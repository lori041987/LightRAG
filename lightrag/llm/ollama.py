from collections.abc import AsyncIterator
import os
import re

import pipmaster as pm

# install specific modules
if not pm.is_installed("ollama"):
    pm.install("ollama")

import ollama

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from lightrag.exceptions import (
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
from lightrag.api import __api_version__

import numpy as np
from typing import Optional, Union
from lightrag.utils import (
    wrap_embedding_func_with_attrs,
    logger,
    verbose_debug,
)
from lightrag.wnc.wnc_logging import wnc_log


_OLLAMA_CLOUD_HOST = "https://ollama.com"
_CLOUD_MODEL_SUFFIX_PATTERN = re.compile(r"(?:-cloud|:cloud)$")


def _format_ollama_durations(response: dict) -> str:
    """
    [WNC] Format Ollama response duration metrics into human-readable seconds.

    For detailed explanation of these metrics, see: wnc_docs/ollama_response_metrics.md

    Args:
        response: Response dict from Ollama SDK containing duration fields (in nanoseconds)

    Returns:
        Formatted string with durations in seconds
    """
    def ns_to_s(ns):
        """Convert nanoseconds to seconds"""
        return f"{ns / 1_000_000_000:.2f}s" if ns else "N/A"

    total = response.get("total_duration")
    load = response.get("load_duration")
    prompt_eval = response.get("prompt_eval_duration")
    eval_dur = response.get("eval_duration")
    prompt_count = response.get("prompt_eval_count", "N/A")
    eval_count = response.get("eval_count", "N/A")

    parts = []
    if total:
        parts.append(f"total={ns_to_s(total)}")
    if load or prompt_eval or eval_dur:
        breakdown = []
        if load:
            breakdown.append(f"load={ns_to_s(load)}")
        if prompt_eval:
            breakdown.append(f"prompt_eval={ns_to_s(prompt_eval)}")
        if eval_dur:
            breakdown.append(f"eval={ns_to_s(eval_dur)}")
        if breakdown:
            parts.append(f"({' + '.join(breakdown)})")

    timing = " ".join(parts) if parts else "N/A"
    tokens = f"tokens: prompt={prompt_count}, output={eval_count}"

    return f"{timing} | {tokens}"


def _coerce_host_for_cloud_model(host: Optional[str], model: object) -> Optional[str]:
    if host:
        return host
    try:
        model_name_str = str(model) if model is not None else ""
    except (TypeError, ValueError, AttributeError) as e:
        logger.warning(f"Failed to convert model to string: {e}, using empty string")
        model_name_str = ""
    if _CLOUD_MODEL_SUFFIX_PATTERN.search(model_name_str):
        logger.debug(
            f"Detected cloud model '{model_name_str}', using Ollama Cloud host"
        )
        return _OLLAMA_CLOUD_HOST
    return host


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def _ollama_model_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    enable_cot: bool = False,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    if enable_cot:
        logger.debug("enable_cot=True is not supported for ollama and will be ignored.")
    stream = True if kwargs.get("stream") else False

    kwargs.pop("max_tokens", None)
    # kwargs.pop("response_format", None) # allow json
    host = kwargs.pop("host", None)
    timeout = kwargs.pop("timeout", None)
    if timeout == 0:
        timeout = None
    kwargs.pop("hashing_kv", None)
    api_key = kwargs.pop("api_key", None)
    # fallback to environment variable when not provided explicitly
    if not api_key:
        api_key = os.getenv("OLLAMA_API_KEY")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"LightRAG/{__api_version__}",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    host = _coerce_host_for_cloud_model(host, model)

    ollama_client = ollama.AsyncClient(host=host, timeout=timeout, headers=headers)

    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        # [WNC] Add debug logging to match OpenAI implementation
        logger.debug("===== Entering func of LLM =====")
        logger.debug(f"Model: {model}   Host: {host}")
        logger.debug(f"Additional kwargs: {kwargs}")
        logger.debug(f"Num of history messages: {len(history_messages)}")
        verbose_debug(f"System prompt: {system_prompt}")
        verbose_debug(f"Query: {prompt}")
        logger.debug("===== Sending Query to LLM =====")

        response = await ollama_client.chat(model=model, messages=messages, **kwargs)
        if stream:
            """cannot cache stream response and process reasoning"""

            async def inner():
                try:
                    async for chunk in response:
                        yield chunk["message"]["content"]
                except Exception as e:
                    logger.error(f"Error in stream response: {str(e)}")
                    raise
                finally:
                    try:
                        await ollama_client._client.aclose()
                        logger.debug("Successfully closed Ollama client for streaming")
                    except Exception as close_error:
                        logger.warning(f"Failed to close Ollama client: {close_error}")

            return inner()
        else:
            model_response = response["message"]["content"]

            """
            If the model also wraps its thoughts in a specific tag,
            this information is not needed for the final
            response and can simply be trimmed.
            """

            # [WNC] Add response logging to match OpenAI implementation
            logger.debug(f"Response content len: {len(model_response)}")
            logger.debug(f"Performance: {_format_ollama_durations(response)}")
            verbose_debug(f"Response: {response}")

            return model_response
    except Exception as e:
        try:
            await ollama_client._client.aclose()
            logger.debug("Successfully closed Ollama client after exception")
        except Exception as close_error:
            logger.warning(
                f"Failed to close Ollama client after exception: {close_error}"
            )
        raise e
    finally:
        if not stream:
            try:
                await ollama_client._client.aclose()
                logger.debug(
                    "Successfully closed Ollama client for non-streaming response"
                )
            except Exception as close_error:
                logger.warning(
                    f"Failed to close Ollama client in finally block: {close_error}"
                )


async def ollama_model_complete(
    prompt,
    system_prompt=None,
    history_messages=[],
    enable_cot: bool = False,
    keyword_extraction=False,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["format"] = "json"
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await _ollama_model_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(
    embedding_dim=1024, max_token_size=8192, model_name="bge-m3:latest"
)
async def ollama_embed(
    texts: list[str],
    embed_model: str = "bge-m3:latest",
    max_token_size: int | None = None,
    **kwargs,
) -> np.ndarray:
    """Generate embeddings using Ollama's API.

    Args:
        texts: List of texts to embed.
        embed_model: The Ollama embedding model to use. Default is "bge-m3:latest".
        max_token_size: Maximum tokens per text. This parameter is automatically
            injected by the EmbeddingFunc wrapper when the underlying function
            signature supports it (via inspect.signature check). Ollama will
            automatically truncate texts exceeding the model's context length
            (num_ctx), so no client-side truncation is needed.
        **kwargs: Additional arguments passed to the Ollama client.

    Returns:
        A numpy array of embeddings, one per input text.

    Note:
        - Ollama API automatically truncates texts exceeding the model's context length
        - The max_token_size parameter is received but not used for client-side truncation
    """
    # [WNC] Initial log
    wnc_log(
        purpose="Generate embeddings for texts using Ollama embeddings API",
        inputs={
            "texts_count": len(texts),
            "texts": texts,  # These are the actual texts to embed (chunks/entities/relations), NOT prompts
            "embed_model": embed_model,
            "max_token_size": max_token_size,
        },
        side_effects="Network call to Ollama embeddings API; Ollama automatically handles text truncation based on model's num_ctx",
        note="Embedding models convert text to vectors WITHOUT prompts/instructions. Just sends raw text to model. "
             "Used by vector DB operations for entity/relation/chunk embeddings. Returns np.ndarray of shape (len(texts), embedding_dim).",
        level="info",
    )

    # Note: max_token_size is received but not used for client-side truncation.
    # Ollama API handles truncation automatically based on the model's num_ctx setting.
    _ = max_token_size  # Acknowledge parameter to avoid unused variable warning

    # [WNC] Add debug logging to match LLM implementation
    host = kwargs.get("host", None)
    logger.debug("===== Entering func of Embedding =====")
    logger.debug(f"Model: {embed_model}   Host: {host}")
    logger.debug(f"Max token size: {max_token_size} (not used, Ollama handles truncation)")
    logger.debug(f"Num of texts: {len(texts)}")
    verbose_debug(f"Texts to embed: {texts}")
    logger.debug("===== Sending Texts to Embedding Model =====")

    api_key = kwargs.pop("api_key", None)
    if not api_key:
        api_key = os.getenv("OLLAMA_API_KEY")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"LightRAG/{__api_version__}",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    host = kwargs.pop("host", None)
    timeout = kwargs.pop("timeout", None)

    host = _coerce_host_for_cloud_model(host, embed_model)

    ollama_client = ollama.AsyncClient(host=host, timeout=timeout, headers=headers)
    try:
        options = kwargs.pop("options", {})
        data = await ollama_client.embed(
            model=embed_model, input=texts, options=options
        )

        # [WNC] Add debug logging after receiving response to match LLM implementation
        logger.debug(f"Received embeddings for {len(data['embeddings'])} texts")
        logger.debug(f"Embeddings shape: {np.array(data['embeddings']).shape}")
        verbose_debug(f"Response: {data}")

        # [WNC] Output log
        wnc_log(
            purpose="[OUTPUT] ollama_embed - embeddings generated successfully",
            outputs={
                "embeddings": np.array(data["embeddings"]),
                "embeddings_shape": np.array(data["embeddings"]).shape,
                "note": "Successfully generated embeddings from Ollama API",
            },
            level="info",
        )

        return np.array(data["embeddings"])
    except Exception as e:
        logger.error(f"Error in ollama_embed: {str(e)}")
        try:
            await ollama_client._client.aclose()
            logger.debug("Successfully closed Ollama client after exception in embed")
        except Exception as close_error:
            logger.warning(
                f"Failed to close Ollama client after exception in embed: {close_error}"
            )
        raise e
    finally:
        try:
            await ollama_client._client.aclose()
            logger.debug("Successfully closed Ollama client after embed")
        except Exception as close_error:
            logger.warning(f"Failed to close Ollama client after embed: {close_error}")
