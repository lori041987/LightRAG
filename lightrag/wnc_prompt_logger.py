"""
[WNC] Prompt Logging Infrastructure for LightRAG

This module provides comprehensive logging of all LLM prompts and contexts
sent during both query-time and indexing-time operations.

Purpose:
- Log exact prompts sent to LLM for debugging
- Help distinguish between LLM errors vs index DB errors vs instruction errors
- Support both console logging and file dumps
- Enable trace_id-based filtering
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


# [WNC] Configuration flags
ENABLE_PROMPT_LOGGING = os.getenv("WNC_ENABLE_PROMPT_LOGGING", "true").lower() == "true"
ENABLE_PROMPT_FILE_DUMP = os.getenv("WNC_ENABLE_PROMPT_FILE_DUMP", "false").lower() == "true"
PROMPT_DUMP_DIR = os.getenv("WNC_PROMPT_DUMP_DIR", "./prompt_logs")


def _truncate_for_display(text: str, max_length: int = 500) -> str:
    """[WNC] Truncate text for console display while preserving readability"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + f"... [truncated, total {len(text)} chars]"


def _format_messages_for_display(messages: list[dict[str, str]]) -> str:
    """[WNC] Format message list for readable console output"""
    lines = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        truncated_content = _truncate_for_display(content, max_length=300)
        lines.append(f"    [{i}] {role}: {truncated_content}")
    return "\n".join(lines)


def log_llm_prompt(
    stage: str,
    cache_type: str,
    system_prompt: Optional[str],
    user_prompt: str,
    history_messages: Optional[list[dict[str, str]]],
    trace_id: Optional[str] = None,
    chunk_id: Optional[str] = None,
    cache_key: Optional[str] = None,
    is_cache_hit: bool = False,
    **extra_context: Any,
) -> None:
    """
    [WNC] Log LLM prompt details to console and optionally to file.

    Args:
        stage: Stage name (e.g., "entity_extraction", "kg_query", "summarization")
        cache_type: Cache type ("extract", "summary", "query", "keywords")
        system_prompt: System prompt sent to LLM
        user_prompt: User prompt sent to LLM
        history_messages: Conversation history
        trace_id: Optional trace ID for filtering
        chunk_id: Optional chunk identifier
        cache_key: Cache key used
        is_cache_hit: Whether this was a cache hit
        **extra_context: Additional context to log
    """
    if not ENABLE_PROMPT_LOGGING:
        return

    # [WNC] Build log prefix
    prefix = f"[WNC][LLM_PROMPT][{stage}][{cache_type}]"
    if trace_id:
        prefix += f"[trace_id={trace_id}]"
    if chunk_id:
        prefix += f"[chunk={chunk_id}]"

    # [WNC] Log prompt structure to console
    logger.info(f"{prefix} {'='*60}")

    logger.info(f"{prefix} LightRAG prompt that will sent to the LLM:")

    if is_cache_hit:
        logger.info(f"{prefix} CACHE HIT - using cached result (would have sent this prompt:)")
    else:
        # [WNC] Explain cache status clearly
        if cache_type == "direct_call":
            logger.info(f"{prefix} Cache disabled - calling LLM directly")
        elif cache_type in ["extract", "summary", "query", "keywords"]:
            logger.info(f"{prefix} Cache miss - calling LLM (will cache result for {cache_type})")
        else:
            logger.info(f"{prefix} Calling LLM (cache_type={cache_type})")

    if cache_key:
        logger.debug(f"{prefix} cache_key={cache_key}")

    # [WNC] Log system prompt
    # [WNC] The instructions telling the LLM how to answer (role, format, etc.) + the retrieved context (entities, relations, chunks)
    if system_prompt:
        logger.info(f"{prefix} SYSTEM PROMPT ({len(system_prompt)} chars):")
        logger.info(f"{prefix}   {_truncate_for_display(system_prompt, 500)}")
    else:
        logger.info(f"{prefix} SYSTEM PROMPT: None")

    # [WNC] Log history messages
    # [WNC] Previous conversation messages (if any)
    if history_messages:
        logger.info(f"{prefix} HISTORY ({len(history_messages)} messages):")
        logger.info(f"{prefix}\n{_format_messages_for_display(history_messages)}")
    else:
        logger.info(f"{prefix} HISTORY: None")

    # [WNC] Log user prompt
    # [WNC] User's question
    logger.info(f"{prefix} USER PROMPT ({len(user_prompt)} chars):")
    logger.info(f"{prefix}   {_truncate_for_display(user_prompt, 500)}")

    # [WNC] Log extra context
    if extra_context:
        for key, value in extra_context.items():
            if isinstance(value, str):
                logger.debug(f"{prefix} {key}={_truncate_for_display(value, 200)}")
            else:
                logger.debug(f"{prefix} {key}={value}")

    logger.info(f"{prefix} {'='*60}")

    # [WNC] Dump to file if enabled (works for both cache hits and misses)
    if ENABLE_PROMPT_FILE_DUMP:
        _dump_prompt_to_file(
            stage=stage,
            cache_type=cache_type,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            history_messages=history_messages,
            trace_id=trace_id,
            chunk_id=chunk_id,
            cache_key=cache_key,
            **extra_context,
        )


def _dump_prompt_to_file(
    stage: str,
    cache_type: str,
    system_prompt: Optional[str],
    user_prompt: str,
    history_messages: Optional[list[dict[str, str]]],
    trace_id: Optional[str],
    chunk_id: Optional[str],
    cache_key: Optional[str],
    **extra_context: Any,
) -> None:
    """[WNC] Dump full prompt details to a JSON file"""
    try:
        # [WNC] Create dump directory
        dump_dir = Path(PROMPT_DUMP_DIR)
        dump_dir.mkdir(parents=True, exist_ok=True)

        # [WNC] Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        trace_suffix = f"_trace_{trace_id}" if trace_id else ""
        filename = f"{timestamp}_{stage}_{cache_type}{trace_suffix}.json"
        filepath = dump_dir / filename

        # [WNC] Prepare dump data
        dump_data = {
            "timestamp": timestamp,
            "stage": stage,
            "cache_type": cache_type,
            "trace_id": trace_id,
            "chunk_id": chunk_id,
            "cache_key": cache_key,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "history_messages": history_messages,
            "extra_context": extra_context,
            "message_sequence": [],
        }

        # [WNC] Build full message sequence as it would be sent to LLM
        if system_prompt:
            dump_data["message_sequence"].append({
                "role": "system",
                "content": system_prompt,
            })
        if history_messages:
            dump_data["message_sequence"].extend(history_messages)
        dump_data["message_sequence"].append({
            "role": "user",
            "content": user_prompt,
        })

        # [WNC] Write to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(dump_data, f, indent=2, ensure_ascii=False)

        logger.debug(f"[WNC][LLM_PROMPT] Dumped prompt to file: {filepath}")

    except Exception as e:
        logger.warning(f"[WNC][LLM_PROMPT] Failed to dump prompt to file: {e}")


def log_query_context(
    trace_id: Optional[str],
    context_type: str,
    entities: Optional[list] = None,
    relations: Optional[list] = None,
    chunks: Optional[list] = None,
    **extra_info: Any,
) -> None:
    """
    [WNC] Log the retrieval context used for query generation.

    This logs what LightRAG retrieved from the knowledge base (vector DB + graph) BEFORE sending to the LLM:

    Args:
        trace_id: Trace ID for this query
        context_type: Type of context ("kg_query" or "naive_query")
        entities: List of entity data
        relations: List of relation data
        chunks: List of document chunks
        **extra_info: Additional info (e.g., token counts, top_k values)
    """
    if not ENABLE_PROMPT_LOGGING:
        return

    prefix = f"[WNC][QUERY_CONTEXT][{context_type}]"
    if trace_id:
        prefix += f"[trace_id={trace_id}]"

    logger.info(f"{prefix} {'='*60}")
    logger.info(f"{prefix} LightRAG Retrieved Context from vector DB + graph:")

    if entities is not None:
        logger.info(f"{prefix}   Entities: {len(entities)} items")
        if entities and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{prefix}   First 3 entities: {entities[:3]}")

    if relations is not None:
        logger.info(f"{prefix}   Relations: {len(relations)} items")
        if relations and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{prefix}   First 3 relations: {relations[:3]}")

    if chunks is not None:
        logger.info(f"{prefix}   Chunks: {len(chunks)} items")
        if chunks and logger.isEnabledFor(logging.DEBUG):
            for i, chunk in enumerate(chunks[:3]):
                content = chunk.get("content", "") if isinstance(chunk, dict) else str(chunk)
                logger.debug(f"{prefix}   Chunk[{i}]: {_truncate_for_display(content, 150)}")

    logger.info(f"{prefix} Extra Info:")
    for key, value in extra_info.items():
        logger.info(f"{prefix}   {key}: {value}")

    logger.info(f"{prefix} {'='*60}")


def log_indexing_operation(
    operation: str,
    trace_id: Optional[str],
    chunk_content: Optional[str] = None,
    entity_count: Optional[int] = None,
    relation_count: Optional[int] = None,
    **extra_info: Any,
) -> None:
    """
    [WNC] Log indexing operations (entity extraction, summarization, etc.)

    Args:
        operation: Operation name (e.g., "entity_extraction", "entity_summarization")
        trace_id: Trace ID for this operation
        chunk_content: Content being processed
        entity_count: Number of entities extracted/processed
        relation_count: Number of relations extracted/processed
        **extra_info: Additional info
    """
    if not ENABLE_PROMPT_LOGGING:
        return

    prefix = f"[WNC][INDEXING][{operation}]"
    if trace_id:
        prefix += f"[trace_id={trace_id}]"

    logger.info(f"{prefix} Starting operation")

    if chunk_content:
        logger.info(f"{prefix}   Input content: {_truncate_for_display(chunk_content, 200)}")

    if entity_count is not None:
        logger.info(f"{prefix}   Entities: {entity_count}")

    if relation_count is not None:
        logger.info(f"{prefix}   Relations: {relation_count}")

    for key, value in extra_info.items():
        logger.debug(f"{prefix}   {key}: {value}")

