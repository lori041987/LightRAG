"""
OpenAI-powered LightRAG demo script.

What this script does
---------------------
1) Loads all `*.json` under `--kdb-dir` (default: `wnc_kdb/`)
2) Converts each JSON file into a clean *text* document (LightRAG indexes text)
3) Indexes documents into `--working-dir` (default: `./rag_storage_openai/`)
4) Runs ONE query and prints the answer

Configuration
-------------
Most settings are loaded from `wnc_scripts/config_test.py` (or `--config <path>`).

Key flags
---------
- `skip_index` can be set in `wnc_scripts/config_test.py` to run new questions
  WITHOUT re-indexing (fast). This assumes `working_dir` already contains previously indexed data.

Storage output (default backends)
--------------------------------
With default LightRAG storages, `--working-dir` will contain JSON + GraphML files:
- `kv_store_*.json`: key-value stores (docs, chunks, entities, cache, status)
- `vdb_*.json`: NanoVectorDB vector index files (embeddings)
- `graph_*.graphml`: knowledge graph (NetworkX)
This is normal; SQLite is optional via different storage adapters.

Requirements
------------
- Export `OPENAI_API_KEY` (or set `openai.api_key_env` in the config file)
"""

import argparse
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from contextlib import contextmanager
from time import perf_counter

import numpy as np

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import (
    always_get_an_event_loop,
    compute_mdhash_id,
    logger,
    setup_logger,
    wrap_embedding_func_with_attrs,
)

# [WNC] Import shared utilities from utils_test module in the same directory
sys.path.insert(0, str(Path(__file__).parent))
from utils_test import (
    enable_console_timestamps,
    enable_wnc_prefix,
    phase,
    load_mixed_docs,
    load_docs_with_textract,
    write_index_input_manifest,
    load_test_config,
    initialize_rag_storages,
    finalize_rag_storages,
    index_documents,
    run_query,
    rerank_model_func,
)

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config_test.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to config file (default: wnc_scripts/config_test.py).",
    )
    parser.add_argument("--kdb-dir", default=None, help="Override kdb_dir from config.")
    parser.add_argument(
        "--working-dir", default=None, help="Override working_dir from config."
    )
    parser.add_argument(
        "--mode",
        default=None,
        choices=["naive", "local", "global", "hybrid", "mix", "bypass"],
        help="Override query mode from config.",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="Override question from config. If omitted, uses `CONFIG.question`.",
    )
    skip_group = parser.add_mutually_exclusive_group()
    skip_group.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip indexing and only run the query against existing storage in working_dir/config.",
    )
    skip_group.add_argument(
        "--do-index",
        action="store_true",
        help="Force indexing even if config.skip_index is true.",
    )
    parser.add_argument(
        "--dump-index-input",
        action="store_true",
        help="Write a manifest of (file_path, doc text preview) that is fed into the indexer.",
    )
    parser.add_argument(
        "--dump-index-input-path",
        default=None,
        help="Output path for --dump-index-input (default: <working-dir>/index_input_manifest.txt).",
    )
    parser.add_argument(
        "--log-index-input",
        action="store_true",
        help="Log each (file_path, doc text preview) that is fed into the indexer.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=500,
        help="Preview length used by --dump-index-input/--log-index-input.",
    )

    args = parser.parse_args()

    # Important argparse behavior:
    # option names like `--skip-index` become attributes like `args.skip_index` (dashes -> underscores).
    config = load_test_config(Path(args.config))
    kdb_dir = Path(args.kdb_dir) if args.kdb_dir else Path(config.kdb_dir)
    working_dir = str(args.working_dir) if args.working_dir else str(config.working_dir)
    mode = str(args.mode) if args.mode else str(config.mode)
    question = args.question if args.question is not None else str(config.question)
    if not question.strip():
        raise SystemExit(
            "Missing question. Provide `--question ...` or set `CONFIG.question` in the config file."
        )

    # Determine indexing behavior from reindex_strategy or legacy skip_index
    reindex_strategy = getattr(config, "reindex_strategy", "incremental")
    if args.skip_index:
        skip_index = True
        reindex_strategy = "skip"
    elif args.do_index:
        skip_index = False
        # Keep reindex_strategy from config
    else:
        skip_index = bool(getattr(config, "skip_index", False))
        if skip_index:
            reindex_strategy = "skip"

    # Handle force re-indexing: clear storage directory if it exists
    if reindex_strategy == "force" and os.path.exists(working_dir):
        logger.info(
            f"Force re-indexing enabled: clearing existing storage at {working_dir}"
        )
        import shutil
        shutil.rmtree(working_dir)
        logger.info(f"Cleared storage directory: {working_dir}")

    # Setup logging based on config (can be overridden by LOG_LEVEL env var)
    lightrag_level = os.getenv("LOG_LEVEL", config.lightrag_log_level)
    setup_logger("lightrag", level=lightrag_level)

    # Setup WNC logger separately using setup_logger to get same format
    import lightrag.wnc.wnc_logging  # Import to register TRACE level
    setup_logger("lightrag.wnc", level=config.wnc_log_level.upper(), enable_file_logging=False)
    enable_console_timestamps("lightrag.wnc", time_only=True)

    # [WNC] Set verbose debug mode from config
    if config.verbose_debug:
        import lightrag.utils
        lightrag.utils.VERBOSE_DEBUG = True

    # [WNC] Set WNC log delimiter from config
    from lightrag.wnc import set_delimiter, set_trace_content_limit, set_log_content_limit
    wnc_delimiter = getattr(config, "wnc_log_delimiter", "pipe")
    set_delimiter(wnc_delimiter)

    # [WNC] Set trace content limit from config
    wnc_trace_limit = getattr(config, "wnc_log_trace_content_limit", 10000)
    set_trace_content_limit(wnc_trace_limit)

    # [WNC] Set log content limit from config
    wnc_content_limit = getattr(config, "wnc_log_content_limit", 0)
    set_log_content_limit(wnc_content_limit)

    # [WNC] Use time_only=True to show only time (09:20:06,054) instead of full datetime
    enable_console_timestamps("lightrag", time_only=True)
    enable_wnc_prefix("lightrag", script_name="openai_test.py")

    # For standard OpenAI endpoints this must be set (or pass `--api-key-env`).
    # For Azure mode, LightRAG's OpenAI binding can also read Azure env vars,
    # but we still pass `api_key` through for consistency.
    api_key_env = config.openai.api_key_env
    api_key = os.getenv(api_key_env)
    if not api_key and not config.openai.use_azure:
        raise SystemExit(f"Missing {api_key_env} in environment.")

    logger.info(
        "Run config: config=%s working_dir=%s kdb_dir=%s backend=%s mode=%s reindex_strategy=%s",
        args.config,
        working_dir,
        kdb_dir,
        config.ingest.backend,
        mode,
        reindex_strategy,
    )
    if config.ingest.backend == "raganything":
        ra = config.ingest.raganything
        logger.info(
            "RAG-Anything config: parser=%s parse_method=%s image=%s table=%s equation=%s max_concurrent_files=%s multimodal_query=%s",
            ra.parser,
            ra.parse_method,
            ra.enable_image_processing,
            ra.enable_table_processing,
            ra.enable_equation_processing,
            ra.max_concurrent_files,
            ra.use_multimodal_query,
        )

    # Load and format documents only when we are indexing.
    # When `skip_index` is enabled (via config or CLI), we assume `working_dir` already contains:
    # - kv_store_*.json + vdb_*.json + graph_*.graphml
    # created by a previous run of this script.
    docs: list[str] = []
    file_paths: list[str] = []
    if not skip_index:
        with phase("Prepare index inputs"):
            if not kdb_dir.exists():
                raise SystemExit(f"kdb dir not found: {kdb_dir}")

            ingest_backend = config.ingest.backend
            if ingest_backend == "simple":
                docs, file_paths = load_mixed_docs(
                    kdb_dir=kdb_dir,
                    include_ground_truth=config.ingest.include_ground_truth,
                    allow_pdf=config.ingest.allow_pdf,
                    pdf_extractor=config.ingest.pdf_extractor,
                    preprocess_json=config.ingest.preprocess_json,
                )
                if not docs:
                    raise SystemExit(
                        f"No supported documents found under: {kdb_dir} "
                        "(supported: .json, .txt, .md, and .pdf when allow_pdf=True in config)"
                    )
                logger.info("Prepared %s documents for indexing.", len(docs))
            elif ingest_backend == "textract":
                docs, file_paths = load_docs_with_textract(
                    kdb_dir=kdb_dir,
                    include_ground_truth=config.ingest.include_ground_truth,
                    preprocess_json=config.ingest.preprocess_json,
                )
                if not docs:
                    raise SystemExit(
                        f"No documents produced by textract under: {kdb_dir} "
                        "(did you install `textract` and its OS dependencies?)"
                    )
                logger.info("Prepared %s documents for indexing.", len(docs))
            elif ingest_backend == "raganything":
                # RAG-Anything ingests files directly into LightRAG; no `docs` list needed here.
                logger.info(
                    "Using RAG-Anything ingestion; documents will be parsed and inserted file-by-file."
                )

        if args.dump_index_input:
            out_path = (
                Path(args.dump_index_input_path)
                if args.dump_index_input_path
                else Path(working_dir) / "index_input_manifest.txt"
            )
            write_index_input_manifest(
                docs=docs,
                file_paths=file_paths,
                out_path=out_path,
                preview_chars=max(0, int(args.preview_chars)),
            )
            logger.info("Wrote index input manifest: %s", out_path)

        if args.log_index_input:
            preview_chars = max(0, int(args.preview_chars))
            logger.info("Index input count: %s", len(docs))
            for idx, (doc, path) in enumerate(zip(docs, file_paths), start=1):
                preview = doc[:preview_chars]
                logger.info("[%s] file_path: %s", idx, path)
                logger.info("[%s] doc_chars: %s", idx, len(doc))
            logger.info("[%s] doc_preview:\n%s", idx, preview)

    async def llm_model_func(
        prompt,
        system_prompt=None,
        history_messages=[],
        keyword_extraction=False,
        **kwargs,
    ) -> str:
        # LLM (chat completion) function used by LightRAG for:
        # - keyword extraction
        # - answer synthesis (and sometimes summarization/entity extraction)
        return await openai_complete_if_cache(
            config.openai.chat_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=config.openai.base_url,
            timeout=config.openai.timeout,
            use_azure=config.openai.use_azure,
            azure_deployment=config.openai.azure_deployment,
            api_version=config.openai.api_version,
            keyword_extraction=keyword_extraction,
            **kwargs,
        )

    @wrap_embedding_func_with_attrs(
        embedding_dim=config.openai.embed_dim,
        max_token_size=8192,
        model_name=config.openai.embed_model,
        send_dimensions=True,
    )
    async def embedding_func(
        texts: list[str],
        embedding_dim: int | None = None,
        max_token_size: int | None = None,
    ) -> np.ndarray:
        # Embedding function used by LightRAG for vector indexing + query vectors.
        # The decorator injects `embedding_dim` / `max_token_size` automatically.
        #
        # NOTE: `openai_embed` does not accept a top-level `timeout=` kwarg;
        # timeout must be passed via `client_configs`.
        client_configs = (
            {"timeout": config.openai.timeout} if config.openai.timeout is not None else None
        )
        return await openai_embed.func(
            texts,
            model=config.openai.embed_model,
            api_key=api_key,
            base_url=config.openai.base_url,
            embedding_dim=embedding_dim,
            max_token_size=max_token_size,
            client_configs=client_configs,
            use_azure=config.openai.use_azure,
            azure_deployment=config.openai.azure_deployment,
            api_version=config.openai.api_version,
        )

    # Get query parameters from config
    cosine_threshold = getattr(config, "cosine_threshold", 0.2)
    chunk_top_k = getattr(config, "chunk_top_k", None)
    # [WNC] Get source path boost config
    enable_source_path_boost = getattr(config, "enable_source_path_boost", False)
    source_path_boosts = getattr(config, "source_path_boosts", [])

    # [WNC] Configure reranker if enabled
    rerank_func = None
    min_rerank_score = 0.0
    if getattr(config, "enable_rerank", False):
        rerank_model_path = getattr(config, "rerank_model_path", None)
        if rerank_model_path:
            # Store model_path in function attribute for access in rerank_model_func
            rerank_model_func._model_path = rerank_model_path
            rerank_func = rerank_model_func
            min_rerank_score = getattr(config, "min_rerank_score", 0.0)
            logger.info("Reranking enabled with model: %s", rerank_model_path)
            logger.info("Rerank top_n: %s (None = use chunk_top_k)", getattr(config, "rerank_top_n", None))
            logger.info("Min rerank score: %.4f", min_rerank_score)
        else:
            logger.warning("enable_rerank=True but rerank_model_path not configured, reranking disabled")
    else:
        logger.info("Reranking disabled (enable_rerank=False)")

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        llm_model_name=config.openai.chat_model,
        embedding_func=embedding_func,
        llm_model_max_async=config.llm_model_max_async,
        max_parallel_insert=int(getattr(config, "max_parallel_insert", 2)),
        enable_llm_cache=config.enable_llm_cache,
        enable_llm_cache_for_entity_extract=config.enable_llm_cache_for_entity_extract,
        cosine_threshold=cosine_threshold,                  # For __post_init__
        cosine_better_than_threshold=cosine_threshold,      # For real vector DB cutoff
        chunk_top_k=chunk_top_k if chunk_top_k else 60,     # For __post_init__, the final cap is enforced in process_chunks_unified using QueryParam.chunk_top_k, therefore, we must set during QueryParam again later.
        # [WNC] Source path boost configuration
        enable_source_path_boost=enable_source_path_boost,
        source_path_boosts=source_path_boosts,
        # [WNC] Reranking configuration
        rerank_model_func=rerank_func,
        min_rerank_score=min_rerank_score,
    )

    # LightRAG requires explicit storage lifecycle management.
    # We initialize storages before indexing/querying and finalize at the end.
    loop = always_get_an_event_loop()
    initialize_rag_storages(rag)
    try:
        def _make_raganything(vision_model_func):
            try:
                from raganything import RAGAnything, RAGAnythingConfig  # type: ignore
            except Exception:
                from raganything import RAGAnything  # type: ignore

                return RAGAnything(lightrag=rag, vision_model_func=vision_model_func)

            rag_settings = config.ingest.raganything
            config_kwargs = {
                "working_dir": working_dir,
                "parse_method": rag_settings.parse_method,
                "parser": rag_settings.parser,
                "enable_image_processing": rag_settings.enable_image_processing,
                "enable_table_processing": rag_settings.enable_table_processing,
                "enable_equation_processing": rag_settings.enable_equation_processing,
                "max_concurrent_files": rag_settings.max_concurrent_files,
            }

            try:
                config = RAGAnythingConfig(**config_kwargs)
            except TypeError:
                # Older raganything versions may not accept some fields.
                config_kwargs.pop("max_concurrent_files", None)
                config = RAGAnythingConfig(**config_kwargs)

            try:
                return RAGAnything(
                    lightrag=rag,
                    vision_model_func=vision_model_func,
                    config=config,
                )
            except TypeError:
                # Older raganything versions may not accept `config=...`.
                logger.warning(
                    "RAGAnything does not accept `config=...`; falling back to defaults."
                )
                return RAGAnything(lightrag=rag, vision_model_func=vision_model_func)

        if not skip_index:
            with phase("Index documents"):
                # This enqueues documents and runs the internal pipeline:
                # chunking -> embeddings -> entity/relation extraction -> graph construction.
                ingest_backend = config.ingest.backend
                if ingest_backend in {"simple", "textract"}:
                    index_documents(
                        rag,
                        docs,
                        file_paths,
                        max_parallel_insert=int(getattr(config, "max_parallel_insert", 2)),
                    )
                elif ingest_backend == "raganything":
                    try:
                        from raganything import RAGAnything  # type: ignore
                    except Exception as e:
                        raise SystemExit(
                            "Ingest backend 'raganything' requires `raganything`. "
                            "Install it (e.g. `pip install raganything`) and re-run."
                        ) from e

                    rag_settings = config.ingest.raganything
                    raganything_output_dir = (
                        rag_settings.output_dir
                        if rag_settings.output_dir
                        else str(Path(working_dir) / "raganything_output")
                    )

                    async def vision_model_func(
                        prompt,
                        system_prompt=None,
                        history_messages=[],
                        image_data=None,
                        **kwargs,
                    ) -> str:
                        # RAG-Anything passes base64 image data when it needs vision.
                        messages: list[dict] = []
                        if system_prompt:
                            messages.append({"role": "system", "content": system_prompt})
                        if history_messages:
                            messages.extend(history_messages)
                        if image_data:
                            messages.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": prompt},
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/jpeg;base64,{image_data}"
                                            },
                                        },
                                    ],
                                }
                            )
                        else:
                            messages.append({"role": "user", "content": prompt})

                        return await openai_complete_if_cache(
                            config.openai.vision_model,
                            "",
                            messages=messages,
                            api_key=api_key,
                            base_url=config.openai.base_url,
                            timeout=config.openai.timeout,
                            use_azure=config.openai.use_azure,
                            azure_deployment=config.openai.azure_deployment,
                            api_version=config.openai.api_version,
                            **kwargs,
                        )

                    rag_multi = _make_raganything(vision_model_func)

                    # Ingest all non-JSON files via RAG-Anything; JSON stays as text documents.
                    # If you want RAG-Anything for JSON too, convert JSON to a text file first.
                    ingest_paths: list[Path] = []
                    for p in sorted(kdb_dir.rglob("*")):
                        if p.is_file() and p.suffix.lower() != ".json":
                            ingest_paths.append(p)

                    logger.info(
                        "RAG-Anything will ingest %s non-JSON files; output_dir=%s",
                        len(ingest_paths),
                        raganything_output_dir,
                    )
                    if not ingest_paths:
                        logger.warning(
                            "No non-JSON files found under %s for RAG-Anything ingestion.",
                            kdb_dir,
                        )

                    for p in ingest_paths:
                        with phase(f"RAG-Anything process file: {p}"):
                            loop.run_until_complete(
                                rag_multi.process_document_complete(
                                    file_path=str(p),
                                    output_dir=raganything_output_dir,
                                    parse_method=rag_settings.parse_method,
                                )
                            )

        if (
            config.ingest.backend == "raganything"
            and config.ingest.raganything.use_multimodal_query
        ):
            try:
                from raganything import RAGAnything  # type: ignore
            except Exception as e:
                raise SystemExit(
                    "Multimodal query requires `raganything`. Install it (e.g. `pip install raganything`)."
                ) from e

            async def vision_model_func(
                prompt,
                system_prompt=None,
                history_messages=[],
                image_data=None,
                **kwargs,
            ) -> str:
                messages: list[dict] = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                if history_messages:
                    messages.extend(history_messages)
                if image_data:
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_data}"
                                    },
                                },
                            ],
                        }
                    )
                else:
                    messages.append({"role": "user", "content": prompt})

                return await openai_complete_if_cache(
                    config.openai.vision_model,
                    "",
                    messages=messages,
                    api_key=api_key,
                    base_url=config.openai.base_url,
                    timeout=config.openai.timeout,
                    use_azure=config.openai.use_azure,
                    azure_deployment=config.openai.azure_deployment,
                    api_version=config.openai.api_version,
                    **kwargs,
                )

            rag_multi = _make_raganything(vision_model_func)
            with phase("Query (multimodal)"):
                logger.info("Question:\n%s", question.strip())
                answer = loop.run_until_complete(
                    rag_multi.query_with_multimodal(question, mode=mode)
                )
        else:
            # Pass rerank configuration to query
            enable_rerank = getattr(config, "enable_rerank", None)
            rerank_top_n = getattr(config, "rerank_top_n", None)
            answer = run_query(
                rag,
                question,
                mode=mode,
                chunk_top_k=chunk_top_k,
                enable_rerank=enable_rerank,
                rerank_top_n=rerank_top_n,
            )

        logger.info("Question:\n%s", question.strip())
        logger.info("Answer:\n%s", answer)
    finally:
        finalize_rag_storages(rag)


if __name__ == "__main__":
    main()
