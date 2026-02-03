"""
Ollama-powered LightRAG demo script (text-only, offline).

What this script does
---------------------
1) Loads documents from `--kdb-dir` (default: `wnc_kdb/`)
   - Supported formats depend on ingest backend (simple/textract):
     - JSON files (optionally preprocessed into flat text if preprocess_json=True)
     - Text files (.txt, .md)
     - PDFs (when allow_pdf=True and backend=simple)
     - Other formats (when backend=textract with proper OS dependencies)
2) Indexes documents into `--working-dir` (default: `./rag_storage_ollama/`)
3) Runs ONE query and prints the answer

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
- Ollama running locally (default: localhost:11434)
- Ollama models pulled: qwen3:8b (chat), bge-m3:567m (embeddings)
- Text-only mode enforced (use_multimodal_query=False)

Notes
-----
This script is designed for OFFLINE use (no external network except local Ollama).
It does NOT support multimodal features (images/tables/equations).
"""

import argparse
import logging
import os
import shutil
import sys
from functools import partial
from pathlib import Path

import numpy as np

from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import (
    EmbeddingFunc,
    logger,
    setup_logger,
)
import lightrag.utils
import lightrag.wnc.wnc_logging
from lightrag.wnc import set_delimiter, set_trace_content_limit, set_log_content_limit

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
    enforce_text_only_mode,
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

    # [WNC] Enforce text-only mode (no multimodal features)
    enforce_text_only_mode(config)

    kdb_dir = Path(args.kdb_dir) if args.kdb_dir else Path(config.kdb_dir)
    # Determine working_dir: CLI arg > config > default
    if args.working_dir:
        working_dir = str(args.working_dir)
    elif hasattr(config, 'working_dir') and config.working_dir:
        working_dir = str(config.working_dir)
    else:
        # No working_dir configured, use default
        working_dir = "./rag_storage_ollama"

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
        shutil.rmtree(working_dir)
        logger.info(f"Cleared storage directory: {working_dir}")

    # Setup logging based on config (can be overridden by LOG_LEVEL env var)
    lightrag_level = os.getenv("LOG_LEVEL", config.lightrag_log_level)
    setup_logger("lightrag", level=lightrag_level)

    # Setup WNC logger separately using setup_logger to get same format
    setup_logger("lightrag.wnc", level=config.wnc_log_level.upper(), enable_file_logging=False)
    enable_console_timestamps("lightrag.wnc", time_only=True)

    # [WNC] Set verbose debug mode from config
    if config.verbose_debug:
        lightrag.utils.VERBOSE_DEBUG = True

    # [WNC] Set WNC log delimiter from config
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
    enable_wnc_prefix("lightrag", script_name="ollama_test.py")

    logger.info(
        "Run config: config=%s working_dir=%s kdb_dir=%s backend=%s mode=%s reindex_strategy=%s",
        args.config,
        working_dir,
        kdb_dir,
        config.ingest.backend,
        mode,
        reindex_strategy,
    )
    logger.info(
        "Ollama config: chat_model=%s embed_model=%s embed_dim=%s host=%s timeout=%s",
        config.ollama.chat_model,
        config.ollama.embed_model,
        config.ollama.embed_dim,
        config.ollama.host or "default (http://localhost:11434)",
        config.ollama.timeout or "default (180s)",
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
                raise SystemExit(
                    "RAG-Anything backend is not supported in text-only Ollama mode. "
                    "Use backend='simple' or 'textract' instead."
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

    # Build llm_model_kwargs for Ollama from config
    llm_model_kwargs = {}
    if config.ollama.host:
        llm_model_kwargs["host"] = config.ollama.host
    if config.ollama.timeout is not None:
        llm_model_kwargs["timeout"] = config.ollama.timeout
    if config.ollama.api_key_env:
        api_key = os.getenv(config.ollama.api_key_env)
        if api_key:
            llm_model_kwargs["api_key"] = api_key
    # Set think mode if configured (for qwen3, gpt-oss, deepseek-v3, deepseek-r1)
    if config.ollama.think is not None:
        llm_model_kwargs["think"] = config.ollama.think
        logger.info("Think mode: %s", config.ollama.think)

    # Get query parameters from config
    cosine_threshold = getattr(config, "cosine_threshold", 0.2)
    chunk_top_k = getattr(config, "chunk_top_k", None)
    # [WNC] Get source path boost config
    enable_source_path_boost = getattr(config, "enable_source_path_boost", False)
    source_path_boosts = getattr(config, "source_path_boosts", [])

    # Build embedding kwargs for Ollama from config
    embedding_kwargs = {}
    if config.ollama.host:
        embedding_kwargs["host"] = config.ollama.host
    if config.ollama.timeout is not None:
        embedding_kwargs["timeout"] = config.ollama.timeout
    if config.ollama.api_key_env:
        api_key = os.getenv(config.ollama.api_key_env)
        if api_key:
            embedding_kwargs["api_key"] = api_key

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
        llm_model_func=ollama_model_complete,  # Pass function directly, not a wrapper
        llm_model_name=config.ollama.chat_model,
        llm_model_kwargs=llm_model_kwargs,  # Pass Ollama-specific kwargs
        # Note: ollama_embed is decorated with @wrap_embedding_func_with_attrs,
        # which wraps it in an EmbeddingFunc. Using .func accesses the original
        # unwrapped function to avoid double wrapping when we create our own
        # EmbeddingFunc with custom configuration (embedding_dim, max_token_size).
        embedding_func=EmbeddingFunc(
            embedding_dim=config.ollama.embed_dim,
            max_token_size=8192,
            func=partial(
                ollama_embed.func,  # Access unwrapped function to avoid double wrapping
                embed_model=config.ollama.embed_model,
                **embedding_kwargs,
            ),
        ),
        llm_model_max_async=config.llm_model_max_async,
        max_parallel_insert=int(getattr(config, "max_parallel_insert", 2)),
        enable_llm_cache=config.enable_llm_cache,
        enable_llm_cache_for_entity_extract=config.enable_llm_cache_for_entity_extract,
        cosine_threshold=cosine_threshold,                  # For __post_init__
        cosine_better_than_threshold=cosine_threshold,      # For real vector DB cutoff
        chunk_top_k=chunk_top_k if chunk_top_k else 60,     # For __post_init__
        # [WNC] Source path boost configuration
        enable_source_path_boost=enable_source_path_boost,
        source_path_boosts=source_path_boosts,
        # [WNC] Reranking configuration
        rerank_model_func=rerank_func,
        min_rerank_score=min_rerank_score,
    )

    # Log effective concurrency settings for transparency
    logger.info("=" * 60)
    logger.info("Concurrency Configuration:")
    logger.info("  llm_model_max_async: %s (chunk-level entity extraction)", rag.llm_model_max_async)
    logger.info("  graph_max_async: %s (graph merging, 2x multiplier)", rag.llm_model_max_async * 2)
    logger.info("  max_parallel_insert: %s (document-level indexing)", rag.max_parallel_insert)
    logger.info("=" * 60)

    # LightRAG requires explicit storage lifecycle management.
    # We initialize storages before indexing/querying and finalize at the end.
    initialize_rag_storages(rag)
    try:
        if not skip_index:
            # Index documents using shared helper
            index_documents(
                rag,
                docs,
                file_paths,
                max_parallel_insert=int(getattr(config, "max_parallel_insert", 2)),
            )

        # Run query using shared helper (text-only, no multimodal)
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
