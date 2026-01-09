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
Most settings are loaded from `wnc_scripts/openai_test_config.py` (or `--config <path>`).

Key flags
---------
- `skip_index` can be set in `wnc_scripts/openai_test_config.py` to run new questions
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

DEFAULT_CONFIG_PATH = Path(__file__).with_name("openai_test_config.py")


def _load_config_module(config_path: Path) -> ModuleType:
    module_name = f"openai_test_config_{abs(hash(str(config_path.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load config module: {config_path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses (and other reflection) expect the module to exist in sys.modules.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_openai_test_config(config_path: Path):
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    module = _load_config_module(config_path)
    config = getattr(module, "CONFIG", None)
    if config is None:
        raise SystemExit(f"Config file must define `CONFIG`: {config_path}")
    return config


def _enable_console_timestamps(logger_name: str = "lightrag") -> None:
    """
    LightRAG's default console logging format omits timestamps.
    For long-running ingestion jobs, timestamps make phase boundaries much easier to read.
    """
    logger_instance = logging.getLogger(logger_name)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in logger_instance.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(formatter)


class _WNCModulePrefixFilter(logging.Filter):
    """Prefix log records originating from this script to make them easy to spot."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            pathname = getattr(record, "pathname", "") or ""
            if pathname.endswith(str(Path(__file__).name)):
                msg = str(getattr(record, "msg", ""))
                if not msg.startswith("[WNC]"):
                    record.msg = f"[WNC] {msg}"
        except Exception:
            # Never block logging due to filter errors.
            pass
        return True


def _enable_wnc_prefix(logger_name: str = "lightrag") -> None:
    """Add a handler filter that prefixes logs from this script with [WNC]."""
    logger_instance = logging.getLogger(logger_name)
    prefix_filter = _WNCModulePrefixFilter()
    for handler in logger_instance.handlers:
        handler.addFilter(prefix_filter)


@contextmanager
def _phase(name: str):
    start = perf_counter()
    logger.info("==> %s", name)
    try:
        yield
    finally:
        elapsed = perf_counter() - start
        logger.info("<== %s (%.2fs)", name, elapsed)


def _json_value_to_text(value: object) -> str:
    """Convert JSON values to readable text for embedding."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _sample_json_to_doc_text(sample: dict, include_ground_truth: bool) -> str:
    """
    Convert ONE JSON sample into a compact, embedding-friendly document.

    We prioritize "signal" fields:
    - metadata: id/tags/task_types
    - payload: input_datas[*].content_raw (your XML snippets)
    - optionally: ground_truths (labels/answers) if you explicitly enable it

    Tip: Usually keep `include_ground_truth=False` to avoid "training on the answer".
    """
    lines: list[str] = []

    for key in ("id", "tags", "task_types"):
        if key in sample:
            lines.append(f"{key}: {_json_value_to_text(sample.get(key))}")

    input_datas = sample.get("input_datas") or []
    if isinstance(input_datas, list) and input_datas:
        lines.append("input_datas:")
        for idx, item in enumerate(input_datas, start=1):
            if not isinstance(item, dict):
                continue
            filename = _json_value_to_text(item.get("filename"))
            timestamp = _json_value_to_text(item.get("timestamp"))
            content_raw = _json_value_to_text(item.get("content_raw"))

            header = f"- item_{idx}"
            if filename:
                header += f" filename={filename}"
            if timestamp:
                header += f" timestamp={timestamp}"
            lines.append(header)
            if content_raw:
                lines.append(content_raw)

    references = sample.get("references") or []
    if references:
        lines.append(f"references: {_json_value_to_text(references)}")

    if include_ground_truth and "ground_truths" in sample:
        lines.append(f"ground_truths: {_json_value_to_text(sample.get('ground_truths'))}")

    return "\n".join([line for line in lines if line.strip()])


def _load_json_docs(
    kdb_dir: Path, include_ground_truth: bool
) -> tuple[list[str], list[str]]:
    """Load and format all `*.json` files under `kdb_dir`."""
    docs: list[str] = []
    file_paths: list[str] = []

    for path in sorted(kdb_dir.rglob("*.json")):
        try:
            sample = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Skipping %s: failed to parse JSON: %s", path, e)
            continue

        if not isinstance(sample, dict):
            logger.warning("Skipping %s: top-level JSON is not an object", path)
            continue

        doc_text = _sample_json_to_doc_text(sample, include_ground_truth)
        if not doc_text.strip():
            logger.warning("Skipping %s: empty doc text after formatting", path)
            continue

        docs.append(f"source_file: {path}\n{doc_text}")
        file_paths.append(str(path))

    return docs, file_paths


def _extract_pdf_text_pypdf(path: Path) -> str:
    """
    Extract text from a PDF using pypdf (best-effort).

    This is intentionally lightweight and offline. If you need higher quality PDF
    extraction (tables/layout), consider using the LightRAG API server's upload
    pipeline or a dedicated extractor.
    """
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PDF support requires `pypdf`. Install it in your venv (e.g. `pip install pypdf`)."
        ) from e

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text)
    return "\n".join(parts).strip()


def _extract_with_textract(path: Path) -> str:
    """Extract text with `textract` (matches README insert multi-file snippet)."""
    """
    Note: `textract` is an optional dependency; install it yourself if you choose
    the textract ingest backend.
    """
    try:
        import textract  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Ingest backend 'textract' requires `textract`. Install it in your venv (e.g. `pip install textract`)."
        ) from e

    data = textract.process(str(path))
    return data.decode("utf-8", errors="ignore").strip()


def _extract_pdf_text(path: Path, extractor: str) -> str:
    """Extract text from PDF using chosen extractor: auto|textract|pypdf."""
    if extractor == "textract":
        return _extract_with_textract(path)
    if extractor == "pypdf":
        return _extract_pdf_text_pypdf(path)

    # auto: prefer textract if installed, else pypdf
    try:
        return _extract_with_textract(path)
    except ModuleNotFoundError:
        return _extract_pdf_text_pypdf(path)


def _load_mixed_docs(
    kdb_dir: Path,
    include_ground_truth: bool,
    allow_pdf: bool,
    pdf_extractor: str,
) -> tuple[list[str], list[str]]:
    """
    Load documents from a directory tree.

    Supported formats:
    - `.json`: your structured samples (converted into a text document)
    - `.txt` / `.md`: raw text files
    - `.pdf`: extracted to text (requires `pypdf`) when `allow_pdf=True`
    """
    docs: list[str] = []
    file_paths: list[str] = []

    for path in sorted(kdb_dir.rglob("*")):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        try:
            if suffix == ".json":
                sample = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(sample, dict):
                    logger.warning(
                        "Skipping %s: top-level JSON is not an object", path
                    )
                    continue
                doc_text = _sample_json_to_doc_text(sample, include_ground_truth)
                if not doc_text.strip():
                    logger.warning(
                        "Skipping %s: empty doc text after formatting", path
                    )
                    continue
                docs.append(f"source_file: {path}\n{doc_text}")
                file_paths.append(str(path))
                continue

            if suffix in {".txt", ".md"}:
                text = path.read_text(encoding="utf-8", errors="ignore").strip()
                if not text:
                    logger.warning("Skipping %s: empty text file", path)
                    continue
                docs.append(f"source_file: {path}\n{text}")
                file_paths.append(str(path))
                continue

            if suffix == ".pdf" and allow_pdf:
                try:
                    text = _extract_pdf_text(path, extractor=pdf_extractor)
                except Exception as e:
                    logger.warning("Skipping %s: failed PDF extraction: %s", path, e)
                    continue
                if not text:
                    logger.warning("Skipping %s: empty extracted PDF text", path)
                    continue
                docs.append(f"source_file: {path}\n{text}")
                file_paths.append(str(path))
                continue
        except Exception as e:
            logger.warning("Skipping %s: failed to read/format: %s", path, e)
            continue

    return docs, file_paths


def _load_docs_with_textract(
    kdb_dir: Path, include_ground_truth: bool
) -> tuple[list[str], list[str]]:
    """
    Ingest backend: textract

    - `.json`: formatted into text (same as simple mode)
    - other files: extracted to text by `textract` (PDF/DOCX/PPTX/CSV/etc.)
    """
    docs: list[str] = []
    file_paths: list[str] = []

    for path in sorted(kdb_dir.rglob("*")):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        try:
            if suffix == ".json":
                sample = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(sample, dict):
                    logger.warning(
                        "Skipping %s: top-level JSON is not an object", path
                    )
                    continue
                doc_text = _sample_json_to_doc_text(sample, include_ground_truth)
                if not doc_text.strip():
                    logger.warning(
                        "Skipping %s: empty doc text after formatting", path
                    )
                    continue
                docs.append(f"source_file: {path}\n{doc_text}")
                file_paths.append(str(path))
                continue

            text = _extract_with_textract(path)
            if not text:
                logger.warning("Skipping %s: empty textract output", path)
                continue
            docs.append(f"source_file: {path}\n{text}")
            file_paths.append(str(path))
        except Exception as e:
            logger.warning("Skipping %s: failed to read/format: %s", path, e)

    return docs, file_paths


def _write_index_input_manifest(
    docs: list[str],
    file_paths: list[str],
    out_path: Path,
    preview_chars: int,
) -> None:
    """
    Save exactly what will be fed into `rag.insert(docs, file_paths=...)`.

    This helps you verify the text formatting (especially for JSON-to-text conversion)
    and confirm which source files were indexed.
    """
    if len(docs) != len(file_paths):
        raise ValueError(
            f"docs/file_paths length mismatch: {len(docs)} vs {len(file_paths)}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"count: {len(docs)}\n")
        f.write(f"preview_chars: {preview_chars}\n")
        f.write("\n")
        for idx, (doc, path) in enumerate(zip(docs, file_paths), start=1):
            preview = doc[:preview_chars].replace("\n", "\\n")
            f.write(f"[{idx}] file_path: {path}\n")
            f.write(f"[{idx}] doc_chars: {len(doc)}\n")
            f.write(f"[{idx}] doc_preview: {preview}\n")
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to config file (default: wnc_scripts/openai_test_config.py).",
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
    config = _load_openai_test_config(Path(args.config))
    kdb_dir = Path(args.kdb_dir) if args.kdb_dir else Path(config.kdb_dir)
    working_dir = str(args.working_dir) if args.working_dir else str(config.working_dir)
    mode = str(args.mode) if args.mode else str(config.mode)
    question = args.question if args.question is not None else str(config.question)
    if not question.strip():
        raise SystemExit(
            "Missing question. Provide `--question ...` or set `CONFIG.question` in the config file."
        )

    if args.skip_index:
        skip_index = True
    elif args.do_index:
        skip_index = False
    else:
        skip_index = bool(getattr(config, "skip_index", False))

    setup_logger("lightrag", level=os.getenv("LOG_LEVEL", "INFO"))
    _enable_console_timestamps("lightrag")
    _enable_wnc_prefix("lightrag")

    # For standard OpenAI endpoints this must be set (or pass `--api-key-env`).
    # For Azure mode, LightRAG's OpenAI binding can also read Azure env vars,
    # but we still pass `api_key` through for consistency.
    api_key_env = config.openai.api_key_env
    api_key = os.getenv(api_key_env)
    if not api_key and not config.openai.use_azure:
        raise SystemExit(f"Missing {api_key_env} in environment.")

    logger.info(
        "Run config: config=%s working_dir=%s kdb_dir=%s backend=%s mode=%s skip_index=%s",
        args.config,
        working_dir,
        kdb_dir,
        config.ingest.backend,
        mode,
        skip_index,
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
        with _phase("Prepare index inputs"):
            if not kdb_dir.exists():
                raise SystemExit(f"kdb dir not found: {kdb_dir}")

            ingest_backend = config.ingest.backend
            if ingest_backend == "simple":
                docs, file_paths = _load_mixed_docs(
                    kdb_dir=kdb_dir,
                    include_ground_truth=config.ingest.include_ground_truth,
                    allow_pdf=config.ingest.allow_pdf,
                    pdf_extractor=config.ingest.pdf_extractor,
                )
                if not docs:
                    raise SystemExit(
                        f"No supported documents found under: {kdb_dir} "
                        "(supported: .json, .txt, .md, and .pdf when allow_pdf=True in config)"
                    )
                logger.info("Prepared %s documents for indexing.", len(docs))
            elif ingest_backend == "textract":
                docs, file_paths = _load_docs_with_textract(
                    kdb_dir=kdb_dir,
                    include_ground_truth=config.ingest.include_ground_truth,
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
            _write_index_input_manifest(
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

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        llm_model_name=config.openai.chat_model,
        embedding_func=embedding_func,
        max_parallel_insert=int(getattr(config, "max_parallel_insert", 2)),
    )

    # LightRAG requires explicit storage lifecycle management.
    # We initialize storages before indexing/querying and finalize at the end.
    loop = always_get_an_event_loop()
    with _phase("Initialize storages"):
        loop.run_until_complete(rag.initialize_storages())
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
            with _phase("Index documents"):
                # This enqueues documents and runs the internal pipeline:
                # chunking -> embeddings -> entity/relation extraction -> graph construction.
                ingest_backend = config.ingest.backend
                if ingest_backend in {"simple", "textract"}:
                    logger.info(
                        "Indexing %s documents into %s (max_parallel_insert=%s)",
                        len(docs),
                        working_dir,
                        int(getattr(config, "max_parallel_insert", 2)),
                    )
                    doc_ids = [compute_mdhash_id(d, prefix="doc-") for d in docs]
                    for doc_id, path in zip(doc_ids, file_paths):
                        logger.info("Index input: doc_id=%s file=%s", doc_id, path)
                    rag.insert(docs, file_paths=file_paths, ids=doc_ids)
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
                        with _phase(f"RAG-Anything process file: {p}"):
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
            with _phase("Query (multimodal)"):
                logger.info("Question:\n%s", question.strip())
                answer = loop.run_until_complete(
                    rag_multi.query_with_multimodal(question, mode=mode)
                )
        else:
            with _phase("Query"):
                logger.info("Question:\n%s", question.strip())
                answer = rag.query(question, param=QueryParam(mode=mode))

        logger.info("Question:\n%s", question.strip())
        logger.info("Answer:\n%s", answer)
    finally:
        with _phase("Finalize storages"):
            loop.run_until_complete(rag.finalize_storages())


if __name__ == "__main__":
    main()
