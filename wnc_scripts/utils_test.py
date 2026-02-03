"""
[WNC] Shared utility functions for LightRAG test scripts.

This module provides reusable utilities for testing LightRAG with different LLM providers
(OpenAI, Gemini, Ollama, etc.). Functions include:
- Custom logging formatters and filters
- Document loading and formatting (JSON, text, PDF)
- Phase timing context manager
- Index input manifest generation
- Configuration loading
- LightRAG initialization and indexing logic

Usage:
    from wnc_scripts.utils_test import (
        WNCFormatter,
        enable_console_timestamps,
        enable_wnc_prefix,
        phase,
        load_mixed_docs,
        write_index_input_manifest,
        load_config_module,
        initialize_rag_storages,
        index_documents,
        run_query,
    )
"""

import importlib.util
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from types import ModuleType
from typing import Tuple, List, Any, Callable

from lightrag import LightRAG, QueryParam
from lightrag.utils import (
    always_get_an_event_loop,
    compute_mdhash_id,
    logger,
)


# [WNC] Custom formatter to include file path and line number in logs
class WNCFormatter(logging.Formatter):
    """
    Custom formatter that includes file path and line number.
    Two format options:
    - Full: %(asctime)s - %(levelname)s - %(pathname)s +%(lineno)d - %(message)s
    - Time-only: HH:MM:SS,mmm - %(levelname)s - %(pathname)s +%(lineno)d - %(message)s
    """
    def __init__(self, fmt=None, datefmt=None, time_only=False, project_root="/srv/ai/LightRAG"):
        super().__init__(fmt, datefmt)
        self.time_only = time_only
        self.project_root = project_root

    def formatTime(self, record, datefmt=None):
        """Override formatTime to support time-only format."""
        if self.time_only:
            # Format as HH:MM:SS,mmm (time only, no date)
            ct = self.converter(record.created)
            t = time.strftime("%H:%M:%S", ct)
            s = "%s,%03d" % (t, record.msecs)
            return s
        else:
            # Use default formatting (includes date)
            return super().formatTime(record, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        # Get relative path from project root
        pathname = record.pathname
        try:
            # Convert to relative path from project root
            if pathname.startswith(self.project_root + "/"):
                pathname = pathname.replace(self.project_root + "/", "")
        except Exception:
            pass

        # Store the modified pathname
        original_pathname = record.pathname
        record.pathname = pathname

        # Format the record
        result = super().format(record)

        # Restore original pathname
        record.pathname = original_pathname

        return result


def enable_console_timestamps(
    logger_name: str = "lightrag",
    time_only: bool = False,
    project_root: str = "/srv/ai/LightRAG"
) -> None:
    """
    [WNC] Enable custom timestamp formatting for console logs.

    Args:
        logger_name: Name of the logger to configure
        time_only: If True, show only time (HH:MM:SS,mmm) instead of full datetime
        project_root: Project root path for relative path conversion
    """
    logger_instance = logging.getLogger(logger_name)
    formatter = WNCFormatter(
        "%(asctime)s - %(levelname)s - %(pathname)s +%(lineno)d - %(message)s",
        time_only=time_only,
        project_root=project_root
    )
    for handler in logger_instance.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(formatter)


# [WNC] Custom filter to prefix log records from calling script
class WNCModulePrefixFilter(logging.Filter):
    """Prefix log records originating from a specific script to make them easy to spot."""

    def __init__(self, script_name: str, prefix: str = "[WNC]"):
        super().__init__()
        self.script_name = script_name
        self.prefix = prefix

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            pathname = getattr(record, "pathname", "") or ""
            if pathname.endswith(self.script_name):
                msg = str(getattr(record, "msg", ""))
                if not msg.startswith(self.prefix):
                    record.msg = f"{self.prefix} {msg}"
        except Exception:
            # [WNC] Never block logging due to filter errors.
            pass
        return True


def enable_wnc_prefix(
    logger_name: str = "lightrag",
    script_name: str = None,
    prefix: str = "[WNC]"
) -> None:
    """
    [WNC] Add a handler filter that prefixes logs from calling script.

    Args:
        logger_name: Name of the logger to configure
        script_name: Name of the script file (e.g., "openai_test.py")
        prefix: Prefix to add to log messages (default: "[WNC]")
    """
    if script_name is None:
        raise ValueError("script_name must be provided")

    logger_instance = logging.getLogger(logger_name)
    prefix_filter = WNCModulePrefixFilter(script_name, prefix)
    for handler in logger_instance.handlers:
        handler.addFilter(prefix_filter)


# [WNC] Context manager to track and log execution phases with timing
@contextmanager
def phase(name: str):
    """
    Context manager to log and time execution phases.

    Usage:
        with phase("Index documents"):
            # your code here
            pass
    """
    start = perf_counter()
    logger.info("==> %s", name)
    try:
        yield
    finally:
        elapsed = perf_counter() - start
        logger.info("<== %s (%.2fs)", name, elapsed)


# [WNC] Helper to convert JSON values to readable text for embedding
def json_value_to_text(value: object) -> str:
    """Convert JSON values to readable text for embedding."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


# [WNC] Convert JSON sample to embedding-friendly document text
def sample_json_to_doc_text(sample: dict, include_ground_truth: bool, preprocess: bool = True) -> str:
    """
    Convert ONE JSON sample into a compact, embedding-friendly document.

    Args:
        sample: JSON sample dictionary
        include_ground_truth: Whether to include ground_truths field
        preprocess: If True, convert to flattened text format; if False, return raw JSON

    We prioritize "signal" fields:
    - metadata: id/tags/task_types
    - payload: input_datas[*].content_raw (your XML snippets)
    - optionally: ground_truths (labels/answers) if you explicitly enable it

    Tip: Usually keep `include_ground_truth=False` to avoid "training on the answer".
    """
    # If no preprocessing, return raw JSON
    if not preprocess:
        import json
        # Remove ground_truths if not included
        if not include_ground_truth and "ground_truths" in sample:
            sample = sample.copy()
            del sample["ground_truths"]
        return json.dumps(sample, indent=2, ensure_ascii=False)

    lines: List[str] = []

    for key in ("id", "tags", "task_types"):
        if key in sample:
            lines.append(f"{key}: {json_value_to_text(sample.get(key))}")

    input_datas = sample.get("input_datas") or []
    if isinstance(input_datas, list) and input_datas:
        lines.append("input_datas:")
        for idx, item in enumerate(input_datas, start=1):
            if not isinstance(item, dict):
                continue
            filename = json_value_to_text(item.get("filename"))
            timestamp = json_value_to_text(item.get("timestamp"))
            content_raw = json_value_to_text(item.get("content_raw"))

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
        lines.append(f"references: {json_value_to_text(references)}")

    if include_ground_truth and "ground_truths" in sample:
        lines.append(f"ground_truths: {json_value_to_text(sample.get('ground_truths'))}")

    return "\n".join([line for line in lines if line.strip()])


# [WNC] Load and format JSON documents from knowledge base directory
def load_json_docs(
    kdb_dir: Path, include_ground_truth: bool, preprocess_json: bool = True
) -> Tuple[List[str], List[str]]:
    """Load and format all `*.json` files under `kdb_dir`."""
    docs: List[str] = []
    file_paths: List[str] = []

    for path in sorted(kdb_dir.rglob("*.json")):
        try:
            sample = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Skipping %s: failed to parse JSON: %s", path, e)
            continue

        if not isinstance(sample, dict):
            logger.warning("Skipping %s: top-level JSON is not an object", path)
            continue

        doc_text = sample_json_to_doc_text(sample, include_ground_truth, preprocess_json)
        if not doc_text.strip():
            logger.warning("Skipping %s: empty doc text after formatting", path)
            continue

        docs.append(f"source_file: {path}\n{doc_text}")
        file_paths.append(str(path))

    return docs, file_paths


# [WNC] Extract text from PDF using pypdf library
def extract_pdf_text_pypdf(path: Path) -> str:
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
    parts: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text)
    return "\n".join(parts).strip()


# [WNC] Extract text using textract library for various file formats
def extract_with_textract(path: Path) -> str:
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


# [WNC] Extract PDF text using specified extractor (auto|textract|pypdf)
def extract_pdf_text(path: Path, extractor: str) -> str:
    """Extract text from PDF using chosen extractor: auto|textract|pypdf."""
    if extractor == "textract":
        return extract_with_textract(path)
    if extractor == "pypdf":
        return extract_pdf_text_pypdf(path)

    # auto: prefer textract if installed, else pypdf
    try:
        return extract_with_textract(path)
    except ModuleNotFoundError:
        return extract_pdf_text_pypdf(path)


# [WNC] Load mixed document types (JSON, text, PDF) from directory
def load_mixed_docs(
    kdb_dir: Path,
    include_ground_truth: bool,
    allow_pdf: bool,
    pdf_extractor: str,
    preprocess_json: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Load documents from a directory tree.

    Supported formats:
    - `.json`: your structured samples (converted into a text document)
    - `.txt` / `.md`: raw text files
    - `.pdf`: extracted to text (requires `pypdf`) when `allow_pdf=True`
    """
    docs: List[str] = []
    file_paths: List[str] = []

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
                doc_text = sample_json_to_doc_text(sample, include_ground_truth, preprocess_json)
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
                    text = extract_pdf_text(path, extractor=pdf_extractor)
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


# [WNC] Load documents using textract backend for various file types
def load_docs_with_textract(
    kdb_dir: Path, include_ground_truth: bool, preprocess_json: bool = False
) -> Tuple[List[str], List[str]]:
    """
    Ingest backend: textract

    - `.json`: formatted into text (same as simple mode)
    - other files: extracted to text by `textract` (PDF/DOCX/PPTX/CSV/etc.)
    """
    docs: List[str] = []
    file_paths: List[str] = []

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
                doc_text = sample_json_to_doc_text(sample, include_ground_truth, preprocess_json)
                if not doc_text.strip():
                    logger.warning(
                        "Skipping %s: empty doc text after formatting", path
                    )
                    continue
                docs.append(f"source_file: {path}\n{doc_text}")
                file_paths.append(str(path))
                continue

            text = extract_with_textract(path)
            if not text:
                logger.warning("Skipping %s: empty textract output", path)
                continue
            docs.append(f"source_file: {path}\n{text}")
            file_paths.append(str(path))
        except Exception as e:
            logger.warning("Skipping %s: failed to read/format: %s", path, e)

    return docs, file_paths


# [WNC] Write manifest of index inputs for verification
def write_index_input_manifest(
    docs: List[str],
    file_paths: List[str],
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


# [WNC] Configuration loading helpers
def load_config_module(config_path: Path) -> ModuleType:
    """
    Load a Python module from a file path as a configuration module.

    Args:
        config_path: Path to the config file (e.g., test_config.py)

    Returns:
        Loaded module object
    """
    module_name = f"test_config_{abs(hash(str(config_path.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load config module: {config_path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses (and other reflection) expect the module to exist in sys.modules.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_test_config(config_path: Path) -> Any:
    """
    Load test configuration from a Python file.

    Args:
        config_path: Path to the config file (must define CONFIG variable)

    Returns:
        The CONFIG object from the module
    """
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    module = load_config_module(config_path)
    config = getattr(module, "CONFIG", None)
    if config is None:
        raise SystemExit(f"Config file must define `CONFIG`: {config_path}")
    return config


# [WNC] LightRAG initialization and operation helpers
def initialize_rag_storages(rag: LightRAG) -> None:
    """
    Initialize LightRAG storages (must be called before indexing/querying).

    Args:
        rag: LightRAG instance
    """
    loop = always_get_an_event_loop()
    with phase("Initialize storages"):
        loop.run_until_complete(rag.initialize_storages())


def finalize_rag_storages(rag: LightRAG) -> None:
    """
    Finalize LightRAG storages (must be called at the end).

    Args:
        rag: LightRAG instance
    """
    loop = always_get_an_event_loop()
    with phase("Finalize storages"):
        loop.run_until_complete(rag.finalize_storages())


def index_documents(
    rag: LightRAG,
    docs: List[str],
    file_paths: List[str],
    max_parallel_insert: int = 2,
) -> None:
    """
    Index documents into LightRAG using simple/textract backend.

    Args:
        rag: LightRAG instance (storages must be initialized)
        docs: List of document texts
        file_paths: List of source file paths (same length as docs)
        max_parallel_insert: Concurrent document processing limit
    """
    with phase("Index documents"):
        logger.info(
            "Indexing %s documents (max_parallel_insert=%s)",
            len(docs),
            max_parallel_insert,
        )
        doc_ids = [compute_mdhash_id(d, prefix="doc-") for d in docs]
        for doc_id, path in zip(doc_ids, file_paths):
            logger.info("Index input: doc_id=%s file=%s", doc_id, path)
        rag.insert(docs, file_paths=file_paths, ids=doc_ids)


def run_query(
    rag: LightRAG,
    question: str,
    mode: str = "hybrid",
    chunk_top_k: int | None = None,
    enable_rerank: bool | None = None,
    rerank_top_n: int | None = None,
) -> str:
    """
    Run a query against LightRAG (text-only, no multimodal).

    Args:
        rag: LightRAG instance (storages must be initialized)
        question: Query text
        mode: Query mode (naive|local|global|hybrid|mix|bypass)
        chunk_top_k: Maximum number of chunks to return (optional)
        enable_rerank: Enable reranking (optional, uses QueryParam default if None)
        rerank_top_n: Maximum chunks after reranking (optional)

    Returns:
        Answer text
    """
    loop = always_get_an_event_loop()
    with phase("Query"):
        logger.info("Question:\n%s", question.strip())
        query_param = QueryParam(mode=mode)
        if chunk_top_k is not None:
            query_param.chunk_top_k = int(chunk_top_k)
        if enable_rerank is not None:
            query_param.enable_rerank = enable_rerank
        if rerank_top_n is not None:
            query_param.rerank_top_n = rerank_top_n
        answer = rag.query(question, param=query_param)
    return answer


def enforce_text_only_mode(config: Any) -> None:
    """
    Enforce text-only mode by disabling multimodal features in config.

    This function modifies the config object in-place to ensure:
    - use_multimodal_query is set to False
    - RAG-Anything multimodal features are disabled

    Args:
        config: Configuration object with ingest settings
    """
    # Disable multimodal query for RAG-Anything if configured
    if hasattr(config, "ingest") and hasattr(config.ingest, "raganything"):
        if hasattr(config.ingest.raganything, "use_multimodal_query"):
            config.ingest.raganything.use_multimodal_query = False
            logger.info("Enforced use_multimodal_query=False for text-only mode")


# ========== In-process Reranker (Sentence-Transformers CrossEncoder) ==========

# Lazy-loaded singleton CrossEncoder for reranking
_reranker: "CrossEncoder | None" = None  # type: ignore


def get_reranker(model_path: str) -> "CrossEncoder":  # type: ignore
    """
    Lazy-load and return the CrossEncoder reranker singleton.

    Args:
        model_path: Path to local model or HuggingFace model name

    Returns:
        CrossEncoder instance for reranking
    """
    global _reranker

    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for reranking. "
                "Install it with: pip install sentence-transformers"
            )

        logger.info("Loading reranker model from: %s", model_path)
        start = perf_counter()

        # Load on CPU to avoid GPU memory issues with embeddings
        _reranker = CrossEncoder(model_path, device="cpu")

        elapsed = perf_counter() - start
        logger.info("Reranker model loaded in %.2fs", elapsed)

    return _reranker


async def rerank_model_func(
    query: str,
    documents: list[str],
    top_n: int | None = None,
    extra_body: dict | None = None,
) -> list[dict]:
    """
    In-process reranker using sentence-transformers CrossEncoder.

    Args:
        query: Search query
        documents: List of document texts to rerank
        top_n: Maximum number of results to return (None = return all)
        extra_body: Extra parameters (unused, for API compatibility)

    Returns:
        List of dicts with format: [{"index": idx, "relevance_score": score}, ...]
        Sorted by relevance_score descending
    """
    if not documents:
        logger.warning("rerank_model_func: No documents to rerank")
        return []

    # Get reranker from config (passed via closure in main())
    # This will be set when creating the partial function
    model_path = getattr(rerank_model_func, "_model_path", None)
    if not model_path:
        raise ValueError("Reranker model_path not configured")

    reranker = get_reranker(model_path)

    # Create query-document pairs
    pairs = [(query, doc) for doc in documents]

    # Score all pairs
    logger.debug("Reranking %d documents with query: %s", len(documents), query[:100])
    start = perf_counter()

    scores = reranker.predict(pairs)

    elapsed = perf_counter() - start
    logger.debug("Reranking completed in %.2fs (%.0f docs/sec)",
                 elapsed, len(documents) / elapsed if elapsed > 0 else 0)

    # Create index-score pairs and sort by score descending
    results = [
        {"index": idx, "relevance_score": float(score)}
        for idx, score in enumerate(scores)
    ]
    results.sort(key=lambda x: x["relevance_score"], reverse=True)

    # Apply top_n if specified
    if top_n is not None and top_n > 0:
        results = results[:top_n]

    logger.debug("Rerank scores - min: %.4f, max: %.4f, mean: %.4f",
                 min(r["relevance_score"] for r in results) if results else 0,
                 max(r["relevance_score"] for r in results) if results else 0,
                 sum(r["relevance_score"] for r in results) / len(results) if results else 0)

    return results
