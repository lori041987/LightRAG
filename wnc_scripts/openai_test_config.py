"""
Configuration for `wnc_scripts/openai_test.py`.

Edit this file to change defaults instead of relying on CLI flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class RagAnythingSettings:
    output_dir: str | None = None
    parse_method: Literal["auto", "txt", "ocr"] = "auto"
    parser: Literal["mineru", "docling"] = "mineru"
    enable_image_processing: bool = True
    enable_table_processing: bool = True
    enable_equation_processing: bool = True
    max_concurrent_files: int = 1
    use_multimodal_query: bool = False


@dataclass
class IngestSettings:
    backend: Literal["simple", "textract", "raganything"] = "simple"
    include_ground_truth: bool = True
    # Pre-process JSON files into flattened text format (key: value)
    # If False, index raw JSON as-is
    preprocess_json: bool = False

    # Only used when `backend == "simple"`
    allow_pdf: bool = True
    pdf_extractor: Literal["auto", "textract", "pypdf"] = "auto"

    raganything: RagAnythingSettings = RagAnythingSettings()


@dataclass
class OpenAISettings:
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    timeout: int | None = None

    # Azure (optional)
    use_azure: bool = False
    azure_deployment: str | None = None
    api_version: str | None = None

    # Models
    chat_model: str = "gpt-4o-mini"
    vision_model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"
    embed_dim: int = 1536


@dataclass
class OpenAITestConfig:
    kdb_dir: str = "/srv/ai/LightRAG/wnc_kdb"
    #kdb_dir: str = "/srv/ai/LightRAG/wnc_kdb/3gpp"
    #kdb_dir: str = "/srv/ai/LightRAG/wnc_kdb/test_json_260123"

    #working_dir: str = "/srv/ai/LightRAG/rag_storage/openai_3gpp"
    working_dir: str = "/srv/ai/LightRAG/rag_storage/wnc_kdb"

    mode: Literal["naive", "local", "global", "hybrid", "mix", "bypass"] = "hybrid"
    skip_index: bool = False

    #question: str = "What is client's IP address?"
    question: str = "What is device's IP address?"

    # Query parameters
    # chunk_top_k: Maximum number of text chunks sent to LLM for answer generation
    chunk_top_k: int = 6

    # Vector similarity threshold
    # cosine_threshold: Minimum cosine similarity score for entity/edge retrieval (0.0-1.0)
    # Higher values = stricter filtering (only very similar entities), lower recall
    # Lower values = looser filtering (more entities), higher recall but more noise
    cosine_threshold: float = 0.3

    # Number of documents processed concurrently during `rag.insert(...)`.
    # Set to 1 for easier-to-read logs (no interleaving).
    max_parallel_insert: int = 1

    # Re-indexing strategy
    # - "skip": Skip indexing entirely (same as skip_index=True)
    # - "incremental": Only index new files not already in storage (default)
    # - "force": Force re-index all files, clearing existing storage first
    reindex_strategy: Literal["skip", "incremental", "force"] = "force"

    # Logging configuration
    # LightRAG log level: DEBUG, INFO, WARNING, ERROR
    lightrag_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "DEBUG"
    # WNC custom log level: trace, debug, info, warning, error
    # "trace" will enable freq logs: sanitize_text_for_encoding(), compute_mdhash_id()
    wnc_log_level: Literal["trace", "debug", "info", "warning", "error"] = "info"
    # Enable verbose debug mode (adds extra detailed logging)
    verbose_debug: bool = True
    # WNC log delimiter: separator between log fields
    # - "pipe": use " | " (compact, single line)
    # - "newline": use "\n  " (multi-line, easier to read)
    # - "comma": use ", " (CSV-like)
    wnc_log_delimiter: Literal["pipe", "newline", "comma"] = "newline"
    # WNC trace log content limit: max chars to show for content in trace logs
    # Set to 0 for unlimited, or positive int for max length
    wnc_log_trace_content_limit: int = 10000

    # Cache configuration
    # Enable LLM response caching to avoid redundant API calls
    enable_llm_cache: bool = False
    # Enable caching specifically for entity extraction steps
    enable_llm_cache_for_entity_extract: bool = False

    ingest: IngestSettings = IngestSettings()
    openai: OpenAISettings = OpenAISettings()


# Default config used by `wnc_scripts/openai_test.py` unless `--config` points elsewhere.
#
# You can either change the defaults above, or mutate the instance here, e.g.:
#   CONFIG.ingest.backend = "raganything"
#   CONFIG.ingest.raganything.parse_method = "txt"
#   CONFIG.ingest.raganything.enable_equation_processing = False
CONFIG = OpenAITestConfig()
