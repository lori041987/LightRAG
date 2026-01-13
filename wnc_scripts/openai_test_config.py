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
    include_ground_truth: bool = False

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
    #kdb_dir: str = "/srv/ai/LightRAG/wnc_kdb"
    #kdb_dir: str = "/srv/ai/LightRAG/wnc_kdb/3gpp"
    kdb_dir: str = "/srv/ai/LightRAG/wnc_kdb/test_json_251229_1"

    #working_dir: str = "/srv/ai/LightRAG/rag_storage/openai_3gpp"
    working_dir: str = "/srv/ai/LightRAG/rag_storage/openai_test_json_251229_1"

    mode: Literal["naive", "local", "global", "hybrid", "mix", "bypass"] = "hybrid"

    # [WNC] Indexing behavior control
    # Determines how documents are indexed:
    #
    # "skip": Skip indexing entirely, only run query on existing storage
    #         Use when: Testing queries on already-indexed data
    #
    # "incremental": Index new documents only, skip documents that already exist (DEFAULT)
    #                Use when: Adding new documents to existing index
    #                Note: Won't see entity extraction logs for existing docs
    #
    # "force": Delete working_dir and re-index everything from scratch
    #          Use when: Need to see entity extraction logs, test cache disabled, verify changes
    #          WARNING: Deletes all indexed data!
    index_mode: Literal["skip", "incremental", "force"] = "force"

    question: str = "What is client's IP address?"
    # Number of documents processed concurrently during `rag.insert(...)`.
    # Set to 1 for easier-to-read logs (no interleaving).
    max_parallel_insert: int = 1

    # [WNC] Logging configuration
    # Set log level: DEBUG, INFO, WARNING, ERROR
    # Can also be controlled via LOG_LEVEL environment variable
    log_level: str = "INFO"

    # [WNC] Prompt logging controls
    # When True, logs all LLM prompts (system, user, history) to console
    # Can also be controlled via WNC_ENABLE_PROMPT_LOGGING environment variable
    enable_prompt_logging: bool = True

    # When True, dumps full prompts to JSON files in prompt_dump_dir
    # Can also be controlled via WNC_ENABLE_PROMPT_FILE_DUMP environment variable
    enable_prompt_file_dump: bool = False

    # Directory for prompt JSON dumps (only used if enable_prompt_file_dump=True)
    # Can also be controlled via WNC_PROMPT_DUMP_DIR environment variable
    prompt_dump_dir: str = "./prompt_logs"

    # [WNC] Trace ID configuration
    # Prefix for auto-generated trace IDs
    # Set to None to disable trace_id entirely
    trace_id_prefix: str | None = "query"

    # When True, includes timestamp in trace_id
    # Examples:
    #   - True + auto_increment=True:  query_20260113_113008_001
    #   - True + auto_increment=False: query_20260113_113008
    #   - False + auto_increment=True: query_001
    #   - False + auto_increment=False: query (not recommended - all queries have same ID)
    trace_id_include_timestamp: bool = False

    # When True, trace_id will auto-increment for each query (001, 002, 003, ...)
    # When False, no counter is added
    trace_id_auto_increment: bool = True

    # [WNC] Cache configuration
    # Cache exists in TWO stages:
    # 1. Indexing stage: LLM calls for entity extraction & summarization
    # 2. Query stage: LLM calls for keyword extraction & answer generation
    enable_llm_cache: bool = False  # Query-time LLM cache
    enable_llm_cache_for_entity_extract: bool = False  # Indexing-time entity extraction cache

    ingest: IngestSettings = IngestSettings()
    openai: OpenAISettings = OpenAISettings()


# Default config used by `wnc_scripts/openai_test.py` unless `--config` points elsewhere.
#
# You can either change the defaults above, or mutate the instance here, e.g.:
#   CONFIG.ingest.backend = "raganything"
#   CONFIG.ingest.raganything.parse_method = "txt"
#   CONFIG.ingest.raganything.enable_equation_processing = False
CONFIG = OpenAITestConfig()
