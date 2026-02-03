"""
Configuration for LightRAG test scripts (openai_test.py, ollama_test.py, etc.).

Edit this file to change defaults instead of relying on CLI flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


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
    backend: Literal["simple", "textract", "raganything"] = "textract"
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
    chat_model: str = "gpt-5.1-chat-latest"  # Changed from gpt-4o-mini for better reliability (but lower TPM limit)
    vision_model: str = "gpt-5.1-chat-latest"
    embed_model: str = "text-embedding-3-small"
    # embed_dim: Embedding dimension - MUST match the model's actual output dimension
    # This tells LightRAG what vector size to expect from the embedding model
    # IMPORTANT: This is NOT a parameter sent to the model - it's a declaration of what the model outputs
    # Common dimensions (native/default):
    #   - text-embedding-3-small: 1536 (OpenAI supports API reduction, LightRAG passes it)
    #   - text-embedding-3-large: 3072 (OpenAI supports API reduction, LightRAG passes it)
    # Setting the wrong dimension will cause mismatch errors!
    embed_dim: int = 1536  # Correct dimension for text-embedding-3-small (native output)


@dataclass
class OllamaSettings:
    # Ollama host (None = use default localhost:11434)
    host: str | None = None
    timeout: int | None = None
    api_key_env: str | None = None  # Optional: set to env var name if using auth

    # Models (must be pulled via `ollama pull` first)
    chat_model: str = "qwen3:8b"
    #chat_model: str = "qwen2.5:7b"
    embed_model: str = "bge-m3:567m"
    #embed_model: str = "nomic-embed-text:v1.5"
    # embed_dim: MUST match the model's actual output dimension (see OpenAISettings for detailed explanation)
    # bge-m3 native output: 1024 dimensions (fixed)
    # nomic-embed-text:v1.5 output: 768 dimensions (fixed)
    # Note: Ollama API supports dimension reduction via 'dimensions' parameter,
    # but LightRAG does NOT currently pass this parameter to Ollama embeddings
    embed_dim: int = 1024

    # Think mode for models that support it (qwen3, gpt-oss, deepseek-v3, deepseek-r1)
    # False = disable chain-of-thought reasoning (much faster)
    # True = enable chain-of-thought reasoning (slower but more detailed)
    # None = don't set (use model default)
    think: bool | None = False


@dataclass
class OpenAITestConfig:
    #kdb_dir: str = "/srv/ai/LightRAG/wnc_kdb"
    #kdb_dir: str = "/srv/ai/LightRAG/wnc_kdb/test_json_260126"
    kdb_dir: str = "/srv/ai/LightRAG/wnc_kdb/test_3gpp"

    #working_dir: str = "/srv/ai/LightRAG/rag_storage/test_json_260126"
    #working_dir: str = "/srv/ai/LightRAG/rag_storage/wnc_kdb"
    working_dir: str = "/srv/ai/LightRAG/rag_storage/test_3gpp_openai"
    #working_dir: str = "/srv/ai/LightRAG/rag_storage/test_3gpp_ollama"

    # Query mode - controls retrieval strategy
    # Options: naive, local, global, hybrid, mix, bypass
    # For detailed explanation with examples, see: wnc_docs/query_modes_explanation.md
    # Quick summary:
    #   - hybrid (recommended): combines entities + relationships for comprehensive results
    #   - local: entity-focused, best for "What is X?" questions
    #   - global: relationship-focused, best for "How are things connected?"
    #   - naive: simple vector search only (fastest)
    mode: Literal["naive", "local", "global", "hybrid", "mix", "bypass"] = "hybrid"
    skip_index: bool = False

    #question: str = "What is client's IP address?"
    question: str = "When a UE receives a `Manage UE Policy Command` message from the network, what is the expected response from the UE to confirm it has applied the new rules?"

    # Query parameters
    # chunk_top_k: Maximum number of text chunks sent to LLM for answer generation
    chunk_top_k: int = 3

    # Vector similarity threshold
    # cosine_threshold: Minimum cosine similarity score for entity/edge retrieval (0.0-1.0)
    # Higher values = stricter filtering (only very similar entities), lower recall
    # Lower values = looser filtering (more entities), higher recall but more noise
    cosine_threshold: float = 0.3

    # Reranking configuration (in-process reranker using sentence-transformers)
    # enable_rerank: Enable reranking of retrieved chunks before sending to LLM
    # - True: Apply reranking to improve relevance of top chunks (recommended)
    # - False: Use original vector similarity scores only
    enable_rerank: bool = False
    # rerank_model_path: Path to local CrossEncoder model or HuggingFace model name
    # Recommended models:
    #   - ms-marco-MiniLM-L6-v2: Fast, good quality (50MB)
    #   - ms-marco-MiniLM-L12-v2: Better quality, slower (130MB)
    #   - bge-reranker-base: High quality, multilingual (280MB)
    rerank_model_path: str = "/srv/ai/models/ms-marco-MiniLM-L6-v2"
    # rerank_top_n: Maximum number of chunks to keep after reranking
    # - None: Use chunk_top_k value (recommended)
    # - int: Override with specific value
    rerank_top_n: int | None = None
    # min_rerank_score: Minimum rerank score threshold (filters low-relevance chunks)
    # - 0.0: Keep all reranked chunks (default)
    # - Higher values: More aggressive filtering
    min_rerank_score: float = 0.0

    # Text chunk boost configuration (affects only chunks sent to LLM, not entities/edges)
    # enable_source_path_boost: If True, prioritizes chunks from specific source paths
    # source_path_boosts: List of {"prefix": "/path/", "boost": 0.05} rules
    # Boost value is added to chunk's cosine similarity score before ranking/selection
    # Example: chunk with 0.40 similarity + 0.05 boost = 0.45 adjusted score
    enable_source_path_boost: bool = False
    source_path_boosts: list[dict[str, Any]] = field(default_factory=lambda: [
        {"prefix": "/srv/ai/LightRAG/wnc_kdb/test_json_260126/", "boost": 0.1}
    ])

    # Document-level concurrency: how many documents to index in parallel
    # Independent from llm_model_max_async (which controls chunk-level concurrency)
    # Set to 1 for easier-to-read logs (no interleaving).
    max_parallel_insert: int = 1

    # LLM concurrency control
    # Controls maximum number of concurrent LLM calls (entity extraction, graph merging)
    # - Entity extraction: uses llm_model_max_async directly (chunk_max_async = llm_model_max_async)
    # - Graph merging: uses 2x multiplier (graph_max_async = llm_model_max_async * 2)
    # Recommended values:
    #   - 1: Sequential processing, no concurrency (safest for slow/limited hardware)
    #   - 2-4: Moderate concurrency (good for local Ollama with decent hardware)
    #   - 8+: High concurrency (only for powerful hardware or remote APIs like OpenAI)
    # Default: 1 for Ollama (conservative), can increase for OpenAI
    llm_model_max_async: int = 1

    # Re-indexing strategy
    # - "skip": Skip indexing entirely (same as skip_index=True)
    # - "incremental": Only index new files not already in storage (default)
    # - "force": Force re-index all files, clearing existing storage first
    reindex_strategy: Literal["skip", "incremental", "force"] = "incremental"

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
    # WNC log length controls (comparison):
    # - `wnc_log_trace_content_limit`: caps verbose TRACE-level logs only.
    #   Use this to reduce the noise from very chatty trace instrumentation.
    # - `wnc_log_content_limit`: would cap inputs/outputs for ALL
    #   WNC log levels (INFO/DEBUG/WARNING/ERROR/TRACE). Use that when INFO logs
    #   are still too large because they include big fields like `content`.
    #
    # WNC trace log content limit: max chars to show for content in TRACE logs.
    # Set to 0 for unlimited, or positive int for max length.
    wnc_log_trace_content_limit: int = 10000
    # WNC log content limit: max chars for large string fields in ALL logs (inputs/outputs/note/side_effects)
    # Set to 0 for unlimited, or positive int for max length
    # Truncates string values inside inputs/outputs dicts before JSON serialization to keep JSON valid
    wnc_log_content_limit: int = 0  # 0 = unlimited

    # Cache configuration
    # Enable LLM response caching to avoid redundant API calls
    enable_llm_cache: bool = False
    # Enable caching specifically for entity extraction steps
    enable_llm_cache_for_entity_extract: bool = False

    ingest: IngestSettings = IngestSettings()
    openai: OpenAISettings = OpenAISettings()
    ollama: OllamaSettings = OllamaSettings()


# Default config used by test scripts (openai_test.py, ollama_test.py) unless `--config` points elsewhere.
#
# You can either change the defaults above, or mutate the instance here, e.g.:
#   CONFIG.ingest.backend = "raganything"
#   CONFIG.ingest.raganything.parse_method = "txt"
#   CONFIG.ingest.raganything.enable_equation_processing = False
CONFIG = OpenAITestConfig()
