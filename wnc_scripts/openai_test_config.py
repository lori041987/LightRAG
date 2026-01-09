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

    working_dir: str = "/srv/ai/LightRAG/rag_storage/openai_3gpp"
    working_dir: str = "/srv/ai/LightRAG/rag_storage/openai_test_json_251229_1"

    mode: Literal["naive", "local", "global", "hybrid", "mix", "bypass"] = "hybrid"
    skip_index: bool = False
    question: str = "What is client's IP address?"
    # Number of documents processed concurrently during `rag.insert(...)`.
    # Set to 1 for easier-to-read logs (no interleaving).
    max_parallel_insert: int = 1

    ingest: IngestSettings = IngestSettings()
    openai: OpenAISettings = OpenAISettings()


# Default config used by `wnc_scripts/openai_test.py` unless `--config` points elsewhere.
#
# You can either change the defaults above, or mutate the instance here, e.g.:
#   CONFIG.ingest.backend = "raganything"
#   CONFIG.ingest.raganything.parse_method = "txt"
#   CONFIG.ingest.raganything.enable_equation_processing = False
CONFIG = OpenAITestConfig()
