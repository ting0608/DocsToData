from __future__ import annotations

from rag.base_pipeline import BaseRagPipeline
from rag.llm_gateway import OllamaGateway
from rag_local.config import LocalSettings, load_local_settings


class LocalRagPipeline(BaseRagPipeline):
    """RAG pipeline backed by a local Ollama server.

    English: Thin subclass of `BaseRagPipeline` that only wires up the
    Ollama-specific `LLMGateway` — same ingestion/retrieval/generation shape
    as the OpenAI pipeline, per architecturev2.txt section 5's LLM Gateway
    pattern.
    中文: 只負責組裝 Ollama 專屬的 `LLMGateway`，ingestion/檢索/生成流程與
    OpenAI pipeline 完全相同，對應 architecturev2.txt 第 5 節的 LLM Gateway
    模式。
    """

    def __init__(self, settings: LocalSettings | None = None) -> None:
        self.settings = settings or load_local_settings()
        gateway = OllamaGateway(self.settings)
        super().__init__(
            gateway=gateway,
            vector_dim=self.settings.vector_dim,
            embedding_model_name=self.settings.ollama_embed_model,
            default_index_dir="data/index_local",
        )
