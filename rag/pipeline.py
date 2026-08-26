from __future__ import annotations

from rag.base_pipeline import BaseRagPipeline
from rag.config import Settings, load_settings
from rag.llm_gateway import OpenAIGateway


class RagPipeline(BaseRagPipeline):
    """High-level RAG workflow backed by OpenAI (chat + embeddings).

    English: Thin subclass of `BaseRagPipeline` that only wires up the
    OpenAI-specific `LLMGateway`. All ingestion/retrieval/generation logic
    lives in `BaseRagPipeline` so it is shared with `LocalRagPipeline`
    (Ollama) and `BedrockRagPipeline` (AWS) per architecturev2.txt section 5.
    中文: 只負責組裝 OpenAI 專屬的 `LLMGateway`，其餘 ingestion/檢索/生成邏輯
    全部放在 `BaseRagPipeline`，與 `LocalRagPipeline`（Ollama）、
    `BedrockRagPipeline`（AWS）共用，對應 architecturev2.txt 第 5 節。
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        gateway = OpenAIGateway(self.settings)
        super().__init__(
            gateway=gateway,
            vector_dim=self.settings.vector_dim,
            embedding_model_name=self.settings.openai_embed_model,
            default_index_dir="data/index",
            chat_model_name=self.settings.openai_chat_model,
        )
