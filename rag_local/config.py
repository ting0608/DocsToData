from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class LocalSettings:
    """Runtime settings for local Ollama RAG flow.

    English: Keeps Ollama endpoint/model settings in one place.
    中文: 集中管理 Ollama 的端點與模型設定。
    """

    ollama_base_url: str
    ollama_chat_model: str
    ollama_embed_model: str
    vector_dim: int


def load_local_settings() -> LocalSettings:
    """Load local RAG settings from `.env`.

    English: Uses sensible defaults matching your installed models.
    中文: 預設值對應你目前已安裝的模型。
    """

    load_dotenv()
    return LocalSettings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip(),
        ollama_chat_model=os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b").strip(),
        ollama_embed_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest").strip(),
        vector_dim=int(os.getenv("OLLAMA_VECTOR_DIM", "768")),
    )

