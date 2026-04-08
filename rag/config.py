from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables.

    English: Centralized config for OpenAI models and vector dimensions.
    中文: 從環境變數讀取統一設定，包含 OpenAI 模型與向量維度。
    """

    openai_api_key: str
    openai_chat_model: str
    openai_embed_model: str
    vector_dim: int


def load_settings() -> Settings:
    """Load `.env` values and return strongly typed settings.

    English: Fails early if API key is missing to avoid runtime surprises.
    中文: 若缺少 API Key 會立即報錯，避免執行中才發現設定問題。
    """

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required. Add it to your .env file.")

    chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip()
    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small").strip()
    vector_dim = int(os.getenv("VECTOR_DIM", "1536"))

    return Settings(
        openai_api_key=api_key,
        openai_chat_model=chat_model,
        openai_embed_model=embed_model,
        vector_dim=vector_dim,
    )
