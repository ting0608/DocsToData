from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class BedrockSettings:
    """Runtime settings for the Amazon Bedrock LLM/embedding provider.

    English: Kept separate from `rag/config.py` (OpenAI) and
    `rag_local/config.py` (Ollama) so each provider's settings can evolve
    independently, mirroring the existing project convention. Model IDs
    default to Anthropic Claude 3 Haiku (chat) and Titan Text Embeddings V2
    (embedding) — both are commonly available Bedrock models, but you must
    still request model access for your AWS account/region in the Bedrock
    console before these will work.
    中文: 與 `rag/config.py`（OpenAI）、`rag_local/config.py`（Ollama）分開管理，
    延續本專案「各 provider 各自一份設定」的慣例。預設 model ID 為 Anthropic
    Claude 3 Haiku（對話）與 Titan Text Embeddings V2（embedding），這兩個都是
    Bedrock 常見可用模型，但仍需先在 AWS 帳號/region 的 Bedrock 主控台申請該
    模型的存取權限才能實際呼叫成功。
    """

    bedrock_region: str
    bedrock_chat_model_id: str
    bedrock_embed_model_id: str
    vector_dim: int
    s3_bucket: str
    s3_prefix: str


def load_bedrock_settings() -> BedrockSettings:
    """Load Bedrock/S3 settings from `.env`.

    English: Unlike `rag/config.py`, this does NOT raise when values are
    missing — Bedrock is an optional provider and should only be required
    when `LLM_PROVIDER=bedrock` is actually selected (checked by the caller).
    中文: 與 `rag/config.py` 不同，這裡不會在缺少設定時直接報錯——因為 Bedrock
    是可選的 provider，只有在真正選用 `LLM_PROVIDER=bedrock` 時才需要驗證
    (由呼叫端負責檢查)。
    """

    load_dotenv()
    return BedrockSettings(
        bedrock_region=os.getenv("BEDROCK_REGION", os.getenv("AWS_REGION", "us-east-1")).strip(),
        bedrock_chat_model_id=os.getenv(
            "BEDROCK_CHAT_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"
        ).strip(),
        bedrock_embed_model_id=os.getenv("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0").strip(),
        vector_dim=int(os.getenv("BEDROCK_VECTOR_DIM", "1024")),
        s3_bucket=os.getenv("S3_BUCKET", "").strip(),
        s3_prefix=os.getenv("S3_PREFIX", "go-invoice").strip().strip("/"),
    )
