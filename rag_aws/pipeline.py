from __future__ import annotations

from rag.base_pipeline import BaseRagPipeline
from rag.llm_gateway import BedrockGateway
from rag_aws.config import BedrockSettings, load_bedrock_settings


class BedrockRagPipeline(BaseRagPipeline):
    """RAG pipeline backed by Amazon Bedrock (chat + embeddings).

    English: Thin subclass of `BaseRagPipeline` wiring up the Bedrock
    `LLMGateway` — this is the "Cloud Mode" provider from
    architecturev2.txt sections 5 and 9. Requires `boto3`, valid AWS
    credentials, and Bedrock model access granted for
    `BEDROCK_CHAT_MODEL_ID`/`BEDROCK_EMBED_MODEL_ID` in `BEDROCK_REGION`.
    This class has not been exercised against a live AWS account in this
    environment; treat it as a starting point to verify against your own
    account (see docs/aws_deployment.md for the manual verification steps).
    中文: 只負責組裝 Bedrock 專屬的 `LLMGateway`——對應
    architecturev2.txt 第 5、9 節的「雲端模式」provider。需要 `boto3`、有效的
    AWS 憑證，以及在 `BEDROCK_REGION` 已開通
    `BEDROCK_CHAT_MODEL_ID`/`BEDROCK_EMBED_MODEL_ID` 的模型存取權限。此類別
    尚未在真實 AWS 帳號中實際驗證過，請視為起點，依 docs/aws_deployment.md
    中的手動驗證步驟在自己的帳號中確認可用性。
    """

    def __init__(self, settings: BedrockSettings | None = None) -> None:
        self.settings = settings or load_bedrock_settings()
        gateway = BedrockGateway(self.settings)
        super().__init__(
            gateway=gateway,
            vector_dim=self.settings.vector_dim,
            embedding_model_name=self.settings.bedrock_embed_model_id,
            default_index_dir="data/index_bedrock",
            chat_model_name=self.settings.bedrock_chat_model_id,
        )
