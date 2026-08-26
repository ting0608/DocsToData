from __future__ import annotations

import json
from typing import Protocol

import requests
from openai import OpenAI

from rag.config import Settings as OpenAISettings
from rag.models import TokenUsage
from rag_aws.config import BedrockSettings
from rag_local.config import LocalSettings


class LLMGateway(Protocol):
    """Common interface every LLM/embedding provider must satisfy.

    English: This is the "LLM Gateway" from architecturev2.txt section 5 —
    the application talks to `generate()`/`embed()` only, so swapping Ollama
    for Bedrock (or OpenAI) never requires touching pipeline, retrieval, or
    graph code. Every implementation also exposes `last_usage`, updated
    after each `generate()` call, so callers can accumulate token counts for
    cost estimation (`rag/pricing.py`) without changing `generate()`'s
    return type.
    中文: 對應 architecturev2.txt 第 5 節的「LLM Gateway」。應用程式只透過
    `generate()`/`embed()` 溝通，因此更換 Ollama、Bedrock 或 OpenAI 時完全
    不需要修改 pipeline、檢索或 graph 相關程式碼。每個實作都會在每次
    `generate()` 呼叫後更新 `last_usage`，讓呼叫端可以累加 token 數以估算成本
    （`rag/pricing.py`），且不需要更動 `generate()` 的回傳型別。
    """

    last_usage: TokenUsage

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_one(self, text: str) -> list[float]: ...


class OpenAIGateway:
    """LLM Gateway backed by the OpenAI API."""

    def __init__(self, settings: OpenAISettings) -> None:
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.last_usage = TokenUsage()

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        completion = self.client.chat.completions.create(
            model=model or self.settings.openai_chat_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage = completion.usage
        self.last_usage = TokenUsage(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )
        return completion.choices[0].message.content or ""

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        batch_size = 128
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(
                model=self.settings.openai_embed_model,
                input=batch,
            )
            vectors.extend([row.embedding for row in response.data])
        return vectors

    def embed_one(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.settings.openai_embed_model,
            input=text,
        )
        return response.data[0].embedding


class OllamaGateway:
    """LLM Gateway backed by a local Ollama server."""

    def __init__(self, settings: LocalSettings) -> None:
        self.settings = settings
        self.last_usage = TokenUsage()

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> str:
        url = f"{self.settings.ollama_base_url}/api/chat"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        options: dict[str, object] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        payload = {
            "model": model or self.settings.ollama_chat_model,
            "stream": False,
            "messages": messages,
            "options": options,
        }
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        # Ollama's non-streamed /api/chat response includes prompt_eval_count
        # (input tokens) and eval_count (output tokens) — free to run, but we
        # still capture these so latency/throughput can be compared apples to
        # apples against OpenAI/Bedrock in the evaluation dashboard.
        self.last_usage = TokenUsage(
            prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(data.get("eval_count", 0) or 0),
        )
        return data.get("message", {}).get("content", "")

    def _embed_one_raw(self, text: str) -> list[float]:
        url = f"{self.settings.ollama_base_url}/api/embeddings"
        payload = {"model": self.settings.ollama_embed_model, "prompt": text}
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        if "embedding" not in data:
            raise ValueError(f"Invalid embedding response: {data}")
        return data["embedding"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one_raw(text) for text in texts]

    def embed_one(self, text: str) -> list[float]:
        return self._embed_one_raw(text)


class BedrockGateway:
    """LLM Gateway backed by Amazon Bedrock.

    English: Uses the Bedrock Converse API for generation (works uniformly
    across Anthropic/Meta/Amazon models) and Titan Text Embeddings V2 for
    embeddings via InvokeModel. Requires `boto3` and AWS credentials with
    `bedrock:InvokeModel`/`bedrock:Converse` permissions, plus explicit model
    access granted in the target region (Bedrock model access is opt-in per
    account/region). This class has NOT been exercised against a live AWS
    account in this environment — verify model IDs and IAM permissions in
    your own account before relying on it.
    中文: 使用 Bedrock 的 Converse API 進行生成（可統一支援 Anthropic/Meta/
    Amazon 等模型），並透過 InvokeModel 呼叫 Titan Text Embeddings V2 產生
    embedding。需要 `boto3` 以及具備 `bedrock:InvokeModel`/`bedrock:Converse`
    權限的 AWS 憑證，且目標 region 需先開通對應模型的存取權限（Bedrock 的模型
    存取權是逐帳號/逐區域申請的）。此類別尚未在真實 AWS 帳號中實際驗證過，
    正式使用前請自行確認 model ID 與 IAM 權限設定。
    """

    def __init__(self, settings: BedrockSettings) -> None:
        self.settings = settings
        self._client = None
        self.last_usage = TokenUsage()

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise RuntimeError(
                    "boto3 is required for the Bedrock provider. Install it with "
                    "`pip install boto3` and configure AWS credentials."
                ) from exc
            self._client = boto3.client("bedrock-runtime", region_name=self.settings.bedrock_region)
        return self._client

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> str:
        client = self._get_client()
        kwargs: dict[str, object] = {
            "modelId": model or self.settings.bedrock_chat_model_id,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {
                "temperature": temperature,
                "maxTokens": max_tokens or 1024,
            },
        }
        if system:
            kwargs["system"] = [{"text": system}]

        response = client.converse(**kwargs)
        usage = response.get("usage", {}) or {}
        self.last_usage = TokenUsage(
            prompt_tokens=int(usage.get("inputTokens", 0) or 0),
            completion_tokens=int(usage.get("outputTokens", 0) or 0),
        )
        content = response.get("output", {}).get("message", {}).get("content", [])
        for block in content:
            if "text" in block:
                return block["text"]
        return ""

    def _embed_one_raw(self, text: str) -> list[float]:
        client = self._get_client()
        body = json.dumps(
            {
                "inputText": text,
                "dimensions": self.settings.vector_dim,
                "normalize": True,
            }
        )
        response = client.invoke_model(
            modelId=self.settings.bedrock_embed_model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        if "embedding" not in result:
            raise ValueError(f"Invalid Bedrock embedding response: {result}")
        return result["embedding"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Titan Text Embeddings V2 does not support multi-text batches per
        # InvokeModel call, so we embed sequentially.
        return [self._embed_one_raw(text) for text in texts]

    def embed_one(self, text: str) -> list[float]:
        return self._embed_one_raw(text)
