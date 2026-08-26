from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A retrievable text chunk with source metadata.

    English: This is the atomic unit stored in the vector index. Metadata fields
    added in v2 (document_id, created_at, content_hash, embedding_model, chunk_size,
    parent_id, parent_text) are optional so existing `chunks.json` files created by
    v1 can still be loaded without migration.
    中文: 這是向量索引中的最小檢索單位，附帶來源資訊。v2 新增的 metadata 欄位皆為
    optional，讓舊版 `chunks.json` 仍可正常載入，不需要資料遷移。
    """

    chunk_id: str
    source: str
    page: int
    text: str
    document_id: str | None = None
    created_at: str | None = None
    content_hash: str | None = None
    embedding_model: str | None = None
    chunk_size: int | None = None
    chunking_strategy: str | None = None
    parent_id: str | None = None
    parent_text: str | None = None


@dataclass
class SearchHit:
    """A retrieval result item with similarity score.

    English: `score` is cosine-like similarity after vector normalization
    when it comes straight from `FaissStore.search`. After the hybrid
    retrieve/rerank stage (`rag/retrieval.py`) it instead holds a fused
    RRF score or a lexical rerank score — always treat `score` as a
    ranking signal, not a calibrated probability.
    中文: 直接來自 `FaissStore.search` 時，`score` 是向量正規化後的相似度
    （近似 cosine）。經過混合檢索/重排序（`rag/retrieval.py`）後，則會變成
    RRF 融合分數或詞彙重排序分數——請一律視為排序訊號，而非經過校準的機率值。
    """

    chunk: Chunk
    score: float


@dataclass
class TokenUsage:
    """Prompt/completion token counts for one or more LLM generate() calls.

    English: Accumulated per-request by `LLMGateway` implementations
    (`rag/llm_gateway.py`) so the evaluation framework (architecturev2.txt
    section 7, "System Metrics") can turn token counts into an estimated
    cost per query via `rag/pricing.py`.
    中文: 由各 `LLMGateway` 實作（`rag/llm_gateway.py`）依請求累加，讓評估框架
    （對應 architecturev2.txt 第 7 節「System Metrics」）可透過
    `rag/pricing.py` 將 token 數換算成每次查詢的預估成本。
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class SourceCitation:
    """A structured citation returned alongside an answer.

    English: This is the v2 response shape from architecturev2.txt section 6
    — replaces the v1 flat `sources: list[str]` (document names only) with
    per-chunk citations that also carry page number and chunk_id so the
    frontend can deep-link back to the exact retrieved passage.
    中文: 對應 architecturev2.txt 第 6 節的 v2 回應格式——取代 v1 單純的
    `sources: list[str]`（僅有檔名），改為每個 chunk 各自一筆的 citation，
    附帶頁碼與 chunk_id，方便前端之後可以連結回精確的檢索段落。
    """

    document: str
    page: int
    chunk_id: str
    score: float
