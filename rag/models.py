from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    """A retrievable text chunk with source metadata.

    English: This is the atomic unit stored in the vector index.
    中文: 這是向量索引中的最小檢索單位，附帶來源資訊。
    """

    chunk_id: str
    source: str
    page: int
    text: str


@dataclass
class SearchHit:
    """A retrieval result item with similarity score.

    English: `score` is cosine-like similarity after vector normalization.
    中文: `score` 是向量正規化後的相似度分數（近似 cosine）。
    """

    chunk: Chunk
    score: float
