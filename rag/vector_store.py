from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from rag.models import Chunk, SearchHit


class FaissStore:
    """Thin wrapper around FAISS with chunk metadata persistence.

    English: Stores vectors in FAISS and chunk metadata in JSON.
    中文: 向量存到 FAISS，文字與來源 metadata 存到 JSON。
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalize vectors for inner-product similarity search.

        English: With normalization, inner product approximates cosine similarity.
        中文: 正規化後用內積搜尋，可近似 cosine similarity。
        """

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return vectors / norms

    def add(self, embeddings: list[list[float]], chunks: list[Chunk]) -> None:
        """Add embeddings and matching chunk metadata into the index.

        English: Embedding count must match chunk count one-to-one.
        中文: 向量數量必須與 chunk 數量一一對應。
        """

        if len(embeddings) != len(chunks):
            raise ValueError("embeddings and chunks length mismatch")
        if not embeddings:
            return

        arr = np.array(embeddings, dtype="float32")
        if arr.shape[1] != self.dim:
            raise ValueError(f"embedding dim {arr.shape[1]} does not match index dim {self.dim}")
        arr = self._normalize(arr)
        self.index.add(arr)
        self.chunks.extend(chunks)

    def search(self, embedding: list[float], top_k: int = 5) -> list[SearchHit]:
        """Search top-k nearest chunks for a query embedding.

        English: Returns chunk objects plus similarity score.
        中文: 回傳包含 chunk 與相似度分數的搜尋結果。
        """

        if self.index.ntotal == 0:
            return []
        query = np.array([embedding], dtype="float32")
        if query.shape[1] != self.dim:
            raise ValueError(f"query dim {query.shape[1]} does not match index dim {self.dim}")
        query = self._normalize(query)
        scores, ids = self.index.search(query, top_k)

        hits: list[SearchHit] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            hits.append(SearchHit(chunk=self.chunks[idx], score=float(score)))
        return hits

    def save(self, out_dir: str) -> None:
        """Persist FAISS index and metadata files.

        English: Writes `index.faiss` and `chunks.json`.
        中文: 會輸出 `index.faiss` 與 `chunks.json`。
        """

        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "index.faiss"))
        with (path / "chunks.json").open("w", encoding="utf-8") as f:
            json.dump([c.__dict__ for c in self.chunks], f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, in_dir: str, dim: int) -> "FaissStore":
        """Load index and metadata from disk.

        English: Raises file errors when required files are missing.
        中文: 若必要檔案不存在，會拋出檔案錯誤。
        """

        path = Path(in_dir)
        if not (path / "index.faiss").exists():
            raise FileNotFoundError(f"Missing FAISS index at {(path / 'index.faiss')}")
        if not (path / "chunks.json").exists():
            raise FileNotFoundError(f"Missing chunks metadata at {(path / 'chunks.json')}")

        store = cls(dim=dim)
        store.index = faiss.read_index(str(path / "index.faiss"))
        with (path / "chunks.json").open("r", encoding="utf-8") as f:
            raw_chunks = json.load(f)
        store.chunks = [Chunk(**item) for item in raw_chunks]
        return store
