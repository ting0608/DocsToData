from __future__ import annotations

from pathlib import Path

import requests

from rag.chunking import chunk_pages
from rag.models import SearchHit
from rag.pdf_parser import extract_pdf_pages
from rag.vector_store import FaissStore
from rag_local.config import LocalSettings, load_local_settings


class LocalRagPipeline:
    """RAG pipeline powered by Ollama local models.

    English: Same ingestion/retrieval shape as cloud pipeline, but fully local.
    中文: 與雲端版本流程相同，但模型推論改為本機 Ollama。
    """

    def __init__(self, settings: LocalSettings | None = None) -> None:
        self.settings = settings or load_local_settings()

    def _embed_text(self, text: str) -> list[float]:
        url = f"{self.settings.ollama_base_url}/api/embeddings"
        payload = {"model": self.settings.ollama_embed_model, "prompt": text}
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        if "embedding" not in data:
            raise ValueError(f"Invalid embedding response: {data}")
        return data["embedding"]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed chunk texts one-by-one via Ollama embeddings API."""
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed user question for FAISS retrieval."""
        return self._embed_text(text)

    def ingest_pdf(
        self,
        pdf_path: str,
        out_dir: str = "data/index_local",
        chunk_size: int = 700,
        chunk_overlap: int = 120,
    ) -> dict[str, int]:
        """Parse PDF and create local FAISS index for Ollama embeddings."""
        pages = extract_pdf_pages(pdf_path)
        source_name = Path(pdf_path).stem
        chunks = chunk_pages(
            pages=pages,
            source_name=source_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        embeddings = self.embed_texts([c.text for c in chunks])

        store = FaissStore(dim=self.settings.vector_dim)
        store.add(embeddings=embeddings, chunks=chunks)
        store.save(out_dir=out_dir)

        return {
            "pages": len(pages),
            "chunks": len(chunks),
            "vectors": len(embeddings),
        }

    def retrieve(self, question: str, in_dir: str = "data/index_local", top_k: int = 5) -> list[SearchHit]:
        """Retrieve top-k chunks from local FAISS index."""
        store = FaissStore.load(in_dir=in_dir, dim=self.settings.vector_dim)
        query_vec = self.embed_query(question)
        return store.search(query_vec, top_k=top_k)

    def answer(self, question: str, in_dir: str = "data/index_local", top_k: int = 5) -> dict[str, object]:
        """Generate answer from retrieved context using Ollama chat model."""
        hits = self.retrieve(question=question, in_dir=in_dir, top_k=top_k)
        context = self._build_context(hits)

        url = f"{self.settings.ollama_base_url}/api/chat"
        payload = {
            "model": self.settings.ollama_chat_model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful RAG assistant. Answer only from the provided context. "
                        "If context is insufficient, say you are not sure."
                    ),
                },
                {"role": "user", "content": f"Question:\n{question}\n\nContext:\n{context}"},
            ],
        }
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        answer_text = data.get("message", {}).get("content", "")

        # English: Keep sources hidden for now to match your current output style.
        # 中文: 先維持隱藏 sources，與你目前 OpenAI 版本一致。
        return {"answer": answer_text}

    @staticmethod
    def _build_context(hits: list[SearchHit]) -> str:
        """Format retrieval hits into model-ready context text."""
        if not hits:
            return "(no relevant chunks found)"
        blocks: list[str] = []
        for i, hit in enumerate(hits, start=1):
            blocks.append(
                f"[{i}] source={hit.chunk.source} page={hit.chunk.page} score={hit.score:.4f}\n{hit.chunk.text}"
            )
        return "\n\n".join(blocks)

