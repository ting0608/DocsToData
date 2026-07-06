from __future__ import annotations

from pathlib import Path

import requests

from rag.chunking import chunk_pages
from rag.graph.workflow import run_rag_graph
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

    def _load_or_create_store(self, out_dir: str) -> FaissStore:
        index_path = Path(out_dir) / "index.faiss"
        if index_path.exists():
            return FaissStore.load(in_dir=out_dir, dim=self.settings.vector_dim)
        return FaissStore(dim=self.settings.vector_dim)

    def ingest_pdf(
        self,
        pdf_path: str,
        out_dir: str = "data/index_local",
        chunk_size: int = 700,
        chunk_overlap: int = 120,
        replace_existing: bool = True,
    ) -> dict[str, int]:
        """Parse PDF and append chunks to the local FAISS index."""

        pages = extract_pdf_pages(pdf_path)
        source_name = Path(pdf_path).stem
        chunks = chunk_pages(
            pages=pages,
            source_name=source_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        embeddings = self.embed_texts([c.text for c in chunks])

        store = self._load_or_create_store(out_dir)
        removed = store.remove_source(source_name) if replace_existing else 0
        store.add(embeddings=embeddings, chunks=chunks)
        store.save(out_dir=out_dir)

        return {
            "source": source_name,
            "pages": len(pages),
            "chunks": len(chunks),
            "vectors": len(embeddings),
            "replaced_chunks": removed,
            "total_documents": len(store.list_sources()),
        }

    def list_sources(self, in_dir: str = "data/index_local") -> list[str]:
        store = FaissStore.load(in_dir=in_dir, dim=self.settings.vector_dim)
        return store.list_sources()

    def retrieve(
        self,
        question: str,
        in_dir: str = "data/index_local",
        top_k: int = 5,
        source_filter: set[str] | None = None,
    ) -> list[SearchHit]:
        """Retrieve top-k chunks from local FAISS index."""

        store = FaissStore.load(in_dir=in_dir, dim=self.settings.vector_dim)
        query_vec = self.embed_query(question)
        return store.search(query_vec, top_k=top_k, source_filter=source_filter)

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.settings.ollama_base_url}/api/chat"
        payload = {
            "model": self.settings.ollama_chat_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")

    def answer(
        self,
        question: str,
        in_dir: str = "data/index_local",
        top_k: int = 5,
        intent: str | None = None,
        source_filter: list[str] | None = None,
    ) -> dict[str, object]:
        """Run LangGraph workflow with intent routing."""

        return run_rag_graph(
            backend=self,
            question=question,
            in_dir=in_dir,
            top_k=top_k,
            intent=intent,  # type: ignore[arg-type]
            source_filter=source_filter,
        )

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
