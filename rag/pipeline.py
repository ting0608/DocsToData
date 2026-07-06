from __future__ import annotations

from pathlib import Path

from openai import OpenAI

from rag.chunking import chunk_pages
from rag.config import Settings, load_settings
from rag.graph.workflow import run_rag_graph
from rag.models import SearchHit
from rag.pdf_parser import extract_pdf_pages
from rag.vector_store import FaissStore


def _batched_texts(texts: list[str], batch_size: int = 128) -> list[list[str]]:
    """Split long text list into OpenAI-friendly mini-batches.

    English: Helps avoid overly large requests.
    中文: 避免一次送出過大的 API 請求。
    """

    return [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]


class RagPipeline:
    """High-level RAG workflow: ingest, retrieve, and answer.

    English: Main facade used by CLI/API layers.
    中文: 提供給 CLI/API 的主要流程入口。
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Create embeddings for multiple text chunks.

        English: Calls OpenAI embedding endpoint in batches.
        中文: 分批呼叫 OpenAI embedding API 產生向量。
        """

        vectors: list[list[float]] = []
        for batch in _batched_texts(texts, batch_size=128):
            response = self.client.embeddings.create(
                model=self.settings.openai_embed_model,
                input=batch,
            )
            vectors.extend([row.embedding for row in response.data])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Create one embedding vector for user query.

        English: Used as retrieval key against FAISS.
        中文: 作為在 FAISS 進行檢索的查詢向量。
        """

        response = self.client.embeddings.create(
            model=self.settings.openai_embed_model,
            input=text,
        )
        return response.data[0].embedding

    def _load_or_create_store(self, out_dir: str) -> FaissStore:
        index_path = Path(out_dir) / "index.faiss"
        if index_path.exists():
            return FaissStore.load(in_dir=out_dir, dim=self.settings.vector_dim)
        return FaissStore(dim=self.settings.vector_dim)

    def ingest_pdf(
        self,
        pdf_path: str,
        out_dir: str = "data/index",
        chunk_size: int = 700,
        chunk_overlap: int = 120,
        replace_existing: bool = True,
    ) -> dict[str, int]:
        """End-to-end ingestion: parse -> chunk -> embed -> append to index.

        English: Appends to an existing FAISS index; replaces same source when re-uploaded.
        中文: 會附加到既有索引；若同名來源已存在則先移除再寫入。
        """

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

    def list_sources(self, in_dir: str = "data/index") -> list[str]:
        store = FaissStore.load(in_dir=in_dir, dim=self.settings.vector_dim)
        return store.list_sources()

    def retrieve(
        self,
        question: str,
        in_dir: str = "data/index",
        top_k: int = 5,
        source_filter: set[str] | None = None,
    ) -> list[SearchHit]:
        """Retrieve top-k chunks relevant to a question."""

        store = FaissStore.load(in_dir=in_dir, dim=self.settings.vector_dim)
        query_vec = self.embed_query(question)
        return store.search(query_vec, top_k=top_k, source_filter=source_filter)

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.settings.openai_chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        return completion.choices[0].message.content or ""

    def answer(
        self,
        question: str,
        in_dir: str = "data/index",
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
        """Format retrieval hits into a prompt context block."""

        if not hits:
            return "(no relevant chunks found)"
        blocks: list[str] = []
        for i, hit in enumerate(hits, start=1):
            blocks.append(
                f"[{i}] source={hit.chunk.source} page={hit.chunk.page} score={hit.score:.4f}\n{hit.chunk.text}"
            )
        return "\n\n".join(blocks)
