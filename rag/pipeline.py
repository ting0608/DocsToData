from __future__ import annotations

from pathlib import Path

from openai import OpenAI

from rag.chunking import chunk_pages
from rag.config import Settings, load_settings
from rag.models import Chunk, SearchHit
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

# During this time, we call openai embedding model to embed the question into a vector so that we can search from index
        response = self.client.embeddings.create(
            model=self.settings.openai_embed_model,
            input=text,
        )
        return response.data[0].embedding

    def ingest_pdf(
        self,
        pdf_path: str,
        out_dir: str = "data/index",
        chunk_size: int = 700,
        chunk_overlap: int = 120,
    ) -> dict[str, int]:
        """End-to-end ingestion: parse -> chunk -> embed -> index save.

        English: Builds a fresh FAISS index in `out_dir`.
        中文: 在 `out_dir` 建立新的 FAISS 索引資料。
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

        store = FaissStore(dim=self.settings.vector_dim)
        store.add(embeddings=embeddings, chunks=chunks)
        store.save(out_dir=out_dir)

        return {
            "pages": len(pages),
            "chunks": len(chunks),
            "vectors": len(embeddings),
        }

    def retrieve(self, question: str, in_dir: str = "data/index", top_k: int = 5) -> list[SearchHit]:
        """Retrieve top-k chunks relevant to a question.

        English: Loads persisted FAISS index from disk each call.
        中文: 每次呼叫都會從磁碟載入已保存的 FAISS 索引。
        """

        store = FaissStore.load(in_dir=in_dir, dim=self.settings.vector_dim)

        # This is the key to embed the question into a vector so that we can search the index, refer to "embed_query" function
        query_vec = self.embed_query(question)
        return store.search(query_vec, top_k=top_k)

    def answer(self, question: str, in_dir: str = "data/index", top_k: int = 5) -> dict[str, object]:
        """Generate final answer with retrieved context and citations.

        English: Returns `answer` plus source metadata for traceability.
        中文: 回傳 `answer` 及來源資訊，方便追溯引用。
        """

        hits = self.retrieve(question=question, in_dir=in_dir, top_k=top_k)
        context = self._build_context(hits)

        completion = self.client.chat.completions.create(
            model=self.settings.openai_chat_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful RAG assistant. Answer only from the provided context. "
                        "If context is insufficient, say you are not sure."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question:\n{question}\n\nContext:\n{context}",
                },
            ],
            temperature=0.1,
        )
        answer_text = completion.choices[0].message.content or ""

        # English: Temporarily hide source payload from CLI/API response.
        # 中文: 先暫時隱藏回傳中的來源資訊（sources）。
        #
        # To enable it again, uncomment the block below and return it.
        # 若要恢復來源資訊，取消註解以下區塊並回傳 sources。
        #
        # sources = [
        #     {
        #         "chunk_id": hit.chunk.chunk_id,
        #         "source": hit.chunk.source,
        #         "page": hit.chunk.page,
        #         "score": hit.score,
        #     }
        #     for hit in hits
        # ]
        # return {"answer": answer_text, "sources": sources}
        return {"answer": answer_text}

    @staticmethod
    def _build_context(hits: list[SearchHit]) -> str:
        """Format retrieval hits into a prompt context block.

        English: Includes source/page/score for grounded answering.
        中文: 會附上來源/頁碼/分數，幫助回答更可追蹤。
        """

        if not hits:
            return "(no relevant chunks found)"
        blocks: list[str] = []
        for i, hit in enumerate(hits, start=1):
            blocks.append(
                f"[{i}] source={hit.chunk.source} page={hit.chunk.page} score={hit.score:.4f}\n{hit.chunk.text}"
            )
        return "\n\n".join(blocks)
