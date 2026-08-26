from __future__ import annotations

import os
from pathlib import Path

from rag.chunk_metadata import stamp_chunk_metadata
from rag.chunking_strategies import DEFAULT_STRATEGY, ChunkingStrategy, chunk_document
from rag.graph.workflow import run_rag_graph
from rag.llm_gateway import LLMGateway
from rag.models import SearchHit, TokenUsage
from rag.pdf_parser import extract_pdf_pages
from rag.retrieval import hybrid_retrieve
from rag.vector_store import FaissStore


def _default_chunking_strategy() -> ChunkingStrategy:
    value = os.getenv("CHUNK_STRATEGY", DEFAULT_STRATEGY).strip().lower()
    return value  # type: ignore[return-value]  # validated inside chunk_document()


class BaseRagPipeline:
    """Provider-agnostic RAG pipeline built on top of an `LLMGateway`.

    English: This is the shared implementation used by `RagPipeline`
    (OpenAI), `LocalRagPipeline` (Ollama), and `BedrockRagPipeline` (AWS) —
    architecturev2.txt section 5's "LLM Gateway" pattern applied to the
    ingestion/retrieval/generation pipeline so all three providers share one
    code path instead of three near-duplicate ones. Subclasses only need to
    construct the right `LLMGateway` and pass provider-specific defaults
    (vector dimension, embedding model name, default index directory).
    中文: 這是 `RagPipeline`（OpenAI）、`LocalRagPipeline`（Ollama）、
    `BedrockRagPipeline`（AWS）共用的實作——把 architecturev2.txt 第 5 節的
    「LLM Gateway」模式套用到 ingestion/檢索/生成流程，讓三個 provider 共用
    同一份程式碼，而不是各自維護三份幾乎相同的版本。子類別只需要建立對應的
    `LLMGateway`，並傳入各自的預設值（向量維度、embedding 模型名稱、預設索引
    目錄）即可。
    """

    def __init__(
        self,
        gateway: LLMGateway,
        vector_dim: int,
        embedding_model_name: str,
        default_index_dir: str,
        chat_model_name: str = "",
        chunking_strategy: ChunkingStrategy | None = None,
    ) -> None:
        self.gateway = gateway
        self.vector_dim = vector_dim
        self.embedding_model_name = embedding_model_name
        self.chat_model_name = chat_model_name
        self.default_index_dir = default_index_dir
        self.chunking_strategy = chunking_strategy or _default_chunking_strategy()
        self._usage_accumulator = TokenUsage()

    def _load_or_create_store(self, out_dir: str) -> FaissStore:
        index_path = Path(out_dir) / "index.faiss"
        if index_path.exists():
            return FaissStore.load(in_dir=out_dir, dim=self.vector_dim)
        return FaissStore(dim=self.vector_dim)

    def ingest_pdf(
        self,
        pdf_path: str,
        out_dir: str | None = None,
        chunk_size: int = 700,
        chunk_overlap: int = 120,
        replace_existing: bool = True,
        chunking_strategy: ChunkingStrategy | None = None,
    ) -> dict[str, object]:
        """End-to-end ingestion: parse -> chunk -> embed -> append to index.

        English: `chunking_strategy` defaults to the pipeline's configured
        strategy (env `CHUNK_STRATEGY`, default "page" — the pre-v2
        behavior) but can be overridden per call, e.g. from an API request,
        to support the experiment framework in architecturev2.txt section 8.
        中文: `chunking_strategy` 預設使用 pipeline 設定的策略（環境變數
        `CHUNK_STRATEGY`，預設為 "page"，即 v2 之前的行為），但可依呼叫（例如
        API 請求）逐次覆寫，以支援 architecturev2.txt 第 8 節的實驗框架。
        """

        out_dir = out_dir or self.default_index_dir
        strategy = chunking_strategy or self.chunking_strategy

        pages = extract_pdf_pages(pdf_path)
        source_name = Path(pdf_path).stem
        chunks = chunk_document(
            pages=pages,
            source_name=source_name,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        stamp_chunk_metadata(
            chunks,
            source_name=source_name,
            embedding_model=self.embedding_model_name,
            chunk_size=chunk_size,
            chunking_strategy=strategy,
        )

        embeddings = self.gateway.embed([c.text for c in chunks])

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
            "chunking_strategy": strategy,
        }

    def list_sources(self, in_dir: str | None = None) -> list[str]:
        store = FaissStore.load(in_dir=in_dir or self.default_index_dir, dim=self.vector_dim)
        return store.list_sources()

    def retrieve(
        self,
        question: str,
        in_dir: str | None = None,
        top_k: int = 5,
        source_filter: set[str] | None = None,
        use_hybrid: bool = True,
        candidate_k: int = 20,
    ) -> list[SearchHit]:
        """Retrieve top-k chunks relevant to a question.

        English: `use_hybrid=True` (default) runs the full vector + keyword
        + rerank pipeline from architecturev2.txt section 4
        (`rag/retrieval.py`); set it to False to fall back to plain vector
        search (useful for A/B comparisons in the evaluation framework).
        中文: `use_hybrid=True`（預設）會執行完整的向量 + 關鍵字 + 重排序流程
        （對應 architecturev2.txt 第 4 節，實作於 `rag/retrieval.py`）；設為
        False 則會退回單純向量搜尋（方便評估框架做 A/B 比較）。
        """

        store = FaissStore.load(in_dir=in_dir or self.default_index_dir, dim=self.vector_dim)
        query_vec = self.gateway.embed_one(question)

        if not use_hybrid:
            return store.search(query_vec, top_k=top_k, source_filter=source_filter)

        return hybrid_retrieve(
            store=store,
            query_text=question,
            query_vector=query_vec,
            top_k=top_k,
            candidate_k=max(candidate_k, top_k),
            source_filter=source_filter,
        )

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        result = self.gateway.generate(user_prompt, system=system_prompt, temperature=0.1)
        self._usage_accumulator.prompt_tokens += self.gateway.last_usage.prompt_tokens
        self._usage_accumulator.completion_tokens += self.gateway.last_usage.completion_tokens
        return result

    def answer(
        self,
        question: str,
        in_dir: str | None = None,
        top_k: int = 5,
        intent: str | None = None,
        source_filter: list[str] | None = None,
    ) -> dict[str, object]:
        """Run LangGraph workflow with intent routing.

        English: Returns the v2 structured response shape from
        architecturev2.txt section 6: `{answer, intent, sources, citations,
        confidence}`, plus a `usage: TokenUsage` field accumulated across
        every `chat()` call made during this run (intent classification +
        generation) so the evaluation framework can compute a per-query cost
        estimate via `rag/pricing.py`.
        中文: 回傳對應 architecturev2.txt 第 6 節的 v2 結構化回應格式：
        `{answer, intent, sources, citations, confidence}`，並額外附上
        `usage: TokenUsage`，累加本次執行過程中所有 `chat()` 呼叫的 token 數
        （意圖分類 + 生成回答），讓評估框架可透過 `rag/pricing.py` 估算每次
        查詢的成本。
        """

        self._usage_accumulator = TokenUsage()
        result = run_rag_graph(
            backend=self,
            question=question,
            in_dir=in_dir or self.default_index_dir,
            top_k=top_k,
            intent=intent,  # type: ignore[arg-type]
            source_filter=source_filter,
        )
        # Copy, not the same object: callers may make further `chat()` calls
        # after `answer()` returns (e.g. an LLM-judge pass in the evaluation
        # framework) which must not silently mutate an already-returned result.
        result["usage"] = TokenUsage(
            prompt_tokens=self._usage_accumulator.prompt_tokens,
            completion_tokens=self._usage_accumulator.completion_tokens,
        )
        return result
