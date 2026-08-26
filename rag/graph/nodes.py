from __future__ import annotations

from rag.graph.backend import RagBackend
from rag.graph.state import RagState
from rag.models import SearchHit

_QA_SYSTEM = (
    "You are a helpful RAG assistant. Answer only from the provided context. "
    "If context is insufficient, say you are not sure."
)

_SUMMARIZE_SYSTEM = (
    "You are a document summarization assistant. Produce a clear, structured summary "
    "using only the provided context. Include key themes, decisions, numbers, and scope. "
    "If information is missing, say so explicitly."
)

_COMPARE_SYSTEM = (
    "You are a document comparison assistant. Compare the provided sources side by side. "
    "Highlight similarities, differences, trade-offs, and contradictions. "
    "Use only the provided context."
)


def _build_context(hits: list[SearchHit]) -> str:
    if not hits:
        return "(no relevant chunks found)"
    blocks: list[str] = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(
            f"[{i}] source={hit.chunk.source} page={hit.chunk.page} score={hit.score:.4f}\n{hit.chunk.text}"
        )
    return "\n\n".join(blocks)


def _hits_to_sources(hits: list[SearchHit]) -> list[str]:
    seen: set[str] = set()
    sources: list[str] = []
    for hit in hits:
        if hit.chunk.source not in seen:
            seen.add(hit.chunk.source)
            sources.append(hit.chunk.source)
    return sources


def _hits_to_citations(hits: list[SearchHit]) -> list[dict[str, object]]:
    """Build the v2 structured citation list (architecturev2.txt section 6).

    English: One entry per retrieved chunk (not deduplicated by document,
    unlike `_hits_to_sources`) so the frontend can show exactly which page
    and chunk backed each part of the answer.
    中文: 每個檢索到的 chunk 各佔一筆（不像 `_hits_to_sources` 會依文件去重），
    讓前端可以顯示答案究竟是根據哪一頁、哪一個 chunk。
    """

    return [
        {
            "document": hit.chunk.source,
            "page": hit.chunk.page,
            "chunk_id": hit.chunk.chunk_id,
            "score": round(hit.score, 4),
        }
        for hit in hits
    ]


def _compute_confidence(hits: list[SearchHit]) -> float:
    """Heuristic confidence score in [0, 1] from retrieval hit scores.

    English: Averages the top hits' scores (already similarity-like for
    plain vector search, or a fused/lexical rerank score for hybrid
    retrieval) and clamps to [0, 1]. This is a proxy for "how well did the
    retrieved context match the query", not a calibrated probability that
    the generated answer is correct — architecturev2.txt section 6 asks for
    a confidence field, and section 7 flags proper faithfulness/correctness
    evaluation as future work.
    中文: 將排名最前面的檢索結果分數取平均並限制在 [0, 1] 範圍內（單純向量
    搜尋時本身就類似相似度；混合檢索時則是融合/詞彙重排序分數）。這只是「檢索
    到的內容與查詢的相符程度」的粗略代理值，並非經過校準、代表生成答案正確性
    的機率——architecturev2.txt 第 6 節要求提供 confidence 欄位，第 7 節則將
    正式的 faithfulness/correctness 評估列為後續工作。
    """

    if not hits:
        return 0.0
    top = hits[:5]
    avg_score = sum(hit.score for hit in top) / len(top)
    return round(max(0.0, min(1.0, avg_score)), 4)


def _selected_sources(backend: RagBackend, state: RagState) -> list[str]:
    """Return document sources to use for this query, honoring optional filter."""

    all_sources = backend.list_sources(state["in_dir"])
    filter_list = state.get("source_filter")
    if not filter_list:
        return all_sources
    allowed = set(filter_list)
    return [source for source in all_sources if source in allowed]


def _retrieve_per_source(
    backend: RagBackend,
    question: str,
    in_dir: str,
    top_k: int,
    sources: list[str],
) -> list[SearchHit]:
    """Retrieve chunks from each selected document for compare/summarize flows."""

    if not sources:
        return []

    per_source_k = max(2, top_k // max(len(sources), 1))
    hits: list[SearchHit] = []
    for source in sources:
        hits.extend(
            backend.retrieve(
                question=question,
                in_dir=in_dir,
                top_k=per_source_k,
                source_filter={source},
            )
        )
    return hits


def answer_qa(backend: RagBackend, state: RagState) -> RagState:
    sources = _selected_sources(backend, state)
    if not sources:
        return {
            **state,
            "answer": "No PDFs selected. Open PDFs and check at least one document to query.",
            "sources": [],
            "citations": [],
            "confidence": 0.0,
        }

    hits = backend.retrieve(
        question=state["question"],
        in_dir=state["in_dir"],
        top_k=state["top_k"],
        source_filter=set(sources),
    )
    context = _build_context(hits)
    answer = backend.chat(
        system_prompt=_QA_SYSTEM,
        user_prompt=f"Question:\n{state['question']}\n\nContext:\n{context}",
    )
    return {
        **state,
        "answer": answer,
        "sources": _hits_to_sources(hits),
        "citations": _hits_to_citations(hits),
        "confidence": _compute_confidence(hits),
    }


def answer_summarize(backend: RagBackend, state: RagState) -> RagState:
    sources = _selected_sources(backend, state)
    if not sources:
        return {
            **state,
            "answer": "No PDFs selected. Open PDFs and check at least one document to summarize.",
            "sources": [],
            "citations": [],
            "confidence": 0.0,
        }

    question = state["question"] or "Summarize the uploaded documents."
    hits = _retrieve_per_source(
        backend,
        question=question,
        in_dir=state["in_dir"],
        top_k=state["top_k"] * 2,
        sources=sources,
    )
    context = _build_context(hits)
    answer = backend.chat(
        system_prompt=_SUMMARIZE_SYSTEM,
        user_prompt=f"Summarize request:\n{question}\n\nContext:\n{context}",
    )
    return {
        **state,
        "answer": answer,
        "sources": _hits_to_sources(hits),
        "citations": _hits_to_citations(hits),
        "confidence": _compute_confidence(hits),
    }


def answer_compare(backend: RagBackend, state: RagState) -> RagState:
    sources = _selected_sources(backend, state)
    if not sources:
        return {
            **state,
            "answer": "No PDFs selected. Open PDFs and check at least two documents to compare.",
            "sources": [],
            "citations": [],
            "confidence": 0.0,
        }
    if len(sources) < 2:
        return {
            **state,
            "answer": (
                "Compare mode needs at least two selected PDFs. "
                f"Currently selected: {', '.join(sources)}."
            ),
            "sources": sources,
            "citations": [],
            "confidence": 0.0,
        }

    question = state["question"] or "Compare the uploaded documents."
    hits = _retrieve_per_source(
        backend,
        question=question,
        in_dir=state["in_dir"],
        top_k=state["top_k"] * 2,
        sources=sources,
    )
    context = _build_context(hits)
    answer = backend.chat(
        system_prompt=_COMPARE_SYSTEM,
        user_prompt=(
            f"Comparison request:\n{question}\n\n"
            f"Documents to compare: {', '.join(sources)}\n\nContext:\n{context}"
        ),
    )
    return {
        **state,
        "answer": answer,
        "sources": _hits_to_sources(hits),
        "citations": _hits_to_citations(hits),
        "confidence": _compute_confidence(hits),
    }
