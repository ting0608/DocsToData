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
    return {**state, "answer": answer, "sources": _hits_to_sources(hits)}


def answer_summarize(backend: RagBackend, state: RagState) -> RagState:
    sources = _selected_sources(backend, state)
    if not sources:
        return {
            **state,
            "answer": "No PDFs selected. Open PDFs and check at least one document to summarize.",
            "sources": [],
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
    return {**state, "answer": answer, "sources": _hits_to_sources(hits)}


def answer_compare(backend: RagBackend, state: RagState) -> RagState:
    sources = _selected_sources(backend, state)
    if not sources:
        return {
            **state,
            "answer": "No PDFs selected. Open PDFs and check at least two documents to compare.",
            "sources": [],
        }
    if len(sources) < 2:
        return {
            **state,
            "answer": (
                "Compare mode needs at least two selected PDFs. "
                f"Currently selected: {', '.join(sources)}."
            ),
            "sources": sources,
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
    return {**state, "answer": answer, "sources": _hits_to_sources(hits)}
