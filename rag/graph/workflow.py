from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from rag.graph.backend import RagBackend
from rag.graph.intent import classify_intent
from rag.graph.nodes import answer_compare, answer_qa, answer_summarize
from rag.graph.state import Intent, RagState


def _route_intent(state: RagState) -> Intent:
    return state.get("intent") or "qa"


def build_rag_graph(backend: RagBackend):
    """Build LangGraph workflow: classify intent -> summarize | qa | compare."""

    graph = StateGraph(RagState)

    graph.add_node("classify", lambda state: classify_intent(backend, state))
    graph.add_node("qa", lambda state: answer_qa(backend, state))
    graph.add_node("summarize", lambda state: answer_summarize(backend, state))
    graph.add_node("compare", lambda state: answer_compare(backend, state))

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        _route_intent,
        {
            "summarize": "summarize",
            "qa": "qa",
            "compare": "compare",
        },
    )
    graph.add_edge("qa", END)
    graph.add_edge("summarize", END)
    graph.add_edge("compare", END)

    return graph.compile()


def run_rag_graph(
    backend: RagBackend,
    question: str,
    in_dir: str,
    top_k: int = 5,
    intent: Intent | None = None,
    source_filter: list[str] | None = None,
) -> dict[str, object]:
    """Run the full LangGraph RAG workflow and return answer metadata."""

    app = build_rag_graph(backend)
    initial: RagState = {
        "question": question,
        "in_dir": in_dir,
        "top_k": top_k,
    }
    if intent:
        initial["intent"] = intent
    if source_filter:
        initial["source_filter"] = source_filter

    final = app.invoke(initial)
    return {
        "answer": final.get("answer", ""),
        "intent": final.get("intent", "qa"),
        "sources": final.get("sources", []),
    }
