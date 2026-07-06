from __future__ import annotations

import json
import re

from rag.graph.backend import RagBackend
from rag.graph.nodes import _selected_sources
from rag.graph.state import Intent, RagState

_CLASSIFY_SYSTEM = """You classify user requests about uploaded PDF documents into exactly one intent:
- summarize: user wants a summary, overview, key points, or report of document(s)
- qa: user asks a specific factual question to be answered from the documents
- compare: user wants to compare, contrast, or find differences/similarities across documents

Reply with JSON only: {"intent": "summarize" | "qa" | "compare"}"""


def _parse_intent(raw: str) -> Intent:
    """Extract intent label from model output."""

    text = raw.strip()
    try:
        payload = json.loads(text)
        value = str(payload.get("intent", "")).lower()
        if value in {"summarize", "qa", "compare"}:
            return value  # type: ignore[return-value]
    except json.JSONDecodeError:
        pass

    match = re.search(r"\b(summarize|qa|compare)\b", text.lower())
    if match:
        return match.group(1)  # type: ignore[return-value]

    lowered = text.lower()
    if any(word in lowered for word in ("compare", "contrast", "difference", "versus", "vs")):
        return "compare"
    if any(word in lowered for word in ("summarize", "summary", "overview", "key points")):
        return "summarize"
    return "qa"


def classify_intent(backend: RagBackend, state: RagState) -> RagState:
    """Classify user question into summarize, qa, or compare."""

    if state.get("intent"):
        return state

    question = state["question"]
    sources = _selected_sources(backend, state)
    source_hint = ", ".join(sources) if sources else "none"

    user_prompt = (
        f"Indexed documents: {source_hint}\n"
        f"User message: {question}\n"
        "Return JSON with one intent."
    )
    raw = backend.chat(system_prompt=_CLASSIFY_SYSTEM, user_prompt=user_prompt)
    return {**state, "intent": _parse_intent(raw)}
