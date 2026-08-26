from __future__ import annotations

from typing import Literal, TypedDict

Intent = Literal["summarize", "qa", "compare"]


class RagState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""

    question: str
    in_dir: str
    top_k: int
    intent: Intent | None
    source_filter: list[str] | None
    answer: str
    sources: list[str]
    citations: list[dict[str, object]]
    confidence: float
