from __future__ import annotations

from typing import Protocol

from rag.models import SearchHit


class RagBackend(Protocol):
    """Minimal interface the LangGraph workflow needs from a RAG pipeline."""

    def retrieve(
        self,
        question: str,
        in_dir: str,
        top_k: int = 5,
        source_filter: set[str] | None = None,
        **kwargs: object,
    ) -> list[SearchHit]: ...

    def chat(self, system_prompt: str, user_prompt: str) -> str: ...

    def list_sources(self, in_dir: str) -> list[str]: ...
