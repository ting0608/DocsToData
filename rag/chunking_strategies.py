from __future__ import annotations

import bisect
import re
from typing import Callable, Literal

import tiktoken

from rag.chunking import _split_tokens, chunk_pages
from rag.models import Chunk

ChunkingStrategy = Literal["page", "fixed_token", "semantic", "parent_child"]

DEFAULT_STRATEGY: ChunkingStrategy = "page"


def _encoder(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    return tiktoken.get_encoding(encoding_name)


def chunk_fixed_token(
    pages: list[tuple[int, str]],
    source_name: str,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
    encoding_name: str = "cl100k_base",
) -> list[Chunk]:
    """Fixed token-size sliding window across the WHOLE document.

    English: Unlike the page-scoped strategy, windows are allowed to span
    across page boundaries so no context is lost right at a page break. Each
    chunk is tagged with the page that contains the start of the window.
    中文: 與 page-scoped 策略不同，這裡的滑動視窗可以跨越分頁邊界，避免在換頁處
    截斷上下文。每個 chunk 會標記為視窗起始位置所在的頁碼。
    """

    encoder = _encoder(encoding_name)

    all_tokens: list[int] = []
    # cumulative token count at the END of each page, used to map a token
    # offset back to its page number via binary search.
    page_boundaries: list[int] = []
    page_numbers: list[int] = []
    for page_no, page_text in pages:
        token_ids = encoder.encode(page_text)
        all_tokens.extend(token_ids)
        page_boundaries.append(len(all_tokens))
        page_numbers.append(page_no)

    if not all_tokens:
        return []

    chunks: list[Chunk] = []
    for chunk_idx, token_slice in enumerate(
        _split_tokens(all_tokens, chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        start=1,
    ):
        text = encoder.decode(token_slice).strip()
        if not text:
            continue

        start_offset = (chunk_idx - 1) * (chunk_size - chunk_overlap)
        page_idx = bisect.bisect_right(page_boundaries, start_offset)
        page_idx = min(page_idx, len(page_numbers) - 1)
        page_no = page_numbers[page_idx]

        chunk_id = f"{source_name}-fx{chunk_idx:04d}-p{page_no:04d}"
        chunks.append(Chunk(chunk_id=chunk_id, source=source_name, page=page_no, text=text))
    return chunks


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")


def _split_paragraphs(text: str) -> list[str]:
    """Split page text into paragraph-like units.

    English: Splits on blank lines first; falls back to sentence boundaries
    for paragraphs that are themselves very long (dense PDF text often has no
    blank lines at all).
    中文: 先以空白行分段，若段落本身過長（PDF 常見整段沒有空行）則再以句子邊界
    細分。
    """

    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not raw_paragraphs:
        raw_paragraphs = [text.strip()] if text.strip() else []

    units: list[str] = []
    for para in raw_paragraphs:
        if len(para) <= 1200:
            units.append(para)
            continue
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(para) if s.strip()]
        units.extend(sentences or [para])
    return units


def chunk_semantic(
    pages: list[tuple[int, str]],
    source_name: str,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
    encoding_name: str = "cl100k_base",
) -> list[Chunk]:
    """Paragraph/sentence-boundary aware chunking, still bounded by token size.

    English: Rather than blindly slicing tokens, this greedily packs whole
    paragraphs (or sentences for long paragraphs) into a chunk until adding
    the next unit would exceed `chunk_size` tokens, then starts a new chunk
    carrying over the trailing ~`chunk_overlap` tokens of context. This keeps
    sentences/paragraphs intact instead of cutting mid-sentence, which is the
    main practical win of "semantic" chunking without requiring an extra
    embedding-similarity pass.
    中文: 不再單純依 token 數硬切，而是把完整段落（過長段落則以句子為單位）
    貪婪地塞進一個 chunk，直到加入下一段會超過 `chunk_size` 才換下一個 chunk，
    並保留約 `chunk_overlap` token 的重疊內容。這樣可以避免在句子中間截斷，
    是「語意分塊」在不引入額外 embedding 相似度運算下最主要的實務效益。
    """

    encoder = _encoder(encoding_name)
    chunks: list[Chunk] = []

    for page_no, page_text in pages:
        units = _split_paragraphs(page_text)
        if not units:
            continue

        chunk_idx = 0
        current_units: list[str] = []
        current_tokens = 0

        def flush() -> None:
            nonlocal chunk_idx, current_units, current_tokens
            if not current_units:
                return
            text = "\n\n".join(current_units).strip()
            if text:
                chunk_idx += 1
                chunk_id = f"{source_name}-sm{page_no:04d}-c{chunk_idx:04d}"
                chunks.append(Chunk(chunk_id=chunk_id, source=source_name, page=page_no, text=text))

        for unit in units:
            unit_tokens = len(encoder.encode(unit))
            if current_units and current_tokens + unit_tokens > chunk_size:
                flush()
                # carry over trailing units worth ~chunk_overlap tokens for continuity
                overlap_units: list[str] = []
                overlap_tokens = 0
                for prev_unit in reversed(current_units):
                    prev_tokens = len(encoder.encode(prev_unit))
                    if overlap_tokens + prev_tokens > chunk_overlap:
                        break
                    overlap_units.insert(0, prev_unit)
                    overlap_tokens += prev_tokens
                current_units = overlap_units
                current_tokens = overlap_tokens

            current_units.append(unit)
            current_tokens += unit_tokens

        flush()

    return chunks


def chunk_parent_child(
    pages: list[tuple[int, str]],
    source_name: str,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
    encoding_name: str = "cl100k_base",
    parent_chunk_multiplier: int = 3,
) -> list[Chunk]:
    """Small child chunks (used for retrieval) linked to a larger parent chunk.

    English: Each page is first split into "parent" windows of
    `chunk_size * parent_chunk_multiplier` tokens. Each parent window is then
    split again into small "child" windows of `chunk_size` tokens (the
    ordinary sliding-window logic). Children are what gets embedded and
    searched (precise retrieval), but each child carries `parent_id` +
    `parent_text` so the context builder can expand back to the larger parent
    window at generation time for more complete context. This is the classic
    "parent-document retrieval" pattern.
    中文: 每一頁先切成較大的「parent」視窗（大小為 `chunk_size * parent_chunk_multiplier`
    tokens），每個 parent 視窗再切成一般大小的「child」視窗（用於 embedding 與
    檢索，確保檢索精準）。每個 child 都帶有 `parent_id` 與 `parent_text`，讓
    context builder 在生成答案時可以還原回較大的 parent 視窗，取得更完整的上下文。
    這是常見的 parent-document retrieval 模式。
    """

    encoder = _encoder(encoding_name)
    parent_chunk_size = chunk_size * parent_chunk_multiplier
    chunks: list[Chunk] = []

    for page_no, page_text in pages:
        token_ids = encoder.encode(page_text)
        if not token_ids:
            continue

        for parent_idx, parent_tokens in enumerate(
            _split_tokens(token_ids, chunk_size=parent_chunk_size, chunk_overlap=0),
            start=1,
        ):
            parent_text = encoder.decode(parent_tokens).strip()
            if not parent_text:
                continue
            parent_id = f"{source_name}-p{page_no:04d}-parent{parent_idx:04d}"

            child_idx = 0
            for child_tokens in _split_tokens(parent_tokens, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
                child_text = encoder.decode(child_tokens).strip()
                if not child_text:
                    continue
                child_idx += 1
                chunk_id = f"{parent_id}-c{child_idx:04d}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        source=source_name,
                        page=page_no,
                        text=child_text,
                        parent_id=parent_id,
                        parent_text=parent_text,
                    )
                )
    return chunks


_STRATEGIES: dict[ChunkingStrategy, Callable[..., list[Chunk]]] = {
    "page": chunk_pages,
    "fixed_token": chunk_fixed_token,
    "semantic": chunk_semantic,
    "parent_child": chunk_parent_child,
}


def chunk_document(
    pages: list[tuple[int, str]],
    source_name: str,
    strategy: ChunkingStrategy = DEFAULT_STRATEGY,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
    encoding_name: str = "cl100k_base",
) -> list[Chunk]:
    """Dispatch to the configured chunking strategy.

    English: `strategy` is validated against the registry so unknown values
    fail fast with a clear error instead of silently falling back.
    中文: `strategy` 會對照策略清單驗證，未知的值會直接報錯，而不是悄悄使用
    預設策略。
    """

    if strategy not in _STRATEGIES:
        valid = ", ".join(sorted(_STRATEGIES))
        raise ValueError(f"Unknown chunking strategy '{strategy}'. Valid options: {valid}")

    fn = _STRATEGIES[strategy]
    return fn(
        pages=pages,
        source_name=source_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name=encoding_name,
    )
