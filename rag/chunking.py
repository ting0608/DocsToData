from __future__ import annotations

from typing import Iterable

import tiktoken

from rag.models import Chunk


def _split_tokens(tokens: list[int], chunk_size: int, chunk_overlap: int) -> Iterable[list[int]]:
    """Split token IDs into overlapping windows.

    English: Uses sliding windows for better context continuity.
    中文: 使用滑動視窗切分，確保上下文連續性更好。
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    start = 0
    step = chunk_size - chunk_overlap
    while start < len(tokens):
        end = start + chunk_size
        yield tokens[start:end]
        start += step


def chunk_pages(
    pages: list[tuple[int, str]],
    source_name: str,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
    encoding_name: str = "cl100k_base",
) -> list[Chunk]:
    """Convert page text into token-aware chunks.

    English: Chunk IDs include source, page number, and chunk index.
    中文: Chunk ID 會包含檔名來源、頁碼與分塊序號。
    """

    encoder = tiktoken.get_encoding(encoding_name)
    chunks: list[Chunk] = []

    for page_no, page_text in pages:
        token_ids = encoder.encode(page_text)
        for chunk_idx, token_slice in enumerate(
            _split_tokens(token_ids, chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            start=1,
        ):
            text = encoder.decode(token_slice).strip()
            if not text:
                continue
            chunk_id = f"{source_name}-p{page_no:04d}-c{chunk_idx:04d}"
            chunks.append(Chunk(chunk_id=chunk_id, source=source_name, page=page_no, text=text))
    return chunks
