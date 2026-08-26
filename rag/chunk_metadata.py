from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone

from rag.models import Chunk


def stamp_chunk_metadata(
    chunks: list[Chunk],
    source_name: str,
    embedding_model: str,
    chunk_size: int | None = None,
    chunking_strategy: str | None = None,
) -> None:
    """Populate v2 metadata fields on freshly created chunks, in place.

    English: `document_id` is shared by every chunk produced from the same
    ingest call, `created_at` is a UTC ISO-8601 timestamp, and `content_hash`
    is a per-chunk SHA-256 of its text (useful later for de-duplication or
    detecting whether a source PDF changed). See architecturev2.txt section 2.
    中文: 同一次匯入產生的所有 chunk 會共用同一個 `document_id`；`created_at`
    為 UTC ISO-8601 時間；`content_hash` 是每個 chunk 文字內容的 SHA-256，方便
    日後去重或偵測來源 PDF 是否變更。對應 architecturev2.txt 第 2 節。
    """

    document_id = f"{source_name}-{uuid.uuid4().hex[:12]}"
    created_at = datetime.now(timezone.utc).isoformat()

    for chunk in chunks:
        chunk.document_id = document_id
        chunk.created_at = created_at
        chunk.content_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
        chunk.embedding_model = embedding_model
        chunk.chunk_size = chunk_size
        chunk.chunking_strategy = chunking_strategy
