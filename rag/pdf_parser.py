from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF


def extract_pdf_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Extract plain text per page from a PDF.

    English: Returns `(page_number, text)` and skips empty-text pages.
    中文: 回傳 `(頁碼, 文字)`，並略過沒有文字內容的頁面。
    """

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: list[tuple[int, str]] = []
    with fitz.open(path) as doc:
        for page_idx, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append((page_idx, text))
    return pages
