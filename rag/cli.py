from __future__ import annotations

import argparse
import json

from rag.pipeline import RagPipeline


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser.

    English: Supports `ingest` and `query` subcommands.
    中文: 支援 `ingest` 與 `query` 兩種子命令。
    """

    parser = argparse.ArgumentParser(description="DocsToData RAG CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Parse PDF and build FAISS index")
    ingest.add_argument("--pdf", required=True, help="Path to PDF file")
    ingest.add_argument("--out-dir", default="data/index", help="Output directory for FAISS files")
    ingest.add_argument("--chunk-size", type=int, default=700, help="Chunk size in tokens")
    ingest.add_argument("--chunk-overlap", type=int, default=120, help="Chunk overlap in tokens")

    query = sub.add_parser("query", help="Ask question from existing FAISS index")
    query.add_argument("--question", required=True, help="Question text")
    query.add_argument("--in-dir", default="data/index", help="Input directory for FAISS files")
    query.add_argument("--top-k", type=int, default=5, help="Top-K retrieved chunks")

    return parser


def main() -> None:
    """CLI entrypoint.

    English: Executes ingestion or QA flow and prints JSON output.
    中文: 執行匯入或問答流程，並輸出 JSON 結果。
    """

    parser = build_parser()
    args = parser.parse_args()
    rag = RagPipeline()

# Ingest here will chunk the pdf and save the chunks to the index directory
    if args.command == "ingest":
        stats = rag.ingest_pdf(
            pdf_path=args.pdf,
            out_dir=args.out_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        print(json.dumps({"status": "ok", "ingest": stats}, ensure_ascii=False, indent=2))
        return

# Query here will retrieve the chunks from the index directory and answer the question
    if args.command == "query":
        result = rag.answer(
            question=args.question,
            in_dir=args.in_dir,
            top_k=args.top_k,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
