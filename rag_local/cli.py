from __future__ import annotations

import argparse
import json

from rag_local.pipeline import LocalRagPipeline


def build_parser() -> argparse.ArgumentParser:
    """Build CLI for local Ollama-based RAG.

    English: Supports `ingest` and `query` commands like cloud CLI.
    中文: 指令與雲端版一致，方便直接切換測試。
    """

    parser = argparse.ArgumentParser(description="DocsToData Local RAG CLI (Ollama)")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Parse PDF and build local FAISS index")
    ingest.add_argument("--pdf", required=True, help="Path to PDF file")
    ingest.add_argument("--out-dir", default="data/index_local", help="Output directory for local FAISS files")
    ingest.add_argument("--chunk-size", type=int, default=700, help="Chunk size in tokens")
    ingest.add_argument("--chunk-overlap", type=int, default=120, help="Chunk overlap in tokens")

    query = sub.add_parser("query", help="Ask question from local FAISS index")
    query.add_argument("--question", required=True, help="Question text")
    query.add_argument("--in-dir", default="data/index_local", help="Input directory for local FAISS files")
    query.add_argument("--top-k", type=int, default=5, help="Top-K retrieved chunks")

    return parser


def main() -> None:
    """CLI entrypoint for local Ollama flow."""
    parser = build_parser()
    args = parser.parse_args()
    rag = LocalRagPipeline()

    if args.command == "ingest":
        stats = rag.ingest_pdf(
            pdf_path=args.pdf,
            out_dir=args.out_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        print(json.dumps({"status": "ok", "ingest": stats}, ensure_ascii=False, indent=2))
        return

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

