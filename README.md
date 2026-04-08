# DocsToData

Python-first RAG starter using OpenAI + FAISS.

## Quick start

1. Create virtual environment and install deps:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Configure environment:
   - `cp .env.example .env`
   - edit `.env` and set `OPENAI_API_KEY`
3. Ingest a PDF:
   - `python -m rag.cli ingest --pdf "path/to/file.pdf"`
   - sample: 'python -m rag.cli ingest --pdf "/Users/tingcccc/Desktop/all those paperworks/Go Daikin Phase 4/Go Daikin Phase 4 Proposal (DMSS + DAMA) - 1.4.pdf"'
4. Ask a question:
   - `python -m rag.cli query --question "What is this document about?"`

## Project structure

- `rag/pdf_parser.py`: extracts page text from PDF via PyMuPDF
- `rag/chunking.py`: token-aware chunking with overlap
- `rag/pipeline.py`: OpenAI embedding + retrieval + answer flow
- `rag/vector_store.py`: FAISS index and metadata persistence
- `rag/cli.py`: CLI interface for ingest/query
