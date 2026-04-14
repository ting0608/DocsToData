# DocsToData

Python-first RAG starter using OpenAI + FAISS, plus local Ollama + FAISS.

## Quick start

1. Create virtual environment and install deps:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Configure environment:
   - `cp .env.example .env`
   - edit `.env` and set `OPENAI_API_KEY` (for cloud flow)
3. Ingest a PDF:
   - `python -m rag.cli ingest --pdf "path/to/file.pdf"`
   - sample: 'python -m rag.cli ingest --pdf "/Users/tingcccc/Desktop/all those paperworks/Go Daikin Phase 4/Go Daikin Phase 4 Proposal (DMSS + DAMA) - 1.4.pdf"'
4. Ask a question:
   - `python -m rag.cli query --question "What is this document about?"`

## Local Ollama flow

1. Make sure Ollama is running and models exist:
   - `ollama list`
2. (Optional) tune local model config in `.env`:
   - `OLLAMA_CHAT_MODEL=llama3.1:8b`
   - `OLLAMA_EMBED_MODEL=nomic-embed-text:latest`
   - `OLLAMA_VECTOR_DIM=768`
3. Ingest with local embedding model:
   - `python -m rag_local.cli ingest --pdf "path/to/file.pdf"`
   - sample: 'python -m rag_local.cli ingest --pdf "/Users/tingcccc/Desktop/all those paperworks/Go Daikin Phase 4/Go Daikin Phase 4 Proposal (DMSS + DAMA) - 1.4.pdf"'
4. Query with local chat model:
   - `python -m rag_local.cli query --question "What is this document about?"`

## Project structure

- `rag/pdf_parser.py`: extracts page text from PDF via PyMuPDF
- `rag/chunking.py`: token-aware chunking with overlap
- `rag/pipeline.py`: OpenAI embedding + retrieval + answer flow
- `rag/vector_store.py`: FAISS index and metadata persistence
- `rag/cli.py`: CLI interface for ingest/query
- `rag_local/pipeline.py`: Ollama embedding + retrieval + answer flow
- `rag_local/cli.py`: local CLI interface for ingest/query
