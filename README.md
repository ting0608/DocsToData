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

## FastAPI backend (localhost)

1. Start server:
   - `uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000`
2. Health check:
   - `GET http://localhost:8000/health`
3. Swagger docs:
   - `http://localhost:8000/docs`

### Example API requests (Postman)

- `POST http://localhost:8000/ingest`
```json
{
  "provider": "ollama",
  "pdf_path": "/Users/tingcccc/Desktop/all those paperworks/Go Daikin Phase 4/Go Daikin Phase 4 Proposal (DMSS + DAMA) - 1.4.pdf"
}
```

- `POST http://localhost:8000/query`
```json
{
  "provider": "ollama",
  "question": "What is this document about?"
}
```

Use `"provider": "openai"` to run the OpenAI pipeline instead.

## Simple web UI

- Start backend:
  - `uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000`
- Open:
  - `http://localhost:8000/`

UI behavior:
- `+` button uploads PDF and calls `/ingest-upload`
- Send button (or Enter) calls `/query`
- Provider dropdown lets you switch between `ollama` and `openai`

## Deploy to Google Cloud Run (OpenAI + API key)

Ollama does not run inside this container by default. For cloud, use **OpenAI** (`provider: "openai"`) and store `OPENAI_API_KEY` in **Secret Manager**, then wire it as an environment variable on the service.

1. Build and push an image (replace `PROJECT_ID` and region):
   - `gcloud auth configure-docker`
   - `docker build -t gcr.io/PROJECT_ID/docstodata:latest .`
   - `docker push gcr.io/PROJECT_ID/docstodata:latest`
2. Create a secret for the API key (one-time):
   - `echo -n 'sk-...' | gcloud secrets create openai-api-key --data-file=-`
3. Deploy to Cloud Run:
   - `gcloud run deploy docstodata --image gcr.io/PROJECT_ID/docstodata:latest --region us-central1 --allow-unauthenticated --set-env-vars DISABLE_OLLAMA=true --set-secrets OPENAI_API_KEY=openai-api-key:latest`
4. Open the service URL; use the UI with **OpenAI** selected, or call `/ingest` and `/query` with `"provider": "openai"`.

**Ephemeral disk:** FAISS indexes and uploads live on the container filesystem. They are lost when the instance is replaced or scaled to zero. For production persistence, plan **Cloud Storage** (or a database) for `index.faiss` / `chunks.json` and uploads.

### Optional persistent storage with GCS

Set these env vars on Cloud Run:
- `GCS_BUCKET=<your_bucket_name>`
- `GCS_PREFIX=docstodata` (optional)

Behavior:
- `/ingest` and `/ingest-upload` upload `index.faiss` and `chunks.json` to `gs://<bucket>/<prefix>/indexes/<provider>/...`
- uploaded PDFs are mirrored to `gs://<bucket>/<prefix>/uploads/...`
- `/query` auto-downloads index files from GCS if local files are missing
- check `/storage-status` to verify GCS config at runtime

**Local Docker test:**

```bash
docker build -t docstodata:local .
docker run --rm -p 8080:8080 -e DISABLE_OLLAMA=true -e OPENAI_API_KEY=sk-... docstodata:local
```

Then open `http://localhost:8080/`.

## Project structure

- `rag/pdf_parser.py`: extracts page text from PDF via PyMuPDF
- `rag/chunking.py`: token-aware chunking with overlap
- `rag/pipeline.py`: OpenAI embedding + retrieval + answer flow
- `rag/vector_store.py`: FAISS index and metadata persistence
- `rag/cli.py`: CLI interface for ingest/query
- `rag_local/pipeline.py`: Ollama embedding + retrieval + answer flow
- `rag_local/cli.py`: local CLI interface for ingest/query
