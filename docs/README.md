# Go Invoice

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

## Web UI

- Start backend:
  - `uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000`
- Open:
  - `http://localhost:8000/`

The UI has a top navigation bar with four sections:

- **Upload & RAG** — the original chat flow. `+` uploads PDFs (`/ingest-upload`), Send/Enter asks a question (`/query`), Summarize builds an exportable report. Provider dropdown (top right) switches `ollama` / `openai` for the whole app.
- **Evaluation Dashboard** — save reusable test questions, run them against the live pipeline, grade each answer pass/fail, and see aggregate pass rate / latency. Backed by `/evaluate/*` endpoints, persisted to `data/eval_cases.json` and `data/eval_history.json`.
- **Document Library** — browse indexed PDFs per provider with page/chunk counts and delete a document from the index. Backed by `/documents/details` and `DELETE /documents/{provider}/{source}`.
- **Account** — sign in/out via AWS Cognito (see below). Runs in an open "dev mode" until Cognito is configured.

### AWS Cognito authentication setup

Authentication is optional and off by default (dev mode: every request is treated as a local dev user). To require sign-in:

1. In the AWS Console, create a **Cognito User Pool** (or reuse an existing one).
   - Enable the **Hosted UI** and add an **App client** (public client, no client secret — this app uses PKCE).
   - Under the app client's Hosted UI settings, set the **Allowed callback URLs** and **Allowed sign-out URLs** to your app's URL, e.g. `http://localhost:8000/` for local dev and your real domain in production.
   - Enable the **Authorization code grant** flow with scopes `openid email profile`.
   - (Optional) Add a SAML/OIDC identity provider under "Sign-in experience" for enterprise SSO, then set `COGNITO_SSO_PROVIDER` to its name to enable the "Continue with SSO" button.
   - Pick a **Cognito domain** (Hosted UI domain) under "App integration".
2. Set these in `.env`:
   - `COGNITO_REGION=us-east-1` (your pool's region)
   - `COGNITO_USER_POOL_ID=us-east-1_xxxxxxxxx`
   - `COGNITO_CLIENT_ID=<app client id>`
   - `COGNITO_DOMAIN=https://your-app.auth.us-east-1.amazoncognito.com`
   - `COGNITO_SSO_PROVIDER=` (optional, e.g. `MyCompanySSO`)
3. Restart the backend. Once all three of `COGNITO_REGION` / `COGNITO_USER_POOL_ID` / `COGNITO_CLIENT_ID` are set, every API route (`/ingest`, `/ingest-upload`, `/query`, `/documents*`, `/evaluate/*`) requires a valid Cognito access token, and the frontend automatically shows the sign-in screen.

How it works: the frontend redirects to the Cognito Hosted UI (Authorization Code + PKCE, no client secret needed in the browser), exchanges the code for tokens client-side, and sends the access token as `Authorization: Bearer <token>` on every API call. The backend (`backend/auth.py`) verifies the token's signature against the user pool's JWKS, and checks `token_use=access` and `client_id` match.

## Deploy to Google Cloud Run (OpenAI + API key)

Ollama does not run inside this container by default. For cloud, use **OpenAI** (`provider: "openai"`) and store `OPENAI_API_KEY` in **Secret Manager**, then wire it as an environment variable on the service.

1. Build and push an image (replace `PROJECT_ID` and region):
   - `gcloud auth configure-docker`
   - `docker build -t gcr.io/PROJECT_ID/go-invoice:latest .`
   - `docker push gcr.io/PROJECT_ID/go-invoice:latest`
2. Create a secret for the API key (one-time):
   - `echo -n 'sk-...' | gcloud secrets create openai-api-key --data-file=-`
3. Deploy to Cloud Run:
   - `gcloud run deploy go-invoice --image gcr.io/PROJECT_ID/go-invoice:latest --region us-central1 --allow-unauthenticated --set-env-vars DISABLE_OLLAMA=true --set-secrets OPENAI_API_KEY=openai-api-key:latest`
4. Open the service URL; use the UI with **OpenAI** selected, or call `/ingest` and `/query` with `"provider": "openai"`.

Note: `--allow-unauthenticated` only controls whether Cloud Run's own IAM layer gates *inbound requests to the service*. It is independent of the app-level Cognito login above — if you configure Cognito env vars, the app itself will still require a valid Cognito token on every API call even though Cloud Run lets the request through.

**Ephemeral disk:** FAISS indexes and uploads live on the container filesystem. They are lost when the instance is replaced or scaled to zero. For production persistence, plan **Cloud Storage** (or a database) for `index.faiss` / `chunks.json` and uploads.

### Optional persistent storage with GCS

Set these env vars on Cloud Run:
- `GCS_BUCKET=<your_bucket_name>`
- `GCS_PREFIX=go-invoice` (optional)

Behavior:
- `/ingest` and `/ingest-upload` upload `index.faiss` and `chunks.json` to `gs://<bucket>/<prefix>/indexes/<provider>/...`
- uploaded PDFs are mirrored to `gs://<bucket>/<prefix>/uploads/...`
- `/query` auto-downloads index files from GCS if local files are missing
- check `/storage-status` to verify GCS config at runtime

**Local Docker test:**

```bash
docker build -t go-invoice:local .
docker run --rm -p 8080:8080 -e DISABLE_OLLAMA=true -e OPENAI_API_KEY=sk-... go-invoice:local
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
- `backend/app.py`: FastAPI app — RAG, Document Library, and Evaluation routes
- `backend/auth.py`: Cognito JWT verification (falls back to open dev mode)
- `backend/eval_store.py`: JSON-file persistence for evaluation cases/history
- `frontend/index.html` + `frontend/js/*.js`: nav-based SPA (Upload & RAG, Evaluation Dashboard, Document Library, Account) — plain ES modules, no build step
