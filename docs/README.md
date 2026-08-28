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

How it works (classic email/password flow): the in-app form posts the email + password to the backend (`POST /auth/login`), which calls Cognito's `InitiateAuth` (`USER_PASSWORD_AUTH`) server-side via boto3 and returns the access / id / refresh tokens. The frontend stores them in `sessionStorage` and sends the access token as `Authorization: Bearer <token>` on every API call. When the access token nears expiry (or a call returns 401), the frontend calls `POST /auth/refresh` with the refresh token to get a new access token transparently. Sign-up (`POST /auth/signup` + emailed code -> `POST /auth/confirm-signup`) and sign-out (`POST /auth/logout` -> Cognito `GlobalSignOut`) go through the same backend. The backend (`backend/auth.py`) still verifies every access token's signature against the user pool's JWKS and checks `token_use=access` and `client_id` match. The Hosted UI redirect (Authorization Code + PKCE) remains available for enterprise SSO via the optional "Continue with SSO" button.

Because the app client uses `ALLOW_USER_PASSWORD_AUTH`, enable that flow on the app client if you create the pool manually (SAM/`template.yaml` sets it automatically). Cognito access tokens do not carry the `email` claim; the frontend reads it from the id token for display only.

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

Top-level layout:

```
.
├── backend/            FastAPI app + AWS Lambda handler
├── rag/                OpenAI RAG pipeline (default provider)
├── rag_local/          Ollama RAG pipeline (local provider)
├── rag_aws/            Bedrock RAG pipeline (AWS provider)
├── frontend/           Static SPA (no build step)
├── data/               Local uploads + FAISS indexes (gitignored)
├── docs/               README, changelog, architecture notes
├── Dockerfile          Cloud Run / local uvicorn image
├── Dockerfile.lambda   AWS Lambda container image
├── docker-compose.yml  Local dev (mounts data/, points Ollama at host)
├── template.yaml       AWS SAM stack (Lambda + API Gateway + S3 + DynamoDB + Cognito)
├── samconfig.toml      SAM deploy defaults (no secrets)
├── amplify.yml         Amplify Hosting build spec for the frontend
├── requirements.txt          Runtime deps
└── requirements-lambda.txt   Extra deps only needed on Lambda
```

Backend (`backend/`):
- `app.py`: FastAPI app — RAG, Document Library, Evaluation, and auth routes
- `auth.py`: Cognito JWT verification + email/password auth (falls back to open dev mode)
- `eval_store.py`: evaluation persistence — JSON files locally (`EVAL_STORE_BACKEND=json`, default) or DynamoDB on Lambda (`EVAL_STORE_BACKEND=dynamodb`)
- `lambda_handler.py`: AWS Lambda entrypoint wrapping the FastAPI app
- `runtime_paths.py`: resolves writable paths (local disk vs Lambda `/tmp`)

RAG pipelines (one per provider, same interface):
- `rag/`: OpenAI embedding + retrieval + answer flow. Key modules: `pdf_parser.py` (PyMuPDF text extraction), `chunking.py` (token-aware chunking), `retrieval.py`, `vector_store.py` (FAISS + metadata), `evaluation.py` (retrieval/judge metrics), `pricing.py` (cost estimates), `pipeline.py`, `cli.py`.
- `rag_local/`: Ollama pipeline + CLI (local, no API key).
- `rag_aws/`: Bedrock pipeline (AWS cloud provider).

Frontend (`frontend/`, plain ES modules, no build step):
- `index.html`, `styles.css`: nav-based SPA shell (Upload & RAG, Evaluation Dashboard, Document Library, Account).
- `js/`: `main.js` (bootstrap/nav wiring), `api.js` (fetch + token refresh), `auth.js` + `authView.js` (Cognito flows), `guest.js` (guest-preview mode), `uploadRag.js`, `library.js`, `evaluation.js`, `state.js`, `nav.js`, `toast.js`, `config.js`.
- `assets/`: logo + provider icons. `data/guest-data.json`: hardcoded sample data shown in guest-preview mode.

### Guest preview mode

The Account screen offers a **Continue as guest** button that bypasses login and populates every screen with hardcoded sample data from `frontend/data/guest-data.json`. In guest mode the app makes no backend calls and every mutating action (upload, query, run evaluation, save/delete/grade) is disabled with a prompt to sign in. Guest state is not persisted — a page refresh returns to the real login screen.

### AWS deployment (Lambda + Amplify)

The backend also deploys as a container-image Lambda behind an HTTP API via AWS SAM (`template.yaml`), with S3 for uploads/index sync, DynamoDB for the evaluation store, and an optional Cognito user pool. The frontend deploys separately to Amplify Hosting (`amplify.yml`), which injects the API Gateway URL into `frontend/js/config.js` at build time via the `API_BASE_URL` environment variable. Build/deploy:

```bash
sam build
sam deploy --parameter-overrides OpenAiApiKey=sk-... BudgetAlertEmail=you@example.com
```
