from __future__ import annotations

import os
import time
from pathlib import Path
import shutil

from typing import Any, Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend import eval_store
from backend.auth import auth_public_config, get_current_user
from rag.pipeline import RagPipeline
from rag.vector_store import FaissStore
from rag_local.pipeline import LocalRagPipeline


def _ollama_enabled() -> bool:
    """Return False when Ollama routes are disabled (e.g. Cloud Run OpenAI-only)."""
    return os.getenv("DISABLE_OLLAMA", "").lower() not in ("1", "true", "yes")


def _require_ollama(provider: str) -> None:
    if provider == "ollama" and not _ollama_enabled():
        raise HTTPException(
            status_code=503,
            detail="Ollama is disabled on this server. Use provider=openai (API key).",
        )


def _gcs_bucket_name() -> str:
    return os.getenv("GCS_BUCKET", "").strip()


def _gcs_prefix() -> str:
    return os.getenv("GCS_PREFIX", "go-invoice").strip().strip("/")


def _gcs_enabled() -> bool:
    return bool(_gcs_bucket_name())


def _blob_path(*parts: str) -> str:
    clean_parts = [p.strip("/") for p in parts if p]
    prefix = _gcs_prefix()
    return "/".join([prefix, *clean_parts]) if prefix else "/".join(clean_parts)


def _upload_to_gcs(local_path: str, remote_path: str) -> None:
    if not _gcs_enabled():
        return
    client = storage.Client()
    bucket = client.bucket(_gcs_bucket_name())
    bucket.blob(remote_path).upload_from_filename(local_path)


def _download_from_gcs(remote_path: str, local_path: str) -> bool:
    if not _gcs_enabled():
        return False
    client = storage.Client()
    bucket = client.bucket(_gcs_bucket_name())
    blob = bucket.blob(remote_path)
    if not blob.exists():
        return False
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(local_path)
    return True


def _sync_index_to_gcs(index_dir: str, provider: str) -> None:
    if not _gcs_enabled():
        return
    index_path = Path(index_dir)
    files = ["index.faiss", "chunks.json"]
    for name in files:
        local = index_path / name
        if local.exists():
            _upload_to_gcs(str(local), _blob_path("indexes", provider, name))


def _ensure_index_from_gcs(index_dir: str, provider: str) -> None:
    if not _gcs_enabled():
        return
    files = ["index.faiss", "chunks.json"]
    for name in files:
        local = str(Path(index_dir) / name)
        if not Path(local).exists():
            _download_from_gcs(_blob_path("indexes", provider, name), local)


class IngestRequest(BaseModel):
    """Request body for ingestion endpoint.

    English: Choose provider and pass PDF path/chunk settings.
    中文: 指定 provider，並傳入 PDF 路徑與切塊參數。
    """

    provider: str = Field(default="openai", pattern="^(openai|ollama)$")
    pdf_path: str
    out_dir: str | None = None
    chunk_size: int = 700
    chunk_overlap: int = 120


class QueryRequest(BaseModel):
    """Request body for query endpoint.

    English: Ask question from a selected vector index.
    中文: 對指定索引提問並取得回答。
    """

    provider: str = Field(default="openai", pattern="^(openai|ollama)$")
    question: str
    in_dir: str | None = None
    top_k: int = 5
    intent: Literal["summarize", "qa", "compare"] | None = None
    source_filter: list[str] | None = None


app = FastAPI(title="Go Invoice API", version="0.1.0")
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# CORS: allows the Cognito Hosted UI redirect flow and any separately-hosted
# frontend build to call this API. Restrict via CORS_ALLOW_ORIGINS in prod.
_allow_origins = [o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NoCacheStaticFiles(StaticFiles):
    """Serves frontend files with `Cache-Control: no-cache`.

    English: Browsers still use the ETag/Last-Modified headers to skip
    re-downloading unchanged files (a cheap 304), but they always
    revalidate first instead of assuming a previously cached copy of
    index.html/styles.css/*.js is still current. Without this, a plain
    reload after deploying new frontend files can keep serving a stale
    cached version indefinitely.
    中文: 瀏覽器仍可透過 ETag/Last-Modified 判斷檔案未變更而回傳 304（不必重新
    下載），但每次都會先重新驗證，不會直接假設先前快取的 index.html/
    styles.css/*.js 仍是最新版本。沒有這個設定，部署新前端檔案後單純重新
    整理頁面可能仍會一直讀到舊的快取版本。
    """

    def file_response(self, *args, **kwargs):  # type: ignore[override]
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


if FRONTEND_DIR.exists():
    app.mount("/frontend", NoCacheStaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


@app.get("/")
def home() -> FileResponse:
    if not FRONTEND_DIR.exists():
        raise HTTPException(
            status_code=404,
            detail="frontend directory not found. Please ensure ./frontend exists.",
        )
    return FileResponse(FRONTEND_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/storage-status")
def storage_status() -> dict[str, object]:
    return {
        "status": "ok",
        "gcs_enabled": _gcs_enabled(),
        "gcs_bucket": _gcs_bucket_name() if _gcs_enabled() else None,
        "gcs_prefix": _gcs_prefix(),
    }


@app.get("/auth/config")
def auth_config() -> dict[str, object]:
    """Public (non-secret) Cognito settings the frontend needs to build the Hosted UI URL."""

    return auth_public_config()


@app.get("/auth/me")
def auth_me(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, object]:
    """Return the identity resolved from the caller's bearer token."""

    return {"status": "ok", "user": user}


@app.post("/ingest")
def ingest(req: IngestRequest, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, object]:
    try:
        _require_ollama(req.provider)
        if req.provider == "openai":
            index_dir = req.out_dir or "data/index"
            pipeline = RagPipeline()
            stats = pipeline.ingest_pdf(
                pdf_path=req.pdf_path,
                out_dir=index_dir,
                chunk_size=req.chunk_size,
                chunk_overlap=req.chunk_overlap,
            )
            _sync_index_to_gcs(index_dir=index_dir, provider="openai")
            return {"status": "ok", "provider": "openai", "ingest": stats}

        index_dir = req.out_dir or "data/index_local"
        pipeline = LocalRagPipeline()
        stats = pipeline.ingest_pdf(
            pdf_path=req.pdf_path,
            out_dir=index_dir,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
        )
        _sync_index_to_gcs(index_dir=index_dir, provider="ollama")
        return {"status": "ok", "provider": "ollama", "ingest": stats}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/documents")
def list_documents(
    provider: str = "openai", user: dict[str, Any] = Depends(get_current_user)
) -> dict[str, object]:
    """List indexed document sources for the selected provider."""

    if provider not in {"openai", "ollama"}:
        raise HTTPException(status_code=400, detail="provider must be openai or ollama")
    _require_ollama(provider)

    index_dir = "data/index" if provider == "openai" else "data/index_local"
    _ensure_index_from_gcs(index_dir=index_dir, provider=provider)

    try:
        if provider == "openai":
            sources = RagPipeline().list_sources(in_dir=index_dir)
        else:
            sources = LocalRagPipeline().list_sources(in_dir=index_dir)
    except FileNotFoundError:
        sources = []

    return {"status": "ok", "provider": provider, "documents": sources, "count": len(sources)}


@app.get("/documents/details")
def list_document_details(
    provider: str = "openai", user: dict[str, Any] = Depends(get_current_user)
) -> dict[str, object]:
    """List indexed documents with page/chunk counts, for the Document Library view."""

    if provider not in {"openai", "ollama"}:
        raise HTTPException(status_code=400, detail="provider must be openai or ollama")
    _require_ollama(provider)

    index_dir = "data/index" if provider == "openai" else "data/index_local"
    _ensure_index_from_gcs(index_dir=index_dir, provider=provider)
    dim = 1536 if provider == "openai" else int(os.getenv("OLLAMA_VECTOR_DIM", "768"))

    try:
        store = FaissStore.load(in_dir=index_dir, dim=dim)
    except FileNotFoundError:
        return {"status": "ok", "provider": provider, "documents": [], "count": 0}

    by_source: dict[str, dict[str, object]] = {}
    for chunk in store.chunks:
        entry = by_source.setdefault(
            chunk.source,
            {"source": chunk.source, "chunks": 0, "pages": set()},
        )
        entry["chunks"] = int(entry["chunks"]) + 1
        entry["pages"].add(chunk.page)

    documents = [
        {
            "source": name,
            "chunks": info["chunks"],
            "pages": len(info["pages"]),
        }
        for name, info in by_source.items()
    ]

    return {"status": "ok", "provider": provider, "documents": documents, "count": len(documents)}


@app.delete("/documents/{provider}/{source}")
def delete_document(
    provider: str, source: str, user: dict[str, Any] = Depends(get_current_user)
) -> dict[str, object]:
    """Remove one document (all its chunks/vectors) from a provider's index."""

    if provider not in {"openai", "ollama"}:
        raise HTTPException(status_code=400, detail="provider must be openai or ollama")
    _require_ollama(provider)

    index_dir = "data/index" if provider == "openai" else "data/index_local"
    dim = 1536 if provider == "openai" else int(os.getenv("OLLAMA_VECTOR_DIM", "768"))

    try:
        store = FaissStore.load(in_dir=index_dir, dim=dim)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No index found for this provider") from None

    removed = store.remove_source(source)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"No document named '{source}' found")

    store.save(out_dir=index_dir)
    _sync_index_to_gcs(index_dir=index_dir, provider=provider)

    return {
        "status": "ok",
        "provider": provider,
        "source": source,
        "removed_chunks": removed,
        "total_documents": len(store.list_sources()),
    }


@app.post("/query")
def query(req: QueryRequest, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, object]:
    try:
        _require_ollama(req.provider)
        if req.provider == "openai":
            index_dir = req.in_dir or "data/index"
            _ensure_index_from_gcs(index_dir=index_dir, provider="openai")
            pipeline = RagPipeline()
            result = pipeline.answer(
                question=req.question,
                in_dir=index_dir,
                top_k=req.top_k,
                intent=req.intent,
                source_filter=req.source_filter,
            )
            return {"status": "ok", "provider": "openai", **result}

        index_dir = req.in_dir or "data/index_local"
        _ensure_index_from_gcs(index_dir=index_dir, provider="ollama")
        pipeline = LocalRagPipeline()
        result = pipeline.answer(
            question=req.question,
            in_dir=index_dir,
            top_k=req.top_k,
            intent=req.intent,
            source_filter=req.source_filter,
        )
        return {"status": "ok", "provider": "ollama", **result}
    except FileNotFoundError:
        raise HTTPException(
            status_code=400,
            detail=(
                "No vector index found. Please ingest at least one PDF first "
                "(use /ingest-upload from UI or call /ingest API)."
            ),
        ) from None
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ingest-upload")
def ingest_upload(
    provider: str = Form("ollama"),
    files: list[UploadFile] = File(...),
    out_dir: str | None = Form(default=None),
    chunk_size: int = Form(default=700),
    chunk_overlap: int = Form(default=120),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, object]:
    """Upload one or more PDFs via browser and ingest them into the shared index.

    English: Saves uploads under `data/uploads`, then appends each PDF to the index.
    中文: 將上傳 PDF 存到 `data/uploads`，並逐一附加到共用索引。
    """

    if provider not in {"openai", "ollama"}:
        raise HTTPException(status_code=400, detail="provider must be openai or ollama")

    _require_ollama(provider)

    if not files:
        raise HTTPException(status_code=400, detail="at least one PDF file is required")

    index_dir = out_dir or ("data/index" if provider == "openai" else "data/index_local")
    pipeline = RagPipeline() if provider == "openai" else LocalRagPipeline()

    ingested: list[dict[str, object]] = []
    saved_paths: list[str] = []

    try:
        for file in files:
            if not file.filename:
                raise HTTPException(status_code=400, detail="missing filename")

            safe_name = Path(file.filename).name
            if not safe_name.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail=f"only PDF files are supported: {safe_name}")

            saved_path = UPLOAD_DIR / safe_name
            with saved_path.open("wb") as out:
                shutil.copyfileobj(file.file, out)
            _upload_to_gcs(str(saved_path), _blob_path("uploads", safe_name))
            saved_paths.append(str(saved_path))

            stats = pipeline.ingest_pdf(
                pdf_path=str(saved_path),
                out_dir=index_dir,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            ingested.append(stats)

        _sync_index_to_gcs(index_dir=index_dir, provider=provider)
        return {
            "status": "ok",
            "provider": provider,
            "saved_paths": saved_paths,
            "ingest": ingested,
            "total_documents": ingested[-1].get("total_documents", len(ingested)) if ingested else 0,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

class EvalCaseRequest(BaseModel):
    """A saved test question for the Evaluation Dashboard.

    English: Optional expected_answer lets a human compare against the live answer.
    中文: expected_answer 為選填，方便人工比對實際回答與預期答案。
    """

    provider: str = Field(default="openai", pattern="^(openai|ollama)$")
    question: str
    expected_answer: str | None = None
    source_filter: list[str] | None = None


class EvalRunRequest(BaseModel):
    """Run one evaluation case (or an ad-hoc question) and record the result."""

    provider: str = Field(default="openai", pattern="^(openai|ollama)$")
    question: str
    case_id: str | None = None
    expected_answer: str | None = None
    top_k: int = 5
    intent: Literal["summarize", "qa", "compare"] | None = None
    source_filter: list[str] | None = None


class EvalRateRequest(BaseModel):
    rating: Literal["pass", "fail"]
    notes: str | None = None


@app.get("/evaluate/cases")
def evaluate_list_cases(
    provider: str | None = None, user: dict[str, Any] = Depends(get_current_user)
) -> dict[str, object]:
    cases = eval_store.list_cases(provider=provider)
    return {"status": "ok", "cases": cases, "count": len(cases)}


@app.post("/evaluate/cases")
def evaluate_add_case(
    req: EvalCaseRequest, user: dict[str, Any] = Depends(get_current_user)
) -> dict[str, object]:
    case = eval_store.add_case(
        provider=req.provider,
        question=req.question,
        expected_answer=req.expected_answer,
        source_filter=req.source_filter,
    )
    return {"status": "ok", "case": case}


@app.delete("/evaluate/cases/{case_id}")
def evaluate_delete_case(
    case_id: str, user: dict[str, Any] = Depends(get_current_user)
) -> dict[str, object]:
    removed = eval_store.delete_case(case_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Test case not found")
    return {"status": "ok", "case_id": case_id}


@app.post("/evaluate/run")
def evaluate_run(
    req: EvalRunRequest, user: dict[str, Any] = Depends(get_current_user)
) -> dict[str, object]:
    """Execute a question against the live RAG pipeline and log the run for grading."""

    try:
        _require_ollama(req.provider)
        index_dir = "data/index" if req.provider == "openai" else "data/index_local"
        _ensure_index_from_gcs(index_dir=index_dir, provider=req.provider)
        pipeline = RagPipeline() if req.provider == "openai" else LocalRagPipeline()

        started = time.perf_counter()
        result = pipeline.answer(
            question=req.question,
            in_dir=index_dir,
            top_k=req.top_k,
            intent=req.intent,
            source_filter=req.source_filter,
        )
        latency_ms = (time.perf_counter() - started) * 1000

        run = eval_store.add_run(
            provider=req.provider,
            question=req.question,
            answer=result.get("answer", ""),
            intent=result.get("intent", "qa"),
            sources=result.get("sources", []),
            latency_ms=latency_ms,
            case_id=req.case_id,
            expected_answer=req.expected_answer,
        )
        return {"status": "ok", "run": run}
    except FileNotFoundError:
        raise HTTPException(
            status_code=400,
            detail="No vector index found. Ingest at least one PDF before running an evaluation.",
        ) from None
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/evaluate/history")
def evaluate_history(
    provider: str | None = None, limit: int = 200, user: dict[str, Any] = Depends(get_current_user)
) -> dict[str, object]:
    history = eval_store.list_history(provider=provider, limit=limit)
    return {"status": "ok", "history": history, "count": len(history)}


@app.post("/evaluate/history/{run_id}/rate")
def evaluate_rate(
    run_id: str, req: EvalRateRequest, user: dict[str, Any] = Depends(get_current_user)
) -> dict[str, object]:
    updated = eval_store.rate_run(run_id, rating=req.rating, notes=req.notes)
    if updated is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"status": "ok", "run": updated}


@app.get("/evaluate/summary")
def evaluate_summary(
    provider: str | None = None, user: dict[str, Any] = Depends(get_current_user)
) -> dict[str, object]:
    return {"status": "ok", "summary": eval_store.summary(provider=provider)}
