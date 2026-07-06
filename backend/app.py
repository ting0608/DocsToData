from __future__ import annotations

import os
from pathlib import Path
import shutil

from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from google.cloud import storage
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag.pipeline import RagPipeline
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
    return os.getenv("GCS_PREFIX", "docstodata").strip().strip("/")


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


app = FastAPI(title="DocsToData API", version="0.1.0")
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


@app.get("/")
def home() -> FileResponse:
    if not FRONTEND_DIR.exists():
        raise HTTPException(
            status_code=404,
            detail="frontend directory not found. Please ensure ./frontend exists.",
        )
    return FileResponse(FRONTEND_DIR / "index.html")


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


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict[str, object]:
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
def list_documents(provider: str = "openai") -> dict[str, object]:
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


@app.post("/query")
def query(req: QueryRequest) -> dict[str, object]:
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

