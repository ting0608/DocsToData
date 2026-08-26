from __future__ import annotations

import os
import time
from pathlib import Path
import shutil

from typing import Any, Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend import eval_store
from backend.auth import auth_public_config, get_current_user
from backend.runtime_paths import app_data_dir, is_lambda
from rag.evaluation import compute_retrieval_metrics, judge_generation
from rag.pipeline import RagPipeline
from rag.pricing import estimate_chat_cost_usd, is_free_provider
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
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(_gcs_bucket_name())
    bucket.blob(remote_path).upload_from_filename(local_path)


def _download_from_gcs(remote_path: str, local_path: str) -> bool:
    if not _gcs_enabled():
        return False
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(_gcs_bucket_name())
    blob = bucket.blob(remote_path)
    if not blob.exists():
        return False
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(local_path)
    return True


def _s3_bucket_name() -> str:
    return os.getenv("S3_BUCKET", "").strip()


def _s3_prefix() -> str:
    return os.getenv("S3_PREFIX", "go-invoice").strip().strip("/")


def _s3_enabled() -> bool:
    return bool(_s3_bucket_name())


def _s3_key(*parts: str) -> str:
    clean_parts = [p.strip("/") for p in parts if p]
    prefix = _s3_prefix()
    return "/".join([prefix, *clean_parts]) if prefix else "/".join(clean_parts)


def _s3_client():
    import boto3

    return boto3.client("s3")


def _upload_to_s3(local_path: str, remote_key: str) -> None:
    if not _s3_enabled():
        return
    _s3_client().upload_file(local_path, _s3_bucket_name(), remote_key)


def _download_from_s3(remote_key: str, local_path: str) -> bool:
    if not _s3_enabled():
        return False
    from botocore.exceptions import ClientError

    client = _s3_client()
    try:
        client.head_object(Bucket=_s3_bucket_name(), Key=remote_key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
            return False
        raise
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    client.download_file(_s3_bucket_name(), remote_key, local_path)
    return True


def _sync_index_to_gcs(index_dir: str, provider: str) -> None:
    """Mirror a FAISS index to remote object storage (GCS and/or S3).

    English: Both syncs can be active at once (e.g. migrating from GCS to
    AWS) — each is a no-op when its bucket env var is unset. Kept under the
    original `_sync_index_to_gcs` name so no call sites needed to change;
    the S3 half is what actually matters for the Lambda deployment
    (architecturev2.txt section 9), since Lambda's `/tmp` is wiped between
    cold starts and the index must be re-downloaded on every fresh
    execution environment.
    中文: GCS 與 S3 兩種同步可以同時啟用（例如從 GCS 搬到 AWS 的過渡期）——
    只要對應的 bucket 環境變數沒設定，該同步就會直接跳過。維持
    `_sync_index_to_gcs` 這個原本的函式名稱，讓所有呼叫端都不需要修改；
    S3 那一半才是 Lambda 部署真正需要的（對應 architecturev2.txt 第 9 節），
    因為 Lambda 的 `/tmp` 在冷啟動之間會被清空，每個新的執行環境都必須重新
    下載索引。
    """

    index_path = Path(index_dir)
    files = ["index.faiss", "chunks.json"]
    for name in files:
        local = index_path / name
        if not local.exists():
            continue
        if _gcs_enabled():
            _upload_to_gcs(str(local), _blob_path("indexes", provider, name))
        if _s3_enabled():
            _upload_to_s3(str(local), _s3_key("indexes", provider, name))


def _ensure_index_from_gcs(index_dir: str, provider: str) -> None:
    """Download a FAISS index from remote object storage if not present locally."""

    files = ["index.faiss", "chunks.json"]
    for name in files:
        local = str(Path(index_dir) / name)
        if Path(local).exists():
            continue
        if _gcs_enabled() and _download_from_gcs(_blob_path("indexes", provider, name), local):
            continue
        if _s3_enabled():
            _download_from_s3(_s3_key("indexes", provider, name), local)


def _resolve_data_dir(path: str) -> str:
    """Map a data-relative or repo-relative path to a writable location.

    English: On Lambda, only `/tmp` is writable, so both relative paths
    (e.g. "data/index") and absolute paths under the repo (e.g.
    `BASE_DIR / "data" / "uploads"`) must be redirected under `/tmp`. This
    is a no-op everywhere else (local dev, Docker, Cloud Run) where the
    repo directory itself is writable.
    中文: 在 Lambda 上只有 `/tmp` 可寫，因此相對路徑（例如 "data/index"）與
    repo 底下的絕對路徑（例如 `BASE_DIR / "data" / "uploads"`）都必須改為
    導向 `/tmp`。在其他環境（本機開發、Docker、Cloud Run）中，repo 目錄本身
    就是可寫的，此函式不做任何轉換。
    """

    if not is_lambda():
        return path

    candidate = Path(path)
    if candidate.is_absolute():
        try:
            relative = candidate.relative_to(BASE_DIR)
        except ValueError:
            return str(candidate)
        return app_data_dir(BASE_DIR, *relative.parts)

    return app_data_dir(BASE_DIR, *candidate.parts)


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
# On Lambda, only /tmp is writable, so uploads land under /tmp/data/uploads
# instead of <repo>/data/uploads (see backend/runtime_paths.py).
UPLOAD_DIR = Path(app_data_dir(BASE_DIR, "data", "uploads"))
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
        "s3_enabled": _s3_enabled(),
        "s3_bucket": _s3_bucket_name() if _s3_enabled() else None,
        "s3_prefix": _s3_prefix(),
        "runtime": "lambda" if is_lambda() else "server",
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
            index_dir = _resolve_data_dir(req.out_dir or "data/index")
            pipeline = RagPipeline()
            stats = pipeline.ingest_pdf(
                pdf_path=req.pdf_path,
                out_dir=index_dir,
                chunk_size=req.chunk_size,
                chunk_overlap=req.chunk_overlap,
            )
            _sync_index_to_gcs(index_dir=index_dir, provider="openai")
            return {"status": "ok", "provider": "openai", "ingest": stats}

        index_dir = _resolve_data_dir(req.out_dir or "data/index_local")
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

    index_dir = _resolve_data_dir("data/index" if provider == "openai" else "data/index_local")
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

    index_dir = _resolve_data_dir("data/index" if provider == "openai" else "data/index_local")
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

    index_dir = _resolve_data_dir("data/index" if provider == "openai" else "data/index_local")
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
            index_dir = _resolve_data_dir(req.in_dir or "data/index")
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

        index_dir = _resolve_data_dir(req.in_dir or "data/index_local")
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

    index_dir = _resolve_data_dir(out_dir or ("data/index" if provider == "openai" else "data/index_local"))
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
            if _s3_enabled():
                _upload_to_s3(str(saved_path), _s3_key("uploads", safe_name))
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

class GroundTruthItem(BaseModel):
    """One expected retrieval hit for a test case (Recall@K/Precision@K/MRR).

    English: Provide `chunk_id` for an exact match, or `document` (+
    optional `page`) when you only know roughly where the answer lives —
    see `rag/evaluation.py::_citation_matches`.
    中文: 若想要精確比對可填 `chunk_id`；若只知道答案大概在哪份文件（+選填
    頁碼），可填 `document`（+`page`）——見
    `rag/evaluation.py::_citation_matches`。
    """

    chunk_id: str | None = None
    document: str | None = None
    page: int | None = None


class EvalCaseRequest(BaseModel):
    """A saved test question for the Evaluation Dashboard.

    English: Optional expected_answer lets a human (or the LLM judge)
    compare against the live answer. Optional ground_truth enables
    Recall@K/Precision@K/MRR for runs of this case.
    中文: expected_answer 為選填，方便人工（或 LLM 評分）比對實際回答與預期
    答案。ground_truth 為選填，可讓此案例的執行計算 Recall@K/Precision@K/
    MRR。
    """

    provider: str = Field(default="openai", pattern="^(openai|ollama)$")
    question: str
    expected_answer: str | None = None
    source_filter: list[str] | None = None
    ground_truth: list[GroundTruthItem] | None = None


class EvalRunRequest(BaseModel):
    """Run one evaluation case (or an ad-hoc question) and record the result."""

    provider: str = Field(default="openai", pattern="^(openai|ollama)$")
    question: str
    case_id: str | None = None
    expected_answer: str | None = None
    ground_truth: list[GroundTruthItem] | None = None
    top_k: int = 5
    intent: Literal["summarize", "qa", "compare"] | None = None
    source_filter: list[str] | None = None
    run_judge: bool = Field(
        default=True,
        description="Run the LLM-as-judge pass (faithfulness/relevance/groundedness/"
        "correctness/citation_accuracy/hallucination_rate). Costs one extra LLM call.",
    )


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
        ground_truth=[g.model_dump(exclude_none=True) for g in req.ground_truth] if req.ground_truth else None,
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
    """Execute a question against the live RAG pipeline and log the run for grading.

    English: Computes all four evaluation categories from
    docs/evaluation.txt / architecturev2.txt section 7 in one pass:
    Retrieval (Recall@K/Precision@K/MRR, only if the case has ground_truth),
    Generation + Quality (faithfulness/relevance/groundedness/correctness/
    citation_accuracy/hallucination_rate, only if `run_judge=true`), and
    System (latency always, cost when the model is in `rag/pricing.py`).
    A pipeline failure is still recorded as a run (with `error` set) instead
    of being dropped, so System's Error % reflects real failures.
    中文: 一次計算 docs/evaluation.txt / architecturev2.txt 第 7 節列出的
    四大類指標：Retrieval（Recall@K/Precision@K/MRR，僅當案例有 ground_truth
    時計算）、Generation + Quality（faithfulness/relevance/groundedness/
    correctness/citation_accuracy/hallucination_rate，僅當 `run_judge=true`
    時計算）、System（latency 一定會算，cost 則在該模型有登記於
    `rag/pricing.py` 時才會算）。即使 pipeline 執行失敗，仍會記錄一筆執行
    （附上 `error`），而不是直接捨棄，讓 System 的 Error % 能反映真實的失敗
    情況。
    """

    index_dir = _resolve_data_dir("data/index" if req.provider == "openai" else "data/index_local")

    try:
        _require_ollama(req.provider)
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

        citations = result.get("citations", [])
        usage = result.get("usage")

        ground_truth = [g.model_dump(exclude_none=True) for g in req.ground_truth] if req.ground_truth else None
        if ground_truth is None and req.case_id:
            case = eval_store.get_case(req.case_id)
            ground_truth = case.get("ground_truth") if case else None
        retrieval_metrics = compute_retrieval_metrics(citations, ground_truth, k=req.top_k)

        judge_scores = None
        if req.run_judge:
            context = "\n\n".join(
                f"[{c.get('document')} p.{c.get('page')}]" for c in citations
            ) or "(no citations returned)"
            judge_scores = judge_generation(
                chat_fn=pipeline.chat,
                question=req.question,
                answer=result.get("answer", ""),
                context=context,
                citations=citations,
                expected_answer=req.expected_answer,
            )
            # judge_generation's own LLM call adds to the same usage accumulator.
            usage = result.get("usage")

        prompt_tokens = usage.prompt_tokens if usage else None
        completion_tokens = usage.completion_tokens if usage else None
        cost_usd = None
        if usage:
            if is_free_provider(req.provider):
                cost_usd = 0.0
            else:
                cost_usd = estimate_chat_cost_usd(pipeline.chat_model_name, usage)

        run = eval_store.add_run(
            provider=req.provider,
            question=req.question,
            answer=result.get("answer", ""),
            intent=result.get("intent", "qa"),
            sources=result.get("sources", []),
            latency_ms=latency_ms,
            case_id=req.case_id,
            expected_answer=req.expected_answer,
            citations=citations,
            retrieval_metrics=retrieval_metrics,
            judge_scores=judge_scores,
            cost_usd=cost_usd,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        return {"status": "ok", "run": run}
    except FileNotFoundError:
        raise HTTPException(
            status_code=400,
            detail="No vector index found. Ingest at least one PDF before running an evaluation.",
        ) from None
    except Exception as exc:
        # Record the failure as a run (System Metrics -> Error %) rather than
        # silently dropping it, then still surface a 400 to the caller.
        eval_store.add_run(
            provider=req.provider,
            question=req.question,
            answer="",
            intent=req.intent or "qa",
            sources=[],
            latency_ms=0.0,
            case_id=req.case_id,
            expected_answer=req.expected_answer,
            error=str(exc),
        )
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
