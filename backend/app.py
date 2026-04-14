from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.pipeline import RagPipeline
from rag_local.pipeline import LocalRagPipeline


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


app = FastAPI(title="DocsToData API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict[str, object]:
    try:
        if req.provider == "openai":
            pipeline = RagPipeline()
            stats = pipeline.ingest_pdf(
                pdf_path=req.pdf_path,
                out_dir=req.out_dir or "data/index",
                chunk_size=req.chunk_size,
                chunk_overlap=req.chunk_overlap,
            )
            return {"status": "ok", "provider": "openai", "ingest": stats}

        pipeline = LocalRagPipeline()
        stats = pipeline.ingest_pdf(
            pdf_path=req.pdf_path,
            out_dir=req.out_dir or "data/index_local",
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
        )
        return {"status": "ok", "provider": "ollama", "ingest": stats}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/query")
def query(req: QueryRequest) -> dict[str, object]:
    try:
        if req.provider == "openai":
            pipeline = RagPipeline()
            result = pipeline.answer(
                question=req.question,
                in_dir=req.in_dir or "data/index",
                top_k=req.top_k,
            )
            return {"status": "ok", "provider": "openai", **result}

        pipeline = LocalRagPipeline()
        result = pipeline.answer(
            question=req.question,
            in_dir=req.in_dir or "data/index_local",
            top_k=req.top_k,
        )
        return {"status": "ok", "provider": "ollama", **result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

