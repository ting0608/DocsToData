from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

"""Persistence for the Evaluation Dashboard: JSON files (local/Docker) or
DynamoDB (AWS Lambda).

English: Historically this project had no database, so evaluation test
cases and run history were stored as JSON files under `data/`. On AWS
Lambda that no longer works — the filesystem is read-only except for the
ephemeral, per-instance `/tmp` — so this module now supports a second
backend, DynamoDB, selected via `EVAL_STORE_BACKEND=dynamodb`. The public
functions (`list_cases`, `add_case`, `add_run`, etc.) keep the same
signatures either way, so `backend/app.py` and its callers need no changes.
中文: 這個專案原本沒有資料庫，評估用的測試案例與執行紀錄以 JSON 檔案儲存在
`data/` 目錄。但在 AWS Lambda 上檔案系統唯讀（只有暫時性、每個執行環境各自的
`/tmp` 可寫），因此這個模組新增了第二種後端 DynamoDB，透過
`EVAL_STORE_BACKEND=dynamodb` 切換。對外的函式（`list_cases`、`add_case`、
`add_run` 等）在兩種後端下簽章都保持一致，所以 `backend/app.py` 與其呼叫端
不需要任何修改。
"""

_LOCK = threading.Lock()
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_CASES_PATH = _DATA_DIR / "eval_cases.json"
_HISTORY_PATH = _DATA_DIR / "eval_history.json"

_dynamo_resource = None
_dynamo_lock = threading.Lock()


def _backend() -> str:
    return os.getenv("EVAL_STORE_BACKEND", "json").strip().lower()


def _use_dynamo() -> bool:
    return _backend() == "dynamodb"


def _cases_table_name() -> str:
    return os.getenv("EVAL_CASES_TABLE", "").strip()


def _history_table_name() -> str:
    return os.getenv("EVAL_HISTORY_TABLE", "").strip()


def _dynamo():
    """Lazily create (and cache) the boto3 DynamoDB resource.

    English: Imported lazily so `boto3` is only required when the DynamoDB
    backend is actually selected — local/Docker JSON-file usage never needs
    `boto3` installed.
    中文: 延遲匯入，讓 `boto3` 只在真正選用 DynamoDB 後端時才需要——本機/
    Docker 使用 JSON 檔案時完全不需要安裝 `boto3`。
    """

    global _dynamo_resource
    with _dynamo_lock:
        if _dynamo_resource is None:
            import boto3

            _dynamo_resource = boto3.resource("dynamodb")
        return _dynamo_resource


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _write_json_list(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


# ---- Test cases -----------------------------------------------------------


def list_cases(provider: str | None = None) -> list[dict[str, Any]]:
    if _use_dynamo():
        table = _dynamo().Table(_cases_table_name())
        items = table.scan().get("Items", [])
        cases = [dict(item) for item in items]
    else:
        with _LOCK:
            cases = _read_json_list(_CASES_PATH)
    if provider:
        cases = [c for c in cases if c.get("provider") == provider]
    return sorted(cases, key=lambda c: c.get("created_at", ""))


def add_case(
    provider: str,
    question: str,
    expected_answer: str | None = None,
    source_filter: list[str] | None = None,
    ground_truth: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Save a reusable test question.

    English: `ground_truth` is an optional list of `{chunk_id}` or
    `{document, page}` items identifying which chunks *should* be retrieved
    to answer this question — enables Recall@K/Precision@K/MRR (see
    `rag/evaluation.py::compute_retrieval_metrics`). Without it, retrieval
    metrics are simply not computed for runs of this case.
    中文: `ground_truth` 為選填，是一組 `{chunk_id}` 或 `{document, page}`
    項目，標明回答此問題「應該」檢索到哪些 chunk，用來計算 Recall@K/
    Precision@K/MRR（見 `rag/evaluation.py::compute_retrieval_metrics`）。
    若未提供，該案例執行時就不會計算檢索指標。
    """

    case = {
        "id": str(uuid.uuid4()),
        "provider": provider,
        "question": question,
        "expected_answer": expected_answer,
        "source_filter": source_filter or None,
        "ground_truth": ground_truth or None,
        "created_at": _now_iso(),
    }
    if _use_dynamo():
        table = _dynamo().Table(_cases_table_name())
        table.put_item(Item=_dynamo_safe(case))
    else:
        with _LOCK:
            cases = _read_json_list(_CASES_PATH)
            cases.append(case)
            _write_json_list(_CASES_PATH, cases)
    return case


def delete_case(case_id: str) -> bool:
    if _use_dynamo():
        table = _dynamo().Table(_cases_table_name())
        existing = table.get_item(Key={"id": case_id}).get("Item")
        if not existing:
            return False
        table.delete_item(Key={"id": case_id})
        return True

    with _LOCK:
        cases = _read_json_list(_CASES_PATH)
        remaining = [c for c in cases if c.get("id") != case_id]
        removed = len(remaining) != len(cases)
        if removed:
            _write_json_list(_CASES_PATH, remaining)
    return removed


def get_case(case_id: str) -> dict[str, Any] | None:
    if _use_dynamo():
        table = _dynamo().Table(_cases_table_name())
        item = table.get_item(Key={"id": case_id}).get("Item")
        return dict(item) if item else None

    for case in list_cases():
        if case.get("id") == case_id:
            return case
    return None


# ---- Run history -----------------------------------------------------------


def add_run(
    provider: str,
    question: str,
    answer: str,
    intent: str,
    sources: list[str],
    latency_ms: float,
    case_id: str | None = None,
    expected_answer: str | None = None,
    citations: list[dict[str, Any]] | None = None,
    retrieval_metrics: dict[str, Any] | None = None,
    judge_scores: dict[str, Any] | None = None,
    cost_usd: float | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Record one evaluation run across all four metric categories.

    English: `retrieval_metrics` comes from
    `rag/evaluation.py::compute_retrieval_metrics` (None when the case has no
    ground truth). `judge_scores` comes from
    `rag/evaluation.py::judge_generation` (None when the judge step was
    skipped for this run, e.g. `run_judge=false`). `cost_usd` is None when
    the provider/model isn't in `rag/pricing.py`'s table (distinct from
    Ollama's actual $0). `error` is set instead of the run being silently
    dropped when the pipeline raises — this is what lets `summary()` compute
    Error % (System Metrics).
    中文: `retrieval_metrics` 來自
    `rag/evaluation.py::compute_retrieval_metrics`（若該案例沒有 ground
    truth 則為 None）。`judge_scores` 來自
    `rag/evaluation.py::judge_generation`（若本次執行跳過評分步驟，例如
    `run_judge=false`，則為 None）。`cost_usd` 在 provider/model 不在
    `rag/pricing.py` 價格表中時為 None（與 Ollama 實際 $0 成本不同，避免
    混淆「未知」與「已知免費」）。當 pipeline 拋出例外時，改為記錄 `error`
    而非直接捨棄該次執行——這正是讓 `summary()` 能計算 Error %（System
    Metrics）的關鍵。
    """

    run = {
        "id": str(uuid.uuid4()),
        "case_id": case_id,
        "provider": provider,
        "question": question,
        "answer": answer,
        "expected_answer": expected_answer,
        "intent": intent,
        "sources": sources,
        "citations": citations or [],
        "retrieval_metrics": retrieval_metrics,
        "judge_scores": judge_scores,
        "cost_usd": cost_usd,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_ms": round(latency_ms, 1),
        "error": error,
        "rating": None,  # "pass" | "fail" | None (not yet graded)
        "notes": None,
        "created_at": _now_iso(),
    }
    if _use_dynamo():
        table = _dynamo().Table(_history_table_name())
        table.put_item(Item=_dynamo_safe(run))
    else:
        with _LOCK:
            history = _read_json_list(_HISTORY_PATH)
            history.append(run)
            _write_json_list(_HISTORY_PATH, history)
    return run


def list_history(provider: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    if _use_dynamo():
        table = _dynamo().Table(_history_table_name())
        items = table.scan().get("Items", [])
        history = [dict(item) for item in items]
    else:
        with _LOCK:
            history = _read_json_list(_HISTORY_PATH)

    if provider:
        history = [h for h in history if h.get("provider") == provider]
    history = sorted(history, key=lambda h: h.get("created_at", ""))
    return history[-limit:][::-1]  # newest first


def rate_run(run_id: str, rating: str, notes: str | None = None) -> dict[str, Any] | None:
    if _use_dynamo():
        table = _dynamo().Table(_history_table_name())
        existing = table.get_item(Key={"id": run_id}).get("Item")
        if not existing:
            return None
        table.update_item(
            Key={"id": run_id},
            UpdateExpression="SET rating = :r, notes = :n",
            ExpressionAttributeValues={":r": rating, ":n": notes},
        )
        updated = dict(existing)
        updated["rating"] = rating
        updated["notes"] = notes
        return updated

    with _LOCK:
        history = _read_json_list(_HISTORY_PATH)
        updated = None
        for run in history:
            if run.get("id") == run_id:
                run["rating"] = rating
                run["notes"] = notes
                updated = run
                break
        if updated is not None:
            _write_json_list(_HISTORY_PATH, history)
    return updated


def _avg(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def summary(provider: str | None = None) -> dict[str, Any]:
    """Aggregate metrics across all four RAG evaluation categories.

    English: Mirrors the quadrant in architecturev2.txt section 7 —
    Retrieval (recall/precision/MRR, averaged over runs whose case had
    ground truth), Generation (faithfulness/relevance/groundedness, from
    LLM-judge runs), Quality (correctness/citation_accuracy/hallucination,
    also judge-derived, plus the existing human pass/fail rating), and
    System (latency, cost, error %). Every averaged field is None (not 0)
    when no run in the filtered history has that data, so the dashboard can
    show "no data yet" instead of a misleading zero.
    中文: 對應 architecturev2.txt 第 7 節的四個象限——Retrieval（recall/
    precision/MRR，只對案例有 ground truth 的執行取平均）、Generation
    （faithfulness/relevance/groundedness，來自 LLM 評分執行）、Quality
    （correctness/citation_accuracy/hallucination，同樣來自評分，加上原有的
    人工 pass/fail 評分）、System（latency、cost、error %）。當篩選後的歷史
    紀錄中沒有任何一筆有該項資料時，對應的平均欄位會是 None（而非 0），讓
    儀表板可以顯示「尚無資料」而不是誤導性的 0。
    """

    history = list_history(provider=provider, limit=100000)
    total = len(history)

    errored = [h for h in history if h.get("error")]
    ok_runs = [h for h in history if not h.get("error")]

    graded = [h for h in ok_runs if h.get("rating") in ("pass", "fail")]
    passed = [h for h in graded if h.get("rating") == "pass"]
    pass_rate = round((len(passed) / len(graded)) * 100, 1) if graded else None

    latencies = [h["latency_ms"] for h in history if isinstance(h.get("latency_ms"), (int, float))]
    costs = [h["cost_usd"] for h in ok_runs if isinstance(h.get("cost_usd"), (int, float))]
    error_rate_pct = round((len(errored) / total) * 100, 1) if total else None

    retrieval_runs = [h["retrieval_metrics"] for h in ok_runs if h.get("retrieval_metrics")]
    judge_runs = [h["judge_scores"] for h in ok_runs if h.get("judge_scores") and not h["judge_scores"].get("judge_error")]
    correctness_scores = [j["correctness"] for j in judge_runs if isinstance(j.get("correctness"), (int, float))]

    return {
        "total_runs": total,
        "graded_runs": len(graded),
        "ungraded_runs": len(ok_runs) - len(graded),
        "pass_count": len(passed),
        "fail_count": len(graded) - len(passed),
        "pass_rate_pct": pass_rate,
        # Retrieval
        "avg_recall_at_k": _avg([r["recall_at_k"] for r in retrieval_runs]),
        "avg_precision_at_k": _avg([r["precision_at_k"] for r in retrieval_runs]),
        "avg_mrr": _avg([r["mrr"] for r in retrieval_runs]),
        "retrieval_scored_runs": len(retrieval_runs),
        # Generation
        "avg_faithfulness": _avg([j["faithfulness"] for j in judge_runs if isinstance(j.get("faithfulness"), (int, float))]),
        "avg_relevance": _avg([j["relevance"] for j in judge_runs if isinstance(j.get("relevance"), (int, float))]),
        "avg_groundedness": _avg([j["groundedness"] for j in judge_runs if isinstance(j.get("groundedness"), (int, float))]),
        # Quality
        "avg_correctness": _avg(correctness_scores),
        "avg_citation_accuracy": _avg([j["citation_accuracy"] for j in judge_runs if isinstance(j.get("citation_accuracy"), (int, float))]),
        "avg_hallucination_rate": _avg([j["hallucination_rate"] for j in judge_runs if isinstance(j.get("hallucination_rate"), (int, float))]),
        "judge_scored_runs": len(judge_runs),
        # System
        "avg_latency_ms": _avg(latencies) or 0.0,
        "avg_cost_usd": _avg(costs),
        "cost_known_runs": len(costs),
        "error_count": len(errored),
        "error_rate_pct": error_rate_pct,
    }


def _to_dynamo_value(value: Any) -> Any:
    """Recursively convert a value into DynamoDB-safe types.

    English: boto3's DynamoDB Table resource rejects native Python `float`
    for its Number type ("Float types are not supported. Use Decimal
    types instead.") — it only accepts `Decimal`. This project's eval data
    is full of floats (latency_ms, cost_usd, and every metric inside
    retrieval_metrics/judge_scores from rag/evaluation.py), so every float
    anywhere in the item — including nested inside dicts/lists — must be
    converted before `put_item`/`update_item`. `str(value)` avoids the
    binary-float-to-Decimal precision surprises `Decimal(value)` directly on
    a float can produce (e.g. Decimal(0.1) != Decimal("0.1")).
    中文: boto3 的 DynamoDB Table resource 不接受原生 Python `float` 作為
    Number 型別（會出現「Float types are not supported. Use Decimal types
    instead.」），只接受 `Decimal`。這個專案的評估資料裡到處都是 float
    （latency_ms、cost_usd，以及 rag/evaluation.py 產生、放在
    retrieval_metrics/judge_scores 裡的每一項指標），所以在
    `put_item`/`update_item` 之前，物件內任何位置（包含巢狀 dict/list）的
    float 都必須轉換。用 `str(value)` 是為了避免直接對 float 做
    `Decimal(value)` 時的二進位浮點數精度問題（例如 Decimal(0.1) 不等於
    Decimal("0.1")）。
    """

    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {k: _to_dynamo_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_dynamo_value(v) for v in value]
    return value


def _dynamo_safe(item: dict[str, Any]) -> dict[str, Any]:
    """Drop `None` values and convert floats to `Decimal` before writing.

    English: DynamoDB's `put_item` rejects top-level `None`/empty-set values
    for some attribute types; simplest fix is to omit unset optional fields
    rather than writing them as `NULL`. It also rejects native `float`
    values anywhere in the item (see `_to_dynamo_value`).
    中文: DynamoDB 的 `put_item` 對某些屬性型別不接受頂層的 `None`/空集合值，
    最簡單的做法是省略未設定的選填欄位，而不是寫成 `NULL`。它也不接受物件內
    任何位置的原生 `float` 值（見 `_to_dynamo_value`）。
    """

    return {k: _to_dynamo_value(v) for k, v in item.items() if v is not None}
