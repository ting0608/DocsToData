from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

"""Lightweight JSON-file persistence for the Evaluation Dashboard.

English: No database in this project yet, so evaluation test cases and run
history are stored as JSON files under `data/`. A process-wide lock keeps
concurrent requests from corrupting the files.
中文: 專案目前沒有資料庫，因此評估用的測試案例與執行紀錄先以 JSON 檔案儲存在
`data/` 目錄下，並用全域鎖避免併發請求造成檔案損毀。
"""

_LOCK = threading.Lock()
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_CASES_PATH = _DATA_DIR / "eval_cases.json"
_HISTORY_PATH = _DATA_DIR / "eval_history.json"


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
    with _LOCK:
        cases = _read_json_list(_CASES_PATH)
    if provider:
        cases = [c for c in cases if c.get("provider") == provider]
    return cases


def add_case(
    provider: str,
    question: str,
    expected_answer: str | None = None,
    source_filter: list[str] | None = None,
) -> dict[str, Any]:
    case = {
        "id": str(uuid.uuid4()),
        "provider": provider,
        "question": question,
        "expected_answer": expected_answer,
        "source_filter": source_filter or None,
        "created_at": _now_iso(),
    }
    with _LOCK:
        cases = _read_json_list(_CASES_PATH)
        cases.append(case)
        _write_json_list(_CASES_PATH, cases)
    return case


def delete_case(case_id: str) -> bool:
    with _LOCK:
        cases = _read_json_list(_CASES_PATH)
        remaining = [c for c in cases if c.get("id") != case_id]
        removed = len(remaining) != len(cases)
        if removed:
            _write_json_list(_CASES_PATH, remaining)
    return removed


def get_case(case_id: str) -> dict[str, Any] | None:
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
) -> dict[str, Any]:
    run = {
        "id": str(uuid.uuid4()),
        "case_id": case_id,
        "provider": provider,
        "question": question,
        "answer": answer,
        "expected_answer": expected_answer,
        "intent": intent,
        "sources": sources,
        "latency_ms": round(latency_ms, 1),
        "rating": None,  # "pass" | "fail" | None (not yet graded)
        "notes": None,
        "created_at": _now_iso(),
    }
    with _LOCK:
        history = _read_json_list(_HISTORY_PATH)
        history.append(run)
        _write_json_list(_HISTORY_PATH, history)
    return run


def list_history(provider: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    with _LOCK:
        history = _read_json_list(_HISTORY_PATH)
    if provider:
        history = [h for h in history if h.get("provider") == provider]
    return history[-limit:][::-1]  # newest first


def rate_run(run_id: str, rating: str, notes: str | None = None) -> dict[str, Any] | None:
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


def summary(provider: str | None = None) -> dict[str, Any]:
    history = list_history(provider=provider, limit=100000)
    total = len(history)
    graded = [h for h in history if h.get("rating") in ("pass", "fail")]
    passed = [h for h in graded if h.get("rating") == "pass"]
    latencies = [h.get("latency_ms", 0) for h in history if isinstance(h.get("latency_ms"), (int, float))]
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0.0
    pass_rate = round((len(passed) / len(graded)) * 100, 1) if graded else None
    return {
        "total_runs": total,
        "graded_runs": len(graded),
        "ungraded_runs": total - len(graded),
        "pass_count": len(passed),
        "fail_count": len(graded) - len(passed),
        "pass_rate_pct": pass_rate,
        "avg_latency_ms": avg_latency,
    }
