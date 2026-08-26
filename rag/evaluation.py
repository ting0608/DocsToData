from __future__ import annotations

import json
import re
from typing import Callable, TypedDict

"""RAG evaluation metrics: retrieval (ground-truth based) + generation
(LLM-as-judge based), per architecturev2.txt section 7.

English: Split into two independent halves on purpose:
- `compute_retrieval_metrics` is pure math over citations vs. a ground-truth
  set — deterministic, free, no LLM call.
- `judge_generation` asks an LLM to score the answer against its own
  retrieved context — this is a real LLM call (costs tokens, is
  non-deterministic) and is therefore optional per run (see
  `EvalRunRequest.run_judge` in backend/app.py).
中文: 刻意分成兩個獨立部分：
- `compute_retrieval_metrics` 是純數學運算，比較 citations 與事先設定的
  ground truth——結果穩定、不花錢，也不需要呼叫 LLM。
- `judge_generation` 會請 LLM 依據自己檢索到的 context 為答案評分——這是一次
  真正的 LLM 呼叫（會消耗 token、結果不保證穩定），因此每次執行是否要跑這一步
  是可選的（見 backend/app.py 的 `EvalRunRequest.run_judge`）。
"""


class GroundTruthItem(TypedDict, total=False):
    chunk_id: str
    document: str
    page: int


def _citation_matches(citation: dict[str, object], item: GroundTruthItem) -> bool:
    """A retrieved citation counts as a hit against one ground-truth item.

    English: Prefer exact `chunk_id` match when the ground truth specifies
    one (most precise). Otherwise fall back to `document` + `page`, since
    hand-authoring exact chunk_ids for a test set is tedious but knowing
    "the answer is on page 14 of employee_handbook.pdf" is easy — this
    mirrors the `expected_pages` example in architecturev2.txt section 7.
    中文: 若 ground truth 有指定 `chunk_id`，優先做精確比對（最準確）。否則退回
    用 `document` + `page` 比對，因為手動標註精確 chunk_id 很費工，但標註「答案
    在 employee_handbook.pdf 第 14 頁」相對容易——對應 architecturev2.txt
    第 7 節 `expected_pages` 的範例。
    """

    gt_chunk_id = item.get("chunk_id")
    if gt_chunk_id:
        return citation.get("chunk_id") == gt_chunk_id

    gt_document = item.get("document")
    gt_page = item.get("page")
    if gt_document is not None and gt_page is not None:
        return citation.get("document") == gt_document and citation.get("page") == gt_page
    if gt_document is not None:
        return citation.get("document") == gt_document
    return False


def compute_retrieval_metrics(
    citations: list[dict[str, object]],
    ground_truth: list[GroundTruthItem] | None,
    k: int | None = None,
) -> dict[str, object] | None:
    """Recall@K, Precision@K, and MRR of retrieved citations vs. ground truth.

    English: Returns None (not zeros) when no ground truth was supplied for
    the case, so callers/UI can distinguish "not evaluated" from "evaluated
    and scored 0". `citations` is the ordered list already returned by the
    pipeline (`rag/graph/nodes.py::_hits_to_citations`) — order encodes rank.
    中文: 若該測試案例沒有提供 ground truth，回傳 None（而非 0），讓呼叫端/UI
    可以區分「尚未評估」與「評估後得到 0 分」。`citations` 是 pipeline 已回傳的
    排序後清單（`rag/graph/nodes.py::_hits_to_citations`）——清單順序即代表
    排名。
    """

    if not ground_truth:
        return None

    top_citations = citations[:k] if k is not None else citations

    matched_gt_indices: set[int] = set()
    first_hit_rank: int | None = None
    for rank, citation in enumerate(top_citations, start=1):
        for gt_index, item in enumerate(ground_truth):
            if gt_index in matched_gt_indices:
                continue
            if _citation_matches(citation, item):
                matched_gt_indices.add(gt_index)
                if first_hit_rank is None:
                    first_hit_rank = rank

    precision_at_k = len(matched_gt_indices) / len(top_citations) if top_citations else 0.0
    recall_at_k = len(matched_gt_indices) / len(ground_truth)
    mrr = round(1.0 / first_hit_rank, 4) if first_hit_rank else 0.0

    return {
        "k": k if k is not None else len(top_citations),
        "recall_at_k": round(recall_at_k, 4),
        "precision_at_k": round(precision_at_k, 4),
        "mrr": mrr,
        "matched_count": len(matched_gt_indices),
        "ground_truth_count": len(ground_truth),
        "first_hit_rank": first_hit_rank,
    }


# ---- Generation quality: LLM-as-judge --------------------------------------

_JUDGE_SYSTEM = """You are a strict RAG answer evaluator. You will be given a question, the \
retrieved context that was available to the assistant, the assistant's answer, its citations, \
and optionally a human-written expected answer. Score the answer on these dimensions, each a \
float from 0.0 (worst) to 1.0 (best), except hallucination_rate which is 0.0 (no hallucination) \
to 1.0 (fully hallucinated):

- faithfulness: does the answer only state things supported by the retrieved context?
- relevance: does the answer actually address the question asked?
- groundedness: is every specific claim in the answer traceable to a specific citation?
- correctness: how well does the answer match the expected answer (0.0 if no expected answer given, use null instead)
- citation_accuracy: do the citations actually contain the information used to answer?
- hallucination_rate: fraction of the answer that is fabricated / not supported by context or citations

Reply with JSON only, no markdown fences, no extra text:
{"faithfulness": 0.0, "relevance": 0.0, "groundedness": 0.0, "correctness": 0.0 or null, \
"citation_accuracy": 0.0, "hallucination_rate": 0.0, "reasoning": "one short sentence"}"""

_JUDGE_KEYS = (
    "faithfulness",
    "relevance",
    "groundedness",
    "correctness",
    "citation_accuracy",
    "hallucination_rate",
)


def _extract_json_object(raw: str) -> dict[str, object] | None:
    """Best-effort JSON extraction from an LLM judge response.

    English: Judge models sometimes wrap JSON in markdown fences or add a
    sentence before/after despite instructions — this grabs the first
    balanced-looking `{...}` block rather than failing outright.
    中文: 評分模型有時會用 markdown 包住 JSON，或在前後多加一句話——這裡會
    嘗試抓出第一段看起來完整的 `{...}` 區塊，而不是直接判定失敗。
    """

    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def judge_generation(
    chat_fn: Callable[[str, str], str],
    question: str,
    answer: str,
    context: str,
    citations: list[dict[str, object]],
    expected_answer: str | None = None,
) -> dict[str, object]:
    """Score one answer on faithfulness/relevance/groundedness/correctness/
    citation_accuracy/hallucination_rate using an LLM judge.

    English: `chat_fn` is deliberately just `(system_prompt, user_prompt) ->
    str` — the exact shape of `BaseRagPipeline.chat` — so callers can pass
    the *same* pipeline (cheap, consistent with the run's own provider) or a
    different, stronger judge model by wrapping another gateway with the
    same signature. Returns a dict with all `_JUDGE_KEYS` plus `reasoning`;
    on any parse failure all score keys are None and `judge_error` is set,
    rather than raising, so a flaky judge call doesn't fail the whole
    evaluation run.
    中文: `chat_fn` 刻意設計成單純的 `(system_prompt, user_prompt) -> str`——
    與 `BaseRagPipeline.chat` 的介面完全相同，因此呼叫端可以直接傳入*同一個*
    pipeline（成本低、與該次執行使用的 provider 一致），或包一層別的、更強的
    評分模型 gateway，只要符合相同介面即可。回傳值包含所有 `_JUDGE_KEYS` 以及
    `reasoning`；若解析失敗，所有分數欄位都會是 None，並附上 `judge_error`，
    而不會直接拋出例外，避免評分呼叫不穩定時讓整個評估執行失敗。
    """

    citations_text = "\n".join(
        f"- {c.get('document')} p.{c.get('page')} (chunk_id={c.get('chunk_id')})" for c in citations
    ) or "(no citations)"
    expected_block = f"\nExpected answer:\n{expected_answer}" if expected_answer else "\n(no expected answer provided)"

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Retrieved context given to the assistant:\n{context}\n\n"
        f"Assistant's answer:\n{answer}\n\n"
        f"Citations returned with the answer:\n{citations_text}"
        f"{expected_block}"
    )

    raw = chat_fn(_JUDGE_SYSTEM, user_prompt)
    parsed = _extract_json_object(raw)
    if parsed is None:
        result: dict[str, object] = {key: None for key in _JUDGE_KEYS}
        result["reasoning"] = None
        result["judge_error"] = f"Could not parse judge response: {raw[:200]!r}"
        return result

    result = {key: parsed.get(key) for key in _JUDGE_KEYS}
    result["reasoning"] = parsed.get("reasoning")
    result["judge_error"] = None
    return result
