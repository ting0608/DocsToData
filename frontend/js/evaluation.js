// Evaluation Dashboard view: save test questions, run them against the live
// RAG pipeline, grade the answers pass/fail, and track aggregate metrics.

import { api } from "./api.js";
import { currentProvider, onProviderChange } from "./state.js";
import { showToast } from "./toast.js";

const totalRunsEl = document.getElementById("evalTotalRuns");
const passRateEl = document.getElementById("evalPassRate");
const avgLatencyEl = document.getElementById("evalAvgLatency");
const avgCostEl = document.getElementById("evalAvgCost");
const errorRateEl = document.getElementById("evalErrorRate");
const recallEl = document.getElementById("evalRecall");
const precisionEl = document.getElementById("evalPrecision");
const mrrEl = document.getElementById("evalMrr");
const faithfulnessEl = document.getElementById("evalFaithfulness");
const relevanceEl = document.getElementById("evalRelevance");
const groundednessEl = document.getElementById("evalGroundedness");
const correctnessEl = document.getElementById("evalCorrectness");
const citationAccuracyEl = document.getElementById("evalCitationAccuracy");
const hallucinationEl = document.getElementById("evalHallucination");

const caseForm = document.getElementById("evalCaseForm");
const caseQuestionInput = document.getElementById("evalCaseQuestion");
const caseExpectedInput = document.getElementById("evalCaseExpected");
const caseGroundTruthDocInput = document.getElementById("evalCaseGroundTruthDoc");
const caseGroundTruthPageInput = document.getElementById("evalCaseGroundTruthPage");
const caseListEl = document.getElementById("evalCaseList");

const runForm = document.getElementById("evalRunForm");
const runQuestionInput = document.getElementById("evalRunQuestion");
const runJudgeInput = document.getElementById("evalRunJudge");
const runBtn = document.getElementById("evalRunBtn");
const runResultEl = document.getElementById("evalRunResult");

const historyListEl = document.getElementById("evalHistoryList");

let loaded = false;

function formatLatency(ms) {
  if (!ms) return "0 ms";
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
}

function formatPct(value) {
  return value === null || value === undefined ? "–" : `${Math.round(value * 100)}%`;
}

function formatCost(value) {
  if (value === null || value === undefined) return "–";
  return value === 0 ? "$0 (local)" : `$${value.toFixed(5)}`;
}

async function loadSummary() {
  try {
    const data = await api.get(`/evaluate/summary?provider=${encodeURIComponent(currentProvider())}`);
    const s = data.summary;
    totalRunsEl.textContent = String(s.total_runs);
    passRateEl.textContent = s.pass_rate_pct === null ? "–" : `${s.pass_rate_pct}%`;
    avgLatencyEl.textContent = formatLatency(s.avg_latency_ms);
    avgCostEl.textContent = formatCost(s.avg_cost_usd);
    errorRateEl.textContent = s.error_rate_pct === null ? "–" : `${s.error_rate_pct}%`;

    recallEl.textContent = formatPct(s.avg_recall_at_k);
    precisionEl.textContent = formatPct(s.avg_precision_at_k);
    mrrEl.textContent = s.avg_mrr === null || s.avg_mrr === undefined ? "–" : s.avg_mrr.toFixed(2);

    faithfulnessEl.textContent = formatPct(s.avg_faithfulness);
    relevanceEl.textContent = formatPct(s.avg_relevance);
    groundednessEl.textContent = formatPct(s.avg_groundedness);

    correctnessEl.textContent = formatPct(s.avg_correctness);
    citationAccuracyEl.textContent = formatPct(s.avg_citation_accuracy);
    hallucinationEl.textContent = formatPct(s.avg_hallucination_rate);
  } catch {
    /* summary is a nice-to-have; ignore failures silently */
  }
}

function renderCases(cases) {
  caseListEl.replaceChildren();
  for (const c of cases) {
    const li = document.createElement("li");
    li.className = "eval-case-item";

    const info = document.createElement("div");
    const q = document.createElement("div");
    q.className = "eval-case-question";
    q.textContent = c.question;
    info.appendChild(q);
    if (c.expected_answer) {
      const exp = document.createElement("div");
      exp.className = "eval-case-expected";
      exp.textContent = `Expected: ${c.expected_answer}`;
      info.appendChild(exp);
    }
    if (c.ground_truth && c.ground_truth.length) {
      const gt = document.createElement("div");
      gt.className = "eval-case-expected";
      gt.textContent = `Ground truth: ${c.ground_truth
        .map((g) => g.chunk_id || `${g.document}${g.page ? " p." + g.page : ""}`)
        .join(", ")}`;
      info.appendChild(gt);
    }

    const actions = document.createElement("div");
    actions.className = "eval-case-actions";

    const runOneBtn = document.createElement("button");
    runOneBtn.className = "btn btn-secondary btn-sm";
    runOneBtn.type = "button";
    runOneBtn.textContent = "Run";
    runOneBtn.addEventListener("click", () => runEvaluation(c.question, c));

    const delBtn = document.createElement("button");
    delBtn.className = "btn btn-ghost btn-sm";
    delBtn.type = "button";
    delBtn.textContent = "Delete";
    delBtn.addEventListener("click", async () => {
      try {
        await api.del(`/evaluate/cases/${c.id}`);
        await loadCases();
        showToast("Test case deleted", "success");
      } catch (err) {
        showToast(err.message || "Failed to delete test case", "error");
      }
    });

    actions.appendChild(runOneBtn);
    actions.appendChild(delBtn);
    li.appendChild(info);
    li.appendChild(actions);
    caseListEl.appendChild(li);
  }
}

async function loadCases() {
  try {
    const data = await api.get(`/evaluate/cases?provider=${encodeURIComponent(currentProvider())}`);
    renderCases(data.cases || []);
  } catch (err) {
    showToast(err.message || "Failed to load test cases", "error");
  }
}

function ratingBadge(rating) {
  const span = document.createElement("span");
  span.className = `rating-badge ${rating || "pending"}`;
  span.textContent = rating === "pass" ? "Pass" : rating === "fail" ? "Fail" : "Ungraded";
  return span;
}

function renderHistory(history) {
  historyListEl.replaceChildren();
  if (!history.length) {
    const empty = document.createElement("p");
    empty.className = "card-hint";
    empty.textContent = "No runs yet. Save a test case or run an ad-hoc question above.";
    historyListEl.appendChild(empty);
    return;
  }

  for (const run of history) {
    const item = document.createElement("div");
    item.className = "eval-history-item";
    if (run.error) item.classList.add("has-error");

    const top = document.createElement("div");
    top.className = "eval-history-top";
    const q = document.createElement("span");
    q.className = "eval-history-question";
    q.textContent = run.question;
    const costPart = typeof run.cost_usd === "number" ? ` · ${formatCost(run.cost_usd)}` : "";
    const meta = document.createElement("span");
    meta.className = "eval-history-meta";
    meta.textContent = `${run.intent} · ${formatLatency(run.latency_ms)}${costPart} · ${new Date(run.created_at).toLocaleString()}`;
    top.appendChild(q);
    top.appendChild(meta);

    const answer = document.createElement("div");
    answer.className = "eval-history-answer";
    answer.textContent = run.error ? `Error: ${run.error}` : run.answer;

    item.appendChild(top);
    item.appendChild(answer);

    if (run.retrieval_metrics || run.judge_scores) {
      const metrics = document.createElement("div");
      metrics.className = "eval-history-metrics";
      if (run.retrieval_metrics) {
        const r = run.retrieval_metrics;
        metrics.appendChild(
          metricChip(`Recall@${r.k} ${formatPct(r.recall_at_k)} · Precision@${r.k} ${formatPct(r.precision_at_k)} · MRR ${r.mrr.toFixed(2)}`)
        );
      }
      if (run.judge_scores && !run.judge_scores.judge_error) {
        const j = run.judge_scores;
        metrics.appendChild(
          metricChip(
            `Faithfulness ${formatPct(j.faithfulness)} · Relevance ${formatPct(j.relevance)} · Groundedness ${formatPct(j.groundedness)}`
          )
        );
        metrics.appendChild(
          metricChip(
            `Correctness ${formatPct(j.correctness)} · Citation Acc. ${formatPct(j.citation_accuracy)} · Hallucination ${formatPct(j.hallucination_rate)}`
          )
        );
      }
      item.appendChild(metrics);
    }

    const actions = document.createElement("div");
    actions.className = "eval-history-actions";
    actions.appendChild(ratingBadge(run.rating));

    const passBtn = document.createElement("button");
    passBtn.className = "btn btn-secondary btn-sm";
    passBtn.type = "button";
    passBtn.textContent = "Mark pass";
    passBtn.addEventListener("click", () => rate(run.id, "pass"));

    const failBtn = document.createElement("button");
    failBtn.className = "btn btn-danger btn-sm";
    failBtn.type = "button";
    failBtn.textContent = "Mark fail";
    failBtn.addEventListener("click", () => rate(run.id, "fail"));

    actions.appendChild(passBtn);
    actions.appendChild(failBtn);
    item.appendChild(actions);
    historyListEl.appendChild(item);
  }
}

function metricChip(text) {
  const span = document.createElement("span");
  span.className = "eval-metric-chip";
  span.textContent = text;
  return span;
}

async function loadHistory() {
  try {
    const data = await api.get(`/evaluate/history?provider=${encodeURIComponent(currentProvider())}`);
    renderHistory(data.history || []);
  } catch (err) {
    showToast(err.message || "Failed to load run history", "error");
  }
}

async function rate(runId, rating) {
  try {
    await api.post(`/evaluate/history/${runId}/rate`, { rating });
    await Promise.all([loadHistory(), loadSummary()]);
  } catch (err) {
    showToast(err.message || "Failed to save rating", "error");
  }
}

async function runEvaluation(question, testCase = null) {
  runBtn.disabled = true;
  runBtn.innerHTML = '<span class="spinner"></span> Running...';
  runResultEl.classList.add("hidden");

  try {
    const payload = {
      provider: currentProvider(),
      question,
      case_id: testCase?.id || null,
      expected_answer: testCase?.expected_answer || null,
      source_filter: testCase?.source_filter || null,
      ground_truth: testCase?.ground_truth || null,
      run_judge: runJudgeInput.checked,
    };
    const data = await api.post("/evaluate/run", payload);
    runResultEl.textContent = data.run.answer;
    runResultEl.classList.remove("hidden");
    showToast("Evaluation run complete", "success");
    await Promise.all([loadHistory(), loadSummary()]);
  } catch (err) {
    showToast(err.message || "Evaluation run failed", "error");
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "Run evaluation";
  }
}

export function initEvaluation() {
  caseForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = caseQuestionInput.value.trim();
    if (!question) return;
    const gtDoc = caseGroundTruthDocInput.value.trim();
    const gtPage = caseGroundTruthPageInput.value.trim();
    const groundTruth = gtDoc
      ? [{ document: gtDoc, page: gtPage ? Number(gtPage) : null }]
      : null;
    try {
      await api.post("/evaluate/cases", {
        provider: currentProvider(),
        question,
        expected_answer: caseExpectedInput.value.trim() || null,
        ground_truth: groundTruth,
      });
      caseQuestionInput.value = "";
      caseExpectedInput.value = "";
      caseGroundTruthDocInput.value = "";
      caseGroundTruthPageInput.value = "";
      await loadCases();
      showToast("Test case saved", "success");
    } catch (err) {
      showToast(err.message || "Failed to save test case", "error");
    }
  });

  runForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = runQuestionInput.value.trim();
    if (!question) return;
    await runEvaluation(question);
  });

  onProviderChange(() => {
    if (loaded) {
      loadCases();
      loadHistory();
      loadSummary();
    }
  });
}

export function activateEvaluation() {
  loaded = true;
  loadCases();
  loadHistory();
  loadSummary();
}
