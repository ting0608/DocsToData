// Evaluation Dashboard view: save test questions, run them against the live
// RAG pipeline, grade the answers pass/fail, and track aggregate metrics.

import { api } from "./api.js";
import { blockIfGuest, getGuestData, isGuest } from "./guest.js";
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
const pagePrevBtn = document.getElementById("evalPagePrev");
const pageNextBtn = document.getElementById("evalPageNext");
const pageStatusEl = document.getElementById("evalPageStatus");

const HISTORY_PAGE_SIZE = 5;

const viewToggleEl = document.getElementById("evalViewToggle");
const scoreGridCardEl = document.getElementById("evalScoreGridCard");
const scoreGridGraphEl = document.getElementById("evalScoreGridGraph");
const chartCanvas = document.getElementById("evalScoreChart");
const chartEmptyHintEl = document.getElementById("evalChartEmptyHint");

const metricTooltipEl = document.getElementById("evalMetricTooltip");
const metricTooltipTitleEl = document.getElementById("evalMetricTooltipTitle");
const metricTooltipBodyEl = document.getElementById("evalMetricTooltipBody");

// How each metric is calculated, keyed by the card's data-metric attribute.
// Kept to 3-5 short lines each; wording mirrors rag/evaluation.py so the
// explanations stay accurate to the actual computation.
const METRIC_EXPLANATIONS = {
  recall: {
    title: "Recall@K",
    body: "Of all the ground-truth passages for a question, the fraction that appear in the top-K retrieved citations. Computed as matched ground-truth items ÷ total ground-truth items. Needs a test case with ground truth. Higher is better (1.0 = every expected passage was retrieved).",
  },
  precision: {
    title: "Precision@K",
    body: "Of the top-K citations actually retrieved, the fraction that match a ground-truth passage. Computed as matched items ÷ retrieved items. Measures how much of what you retrieved was relevant. Higher is better (1.0 = no irrelevant passages retrieved).",
  },
  mrr: {
    title: "MRR (Mean Reciprocal Rank)",
    body: "1 divided by the rank of the first correct citation (1st place = 1.0, 2nd = 0.5, 3rd = 0.33, …). Rewards putting a relevant passage near the top. 0 if no correct passage is retrieved. Averaged across runs.",
  },
  faithfulness: {
    title: "Faithfulness",
    body: "An LLM judge scores 0-1 whether the answer only states things supported by the retrieved context (no unsupported additions). Requires judge scoring enabled on the run. Higher is better.",
  },
  relevance: {
    title: "Relevance",
    body: "An LLM judge scores 0-1 whether the answer actually addresses the question that was asked, rather than drifting off-topic. Requires judge scoring enabled. Higher is better.",
  },
  groundedness: {
    title: "Groundedness",
    body: "An LLM judge scores 0-1 whether every specific claim in the answer is traceable to a specific citation. Requires judge scoring enabled. Higher is better.",
  },
  correctness: {
    title: "Correctness",
    body: "An LLM judge scores 0-1 how well the answer matches the human-written expected answer. Only computed when a test case includes an expected answer; otherwise null (shown as –). Higher is better.",
  },
  citation_accuracy: {
    title: "Citation Accuracy",
    body: "An LLM judge scores 0-1 whether the cited passages actually contain the information used to answer. Catches citations that look relevant but don't support the claim. Requires judge scoring enabled. Higher is better.",
  },
  hallucination: {
    title: "Hallucination Rate",
    body: "An LLM judge estimates the fraction of the answer that is fabricated or unsupported by the context/citations. Unlike the others, LOWER is better (0.0 = nothing fabricated, 1.0 = fully hallucinated).",
  },
  pass_rate: {
    title: "Pass rate (human)",
    body: "The percentage of graded runs a human marked as Pass (Pass ÷ graded runs). Ungraded runs are excluded. This is your manual quality signal, independent of the LLM judge. Higher is better.",
  },
  total_runs: {
    title: "Total runs",
    body: "The number of evaluation runs recorded for the selected provider, including failed runs. Every question you run (ad-hoc or from a saved case) adds one run to this count.",
  },
  avg_latency: {
    title: "Avg latency",
    body: "The mean wall-clock time for the pipeline to answer a question, measured server-side around the answer call and averaged across all runs. Lower is better.",
  },
  avg_cost: {
    title: "Avg cost / query",
    body: "The mean estimated USD cost per run, from token usage priced via rag/pricing.py. Local (Ollama) runs are $0. Runs with no known pricing are skipped in the average. Lower is better.",
  },
  error_rate: {
    title: "Error rate",
    body: "The percentage of runs that failed with an error (error runs ÷ total runs). A pipeline failure is still recorded as a run so it counts here rather than being silently dropped. Lower is better.",
  },
};

function positionMetricTooltip(cardEl) {
  const rect = cardEl.getBoundingClientRect();
  // Show first (still hidden via visibility) to measure, then place below the
  // card, clamped to the viewport so it never runs off-screen.
  metricTooltipEl.classList.remove("hidden");
  const ttRect = metricTooltipEl.getBoundingClientRect();
  const margin = 8;
  let left = rect.left;
  if (left + ttRect.width > window.innerWidth - margin) {
    left = window.innerWidth - margin - ttRect.width;
  }
  left = Math.max(margin, left);

  let top = rect.bottom + 8;
  if (top + ttRect.height > window.innerHeight - margin) {
    // Not enough room below; place above the card instead.
    top = rect.top - ttRect.height - 8;
  }
  metricTooltipEl.style.left = `${left}px`;
  metricTooltipEl.style.top = `${top}px`;
}

function showMetricTooltip(cardEl) {
  const key = cardEl.dataset.metric;
  const info = METRIC_EXPLANATIONS[key];
  if (!info) return;
  metricTooltipTitleEl.textContent = info.title;
  metricTooltipBodyEl.textContent = info.body;
  positionMetricTooltip(cardEl);
}

function hideMetricTooltip() {
  metricTooltipEl.classList.add("hidden");
}

function initMetricTooltips() {
  const cards = document.querySelectorAll("#view-evaluation .summary-card[data-metric]");
  for (const card of cards) {
    card.addEventListener("mouseenter", () => showMetricTooltip(card));
    card.addEventListener("mouseleave", hideMetricTooltip);
    // Keyboard/focus accessibility: tabbing to a card reveals its explanation.
    card.setAttribute("tabindex", "0");
    card.addEventListener("focus", () => showMetricTooltip(card));
    card.addEventListener("blur", hideMetricTooltip);
  }
}

let loaded = false;
let evalView = "card"; // "card" | "graph"
let scoreChart = null;
let latestSummary = null;
let fullHistory = [];
let historyPage = 1;

// Category colors, shared between the card-view left border accents and the
// graph-view bar colors, so the legend above the grid means the same thing
// in either view.
const METRIC_META = [
  { key: "avg_recall_at_k", label: "Recall@K", category: "retrieval", color: "#2fa084" },
  { key: "avg_precision_at_k", label: "Precision@K", category: "retrieval", color: "#2fa084" },
  { key: "avg_mrr", label: "MRR", category: "retrieval", color: "#2fa084", isRatio: true },
  { key: "avg_faithfulness", label: "Faithfulness", category: "generation", color: "#4c8bd9" },
  { key: "avg_relevance", label: "Relevance", category: "generation", color: "#4c8bd9" },
  { key: "avg_groundedness", label: "Groundedness", category: "generation", color: "#4c8bd9" },
  { key: "avg_correctness", label: "Correctness", category: "quality", color: "#d19a2f" },
  { key: "avg_citation_accuracy", label: "Citation Acc.", category: "quality", color: "#d19a2f" },
  { key: "avg_hallucination_rate", label: "Hallucination", category: "quality", color: "#d19a2f" },
  { key: "pass_rate_pct", label: "Pass rate (human)", category: "quality", color: "#d19a2f", isPercentAlready: true },
];

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
    const data = isGuest()
      ? { summary: getGuestData()?.evaluation?.summary }
      : await api.get(`/evaluate/summary?provider=${encodeURIComponent(currentProvider())}`);
    const s = data.summary;
    latestSummary = s;

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

    // Only touch the chart while its container is actually visible — Chart.js
    // measures the canvas at creation/update time, and a `display: none`
    // ancestor (the card view being active) reports zero size, leaving the
    // chart blank even after switching to graph view later.
    if (evalView === "graph") renderScoreChart();
  } catch {
    /* summary is a nice-to-have; ignore failures silently */
  }
}

function metricValueAsRatio(meta, summary) {
  const raw = summary[meta.key];
  if (raw === null || raw === undefined) return null;
  if (meta.isPercentAlready) return raw / 100;
  return raw; // avg_mrr and the *_at_k / judge scores are already 0..1
}

function renderScoreChart() {
  if (!chartCanvas || typeof window.Chart === "undefined") return;
  if (!latestSummary) return;

  const points = METRIC_META.map((meta) => ({
    meta,
    ratio: metricValueAsRatio(meta, latestSummary),
  }));
  const hasAnyData = points.some((p) => p.ratio !== null);
  chartEmptyHintEl.classList.toggle("hidden", hasAnyData);
  chartCanvas.classList.toggle("hidden", !hasAnyData);
  if (!hasAnyData) {
    if (scoreChart) {
      scoreChart.destroy();
      scoreChart = null;
    }
    return;
  }

  const labels = points.map((p) => p.meta.label);
  const values = points.map((p) => (p.ratio === null ? 0 : Math.round(p.ratio * 100)));
  const colors = points.map((p) => p.meta.color);

  if (scoreChart) {
    scoreChart.data.labels = labels;
    scoreChart.data.datasets[0].data = values;
    scoreChart.data.datasets[0].backgroundColor = colors;
    scoreChart.update();
    return;
  }

  scoreChart = new window.Chart(chartCanvas, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Score (%)",
          data: values,
          backgroundColor: colors,
          borderRadius: 6,
          maxBarThickness: 36,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.formattedValue}%`,
          },
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          ticks: { callback: (v) => `${v}%` },
        },
      },
    },
  });
}

function setEvalView(view) {
  evalView = view;
  for (const btn of viewToggleEl.querySelectorAll(".view-toggle-btn")) {
    btn.classList.toggle("active", btn.dataset.evalView === view);
  }
  scoreGridCardEl.classList.toggle("hidden", view !== "card");
  scoreGridGraphEl.classList.toggle("hidden", view !== "graph");
  if (view === "graph") renderScoreChart();
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
      if (blockIfGuest("Sign in to manage test cases. You're viewing sample data as a guest.")) return;
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
    const data = isGuest()
      ? { cases: getGuestData()?.evaluation?.cases || [] }
      : await api.get(`/evaluate/cases?provider=${encodeURIComponent(currentProvider())}`);
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

function totalHistoryPages() {
  return Math.max(1, Math.ceil(fullHistory.length / HISTORY_PAGE_SIZE));
}

function renderHistoryPage() {
  const totalPages = totalHistoryPages();
  historyPage = Math.min(Math.max(1, historyPage), totalPages);

  const start = (historyPage - 1) * HISTORY_PAGE_SIZE;
  const pageItems = fullHistory.slice(start, start + HISTORY_PAGE_SIZE);
  renderHistory(pageItems);

  pageStatusEl.textContent = `Page ${historyPage} of ${totalPages} (${fullHistory.length} run${fullHistory.length === 1 ? "" : "s"})`;
  pagePrevBtn.disabled = historyPage <= 1;
  pageNextBtn.disabled = historyPage >= totalPages;
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

async function loadHistory({ resetPage = false } = {}) {
  try {
    const data = isGuest()
      ? { history: getGuestData()?.evaluation?.history || [] }
      : await api.get(`/evaluate/history?provider=${encodeURIComponent(currentProvider())}`);
    fullHistory = data.history || [];
    if (resetPage) historyPage = 1;
    renderHistoryPage();
  } catch (err) {
    showToast(err.message || "Failed to load run history", "error");
  }
}

async function rate(runId, rating) {
  if (blockIfGuest("Sign in to grade runs. You're viewing sample data as a guest.")) return;
  try {
    await api.post(`/evaluate/history/${runId}/rate`, { rating });
    await Promise.all([loadHistory({ resetPage: false }), loadSummary()]);
  } catch (err) {
    showToast(err.message || "Failed to save rating", "error");
  }
}

async function runEvaluation(question, testCase = null) {
  if (blockIfGuest("Sign in to run evaluations. You're viewing sample data as a guest.")) return;
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
    // Jump back to page 1 so the just-completed run (newest-first) is visible.
    await Promise.all([loadHistory({ resetPage: true }), loadSummary()]);
  } catch (err) {
    showToast(err.message || "Evaluation run failed", "error");
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "Run evaluation";
  }
}

export function initEvaluation() {
  initMetricTooltips();

  viewToggleEl.addEventListener("click", (e) => {
    const btn = e.target.closest(".view-toggle-btn");
    if (!btn) return;
    setEvalView(btn.dataset.evalView);
  });

  caseForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (blockIfGuest("Sign in to save test cases. You're viewing sample data as a guest.")) return;
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

  pagePrevBtn.addEventListener("click", () => {
    historyPage -= 1;
    renderHistoryPage();
  });

  pageNextBtn.addEventListener("click", () => {
    historyPage += 1;
    renderHistoryPage();
  });

  onProviderChange(() => {
    if (loaded) {
      loadCases();
      loadHistory({ resetPage: true });
      loadSummary();
    }
  });
}

export function activateEvaluation() {
  loaded = true;
  loadCases();
  loadHistory({ resetPage: true });
  loadSummary();
}

/**
 * Wipe all rendered evaluation data + cached state so a signed-out (or
 * newly signed-in) user never sees the previous session's dashboard.
 * Called from the sign-out handler in authView.js.
 */
export function clearEvaluation() {
  loaded = false;
  latestSummary = null;
  fullHistory = [];
  historyPage = 1;

  caseListEl.replaceChildren();
  historyListEl.replaceChildren();
  runResultEl.classList.add("hidden");
  runResultEl.textContent = "";
  if (pageStatusEl) pageStatusEl.textContent = "";

  // Reset all summary metric tiles to their empty placeholder.
  const dash = "–";
  [
    totalRunsEl, passRateEl, avgLatencyEl, avgCostEl, errorRateEl,
    recallEl, precisionEl, mrrEl, faithfulnessEl, relevanceEl,
    groundednessEl, correctnessEl, citationAccuracyEl, hallucinationEl,
  ].forEach((el) => { if (el) el.textContent = dash; });

  if (scoreChart) {
    scoreChart.destroy();
    scoreChart = null;
  }
}
