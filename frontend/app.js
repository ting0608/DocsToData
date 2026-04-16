const messagesEl = document.getElementById("messages");
const providerEl = document.getElementById("provider");
const uploadBtn = document.getElementById("uploadBtn");
const fileInput = document.getElementById("fileInput");
const questionInput = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");
const summarizeBtn = document.getElementById("summarizeBtn");
const reportOverlay = document.getElementById("reportOverlay");
const reportContent = document.getElementById("reportContent");
const closeReportBtn = document.getElementById("closeReportBtn");
const exportReportBtn = document.getElementById("exportReportBtn");
const exportFormatEl = document.getElementById("exportFormat");

let latestReportText = "";
const MAX_CHAT_MESSAGES = 80;

function getProviderAvatar(provider) {
  return provider === "openai" ? "/frontend/assets/openAI-icon.png" : "/frontend/assets/ollama-icon.png";
}

function trimMessagesIfNeeded() {
  while (messagesEl.children.length > MAX_CHAT_MESSAGES) {
    messagesEl.removeChild(messagesEl.firstElementChild);
  }
}

function addMessage(type, text) {
  if (type === "bot") {
    const row = document.createElement("div");
    row.className = "bot-row";

    const avatar = document.createElement("img");
    avatar.className = "bot-avatar";
    avatar.src = getProviderAvatar(providerEl.value);
    avatar.alt = `${providerEl.value} avatar`;

    const bubble = document.createElement("div");
    bubble.className = "message bot";
    bubble.textContent = text;

    row.appendChild(avatar);
    row.appendChild(bubble);
    messagesEl.appendChild(row);
    trimMessagesIfNeeded();
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return;
  }

  const div = document.createElement("div");
  div.className = `message ${type}`;
  div.textContent = text;
  messagesEl.appendChild(div);
  trimMessagesIfNeeded();
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addTypingMessage() {
  const row = document.createElement("div");
  row.className = "bot-row";

  const avatar = document.createElement("img");
  avatar.className = "bot-avatar";
  avatar.src = getProviderAvatar(providerEl.value);
  avatar.alt = `${providerEl.value} avatar`;

  const bubble = document.createElement("div");
  bubble.className = "message bot";
  bubble.innerHTML = `
    <span class="typing" aria-label="AI is typing">
      <span class="dot"></span>
      <span class="dot"></span>
      <span class="dot"></span>
    </span>
  `;
  row.appendChild(avatar);
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  trimMessagesIfNeeded();
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return row;
}

async function ingestFile(file) {
  const form = new FormData();
  form.append("provider", providerEl.value);
  form.append("file", file);

  addMessage("system", `Uploading and ingesting: ${file.name}`);

  const res = await fetch("/ingest-upload", {
    method: "POST",
    body: form,
  });
  const data = await res.json();

  if (!res.ok) {
    throw new Error(data.detail || "Ingest failed");
  }

  const ingest = data.ingest;
  addMessage(
    "system",
    `Ingest done (${data.provider}). Pages: ${ingest.pages}, Chunks: ${ingest.chunks}, Vectors: ${ingest.vectors}`
  );
}

async function fetchAnswer(question) {
  const res = await fetch("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      provider: providerEl.value,
      question,
    }),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || "Query failed");
  }
  return data.answer || "(no answer)";
}

async function askQuestion(question) {
  addMessage("user", question);
  const typingEl = addTypingMessage();

  try {
    const answer = await fetchAnswer(question);
    addMessage("bot", answer);
  } finally {
    typingEl.remove();
  }
}

function buildReportText(items) {
  const lines = [
    "DocsToData Summary Report",
    `Generated at: ${new Date().toLocaleString()}`,
    `Provider: ${providerEl.value}`,
    "",
    `1) Budget / Cost`,
    items.budget,
    "",
    `2) Timeline (External + Internal Days)`,
    items.timeline,
    "",
    `3) Included Features`,
    items.features,
    "",
    `4) Requestor`,
    items.requestor,
    "",
    "Note: This report is generated from current indexed document context.",
  ];
  return lines.join("\n");
}

function showReport(text) {
  latestReportText = text;
  reportContent.textContent = text;
  reportOverlay.classList.remove("hidden");
}

function hideReport() {
  reportOverlay.classList.add("hidden");
}

function downloadBlob(content, mimeType, extension) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `summary-report-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.${extension}`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function downloadPngFromText(text) {
  const lines = text.split("\n");
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  const padding = 32;
  const lineHeight = 24;
  const width = 1200;
  const height = padding * 2 + lines.length * lineHeight;
  canvas.width = width;
  canvas.height = Math.max(height, 600);

  ctx.fillStyle = "#EEEEEE";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#1F6F5F";
  ctx.font = "20px Menlo, Monaco, Consolas, monospace";

  let y = padding + 10;
  for (const line of lines) {
    ctx.fillText(line, padding, y);
    y += lineHeight;
  }

  canvas.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `summary-report-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }, "image/png");
}

function downloadPdfFromText(text) {
  const jsPdf = window.jspdf?.jsPDF;
  if (!jsPdf) {
    throw new Error("PDF library not loaded. Please refresh and try again.");
  }
  const doc = new jsPdf({ unit: "pt", format: "a4" });
  const margin = 40;
  const maxWidth = 515;
  const lines = doc.splitTextToSize(text, maxWidth);
  doc.setFont("courier", "normal");
  doc.setFontSize(11);
  doc.text(lines, margin, margin);
  doc.save(`summary-report-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.pdf`);
}

async function runSummarizeReport() {
  addMessage("system", "Generating summary report...");
  const typingEl = addTypingMessage();

  summarizeBtn.disabled = true;
  sendBtn.disabled = true;
  uploadBtn.disabled = true;
  questionInput.disabled = true;

  try {
    const budgetQ = "What is the total budget/cost of this project? Include currency if available.";
    const timelineQ =
      "How many days are needed externally and internally? If one side is missing, say not explicitly stated.";
    const featuresQ = "What key features/scope are included in this project?";
    const requestorQ = "Who is the requestor/client for this project?";

    const [budget, timeline, features, requestor] = await Promise.all([
      fetchAnswer(budgetQ),
      fetchAnswer(timelineQ),
      fetchAnswer(featuresQ),
      fetchAnswer(requestorQ),
    ]);

    const report = buildReportText({ budget, timeline, features, requestor });
    showReport(report);
    addMessage("system", "Summary report ready. You can export it.");
  } finally {
    typingEl.remove();
    summarizeBtn.disabled = false;
    sendBtn.disabled = false;
    uploadBtn.disabled = false;
    questionInput.disabled = false;
  }
}

uploadBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", async () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  try {
    await ingestFile(file);
  } catch (err) {
    addMessage("system", `Error: ${err.message}`);
  } finally {
    fileInput.value = "";
  }
});

sendBtn.addEventListener("click", async () => {
  const question = questionInput.value.trim();
  if (!question) return;
  questionInput.value = "";
  try {
    await askQuestion(question);
  } catch (err) {
    addMessage("system", `Error: ${err.message}`);
  }
});

questionInput.addEventListener("keydown", async (e) => {
  if (e.key !== "Enter") return;
  e.preventDefault();
  sendBtn.click();
});

summarizeBtn.addEventListener("click", async () => {
  try {
    await runSummarizeReport();
  } catch (err) {
    addMessage("system", `Error: ${err.message}`);
  }
});

closeReportBtn.addEventListener("click", hideReport);

reportOverlay.addEventListener("click", (e) => {
  if (e.target === reportOverlay) hideReport();
});

exportReportBtn.addEventListener("click", () => {
  if (!latestReportText) return;
  const format = exportFormatEl.value || "txt";
  if (format === "png") {
    downloadPngFromText(latestReportText);
    return;
  }
  if (format === "pdf") {
    downloadPdfFromText(latestReportText);
    return;
  }
  downloadBlob(latestReportText, "text/plain;charset=utf-8", "txt");
});

addMessage("system", "Ready. Pick provider, upload a PDF with +, then ask.");
