const messagesEl = document.getElementById("messages");
const providerEl = document.getElementById("provider");
const uploadBtn = document.getElementById("uploadBtn");
const fileInput = document.getElementById("fileInput");
const questionInput = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");
const summarizeBtn = document.getElementById("summarizeBtn");
const documentsBtn = document.getElementById("documentsBtn");
const documentsOverlay = document.getElementById("documentsOverlay");
const documentsList = document.getElementById("documentsList");
const closeDocumentsBtn = document.getElementById("closeDocumentsBtn");
const selectAllDocumentsBtn = document.getElementById("selectAllDocumentsBtn");
const clearDocumentsBtn = document.getElementById("clearDocumentsBtn");
const reportOverlay = document.getElementById("reportOverlay");
const reportContent = document.getElementById("reportContent");
const closeReportBtn = document.getElementById("closeReportBtn");
const exportReportBtn = document.getElementById("exportReportBtn");
const exportFormatEl = document.getElementById("exportFormat");

let latestReportText = "";
const MAX_CHAT_MESSAGES = 80;
const allDocuments = [];
const selectedSources = new Set();
const knownDocumentsByProvider = { openai: [], ollama: [] };
const selectedSourcesByProvider = { openai: new Set(), ollama: new Set() };

function currentProvider() {
  return providerEl.value;
}

function getKnownDocuments() {
  return knownDocumentsByProvider[currentProvider()] || [];
}

function getSelectedSourcesSet() {
  if (!selectedSourcesByProvider[currentProvider()]) {
    selectedSourcesByProvider[currentProvider()] = new Set();
  }
  return selectedSourcesByProvider[currentProvider()];
}

function syncSelectionState() {
  allDocuments.length = 0;
  allDocuments.push(...getKnownDocuments());
  selectedSources.clear();
  for (const name of getSelectedSourcesSet()) {
    selectedSources.add(name);
  }
}

function updateDocumentsBtnLabel() {
  const total = allDocuments.length;
  const selected = selectedSources.size;
  if (!total) {
    documentsBtn.classList.add("hidden");
    documentsBtn.textContent = "PDFs (0)";
    return;
  }

  documentsBtn.classList.remove("hidden");
  if (selected === total) {
    documentsBtn.textContent = `PDFs (${total})`;
  } else {
    documentsBtn.textContent = `PDFs (${selected}/${total})`;
  }
}

function getSelectedSources() {
  return allDocuments.filter((name) => selectedSources.has(name));
}

function getSourceFilterPayload() {
  const selected = getSelectedSources();
  if (!selected.length || selected.length === allDocuments.length) {
    return null;
  }
  return selected;
}

function ensureDocumentsSelected(actionLabel) {
  if (!allDocuments.length) {
    addMessage("system", "No indexed PDFs yet. Upload at least one PDF first.");
    return false;
  }
  if (!selectedSources.size) {
    addMessage("system", `Select at least one PDF in the PDFs list before you ${actionLabel}.`);
    return false;
  }
  return true;
}

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
  bubble.className = "message bot typing-bubble";
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

function showDocumentsModal() {
  documentsOverlay.classList.remove("hidden");
}

function hideDocumentsModal() {
  documentsOverlay.classList.add("hidden");
}

function renderDocumentsList(docs) {
  documentsList.replaceChildren();
  for (const name of docs) {
    const item = document.createElement("li");
    item.className = "documents-item";

    const label = document.createElement("label");
    label.className = "documents-item-label";
    if (selectedSources.has(name)) {
      label.classList.add("is-selected");
    }

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.className = "documents-checkbox";
    checkbox.checked = selectedSources.has(name);

    const checkmark = document.createElement("span");
    checkmark.className = "documents-checkmark";
    checkmark.setAttribute("aria-hidden", "true");

    const text = document.createElement("span");
    text.className = "documents-name";
    text.textContent = name;

    checkbox.addEventListener("change", () => {
      const selected = getSelectedSourcesSet();
      if (checkbox.checked) {
        selected.add(name);
        selectedSources.add(name);
        label.classList.add("is-selected");
      } else {
        selected.delete(name);
        selectedSources.delete(name);
        label.classList.remove("is-selected");
      }
      updateDocumentsBtnLabel();
    });

    label.appendChild(checkbox);
    label.appendChild(checkmark);
    label.appendChild(text);
    item.appendChild(label);
    documentsList.appendChild(item);
  }
}

function setAllDocumentSelection(checked) {
  const selected = getSelectedSourcesSet();
  selectedSources.clear();
  selected.clear();
  if (checked) {
    for (const name of allDocuments) {
      selected.add(name);
      selectedSources.add(name);
    }
  }
  renderDocumentsList(allDocuments);
  updateDocumentsBtnLabel();
}

async function refreshDocuments() {
  const res = await fetch(`/documents?provider=${encodeURIComponent(providerEl.value)}`);
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || "Failed to load documents");
  }

  const docs = data.documents || [];
  const provider = currentProvider();
  const prevKnown = knownDocumentsByProvider[provider] || [];
  const selected = getSelectedSourcesSet();

  if (!prevKnown.length) {
    selected.clear();
    for (const name of docs) {
      selected.add(name);
    }
  } else {
    for (const name of [...selected]) {
      if (!docs.includes(name)) {
        selected.delete(name);
      }
    }
    for (const name of docs) {
      if (!prevKnown.includes(name)) {
        selected.add(name);
      }
    }
  }

  knownDocumentsByProvider[provider] = docs;
  syncSelectionState();

  if (!docs.length) {
    updateDocumentsBtnLabel();
    renderDocumentsList([]);
    return;
  }

  updateDocumentsBtnLabel();
  renderDocumentsList(docs);
}

async function ingestFiles(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) return;

  const form = new FormData();
  form.append("provider", providerEl.value);
  for (const file of files) {
    form.append("files", file);
  }

  addMessage("system", `Uploading and ingesting ${files.length} PDF(s)...`);

  const res = await fetch("/ingest-upload", {
    method: "POST",
    body: form,
  });
  const data = await res.json();

  if (!res.ok) {
    throw new Error(data.detail || "Ingest failed");
  }

  const ingested = Array.isArray(data.ingest) ? data.ingest : [data.ingest];
  for (const item of ingested) {
    addMessage(
      "system",
      `Ingested ${item.source}: pages ${item.pages}, chunks ${item.chunks}, vectors ${item.vectors}`
    );
  }
  addMessage("system", `Total indexed documents: ${data.total_documents ?? ingested.length}`);
  await refreshDocuments();
}

async function fetchAnswer(question, intent = null) {
  const payload = {
    provider: providerEl.value,
    question,
  };
  if (intent) {
    payload.intent = intent;
  }
  const sourceFilter = getSourceFilterPayload();
  if (sourceFilter) {
    payload.source_filter = sourceFilter;
  }

  const res = await fetch("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || "Query failed");
  }
  return data;
}

async function askQuestion(question) {
  addMessage("user", question);
  const typingEl = addTypingMessage();

  try {
    const data = await fetchAnswer(question);
    const intentLabel = data.intent ? ` [${data.intent}]` : "";
    addMessage("bot", `${data.answer || "(no answer)"}${intentLabel}`);
  } finally {
    typingEl.remove();
  }
}

function buildReportText(summaryText, intent) {
  const selected = getSelectedSources();
  const focusLine =
    selected.length && selected.length < allDocuments.length
      ? `Focused on: ${selected.join(", ")}`
      : "Focused on: all indexed PDFs";

  const lines = [
    "DocsToData Summary Report",
    `Generated at: ${new Date().toLocaleString()}`,
    `Provider: ${providerEl.value}`,
    `Intent: ${intent || "summarize"}`,
    focusLine,
    "",
    summaryText,
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
  if (!ensureDocumentsSelected("summarize")) return;

  addMessage("system", "Generating summary report...");
  const typingEl = addTypingMessage();

  summarizeBtn.disabled = true;
  sendBtn.disabled = true;
  uploadBtn.disabled = true;
  questionInput.disabled = true;

  try {
    const selected = getSelectedSources();
    const focusHint =
      selected.length < allDocuments.length
        ? ` Focus only on these selected documents: ${selected.join(", ")}.`
        : "";
    const question =
      `Summarize the indexed documents.${focusHint} Include budget/cost, timeline (external and internal days), key features/scope, and requestor/client when available.`;
    const data = await fetchAnswer(question, "summarize");
    const report = buildReportText(data.answer || "(no answer)", data.intent);
    showReport(report);
    addMessage("system", "Summary report ready. You can export it.");
    addMessage("bot", `${data.answer || "(no answer)"} [summarize]`);
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
  const files = fileInput.files;
  if (!files?.length) return;
  try {
    await ingestFiles(files);
  } catch (err) {
    addMessage("system", `Error: ${err.message}`);
  } finally {
    fileInput.value = "";
  }
});

sendBtn.addEventListener("click", async () => {
  const question = questionInput.value.trim();
  if (!question) return;
  if (!ensureDocumentsSelected("ask a question")) return;
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

providerEl.addEventListener("change", async () => {
  try {
    await refreshDocuments();
  } catch (err) {
    addMessage("system", `Error loading documents: ${err.message}`);
  }
});

documentsBtn.addEventListener("click", showDocumentsModal);
closeDocumentsBtn.addEventListener("click", hideDocumentsModal);
selectAllDocumentsBtn.addEventListener("click", () => setAllDocumentSelection(true));
clearDocumentsBtn.addEventListener("click", () => setAllDocumentSelection(false));
documentsOverlay.addEventListener("click", (e) => {
  if (e.target === documentsOverlay) hideDocumentsModal();
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

addMessage("system", "Ready. Pick provider, upload PDFs with +, then ask, summarize, or compare.");
refreshDocuments().catch(() => {});
