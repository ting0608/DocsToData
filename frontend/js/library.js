// Document Library view: browse indexed PDFs per provider with page/chunk
// counts, delete a document, and jump into the upload flow.

import { api } from "./api.js";
import { currentProvider, onProviderChange } from "./state.js";
import { showToast } from "./toast.js";

const grid = document.getElementById("libraryGrid");
const countEl = document.getElementById("libraryCount");
const emptyEl = document.getElementById("libraryEmpty");
const uploadBtn = document.getElementById("libraryUploadBtn");
const emptyUploadBtn = document.getElementById("libraryEmptyUploadBtn");
const fileInput = document.getElementById("fileInput");

let loaded = false;

function docIcon(name) {
  return (name || "?").slice(0, 2).toUpperCase();
}

function renderSkeleton() {
  grid.replaceChildren();
  for (let i = 0; i < 4; i++) {
    const card = document.createElement("div");
    card.className = "card library-doc-card";
    card.style.setProperty("--stagger", String(i));
    card.innerHTML = `
      <div class="skeleton" style="width:36px;height:36px;border-radius:10px;"></div>
      <div class="skeleton" style="width:80%;"></div>
      <div class="skeleton" style="width:50%;"></div>
    `;
    grid.appendChild(card);
  }
}

function renderDocuments(docs) {
  grid.replaceChildren();
  countEl.textContent = `${docs.length} document${docs.length === 1 ? "" : "s"}`;

  if (!docs.length) {
    emptyEl.classList.remove("hidden");
    return;
  }
  emptyEl.classList.add("hidden");

  docs.forEach((doc, i) => {
    const card = document.createElement("div");
    card.className = "card library-doc-card";
    card.style.setProperty("--stagger", String(i));

    const icon = document.createElement("div");
    icon.className = "library-doc-icon";
    icon.textContent = docIcon(doc.source);

    const name = document.createElement("div");
    name.className = "library-doc-name";
    name.textContent = doc.source;

    const stats = document.createElement("div");
    stats.className = "library-doc-stats";
    stats.innerHTML = `<span>${doc.pages} page${doc.pages === 1 ? "" : "s"}</span><span>${doc.chunks} chunk${doc.chunks === 1 ? "" : "s"}</span>`;

    const actions = document.createElement("div");
    actions.className = "library-doc-actions";
    const delBtn = document.createElement("button");
    delBtn.className = "btn btn-danger btn-sm";
    delBtn.type = "button";
    delBtn.textContent = "Delete";
    delBtn.addEventListener("click", () => deleteDocument(doc.source, card));
    actions.appendChild(delBtn);

    card.appendChild(icon);
    card.appendChild(name);
    card.appendChild(stats);
    card.appendChild(actions);
    grid.appendChild(card);
  });
}

async function deleteDocument(source, card) {
  if (!window.confirm(`Delete "${source}" from the index? This removes all its chunks.`)) {
    return;
  }
  card.classList.add("removing");
  try {
    await api.del(`/documents/${encodeURIComponent(currentProvider())}/${encodeURIComponent(source)}`);
    showToast(`Deleted "${source}"`, "success");
    await loadDocuments();
  } catch (err) {
    card.classList.remove("removing");
    showToast(err.message || "Failed to delete document", "error");
  }
}

export async function loadDocuments() {
  renderSkeleton();
  try {
    const data = await api.get(`/documents/details?provider=${encodeURIComponent(currentProvider())}`);
    renderDocuments(data.documents || []);
  } catch (err) {
    grid.replaceChildren();
    countEl.textContent = "0 documents";
    showToast(err.message || "Failed to load documents", "error");
  }
}

export function initLibrary() {
  const goToUpload = () => {
    document.querySelector('.nav-link[data-view="rag"]')?.click();
    window.setTimeout(() => fileInput?.click(), 260);
  };
  uploadBtn.addEventListener("click", goToUpload);
  emptyUploadBtn.addEventListener("click", goToUpload);

  onProviderChange(() => {
    if (loaded) loadDocuments();
  });
}

export function activateLibrary() {
  loaded = true;
  loadDocuments();
}

/** Wipe rendered documents + cached state on sign-out. */
export function clearLibrary() {
  loaded = false;
  grid.replaceChildren();
  countEl.textContent = "0 documents";
  emptyEl.classList.add("hidden");
}
