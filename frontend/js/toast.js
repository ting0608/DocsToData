// Small animated toast notifications, mounted into #toastContainer.

const container = document.getElementById("toastContainer");

export function showToast(message, type = "info", duration = 3200) {
  if (!container) return;
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = message;
  container.appendChild(el);

  window.setTimeout(() => {
    el.classList.add("leaving");
    window.setTimeout(() => el.remove(), 220);
  }, duration);
}
