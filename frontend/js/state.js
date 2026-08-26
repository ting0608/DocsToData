// App-wide shared state: currently selected model provider.
// Kept tiny and dependency-free so every view module can read/react to it.

const providerEl = document.getElementById("provider");
const listeners = new Set();

export function currentProvider() {
  return providerEl.value;
}

export function onProviderChange(fn) {
  listeners.add(fn);
}

providerEl.addEventListener("change", () => {
  for (const fn of listeners) fn(currentProvider());
});
