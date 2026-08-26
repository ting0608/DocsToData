// Shared fetch wrapper: attaches the Cognito bearer token (when signed in)
// and normalizes error handling across the app.

import { getAccessToken } from "./auth.js";
import { API_BASE_URL } from "./config.js";

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

export async function apiFetch(path, options = {}) {
  const headers = new Headers(options.headers || {});
  const token = getAccessToken();
  if (token && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const res = await fetch(`${API_BASE_URL}${path}`, { ...options, headers });
  const contentType = res.headers.get("content-type") || "";
  const data = contentType.includes("application/json") ? await res.json() : null;

  if (!res.ok) {
    const message = data?.detail || res.statusText || "Request failed";
    throw new ApiError(message, res.status);
  }
  return data;
}

export const api = {
  get: (path) => apiFetch(path),
  post: (path, body) =>
    apiFetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  del: (path) => apiFetch(path, { method: "DELETE" }),
};
