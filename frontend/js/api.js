// Shared fetch wrapper: attaches the Cognito bearer token (when signed in,
// refreshing it first if it's stale) and normalizes error handling across
// the app. On a 401 with a signed-in session, retries exactly once after a
// forced token refresh -- covers the case where the access token expired
// between our proactive refresh and the server actually processing it.

import { forceRefreshAccessToken, getValidAccessToken, isSignedIn } from "./auth.js";
import { API_BASE_URL } from "./config.js";

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function doFetch(path, options, token) {
  const headers = new Headers(options.headers || {});
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const res = await fetch(`${API_BASE_URL}${path}`, { ...options, headers });
  const contentType = res.headers.get("content-type") || "";
  const data = contentType.includes("application/json") ? await res.json() : null;
  return { res, data };
}

export async function apiFetch(path, options = {}) {
  const token = await getValidAccessToken();
  let { res, data } = await doFetch(path, options, token);

  if (res.status === 401 && isSignedIn()) {
    const refreshed = await forceRefreshAccessToken();
    if (refreshed) {
      ({ res, data } = await doFetch(path, options, refreshed));
    }
  }

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
