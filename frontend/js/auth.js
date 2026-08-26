// AWS Cognito authentication: Hosted UI + Authorization Code flow with PKCE.
//
// English: Tokens are kept in memory + sessionStorage (not localStorage) to
// reduce persistence of access tokens across browser sessions/tabs.
// 中文: Token 存放在記憶體與 sessionStorage（不用 localStorage），降低 access
// token 在瀏覽器重啟後仍被保留的風險。

import { API_BASE_URL } from "./config.js";

const STORAGE_KEY = "d2d_auth_tokens";
const PKCE_VERIFIER_KEY = "d2d_pkce_verifier";
const PKCE_STATE_KEY = "d2d_pkce_state";

let cachedConfig = null;
let tokens = loadTokensFromStorage();

function loadTokensFromStorage() {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function saveTokens(next) {
  tokens = next;
  if (next) {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } else {
    sessionStorage.removeItem(STORAGE_KEY);
  }
}

export async function fetchAuthConfig() {
  if (cachedConfig) return cachedConfig;
  const res = await fetch(`${API_BASE_URL}/auth/config`);
  cachedConfig = await res.json();
  return cachedConfig;
}

export function isAuthEnabled() {
  return Boolean(cachedConfig?.enabled);
}

export function getAccessToken() {
  return tokens?.access_token || null;
}

export function isSignedIn() {
  return Boolean(tokens?.access_token);
}

function base64UrlEncode(bytes) {
  let str = "";
  for (const b of bytes) str += String.fromCharCode(b);
  return btoa(str).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function randomString(length = 64) {
  const bytes = new Uint8Array(length);
  crypto.getRandomValues(bytes);
  return base64UrlEncode(bytes);
}

async function sha256Base64Url(input) {
  const data = new TextEncoder().encode(input);
  const digest = await crypto.subtle.digest("SHA-256", data);
  return base64UrlEncode(new Uint8Array(digest));
}

function redirectUri() {
  return `${window.location.origin}${window.location.pathname}`;
}

async function buildAuthorizeUrl({ identityProvider } = {}) {
  const cfg = await fetchAuthConfig();
  if (!cfg.enabled || !cfg.domain) {
    throw new Error("Cognito is not configured on this server.");
  }

  const verifier = randomString(64);
  const state = randomString(24);
  sessionStorage.setItem(PKCE_VERIFIER_KEY, verifier);
  sessionStorage.setItem(PKCE_STATE_KEY, state);
  const challenge = await sha256Base64Url(verifier);

  const params = new URLSearchParams({
    client_id: cfg.client_id,
    response_type: "code",
    scope: "openid email profile",
    redirect_uri: redirectUri(),
    state,
    code_challenge: challenge,
    code_challenge_method: "S256",
  });
  if (identityProvider) {
    params.set("identity_provider", identityProvider);
  }

  return `${cfg.domain}/oauth2/authorize?${params.toString()}`;
}

/** Redirect the browser to the Cognito Hosted UI sign-in page. */
export async function signIn({ identityProvider } = {}) {
  const url = await buildAuthorizeUrl({ identityProvider });
  window.location.assign(url);
}

/** Clear local tokens and redirect to the Cognito Hosted UI logout endpoint. */
export async function signOut() {
  const cfg = await fetchAuthConfig();
  saveTokens(null);
  if (!cfg.enabled || !cfg.domain) {
    window.location.reload();
    return;
  }
  const params = new URLSearchParams({
    client_id: cfg.client_id,
    logout_uri: redirectUri(),
  });
  window.location.assign(`${cfg.domain}/logout?${params.toString()}`);
}

async function exchangeCodeForTokens(code) {
  const cfg = await fetchAuthConfig();
  const verifier = sessionStorage.getItem(PKCE_VERIFIER_KEY) || "";

  const body = new URLSearchParams({
    grant_type: "authorization_code",
    client_id: cfg.client_id,
    code,
    redirect_uri: redirectUri(),
    code_verifier: verifier,
  });

  const res = await fetch(`${cfg.domain}/oauth2/token`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body,
  });
  if (!res.ok) {
    throw new Error("Failed to exchange authorization code for tokens.");
  }
  const data = await res.json();
  saveTokens({
    access_token: data.access_token,
    id_token: data.id_token,
    refresh_token: data.refresh_token,
    obtained_at: Date.now(),
  });
}

/**
 * Handle the `?code=...&state=...` redirect from the Cognito Hosted UI, if present.
 * Safe to call on every page load; it is a no-op when there is no auth redirect.
 */
export async function completeSignInRedirect() {
  const url = new URL(window.location.href);
  const code = url.searchParams.get("code");
  const state = url.searchParams.get("state");
  if (!code) return false;

  const expectedState = sessionStorage.getItem(PKCE_STATE_KEY);
  sessionStorage.removeItem(PKCE_VERIFIER_KEY);
  sessionStorage.removeItem(PKCE_STATE_KEY);

  if (!expectedState || state !== expectedState) {
    throw new Error("Auth state mismatch. Please sign in again.");
  }

  await exchangeCodeForTokens(code);

  url.searchParams.delete("code");
  url.searchParams.delete("state");
  window.history.replaceState({}, document.title, url.pathname + url.search + url.hash);
  return true;
}

/** Fetch the resolved identity from the backend (works in dev mode too). */
export async function fetchCurrentUser() {
  const headers = {};
  const token = getAccessToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  const res = await fetch(`${API_BASE_URL}/auth/me`, { headers });
  if (!res.ok) return null;
  const data = await res.json();
  return data.user;
}
