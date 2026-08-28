// AWS Cognito authentication. Two flows live here:
//   1. Classic email/password login (primary): the frontend posts
//      credentials to our own backend (/auth/login, /auth/refresh, ...),
//      which calls Cognito's InitiateAuth server-side. This is what
//      signInWithPassword()/signUp()/etc below drive.
//   2. Hosted UI + Authorization Code + PKCE redirect (kept for SSO): still
//      used by signIn()/completeSignInRedirect() for the "Continue with SSO"
//      button, unchanged from before.
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
// Serializes concurrent refresh attempts so multiple in-flight 401s don't
// each spend their own refresh_token exchange.
let refreshInFlight = null;

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

/**
 * Decode display claims (email, name) from the stored id token.
 *
 * English: The backend verifies the *access* token, which for Cognito does
 * not carry the email claim -- only the *id* token does. This reads email
 * from the id token for display purposes only (never for authorization).
 * Returns {} if there's no id token or it can't be parsed.
 * 中文: 後端驗證的是 access token，而 Cognito 的 access token 不帶 email，只有
 * id token 才有。這裡只為了「顯示」而從 id token 讀出 email（絕不用於授權）。
 */
export function getIdentityFromIdToken() {
  const idToken = tokens?.id_token;
  if (!idToken) return {};
  try {
    const payload = idToken.split(".")[1];
    const json = atob(payload.replace(/-/g, "+").replace(/_/g, "/"));
    const claims = JSON.parse(json);
    return {
      email: claims.email || null,
      name: claims.name || claims["cognito:username"] || claims.email || null,
    };
  } catch {
    return {};
  }
}

async function postJson(path, body) {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    throw new Error(data?.detail || res.statusText || "Request failed");
  }
  return data;
}

function storeAuthResult(tokenSet) {
  saveTokens({
    access_token: tokenSet.access_token,
    id_token: tokenSet.id_token,
    refresh_token: tokenSet.refresh_token,
    // expires_in is seconds from Cognito; obtained_at lets getValidAccessToken()
    // know when to proactively refresh instead of waiting for a 401.
    expires_in: tokenSet.expires_in,
    obtained_at: Date.now(),
  });
}

/** Classic email/password sign-in against our backend's /auth/login (Cognito InitiateAuth). */
export async function signInWithPassword(email, password) {
  const data = await postJson("/auth/login", { email, password });
  storeAuthResult(data.tokens);
  return data.tokens;
}

/** Register a new account. Cognito emails a confirmation code to `email`. */
export async function signUp(email, password) {
  return postJson("/auth/signup", { email, password });
}

/** Confirm a new account with the code emailed by Cognito. */
export async function confirmSignUp(email, code) {
  return postJson("/auth/confirm-signup", { email, code });
}

/** Re-send the sign-up confirmation code. */
export async function resendConfirmationCode(email) {
  return postJson("/auth/resend-code", { email });
}

/** True once we're within `marginSeconds` of the access token's expiry (or already expired). */
function isAccessTokenStale(marginSeconds = 60) {
  if (!tokens?.access_token || !tokens?.expires_in || !tokens?.obtained_at) {
    return !tokens?.access_token;
  }
  const expiresAt = tokens.obtained_at + tokens.expires_in * 1000;
  return Date.now() + marginSeconds * 1000 >= expiresAt;
}

async function doRefresh() {
  if (!tokens?.refresh_token) {
    throw new Error("No refresh token available; please sign in again.");
  }
  const data = await postJson("/auth/refresh", { refresh_token: tokens.refresh_token });
  storeAuthResult({ ...data.tokens, refresh_token: tokens.refresh_token });
  return tokens.access_token;
}

/**
 * Return a currently-valid access token, transparently refreshing it first
 * if it's expired/near-expiry. Used by api.js before every request, and
 * again after a 401 as a fallback for the classic login flow.
 */
export async function getValidAccessToken() {
  if (!tokens?.access_token) return null;
  if (!isAccessTokenStale()) return tokens.access_token;

  if (!refreshInFlight) {
    refreshInFlight = doRefresh().finally(() => {
      refreshInFlight = null;
    });
  }
  try {
    return await refreshInFlight;
  } catch {
    saveTokens(null);
    return null;
  }
}

/** Force a refresh regardless of staleness (used by api.js's 401 retry-once). */
export async function forceRefreshAccessToken() {
  if (!tokens?.refresh_token) return null;
  if (!refreshInFlight) {
    refreshInFlight = doRefresh().finally(() => {
      refreshInFlight = null;
    });
  }
  try {
    return await refreshInFlight;
  } catch {
    saveTokens(null);
    return null;
  }
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

/**
 * Sign out of the classic email/password session: invalidates all refresh
 * tokens for this user server-side (global sign-out) and clears local state.
 * Does not redirect (there's no Hosted UI page involved in this flow).
 */
export async function signOutPassword() {
  const token = getAccessToken();
  saveTokens(null);
  if (token) {
    try {
      await fetch(`${API_BASE_URL}/auth/logout`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
      });
    } catch {
      // Best-effort: local tokens are already cleared either way.
    }
  }
}

/** Clear local tokens and redirect to the Cognito Hosted UI logout endpoint (SSO/Hosted UI flow). */
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

/** Fetch the resolved identity from the backend (works in dev mode too).
 *
 * English: If we hold a token but the server rejects it (401 -- expired and
 * un-refreshable), clear it so the UI resolves cleanly to "signed out"
 * instead of getting stuck half-signed-in (stale token in storage but no
 * real identity). This is what makes refreshAuthUi() reliable across
 * expired sessions.
 */
export async function fetchCurrentUser() {
  const headers = {};
  const token = await getValidAccessToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  const res = await fetch(`${API_BASE_URL}/auth/me`, { headers });
  if (!res.ok) {
    if (res.status === 401 && tokens) saveTokens(null);
    return null;
  }
  const data = await res.json();
  return data.user;
}
