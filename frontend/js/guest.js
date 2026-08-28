// Guest-preview mode: lets a visitor bypass login and browse every screen
// populated with hardcoded, non-usable sample data.
//
// English: When guest mode is active, the view modules render from
// frontend/data/guest-data.json instead of calling the backend, and every
// mutating action (upload, query, run eval, save/delete/rate) is blocked
// with a toast prompting sign-in. Guest state is intentionally NOT persisted
// (no sessionStorage) so it never survives a reload -- a reload returns the
// visitor to the real login screen.
// 中文: 啟用訪客模式時，各視圖改為從 frontend/data/guest-data.json 讀取資料，
// 不呼叫後端；所有會變更資料的動作（上傳、提問、跑評估、儲存/刪除/評分）都會
// 被擋下並提示登入。訪客狀態刻意不做持久化（不用 sessionStorage），因此重新
// 整理頁面後會回到真正的登入畫面。

import { showToast } from "./toast.js";

// Resolve the JSON path from this module's own URL so it works both locally
// ("/frontend/data/...") and on Amplify ("/data/..."), matching how
// uploadRag.js resolves its asset paths. See uploadRag.js for the full
// rationale on why hardcoded "/frontend/..." paths break on Amplify.
const GUEST_DATA_URL = new URL("../data/guest-data.json", import.meta.url);

let guestActive = false;
let guestData = null;

export function isGuest() {
  return guestActive;
}

/** Load (and cache) the sample dataset. */
export async function loadGuestData() {
  if (guestData) return guestData;
  const res = await fetch(GUEST_DATA_URL);
  if (!res.ok) throw new Error("Failed to load guest sample data.");
  guestData = await res.json();
  return guestData;
}

/** Synchronous accessor; returns null until loadGuestData() has resolved. */
export function getGuestData() {
  return guestData;
}

export async function enterGuestMode() {
  await loadGuestData();
  guestActive = true;
}

export function exitGuestMode() {
  guestActive = false;
}

/**
 * Guard for mutating actions. Returns true (and shows a toast) when the app
 * is in guest mode, so callers can early-return. Usage:
 *   if (blockIfGuest()) return;
 */
export function blockIfGuest(message = "Sign in to use this feature. You're viewing sample data as a guest.") {
  if (!guestActive) return false;
  showToast(message, "info");
  return true;
}
