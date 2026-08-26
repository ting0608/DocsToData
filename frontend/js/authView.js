// Wires the Account view + topbar user badge to the Cognito auth module.

import {
  completeSignInRedirect,
  fetchAuthConfig,
  fetchCurrentUser,
  isSignedIn,
  signIn,
  signOut,
} from "./auth.js";
import { showToast } from "./toast.js";

const devModePanel = document.getElementById("authDevModePanel");
const loggedOutPanel = document.getElementById("authLoggedOutPanel");
const loggedInPanel = document.getElementById("authLoggedInPanel");

const signInEmailBtn = document.getElementById("signInEmailBtn");
const signInSsoBtn = document.getElementById("signInSsoBtn");
const signOutBtn = document.getElementById("signOutBtn");

const authUserAvatar = document.getElementById("authUserAvatar");
const authUserName = document.getElementById("authUserName");
const authUserEmail = document.getElementById("authUserEmail");
const authUserSub = document.getElementById("authUserSub");

const userBadge = document.getElementById("userBadge");
const userAvatar = document.getElementById("userAvatar");
const userName = document.getElementById("userName");
const topbarSignInBtn = document.getElementById("topbarSignInBtn");
const topbarSignOutBtn = document.getElementById("topbarSignOutBtn");

function initials(name) {
  return (name || "U").trim().slice(0, 1).toUpperCase();
}

function renderSignedOut(cfg) {
  devModePanel.classList.toggle("hidden", cfg.enabled);
  loggedOutPanel.classList.toggle("hidden", !cfg.enabled);
  loggedInPanel.classList.add("hidden");
  signInSsoBtn.classList.toggle("hidden", !cfg.sso_provider);

  userBadge.classList.add("hidden");
  topbarSignInBtn.classList.toggle("hidden", !cfg.enabled);
}

function renderSignedIn(user) {
  devModePanel.classList.add("hidden");
  loggedOutPanel.classList.add("hidden");
  loggedInPanel.classList.remove("hidden");
  topbarSignInBtn.classList.add("hidden");

  const displayName = user.username || user.email || "User";
  authUserName.textContent = displayName;
  authUserEmail.textContent = user.email || (user.dev_mode ? "Dev mode (no login configured)" : "");
  authUserSub.textContent = user.dev_mode ? "" : `Subject: ${user.sub}`;
  authUserAvatar.textContent = initials(displayName);

  userBadge.classList.remove("hidden");
  userName.textContent = displayName;
  userAvatar.textContent = initials(displayName);
}

export async function refreshAuthUi() {
  const cfg = await fetchAuthConfig();
  const user = await fetchCurrentUser();

  if (!cfg.enabled) {
    // Dev mode: backend accepts every request, so treat the app as "signed in".
    renderSignedIn(user || { username: "dev", dev_mode: true });
    return;
  }

  if (isSignedIn() && user) {
    renderSignedIn(user);
  } else {
    renderSignedOut(cfg);
  }
}

export async function initAuthView() {
  try {
    const redirected = await completeSignInRedirect();
    if (redirected) showToast("Signed in successfully", "success");
  } catch (err) {
    showToast(err.message || "Sign-in failed", "error");
  }

  signInEmailBtn.addEventListener("click", () => signIn().catch((err) => showToast(err.message, "error")));
  signInSsoBtn.addEventListener("click", async () => {
    const cfg = await fetchAuthConfig();
    signIn({ identityProvider: cfg.sso_provider }).catch((err) => showToast(err.message, "error"));
  });

  const doSignOut = () => signOut().catch((err) => showToast(err.message, "error"));
  signOutBtn.addEventListener("click", doSignOut);
  topbarSignOutBtn.addEventListener("click", doSignOut);
  topbarSignInBtn.addEventListener("click", () => signIn().catch((err) => showToast(err.message, "error")));

  await refreshAuthUi();
}
