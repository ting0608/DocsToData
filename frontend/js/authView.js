// Wires the Account view + topbar user badge to the Cognito auth module.

import {
  completeSignInRedirect,
  confirmSignUp,
  fetchAuthConfig,
  fetchCurrentUser,
  isSignedIn,
  resendConfirmationCode,
  getIdentityFromIdToken,
  signIn,
  signInWithPassword,
  signOutPassword,
  signUp,
} from "./auth.js";
import { clearEvaluation } from "./evaluation.js";
import { clearLibrary } from "./library.js";
import { goToView } from "./nav.js";
import { showToast } from "./toast.js";
import { clearUploadRag } from "./uploadRag.js";

const devModePanel = document.getElementById("authDevModePanel");
const loggedOutPanel = document.getElementById("authLoggedOutPanel");
const loggedInPanel = document.getElementById("authLoggedInPanel");

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

// Login / signup / confirm sub-forms within authLoggedOutPanel.
const authLoginForm = document.getElementById("authLoginForm");
const authSignupForm = document.getElementById("authSignupForm");
const authConfirmForm = document.getElementById("authConfirmForm");

const loginForm = document.getElementById("loginForm");
const loginEmail = document.getElementById("loginEmail");
const loginPassword = document.getElementById("loginPassword");

const signupForm = document.getElementById("signupForm");
const signupEmail = document.getElementById("signupEmail");
const signupPassword = document.getElementById("signupPassword");

const confirmForm = document.getElementById("confirmForm");
const confirmCode = document.getElementById("confirmCode");
const confirmEmailLabel = document.getElementById("confirmEmailLabel");
const resendCodeBtn = document.getElementById("resendCodeBtn");

const showSignupBtn = document.getElementById("showSignupBtn");
const showLoginBtn = document.getElementById("showLoginBtn");

// Email pending confirmation, set right after a successful sign-up.
let pendingConfirmEmail = null;

function showAuthSubForm(name) {
  authLoginForm.classList.toggle("hidden", name !== "login");
  authSignupForm.classList.toggle("hidden", name !== "signup");
  authConfirmForm.classList.toggle("hidden", name !== "confirm");
}

function initials(name) {
  return (name || "U").trim().slice(0, 1).toUpperCase();
}

function renderSignedOut(cfg) {
  devModePanel.classList.toggle("hidden", cfg.enabled);
  loggedOutPanel.classList.toggle("hidden", !cfg.enabled);
  loggedInPanel.classList.add("hidden");
  signInSsoBtn.classList.toggle("hidden", !cfg.sso_provider);
  showAuthSubForm("login");

  userBadge.classList.add("hidden");
  topbarSignInBtn.classList.toggle("hidden", !cfg.enabled);
}

function renderSignedIn(user) {
  devModePanel.classList.add("hidden");
  loggedOutPanel.classList.add("hidden");
  loggedInPanel.classList.remove("hidden");
  topbarSignInBtn.classList.add("hidden");

  // The access token (verified by the backend) has no email claim; pull the
  // email/name from the id token for display when available.
  const idClaims = user.dev_mode ? {} : getIdentityFromIdToken();
  const email = user.email || idClaims.email || null;
  const displayName = idClaims.name || email || user.username || "User";

  authUserName.textContent = displayName;
  authUserEmail.textContent = email || (user.dev_mode ? "Dev mode (no login configured)" : "");
  authUserSub.textContent = user.dev_mode ? "" : `Subject: ${user.sub}`;
  authUserAvatar.textContent = initials(displayName);

  userBadge.classList.remove("hidden");
  userName.textContent = displayName;
  userAvatar.textContent = initials(displayName);
}

export async function refreshAuthUi() {
  let cfg;
  let user = null;
  try {
    cfg = await fetchAuthConfig();
    user = await fetchCurrentUser();
  } catch {
    // Network/parse failure: fall back to a safe signed-out view rather than
    // leaving both panels visible (half-rendered state).
    cfg = cfg || { enabled: true, sso_provider: null };
  }

  if (cfg && !cfg.enabled) {
    // Dev mode: backend accepts every request, so treat the app as "signed in".
    renderSignedIn(user || { username: "dev", dev_mode: true });
    return;
  }

  if (isSignedIn() && user) {
    renderSignedIn(user);
  } else {
    renderSignedOut(cfg || { enabled: true, sso_provider: null });
  }
}

export async function initAuthView() {
  try {
    const redirected = await completeSignInRedirect();
    if (redirected) showToast("Signed in successfully", "success");
  } catch (err) {
    showToast(err.message || "Sign-in failed", "error");
  }

  loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      await signInWithPassword(loginEmail.value.trim(), loginPassword.value);
      loginForm.reset();
      showToast("Signed in successfully", "success");
      await refreshAuthUi();
    } catch (err) {
      showToast(err.message || "Sign-in failed", "error");
    }
  });

  signupForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const email = signupEmail.value.trim();
    try {
      const result = await signUp(email, signupPassword.value);
      signupForm.reset();
      if (result.user_confirmed) {
        showToast("Account created. You can sign in now.", "success");
        showAuthSubForm("login");
      } else {
        pendingConfirmEmail = email;
        confirmEmailLabel.textContent = email;
        showToast("Check your email for a confirmation code.", "info");
        showAuthSubForm("confirm");
      }
    } catch (err) {
      showToast(err.message || "Sign-up failed", "error");
    }
  });

  confirmForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!pendingConfirmEmail) {
      showToast("No pending confirmation. Please sign up again.", "error");
      showAuthSubForm("signup");
      return;
    }
    try {
      await confirmSignUp(pendingConfirmEmail, confirmCode.value.trim());
      confirmForm.reset();
      showToast("Email verified. You can sign in now.", "success");
      loginEmail.value = pendingConfirmEmail;
      pendingConfirmEmail = null;
      showAuthSubForm("login");
    } catch (err) {
      showToast(err.message || "Verification failed", "error");
    }
  });

  resendCodeBtn.addEventListener("click", async () => {
    if (!pendingConfirmEmail) return;
    try {
      await resendConfirmationCode(pendingConfirmEmail);
      showToast("Confirmation code re-sent.", "success");
    } catch (err) {
      showToast(err.message || "Failed to resend code", "error");
    }
  });

  showSignupBtn.addEventListener("click", () => showAuthSubForm("signup"));
  showLoginBtn.addEventListener("click", () => showAuthSubForm("login"));

  signInSsoBtn.addEventListener("click", async () => {
    const cfg = await fetchAuthConfig();
    signIn({ identityProvider: cfg.sso_provider }).catch((err) => showToast(err.message, "error"));
  });

  const doSignOut = () =>
    signOutPassword()
      .then(() => {
        // Wipe every view's rendered data + cached state so the next user
        // (or the signed-out screen) never shows the previous session's data.
        clearUploadRag();
        clearLibrary();
        clearEvaluation();
      })
      .then(() => refreshAuthUi());
  signOutBtn.addEventListener("click", doSignOut);
  topbarSignOutBtn.addEventListener("click", doSignOut);
  topbarSignInBtn.addEventListener("click", () => goToView("auth"));

  await refreshAuthUi();
}
