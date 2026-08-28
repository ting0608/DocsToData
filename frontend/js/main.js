// App bootstrap: wires navigation, auth, and the three feature views.

import { activateEvaluation, initEvaluation } from "./evaluation.js";
import { initAuthView, refreshAuthUi } from "./authView.js";
import { activateLibrary, initLibrary } from "./library.js";
import { goToView, initNav, onViewChange } from "./nav.js";
import { activateUploadRag, initUploadRag } from "./uploadRag.js";

async function bootstrap() {
  // Auth must initialize first: it consumes the Cognito redirect (?code=...)
  // and resolves the current identity before other views render.
  await initAuthView();

  initUploadRag();
  initLibrary();
  initEvaluation();
  initNav("rag");

  onViewChange((viewName) => {
    if (viewName === "rag") activateUploadRag();
    if (viewName === "library") activateLibrary();
    if (viewName === "evaluation") activateEvaluation();
    if (viewName === "auth") refreshAuthUi();
  });
}

bootstrap();

// Exposed for debugging in the browser console only; not part of the public API.
window.__d2d = { goToView };
