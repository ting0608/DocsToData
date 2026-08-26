// Top navigation: switches between the four views and animates the active
// indicator underline to slide beneath the selected nav link.

const navLinks = Array.from(document.querySelectorAll(".nav-link"));
const indicator = document.getElementById("navIndicator");
const viewSections = Array.from(document.querySelectorAll(".view"));

const listeners = new Set();

function moveIndicatorTo(link) {
  if (!link) return;
  indicator.style.width = `${link.offsetWidth}px`;
  indicator.style.transform = `translateX(${link.offsetLeft}px)`;
}

function activateView(viewName) {
  for (const section of viewSections) {
    section.classList.toggle("active", section.dataset.view === viewName);
  }
  for (const link of navLinks) {
    link.classList.toggle("active", link.dataset.view === viewName);
  }
  const activeLink = navLinks.find((l) => l.dataset.view === viewName);
  moveIndicatorTo(activeLink);

  for (const fn of listeners) fn(viewName);
}

export function onViewChange(fn) {
  listeners.add(fn);
}

export function goToView(viewName) {
  activateView(viewName);
}

export function initNav(defaultView = "rag") {
  for (const link of navLinks) {
    link.addEventListener("click", () => activateView(link.dataset.view));
  }
  window.addEventListener("resize", () => {
    const activeLink = navLinks.find((l) => l.classList.contains("active"));
    moveIndicatorTo(activeLink);
  });

  activateView(defaultView);
}
