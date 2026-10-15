// Draws the plotly figures the server embeds as <div class="plot" data-figure=...>,
// and reports failed requests (htmx otherwise ignores error responses).

function renderPlots() {
  if (typeof Plotly === "undefined") return;
  document.querySelectorAll(".plot[data-figure]").forEach((el) => {
    if (el.dataset.renderedId === el.dataset.figureId) return;
    const figure = JSON.parse(el.dataset.figure);
    Plotly.newPlot(el, figure.data, figure.layout, { responsive: true, ...figure.config })
      .then(() => el.dispatchEvent(new CustomEvent("plot:rendered", { bubbles: true })));
    el.dataset.renderedId = el.dataset.figureId;
  });
}

// Morphing would strip the DOM plotly built inside a plot; leave unchanged figures alone.
// The builder caption is written here, not by the server (see updateBuilder), so keep it too.
Idiomorph.defaults.callbacks.beforeNodeMorphed = (oldNode, newNode) =>
  !(oldNode.classList && oldNode.classList.contains("plot") &&
    oldNode.dataset.figureId === newNode.dataset.figureId) &&
  oldNode.id !== "builder-now";

// A re-rendered page would close an open dropdown (the server never renders `open`); keep it open.
// Likewise the builder strip's current step, which only the browser knows (see the builder section below).
Idiomorph.defaults.callbacks.beforeAttributeUpdated = (attribute, node) =>
  !(attribute === "open" && node.classList && node.classList.contains("multiselect")) &&
  !((attribute === "data-current" || attribute === "aria-current") && node.classList && node.classList.contains("builder-step"));

function showError(message) {
  const toast = document.getElementById("error-toast");
  toast.querySelector(".message").textContent = message;
  toast.hidden = false;
}

document.addEventListener("DOMContentLoaded", renderPlots);
document.addEventListener("htmx:afterSettle", renderPlots);
document.addEventListener("htmx:responseError", (event) => {
  showError(`Request failed (${event.detail.xhr.status}). See the server console for details.`);
});
document.addEventListener("htmx:sendError", () => showError("Could not reach the server. Is it still running?"));

// ---------- multi-select dropdowns (<details class="multiselect"> of checkboxes) ----------
function updateMultiselectLabel(menu) {
  const label = menu.querySelector(".multiselect-label");
  const checked = [...menu.querySelectorAll("input[type=checkbox]:checked")].map((el) => el.dataset.label);
  label.classList.toggle("empty", checked.length === 0);
  label.textContent = checked.length === 0 ? label.dataset.placeholder
    : checked.length <= 3 ? checked.join(", ") : `${checked.length} selected`;
}

document.addEventListener("change", (event) => {
  const menu = event.target.closest(".multiselect");
  if (menu && event.target.type === "checkbox") updateMultiselectLabel(menu);
});

document.addEventListener("click", (event) => {
  const button = event.target.closest(".multiselect [data-select]");
  if (button) {
    const menu = button.closest(".multiselect");
    const on = button.dataset.select === "all";
    // "All" respects the filter: only the options currently shown are selected
    menu.querySelectorAll(".multiselect-options label:not([hidden]) input[type=checkbox]").forEach((el) => { el.checked = on; });
    if (!on) menu.querySelectorAll("input[type=checkbox]").forEach((el) => { el.checked = false; });
    updateMultiselectLabel(menu);
    menu.dispatchEvent(new Event("change", { bubbles: true }));  // forms that post on change (cards) post now
    return;
  }
  // clicking anywhere outside an open dropdown closes it
  document.querySelectorAll(".multiselect[open]").forEach((menu) => {
    if (!menu.contains(event.target)) menu.open = false;
  });
});

document.addEventListener("input", (event) => {
  if (!event.target.classList.contains("multiselect-filter")) return;
  const query = event.target.value.trim().toLowerCase();
  event.target.closest(".multiselect").querySelectorAll(".multiselect-options label").forEach((label) => {
    label.hidden = query !== "" && !label.textContent.toLowerCase().includes(query);
  });
});

document.addEventListener("keydown", (event) => {
  const menu = event.target.closest && event.target.closest(".multiselect");
  if (event.key === "Escape") {
    document.querySelectorAll(".multiselect[open]").forEach((m) => { m.open = false; });
    if (menu) menu.querySelector("summary").focus();
  } else if (event.key === "Enter" && event.target.classList.contains("multiselect-filter")) {
    event.preventDefault();  // don't submit the form from the filter box
  }
});

// only one dropdown open at a time
document.addEventListener("toggle", (event) => {
  if (!event.target.classList || !event.target.classList.contains("multiselect") || !event.target.open) return;
  document.querySelectorAll(".multiselect[open]").forEach((m) => { if (m !== event.target) m.open = false; });
  const filter = event.target.querySelector(".multiselect-filter");
  if (filter) filter.focus();
}, true);

// ---------- top-bar menus: hover/focus opens them in CSS; clicks toggle them for touch screens ----------
// A group's button is also a link to its Get started section. With a mouse, clicking it follows the link.
// On a touch screen (no hover), the first tap opens the menu and a second tap follows the link.
const canHover = window.matchMedia("(hover: hover)");
document.addEventListener("click", (event) => {
  const button = event.target.closest(".nav-button");
  document.querySelectorAll(".nav-group.open").forEach((group) => {
    if (!button || group !== button.parentElement) group.classList.remove("open");
  });
  if (!button) return;
  const group = button.parentElement;
  if (button.tagName === "A" && (canHover.matches || group.classList.contains("open"))) return;
  event.preventDefault();
  group.classList.toggle("open");
});

// ---------- model-builder strip (spectrum page, templates/spectrum/_builder.html) ----------
// The server renders each step's state; here the step whose cards are in view is marked current,
// the caption names it (or the hovered step), and clicking a step scrolls to its cards. Clicking an
// optional step that is switched off jumps to the switch that turns it on.
let hoveredStep = null;

function builderCurrentStep(builder) {
  const sections = [...document.querySelectorAll("#sections > section[data-step]")];
  if (sections.length === 0) return null;
  const line = builder.getBoundingClientRect().bottom + 60;
  let current = sections[0];
  for (const section of sections) if (section.getBoundingClientRect().top <= line) current = section;
  if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4) current = sections[sections.length - 1];
  return current.dataset.step;
}

function updateBuilder() {
  const builder = document.getElementById("builder");
  if (!builder) return;
  document.documentElement.style.setProperty("--builder-height", `${builder.offsetHeight}px`);
  const current = builderCurrentStep(builder);
  builder.querySelectorAll(".builder-step").forEach((step) => {
    const isCurrent = step.dataset.step === current;
    step.dataset.current = String(isCurrent);
    if (isCurrent) step.setAttribute("aria-current", "step"); else step.removeAttribute("aria-current");
  });
  const shown = builder.querySelector(`.builder-step[data-step="${hoveredStep || current}"]`);
  const caption = document.getElementById("builder-now");
  if (!shown || !caption) return;
  caption.replaceChildren(hoveredStep ? "" : "Now: ", Object.assign(document.createElement("b"), { textContent: shown.dataset.label }),
    `: ${shown.dataset.summary}${shown.dataset.state === "off" ? " Optional, currently off." : ""}`);
}

let builderTicking = false;
function scheduleBuilderUpdate() {
  if (builderTicking) return;
  builderTicking = true;
  requestAnimationFrame(() => { builderTicking = false; updateBuilder(); });
}
window.addEventListener("scroll", scheduleBuilderUpdate, { passive: true });
window.addEventListener("resize", scheduleBuilderUpdate);
document.addEventListener("DOMContentLoaded", updateBuilder);
document.addEventListener("htmx:afterSettle", updateBuilder);

const smoothScroll = () => (window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth");

document.addEventListener("click", (event) => {
  const step = event.target.closest(".builder-step");
  if (step) {
    if (step.dataset.state === "off" && step.dataset.toggle) {
      const toggle = document.getElementById(step.dataset.toggle);
      if (!toggle) return;
      toggle.closest("section").scrollIntoView({ behavior: smoothScroll(), block: "start" });
      toggle.focus({ preventScroll: true });
      const field = toggle.closest(".tf-check") || toggle.parentElement;
      field.classList.remove("builder-flash");
      void field.offsetWidth;  // restart the animation
      field.classList.add("builder-flash");
      return;
    }
    const target = document.querySelector(`#sections > section[data-step="${step.dataset.step}"]`);
    if (target) target.scrollIntoView({ behavior: smoothScroll(), block: "start" });
    return;
  }
  // a download is a plain link, so the page isn't re-rendered: mark its step here (the server records it too)
  const download = event.target.closest("[data-marks-step]");
  if (download) {
    const marked = document.getElementById(`step-${download.dataset.marksStep}`);
    if (marked) marked.dataset.state = "done";
  }
});

function setHoveredStep(step) {
  const key = step ? step.dataset.step : null;
  if (key === hoveredStep) return;
  hoveredStep = key;
  updateBuilder();
}
document.addEventListener("mouseover", (event) => setHoveredStep(event.target.closest && event.target.closest(".builder-step")));
document.addEventListener("focusin", (event) => setHoveredStep(event.target.closest(".builder-step")));
