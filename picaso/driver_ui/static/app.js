// Draws the plotly figures the server embeds as <div class="plot" data-figure=...>,
// and reports failed requests (htmx otherwise ignores error responses).

function renderPlots() {
  if (typeof Plotly === "undefined") return;
  document.querySelectorAll(".plot[data-figure]").forEach((el) => {
    if (el.dataset.renderedId === el.dataset.figureId) return;
    const figure = JSON.parse(el.dataset.figure);
    Plotly.newPlot(el, figure.data, figure.layout, { responsive: true });
    el.dataset.renderedId = el.dataset.figureId;
  });
}

// Morphing would strip the DOM plotly built inside a plot; leave unchanged figures alone.
Idiomorph.defaults.callbacks.beforeNodeMorphed = (oldNode, newNode) =>
  !(oldNode.classList && oldNode.classList.contains("plot") &&
    oldNode.dataset.figureId === newNode.dataset.figureId);

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
document.addEventListener("click", (event) => {
  const button = event.target.closest(".nav-button");
  document.querySelectorAll(".nav-group.open").forEach((group) => {
    if (!button || group !== button.parentElement) group.classList.remove("open");
  });
  if (button) button.parentElement.classList.toggle("open");
});
