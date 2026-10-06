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
