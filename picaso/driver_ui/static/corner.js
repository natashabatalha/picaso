// Interactive corner plot (core.analysis.interactive_corner_figure). Shape i of the figure is the black
// line on parameter i's 1D posterior. Dragging it, clicking a histogram bar or typing a value moves the
// selected point, and the server recomputes the spectrum next to the corner plot there. Parameters are
// renamed by editing an axis title on the plot or a label in the table below it.

function cornerPoint(el) {
  return el.layout.shapes.slice(0, el.layout.meta.params.length).map((shape) => shape.x0);
}

function cornerStatus(el, message) {
  const status = el.closest(".corner").querySelector(".corner-status");
  if (status) status.textContent = message;
}

function syncCornerInputs(el) {
  const point = cornerPoint(el);
  el.closest(".corner").querySelectorAll("[data-corner-index]").forEach((input) => {
    input.value = Number(point[input.dataset.cornerIndex].toPrecision(6));
  });
}

function moveCrosshairs(el) {
  const { crosshairs } = el.layout.meta;
  if (!crosshairs.length) return Promise.resolve();
  const point = cornerPoint(el);
  return Plotly.restyle(el, { x: crosshairs.map(([, , j]) => [point[j]]), y: crosshairs.map(([, i]) => [point[i]]) },
                        crosshairs.map(([trace]) => trace));
}

// One request at a time; points chosen meanwhile collapse into one request for the latest.
function evaluateCorner(el) {
  if (el._cornerBusy) { el._cornerQueued = true; return; }
  el._cornerBusy = true;
  cornerStatus(el, "Computing the model…");
  const { params, model_trace: modelTrace, title_annotation: title } = el.layout.meta;
  const point = Object.fromEntries(params.map((param, i) => [param, cornerPoint(el)[i]]));
  fetch(el.closest(".corner").dataset.evaluateUrl, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ point }),
  })
    .then((response) => response.json().then((body) => {
      if (!response.ok) throw new Error(body.error || `request failed (${response.status})`);
      return body;
    }))
    .then((body) => {
      if (el._cornerQueued) return;  // already out of date
      Plotly.restyle(el, { x: [body.wavelength], y: [body.model] }, [modelTrace]);
      Plotly.relayout(el, { [`annotations[${title}].text`]: body.title });
    })
    .catch((error) => showError(error.message))
    .finally(() => {
      el._cornerBusy = false;
      cornerStatus(el, "");
      if (el._cornerQueued) { el._cornerQueued = false; evaluateCorner(el); }
    });
}

// values: {parameter index: new value}
function moveCornerLines(el, values) {
  const update = {};
  for (const [i, x] of Object.entries(values)) {
    Object.assign(update, { [`shapes[${i}].x0`]: x, [`shapes[${i}].x1`]: x, [`shapes[${i}].y0`]: 0,
                            [`shapes[${i}].y1`]: 1 });
  }
  el._cornerUpdating = true;  // our own relayout fires plotly_relayout too
  return Plotly.relayout(el, update).then(() => {
    el._cornerUpdating = false;
    syncCornerInputs(el);
    return moveCrosshairs(el);
  }).then(() => evaluateCorner(el));
}

// Shows `label` everywhere parameter i is named (axis titles, 1D posterior title, table) and saves it.
function setCornerLabel(el, i, label) {
  const meta = el.layout.meta;
  label = String(label ?? "").trim() || meta.labels[i];  // an emptied label goes back to what it was
  meta.labels[i] = label;
  const update = { [`annotations[${meta.label_annotations[i]}].text`]: `${label}<br>${meta.intervals[i]}` };
  for (const [axis, index] of Object.entries(meta.label_axes)) {
    if (index === i) update[`${axis}.title.text`] = label;
  }
  el._cornerUpdating = true;
  Plotly.relayout(el, update).then(() => { el._cornerUpdating = false; });
  const input = el.closest(".corner").querySelector(`[data-corner-label="${i}"]`);
  if (input) input.value = label;
  fetch(el.closest(".corner").dataset.labelUrl, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ param: meta.params[i], label }),
  }).then((response) => { if (!response.ok) showError(`Could not save the label (${response.status}).`); })
    .catch(() => showError("Could not reach the server. Is it still running?"));
}

function onCornerRelayout(el, update) {
  if (el._cornerUpdating) return;
  const { params, ranges, label_axes: labelAxes } = el.layout.meta;
  for (const [key, value] of Object.entries(update)) {
    const match = key.match(/^([xy]axis\d*)\.title(\.text)?$/);
    if (match && match[1] in labelAxes) {
      setCornerLabel(el, labelAxes[match[1]], value && value.text !== undefined ? value.text : value);
    }
  }
  const values = {};
  for (let i = 0; i < params.length; i++) {
    const keys = ["x0", "x1", "y0", "y1"].filter((key) => `shapes[${i}].${key}` in update);
    if (!keys.length) continue;
    const x0 = update[`shapes[${i}].x0`], x1 = update[`shapes[${i}].x1`];
    // dragging the line moves both ends; dragging one end (or only up/down) is snapped back to a vertical line
    let x = x0 !== undefined && x1 !== undefined ? (x0 + x1) / 2 : x0 ?? x1 ?? el.layout.shapes[i].x0;
    const [lo, hi] = ranges[i];
    if (lo < hi) x = Math.min(hi, Math.max(lo, x));
    values[i] = x;
  }
  if (Object.keys(values).length) moveCornerLines(el, values);
}

document.addEventListener("plot:rendered", (event) => {
  const el = event.target;
  if (el.id !== "plot-corner") return;
  el.on("plotly_relayout", (update) => onCornerRelayout(el, update));
  el.on("plotly_click", (data) => {
    const point = data.points[0];
    if (point && point.data.type === "bar") moveCornerLines(el, { [point.data.meta]: point.x });
  });
});

document.addEventListener("change", (event) => {
  const label = event.target.closest && event.target.closest("[data-corner-label]");
  const plot = document.getElementById("plot-corner");
  if (label && plot && plot.layout) setCornerLabel(plot, Number(label.dataset.cornerLabel), label.value);
});

document.addEventListener("change", (event) => {
  const input = event.target.closest && event.target.closest("[data-corner-index]");
  const el = document.getElementById("plot-corner");
  const x = input ? parseFloat(input.value) : NaN;
  if (el && el.layout && Number.isFinite(x)) moveCornerLines(el, { [input.dataset.cornerIndex]: x });
});

document.addEventListener("click", (event) => {
  if (!event.target.closest("[data-corner-reset]")) return;
  const el = document.getElementById("plot-corner");
  if (el && el.layout) moveCornerLines(el, Object.assign({}, el.layout.meta.medians));
});
