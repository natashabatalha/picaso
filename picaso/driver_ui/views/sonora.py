"""
Sonora viewer page: browse the Sonora grids (Bobcat, Elf Owl, Diamondback, Flame Skimmer) that are on disk,
see what each download holds and how big it is, and plot a model's pressure-temperature profile, chemistry,
clouds and published spectrum. Nothing is computed; the models are read as published.
"""
import os

from flask import Blueprint, current_app, render_template, request

import picaso.data as data
from picaso.driver_ui.core import sonora
from picaso.driver_ui.core.plots import plotly_plot

bp = Blueprint("sonora", __name__, url_prefix="/sonora")

MAX_COMPARE = 8


def default_root():
    refdata = current_app.config["PICASO_REFDATA"]
    return os.path.join(refdata, "sonora_grids") if refdata else ""


def sonora_downloads():
    """The sonora_grids download targets, or ({}, error)."""
    try:
        return data.get_data_config()[1].get("sonora_grids", {}), None
    except Exception as e:
        return {}, f"Could not read the download list: {e}"


def parse_value(raw):
    """Form value of a parameter: '' means equilibrium / cloud-free (None)."""
    raw = (raw or "").strip()
    return None if raw in ("", "none") else float(raw)


def number(name, default):
    try:
        return float(request.form.get(name, "")) if request.form.get(name, "").strip() else default
    except ValueError:
        return default


def page_choices(root, grids_on_disk, submitted):
    """The grid, requested parameters and plot options, from the form or the defaults."""
    form = request.form if submitted else {}
    grid = form.get("grid") if form.get("grid") in grids_on_disk else (grids_on_disk[0] if grids_on_disk else None)
    df = sonora.models(root, grid) if grid else None
    same_grid = submitted and form.get("loaded_grid") == grid
    if grid is None:
        request_ = {}
    elif same_grid:
        request_ = {key: parse_value(form.get(key)) for key, _, _ in sonora.PARAMETERS if key in sonora.parameter_values(df)}
        if request_.get("teff") is None or request_.get("grav") is None:
            request_ = sonora.default_request(df, grid)
    else:
        request_ = sonora.default_request(df, grid)
    choices = {
        "grid": grid,
        "request": request_,
        "molecules": request.form.getlist("molecule") if same_grid else None,
        "compare": [f for f in request.form.getlist("compare") if os.path.isfile(f)] if submitted else [],
        "R": number("R", 300.0) if submitted else 300.0,
        "wave_min": number("wave_min", 0.5) if submitted else 0.5,
        "wave_max": number("wave_max", 15.0) if submitted else 15.0,
        "yscale": "linear" if form.get("yscale") == "linear" else "log",
        "cloud_wave": number("cloud_wave", 1.0) if submitted else 1.0,
    }
    if choices["R"] is not None and choices["R"] <= 0:
        choices["R"] = None
    return choices, df


def compare_models(root, choices):
    """The comparison list as (grid, row) pairs; files are kept even when another grid is being browsed."""
    out = []
    for filename in choices["compare"]:
        for grid in sonora.GRIDS:
            df = sonora.models(root, grid)
            match = df[df["filename"] == filename]
            if len(match):
                out.append((grid, match.iloc[0]))
                break
    return out


def render_page(root, submitted):
    downloads, download_error = sonora_downloads()
    root_ok = bool(root) and os.path.isdir(os.path.expanduser(root))
    catalog = sonora.catalog(root if root_ok else "", downloads)  #every download is listed, even with nothing on disk
    counts = {g: len(sonora.models(root, g)) if root_ok else 0 for g in sonora.GRIDS}
    grids_on_disk = [g for g in sonora.GRIDS if counts[g]]
    choices, df = page_choices(root, grids_on_disk, submitted)

    messages, spectrum_messages, figures, values, selected = [], [], {}, {}, None
    if download_error:
        messages.append(("error", download_error))
    if not root_ok:
        messages.append(("warning", f"Sonora grid folder not found: {root or '(no reference data set)'}. "
                                    "Download a grid below, or enter the folder that holds one folder per grid."))
    elif not grids_on_disk:
        messages.append(("warning", "No Sonora models were found in this folder yet. Download a grid below."))

    if choices["grid"]:
        grid = choices["grid"]
        values = sonora.parameter_values(df)
        try:
            row, note = sonora.select(df, grid, choices["request"])
            if note:
                messages.append(("info", note))
            selected = row
        except Exception as e:
            messages.append(("error", str(e)))

        action = request.form.get("action") if submitted else None
        if action == "clear":
            choices["compare"] = []
        elif action == "add" and selected is not None and selected["filename"] not in choices["compare"]:
            choices["compare"] = (choices["compare"] + [selected["filename"]])[-MAX_COMPARE:]
        elif action and action.startswith("remove:"):
            choices["compare"] = [f for f in choices["compare"] if f != action.split(":", 1)[1]]

        shown = compare_models(root, choices)
        if selected is not None and selected["filename"] not in choices["compare"]:
            shown = [(grid, selected)] + shown
        models = []
        for g, r in shown:
            try:
                models.append(sonora.load_model(root, g, r))
            except Exception as e:
                messages.append(("error", f"Could not read {os.path.basename(r['filename'])}: {e}"))

        no_spectrum = {}  #download target -> labels of the models without a published spectrum
        for m in models:
            for what in m.missing:
                target = what.split("(")[-1].rstrip(")")
                if what.startswith("published spectrum"):
                    no_spectrum.setdefault(target, []).append(m.label)
                else:
                    messages.append(("info", f"{m.label}: no {what.split(' (')[0]} found on disk. "
                                             f"Download '{target}' (see the table below) to show it."))
        for target, labels in no_spectrum.items():
            size = downloads.get(target, {}).get("size_gb")
            which = "" if len(labels) == len(models) else f" for {', '.join(labels)}"
            spectrum_messages.append(("error", f"Spectra not downloaded{which}. Download '{target}'"
                                               f"{f' ({size:.3g} GB)' if size else ''} from the table below to show them."))

        available = sonora.molecules_in(models)
        if choices["molecules"] is None:
            choices["molecules"] = [m for m in sonora.DEFAULT_MOLECULES if m in available]
        choices["molecules"] = [m for m in choices["molecules"] if m in available]
        choices["available_molecules"] = available
        choices["shown"] = [m.label for m in models]

        if models:
            figures["pt"] = sonora.pt_figure(models)
            figures["chemistry"] = sonora.chemistry_figure(models, choices["molecules"])
            figures["clouds"] = sonora.cloud_figure(models, current_app.config["PICASO_REFDATA"], choices["cloud_wave"])
            if choices["wave_min"] > 0 and choices["wave_max"] > choices["wave_min"]:
                figures["spectrum"] = sonora.spectrum_figure(models, choices["R"],
                                                             (choices["wave_min"], choices["wave_max"]), choices["yscale"])
            else:
                messages.append(("error", "The wavelength range needs 0 < minimum < maximum."))
        choices["compare_labels"] = [(f, sonora.model_label(g, r)) for f, (g, r) in
                                     zip(choices["compare"], compare_models(root, choices))]

    figures = {k: plotly_plot(v) for k, v in figures.items() if v is not None}
    return render_template("sonora/index.html", root=root, root_ok=root_ok, catalog=catalog, counts=counts,
                           grids=sonora.GRIDS, grids_on_disk=grids_on_disk, choices=choices, values=values,
                           parameters=sonora.PARAMETERS, selected=selected, figures=figures, messages=messages,
                           spectrum_messages=spectrum_messages)


@bp.get("/")
def index():
    root = request.args.get("root")
    return render_page(default_root() if root is None else root.strip(), submitted=False)


@bp.post("/plot")
def plot():
    root = request.form.get("root", "").strip()
    return render_page(root, submitted=root == request.form.get("loaded_root"))
