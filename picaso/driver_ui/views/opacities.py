"""
Opacity Viewer page: plot molecular and continuum cross sections straight from a
resampled opacity database. Defaults to the same file the environment checker
(picaso.data.check_environ) reports, but any database path can be entered.
"""
import os

from flask import Blueprint, current_app, render_template, request

import picaso.data as data
from picaso.driver_ui.core import opacities
from picaso.driver_ui.core.plots import plotly_plot

bp = Blueprint("opacities", __name__, url_prefix="/opacities")

DEFAULT_MOLECULES = ("H2O",)
DEFAULT_TEMPERATURE = 1000.0  # K
DEFAULT_PRESSURE = 1.0        # bar
DEFAULT_R = 1000.0


def default_db():
    refdata = current_app.config["PICASO_REFDATA"]
    path = data.default_opacity_file(refdata)[0] if refdata else None
    return path or ""


def default_choices(info):
    return {
        "molecules": [m for m in DEFAULT_MOLECULES if m in info.molecules] or list(info.molecules[:1]),
        "continuum": [],
        "temperatures": [opacities.nearest(info.temperatures, DEFAULT_TEMPERATURE)] if info.temperatures else [],
        "pressures": [opacities.nearest(info.pressures, DEFAULT_PRESSURE)] if info.pressures else [],
        "R": DEFAULT_R if DEFAULT_R < info.native_R else None,
        "wave_min": round(info.wave_min, 4),
        "wave_max": round(info.wave_max, 4),
        "xscale": "log", "yscale": "log", "xunit": "micron",
    }


def number(name, errors, label):
    raw = request.form.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        errors.append(f"{label} must be a number.")
        return None


def submitted_choices(info):
    """Form values, limited to what this file contains. Returns (choices, errors)."""
    errors = []
    floats = lambda name: sorted({float(v) for v in request.form.getlist(name)})
    choices = {
        "molecules": [m for m in request.form.getlist("molecule") if m in info.molecules],
        "continuum": [c for c in request.form.getlist("continuum") if c in info.continuum],
        "temperatures": [t for t in floats("temperature") if t in info.temperatures],
        "pressures": [p for p in floats("pressure") if p in info.pressures],
        "R": number("R", errors, "Resolution"),
        "wave_min": number("wave_min", errors, "Minimum wavelength"),
        "wave_max": number("wave_max", errors, "Maximum wavelength"),
        "xscale": "linear" if request.form.get("xscale") == "linear" else "log",
        "yscale": "linear" if request.form.get("yscale") == "linear" else "log",
        "xunit": "wavenumber" if request.form.get("xunit") == "wavenumber" else "micron",
    }
    if choices["wave_min"] is None:
        choices["wave_min"] = round(info.wave_min, 4)
    if choices["wave_max"] is None:
        choices["wave_max"] = round(info.wave_max, 4)

    if not (choices["molecules"] or choices["continuum"]):
        errors.append("Select at least one molecule or continuum species.")
    if not choices["temperatures"]:
        errors.append("Select at least one temperature.")
    if choices["molecules"] and not choices["pressures"]:
        errors.append("Select at least one pressure.")
    if choices["R"] is not None and choices["R"] <= 0:
        errors.append("Resolution must be positive (leave it blank for the native resolution).")
    if choices["wave_min"] <= 0 or choices["wave_min"] >= choices["wave_max"]:
        errors.append("The wavelength range needs 0 < minimum < maximum.")
    elif choices["wave_max"] < info.wave_min or choices["wave_min"] > info.wave_max:
        errors.append(f"This file only covers {info.wave_min:.3g}-{info.wave_max:.3g} μm.")
    return choices, errors


# the data reader fields of the retrieval setup's Observational Data Configuration
DATA_FIELDS = ("coord", "coord_unit", "data", "data_unit", "error")


def data_choices():
    """The observational data form. Kept even when the opacity file changes, since it does not depend on it."""
    choices = {key: request.form.get(f"obs_{key}", "").strip() for key in DATA_FIELDS}
    choices["show"] = request.form.get("show_data") == "on"
    choices["filenames"] = [line.strip() for line in request.form.get("obs_filenames", "").splitlines() if line.strip()]
    return choices


def load_data(obs):
    """Reads the observational data if it was asked for. Returns (observations or None, messages)."""
    if not (obs["show"] and obs["filenames"]):
        return None, []
    missing = [key for key in ("coord", "data", "error") if not obs[key]]
    if missing:
        return None, [("error", f"Enter the {', '.join(missing)} name(s) to read the observational data.")]
    try:
        observations, notes = opacities.read_observations(
            [os.path.expanduser(f) for f in obs["filenames"]], obs["coord"], obs["data"], obs["error"],
            coord_unit=obs["coord_unit"] or None, data_unit=obs["data_unit"] or None)
    except Exception as e:
        return None, [("error", f"Could not read the observational data: {e}")]
    if not observations:
        return None, [("error", "No observational data was read.")]
    return observations, notes


def render_page(db, submitted, obs):
    """The page for opacity file `db`, using the posted form if `submitted`, else the defaults."""
    info, choices, figure, messages = None, None, None, []
    if not db:
        messages.append(("warning", "No default opacity file was found (expected $picaso_refdata/opacities/opacities*.db). "
                                    "Enter the path to a resampled opacity database."))
    elif not os.path.isfile(os.path.expanduser(db)):
        messages.append(("error", f"Opacity file not found: {db}"))
    else:
        try:
            info = opacities.describe(db)
        except Exception as e:
            messages.append(("error", f"Could not read {db} as an opacity database: {e}"))

    if info is not None:
        choices, errors = submitted_choices(info) if submitted else (default_choices(info), [])
        messages += [("error", e) for e in errors]
        if not errors:
            observations, notes = load_data(obs)
            messages += notes
            try:
                fig, notes = opacities.opacity_figure(
                    db, choices["molecules"], choices["continuum"], choices["temperatures"], choices["pressures"],
                    (choices["wave_min"], choices["wave_max"]), R=choices["R"],
                    xscale=choices["xscale"], yscale=choices["yscale"], xunit=choices["xunit"],
                    observations=observations, data_label=obs["data"] or "Data")
                figure = plotly_plot(fig)
                messages += notes
            except Exception as e:
                messages.append(("error", f"Could not plot the opacities: {e}"))

    return render_template("opacities/index.html", db=db, info=info, choices=choices, obs=obs, figure=figure,
                           messages=messages)


@bp.get("/")
def index():
    db = request.args.get("db")
    return render_page(default_db() if db is None else db.strip(), submitted=False, obs=data_choices())


@bp.post("/plot")
def plot():
    db = request.form.get("db", "").strip()
    # a new file path starts from that file's defaults; old selections may not exist in it
    return render_page(db, submitted=db == request.form.get("loaded_db"), obs=data_choices())
