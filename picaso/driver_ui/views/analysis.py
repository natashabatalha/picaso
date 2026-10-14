"""Retrieval Analysis page: corner plots, max log-likelihood model, bands, and an export package."""
import os

import matplotlib.pyplot as plt
import numpy as np
import toml
from flask import Blueprint, Response, abort, current_app, render_template, request

from picaso.driver_ui import state
from picaso.driver_ui.core import analysis
from picaso.driver_ui.core.config_schema import Bool, Choice, MultiChoice, Number, Section, Text
from picaso.driver_ui.core.plots import matplotlib_bytes, matplotlib_image, plotly_plot
from picaso.driver_ui.views.cards import Card, card_views, set_messages

bp = Blueprint("analysis", __name__, url_prefix="/analysis")

IMAGE_FORMATS = {"png": "image/png", "pdf": "application/pdf", "svg": "image/svg+xml"}
EXPORT_DETAILS = {"spectrum_tag": "transit_depth", "spectrum_unit": "cm**2/cm**2", "author": "", "contact": "",
                  "code": "PICASO", "model_description": ""}


def page_state(sess):
    a = sess.analysis
    if not a:
        a.update(config=None, retrieval_dir="", loaded=None, corner={}, corner_export={"dpi": 300, "format": "png"},
                 corner_plot=None, corner_point={}, corner_median={}, corner_model=None, corner_model_error=None,
                 corner_median_model=None,
                 bands_n=10, bands_selected=[], bands=None, export=dict(EXPORT_DETAILS),
                 attributes=[{"key": "", "value": ""}], zip=None, contributions_on=False, contribution_options=[],
                 contributions_selected=[], contributions_plot=None)
    return a


def is_loaded(sess):
    a = page_state(sess)
    return a["loaded"] is not None and a["loaded"]["dir"] == a["retrieval_dir"]


# =======================================
# CARDS
# =======================================
class UploadCard(Card):
    name, title = "upload", "Driver configuration"
    extra = "analysis/extras/upload.html"


class DirectoryCard(Card):
    name, title = "directory", "Retrieval output"
    extra = "analysis/extras/load.html"

    def visible(self, sess):
        return page_state(sess)["config"] is not None

    def fields(self, sess):
        return Section(("directory_card",), [
            Text(("analysis", "retrieval_dir"), page_state(sess)["retrieval_dir"], label="Retrieval output directory")])

    def notes(self, sess):
        a = page_state(sess)
        notes = []
        if not a["config"].get("InputOutput", {}).get("retrieval_output"):
            notes.append(("warning", "The uploaded config does not define retrieval_output in [InputOutput]. "
                                     "Please specify the directory below."))
        if a["retrieval_dir"] and not os.path.exists(a["retrieval_dir"]):
            notes.append(("error", f"The path '{a['retrieval_dir']}' does not exist on this machine."))
        if not a["config"].get("retrieval"):
            notes.append(("error", "No 'retrieval' section found in the configuration."))
        else:
            notes.append(("info", f"Parameters in retrieval: {analysis.parameters(a['config'])}"))
        return notes


def ready_to_load(sess):
    a = page_state(sess)
    return bool(a["retrieval_dir"]) and os.path.exists(a["retrieval_dir"]) and bool(a["config"].get("retrieval"))


class CornerCard(Card):
    name, title = "corner", "Corner plot"
    extra = "analysis/extras/corner.html"

    def visible(self, sess):
        return is_loaded(sess)

    def export_fields(self, sess):
        """Rendered by the extra template below the plot (labels are edited on the plot itself, see corner.js)."""
        export = page_state(sess)["corner_export"]
        return Section(("corner_export",), [
            Number(("analysis", "corner_export", "dpi"), export["dpi"], minimum=100, maximum=1200,
                   label="Resolution (DPI)"),
            Choice(("analysis", "corner_export", "format"), export["format"], list(IMAGE_FORMATS),
                   label="Image format"),
        ], label="Export high resolution corner plot")

    def update(self, sess, form):
        return sess.apply(self.export_fields(sess), form)

    def notes(self, sess):
        a = page_state(sess)
        return [("error", error) for error in (a.get("corner_error"), a.get("corner_model_error")) if error]


class MaxLoglCard(Card):
    name, title = "max_logl", "Max log likelihood model vs data"
    extra = "analysis/extras/max_logl.html"

    def visible(self, sess):
        return is_loaded(sess)

    def contribution_fields(self, sess):
        """Rendered by the extra template below the plot rather than as the card's form above it."""
        a = page_state(sess)
        children = [Bool(("analysis", "contributions_on"), a["contributions_on"],
                         label="Do you also want the individual species contribution (leave-one-out) for this model?")]
        if a["contributions_on"]:
            children.append(MultiChoice(
                ("analysis", "contributions_selected"), a["contributions_selected"], a["contribution_options"],
                label="Molecules to leave out",
                hint="One spectrum is computed per molecule, with that molecule's opacity removed."))
        return Section(("contributions_card",), children)

    def update(self, sess, form):
        return sess.apply(self.contribution_fields(sess), form)


class BandsCard(Card):
    name, title = "bands", "Generate banded profiles and spectra"
    extra = "analysis/extras/bands.html"

    def visible(self, sess):
        return is_loaded(sess)

    def fields(self, sess):
        a = page_state(sess)
        return Section(("bands_card",), [
            Number(("analysis", "bands_n"), a["bands_n"], minimum=10, maximum=1000, label="Number of samples"),
            MultiChoice(("analysis", "bands_selected"), a["bands_selected"],
                        analysis.band_options(a["loaded"]["out"]), label="Generate bands for"),
        ])


class ExportCard(Card):
    name, title = "export", "Export & download retrieval results"
    extra = "analysis/extras/export.html"

    def visible(self, sess):
        return is_loaded(sess) and page_state(sess)["bands"] is not None

    def fields(self, sess):
        a = page_state(sess)
        details = [Text(("analysis", "export", key), value, multiline=(key == "model_description"))
                   for key, value in a["export"].items()]
        attributes = [Section(("analysis", "attributes", str(i)), [
            Text(("analysis", "attributes", str(i), "key"), row["key"], label="Attribute key"),
            Text(("analysis", "attributes", str(i), "value"), row["value"], label="Attribute value"),
        ], label=f"Attribute {i + 1}") for i, row in enumerate(a["attributes"])]
        return Section(("export_card",), [
            *details,
            Section(("export_attributes",), attributes,
                    label="Custom attributes (additional metadata embedded in the xarray NetCDF file)"),
        ])

    def notes(self, sess):
        return [("info", "Generate and download a package of retrieval results including the xarray dataset, "
                         "sample pickle, standard plots, the driver TOML and a quick look README.md with DOI info.")]


CARDS = [UploadCard(), DirectoryCard(), CornerCard(), MaxLoglCard(), BandsCard(), ExportCard()]
CARDS_BY_NAME = {card.name: card for card in CARDS}


def render_corner(sess):
    a = page_state(sess)
    try:
        fig = analysis.interactive_corner_figure(a["loaded"]["info"], a["corner"], a["corner_point"],
                                                 a["loaded"]["out"], a["corner_model"], a["config"]["observation_type"],
                                                 a["corner_median_model"])
        a["corner_plot"] = plotly_plot(fig, config=analysis.CORNER_PLOT_CONFIG)
        a["corner_error"], a["corner_stale"] = None, False
    except Exception as e:
        a["corner_plot"], a["corner_error"] = None, f"Could not draw the corner plot: {e}"


def select_corner_point(sess, point):
    """Moves the corner plot's black lines to `point` (see analysis.median_point) and computes the model there."""
    a = page_state(sess)
    a["corner_point"] = point
    try:
        a["corner_model"] = analysis.evaluate_point(a["config"], a["loaded"]["info"], point)
        a["corner_model_error"] = None
    except Exception as e:
        a["corner_model"], a["corner_model_error"] = None, f"Could not compute the model at the selected values: {e}"


def render_page(sess, errors=None):
    a = page_state(sess)
    return render_template("_cards.html", endpoint="analysis.update_card", cards=card_views(CARDS, sess, a),
                           errors=errors or {}, sess=sess, a=a, ready_to_load=ready_to_load(sess) if a["config"] else False,
                           example_available=os.path.isfile(analysis.example_path(current_app.config["PICASO_REFDATA"])))


# =======================================
# ROUTES
# =======================================
@bp.get("/")
def index():
    sess = state.current()
    # the page moves the lines and renames parameters on its copy of the figure; a fresh page needs them drawn in
    if page_state(sess).get("corner_stale") and is_loaded(sess):
        render_corner(sess)
    return render_template("analysis/index.html", page=render_page(sess))


@bp.post("/card/<name>")
def update_card(name):
    sess = state.current()
    card = CARDS_BY_NAME.get(name)
    if card is None or not card.visible(sess):
        abort(404)
    errors = card.update(sess, request.form)
    return render_page(sess, errors)


@bp.post("/upload")
def upload():
    sess = state.current()
    a = page_state(sess)
    try:
        text = request.files["toml"].read().decode("utf-8")
        a["config"], a["config_text"] = toml.loads(text), text
    except Exception as e:
        set_messages(a, "upload", [("error", f"Could not parse the retrieval TOML file: {e}")])
    else:
        a["retrieval_dir"] = a["config"].get("InputOutput", {}).get("retrieval_output", "")
        a["loaded"] = None
        set_messages(a, "upload", [("success", "Successfully loaded configuration file!")])
    return render_page(sess)


@bp.post("/example")
def example():
    sess = state.current()
    a = page_state(sess)
    try:
        a["config"], a["config_text"] = analysis.example_config(current_app.config["PICASO_REFDATA"])
    except Exception as e:
        set_messages(a, "upload", [("error", f"Could not load the example retrieval: {e}")])
        return render_page(sess)
    a["retrieval_dir"] = a["config"]["InputOutput"]["retrieval_output"]
    a["loaded"] = None
    set_messages(a, "upload", [("success", "Loaded the example: a synthetic WASP-39b-like transit (R=100, 30 ppm) "
                                           "retrieving isothermal T, H2O, CO2 and radius. Truths: T=900 K, "
                                           "H2O=1e-3, CO2=1e-4, radius=1.27 Rjup.")])
    return load_retrieval(sess)


@bp.post("/load")
def load():
    sess = state.current()
    if not ready_to_load(sess):
        abort(400)
    return load_retrieval(sess)


def load_retrieval(sess):
    a = page_state(sess)
    try:
        loaded = analysis.load(a["config"], a["retrieval_dir"])
    except Exception as e:
        set_messages(a, "directory", [("error", f"Could not read the retrieval: {e}")])
        return render_page(sess)
    a["loaded"] = {"dir": a["retrieval_dir"], "info": loaded["info"], "out": loaded["out"],
                   "plot": plotly_plot(loaded["figure"])}
    a["corner"] = analysis.default_corner_settings(loaded["info"])
    a["corner_median"] = analysis.median_point(loaded["info"])
    select_corner_point(sess, dict(a["corner_median"]))
    a["corner_median_model"] = a["corner_model"]
    a["bands_selected"] = analysis.band_options(loaded["out"])[:2]
    a["bands"] = a["zip"] = None
    a["contribution_options"], a["contributions_selected"] = analysis.contribution_options(a["config"], loaded["out"])
    a["contributions_plot"] = None
    render_corner(sess)
    messages = [("success", "Successfully loaded retrieval outputs!")]
    info = loaded["info"]
    if not info.get("converged", True):
        messages.append(("warning", f"This retrieval has not converged. Results are intermediate, from {info['niter']} "
                                    f"iterations with an effective sample size of {info['ess']:.0f}, and will change "
                                    "as the run continues."))
    set_messages(a, "directory", messages)
    return render_page(sess)


@bp.post("/save-corner")
def save_corner():
    sess = state.current()
    a = page_state(sess)
    if a["loaded"] is None or a.get("corner_plot") is None:
        abort(404)
    export = a["corner_export"]
    path = os.path.join(a["retrieval_dir"], f"corner_plot_manuscript.{export['format']}")
    fig = analysis.corner_figure(a["loaded"]["info"], a["corner"])
    fig.savefig(path, format=export["format"], dpi=export["dpi"], bbox_inches="tight")
    plt.close(fig)
    set_messages(a, "corner", [("success", f"Successfully saved figure locally at: {path}")])
    return render_page(sess)


@bp.post("/corner/point")
def corner_point():
    """Called by static/corner.js when the user moves a black line: JSON {point: {parameter: value}}."""
    sess = state.current()
    a = page_state(sess)
    if not is_loaded(sess):
        abort(404)
    try:
        requested = (request.get_json(silent=True) or {})["point"]
        point = {param: float(requested[param]) for param in a["loaded"]["info"]["param_names"]}
        if not np.all(np.isfinite(list(point.values()))):
            raise ValueError("values must be finite")
    except (KeyError, TypeError, ValueError) as e:
        return {"error": f"Invalid point: {e}"}, 400
    select_corner_point(sess, point)
    a["corner_stale"] = True
    if a["corner_model"] is None:
        return {"error": a["corner_model_error"]}, 500
    wavelength, model = analysis.model_spectrum(a["corner_model"])
    return {"point": point, "wavelength": wavelength.tolist(), "model": model.tolist(),
            "title": analysis.corner_spectrum_title(a["corner_model"])}


@bp.post("/corner/label")
def corner_label():
    """Called by static/corner.js when the user renames a parameter: JSON {param, label}."""
    sess = state.current()
    a = page_state(sess)
    if not is_loaded(sess):
        abort(404)
    body = request.get_json(silent=True) or {}
    param, label = body.get("param"), str(body.get("label", "")).strip()
    if param not in a["corner"] or not label:
        return {"error": "Invalid parameter label"}, 400
    a["corner"][param]["label"] = label
    a["corner_stale"] = True
    return {"param": param, "label": label}


@bp.post("/contributions")
def generate_contributions():
    sess = state.current()
    a = page_state(sess)
    if not is_loaded(sess):
        abort(404)
    if not a["contributions_selected"]:
        set_messages(a, "max_logl", [("warning", "Select at least one molecule to leave out.")])
        return render_page(sess)
    try:
        excluded = analysis.leave_one_out(a["config"], a["loaded"]["info"], a["contributions_selected"])
        fig = analysis.max_logl_figure(a["loaded"]["out"], a["config"]["observation_type"],
                                       {f"No {mol}": out for mol, out in excluded.items()})
        a["contributions_plot"] = plotly_plot(fig.update_layout(title="Individual Species Contribution (Leave-One-Out)"))
    except Exception as e:
        set_messages(a, "max_logl", [("error", f"Could not compute species contributions: {e}")])
    return render_page(sess)


@bp.post("/bands")
def generate_bands():
    sess = state.current()
    a = page_state(sess)
    try:
        a["bands"] = analysis.bands(a["config"], a["loaded"]["info"], a["bands_n"], a["bands_selected"])
        a["band_images"] = [matplotlib_image(fig) for fig in analysis.band_figures(a["bands"])]
        a["zip"] = None
    except Exception as e:
        set_messages(a, "bands", [("error", f"Could not generate bands: {e}")])
    return render_page(sess)


@bp.post("/add-attribute")
def add_attribute():
    sess = state.current()
    page_state(sess)["attributes"].append({"key": "", "value": ""})
    return render_page(sess)


@bp.post("/export")
def export():
    sess = state.current()
    a = page_state(sess)
    attributes = {row["key"].strip(): row["value"].strip() for row in a["attributes"] if row["key"].strip()}
    try:
        a["zip"] = analysis.export_package(a["bands"], a["loaded"]["info"], a["export"], attributes,
                                           config=a["config"], config_text=a.get("config_text"))
        set_messages(a, "export", [("success", "Successfully generated the export package!")])
    except Exception as e:
        set_messages(a, "export", [("error", f"Error generating export package: {e}")])
    return render_page(sess)


@bp.get("/download/corner")
def download_corner():
    a = page_state(state.current())
    if a["loaded"] is None or a.get("corner_plot") is None:
        abort(404)
    export = a["corner_export"]
    fig = analysis.corner_figure(a["loaded"]["info"], a["corner"])
    data = matplotlib_bytes(fig, export["format"], export["dpi"])
    plt.close(fig)
    return Response(data, mimetype=IMAGE_FORMATS[export["format"]],
                    headers={"Content-Disposition": f"attachment; filename=corner_plot.{export['format']}"})


@bp.get("/download/retrieval_results_export.zip")
def download_zip():
    a = page_state(state.current())
    if a["zip"] is None:
        abort(404)
    return Response(a["zip"], mimetype="application/zip",
                    headers={"Content-Disposition": "attachment; filename=retrieval_results_export.zip"})
