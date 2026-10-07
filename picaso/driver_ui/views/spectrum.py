"""
Spectrum & Retrieval Setup page.

The page is a list of cards (see views/cards.py). Changing any input posts its
card's form, the session is updated, and the whole page is re-rendered and
morphed in place, so dependent cards (star, priors, ...) update immediately.
Long-running actions (previews, model runs, prior sampling) are separate
buttons; their results are stored in the session and shown until replaced.
"""
import math
import os
import tomllib

import numpy as np
import toml
from flask import Blueprint, Response, abort, current_app, render_template, request

import picaso.driver as go
from picaso.references import get_citations
from picaso.driver_ui.core import clouds, free_chemistry, resources, runs
from picaso.driver_ui.core.config_ops import format_references, is_irradiated
from picaso.driver_ui.core.config_schema import (
    Bool, MultiChoice, Number, Section, Text, build_fields, find_field, replace_field)
from picaso.driver_ui.core.plots import plotly_plot
from picaso.driver_ui.views import retrieval
from picaso.driver_ui.views.cards import Card, card_views, set_messages
from picaso.driver_ui.views.spectrum_session import (
    apply_upload, current, exported_config, is_reflected, model_config, opacity, reset, wave_bounds, wave_range)

bp = Blueprint("spectrum", __name__, url_prefix="/spectrum")

SELECTORS = {
    ("star",): ("type", None),  # options read from star.type_options
    ("temperature",): ("profile", go.pt_options),
    ("chemistry",): ("method", go.chem_options),
}


def learn_more(prefix, name):
    """'Learn more' DOI links for a cited pt_/chem_/cloud_ parameterization."""
    dois = get_citations(f"{prefix}_{name}")
    if len(dois) == 1:
        return [("Learn more", f"https://doi.org/{dois[0]}")]
    return [(f"Learn more #{i}", f"https://doi.org/{doi}") for i, doi in enumerate(dois, start=1)]


# =======================================
# CARDS
# =======================================
class AdminCard(Card):
    name, title = "optical", "Administrative"

    def fields(self, sess):
        return build_fields(sess.config["OpticalProperties"], ("OpticalProperties",),
                            choices={("OpticalProperties", "opacity_method"): ["resampled"]})

    def notes(self, sess):
        notes = [("info", f"Reference data: {current_app.config['PICASO_REFDATA']}")]
        _, error = opacity(sess)
        return notes + ([("error", error)] if error else [])


class UploadCard(Card):
    name, title = "upload", "Default Values from a TOML File"
    extra = "spectrum/extras/upload.html"


class SetupCard(Card):
    name, title = "setup", "Calculation"

    def fields(self, sess):
        config = sess.config
        keys = ["observation_type", "observation_type_options"]
        if config["observation_type"] == "thermal":
            keys.append("irradiated")
        tree = build_fields({k: config[k] for k in keys})
        if "irradiated" in keys:
            find_field(tree, "irradiated").hint = "Irradiated objects need star properties"
        return tree

    def notes(self, sess):
        if sess.config.get("calc_type") == "climate":
            return [("warning", "The uploaded driver.toml has calc_type set to climate, which should be run on the "
                                "climate page. Spectrum setup proceeds, but may have issues if the full setup has "
                                "not been provided.")]
        return []


class StarCard(Card):
    name, title = "star", "Star Variables"

    def visible(self, sess):
        return is_irradiated(sess.config)

    def fields(self, sess):
        return build_fields(sess.config["star"], ("star",), SELECTORS)


class ObjectCard(Card):
    name, title = "object", "Object Variables"

    def fields(self, sess):
        return build_fields(sess.config["object"], ("object",))


class GeometryCard(Card):
    name, title = "geometry", "Phase Angle"

    def visible(self, sess):
        return is_reflected(sess.config)

    def fields(self, sess):
        phase = sess.ui["phase"]
        if phase is None:
            phase = sess.config["geometry"]["phase"]["value"]
        return Section(("geometry_card",), [
            Number(("ui", "phase"), phase, minimum=0, maximum=2 * math.pi, label="Phase angle (radian)")])


class TemperatureCard(Card):
    name, title = "temperature", "Pressure & Temperature"
    extra = "spectrum/extras/preview_pt.html"

    def fields(self, sess):
        tree = build_fields(sess.config["temperature"], ("temperature",), SELECTORS,
                            custom={("temperature", "xarray_grid")})
        profile = sess.config["temperature"]["profile"]
        find_field(tree, "temperature.profile").links = learn_more("pt", profile)
        if profile == "xarray_grid":
            replace_field(tree, "temperature.xarray_grid", self._xarray_grid_fields(sess))
        return tree

    def notes(self, sess):
        notes = [("info", "Configure pressure (can be ignored when using a userfile, sonora bobcat, or a custom xarray for temperature)")]
        if sess.config["temperature"]["profile"] == "xarray_grid":
            _, error = xarray_grid_tools(sess.config)
            notes += [("error", error)] if error else []
        return notes

    def _xarray_grid_fields(self, sess):
        path = ("temperature", "xarray_grid")
        grid = sess.config["temperature"]["xarray_grid"]
        children = [Text(path + ("model_dir",), str(grid["model_dir"]))]
        tools, _ = xarray_grid_tools(sess.config)
        if tools is not None:
            grid_kwargs = grid.get("grid_kwargs", {})
            for name, values in resources.grid_parameters(tools).items():
                low, high = float(np.min(values)), float(np.max(values))
                value = grid_kwargs.get(name)
                value = float(value) if value is not None and low <= value <= high else low
                children.append(Number(path + ("grid_kwargs", name), value, minimum=low, maximum=high))
        return Section(path, children)


def xarray_grid_tools(config):
    """(param tools loaded with the temperature xarray grid, None) or (None, error message)."""
    model_dir = str(config["temperature"]["xarray_grid"]["model_dir"])
    if not os.path.isdir(model_dir):
        return None, "Enter a valid path to an xarray model grid to use this option"
    try:
        return resources.param_tools_for(config), None
    except Exception as e:
        return None, f"Could not load the model grid: {e}"


class ChemistryCard(Card):
    name, title = "chemistry", "Chemistry"
    extra = "spectrum/extras/preview_chemistry.html"

    def fields(self, sess):
        chemistry = sess.config["chemistry"]
        method = chemistry["method"]
        tree = build_fields(chemistry, ("chemistry",), SELECTORS,
                            custom={("chemistry", "free"), ("chemistry", "xarray_grid")})
        find_field(tree, "chemistry.method").links = learn_more("chem", method)
        if method == "free":
            opa, _ = opacity(sess)
            molecules = resources.possible_molecules(opa) if opa is not None else []
            replace_field(tree, "chemistry.free", free_chemistry.fields(chemistry["free"], molecules))
        elif method == "xarray_grid":
            replace_field(tree, "chemistry.xarray_grid", self._xarray_fields(sess))
        return tree

    def _xarray_fields(self, sess):
        path = ("chemistry", "xarray_grid")
        tools, error = xarray_grid_tools(sess.config)
        if error or sess.config["temperature"]["profile"] != "xarray_grid":
            return Section(path)
        species = list(tools.species)
        current = sess.config["chemistry"].get("xarray_grid", {}).get("grid_kwargs", {}).get("molecules")
        if not isinstance(current, list):
            opa, _ = opacity(sess)
            available = resources.possible_molecules(opa) if opa is not None else []
            current = [mol for mol in available if mol in species]
        return Section(path, [MultiChoice(path + ("grid_kwargs", "molecules"), current, species,
                                          hint="Subset of molecules to include in spectra")])

    def notes(self, sess):
        method = sess.config["chemistry"]["method"]
        if method == "free":
            return [("error", e) for e in free_chemistry.errors(sess.config["chemistry"]["free"])]
        if method == "xarray_grid":
            _, error = xarray_grid_tools(sess.config)
            if error or sess.config["temperature"]["profile"] != "xarray_grid":
                return [("error", "Using an xarray grid for chemistry requires selecting the grid parameters "
                                  "through the xarray_grid temperature profile, for consistency")]
        return []

    def after_update(self, sess):
        if sess.config["chemistry"]["method"] == "free":
            free_chemistry.normalize(sess.config["chemistry"]["free"])


class CloudsCard(Card):
    name, title = "clouds", "Clouds"
    extra = "spectrum/extras/preview_clouds.html"

    def fields(self, sess):
        children = [Bool(("ui", "include_clouds"), sess.ui["include_clouds"], label="Do you want clouds?")]
        if sess.ui["include_clouds"]:
            config_clouds = sess.config["clouds"]
            children.append(Number(("ui", "num_clouds"), len(clouds.cloud_ids(config_clouds)), minimum=1,
                                   label="How many cloud types?"))
            for cloud in clouds.fields(config_clouds).children:
                selector = cloud.children[0]
                selector.links = learn_more("cloud", selector.value)
                children.append(cloud)
        return Section(("clouds_card",), children)

    def after_update(self, sess):
        count = sess.ui.pop("num_clouds", None)
        if sess.ui["include_clouds"] and count:
            clouds.resize(sess.config["clouds"], count)


class SpectrumCard(Card):
    name, title = "spectrum", "Spectrum"
    extra = "spectrum/extras/run.html"
    clears_previews = False

    def fields(self, sess):
        opa, _ = opacity(sess)
        if opa is None:
            return None
        low, high = wave_bounds(opa)
        current_low, current_high = wave_range(sess)
        return Section(("spectrum_card",), [
            Number(("ui", "wave_min"), current_low, minimum=low, maximum=high, hint="Wavelength range minimum (μm)"),
            Number(("ui", "wave_max"), current_high, minimum=low, maximum=high, hint="Wavelength range maximum (μm)"),
            Number(("ui", "resolution"), sess.ui["resolution"], minimum=10, hint="Spectral resolution"),
        ])

    def notes(self, sess):
        _, error = opacity(sess)
        return [("error", error)] if error else []

    def after_update(self, sess):
        refresh_spectrum_plot(sess)


class ConfigurationCard(Card):
    name, title = "configuration", "Configuration"
    extra = "spectrum/extras/configuration.html"


CARDS = [AdminCard(), UploadCard(), SetupCard(), StarCard(), ObjectCard(), GeometryCard(),
         TemperatureCard(), ChemistryCard(), CloudsCard(), SpectrumCard(),
         *retrieval.CARDS, ConfigurationCard()]
CARDS_BY_NAME = {card.name: card for card in CARDS}


# =======================================
# RENDERING
# =======================================
def render_page(sess, errors=None):
    return render_template(
        "_cards.html",
        endpoint="spectrum.update_card",
        cards=card_views(CARDS, sess, sess.results),
        errors=errors or {},
        sess=sess,
        exported_toml=toml.dumps(exported_config(sess)),
    )


def refresh_spectrum_plot(sess):
    results = sess.results.get("spectrum")
    if results is not None:
        fig = runs.spectrum_figure(results["wavenumber"], results["flux"], wave_range(sess))
        sess.results["spectrum_plot"] = plotly_plot(fig)


# =======================================
# ROUTES
# =======================================
@bp.get("/")
def index():
    return render_template("spectrum/index.html", page=render_page(current()))


@bp.post("/card/<name>")
def update_card(name):
    sess = current()
    card = CARDS_BY_NAME.get(name)
    if card is None or not card.visible(sess):
        abort(404)
    errors = card.update(sess, request.form)
    if card.clears_previews:
        sess.results.pop("previews", None)
    return render_page(sess, errors)


@bp.post("/upload")
def upload():
    sess = current()
    file = request.files.get("toml")
    try:
        apply_upload(sess, tomllib.load(file.stream))
        set_messages(sess.results, "upload", [("success", "Successfully loaded user-provided default values.")])
    except (tomllib.TOMLDecodeError, UnicodeDecodeError, AttributeError) as e:
        set_messages(sess.results, "upload", [("error", f"Could not read the TOML file: {e}")])
    return render_page(sess)


PREVIEWS = {
    "pt": runs.pt_figure,
    "chemistry": runs.mixing_ratio_figure,
    "clouds": runs.cloud_figure,
}


@bp.post("/preview/<kind>")
def preview(kind):
    sess = current()
    if kind not in PREVIEWS:
        abort(404)
    try:
        fig = PREVIEWS[kind](model_config(sess))
        result = {"plot": plotly_plot(fig)} if fig is not None else \
                 {"warning": "No cloud profile generated. Please check the cloud configuration."}
    except Exception as e:
        result = {"warning": "Make sure you have configured temperature and chemistry.", "error": str(e)}
    sess.results.setdefault("previews", {})[kind] = result
    return render_page(sess)


@bp.post("/run")
def run():
    sess = current()
    try:
        results = runs.run_spectrum(model_config(sess), sess.ui["resolution"])
    except Exception as e:
        set_messages(sess.results, "spectrum", [
            ("warning", "Make sure you have configured temperature, pressure, and chemistry before running a spectrum."),
            ("error", str(e))])
    else:
        sess.results["spectrum"] = results
        refresh_spectrum_plot(sess)
        set_messages(sess.results, "spectrum", [("warning", w) for w in results["warnings"]])
    return render_page(sess)


@bp.post("/retrieval/<action>")
def retrieval_action(action):
    sess = current()
    if action not in retrieval.ACTIONS:
        abort(404)
    retrieval.ACTIONS[action](sess)
    return render_page(sess)


@bp.post("/reset")
def reset_all():
    sess = current()
    reset(sess)
    return render_page(sess)


# =======================================
# DOWNLOADS
# =======================================
def attachment(data, filename, mimetype):
    return Response(data, mimetype=mimetype, headers={"Content-Disposition": f"attachment; filename={filename}"})


@bp.get("/download/configured_toml.toml")
def download_config():
    return attachment(toml.dumps(exported_config(current())), "configured_toml.toml", "application/toml")


@bp.get("/download/references.txt")
def download_references():
    references = go.references(driver_dict=exported_config(current()))
    return attachment(format_references(references), "references.txt", "text/plain")


@bp.get("/download/picaso_spectrum.nc")
def download_netcdf():
    results = current().results.get("spectrum") or abort(404)
    return attachment(runs.spectrum_netcdf(results), "picaso_spectrum.nc", "application/x-netcdf")


@bp.get("/download/picaso_spectrum.csv")
def download_csv():
    results = current().results.get("spectrum") or abort(404)
    return attachment(runs.spectrum_csv(results), "picaso_spectrum.csv", "text/csv")
