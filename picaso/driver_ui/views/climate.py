"""
1D Climate Calculations page.

Mirrors the Spectrum & Retrieval Setup page (a list of cards, re-rendered and
morphed in place after every change), but edits a climate.toml config kept in
the session's own `climate` page state, so it never touches the Spectrum
page's driver config. Field paths are prefixed with ("climate", "config").
"""
import os
import threading
import tomllib

import toml
from flask import Blueprint, Response, abort, current_app, render_template, request

import picaso.driver as go
from picaso.driver_ui import state
from picaso.driver_ui.core import climate
from picaso.driver_ui.core.config_ops import (
    climate_template_path, export_climate_config, merge_config, new_climate_config, resolve_climate_defaults)
from picaso.driver_ui.core.config_schema import apply_form, build_fields, find_field
from picaso.climate_log import ConvergenceHistory
from picaso.driver_ui.core.plots import Plot, plotly_plot
from picaso.driver_ui.views.cards import Card, bulk_property_fields, bulk_property_notes, card_views, set_messages
from picaso.driver_ui.views.spectrum import learn_more

bp = Blueprint("climate", __name__, url_prefix="/climate")

CONFIG = ("climate", "config")
SELECTORS = {
    CONFIG + ("star",): ("type", None),  # options read from star.type_options
    CONFIG + ("temperature",): ("profile", go.climate_pt_options),
    CONFIG + ("chemistry",): ("method", go.climate_chem_options),
}


def page_state(sess):
    c = sess.climate
    if not c:
        reset(sess)
    return c


def reset(sess):
    sess.climate.clear()
    sess.climate.update(config=new_climate_config(current_app.config["PICASO_REFDATA"]), results={})
    # older reference climate.toml files do not have this UI option yet
    sess.climate["config"].setdefault("climate", {}).setdefault("watch_convergence", False)


def config_of(sess):
    return page_state(sess)["config"]


def fields_for(sess, section):
    return build_fields(config_of(sess)[section], CONFIG + (section,), SELECTORS)


def model_config(sess):
    """Clean config for driver.run_climate."""
    return export_climate_config(config_of(sess))


def opacity_error(sess):
    try:
        climate.opacity(config_of(sess)["OpticalProperties"])
    except Exception as e:
        return f"Could not load the correlated-k opacities: {e}"
    return None


# =======================================
# CARDS
# =======================================
class ClimatePageCard(Card):
    """
    Applies forms to the climate config only. Session.apply would deep-copy all
    page state on every change, including the stored climate run.
    """
    def update(self, sess, form):
        root, errors = apply_form({"climate": {"config": config_of(sess)}}, self.fields(sess), form)
        page_state(sess)["config"] = root["climate"]["config"]
        self.after_update(sess)
        return errors


class AdminCard(ClimatePageCard):
    name, title = "optical", "Administrative"

    def fields(self, sess):
        return fields_for(sess, "OpticalProperties")

    def notes(self, sess):
        notes = [("info", f"Reference data: {current_app.config['PICASO_REFDATA']}"),
                 ("info", "Climate runs use correlated-k opacities. For preweighted, the opacity file is a "
                          "preweighted ck table (its chemistry is fixed). For resortrebin, it is the directory "
                          "of by-molecule ck tables (leave empty for the default).")]
        error = opacity_error(sess)
        return notes + ([("error", error)] if error else [])


class UploadCard(ClimatePageCard):
    name, title = "upload", "Pre-fill from a TOML file"
    extra = "climate/extras/upload.html"


class SetupCard(ClimatePageCard):
    name, title = "setup", "Calculation"

    def fields(self, sess):
        tree = build_fields({"irradiated": config_of(sess)["irradiated"]}, CONFIG)
        find_field(tree, ".".join(CONFIG + ("irradiated",))).hint = \
            "Irradiated planets need star properties; brown dwarfs do not"
        return tree

    def notes(self, sess):
        if config_of(sess).get("calc_type") != "climate":
            return [("warning", "The uploaded TOML does not have calc_type set to climate. Climate setup proceeds, "
                                "but spectra and retrievals should be set up on the Spectrum page.")]
        return []


class StarCard(ClimatePageCard):
    name, title = "star", "Star variables"

    def visible(self, sess):
        return config_of(sess).get("irradiated", False)

    def fields(self, sess):
        return fields_for(sess, "star")


class ObjectCard(ClimatePageCard):
    name, title = "object", "Object variables"

    def fields(self, sess):
        tree = bulk_property_fields(fields_for(sess, "object"), CONFIG + ("object",))
        teff = find_field(tree, ".".join(CONFIG + ("object", "teff")))
        if teff is not None:
            teff.hint = "Effective temperature for brown dwarfs, intrinsic temperature for irradiated planets"
        return tree

    def notes(self, sess):
        return bulk_property_notes(config_of(sess)["object"])


class TemperatureCard(ClimatePageCard):
    name, title = "temperature", "Initial pressure & temperature guess"
    extra = "climate/extras/preview_pt.html"

    def fields(self, sess):
        tree = fields_for(sess, "temperature")
        profile = config_of(sess)["temperature"]["profile"]
        find_field(tree, ".".join(CONFIG + ("temperature", "profile"))).links = learn_more("pt", profile)
        return tree

    def notes(self, sess):
        return [("info", "The climate model iterates from this guess on its pressure grid. Configure pressure "
                         "(ignored for a userfile or sonora bobcat guess, which bring their own pressure grid).")]


class ChemistryCard(ClimatePageCard):
    name, title = "chemistry", "Chemistry"

    def fields(self, sess):
        tree = fields_for(sess, "chemistry")
        method = config_of(sess)["chemistry"]["method"]
        if method != "preweighted":
            find_field(tree, ".".join(CONFIG + ("chemistry", "method"))).links = learn_more("chem", method)
        return tree

    def notes(self, sess):
        config = config_of(sess)
        method, opacity_method = config["chemistry"]["method"], config["OpticalProperties"]["opacity_method"]
        if method == "preweighted" and opacity_method != "preweighted":
            return [("error", "Preweighted chemistry requires the preweighted opacity method.")]
        if method != "preweighted" and opacity_method != "resortrebin":
            return [("error", f"{method} chemistry is computed on the fly and requires the resortrebin opacity "
                              "method. Preweighted ck tables already include their chemistry.")]
        return []


class ClimateCard(ClimatePageCard):
    name, title = "climate", "Climate model"
    extra = "climate/extras/run.html"
    clears_previews = False

    def fields(self, sess):
        tree = fields_for(sess, "climate")
        watch = find_field(tree, ".".join(CONFIG + ("climate", "watch_convergence")))
        if watch is not None:
            watch.hint = "Draw the temperature profile and the solver log live while the model runs"
        return tree

    def notes(self, sess):
        return [("info", "rcb_guess is the level index at the top of the guessed convective zone. rfacv sets the "
                         "stellar flux in the energy balance: 0 for brown dwarfs, 0.5 for full day-night "
                         "redistribution, 1 for the dayside.")]


class ConfigurationCard(ClimatePageCard):
    name, title = "configuration", "Configuration"
    extra = "climate/extras/configuration.html"


CARDS = [UploadCard(), AdminCard(), SetupCard(), StarCard(), ObjectCard(), TemperatureCard(),
         ChemistryCard(), ClimateCard(), ConfigurationCard()]
CARDS_BY_NAME = {card.name: card for card in CARDS}


# =======================================
# RENDERING
# =======================================
def live_view(sess):
    """What the live panel of a watched climate run needs, or None if no run is being watched."""
    live = page_state(sess)["results"].get("live")
    if live is None:
        return None
    history = live["history"]
    n = len(history.dts)
    stage = history.stages[-1][1] if history.stages else "Setting up"
    # the figure id only changes when there is something new to draw, so polling does not redraw it
    figure = climate.live_figure(history)
    figure.update_layout(width=None, autosize=True)
    return {"plot": Plot(f"live-{id(history)}-{n}-{len(history.stages)}", figure.to_json()),
            "status": f"{stage}: {n} outer iterations so far",
            "log": "\n".join(history.log[-40:])}


def render_page(sess, errors=None):
    return render_template(
        "_cards.html",
        endpoint="climate.update_card",
        cards=card_views(CARDS, sess, page_state(sess)["results"]),
        errors=errors or {},
        sess=sess,
        climate_state=page_state(sess),
        live=live_view(sess),
        exported_toml=toml.dumps(model_config(sess)),
    )


# =======================================
# ROUTES
# =======================================
@bp.get("/")
def index():
    path = climate_template_path(current_app.config["PICASO_REFDATA"])
    if not os.path.isfile(path):
        return render_template("climate/index.html", missing_template=path), 503
    return render_template("climate/index.html", page=render_page(state.current()))


@bp.post("/card/<name>")
def update_card(name):
    sess = state.current()
    card = CARDS_BY_NAME.get(name)
    if card is None or not card.visible(sess):
        abort(404)
    errors = card.update(sess, request.form)
    if card.clears_previews:
        page_state(sess)["results"].pop("preview", None)
    return render_page(sess, errors)


@bp.post("/upload")
def upload():
    sess = state.current()
    c = page_state(sess)
    file = request.files.get("toml")
    try:
        merge_config(c["config"], tomllib.load(file.stream))
        c["config"] = resolve_climate_defaults(c["config"], current_app.config["PICASO_REFDATA"])
        c["results"].pop("preview", None)
        set_messages(c["results"], "upload", [("success", "Successfully loaded user-provided default values.")])
    except (tomllib.TOMLDecodeError, UnicodeDecodeError, AttributeError) as e:
        set_messages(c["results"], "upload", [("error", f"Could not read the TOML file: {e}")])
    return render_page(sess)


@bp.post("/preview")
def preview():
    sess = state.current()
    try:
        result = {"plot": plotly_plot(climate.guess_figure(model_config(sess)))}
    except Exception as e:
        result = {"warning": "Make sure you have configured the opacities, object, initial guess and chemistry.",
                  "error": str(e)}
    page_state(sess)["results"]["preview"] = result
    return render_page(sess)


def store_run(sess, run_results=None, error=None):
    """Puts a finished climate run (or its error) on the page."""
    results = page_state(sess)["results"]
    if error is not None:
        results.pop("run", None)
        set_messages(results, "climate", [
            ("warning", "Make sure you have configured the opacities, object, initial guess and chemistry before "
                        "running the climate model."),
            ("error", str(error))])
        return
    run_results["plots"] = [plotly_plot(fig) for fig in climate.result_figures(run_results)]
    run_results["with_spec"] = model_config(sess).get("climate", {}).get("with_spec", False)
    results["run"] = run_results
    converged = run_results["out"].get("converged", 1)
    set_messages(results, "climate", [("success", "Climate model finished.") if converged else
                                      ("warning", "Climate model finished but did not converge. "
                                                  "See the solver log in the server console.")])


def watch_run(live, config):
    """Runs the climate model in a background thread, feeding `live['history']` as it goes."""
    try:
        live["run_results"] = climate.run_climate(config, watch=live["history"])
    except Exception as e:
        live["error"] = e
    live["done"] = True


@bp.post("/run")
def run():
    sess = state.current()
    results = page_state(sess)["results"]
    config = model_config(sess)
    if results.get("live") is not None:
        set_messages(results, "climate", [("warning", "A climate model is already running.")])
        return render_page(sess)
    if config.get("climate", {}).get("watch_convergence", False):
        live = {"history": ConvergenceHistory(), "done": False, "run_results": None, "error": None}
        results["live"] = live
        results.pop("run", None)
        threading.Thread(target=watch_run, args=(live, config), daemon=True).start()
        return render_page(sess)
    try:
        run_results = climate.run_climate(config)
    except Exception as e:
        store_run(sess, error=e)
    else:
        store_run(sess, run_results)
    return render_page(sess)


@bp.get("/progress")
def progress():
    """Polled by the live panel of a watched run. Once the run is done, re-renders the whole page."""
    sess = state.current()
    results = page_state(sess)["results"]
    live = results.get("live")
    if live is not None and not live["done"]:
        return render_template("climate/extras/live.html", live=live_view(sess))
    if live is not None:
        results.pop("live")
        store_run(sess, live["run_results"], live["error"])
    return Response(render_page(sess), headers={"HX-Retarget": "#sections", "HX-Reswap": "morph:outerHTML"})


@bp.post("/reset")
def reset_all():
    sess = state.current()
    reset(sess)
    return render_page(sess)


# =======================================
# DOWNLOADS
# =======================================
def attachment(data, filename, mimetype):
    return Response(data, mimetype=mimetype, headers={"Content-Disposition": f"attachment; filename={filename}"})


@bp.get("/download/climate.toml")
def download_config():
    return attachment(toml.dumps(model_config(state.current())), "climate.toml", "application/toml")


@bp.get("/download/picaso_climate.nc")
def download_netcdf():
    results = page_state(state.current())["results"].get("run")
    if results is None or not results["with_spec"]:
        abort(404)
    return attachment(climate.climate_netcdf(results), "picaso_climate.nc", "application/x-netcdf")


@bp.get("/download/picaso_climate_pt.csv")
def download_csv():
    results = page_state(state.current())["results"].get("run") or abort(404)
    return attachment(climate.climate_csv(results), "picaso_climate_pt.csv", "text/csv")
