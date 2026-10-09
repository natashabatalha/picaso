"""Spectral Diagnostics/IC Theory page (WIP): DOF and Shannon information for different observation cases."""
import pandas as pd
from flask import Blueprint, abort, redirect, render_template, request, url_for
from markupsafe import Markup

from picaso.driver_ui import state
from picaso.driver_ui.core import info_content as ic
from picaso.driver_ui.core.config_schema import Choice, FileInput, MultiChoice, Number, Section, Text
from picaso.driver_ui.core.plots import plotly_plot
from picaso.driver_ui.views import spectrum_session
from picaso.driver_ui.views.cards import Card, card_views, set_messages

bp = Blueprint("info_content", __name__, url_prefix="/info-content")

METHODS = ["Manual", "CSV Upload"]


def new_case(number):
    return {"name": f"Case {number}", "method": "Manual", "min_wave": 0.3, "max_wave": 2.0, "res": 100,
            "error": 0.01, "csv": None, "csv_name": ""}


def page_state(sess):
    s = sess.info_content
    if not s:
        s.update(cases=[new_case(1)], data=None, priors={}, results=None, selected="", plots={},
                 source=None, jacobian_params=[], jacobian_dparam=1e-2)  # source: {config, wno, options} from the Spectrum page
    return s


def case_path(i, *keys):
    return ("info_content", "cases", str(i), *keys)


# =======================================
# CARDS
# =======================================
class CasesCard(Card):
    name, title = "cases", "Observation Cases"
    extra = "info_content/extras/cases.html"

    def visible(self, sess):
        return page_state(sess)["data"] is not None

    def notes(self, sess):
        if page_state(sess)["results"] is None:
            return [("info", "Set up your observation cases and click 'Compute statistics' to continue.")]
        return []

    def fields(self, sess):
        sections = []
        for i, case in enumerate(page_state(sess)["cases"]):
            items = [Text(case_path(i, "name"), case["name"]), Choice(case_path(i, "method"), case["method"], METHODS)]
            if case["method"] == "Manual":
                items += [
                    Number(case_path(i, "min_wave"), case["min_wave"], label="Min wavelength (um)"),
                    Number(case_path(i, "max_wave"), case["max_wave"], label="Max wavelength (um)"),
                    Number(case_path(i, "res"), case["res"], label="Resolution (R)"),
                    Number(case_path(i, "error"), case["error"], label="Spectral error"),
                ]
            else:
                hint = f"Loaded {case['csv_name']}" if case["csv"] is not None else "Columns: wavelength, error"
                items.append(FileInput(case_path(i, "csv_upload"), accept=".csv", label="CSV (wavelength, error)",
                                       hint=hint))
            sections.append(Section(case_path(i), items, label=case["name"]))
        return Section(("cases_card",), sections)

    def update(self, sess, form):
        errors = super().update(sess, form)
        for i, case in enumerate(page_state(sess)["cases"]):
            upload = request.files.get(".".join(case_path(i, "csv_upload")))
            if upload and upload.filename:
                df = pd.read_csv(upload)
                if {"wavelength", "error"} <= set(df.columns):
                    case["csv"], case["csv_name"] = df, upload.filename
                else:
                    errors[".".join(case_path(i, "csv_upload"))] = "CSV must have 'wavelength' and 'error' columns."
        return errors


class DataCard(Card):
    name, title = "data", "Jacobian Initialization"
    extra = "info_content/extras/data.html"

    def fields(self, sess):
        s = page_state(sess)
        if s["source"] is not None:
            return Section(("data_card",), [
                MultiChoice(("info_content", "jacobian_params"), s["jacobian_params"],
                            s["source"]["options"], label="Jacobian parameters", dropdown=True,
                            hint="One extra spectrum is computed per parameter. You can add parameters later; "
                                 "only the new ones are computed."),
                Number(("info_content", "jacobian_dparam"), s["jacobian_dparam"], minimum=1e-6,
                       label="Perturbation size (fraction of each parameter's value)",
                       hint="Grid-based models (e.g. visscher chemistry) may need a larger step to respond.")])
        if s["data"] is not None:
            return None
        return Section(("data_card",), [
            FileInput(("info_content", "npz_upload"), accept=".npz", label="Upload Jacobian data (.npz)",
                      hint="Must contain 'wno', 'jacobian', and 'params'")])

    def notes(self, sess):
        s = page_state(sess)
        data, notes = s["data"], []
        if s["source"] is not None:
            notes.append(("info", "Using the model from the Spectrum & Retrieval Setup page "
                                  f"({s['source']['config']['observation_type']})."))
            if data is None:
                return notes + [("warning", "Select parameters and click 'Compute Jacobian' to begin.")]
            if data["params"] != s["jacobian_params"] or data["d_param"] != s["jacobian_dparam"]:
                notes.append(("warning", "The selection has changed. Click 'Update Jacobian' to apply it."))
        elif data is None:
            return [("warning", Markup(
                "Please initialize the example or provide Jacobian data to begin. Or, set up a starting case on the "
                '<a href="{}">Spectrum &amp; Retrieval Setup</a> page, run the spectrum, and click '
                "'Compute Jacobian &gt;&gt;'.").format(url_for("spectrum.index")))]
        zeros = ic.zero_columns(data)
        if zeros:
            notes.append(("warning", f"The spectrum did not change for {zeros}, so their Jacobian is zero and the "
                                     "statistics cannot be computed. Try a larger perturbation size or remove them."))
        return notes + [("info", f"Jacobian parameters: {data['params']}")]

    def update(self, sess, form):
        s = page_state(sess)
        if s["source"] is not None:
            return super().update(sess, form)
        upload = request.files.get("info_content.npz_upload")
        if upload and upload.filename:
            try:
                set_data(s, ic.load_npz(upload))
                set_messages(s, self.name, [("success", "Data uploaded successfully!")])
            except Exception as e:
                set_messages(s, self.name, [("error", str(e))])
        return {}


class PriorsCard(Card):
    name, title = "priors", "Input Priors for your Parameter Set"

    def visible(self, sess):
        return page_state(sess)["data"] is not None

    def fields(self, sess):
        s = page_state(sess)
        return Section(("priors_card",), [
            Text(("info_content", "priors", param), s["priors"].get(param, ""), hint="Non-zero number")
            for param in s["data"]["params"]])

    def notes(self, sess):
        if page_state(sess)["results"] is None:
            return [("info", "Enter a prior for each parameter.")]
        return []


class ResultsCard(Card):
    name, title = "results", "Results"
    extra = "info_content/extras/results.html"

    def visible(self, sess):
        return page_state(sess)["data"] is not None and page_state(sess)["results"] is not None

    def fields(self, sess):
        s = page_state(sess)
        return Section(("results_card",), [
            Choice(("info_content", "selected"), s["selected"], s["results"]["names"],
                   label="Case for detailed analysis")])

    def after_update(self, sess):
        build_case_plots(page_state(sess))


CARDS = [DataCard(), PriorsCard(), CasesCard(), ResultsCard()]
CARDS_BY_NAME = {card.name: card for card in CARDS}


def set_data(s, data):
    s["data"] = data
    s["plots"]["jacobian"] = plotly_plot(ic.jacobian_figure(data))


def clear_results(s):
    s["results"] = None
    s["plots"] = {"jacobian": s["plots"]["jacobian"]} if "jacobian" in s["plots"] else {}


def build_case_plots(s):
    index = s["results"]["names"].index(s["selected"])
    figures = ic.case_figures(s["results"], s["data"]["params"], s["data"]["wno"], index)
    s["plots"]["case"] = [plotly_plot(fig) for fig in figures]


def render_page(sess, errors=None):
    s = page_state(sess)
    return render_template("_cards.html", endpoint="info_content.update_card", cards=card_views(CARDS, sess, s),
                           errors=errors or {}, sess=sess, s=s)


# =======================================
# ROUTES
# =======================================
@bp.get("/")
def index():
    return render_template("info_content/index.html", page=render_page(state.current()))


@bp.post("/card/<name>")
def update_card(name):
    sess = state.current()
    card = CARDS_BY_NAME.get(name)
    if card is None or not card.visible(sess):
        abort(404)
    errors = card.update(sess, request.form)
    return render_page(sess, errors)


@bp.post("/cases/add")
def add_case():
    sess = state.current()
    cases = page_state(sess)["cases"]
    cases.append(new_case(len(cases) + 1))
    return render_page(sess)


@bp.post("/cases/<int:index>/remove")
def remove_case(index):
    sess = state.current()
    cases = page_state(sess)["cases"]
    if 0 <= index < len(cases):
        cases.pop(index)
    return render_page(sess)


@bp.post("/example")
def example():
    sess = state.current()
    s = page_state(sess)
    try:
        set_data(s, ic.example_jacobian())
    except Exception as e:
        set_messages(s, "data", [("error", f"Could not compute the example Jacobian: {e}")])
    return render_page(sess)


@bp.post("/from-spectrum")
def from_spectrum():
    """'Compute Jacobian >>' on the Spectrum page: links its model here and opens this page."""
    sess = spectrum_session.current()
    spectrum = sess.results.get("spectrum") or abort(404)
    # options come from the config pruned to the selected profiles (no parameters of unused branches)
    options = ic.jacobian_options(spectrum_session.retrieval_base_config(sess))
    s = page_state(sess)
    case = new_case(1)
    case["min_wave"], case["max_wave"] = (round(w, 4) for w in spectrum_session.wave_range(sess))
    case["res"] = sess.ui["resolution"]
    s.update(cases=[case], data=None, priors={}, results=None, plots={},
             source={"config": spectrum_session.model_config(sess), "wno": spectrum["df"]["wavenumber"],
                     "options": options},
             jacobian_params=ic.default_jacobian_params(options))
    return redirect(url_for("info_content.index"))


@bp.post("/compute-jacobian")
def compute_jacobian():
    sess = state.current()
    s = page_state(sess)
    if s["source"] is None:
        abort(404)
    if not s["jacobian_params"]:
        set_messages(s, "data", [("error", "Please select at least one parameter.")])
        return render_page(sess)
    try:
        set_data(s, ic.model_jacobian(s["source"], s["jacobian_params"], s["jacobian_dparam"], s["data"]))
    except Exception as e:
        set_messages(s, "data", [("error", f"Could not compute the Jacobian: {e}")])
        return render_page(sess)
    clear_results(s)
    return render_page(sess)


@bp.post("/reset-data")
def reset_data():
    sess = state.current()
    s = page_state(sess)
    s.update(data=None, priors={}, results=None, plots={}, source=None, jacobian_params=[],
             jacobian_dparam=1e-2)
    return render_page(sess)


@bp.post("/finalize")
def finalize():
    sess = state.current()
    s = page_state(sess)
    params = s["data"]["params"]
    errors, priors = [], []
    for param in params:
        try:
            priors.append(float(s["priors"].get(param, "")))
            if priors[-1] == 0:
                errors.append(f"Each prior must be a non-zero number. Parameter {param} is 0")
        except ValueError:
            errors.append(f"Each prior must be a number. Parameter {param} == '{s['priors'].get(param, '')}' is not")
    if not s["cases"]:
        errors.append("Please add at least one case.")
    errors += [f"Please upload a CSV file for {c['name']}" for c in s["cases"]
               if c["method"] != "Manual" and c["csv"] is None]
    if errors:
        set_messages(s, "cases", [("error", e) for e in errors])
        return render_page(sess)

    try:
        s["results"] = ic.analyze(s["cases"], s["data"], priors)
    except Exception as e:
        set_messages(s, "cases", [("error", f"Could not compute the statistics: {e}")])
        return render_page(sess)
    s["selected"] = s["results"]["names"][0]
    s["plots"]["comparison"] = {name: plotly_plot(fig) for name, fig in
                                ic.comparison_figures(s["results"], params, s["data"]["wno"]).items()}
    build_case_plots(s)
    return render_page(sess)


@bp.post("/reset-results")
def reset_results():
    sess = state.current()
    clear_results(page_state(sess))
    return render_page(sess)
