"""Information Content page (WIP): DOF and Shannon information for different observation cases."""
import pandas as pd
from flask import Blueprint, abort, render_template, request

from picaso.driver_ui import state
from picaso.driver_ui.core import info_content as ic
from picaso.driver_ui.core.config_schema import Choice, FileInput, Number, Section, Text
from picaso.driver_ui.core.plots import plotly_plot
from picaso.driver_ui.views.cards import Card, card_views, set_messages

bp = Blueprint("info_content", __name__, url_prefix="/info-content")

METHODS = ["Manual", "CSV Upload"]


def new_case(number):
    return {"name": f"Case {number}", "method": "Manual", "min_wave": 0.3, "max_wave": 2.0, "res": 100,
            "error": 0.01, "csv": None, "csv_name": ""}


def page_state(sess):
    s = sess.info_content
    if not s:
        s.update(cases=[new_case(1)], data=None, priors={}, results=None, selected="", plots={})
    return s


def case_path(i, *keys):
    return ("info_content", "cases", str(i), *keys)


# =======================================
# CARDS
# =======================================
class CasesCard(Card):
    name, title = "cases", "Observation Cases"
    extra = "info_content/extras/cases.html"

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
    name, title = "data", "Data Initialization"
    extra = "info_content/extras/data.html"

    def fields(self, sess):
        if page_state(sess)["data"] is not None:
            return None
        return Section(("data_card",), [
            FileInput(("info_content", "npz_upload"), accept=".npz", label="Upload Jacobian data (.npz)",
                      hint="Must contain 'wno', 'jacobian', and 'params'")])

    def notes(self, sess):
        data = page_state(sess)["data"]
        if data is None:
            return [("warning", "Please initialize the example or provide Jacobian data to begin.")]
        return [("info", f"Jacobian parameters: {data['params']}")]

    def update(self, sess, form):
        s = page_state(sess)
        upload = request.files.get("info_content.npz_upload")
        if upload and upload.filename:
            try:
                s["data"] = ic.load_npz(upload)
                set_messages(s, self.name, [("success", "Data uploaded successfully!")])
            except Exception as e:
                set_messages(s, self.name, [("error", str(e))])
        return {}


class PriorsCard(Card):
    name, title = "priors", "Input Priors for your Parameter Set"
    extra = "info_content/extras/priors.html"

    def visible(self, sess):
        return page_state(sess)["data"] is not None

    def fields(self, sess):
        s = page_state(sess)
        return Section(("priors_card",), [
            Text(("info_content", "priors", param), s["priors"].get(param, ""), hint="Non-zero number")
            for param in s["data"]["params"]])

    def notes(self, sess):
        if page_state(sess)["results"] is None:
            return [("info", "Enter a prior for each parameter and click 'Finalize priors' to continue.")]
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


CARDS = [CasesCard(), DataCard(), PriorsCard(), ResultsCard()]
CARDS_BY_NAME = {card.name: card for card in CARDS}


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
        s["data"] = ic.example_jacobian()
    except Exception as e:
        set_messages(s, "data", [("error", f"Could not compute the example Jacobian: {e}")])
    return render_page(sess)


@bp.post("/reset-data")
def reset_data():
    sess = state.current()
    s = page_state(sess)
    s.update(data=None, priors={}, results=None, plots={})
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
        set_messages(s, "priors", [("error", e) for e in errors])
        return render_page(sess)

    try:
        s["results"] = ic.analyze(s["cases"], s["data"], priors)
    except Exception as e:
        set_messages(s, "priors", [("error", f"Could not compute the statistics: {e}")])
        return render_page(sess)
    s["selected"] = s["results"]["names"][0]
    s["plots"]["comparison"] = {name: plotly_plot(fig) for name, fig in
                                ic.comparison_figures(s["results"], params, s["data"]["wno"]).items()}
    build_case_plots(s)
    return render_page(sess)


@bp.post("/reset-results")
def reset_results():
    sess = state.current()
    s = page_state(sess)
    s.update(results=None, plots={})
    return render_page(sess)
