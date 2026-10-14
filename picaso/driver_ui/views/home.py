import os

from flask import Blueprint, current_app, render_template

from picaso.driver_ui.core.config_ops import driver_template_path
from picaso.driver_ui.navigation import Group, Page

bp = Blueprint("home", __name__)

GET_STARTED = (
    Group("Setup", "", (
        Page("/refdata/", "Set up reference data", "Locate reference data and download opacities.",
             visual="card_refdata.svg"),
    )),
    Group("Model", "", (
        Page("/spectrum/", "Run a spectrum", "Compute spectra and set up retrievals.",
             visual="card_transmission.svg"),
        Page("/climate/", "Calculate 1D climate", "Solve for a radiative-convective P-T profile.",
             visual="card_climate.svg"),
    )),
    Group("Analyze", "", (
        Page("/analysis/", "Analyze a retrieval", "Corner plots, best-fit models, banded spectra.",
             visual="card_fit.svg"),
        Page("/info-content/", "Compare observations", "Rank observations by information content.",
             wip=True, visual="card_info_content.svg"),
    )),
    Group("Explore", "", (
        Page("/opacities/", "View opacities", "Plot molecular and continuum cross sections.",
             visual="card_opacities.svg"),
    )),
)


def refdata_ready():
    return os.path.isfile(driver_template_path(current_app.config["PICASO_REFDATA"]))


@bp.get("/")
def index():
    return render_template("home.html")


@bp.get("/get-started")
def get_started():
    return render_template("get_started.html", groups=GET_STARTED, refdata_ready=refdata_ready())


@bp.get("/landing-page")
def landing():
    return render_template("landing.html", refdata_ready=refdata_ready())
