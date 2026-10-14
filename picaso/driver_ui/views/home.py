import os

from flask import Blueprint, current_app, render_template

from picaso.driver_ui.core.config_ops import driver_template_path
from picaso.driver_ui.navigation import Group, Page

bp = Blueprint("home", __name__)

GET_STARTED = (
    Group("Setup", "", (
        Page("/refdata/", "Check environment", "Locate reference data and see what PICASO can find.",
             visual="card_refdata.svg"),
        Page("/refdata/#download", "Download reference data", "Get opacities, stellar grids and other data."),
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
        Page("/info-content/", "Spectral diagnostics and information theory",
             "Rank observations by information content.",
             wip=True, visual="card_info_content.svg"),
    )),
    Group("Explore", "", (
        Page("/opacities/", "View opacities", "Plot molecular and continuum cross sections.",
             visual="card_opacities.svg"),
    )),
)


# Home page card illustration for each top-bar group (navigation.GROUPS); the cards list the group's pages.
HOME_VISUALS = {
    "Setup": "card_refdata.svg",
    "Model": "card_transmission.svg",
    "Analyze": "card_fit.svg",  # a model spectrum against data: analysis of models and observations
    "Explore": "card_opacities.svg",
}


def refdata_ready():
    return os.path.isfile(driver_template_path(current_app.config["PICASO_REFDATA"]))


@bp.get("/")
def index():
    return render_template("home.html", visuals=HOME_VISUALS, refdata_ready=refdata_ready())


@bp.get("/get-started")
def get_started():
    return render_template("get_started.html", groups=GET_STARTED, refdata_ready=refdata_ready())


@bp.get("/landing-page")
def landing():
    return render_template("landing.html", refdata_ready=refdata_ready())
