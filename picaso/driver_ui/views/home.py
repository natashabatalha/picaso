import os

from flask import Blueprint, current_app, render_template

from picaso.driver_ui.core.config_ops import driver_template_path

bp = Blueprint("home", __name__)

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
