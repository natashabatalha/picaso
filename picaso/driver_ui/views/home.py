import os

from flask import Blueprint, current_app, render_template

from picaso.driver_ui.core.config_ops import driver_template_path

bp = Blueprint("home", __name__)


@bp.get("/")
def index():
    refdata = current_app.config["PICASO_REFDATA"]
    return render_template("home.html", refdata_ready=os.path.isfile(driver_template_path(refdata)))
