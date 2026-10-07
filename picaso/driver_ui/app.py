"""
Flask version of the PICASO driver UI.

Run locally with:
    python -m picaso.driver_ui.app
"""
import os
import traceback

import plotly
from flask import Flask, render_template, send_file

from picaso.driver_ui import navigation
from picaso.driver_ui.core.config_ops import driver_template_path
from picaso.driver_ui.views import home, opacities, refdata

# pages that import PICASO, which needs the reference data at import time
PICASO_PAGES = ("/spectrum/", "/analysis/", "/info-content/")


def create_app(refdata_dir=None):
    app = Flask(__name__)
    # sessions only hold an id into server memory, which is lost on restart anyway
    app.secret_key = os.urandom(24)
    app.config["PICASO_REFDATA"] = refdata_dir or os.environ.get("picaso_refdata", "")
    # picaso reads these when it is imported, so set them once at startup
    os.environ["picaso_refdata"] = app.config["PICASO_REFDATA"]
    os.environ.setdefault("PYSYN_CDBS", os.path.join(app.config["PICASO_REFDATA"], "grp", "redcat", "trds"))

    app.register_blueprint(home.bp)
    app.register_blueprint(refdata.bp)
    # reads the opacity database directly, so it works without importing PICASO
    app.register_blueprint(opacities.bp)
    register_picaso_pages(app)

    @app.context_processor
    def nav_context():
        return {"nav_groups": navigation.GROUPS, "nav_resources": navigation.RESOURCES,
                "nav_active": navigation.is_active}

    @app.get("/vendor/plotly.min.js")
    def plotly_js():
        path = os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js")
        return send_file(path, max_age=86400)

    return app


def register_picaso_pages(app):
    """Registers the PICASO pages, or a placeholder explaining why they are unavailable."""
    problem = None
    if not os.path.isfile(driver_template_path(app.config["PICASO_REFDATA"])):
        problem = "missing_refdata"
    else:
        try:
            from picaso.driver_ui.views import analysis, info_content, spectrum
        except Exception:
            problem = traceback.format_exc()

    if problem is None:
        for view in (spectrum, analysis, info_content):
            app.register_blueprint(view.bp)
        return

    def unavailable(path=""):
        return render_template("unavailable.html", refdata=app.config["PICASO_REFDATA"], problem=problem), 503

    for prefix in PICASO_PAGES:
        app.add_url_rule(prefix, f"unavailable{prefix}", unavailable)
        app.add_url_rule(f"{prefix}<path:path>", f"unavailable{prefix}path", unavailable, methods=["GET", "POST"])


if __name__ == "__main__":
    create_app().run(debug=True)
