"""
Setup Reference Data page: point the app at the PICASO reference data, check
the environment, and download data products.

PICASO reads `picaso_refdata` when it is imported, so a path set here applies
to this page right away but needs an app restart for the PICASO pages.
"""
import os

from flask import Blueprint, abort, current_app, render_template, request

import picaso.data as data

bp = Blueprint("refdata", __name__, url_prefix="/refdata")


def refdata_is_valid():
    path = current_app.config["PICASO_REFDATA"]
    return bool(path) and os.path.isdir(path)


def data_config():
    """({category: {target: info}}, error message or None)."""
    try:
        _, config = data.get_data_config()
        return config, None
    except Exception as e:
        return {}, f"Error loading data configuration: {e}"


def render_page(message=None):
    downloads, error = data_config() if refdata_is_valid() else ({}, None)
    return render_template(
        "refdata/index.html",
        refdata=current_app.config["PICASO_REFDATA"],
        valid=refdata_is_valid(),
        status_html=data.check_environ(return_html=True) if refdata_is_valid() else None,
        downloads=downloads,
        downloads_error=error,
        cwd=os.getcwd(),
        message=message,
    )


@bp.get("/")
def index():
    return render_page()


@bp.post("/set")
def set_refdata():
    path = request.form.get("path", "").strip()
    if not os.path.isdir(path):
        return render_page(("error", "Please enter a valid directory path."))
    os.environ["picaso_refdata"] = path
    current_app.config["PICASO_REFDATA"] = path
    return render_page(("success", f"picaso_refdata set to {path}. Restart the app for the PICASO pages to use it."))


@bp.post("/shutdown")
def shutdown():
    os._exit(0)


@bp.post("/download")
def download():
    category, target = request.form["category"], request.form["target"]
    info = data_config()[0].get(category, {}).get(target) or abort(400)
    default = info.get("default_destination", "cwd")
    default = os.getcwd() if default == "cwd" else default
    custom = request.form.get("destination") == "custom"
    destination = request.form.get("custom_path", "").strip() if custom else default

    if custom and not os.path.isdir(destination):
        result = ("error", "Invalid destination directory.")
    else:
        try:
            os.makedirs(destination, exist_ok=True)
            data.get_data(category_download=category, target_download=target, final_destination_dir=destination)
            result = ("success", f"Successfully downloaded {target} to {destination}")
        except Exception as e:
            result = ("error", f"Download failed: {e}")
    return render_template("refdata/_download_result.html", result=result)
