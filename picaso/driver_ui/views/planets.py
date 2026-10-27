"""
Load a planet page: search the NASA Exoplanet Archive (driver.query_exoplanet_archive)
by name and by ranges of planet and star properties, then send a planet to the
Spectrum or 1D climate page, which fills in its config (core/archive.py).

The search form is a GET, so a search can be bookmarked or reloaded.
"""
import math

from flask import Blueprint, render_template, request

import picaso.driver as go

bp = Blueprint("planets", __name__, url_prefix="/planets")

LIMIT = 50


def submitted_search():
    """(name, {column: (min, max)}, form values to redisplay, errors) from the query string."""
    name = request.args.get("name", "").strip()
    ranges, values, errors = {}, {"name": name}, []
    for column, (description, _) in go.ARCHIVE_COLUMNS.items():
        bounds = []
        for end in ("min", "max"):
            raw = request.args.get(f"{column}_{end}", "").strip()
            values[f"{column}_{end}"] = raw
            try:
                bounds.append(float(raw) if raw else None)
            except ValueError:
                errors.append(f"{description} {end}imum must be a number.")
                bounds.append(None)
        if bounds != [None, None]:
            ranges[column] = tuple(bounds)
    return name, ranges, values, errors


def rows(table):
    """The results as dicts, with missing values as None."""
    return [{key: None if isinstance(value, float) and math.isnan(value) else value
             for key, value in record.items()} for record in table.to_dict("records")]


@bp.get("/")
def index():
    name, ranges, values, messages = submitted_search()
    messages = [("error", e) for e in messages]
    results = None
    searched = bool(name or ranges)
    if searched and not messages:
        try:
            results = rows(go.query_exoplanet_archive(name or None, limit=LIMIT, **ranges))
        except Exception as e:
            messages.append(("error", f"Could not search the NASA Exoplanet Archive: {e}"))
    if results is not None:
        if not results:
            messages.append(("info", "No planets match this search."))
        elif len(results) == LIMIT:
            messages.append(("info", f"Showing the first {LIMIT} matches, alphabetically. "
                                     "Narrow the search to see the others."))
    return render_template("planets/index.html", columns=go.ARCHIVE_COLUMNS, values=values, results=results,
                           searched=searched, messages=messages)
