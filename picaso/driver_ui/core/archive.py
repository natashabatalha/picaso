"""
Loading planets from the NASA Exoplanet Archive into a page's config (Spectrum, 1D climate).

The archive often lacks some values (no mass, no stellar metallicity, ...). Nothing
left over from an earlier planet should then pass for this planet's value, so:
  - missing quantities (radius, mass, distance, ...) are cleared, as is gravity,
    which the archive does not give: the form shows them blank;
  - missing plain numbers (star teff, logg, feh, temperatures) cannot be left blank
    in the form, so they keep their value and the page flags them until edited.
"""
import picaso.driver as go


def load_planet(config, planet_name):
    """
    Writes the archive's values for `planet_name` into `config`, in place. Returns the
    record that `notes` reports from: {"name", "filled": [path], "missing": [(path, value left)]}.
    """
    row = go.find_exoplanet(planet_name)
    prefilled = go.prefill_from_exoplanet_archive(row, config)
    filled, missing = [], []
    for path in go.ARCHIVE_TO_CONFIG:
        parent = _section(config, path[:-1])
        key = path[-1]
        if parent is None or key not in parent:
            continue  # not in this page's config
        new = _section(prefilled, path[:-1])[key]
        if isinstance(parent[key], dict):
            new = new["value"]
        if new != "":
            _set(parent, key, new)
            filled.append(path)
        else:
            if isinstance(parent[key], dict):
                parent[key]["value"] = ""
            missing.append((path, _get(parent, key)))
    gravity = config.get("object", {}).get("gravity")
    if isinstance(gravity, dict) and gravity["value"] != "":
        gravity["value"] = ""
        if not {("object", "mass"), ("object", "radius")} <= set(filled):  # else gravity comes from them
            missing.append((("object", "gravity"), ""))
    return {"name": row["pl_name"], "filled": filled, "missing": missing}


def notes(config, record):
    """Notes for the page: what was loaded, and the values not from this planet that are still unedited."""
    if not record:
        return []
    notes = [("success", f"Loaded {record['name']} from the NASA Exoplanet Archive: "
                         f"{', '.join(_label(p) for p in record['filled']) or 'no values'}.")]
    unedited = [(path, value) for path, value in record["missing"]
                if (parent := _section(config, path[:-1])) is not None and _get(parent, path[-1]) == value]
    cleared = [_label(path) for path, value in unedited if value == ""]
    kept = [f"{_label(path)} = {value:g}" for path, value in unedited if value != ""]
    if cleared:
        notes.append(("warning", f"The archive has no value for {record['name']}'s {', '.join(cleared)}. "
                                 "These were cleared: fill them in if needed."))
    if kept:
        notes.append(("warning", f"The archive has no value for {record['name']}'s {', '.join(kept)}. "
                                 "These are still the earlier values, not this planet's: check them."))
    return notes


def _label(path):
    return ".".join(path)


def _section(config, path):
    for key in path:
        config = config.get(key) if isinstance(config, dict) else None
    return config if isinstance(config, dict) else None


def _get(parent, key):
    value = parent[key]
    return value["value"] if isinstance(value, dict) else value


def _set(parent, key, value):
    if isinstance(parent[key], dict):
        parent[key]["value"] = value
    else:
        parent[key] = value
