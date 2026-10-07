"""Loading, merging, cleaning and exporting driver.toml configs."""
import copy
import glob
import json
import os
import tomllib

import numpy as np


# =======================================
# LOADING
# =======================================
def driver_template_path(refdata):
    return os.path.join(refdata, "input_tomls", "driver.toml")


def climate_template_path(refdata):
    return os.path.join(refdata, "input_tomls", "climate.toml")


def new_climate_config(refdata):
    """The reference climate.toml that seeds the climate page, with `_default_` paths resolved."""
    with open(climate_template_path(refdata), "rb") as f:
        return resolve_climate_defaults(tomllib.load(f), refdata)


def resolve_climate_defaults(config, refdata):
    """Like resolve_defaults, but climate opacity files are ck tables, so they are not swapped for the resampled database."""
    return _replace_in_strings(config, "_default_", refdata)


def new_config(refdata):
    """The reference driver.toml that seeds every new UI session, with `_default_` paths resolved."""
    with open(driver_template_path(refdata), "rb") as f:
        return resolve_defaults(tomllib.load(f), refdata)


def resolve_defaults(config, refdata):
    """
    Replaces the `_default_` placeholder in paths with the reference data directory.
    The opacity file follows the same naming lookup as justdoit.opannection().
    """
    optical = config.get("OpticalProperties", {})
    if "_default_" in optical.get("opacity_file", ""):
        optical["opacity_file"] = default_opacity_file(refdata)
    return _replace_in_strings(config, "_default_", refdata)


def default_opacity_file(refdata):
    with open(os.path.join(refdata, "config.json")) as f:
        pattern = json.load(f)["opacities"]["files"]["opacity"]
    matches = glob.glob(os.path.join(refdata, pattern))
    return matches[0] if matches else os.path.join(refdata, "opacities", "opacities.db")


def merge_config(base, overrides):
    """Recursively writes `overrides` (e.g. an uploaded TOML) over `base`, in place."""
    for key, value in overrides.items():
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            merge_config(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def _replace_in_strings(data, old, new):
    if isinstance(data, dict):
        return {k: _replace_in_strings(v, old, new) for k, v in data.items()}
    if isinstance(data, list):
        return [_replace_in_strings(v, old, new) for v in data]
    if isinstance(data, str):
        return data.replace(old, new)
    return data


# =======================================
# EXPORT
# =======================================
def is_irradiated(config):
    """Only thermal emission can be computed without a host star."""
    return config["observation_type"] != "thermal" or config.get("irradiated", False)


def export_config(config, include_clouds=True, retrieval=None, prune=False):
    """
    The config as PICASO should receive it: options stripped, and sections the
    user switched off removed. The UI keeps those in its own state so toggling
    back restores the user's values.

    Parameters
    ----------
    config : dict
        The UI's working config
    include_clouds : bool
        Whether the user wants clouds
    retrieval : dict
        Retrieval section built by the UI (replaces any retrieval section in config)
    prune : bool
        Keep only the selected temperature/chemistry/cloud options (used for retrievals)
    """
    config = clean_dictionary(config)
    config["irradiated"] = is_irradiated(config)
    if not config["irradiated"]:
        config.pop("star", None)
    if not include_clouds:
        config.pop("clouds", None)
    config.pop("retrieval", None)

    chemistry = config.get("chemistry", {})
    if chemistry.get("method") == "xarray_grid":
        # chemistry is read from the same grid (and grid point) as the temperature profile
        molecules = chemistry.get("xarray_grid", {}).get("grid_kwargs", {}).get("molecules", "all")
        chemistry["xarray_grid"] = copy.deepcopy(config["temperature"].get("xarray_grid", {}))
        chemistry["xarray_grid"].setdefault("grid_kwargs", {})["molecules"] = molecules

    if prune:
        config = prune_to_selected(config)
    if retrieval:
        config["retrieval"] = retrieval
    return config


def export_climate_config(config):
    """The climate config as driver.run_climate should receive it: options stripped, star removed unless irradiated."""
    config = clean_dictionary(config)
    if not config.get("irradiated", False):
        config.pop("star", None)
    return config


def prune_to_selected(config):
    """Drops the temperature profiles, chemistry methods, free-chemistry parameters and cloud types that are not selected."""
    config = copy.deepcopy(config)
    temperature = config["temperature"]
    profile = temperature["profile"]
    config["temperature"] = {profile: temperature[profile], "pressure": temperature["pressure"], "profile": profile}

    chemistry = config["chemistry"]
    method = chemistry["method"]
    config["chemistry"] = {method: chemistry[method], "method": method}
    if method == "free":
        free = config["chemistry"]["free"]
        profile_options = free.get("profile_options", {})
        for mol in free.get("species", []):
            if mol in free and free[mol].get("profile") in profile_options:
                allowed = profile_options[free[mol]["profile"]]
                keep = ["profile", "unit"] + (allowed if isinstance(allowed, list) else [])
                free[mol] = {k: v for k, v in free[mol].items() if k in keep}

    if "clouds" in config:
        clouds = {}
        for key, cloud_type in config["clouds"].items():
            if key.endswith("_type"):
                cloud_id = key[: -len("_type")]
                clouds[key] = cloud_type
                clouds[cloud_id] = {cloud_type: config["clouds"][cloud_id][cloud_type]}
        config["clouds"] = clouds
    return config


def clean_dictionary(data, suffix="_options"):
    """
    Recursively removes keys ending in `suffix` (UI-only option lists that PICASO
    does not expect) and converts numpy types to native python types for TOML serialization.
    """
    if isinstance(data, dict):
        return {k: clean_dictionary(v, suffix) for k, v in data.items() if not k.endswith(suffix)}
    if isinstance(data, list):
        return [clean_dictionary(v, suffix) for v in data]
    if isinstance(data, (np.floating, np.integer, np.str_, np.bool_)):
        return data.item()
    if isinstance(data, np.ndarray):
        return data.tolist()
    return data


def format_references(references):
    """Formats driver.references() output (one DOI list per section) as a plain-text file."""
    section_titles = {"temperature": "Temperature Profile", "chemistry": "Chemistry", "clouds": "Clouds"}
    lines = ["PICASO References", "=================="]
    for section, dois in references.items():
        if dois:
            lines += ["", section_titles.get(section, section.capitalize())]
            lines += [f"  https://doi.org/{doi}" for doi in dois]
    if len(lines) == 2:
        lines += ["", "No citations found for the selected parameterizations."]
    return "\n".join(lines)
