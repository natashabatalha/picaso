"""
Retrieval setup: which config values can be free parameters, their priors,
and the [retrieval] section written to the exported TOML.

Free parameters are addressed by dotted paths into the config, as PICASO's
driver expects, e.g. "object.radius.value" or "temperature.knots.T_knots.2".
Data systematics use "<kind>.<data file name>", e.g. "err_inf.wasp39_nirspec".
"""
import ast
import os
import re

SYSTEMATICS = {"err_inf": 0.0, "offset": 0.0, "scaling": 1.0}  # kind -> nominal value
SYSTEMATICS_LABELS = {"err_inf": "Add error inflation term", "offset": "Add instrumental offset",
                      "scaling": "Add scaling term"}
PRIOR_TYPES = ["uniform", "gaussian"]
SAMPLER_CODES = ["dynesty", "ultranest"]
_NOT_FREE = {"nlevel"}


def data_name(filename):
    """Systematics terms are keyed by the data file name without folder or extension."""
    return os.path.splitext(os.path.basename(filename))[0]


def free_parameters(config, path=""):
    """
    {path: current value} for every number in the config that could be retrieved:
    floats, ints (not bools or nlevel), and each element of numeric lists.
    """
    found = {}
    for key, value in config.items():
        key_path = f"{path}.{key}" if path else key
        if isinstance(value, dict):
            found.update(free_parameters(value, key_path))
        elif _is_number(value) and key not in _NOT_FREE:
            found[key_path] = value
        elif isinstance(value, list) and value and all(_is_number(v) for v in value):
            for i, item in enumerate(value):
                found[f"{key_path}.{i}"] = item
    return found


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def find_prior(priors, key):
    """
    Looks up a parameter's prior in a [retrieval] section (from driver.toml or
    an uploaded TOML). Accepts flat ("object.radius") or nested keys, a missing
    trailing ".value", and generic cloud names ("clouds.slab-grey.ptop").
    Systematics without a prior for their data file use the first example of
    their kind (driver.toml has e.g. retrieval.err_inf.filename_example).
    """
    if not priors:
        return None
    candidates = [key, re.sub(r"clouds\.cloud\d+\.", "clouds.", key)]
    candidates += [c[: -len(".value")] for c in candidates if c.endswith(".value")]
    for candidate in candidates:
        if candidate in priors:
            return priors[candidate]
        node = priors
        for part in candidate.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        if _is_prior(node):
            return node
    kind = key.split(".")[0]
    if kind in SYSTEMATICS:
        return next((p for p in priors.get(kind, {}).values() if _is_prior(p)), None)
    return None


def _is_prior(node):
    return isinstance(node, dict) and any(k in node for k in ("prior", "type", "uniform_kwargs", "gaussian_kwargs"))


def default_prior(value, config_prior=None):
    """
    UI prior settings for a newly selected parameter: from the config's prior if
    there is one, otherwise a uniform +/-25% range around the current value.
    """
    p = config_prior or {}
    uniform = p.get("uniform_kwargs") or p.get("uniform_options") or {}
    gaussian = p.get("gaussian_kwargs") or p.get("gaussian_options") or {}
    prior = p.get("prior") or p.get("type") or "uniform"
    low, high = sorted((value * 0.75, value * 1.25))
    return {
        "prior": prior if prior in PRIOR_TYPES else "uniform",
        "log": p.get("log") in (True, "True", "true"),
        "min": float(uniform.get("min", p.get("min", low))),
        "max": float(uniform.get("max", p.get("max", high))),
        "mean": float(gaussian.get("mean", p.get("mean", value))),
        "std": float(gaussian.get("std", p.get("std", 1.0))),
    }


def to_driver_prior(ui_prior):
    """UI prior settings -> the {prior, log, <prior>_kwargs} form PICASO's driver reads."""
    kind = ui_prior["prior"]
    kwargs = {"min": ui_prior["min"], "max": ui_prior["max"]} if kind == "uniform" else \
             {"mean": ui_prior["mean"], "std": ui_prior["std"]}
    return {"prior": kind, "log": ui_prior["log"], f"{kind}_kwargs": kwargs}


def driver_priors(ui_priors):
    """{path: driver prior} for every selected parameter, in selection order."""
    return {key: to_driver_prior(p) for key, p in ui_priors.items()}


def parse_kwargs(text):
    """Parses a python dict literal typed by the user, e.g. "{'live_points': 700}"."""
    value = ast.literal_eval(text.strip() or "{}")
    if not isinstance(value, dict):
        raise ValueError("must be a dictionary, e.g. {'live_points': 700}")
    return value


def retrieval_section(priors, sampler):
    """
    The [retrieval] TOML section: priors nested by their dotted path, plus sampler options.

    Parameters
    ----------
    priors : dict
        {path: driver prior}
    sampler : dict
        {code, sampler_kwargs, run_kwargs}
    """
    section = {"sampler": dict(sampler)}
    for key, prior in priors.items():
        node = section
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = prior
    return section
