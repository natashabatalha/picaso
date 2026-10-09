"""
Session state for the Spectrum & Retrieval Setup page: starting a session,
uploads, and the configs derived from the session for runs and export.
"""
import math

import numpy as np
from flask import current_app

import picaso.driver as go

from picaso.driver_ui import state
from picaso.driver_ui.core import free_chemistry, resources, retrieval_setup
from picaso.driver_ui.core.config_ops import export_config, merge_config, new_config, resolve_defaults


def current():
    """The session, with the Spectrum page state initialized on first use."""
    sess = state.current()
    if sess.config is None:
        reset(sess)
    return sess


def reset(sess):
    sess.config = new_config(current_app.config["PICASO_REFDATA"])
    free_chemistry.normalize(sess.config["chemistry"]["free"])
    sess.ui = {
        "include_clouds": False,
        "phase": None,  # phase angle (radian) for reflected light; None keeps the config's value
        "wave_min": None,  # None = full range of the opacity database
        "wave_max": None,
        "resolution": 150,
        "retrieval": False,
        "systematics": {kind: {} for kind in retrieval_setup.SYSTEMATICS},  # kind -> {data name: bool}
        "free": {},  # parameter path -> selected
        "priors": {},  # parameter path -> prior settings
        "nsamples": 5,
        "sampler": sampler_defaults(sess.config),
    }
    sess.results = {}


def sampler_defaults(config):
    sampler = config.get("retrieval", {}).get("sampler", {})
    code = sampler.get("code")
    return {
        "code": code if code in retrieval_setup.SAMPLER_CODES else retrieval_setup.SAMPLER_CODES[0],
        "sampler_kwargs": repr(sampler.get("sampler_kwargs", {})),
        "run_kwargs": repr(sampler.get("run_kwargs", {})),
    }


def apply_upload(sess, uploaded):
    """Uses an uploaded TOML's values as the new defaults (on top of the current config)."""
    merge_config(sess.config, uploaded)
    sess.config = resolve_defaults(sess.config, current_app.config["PICASO_REFDATA"])
    free_chemistry.normalize(sess.config["chemistry"]["free"])
    sess.ui["include_clouds"] = "clouds" in uploaded  # an uploaded TOML without [clouds] means no clouds
    sess.ui["phase"] = None
    sess.ui["sampler"] = sampler_defaults(sess.config)
    sess.results.pop("previews", None)


# =======================================
# DERIVED CONFIGS
# =======================================
def is_reflected(config):
    return go.OBSERVATION_CALC_MAP.get(config["observation_type"]) == "reflected"


def _export(sess, retrieval=None, prune=False):
    config = export_config(sess.config, include_clouds=sess.ui["include_clouds"], retrieval=retrieval, prune=prune)
    if is_reflected(config) and sess.ui["phase"] is not None:
        config["geometry"]["phase"]["value"] = sess.ui["phase"]  # the phase input only applies to reflected light
    return config


def model_config(sess):
    """Clean config for running PICASO (no retrieval section)."""
    return _export(sess)


def retrieval_base_config(sess):
    """Model config pruned to the selected options: the space free parameters are chosen from."""
    return _export(sess, prune=True)


def exported_config(sess):
    """The config offered for download, including the retrieval section when retrievals are on."""
    if not sess.ui["retrieval"]:
        return model_config(sess)
    return _export(sess, retrieval=retrieval_section(sess), prune=True)


def selected_parameters(sess):
    """{path: nominal value} of selected free parameters: data systematics first, then model parameters."""
    selected = {f"{kind}.{name}": retrieval_setup.SYSTEMATICS[kind]
                for kind, names in sess.ui["systematics"].items() for name, on in names.items() if on}
    available = retrieval_setup.free_parameters(retrieval_base_config(sess))
    selected.update({key: value for key, value in available.items() if sess.ui["free"].get(key)})
    return selected


def retrieval_priors(sess):
    """{path: driver prior} for the selected parameters."""
    return retrieval_setup.driver_priors({key: sess.ui["priors"][key] for key in selected_parameters(sess)})


def retrieval_section(sess):
    sampler = {"code": sess.ui["sampler"]["code"]}
    for key in ("sampler_kwargs", "run_kwargs"):
        try:
            sampler[key] = retrieval_setup.parse_kwargs(sess.ui["sampler"][key])
        except (ValueError, SyntaxError):
            sampler[key] = {}  # the sampler card shows the parse error
    return retrieval_setup.retrieval_section(retrieval_priors(sess), sampler)


# =======================================
# OPACITY-DEPENDENT SETTINGS
# =======================================
def opacity(sess):
    """(opacity connection, None) or (None, error message)."""
    try:
        return resources.opacity(sess.config["OpticalProperties"]), None
    except Exception as e:
        return None, f"Could not load the opacity database: {e}"


def wave_bounds(opacity):
    """Wavelength range (um) of the opacity database, rounded inward to 4 decimals for display."""
    wavelength = 1e4 / opacity.wno
    return math.ceil(np.min(wavelength) * 1e4) / 1e4, math.floor(np.max(wavelength) * 1e4) / 1e4


def wave_range(sess):
    """Wavelength range (um) for spectrum plots."""
    opa, _ = opacity(sess)
    low, high = wave_bounds(opa) if opa is not None else (None, None)
    return (sess.ui["wave_min"] if sess.ui["wave_min"] is not None else low,
            sess.ui["wave_max"] if sess.ui["wave_max"] is not None else high)
