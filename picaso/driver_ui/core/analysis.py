"""Retrieval Analysis computations: reading retrieval output, corner plots, contributions, bands and export packages."""
import copy
import io
import os
import tempfile
import zipfile

import numpy as np

import picaso.driver as go
from picaso import WIP_justplotit as jpi
from picaso import retrieval as ret
from picaso.driver_ui.core import resources
from picaso.driver_ui.core.plots import label_spectrum_yaxis


def parameters(config):
    """Free parameter paths in the config's [retrieval] section, in order."""
    return list(go.prior_finder(config.get("retrieval", {})))


def load(config, retrieval_dir):
    """Reads the samples and evaluates the max log-likelihood model against the data."""
    info = ret.read_retrievals(retrieval_dir, go.prior_finder(config.get("retrieval", {})))
    out = go.check_model_samples(config, N=1, samples=np.atleast_2d(info["max_logl_point"]), full_likelihood=True)
    return {"info": info, "out": out, "chi2": out["chi_sq_per_pt"][0],
            "figure": max_logl_figure(out, config["observation_type"])}


def max_logl_figure(out, observation_type, others=None):
    """
    Max log-likelihood model and data, plus optional extra models (e.g. leave-one-out spectra).

    Parameters
    ----------
    observation_type : str
        config["observation_type"], which sets the y axis units
    others : dict
        {legend label: check_model_samples(full_likelihood=True) output}
    """
    others = others or {}
    models = {"Max LogL Model": out, **others}
    # data can be stitched from several instruments out of order, which would draw the model lines back and forth
    order = {label: np.argsort(np.asarray(m["xdata"])) for label, m in models.items()}
    fig = jpi.spectrum([np.asarray(m["xdata"])[order[label]] for label, m in models.items()],
                       [np.asarray(m["ymodel"][0])[order[label]] for label, m in models.items()],
                       legend=[f"{label} (Chi-sq = {m['chi_sq_per_pt'][0]:.2f})" if others else label
                               for label, m in models.items()], backend="plotly")
    fig = jpi.plot_errorbar(1e4 / out["xdata"], out["ydata"][0], out["edata"][0], plot=fig, backend="plotly")
    fig.update_layout(title=f"Chi-sq = {out['chi_sq_per_pt'][0]:.2f}")
    return label_spectrum_yaxis(fig, observation_type)


# =======================================
# LEAVE-ONE-OUT CONTRIBUTIONS
# =======================================
NOT_MOLECULES = ("pressure", "temperature", "kz")


def contribution_options(config, out):
    """
    Molecules that can be left out of the max log-likelihood model and the default selection: the molecules the user
    ran (free chemistry species or grid molecules), otherwise (equilibrium chemistry, or a grid with molecules='all')
    every molecule in the opacity database. Molecules in the model come first, most abundant first.
    """
    profile = out["profiles"][0]
    abundances = profile.drop(columns=[col for col in NOT_MOLECULES if col in profile]).select_dtypes("number").mean()
    in_model = list(abundances.sort_values(ascending=False).index)
    opacity_molecules = resources.opacity(config["OpticalProperties"]).molecules
    options = in_model + [mol for mol in opacity_molecules if mol not in in_model]

    chemistry = config.get("chemistry", {})
    method = chemistry.get("method", "")
    if method == "free":
        requested = chemistry.get("free", {}).get("species", [])
    else:
        requested = chemistry.get(method, {}).get("grid_kwargs", {}).get("molecules") if method else None
    if isinstance(requested, list):
        default = [mol for mol in options if mol in requested]
    else:
        default = [mol for mol in options if mol in opacity_molecules]
    return options, default


@resources.serialized
def leave_one_out(config, info, molecules):
    """
    Max log-likelihood model recomputed with each molecule's opacity removed in turn.

    Returns
    -------
    {molecule: check_model_samples(full_likelihood=True) style output}
    """
    OPA, param_tools = resources.opacity(config["OpticalProperties"]), resources.param_tools_for(config)
    fitpars = go.prior_finder(config["retrieval"])
    data, conv = go.get_data(config)
    point = np.atleast_2d(info["max_logl_point"])
    results = {}
    for mol in molecules:
        excluded = copy.deepcopy(config)
        excluded.setdefault("chemistry", {})["exclude_mol"] = mol
        results[mol] = go.log_likelihood(point, fitpars, excluded, OPA, data, param_tools, CONV_DICT=conv,
                                         retrieval=False)
    return results


def default_corner_settings(info):
    """{parameter: {label, min, max}} with ranges spanning the samples, in log10 for log sampled parameters."""
    settings = {}
    for i, param in enumerate(info["param_names"]):
        values = info["samples_equal"][:, i]
        label = param
        if param in info.get("log_params", []):
            values, label = np.log10(values), f"log10({param})"
        settings[param] = {"label": label, "min": float(np.min(values)), "max": float(np.max(values))}
    return settings


def corner_figure(info, settings):
    """Matplotlib corner plot of the equally weighted samples."""
    fig, _ = ret.plot_pair(
        info["samples_equal"], info["param_names"],
        pretty_labels={param: s["label"] for param, s in settings.items()},
        ranges={param: [s["min"], s["max"]] for param, s in settings.items()},
        log_params=info.get("log_params"),
    )
    return fig


def band_options(out):
    """Profiles bands can be generated for (everything but pressure)."""
    return [key for key in out["profiles"][0].keys() if "pressure" not in key]


def bands(config, info, nsamples, selected):
    return ret.get_bands(config, info, N=nsamples, pressure_bands=selected, eval_maxlogl=True)


def band_figures(returns):
    """Matplotlib figures of the spectra bands and the pressure-profile bands."""
    spectra, _ = ret.plot_spectra_bands(returns)
    profiles, _ = ret.plot_pressure_bands(returns)
    return [spectra, profiles]


def export_package(evaluations, info, details, attributes):
    """
    Zip of retrieval_results() output: the xarray dataset, sample pickle and standard plots.

    Parameters
    ----------
    details : dict
        spectrum_tag, spectrum_unit, author, contact, model_description, code
    attributes : dict
        Extra metadata attributes embedded in the NetCDF file
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        ret.retrieval_results(evaluations, info, os.path.join(tmpdir, "retrieval_results"), **details, **attributes)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for folder, _, filenames in os.walk(tmpdir):
                for filename in filenames:
                    path = os.path.join(folder, filename)
                    zip_file.write(path, os.path.relpath(path, tmpdir))
        return buffer.getvalue()
