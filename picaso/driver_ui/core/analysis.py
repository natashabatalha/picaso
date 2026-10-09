"""Retrieval Analysis computations: reading retrieval output, corner plots, contributions, bands and export packages."""
import copy
import io
import os
import tempfile
import zipfile

import numpy as np
import toml

import picaso.driver as go
from picaso import WIP_justplotit as jpi
from picaso import retrieval as ret
from picaso.driver_ui.core import resources
from picaso.driver_ui.core.plots import label_spectrum_yaxis


def example_path(refdata):
    """The synthetic transit retrieval shipped with the reference data (built by reference/scripts/make_retrieval_example.py)."""
    return os.path.join(refdata, "base_cases", "retrieval_example", "retrieval_example.toml")


def example_config(refdata):
    """(config with `_default_` paths resolved, raw toml text) of the example retrieval."""
    with open(example_path(refdata)) as f:
        text = f.read()
    # plain replacement: config_ops.resolve_defaults would swap the example's small opacity db for the default one
    return go.resolve_default_paths(toml.loads(text), refdata), text


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
    fig.update_traces(selector=0, line=dict(color="black", width=4))  # others are width 3
    fig = jpi.plot_errorbar(1e4 / out["xdata"], out["ydata"][0], out["edata"][0], plot=fig, backend="plotly",
                            point_kwargs=dict(color="black"), error_kwargs=dict(color="black"))
    fig.update_traces(selector=-1, name="Data", opacity=0.5)
    fig.data = fig.data[-1:] + fig.data[:-1]  # traces draw in order, so the data goes behind the models
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


EXPORT_NAME = "retrieval_results"  # file prefix retrieval_results() writes into the package


def _format_interval(median, lo, hi, digits=3):
    return f"{median:.{digits}g} (+{hi - median:.{digits}g} / -{median - lo:.{digits}g})"


def readme_from_config(config, info, max_logl_chisq=None):
    """
    Quick look README.md summarizing a retrieval: the model setup in the driver config, the fit result and the
    DOIs of the parameterizations used.

    Parameters
    ----------
    config : dict
        Driver config of the retrieval
    info : dict
        ret.read_retrievals output
    max_logl_chisq : float
        Chi-sq per data point of the max log-likelihood model
    """
    lines = ["# PICASO Retrieval Results", "",
             "Quick look summary of this retrieval, generated from the driver TOML file (`inputs.toml`).", ""]

    lines += ["## Model setup", "",
              f"- **Observation type:** {config.get('observation_type', 'not specified')}"]
    temperature = config.get("temperature", {}).get("profile")
    lines.append(f"- **Temperature profile:** {temperature or 'not specified'}")
    chemistry = config.get("chemistry", {})
    method = chemistry.get("method")
    if method == "free":
        # molecule tables have a profile; background is {gases, fraction}
        free = chemistry.get("free", {})
        species = [f"{mol} ({opts['profile']})" for mol, opts in free.items()
                   if isinstance(opts, dict) and "profile" in opts and opts["profile"] != "background"]
        background = free.get("background", {}).get("gases", [])
        method = "free" + (f": {', '.join(species)}" if species else "")
        if background:
            method += f"; background: {', '.join(background)}"
    lines.append(f"- **Chemistry:** {method or 'not specified'}")
    clouds = config.get("clouds", {})
    cloud_types = [f"{key.split('_type')[0]}: {value}" for key, value in clouds.items()
                   if key.endswith("_type") and value] if isinstance(clouds, dict) else []
    lines.append(f"- **Clouds:** {', '.join(cloud_types) if cloud_types else 'none'}")
    sampler = config.get("retrieval", {}).get("sampler", {}).get("code")
    if sampler:
        lines.append(f"- **Sampler:** {sampler}")
    filenames = config.get("ObservationData", {}).get("filenames", [])
    filenames = [filenames] if isinstance(filenames, str) else filenames
    lines += ["", "### Data files", ""] + ([f"- `{name}`" for name in filenames] or ["- none specified"])
    retrieval_output = config.get("InputOutput", {}).get("retrieval_output")
    if retrieval_output:
        lines += ["", f"Retrieval output directory: `{retrieval_output}`"]

    lines += ["", "## Retrieval result", ""]
    if max_logl_chisq is not None:
        lines.append(f"- **Max log-likelihood chi-sq per data point:** {float(np.squeeze(max_logl_chisq)):.3f}")
    if info.get("max_logl") is not None:
        lines.append(f"- **Max log-likelihood:** {float(info['max_logl']):.3f}")
    if not info.get("converged", True):
        lines.append(f"- **Warning:** this retrieval had not converged when exported ({info.get('niter')} iterations)")

    lines += ["", "### 1-sigma constraints (median, +/- 1 sigma)", "",
              "| Parameter | Constraint | log10 constraint | Max LogL value |", "|---|---|---|---|"]
    intervals = info["med_intervals"]
    log_params = info.get("log_params", [])
    for i, param in enumerate(info["param_names"]):
        median, lo, hi = (float(intervals[f"{param}_{key}"].values[0]) for key in ("median", "errlo", "errup"))
        log_interval = _format_interval(*np.log10([median, lo, hi])) if param in log_params else ""
        lines.append(f"| {param} | {_format_interval(median, lo, hi)} | {log_interval} "
                     f"| {float(info['max_logl_point'][i]):.4g} |")

    lines += ["", "## Reading the full results", "",
              f"All of the median and max log-likelihood spectra, 1 and 2 sigma bands, profiles and metadata are in "
              f"`{EXPORT_NAME}_median_and_max_logl.nc`. Open it with xarray:", "",
              "```python",
              "import xarray as xr",
              f"ds = xr.load_dataset('{EXPORT_NAME}_median_and_max_logl.nc')",
              "print(ds)  # data variables, coordinates and units",
              "print(ds.attrs['intervals_params'])  # 1-sigma constraints",
              "print(ds.attrs['max_logl_params'])  # max log-likelihood parameters",
              "```", "",
              f"The equally weighted posterior samples are pickled in `{EXPORT_NAME}_equally_weighted_samples.pk`:", "",
              "```python",
              "import pickle",
              f"param_names, samples = pickle.load(open('{EXPORT_NAME}_equally_weighted_samples.pk', 'rb'))",
              "```", ""]

    lines += ["## References", "",
              "Please cite PICASO along with the parameterizations used in this model:", ""]
    try:
        references = go.references(driver_dict=config)
    except Exception as e:
        references, error = {}, e
    else:
        error = None
    cited = [(section, func, dois) for section, funcs in references.items() for func, dois in funcs.items()]
    for section, func, dois in cited:
        lines.append(f"- **{section}** (`{func}`): " + ", ".join(f"https://doi.org/{doi}" for doi in dois))
    if error is not None:
        lines.append(f"- Could not look up references: {error}")
    elif not cited:
        lines.append("- No DOIs are registered for the parameterizations in this model.")
    return "\n".join(lines) + "\n"


def export_package(evaluations, info, details, attributes, config=None, config_text=None):
    """
    Zip of retrieval_results() output: the xarray dataset, sample pickle and standard plots, plus the driver TOML
    (inputs.toml) and a quick look README.md when `config` is given.

    Parameters
    ----------
    details : dict
        spectrum_tag, spectrum_unit, author, contact, model_description, code
    attributes : dict
        Extra metadata attributes embedded in the NetCDF file
    config : dict
        Driver config of the retrieval
    config_text : str
        The driver TOML as uploaded (keeps its comments); otherwise `config` is written out as TOML
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        ret.retrieval_results(evaluations, info, os.path.join(tmpdir, EXPORT_NAME), **details, **attributes)
        if config is not None:
            with open(os.path.join(tmpdir, "inputs.toml"), "w") as f:
                f.write(config_text if config_text is not None else toml.dumps(config))
            with open(os.path.join(tmpdir, "README.md"), "w") as f:
                f.write(readme_from_config(config, info, evaluations.get("max_logl_chisq")))
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for folder, _, filenames in os.walk(tmpdir):
                for filename in filenames:
                    path = os.path.join(folder, filename)
                    zip_file.write(path, os.path.relpath(path, tmpdir))
        return buffer.getvalue()
