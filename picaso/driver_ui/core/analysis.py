"""Retrieval Analysis computations: reading retrieval output, corner plots, contributions, bands and export packages."""
import copy
import io
import os
import tempfile
import zipfile

import numpy as np
import toml
from plotly import graph_objects as pgo
from scipy.stats import gaussian_kde

import picaso.driver as go
from picaso import justplotit as jpi
from picaso import retrieval as ret
from picaso.driver_ui.core import resources
from picaso.driver_ui.core.plots import SPECTRUM_YAXIS_TITLES, label_spectrum_yaxis


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


def plotted_samples(info):
    """Equally weighted samples as the corner plot shows them: log10 for log sampled parameters."""
    samples = np.array(info["samples_equal"], dtype=float)
    for i, param in enumerate(info["param_names"]):
        if param in info.get("log_params", []):
            samples[:, i] = np.log10(samples[:, i])
    return samples


def default_corner_settings(info):
    """{parameter: {label, min, max}} with ranges spanning the samples, in log10 for log sampled parameters."""
    samples = plotted_samples(info)
    settings = {}
    for i, param in enumerate(info["param_names"]):
        label = f"log10({param})" if param in info.get("log_params", []) else param
        settings[param] = {"label": label, "min": float(np.min(samples[:, i])), "max": float(np.max(samples[:, i]))}
    return settings


def corner_figure(info, settings):
    """Matplotlib corner plot of the equally weighted samples (the high resolution export of the interactive one)."""
    fig, _ = ret.plot_pair(
        info["samples_equal"], info["param_names"],
        pretty_labels={param: s["label"] for param, s in settings.items()},
        ranges={param: [s["min"], s["max"]] for param, s in settings.items()},
        log_params=info.get("log_params"),
    )
    return fig


# =======================================
# INTERACTIVE CORNER PLOT
# =======================================
# 1, 2 and 3 sigma regions, darkest inside (from the GnBu colormap ret.plot_pair uses)
SIGMA_COLORS = ["#0868ac", "#43a2ca", "#a8ddb5"]
# lets the page drag the black lines (layout shapes) of the 1D posteriors
CORNER_PLOT_CONFIG = {"edits": {"shapePosition": True}, "displaylogo": False}


def median_point(info):
    """{parameter: median} of the samples, in the corner plot's space (log10 for log sampled parameters)."""
    return dict(zip(info["param_names"], np.median(plotted_samples(info), axis=0).tolist()))


@resources.serialized
def evaluate_point(config, info, point):
    """
    Model and its fit to the data at a point of the corner plot.

    Parameters
    ----------
    point : dict
        {parameter: value}, in log10 for log sampled parameters (as the corner plot shows them)

    Returns
    -------
    check_model_samples(full_likelihood=True) style output
    """
    log_params = info.get("log_params", [])
    theta = [10**point[param] if param in log_params else point[param] for param in info["param_names"]]
    OPA, param_tools = resources.opacity(config["OpticalProperties"]), resources.retrieval_param_tools(config)
    data, conv = go.get_data(config)
    return go.log_likelihood(np.atleast_2d(theta), go.prior_finder(config["retrieval"]), config, OPA, data,
                             param_tools, CONV_DICT=conv, retrieval=False)


def model_spectrum(out):
    """(wavelength [um], model) of a check_model_samples(full_likelihood=True) output, ordered by wavelength."""
    wavelength = 1e4 / np.asarray(out["xdata"], dtype=float)
    order = np.argsort(wavelength)
    return wavelength[order], np.asarray(out["ymodel"][0], dtype=float)[order]


def corner_spectrum_title(out):
    if out is None:
        return "Model could not be computed at the selected values"
    return f"Model at selected values (Chi-sq = {out['chi_sq_per_pt'][0]:.2f})"


def _interval_text(values):
    """median_{-lo}^{+hi} to the precision of the error, as ret.plot_pair titles its 1D posteriors."""
    lo, mid, hi = np.percentile(values, [15.8655, 50.0, 84.1345])
    errlo, errhi = mid - lo, hi - mid
    err = min(errlo, errhi) if min(errlo, errhi) > 0 else max(errlo, errhi)
    if err <= 0:
        dec = 2
    else:
        log_err = np.log10(err)
        dec = max(0, 2 - int(np.floor(log_err))) if log_err >= 0 else int(np.ceil(-log_err)) + 1
        dec = min(max(dec, 1), 6)
    return f"{mid:.{dec}f}<sub>-{errlo:.{dec}f}</sub><sup>+{errhi:.{dec}f}</sup>"


def _sigma_field(x, y, xx, yy):
    """
    Number of sigma (as for a 2D gaussian) of the highest density region whose edge passes through each grid point,
    so the 1, 2 and 3 sigma contours (the 39.3, 86.5 and 98.9% regions ret.plot_pair draws) are evenly spaced levels.
    """
    density = gaussian_kde(np.vstack([x, y]))(np.vstack([xx.ravel(), yy.ravel()]))
    order = np.argsort(density)[::-1]
    enclosed = np.empty_like(density)
    enclosed[order] = np.cumsum(density[order]) / np.sum(density)
    return np.sqrt(-2 * np.log(np.clip(1 - enclosed, 1e-12, 1))).reshape(xx.shape)


def interactive_corner_figure(info, settings, point, data, model, observation_type):
    """
    Plotly corner plot of the equally weighted samples with the spectrum at `point` to its right. Like ret.plot_pair,
    but the black line on each 1D posterior marks `point`, and the page (static/corner.js) lets the user drag it to
    recompute the spectrum there.

    Parameters
    ----------
    settings : dict
        default_corner_settings style {parameter: {label, min, max}}
    point : dict
        {parameter: value} the black lines are drawn at, in the corner plot's space (see median_point)
    data : dict
        check_model_samples(full_likelihood=True) style output holding the data
    model : dict
        check_model_samples(full_likelihood=True) style output at `point`, or None if it could not be computed
    observation_type : str
        config["observation_type"], which sets the spectrum's y axis units

    Returns
    -------
    plotly figure. layout.meta tells corner.js which shapes and traces to move.
    """
    params = info["param_names"]
    n = len(params)
    samples = plotted_samples(info)
    # kde and drawing cost scale with the number of samples, so thin the ones used for the contours and scatter
    thin = samples[np.random.RandomState(0).permutation(len(samples))[:1000]]
    values = [point[param] for param in params]
    ranges = [[settings[param]["min"], settings[param]["max"]] for param in params]
    labels = [settings[param]["label"] for param in params]

    corner_width, gap = 0.6, 0.012
    size = (1 - gap * (n - 1)) / n

    def domain(k, scale=1.0):
        return [k * (size + gap) * scale, (k * (size + gap) + size) * scale]

    cells = [(i, j) for i in range(n) for j in range(i + 1)]
    number = {cell: k + 1 for k, cell in enumerate(cells)}
    spectrum_number = len(cells) + 1

    def name(axis, k):
        return axis if k == 1 else f"{axis}{k}"

    fig = pgo.Figure()
    layout = {}
    crosshairs = []
    for (i, j), k in number.items():
        xaxis, yaxis = name("x", k), name("y", k)
        layout[name("xaxis", k)] = dict(
            domain=domain(j, corner_width), anchor=yaxis, range=ranges[j], showticklabels=(i == n - 1),
            title_text=labels[j] if i == n - 1 else None, matches=None if i == n - 1 else name("x", number[(n - 1, j)]))
        if i == j:
            counts, edges = np.histogram(samples[:, i], bins=20)
            fig.add_trace(pgo.Bar(x=(edges[1:] + edges[:-1]) / 2, y=counts, width=np.diff(edges), xaxis=xaxis,
                                  yaxis=yaxis, marker_color="grey", opacity=0.7, meta=i, showlegend=False,
                                  hovertemplate=f"{labels[i]}: %{{x:.4g}}<br>%{{y}} samples<extra></extra>"))
            layout[name("yaxis", k)] = dict(domain=domain(n - 1 - i), anchor=xaxis, showticklabels=False,
                                            showgrid=False, rangemode="tozero", fixedrange=True)
            fig.add_annotation(text=f"{labels[i]}<br>{_interval_text(samples[:, i])}", showarrow=False,
                               xref=f"{xaxis} domain", yref=f"{yaxis} domain", x=0.5, y=1, yanchor="bottom",
                               font_size=11)
            # shape i is parameter i's black line, which corner.js drags
            fig.add_shape(type="line", xref=xaxis, yref=f"{yaxis} domain", x0=values[i], x1=values[i], y0=0, y1=1,
                          line=dict(color="black", width=3))
            continue
        layout[name("yaxis", k)] = dict(
            domain=domain(n - 1 - i), anchor=xaxis, range=ranges[i], showticklabels=(j == 0),
            title_text=labels[i] if j == 0 else None, matches=None if j == 0 else name("y", number[(i, 0)]))
        fig.add_trace(pgo.Scatter(x=thin[:, j], y=thin[:, i], xaxis=xaxis, yaxis=yaxis, mode="markers",
                                  marker=dict(color="grey", size=2), opacity=0.5, hoverinfo="skip", showlegend=False))
        try:
            xx, yy = np.meshgrid(np.linspace(samples[:, j].min(), samples[:, j].max(), 60),
                                 np.linspace(samples[:, i].min(), samples[:, i].max(), 60))
            sigma = _sigma_field(thin[:, j], thin[:, i], xx, yy)
        except (np.linalg.LinAlgError, ValueError):
            pass  # parameter has no spread yet (e.g. very early in a run), so just show the scatter
        else:
            # filled 3, 2 then 1 sigma regions, each drawn over the last
            for level, color in zip((3, 2, 1), SIGMA_COLORS[::-1]):
                fig.add_trace(pgo.Contour(x=xx[0], y=yy[:, 0], z=sigma, xaxis=xaxis, yaxis=yaxis,
                                          contours=dict(type="constraint", operation="<", value=level),
                                          fillcolor=color, opacity=0.8, line=dict(color="rgba(0,0,0,0.5)", width=0.8),
                                          hoverinfo="skip", showlegend=False, showscale=False))
        crosshairs.append([len(fig.data), i, j])
        fig.add_trace(pgo.Scatter(x=[values[j]], y=[values[i]], xaxis=xaxis, yaxis=yaxis, mode="markers",
                                  marker=dict(color="black", size=7), hoverinfo="skip", showlegend=False))

    xaxis, yaxis = name("x", spectrum_number), name("y", spectrum_number)
    layout[name("xaxis", spectrum_number)] = dict(domain=[corner_width + 0.08, 1], anchor=yaxis,
                                                  title_text="Wavelength [μm]")
    layout[name("yaxis", spectrum_number)] = dict(
        domain=[0.4, 1], anchor=xaxis, automargin=True, title_standoff=10,
        title_text=SPECTRUM_YAXIS_TITLES.get(observation_type, observation_type))
    wavelength = 1e4 / np.asarray(data["xdata"], dtype=float)
    order = np.argsort(wavelength)
    fig.add_trace(pgo.Scatter(x=wavelength[order], y=np.asarray(data["ydata"][0])[order], xaxis=xaxis, yaxis=yaxis,
                              mode="markers", name="Data", marker=dict(color="black", size=5), opacity=0.5,
                              error_y=dict(array=np.asarray(data["edata"][0])[order], color="black", thickness=1)))
    model_wavelength, model_spectrum_values = model_spectrum(model) if model is not None else ([], [])
    model_trace = len(fig.data)
    fig.add_trace(pgo.Scatter(x=model_wavelength, y=model_spectrum_values, xaxis=xaxis, yaxis=yaxis, mode="lines",
                              name="Model at selected values", line=dict(color="black", width=3)))
    title_annotation = len(fig.layout.annotations)
    fig.add_annotation(text=corner_spectrum_title(model), showarrow=False, xref=f"{xaxis} domain",
                       yref=f"{yaxis} domain", x=0.5, y=1, yanchor="bottom", yshift=6, font_size=14)

    fig.update_layout(
        **layout, height=max(560, min(1100, 150 * n + 160)), bargap=0, plot_bgcolor="white",
        margin=dict(l=80, r=20, t=70, b=90),
        legend=dict(x=corner_width + 0.08, y=0.3, xanchor="left", yanchor="top"),
        meta=dict(params=params, ranges=ranges, medians=list(median_point(info).values()), crosshairs=crosshairs,
                  model_trace=model_trace, title_annotation=title_annotation))
    fig.update_xaxes(showline=True, linecolor="black", mirror=True, ticks="outside", showgrid=False, zeroline=False)
    fig.update_yaxes(showline=True, linecolor="black", mirror=True, zeroline=False)
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
