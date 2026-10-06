"""
PICASO computations behind the Spectrum & Retrieval Setup page. Every function
takes an exported (clean) config, as produced by config_ops.export_config, and
uses the process's cached opacity connection (see core/resources.py).
"""
import copy
import os
import tempfile

import numpy as np
import pandas as pd

import picaso.driver as go
from picaso import WIP_justplotit as jpi
from picaso import justdoit as jdi
from picaso.driver_ui.core import resources
from picaso.driver_ui.core.plots import cloud_layers, describe_nans

MOLECULES_LIMIT = 10  # molecules shown in prior-sample mixing ratio plots


@resources.serialized
def spectrum_class(config, stage=None):
    """picaso inputs class set up through `stage` (object, star, temperature, chemistry; None = everything)."""
    opacity = resources.opacity(config["OpticalProperties"])
    return go.setup_spectrum_class(config, opacity, resources.param_tools_for(config), stage)


@resources.serialized
def run_model(config):
    """Like driver.run() for a spectrum, but without loading a second copy of the opacity database."""
    picaso_class = spectrum_class(config)
    calculation = go.OBSERVATION_CALC_MAP.get(config["observation_type"], config["observation_type"])
    output = picaso_class.spectrum(resources.opacity(config["OpticalProperties"]), full_output=True,
                                   calculation=calculation)
    return output, picaso_class


# =======================================
# PREVIEWS
# =======================================
def pt_figure(config):
    profile = spectrum_class(config, "temperature").inputs["atmosphere"]["profile"]
    return jpi.Analyzer({"layer": profile}, backend="plotly").pt()


def mixing_ratio_figure(config):
    profile = spectrum_class(config, "chemistry").inputs["atmosphere"]["profile"]
    full_output = {"layer": {"pressure": profile["pressure"], "mixingratios": profile}}
    return jpi.Analyzer(full_output, backend="plotly").mixing_ratio()


def cloud_figure(config):
    """None if the configuration produced no cloud profile."""
    clouds = spectrum_class(config).inputs.get("clouds", {})
    if clouds.get("profile") is None:
        return None
    pressure, wavenumber, w0, opd, g0 = cloud_layers(clouds["profile"])
    full_output = {"layer": {"pressure": pressure, "cloud": {"w0": w0, "opd": opd, "g0": g0}},
                   "wavenumber": wavenumber}
    return jpi.Analyzer(full_output, backend="plotly").cloud()


# =======================================
# SPECTRUM
# =======================================
def run_spectrum(config, resolution):
    """Runs the full model and regrids it to `resolution`. Returns a dict of results and warnings."""
    df, picaso_class = run_model(config)
    key = config["observation_type"]
    wavenumber, flux, mask = go.process_model(df["wavenumber"], df[key], config=config, regrid_R=resolution)["model"]
    warnings = []
    if np.any(mask):
        warnings.append(f"Encountered {np.sum(mask)} NaNs out of {len(mask)} model points. "
                        f"NaN wavelength ranges: {describe_nans(1e4 / wavenumber, mask)}")
    return {"df": df, "picaso_class": picaso_class, "wavenumber": wavenumber, "flux": flux,
            "observation_key": key, "warnings": warnings}


def spectrum_figure(wavenumber, flux, wave_range):
    fig = jpi.spectrum(wavenumber, flux, backend="plotly")
    fig.update_xaxes(range=list(wave_range))
    return fig


def spectrum_netcdf(results):
    dataset = jdi.output_xarray(results["df"], results["picaso_class"])
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        path = tmp.name
    try:
        dataset.to_netcdf(path)
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.remove(path)


def spectrum_csv(results):
    return pd.DataFrame({"wavenumber": results["wavenumber"],
                         results["observation_key"]: results["flux"]}).to_csv(index=False)


# =======================================
# OBSERVATIONAL DATA
# =======================================
def check_data(config, reference=None, wave_range=None):
    """
    Parses the observation files and plots them, optionally with a reference model
    (run_spectrum results) binned to each data grid. Returns (messages, figure).
    """
    data_dict, conv_dict = go.get_data(config)
    if not data_dict:
        raise ValueError("No observation data files were given")
    messages = [("success", f"Successfully parsed {len(data_dict)} files: {list(data_dict)}")]
    messages += [("info", f"{key}: {len(x)} points, wavenumber range: {min(x):.2f} - {max(x):.2f}")
                 for key, (x, _, _) in data_dict.items()]

    fig = None
    if reference is not None:
        df = reference["df"]
        key = config["observation_type"]
        processed = go.process_model(df["wavenumber"], df[key], data_dict=data_dict, conv_dict=conv_dict, config=config)
        xs, ys = [], []
        for name, (x, y, mask) in processed.items():
            xs.append(x)
            ys.append(y)
            if np.any(mask):
                messages.append(("warning", f"Reference model binned to {name} data grid encountered "
                                            f"{np.sum(mask)} NaNs out of {len(y)} points."))
        fig = jpi.spectrum(xs, ys, legend=list(processed), backend="plotly")
    for x, y, e in data_dict.values():
        fig = jpi.plot_errorbar(1e4 / x, y, e, plot=fig, backend="plotly")
    if wave_range is not None:
        fig.update_xaxes(range=list(wave_range))
    return messages, fig


# =======================================
# PRIOR SAMPLES
# =======================================
def sample_priors(config, priors, nsamples):
    """
    Draws `nsamples` configs from the priors and sets each one up through clouds.
    Returns (sampled configs, profiles for prior_sample_figures).
    """
    configs, profiles = [], []
    for _ in range(nsamples):
        cube = go.hypercube(np.random.rand(len(priors)), priors)
        sample = go.update_config_w_cube(copy.deepcopy(config), priors, cube)
        inputs = spectrum_class(sample).inputs
        atmosphere = inputs["atmosphere"]["profile"]
        molecules = [m for m in atmosphere.keys() if m not in ("pressure", "temperature", "kz")]
        configs.append(sample)
        profiles.append({
            "temperature": atmosphere["temperature"],
            "pressure": atmosphere["pressure"],
            "mixingratios": atmosphere,
            "molecules": molecules[:MOLECULES_LIMIT],
            "cloudprofile": inputs.get("clouds", {}).get("profile"),
        })
    return configs, profiles


def prior_sample_spectra(configs, config, wave_range):
    """Full spectra for each sampled config, binned to the data and plotted over it. Returns (warnings, figure)."""
    data_dict, conv_dict = go.get_data(config)
    xs, ys, warnings = [], [], []
    for n, sample in enumerate(configs, start=1):
        df, _ = run_model(sample)
        key = sample["observation_type"]
        processed = go.process_model(df["wavenumber"], df[key], data_dict=data_dict, config=sample, conv_dict=conv_dict)
        for name, (x_model, y_model, mask) in processed.items():
            xs.append(x_model)
            ys.append(y_model)
            if not np.any(mask):
                continue
            x, y, e = data_dict[name]
            valid = ~mask
            mock_likelihood = np.sum((y[valid] ** 2 - y_model[valid] ** 2) / e[valid] ** 2)
            where = f"Masked points are here: {describe_nans(1e4 / x_model, mask)}"
            if np.isnan(mock_likelihood):
                warnings.append(f"Sample {n}: Tried masking {np.sum(mask)} points out of {len(mask)} model points "
                                f"but NaN still persisting in mock likelihood for {name}. {where}")
            else:
                warnings.append(f"Sample {n}: Masked {np.sum(mask)} points out of {len(mask)} model points but with "
                                f"the mask a likelihood can be computed for {name}. {where}")

    fig = jpi.spectrum(xs, ys, palette=["rgba(255,0,0,0.3)"], backend="plotly")
    for x, y, e in data_dict.values():
        fig = jpi.plot_errorbar(1e4 / x, y, e, plot=fig, backend="plotly")
    fig.update_xaxes(range=list(wave_range))
    return warnings, fig
