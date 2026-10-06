"""Retrieval Analysis computations: reading retrieval output, corner plots, bands and export packages."""
import io
import os
import tempfile
import zipfile

import numpy as np

import picaso.driver as go
from picaso import WIP_justplotit as jpi
from picaso import retrieval as ret


def parameters(config):
    """Free parameter paths in the config's [retrieval] section, in order."""
    return list(go.prior_finder(config.get("retrieval", {})))


def load(config, retrieval_dir):
    """Reads the samples and evaluates the max log-likelihood model against the data."""
    info = ret.read_retrievals(retrieval_dir, parameters(config))
    out = go.check_model_samples(config, N=1, samples=np.atleast_2d(info["max_logl_point"]), full_likelihood=True)
    chi2 = out["chi_sq_per_pt"][0]
    fig = jpi.spectrum([out["xdata"]], [out["ymodel"][0]], legend=["Max LogL Model"], backend="plotly")
    fig = jpi.plot_errorbar(1e4 / out["xdata"], out["ydata"][0], out["edata"][0], plot=fig, backend="plotly")
    fig.update_layout(title=f"Chi-sq = {chi2:.2f}")
    return {"info": info, "out": out, "chi2": chi2, "figure": fig}


def default_corner_settings(info):
    """{parameter: {label, min, max}} with ranges spanning the samples."""
    samples = info["samples_equal"]
    return {param: {"label": param, "min": float(np.min(samples[:, i])), "max": float(np.max(samples[:, i]))}
            for i, param in enumerate(info["param_names"])}


def corner_figure(info, settings):
    """Matplotlib corner plot of the equally weighted samples."""
    fig, _ = ret.plot_pair(
        info["samples_equal"], info["param_names"],
        pretty_labels={param: s["label"] for param, s in settings.items()},
        ranges={param: [s["min"], s["max"]] for param, s in settings.items()},
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
