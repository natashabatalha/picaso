"""
PICASO computations behind the 1D Climate Calculations page. Every function
takes an exported (clean) climate config, as produced by
config_ops.export_climate_config.
"""
import functools
import json
import os
import tempfile

import pandas as pd
import plotly.graph_objects as go_plotly

import picaso.driver as go
from picaso import WIP_justplotit as jpi
from picaso import justdoit as jdi
from picaso.driver_ui.core import resources


def opacity(optical):
    """Correlated-k opacity connection for a climate config's [OpticalProperties] section."""
    kwargs = json.dumps(optical.get("opacity_kwargs", {}), sort_keys=True)
    return _opacity(optical.get("opacity_file", ""), optical["opacity_method"], kwargs)


@functools.lru_cache(maxsize=1)
def _opacity(filename, method, kwargs_json):
    optical = {"opacity_file": filename, "opacity_method": method, "opacity_kwargs": json.loads(kwargs_json)}
    return go.climate_opacity({"OpticalProperties": optical})


@resources.serialized
def climate_class(config):
    """picaso climate class, set up but not run."""
    return go.setup_climate_class(config, opacity(config["OpticalProperties"]))


@resources.serialized
def run_climate(config):
    """Like driver.run_climate, but reusing the cached opacity connection. Returns a dict of results."""
    opa = opacity(config["OpticalProperties"])
    picaso_class = go.setup_climate_class(config, opa)
    settings = config.get("climate", {})
    out = picaso_class.climate(opa, save_all_profiles=settings.get("save_all_profiles", False),
                               with_spec=settings.get("with_spec", False), verbose=settings.get("verbose", True))
    return {"out": out, "picaso_class": picaso_class,
            "guess": (picaso_class.inputs["climate"]["guess_temp"], picaso_class.inputs["climate"]["pressure"])}


# =======================================
# FIGURES
# =======================================
def pt_figure(profiles):
    """Pressure-temperature figure of [(label, temperature, pressure, dashed)]."""
    fig = go_plotly.Figure()
    for label, temperature, pressure, dashed in profiles:
        fig.add_trace(go_plotly.Scatter(x=temperature, y=pressure, mode="lines", name=label,
                                        line=dict(width=2, dash="dash" if dashed else "solid")))
    fig.update_xaxes(title_text="Temperature (K)")
    fig.update_yaxes(type="log", title_text="Pressure (bars)", autorange="reversed")
    return fig


def guess_figure(config):
    picaso_class = climate_class(config)
    temperature, pressure = picaso_class.inputs["climate"]["guess_temp"], picaso_class.inputs["climate"]["pressure"]
    return pt_figure([("Initial guess", temperature, pressure, True)])


def result_figures(results):
    """Converged pressure-temperature profile (with the guess) and, if computed, the thermal spectrum."""
    out = results["out"]
    temperature, pressure = results["guess"]
    figures = [pt_figure([("Initial guess", temperature, pressure, True),
                          ("Converged", out["temperature"], out["pressure"], False)])]
    spectrum = out.get("spectrum_output")
    if spectrum is not None and "thermal" in spectrum:
        fig = jpi.spectrum(spectrum["wavenumber"], spectrum["thermal"], backend="plotly")
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log", title_text="Thermal flux (erg/s/cm²/cm)")
        figures.append(fig)
    return figures


# =======================================
# DOWNLOADS
# =======================================
def climate_netcdf(results):
    """Requires the climate run to have been run with with_spec=True."""
    dataset = jdi.output_xarray(results["out"], results["picaso_class"])
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        path = tmp.name
    try:
        dataset.to_netcdf(path)
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.remove(path)


def climate_csv(results):
    out = results["out"]
    return pd.DataFrame({"pressure": out["pressure"], "temperature": out["temperature"]}).to_csv(index=False)
