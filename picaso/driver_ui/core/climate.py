"""
PICASO computations behind the 1D Climate Calculations page. Every function
takes an exported (clean) climate config, as produced by
config_ops.export_climate_config.
"""
import functools
import json
import os
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go_plotly
from plotly.subplots import make_subplots

import picaso.driver as go
from picaso import justplotit as jpi
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
def run_climate(config, watch=None):
    """
    Like driver.run_climate, but reusing the cached opacity connection. Returns a dict of results.
    `watch` is passed to picaso's climate() to follow the run (e.g. a climate_log.ConvergenceHistory).
    """
    opa = opacity(config["OpticalProperties"])
    picaso_class = go.setup_climate_class(config, opa)
    settings = config.get("climate", {})
    out = picaso_class.climate(opa, save_all_profiles=settings.get("save_all_profiles", False),
                               with_spec=settings.get("with_spec", False), verbose=settings.get("verbose", True),
                               watch=watch)
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


def live_figure(history):
    """
    Convergence so far of a running climate model (a climate_log.ConvergenceHistory): every outer
    iteration's T(P) profile, the newest in red, and dT per iteration against each stage's tolerance.
    """
    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45], horizontal_spacing=0.12,
                        subplot_titles=("Temperature profile", "Change per outer iteration"))
    pressure = history.pressure
    if pressure is not None and history.guess is not None:
        fig.add_trace(go_plotly.Scatter(x=history.guess, y=pressure, name="Initial guess", mode="lines",
                                        line=dict(color="gray", dash="dash", width=1)), 1, 1)
    n = len(history.profiles)
    for i, (k, temperature) in enumerate(history.profiles):
        newest = i == n - 1
        fig.add_trace(go_plotly.Scatter(
            x=temperature, y=pressure, mode="lines", name=f"Iteration {k}", showlegend=newest,
            line=dict(color="#d62728" if newest else "#1f77b4", width=3 if newest else 1),
            opacity=1 if newest else 0.1 + 0.4 * (i + 1) / n), 1, 1)
    if pressure is not None:
        last = len(pressure) - 1
        for top, bottom in history.zones:
            fig.add_hrect(y0=np.log10(pressure[min(top, last)]), y1=np.log10(pressure[min(bottom, last)]),
                          fillcolor="orange", opacity=0.15, line_width=0, row=1, col=1)
    if history.dts:
        k, dt, tol, _ = map(list, zip(*history.dts))
        fig.add_trace(go_plotly.Scatter(x=k, y=np.maximum(dt, 1e-3), mode="lines+markers", name="dT",
                                        line=dict(color="#1f77b4")), 1, 2)
        fig.add_trace(go_plotly.Scatter(x=k, y=tol, mode="lines", name="Stage tolerance",
                                        line=dict(color="#2ca02c", dash="dash", shape="hvh")), 1, 2)
    fig.update_xaxes(title_text="Temperature (K)", row=1, col=1)
    fig.update_yaxes(type="log", title_text="Pressure (bars)", autorange="reversed", row=1, col=1)
    fig.update_xaxes(title_text="Outer iteration", row=1, col=2)
    fig.update_yaxes(type="log", title_text="mean |dT| / 1.5 (K)", row=1, col=2)
    fig.update_layout(height=450, legend=dict(orientation="h", y=-0.2), uirevision="climate-live")
    return fig


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
