"""Turning figures into something a web page can show, plus the prior-sample plots."""
import base64
import io
import uuid
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")  # render off-screen; the server has no display
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class Plot:
    """A plotly figure ready for the page. `id` changes whenever the figure does."""
    id: str
    json: str


SPECTRUM_YAXIS_TITLES = {
    "transit_depth": "Transit Depth (R<sub>p</sub>/R<sub>s</sub>)<sup>2</sup>",
    "fpfs_reflected": "Planet Flux / Stellar Flux",
    "fpfs_thermal": "Planet Flux / Stellar Flux",
    "thermal": "Flux [erg/cm<sup>2</sup>/s/cm]",
    "albedo": "Apparent Albedo",
}


def label_spectrum_yaxis(fig, observation_type):
    """Titles a spectrum figure's y axis with the physical units of `observation_type` (e.g. transit_depth)."""
    # automargin widens the left margin to fit long tick labels (e.g. 0.02215) so the title doesn't sit on them
    return fig.update_yaxes(title_text=SPECTRUM_YAXIS_TITLES.get(observation_type, observation_type),
                            automargin=True, title_standoff=10)


def plotly_plot(fig):
    fig.update_layout(width=None, autosize=True)  # fill the card instead of a fixed pixel width
    return Plot(uuid.uuid4().hex, fig.to_json())


def matplotlib_bytes(fig, fmt="png", dpi=100):
    buffer = io.BytesIO()
    fig.savefig(buffer, format=fmt, dpi=dpi, bbox_inches="tight")
    return buffer.getvalue()


def matplotlib_image(fig, dpi=100):
    """PNG data URI for an <img>; closes the figure."""
    png = matplotlib_bytes(fig, "png", dpi)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(png).decode()


# =======================================
# MODEL DIAGNOSTICS
# =======================================
def describe_nans(wavelength, mask):
    """e.g. "1.20-1.35 um, 4.10 um" for the contiguous wavelength ranges where mask is True."""
    indices = np.flatnonzero(mask)
    runs = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1) if indices.size else []
    ranges = []
    for run in runs:
        start, end = wavelength[run[0]], wavelength[run[-1]]
        ranges.append(f"{start:.2f} um" if start == end else f"{start:.2f}-{end:.2f} um")
    return ", ".join(ranges)


def cloud_layers(cloud_df):
    """Reshapes PICASO's long-format cloud profile to (pressure, wavenumber, w0, opd, g0) arrays [layer, wavenumber]."""
    df = cloud_df.astype("float")
    pressure = df["pressure"].unique()
    wavenumber = df["wavenumber"].unique()
    shape = (len(pressure), len(wavenumber))
    return (pressure, wavenumber,
            np.reshape(df["w0"].values, shape), np.reshape(df["opd"].values, shape), np.reshape(df["g0"].values, shape))


# =======================================
# PRIOR SAMPLES
# =======================================
def prior_sample_figures(profiles, include_clouds):
    """Pressure-temperature, mixing ratio and (optionally) cloud figures overlaying every prior sample."""
    pt_fig = go.Figure()
    for p in profiles:
        pt_fig.add_trace(go.Scatter(x=p["temperature"], y=p["pressure"], mode="lines",
                                    line=dict(color="red", width=1.5), opacity=0.3, showlegend=False))
    pt_fig.update_xaxes(title_text="Temperature (K)")
    pt_fig.update_yaxes(type="log", title_text="Pressure(Bars)", autorange="reversed")
    pt_fig.update_layout(title=f"Pressure-Temperature Profiles ({len(profiles)} Samples)")

    mixing_fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for n, p in enumerate(profiles):
        for i, mol in enumerate(profiles[0]["molecules"]):
            mixing_fig.add_trace(go.Scatter(x=p["mixingratios"][mol], y=p["pressure"], mode="lines", name=mol,
                                            line=dict(color=colors[i % len(colors)], width=2), opacity=0.3,
                                            showlegend=(n == 0), legendgroup=mol))
    mixing_fig.update_xaxes(type="log", title_text="Mixing Ratio(v/v)", range=[-20, 1])
    mixing_fig.update_yaxes(type="log", title_text="Pressure(Bars)", autorange="reversed")
    mixing_fig.update_layout(title="Mixing Ratios")

    figures = [pt_fig, mixing_fig]
    if include_clouds:
        figures.append(_cloud_sample_figure(profiles))
    return figures


def _cloud_sample_figure(profiles):
    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        "Single scattering albedo vs Pressure", "Asymmetry vs Pressure", "Optical Depth vs Pressure"))
    for p in profiles:
        pressure, _, w0, opd, g0 = cloud_layers(p["cloudprofile"])
        for col, (values, color) in enumerate([(w0, "blue"), (g0, "green"), (opd + 1e-60, "red")], start=1):
            fig.add_trace(go.Scatter(x=np.mean(values, axis=1), y=pressure, mode="lines",
                                     line=dict(color=color, width=1.5), opacity=0.3, showlegend=False), row=1, col=col)
    fig.update_yaxes(type="log", autorange="reversed")
    fig.update_xaxes(title_text="Single Scattering Albedo", row=1, col=1)
    fig.update_xaxes(title_text="Asymmetry", row=1, col=2)
    fig.update_xaxes(type="log", title_text="Optical Depth", row=1, col=3)
    fig.update_layout(height=500)
    return fig
