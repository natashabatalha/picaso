"""Reading a resampled opacity database and plotting its cross sections for the Opacity Viewer page."""
import itertools
import os
from dataclasses import dataclass
from functools import lru_cache

import h5py
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots

from picaso import opacity_factory as opa

MOLECULAR_UNIT = "cm<sup>2</sup>/molecule"
CONTINUUM_UNIT = "cm<sup>-1</sup> amagat<sup>-2</sup>"


@dataclass(frozen=True)
class OpacityFile:
    """What is inside an opacity database, enough to build the form."""
    molecules: tuple
    continuum: tuple
    temperatures: tuple   # molecular grid temperatures [K]
    pressures: tuple      # molecular grid pressures [bar]
    continuum_temperatures: tuple
    wave_min: float       # [micron]
    wave_max: float       # [micron]
    native_R: float       # median resolving power of the wavenumber grid


def describe(path):
    """OpacityFile for `path`. Cached until the file changes, since scanning a big database takes a moment."""
    path = os.path.abspath(os.path.expanduser(path))
    return _describe(path, os.path.getmtime(path))


@lru_cache(maxsize=8)
def _describe(path, mtime):
    molecules, pt_pairs = opa.molecular_avail(path)
    continuum, continuum_temperatures = opa.continuum_avail(path)
    wno = wavenumber_grid(path)
    return OpacityFile(
        molecules=tuple(molecules),
        continuum=tuple(continuum),
        temperatures=tuple(sorted({float(t) for _, _, t in pt_pairs})),
        pressures=tuple(sorted({float(p) for _, p, _ in pt_pairs})),
        continuum_temperatures=tuple(float(t) for t in continuum_temperatures),
        wave_min=float(1e4 / wno.max()),
        wave_max=float(1e4 / wno.min()),
        native_R=native_resolution(wno),
    )


def wavenumber_grid(path):
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as f:
            return np.asarray(f["header/wavenumber_grid"][:])
    cur, conn = opa.open_local(path)
    try:
        cur.execute("SELECT wavenumber_grid FROM header")
        return np.asarray(cur.fetchone()[0])
    finally:
        conn.close()


def native_resolution(wno):
    wno = np.sort(wno)
    return float(np.median(wno[1:] / np.diff(wno)))


def nearest(grid, value):
    grid = np.asarray(grid)
    return float(grid[np.abs(grid - value).argmin()])


def resample(wno, opacity, wave_min, wave_max, R=None):
    """Cuts to [wave_min, wave_max] micron, then bins to constant R (if R is given)."""
    keep = (wno >= 1e4 / wave_max) & (wno <= 1e4 / wave_min)
    wno, opacity = wno[keep], opacity[keep]
    if R and len(wno) > 1:
        wno, opacity = opa.regrid(wno, opacity, R=R)
    return wno, opacity


def opacity_figure(path, molecules, continuum, temperatures, pressures, wave_range, R=None,
                   xscale="log", yscale="log", xunit="micron"):
    """
    Plotly figure of the requested cross sections, plus a list of (level, message) notes.

    Molecules are drawn for every temperature/pressure combination, at the nearest grid point.
    Continuum (CIA) opacities have no pressure dependence and different units, so they go on a
    second y axis, once per temperature.
    """
    info = describe(path)
    wave_min, wave_max = wave_range
    notes = []
    if R and R >= info.native_R:
        notes.append(("warning", f"R={R:g} is not below the native resolution of this file (R~{info.native_R:,.0f}), "
                                 "so the opacities are shown unresampled."))
        R = None

    traces = []  # (label, wno, opacity, is_continuum)
    if molecules:
        pairs = list(itertools.product(temperatures, pressures))
        data = opa.get_molecular(path, list(molecules), [t for t, _ in pairs], [p for _, p in pairs])
        for mol in molecules:
            for t in sorted(data.get(mol, {})):
                for p in sorted(data[mol][t]):
                    wno, y = resample(np.asarray(data["wavenumber"]), np.asarray(data[mol][t][p]), wave_min, wave_max, R)
                    traces.append((f"{mol} at {t:g} K {p:g} bar", wno, y, False))
    if continuum:
        data = opa.get_continuum(path, list(continuum), list(temperatures))
        for spec in continuum:
            for t in sorted(data.get(spec, {})):
                wno, y = resample(np.asarray(data["wavenumber"]), np.asarray(data[spec][t]), wave_min, wave_max, R)
                traces.append((f"{spec} at {t:g} K", wno, y, True))

    has_continuum = any(c for *_, c in traces)
    has_molecular = any(not c for *_, c in traces)
    fig = make_subplots(specs=[[{"secondary_y": has_continuum and has_molecular}]])
    colors = px.colors.qualitative.Plotly
    for i, (label, wno, y, is_continuum) in enumerate(traces):
        x = wno if xunit == "wavenumber" else 1e4 / wno
        if yscale == "log":
            y = np.where(y > 0, y, np.nan)  # zeros cannot be drawn on a log axis
        fig.add_scatter(x=x, y=y, name=label, mode="lines", line=dict(width=1, color=colors[i % len(colors)],
                        dash="dot" if is_continuum and has_molecular else None),
                        secondary_y=is_continuum and has_molecular)

    fig.update_xaxes(type=xscale, title="Wavenumber [cm<sup>-1</sup>]" if xunit == "wavenumber" else "Wavelength [μm]")
    fig.update_yaxes(type=yscale, exponentformat="power")
    if has_molecular:
        fig.update_yaxes(title=f"Cross section [{MOLECULAR_UNIT}]", secondary_y=False)
    if has_continuum:
        fig.update_yaxes(title=f"Continuum [{CONTINUUM_UNIT}]", secondary_y=has_molecular, showgrid=not has_molecular)
    fig.update_layout(height=640, margin=dict(l=70, r=20, t=30, b=60),
                      legend=dict(orientation="h", yanchor="top", y=-0.15, x=0))
    return fig, notes
