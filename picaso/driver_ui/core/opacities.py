"""Reading a resampled opacity database and plotting its cross sections for the Opacity Viewer page."""
import itertools
import os
import warnings
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


def read_observations(filenames, coord, data, error, coord_unit=None, data_unit=None):
    """
    Reads observation files with the same reader as the retrieval setup (picaso.driver.get_data).
    Returns ({name: (wavenumber, y, error)}, notes).
    """
    # picaso.driver imports all of PICASO, which needs the reference data; only pay for it when asked
    import picaso.driver as go
    config = {"ObservationData": {"filenames": list(filenames), "coord": coord, "data": data, "error": error,
                                  "coord_unit": coord_unit, "data_unit": data_unit}}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data_dict, _ = go.get_data(config)
    notes = [("warning", str(w.message)) for w in caught if issubclass(w.category, UserWarning)]
    return {name: tuple(np.asarray(v) for v in values) for name, values in data_dict.items()}, notes


def opacity_figure(path, molecules, continuum, temperatures, pressures, wave_range, R=None,
                   xscale="log", yscale="log", xunit="micron", observations=None, data_label="Data"):
    """
    Plotly figure of the requested cross sections, plus a list of (level, message) notes.

    Molecules are drawn for every temperature/pressure combination, at the nearest grid point.
    Continuum (CIA) opacities have no pressure dependence and different units, so they go on a
    second y axis, once per temperature.

    `observations` ({name: (wavenumber, y, error)}, see read_observations) are drawn in a panel
    half the height of the cross sections, directly above them on a shared x axis.
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
    two_axes = has_continuum and has_molecular
    rows = 2 if observations else 1
    row = rows  # the cross sections always go in the bottom panel
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[1, 2] if observations else None,
                        specs=[[{}]] * (rows - 1) + [[{"secondary_y": two_axes}]])
    colors = px.colors.qualitative.Plotly
    for i, (label, wno, y, is_continuum) in enumerate(traces):
        x = wno if xunit == "wavenumber" else 1e4 / wno
        if yscale == "log":
            y = np.where(y > 0, y, np.nan)  # zeros cannot be drawn on a log axis
        fig.add_scatter(x=x, y=y, name=label, mode="lines", line=dict(width=1, color=colors[i % len(colors)],
                        dash="dot" if is_continuum and has_molecular else None),
                        row=row, col=1, secondary_y=is_continuum and has_molecular)

    for name, (wno, y, err) in (observations or {}).items():
        keep = (wno >= 1e4 / wave_max) & (wno <= 1e4 / wave_min)
        if not keep.any():
            notes.append(("warning", f"{name} has no data points between {wave_min:g} and {wave_max:g} μm."))
        x = wno[keep] if xunit == "wavenumber" else 1e4 / wno[keep]
        fig.add_scatter(x=x, y=y[keep], name=name, mode="lines", line=dict(width=1, color="black"),
                        error_y=dict(type="data", array=err[keep], thickness=1, width=0, color="black"),
                        row=1, col=1)

    fig.update_xaxes(type=xscale)
    fig.update_xaxes(title="Wavenumber [cm<sup>-1</sup>]" if xunit == "wavenumber" else "Wavelength [μm]", row=row, col=1)
    fig.update_yaxes(type=yscale, exponentformat="power", row=row, col=1)
    if has_molecular:
        fig.update_yaxes(title=f"Cross section [{MOLECULAR_UNIT}]", secondary_y=False, row=row, col=1)
    if has_continuum:
        fig.update_yaxes(title=f"Continuum [{CONTINUUM_UNIT}]", secondary_y=has_molecular, showgrid=not has_molecular,
                         row=row, col=1)
    if observations:
        fig.update_yaxes(title=data_label, row=1, col=1)
    height = 960 if observations else 640
    fig.update_layout(height=height, margin=dict(l=70, r=20, t=30, b=60),
                      legend=dict(orientation="h", yanchor="top", y=-96 / height, x=0))
    return fig, notes
