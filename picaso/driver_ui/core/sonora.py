"""
Reading the Sonora grids on disk for the Sonora viewer page: which download targets are present, which
models they hold, and figures of a model's pressure-temperature profile, chemistry, clouds and published
spectrum. Only reads files (through picaso.sonora), so it works without importing PICASO.
"""
import json
import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import xarray as xr
from plotly.subplots import make_subplots

from picaso import sonora

GRIDS = {
    #label, paper, ADS link, what it is for (from "Which Sonora model should I use?", Morley et al., Exoclimes VI)
    "bobcat": ("Bobcat", "Marley et al. 2021", "https://ui.adsabs.harvard.edu/abs/2021ApJ...920...85M/abstract",
               "200-2400 K, cloud-free, chemical equilibrium. Self-consistent evolution models."),
    "elfowl": ("Elf Owl", "Mukherjee et al. 2024", "https://ui.adsabs.harvard.edu/abs/2024ApJ...963...73M/abstract",
               "275-2400 K, cloud-free, disequilibrium chemistry (Kzz), [M/H] -1 to +1, C/O 0.5-2.5x solar."),
    "diamondback": ("Diamondback", "Morley et al. 2024", "https://ui.adsabs.harvard.edu/abs/2024ApJ...975...59M/abstract",
                    "900-2400 K, cloudy (fsed 1-8) or cloud-free, chemical equilibrium. Self-consistent evolution models."),
    "flameskimmer": ("Flame Skimmer", "Mang et al. 2026", "https://zenodo.org/records/20030439",
                     "50-2400 K, log g 2-5.5, cloud-free, equilibrium and disequilibrium chemistry. Best for the coldest objects."),
}

PARAMETERS = (  # (column, label, unit)
    ("teff", "Teff", "K"),
    ("grav", "Gravity", "m/s²"),
    ("log_mh", "[M/H]", "log solar"),
    ("cto_relative", "C/O", "× solar"),
    ("logkzz", "log Kzz", "cm²/s"),
    ("fsed", "fsed", ""),
)
DEFAULT_MOLECULES = ("H2O", "CH4", "CO", "CO2", "NH3")
SCAN_SECONDS = 60  # rescan a grid folder at most this often

# Teff ranges of the download targets that split a grid by temperature
TEFF_RANGES = {"Ytype": (0, 560), "Ttype": (560, 1250), "Ltype": (1250, 3000)}


def grid_folder(root, grid):
    if not root:
        return None
    path = os.path.join(os.path.expanduser(root), grid)
    return path if os.path.isdir(path) else None


# =======================================
# WHAT IS ON DISK
# =======================================
@lru_cache(maxsize=16)
def _scan(folder, stamp):
    """Every file under `folder` as (name, path)."""
    out = []
    for root, _, files in os.walk(folder):
        out += [(f, os.path.join(root, f)) for f in files if not f.startswith(("._", "."))]
    return tuple(out)


def files(folder):
    return _scan(folder, int(time.time() // SCAN_SECONDS)) if folder else ()


@lru_cache(maxsize=16)
def _models(folder, grid, stamp):
    try:
        return sonora.sonora_grid_options(folder, grid=grid)
    except Exception:
        return pd.DataFrame(columns=[p for p, _, _ in PARAMETERS] + ["filename"])


def models(root, grid):
    """sonora.sonora_grid_options for one grid (empty if it is not on disk)."""
    folder = grid_folder(root, grid)
    if folder is None:
        return pd.DataFrame(columns=[p for p, _, _ in PARAMETERS] + ["filename"])
    return _models(folder, grid, int(time.time() // SCAN_SECONDS))


@lru_cache(maxsize=64)
def _elfowl_version(filename):
    """Elf Owl v1 and v2 files have the same names; v2 records the CO2 fix in its attributes."""
    try:
        with xr.open_dataset(filename) as ds:
            return "v2" if "CO2" in json.loads(ds.attrs.get("planet_params", "{}")) else "v1"
    except Exception:
        return None


def _in_range(teff, key):
    low, high = TEFF_RANGES[key]
    return (teff >= low) & (teff < high)


def count_on_disk(root, grid, target):
    """Number of models (or files) of download `target` found on disk, and a short note on what was counted."""
    folder = grid_folder(root, grid)
    if folder is None:
        return 0, ""
    names = [n for n, _ in files(folder)]
    df = models(root, grid)
    if target == "bobcat":
        return len(df), "profiles"
    if target == "bobcat-spectra":
        return sum(1 for n in names if re.match(r"^sp_t.*", n)), "spectra"
    if target == "diamondback":
        nclouds = sum(1 for n in names if n.endswith(".cld"))
        return len(df), f"profiles, {nclouds} cloud files"
    if target == "diamondback-spectra":
        return sum(1 for n in names if n.endswith(".spec")), "spectra"
    if target.startswith("elfowl"):
        _, kind, version = target.split("-")
        part = df[_in_range(df["teff"], kind)]
        if len(part) == 0 or _elfowl_version(part["filename"].iloc[0]) != version:
            return 0, ""
        return len(part), "models"
    if target.startswith("flameskimmer"):
        rows = [sonora._SONORA_PATTERNS["flameskimmer"].match(n) for n in names]
        rows = [m.groupdict() for m in rows if m]
        if target == "flameskimmer-quickstart":
            return sum(1 for r in rows if r["R"] == "3000"), "models (R=3,000)"
        full = [r for r in rows if r["R"] != "3000"]
        if target == "flameskimmer-eq":
            return sum(1 for r in full if not r["logkzz"]), "models"
        kind = target.split("-")[-1]
        return sum(1 for r in full if r["logkzz"] and _in_range(float(r["teff"]), kind)), "models"
    return 0, ""


@dataclass
class Target:
    name: str
    grid: str
    size_gb: float
    description: str
    found: int
    found_note: str


def catalog(root, sonora_config):
    """One Target per sonora_grids download (data.get_data_config()[1]['sonora_grids']), grouped by grid."""
    out = {grid: [] for grid in GRIDS}
    for name, info in sonora_config.items():
        grid = next((g for g in GRIDS if name.startswith(g)), None)
        if grid is None:
            continue
        found, note = count_on_disk(root, grid, name)
        out[grid].append(Target(name, grid, info.get("size_gb", float("nan")), info.get("description", ""), found, note))
    return out


# =======================================
# CHOOSING A MODEL
# =======================================
def parameter_values(df):
    """{parameter: sorted values on disk}; nan (equilibrium / cloud-free) is listed as None, first."""
    values = {}
    for key, _, _ in PARAMETERS:
        if key not in df or len(df) == 0:
            continue
        column = df[key].astype(float)
        values[key] = ([None] if column.isna().any() else []) + sorted(column.dropna().unique().tolist())
    return {k: v for k, v in values.items() if v != [None]}


def default_request(df, grid):
    """Starting choice for a grid: near 1000 K (300 K for Flame Skimmer), g=316 m/s2, solar."""
    values = parameter_values(df)
    nearest = lambda key, target: min((v for v in values.get(key, []) if v is not None), default=None,
                                      key=lambda v: abs(np.log10(v) - np.log10(target)) if key == "grav" else abs(v - target))
    request = {"teff": nearest("teff", 300 if grid == "flameskimmer" else 1000), "grav": nearest("grav", 316),
               "log_mh": nearest("log_mh", 0), "cto_relative": nearest("cto_relative", 1)}
    if "logkzz" in values:
        request["logkzz"] = None if None in values["logkzz"] else nearest("logkzz", 4)
    if "fsed" in values:
        request["fsed"] = None if None in values["fsed"] else nearest("fsed", 2)
    return request


def select(df, grid, request):
    """The model nearest to `request`, and a note when it is not an exact match."""
    row = sonora._nearest_sonora_model(df, grid, teff=request["teff"], grav=request["grav"],
                                       log_mh=request.get("log_mh"), cto_relative=request.get("cto_relative"),
                                       logkzz=request.get("logkzz"), fsed=request.get("fsed"), verbose=False)
    off = [label for key, label, _ in PARAMETERS
           if request.get(key) is not None and not pd.isna(row[key]) and not np.isclose(row[key], request[key])]
    note = f"No model has exactly these parameters; showing the nearest one (different {', '.join(off)})." if off else None
    return row, note


def model_label(grid, row):
    parts = [f"{row['teff']:g} K", f"g={row['grav']:g}"]
    if grid != "bobcat":
        parts += [f"[M/H]={row['log_mh']:+.1f}", f"C/O={row['cto_relative']:g}"]
    if grid in ("elfowl", "flameskimmer"):
        parts.append("eq" if pd.isna(row["logkzz"]) else f"Kzz=1e{row['logkzz']:g}")
    if grid == "diamondback":
        parts.append("cloud-free" if pd.isna(row["fsed"]) else f"fsed={row['fsed']:g}")
    return f"{GRIDS[grid][0]} " + ", ".join(parts)


@dataclass
class Model:
    grid: str
    filename: str
    label: str
    profile: pd.DataFrame
    cloud: pd.DataFrame       # None if there is no cloud
    spectrum: tuple           # (wavelength [micron], flux [erg/cm2/s/cm]) or None
    missing: list             # what could not be found, e.g. ["published spectrum (diamondback-spectra)"]


@lru_cache(maxsize=32)
def load(root, grid, filename, mtime):
    folder = grid_folder(root, grid)
    missing = []
    profile = sonora.read_sonora_model(filename)
    cloud = None
    if grid == "diamondback" and "nc_m" not in os.path.basename(filename):
        cld = sonora.find_sonora_cloud(filename, folder)
        if cld:
            cloud = pd.read_csv(cld, sep=r"\s+").rename(columns={"tau": "opd"})
        else:
            missing.append("cloud optical properties (diamondback)")
    spectrum = None
    published = sonora.find_sonora_spectrum(filename, folder)
    if published:
        spectrum = sonora.read_sonora_spectrum(published)
    else:
        missing.append(f"published spectrum ({grid}-spectra)")
    return profile, cloud, spectrum, missing


def load_model(root, grid, row):
    profile, cloud, spectrum, missing = load(root, grid, row["filename"], os.path.getmtime(row["filename"]))
    return Model(grid, row["filename"], model_label(grid, row), profile, cloud, spectrum, list(missing))


# =======================================
# FIGURES
# =======================================
COLORS = px.colors.qualitative.Plotly
DASHES = ("solid", "dash", "dot", "dashdot", "longdash", "longdashdot")


def _pressure_axis(fig, **kwargs):
    fig.update_yaxes(type="log", autorange="reversed", title_text="Pressure [bar]", **kwargs)


def pt_figure(models):
    fig = go.Figure()
    for i, m in enumerate(models):
        fig.add_trace(go.Scatter(x=m.profile["temperature"], y=m.profile["pressure"], mode="lines", name=m.label,
                                 line=dict(color=COLORS[i % len(COLORS)], width=2)))
    _pressure_axis(fig)
    fig.update_xaxes(title_text="Temperature [K]")
    fig.update_layout(title="Pressure-temperature profile", height=450, legend=dict(font=dict(size=11), orientation="h", yanchor="top", y=-0.18, x=0),
                      margin=dict(l=70, r=20, t=50, b=40))
    return fig


def molecules_in(models):
    """Molecules saved in any of the models, most abundant first."""
    peak = {}
    for m in models:
        for col in m.profile.columns:
            if col in ("pressure", "temperature", "kz"):
                continue
            peak[col] = max(peak.get(col, 0), float(np.nanmax(m.profile[col].values)))
    return sorted(peak, key=lambda c: -peak[c])


def chemistry_figure(models, molecules):
    fig = go.Figure()
    for j, mol in enumerate(molecules):
        for i, m in enumerate(models):
            if mol not in m.profile:
                continue
            fig.add_trace(go.Scatter(x=m.profile[mol], y=m.profile["pressure"], mode="lines",
                                     name=mol if len(models) == 1 else f"{mol}, {m.label}", legendgroup=mol,
                                     line=dict(color=COLORS[j % len(COLORS)], dash=DASHES[i % len(DASHES)], width=2)))
    _pressure_axis(fig)
    fig.update_xaxes(type="log", title_text="Volume mixing ratio", range=[-12, 0], exponentformat="power")
    note = " (line style: model)" if len(models) > 1 else ""
    fig.update_layout(title=f"Chemistry{note}", height=450, legend=dict(font=dict(size=11), orientation="h", yanchor="top", y=-0.18, x=0),
                      margin=dict(l=70, r=20, t=50, b=40))
    return fig


@lru_cache(maxsize=2)
def cloud_wavelengths(refdata):
    """Wavelength (micron) of each of the 196 cloud spectral points (wave_EGP.dat, in the reference data)."""
    path = os.path.join(refdata, "opacities", "wave_EGP.dat")
    if refdata and os.path.exists(path):
        return pd.read_csv(path, sep=r"\s+")["micron."].values
    return None


def cloud_figure(models, refdata, wavelength=1.0):
    """Optical depth per layer, single scattering albedo and asymmetry of the cloudy models at one wavelength."""
    cloudy = [(i, m) for i, m in enumerate(models) if m.cloud is not None]
    if not cloudy:
        return None
    waves = cloud_wavelengths(refdata)
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True,
                        subplot_titles=("Optical depth per layer", "Single scattering albedo", "Asymmetry (g0)"))
    for i, m in cloudy:
        nwave = int(m.cloud["spec_number"].max())
        layers = m.cloud["level_number"].values.reshape(-1, nwave)[:, 0]
        level_p = m.profile["pressure"].values
        pressure = np.sqrt(level_p[layers - 1] * level_p[np.minimum(layers, len(level_p) - 1)])
        iw = int(np.argmin(np.abs(waves - wavelength))) if (waves is not None and len(waves) == nwave) else nwave // 2
        shown = waves[iw] if (waves is not None and len(waves) == nwave) else None
        color = COLORS[i % len(COLORS)]
        opd = m.cloud["opd"].values.reshape(-1, nwave)[:, iw]
        for col, key in enumerate(("opd", "w0", "g0"), start=1):
            values = m.cloud[key].values.reshape(-1, nwave)[:, iw]
            values = np.where(opd > 0, values, np.nan)  # no cloud in this layer
            fig.add_trace(go.Scatter(x=values, y=pressure, mode="lines", name=m.label, legendgroup=m.label,
                                     showlegend=col == 1, line=dict(color=color, width=2)), row=1, col=col)
    _pressure_axis(fig, row=1, col=1)
    fig.update_yaxes(type="log", autorange="reversed")
    fig.update_xaxes(type="log", exponentformat="power", row=1, col=1)
    at = f" at {shown:.3g} μm" if shown else ""
    fig.update_layout(title=f"Cloud optical properties{at}", height=450, legend=dict(font=dict(size=11)))
    return fig


def bin_spectrum(wave, flux, R, wave_range):
    """Mean flux in bins of constant resolving power R between wave_range (micron)."""
    keep = (wave >= wave_range[0]) & (wave <= wave_range[1])
    wave, flux = wave[keep], flux[keep]
    if R is None or len(wave) < 2:
        return wave, flux
    edges = np.exp(np.arange(np.log(wave_range[0]), np.log(wave_range[1]) + 1 / R, 1 / R))
    index = np.digitize(wave, edges)
    sums = np.bincount(index, weights=flux, minlength=len(edges) + 1)
    counts = np.bincount(index, minlength=len(edges) + 1)
    good = counts[1:len(edges)] > 0
    centers = np.sqrt(edges[1:] * edges[:-1])
    return centers[good], (sums[1:len(edges)] / np.maximum(counts[1:len(edges)], 1))[good]


def spectrum_figure(models, R=None, wave_range=(0.5, 15), yscale="log"):
    with_spectra = [(i, m) for i, m in enumerate(models) if m.spectrum is not None]
    if not with_spectra:
        return None
    fig = go.Figure()
    for i, m in with_spectra:
        wave, flux = bin_spectrum(*m.spectrum, R, wave_range)
        fig.add_trace(go.Scatter(x=wave, y=flux, mode="lines", name=m.label,
                                 line=dict(color=COLORS[i % len(COLORS)], width=1.5)))
    fig.update_xaxes(title_text="Wavelength [μm]", range=list(wave_range))
    fig.update_yaxes(type=yscale, title_text="Flux [erg/cm<sup>2</sup>/s/cm]", exponentformat="power")
    resolution = f"R={R:g}" if R else "native resolution"
    fig.update_layout(title=f"Published spectrum ({resolution}, flux at the top of the atmosphere)", height=450,
                      legend=dict(font=dict(size=11)))
    return fig
