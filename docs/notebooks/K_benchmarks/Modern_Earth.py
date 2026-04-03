# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # The spectra of modern Earth
#
# This notebook computes the reflected-light, thermal emission and transmission spectra of Earth, broadly reproducing the spectra computed buy the `rfast` and `SMART` codes in [Robinson and Salvador (2023)](https://doi.org/10.3847/PSJ/acac9a).

# %%
# Imports
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import os
import astropy.units as u
from astropy import constants
from astropy.io import ascii
from picaso import justdoit as jdi 
from matplotlib import pyplot as plt

# %% [markdown]
# ## Opacities for rocky planets
#
# First, we need to download the following resampled opacity database specifically designed for rocky planets: [https://zenodo.org/records/17381172](https://zenodo.org/records/17381172). For this notebook, download the R = 15,000 file that spans 0.1 to 250 microns and, after unzipping, place the file in your PICASO `reference/opacities/` folder. These opacities are similar to those use for climate modeling in the Photochem code ([Wogan et al. (2025)](https://doi.org/10.3847/PSJ/ae0e1c)) just reformatted for spectral calculations with PICASO. You can also optionally download the high resolution file in the zenodo archive (R = 60,000), but it is not needed for this notebook.
#
# These opacities from Photochem are distinct from the "default" PICASO opacities (e.g., []()) in the following ways:
# | Feature | Photochem Opacities | Default PICASO Opacities |
# | --- | --- | --- |
# | Line broadening | Mostly Earth air | Hydrogen and helium |
# | CIA partners | Includes many rocky-planet relevant CIAs (e.g., CO2-CO2, H2O-H2O, etc.) | Limited CIA set (e.g., H2-H2, H2-He, etc.) |
# | Temperature coverage | Fewer absorbers, tuned for temperate atmospheres (roughly < 2000 K) | Broader absorber set relevant to hotter atmospheres (roughly < 6000 K) |
#
# Below, we create two opacities objects, one with small and one with big wavelength ranges:

# %%
filename_db = os.path.join(jdi.__refdata__, 'opacities', 'opacities_photochem_0.1_250.0_R15000.db')

opacities_small = jdi.opannection(
    wave_range=[0.2, 2.0], 
    filename_db=filename_db
)

opacities_big = jdi.opannection(
    wave_range=[0.2, 250.0], 
    filename_db=filename_db
)

# %% [markdown]
# ## Composition and temperature of Earth's atmosphere
#
# Here, our goal is to reproduce the spectral calculation from [Robinson & Salvador (2023)](https://doi.org/10.3847/PSJ/acac9a). They used mixing ratio and temperature profiles from the ICRCCM mid-latitude summer sounding for all their calculations, which we load with the cell below.

# %%
df_earth = pd.read_csv(jdi.earth_icrccm_pt(), sep=r'\s+')


# %% [markdown]
# ## Reproducing Robinson & Salvador 2023
#
# ### Reflected light
#
# First, we will reproduce the Earth reflected light spectrum in Figure 4 (right panel) of Robinson & Salvador (2023), by mirroring their model setup almost exactly. In their calculation, they use the P-T-composition as described above, as well as as cloud that blends water liquid/ice optical properties. 
#
# To dupilcate their setup, you must download the folder "hires_opacities" from Ty Robinson's Dropbox: [https://hablabnet.wordpress.com/research/#data-software](https://hablabnet.wordpress.com/research/#data-software). Follow the link, then click "Dropbox" then the "rfast" folder, then download "hires_opacities.zip" and unzip the folder to the same directory as this notebook.
#
# The functions below use the cloud mie files in that folder to mirror their setup.

# %%
def _interp_extrap(x, xp, fp):
    """Linearly interpolate and extrapolate onto x."""
    x = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    y = np.interp(x, xp, fp)

    if xp.size > 1:
        left = x < xp[0]
        right = x > xp[-1]

        if np.any(left):
            slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
            y[left] = fp[0] + slope * (x[left] - xp[0])

        if np.any(right):
            slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
            y[right] = fp[-1] + slope * (x[right] - xp[-1])

    return y


def _read_rfast_cloud_optics(wavelength_um, opdir, lamc0=0.55):
    """
    Read the rfast liquid/ice cloud optical property tables and blend them 50/50.

    Returns wavelength-dependent cloud single-scattering albedo, asymmetry
    parameter, and extinction efficiency normalized at lamc0, following the
    logic in rfast_opac_routines.py.
    """
    liquid_path = os.path.join(opdir, "strato_cum.mie")
    ice_path = os.path.join(opdir, "baum_cirrus_de100.mie")

    if not (os.path.exists(liquid_path) and os.path.exists(ice_path)):
        raise FileNotFoundError(
            "Could not find rfast cloud Mie tables. Expected "
            f"{liquid_path} and {ice_path}."
        )

    liquid = ascii.read(liquid_path, data_start=20, delimiter=" ")
    ice = ascii.read(ice_path, data_start=2, delimiter=" ")

    wcl = _interp_extrap(wavelength_um, liquid["col1"], liquid["col10"])
    gcl = _interp_extrap(wavelength_um, liquid["col1"], liquid["col11"])
    qcl_raw = _interp_extrap(wavelength_um, liquid["col1"], liquid["col7"])
    qcl_ref = _interp_extrap(np.array([lamc0]), liquid["col1"], liquid["col7"])[0]
    qcl = qcl_raw / qcl_ref

    wci = _interp_extrap(wavelength_um, ice["wl"], ice["omega"])
    gci = _interp_extrap(wavelength_um, ice["wl"], ice["g"])
    qci_raw = _interp_extrap(wavelength_um, ice["wl"], ice["Qe"])
    qci_ref = _interp_extrap(np.array([lamc0]), ice["wl"], ice["Qe"])[0]
    qci = qci_raw / qci_ref

    frac_liquid = 0.5
    w0 = frac_liquid * wcl + (1.0 - frac_liquid) * wci
    g0 = frac_liquid * gcl + (1.0 - frac_liquid) * gci
    qext = frac_liquid * qcl + (1.0 - frac_liquid) * qci
    return w0, g0, qext


def _build_rfast_like_cloud_df(pressure_levels_bar, wavenumber, opdir, ptop=0.6,
                               pbot=0.7, tauc0=10.0, lamc0=0.55):
    """
    Build a PICASO cloud dataframe using the same slab location and optical-depth
    normalization as the rfast Earth reflected-light setup.
    """
    wavelength_um = 1e4 / np.asarray(wavenumber, dtype=float)

    w0_wave, g0_wave, qext_wave = _read_rfast_cloud_optics(wavelength_um, opdir, lamc0=lamc0)
    
    tau_wave = tauc0 * qext_wave

    level_pressure = np.asarray(pressure_levels_bar, dtype=float)
    layer_pressure = np.sqrt(level_pressure[:-1] * level_pressure[1:])
    cloud_thickness = pbot - ptop

    rows = []
    for ilay, pmid in enumerate(layer_pressure):
        player_top = level_pressure[ilay]
        player_bot = level_pressure[ilay + 1]
        overlap = max(0.0, min(player_bot, pbot) - max(player_top, ptop))
        tau_fraction = overlap / cloud_thickness if cloud_thickness > 0 else 0.0
        opd_layer = tau_wave * tau_fraction

        rows.append(
            pd.DataFrame(
                {
                    "pressure": np.full_like(wavenumber, pmid, dtype=float),
                    "wavenumber": np.asarray(wavenumber, dtype=float),
                    "opd": opd_layer,
                    "w0": w0_wave,
                    "g0": g0_wave,
                }
            )
        )

    return pd.concat(rows, ignore_index=True)


def _regrid_picaso_outputs(wno, spectrum_output, R):
    primary_key = None
    secondary = None

    if "fpfs_total" in spectrum_output:
        primary_key = "fpfs_total"
        secondary = {}
        for key in ("albedo", "fpfs_reflected", "thermal", "fpfs_thermal"):
            if key in spectrum_output:
                _, secondary[key] = jdi.mean_regrid(wno, spectrum_output[key], R=R)
    elif "fpfs_reflected" in spectrum_output:
        primary_key = "fpfs_reflected"
        if "albedo" in spectrum_output:
            _, secondary = jdi.mean_regrid(wno, spectrum_output["albedo"], R=R)
    elif "fpfs_thermal" in spectrum_output:
        primary_key = "fpfs_thermal"
        if "thermal" in spectrum_output:
            _, secondary = jdi.mean_regrid(wno, spectrum_output["thermal"], R=R)
    elif "transit_depth" in spectrum_output:
        primary_key = "transit_depth"
    else:
        raise KeyError(
            "No supported rebinnable PICASO output was found. Expected one of "
            "fpfs_reflected, fpfs_thermal, fpfs_total, or transit_depth."
        )

    wno_regrid, primary = jdi.mean_regrid(wno, spectrum_output[primary_key], R=R)
    return wno_regrid, primary, secondary

def initialize_earth(
    opacity,
    phase=0.0,
    surface_albedo=0.05,
    stellar_teff=5780.0,
    stellar_metallicity=0.0,
    stellar_logg=4.0,
    semi_major=1.0,
    stellar_radius=1.0,
    planet_radius=1.0,
    planet_mass=1.0,
):
    """
    Build a PICASO Earth-like input bundle with the geometry, gravity, star,
    and surface settings used by the local rfast Earth benchmark helper.

    Parameters
    ----------
    opacity : picaso opacity connection
        Opacity object created with ``jdi.opannection``.
    phase : float, optional
        Orbital phase angle in radians passed to ``phase_angle``.
        Default ``0.0`` gives full phase.
    surface_albedo : float, optional
        Lambertian surface albedo applied with ``surface_reflect``.
        Default ``0.05`` matches the local rfast Figure 4 setup.
    stellar_teff : float, optional
        Stellar effective temperature in K. Default ``5780.0``.
    stellar_metallicity : float, optional
        Stellar metallicity [M/H]. Default ``0.0``.
    stellar_logg : float, optional
        Stellar surface gravity in cgs log10 units. Default ``4.0``.
    semi_major : float, optional
        Semi-major axis in au. Default ``1.0``.
    stellar_radius : float, optional
        Stellar radius in solar radii. Default ``1.0``.
    planet_radius : float, optional
        Planet radius in Earth radii. Default ``1.0``.
    planet_mass : float, optional
        Planet mass in Earth masses. Default ``1.0``.

    Returns
    -------
    picaso.justdoit.inputs
        Configured PICASO input bundle ready for ``atmosphere(...)`` and
        ``spectrum(...)``.
    """
    earth = jdi.inputs()
    earth.phase_angle(phase)
    earth.gravity(
        radius=planet_radius,
        radius_unit=u.Unit("R_earth"),
        mass=planet_mass,
        mass_unit=u.Unit("M_earth"),
    )
    earth.star(
        opacity,
        stellar_teff,
        stellar_metallicity,
        stellar_logg,
        semi_major=semi_major,
        radius=stellar_radius,
        radius_unit=u.Unit("R_sun"),
        semi_major_unit=u.Unit("au"),
    )
    earth.surface_reflect(surface_albedo, opacity.wno)
    return earth


def earth_spectrum_like_rfast(
    opacity,
    earth,
    df_earth,
    opdir,
    calculation="reflected",
    exclude_mol=None,
    R=140,
    cloud_frac=0.5,
    ptop=0.6,
    pbot=0.7,
    tauc0=10.0,
    lamc0=0.55,
):
    """
    Compute an Earth spectrum in PICASO using the rfast Earth benchmark cloud
    deck and surface settings.

    Parameters
    ----------
    opacity : picaso opacity connection
        Opacity object created with ``jdi.opannection``.
    df_atmo_earth : pandas.DataFrame
        Atmospheric profile dataframe with pressure in bar, temperature in K,
        and gas mixing ratio columns compatible with ``earth.atmosphere(df=...)``.
    opdir : str
        Directory containing the rfast cloud Mie tables
        ``strato_cum.mie`` and ``baum_cirrus_de100.mie``.
    calculation : str, optional
        PICASO calculation mode passed to ``earth.spectrum``. Common values are
        ``"reflected"`` and ``"thermal"``. Default ``"reflected"`` preserves
        the Figure 4 reflected-light benchmark behavior.
    initialize_earth_kwargs : dict, optional
        Keyword arguments passed directly to ``initialize_earth``. This is the
        place to override the shared setup inputs such as ``phase``,
        ``surface_albedo``, ``stellar_teff``, ``stellar_metallicity``,
        ``stellar_logg``, ``semi_major``, ``stellar_radius``,
        ``planet_radius``, and ``planet_mass``. If omitted, the
        ``initialize_earth`` defaults are used, which reproduce the local
        rfast Earth benchmark setup.
    exclude_mol : str, list of str, or dict, optional
        Opacity-only exclusion control passed through to ``earth.atmosphere``.
        Strings/lists exclude all opacity types for the named molecules; dict
        values allow explicit selection of ``line``, ``continuum``,
        ``rayleigh``, or ``all``.
    R : int, optional
        Output resolving power for ``jdi.mean_regrid``.
    cloud_frac : float, optional
        Fractional cloud coverage. Default ``0.5`` matches ``fc`` in the local
        rfast Figure 4 benchmark setup.
    ptop : float, optional
        Cloud-top pressure in bar. Default ``0.6`` matches the local rfast
        Figure 4 benchmark setup.
    pbot : float, optional
        Cloud-base pressure in bar. Default ``0.7`` matches the local rfast
        Figure 4 benchmark setup.
    tauc0 : float, optional
        Cloud extinction optical depth at ``lamc0``. Default ``10.0`` matches
        the local rfast Figure 4 benchmark setup.
    lamc0 : float, optional
        Wavelength in micron where ``tauc0`` is specified. Default ``0.55``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(wno, primary, secondary)`` on the rebinned output grid, where
        ``wno`` is in cm^-1. For ``calculation="reflected"``, ``primary`` is
        ``fpfs_reflected`` and ``secondary`` is geometric albedo. For
        ``calculation="thermal"``, ``primary`` is ``fpfs_thermal`` and
        ``secondary`` is thermal flux. For ``calculation="transmission"``,
        ``primary`` is transit depth and ``secondary`` is ``None``. If both
        reflected and thermal outputs are present, ``primary`` is
        ``fpfs_total`` and ``secondary`` is a dict containing any rebinned
        component arrays that were returned.
    """
   
    # Set atmosphere
    earth.atmosphere(df=df_earth, exclude_mol=exclude_mol)
    
    # Build a rfast-like cloud
    cloud_df = _build_rfast_like_cloud_df(
        pressure_levels_bar=df_earth["pressure"].values,
        wavenumber=opacity.wno,
        opdir=opdir,
        ptop=ptop,
        pbot=pbot,
        tauc0=tauc0,
        lamc0=lamc0,
    )
    
    # Set the cloud
    weight_clear = 1.0 - cloud_frac
    earth.clouds(df=cloud_df, do_holes=True, fhole=weight_clear, fthin_cld=0.0)
    
    # Compute spectrum
    df = earth.spectrum(opacity, calculation=calculation, full_output=True)
    
    wno = df["wavenumber"]
    return _regrid_picaso_outputs(wno, df, R=R)


# %% [markdown]
# Lets compute a couple spectra, excluding one molecule at a time, so we can see their contribution.

# %%
earth = initialize_earth(opacities_small)

exclude_mols = [None, 'H2O', 'CO2', 'CH4', 'O2', 'O3']

res_reflected = {}
for exclude_mol in exclude_mols:
    key = exclude_mol
    if key is None:
        key = 'all'
        
    res_reflected[key] = earth_spectrum_like_rfast(
        opacity=opacities_small,
        earth=earth,
        df_earth=df_earth,
        opdir='hires_opacities',
        exclude_mol=exclude_mol
    )

# %%
fig, ax = plt.subplots(1, 1, figsize=[6, 4.5])
wno, fpfs, albedo = res_reflected['all']
ax.plot(1e4/wno, albedo, c='k', lw=1, label='Modern Earth')

# Plot molecular contributions
for key in res_reflected:
    if key in ['all']:
        continue
    _, fpfs1, albedo1 = res_reflected[key]
    ax.fill_between(1e4/wno, albedo, albedo1, label=key, alpha=0.5)

ax.set_xlim(0.2, 2.0)
ax.set_ylim(0., 0.45)
ax.set_ylabel('Geometric Albedo')
ax.set_xlabel('Wavelength (microns)')
ax.legend()

plt.show()

# %% [markdown]
# The plot above does a decent job of reproducing Figure 4 (right) of Robinson & Salvador (2023). There are minor differences in the continuum (set by clouds largely), which I would guess are probably due to different treatments of mutiple scattering.

# %% [markdown]
# ### Thermal emission
#
# Now, lets reproduce Figure 5 in Robinson & Salvador (2023). We can simply use the `earth_spectrum_like_rfast`, but this time with the clouds zeroed out, to match the paper.

# %%
earth = initialize_earth(opacities_big)

exclude_mols = [None, 'H2O', 'CO2', 'CH4', 'O2', 'O3']

res_thermal = {}
for exclude_mol in exclude_mols:
    key = exclude_mol
    if key is None:
        key = 'all'
        
    res_thermal[key] = earth_spectrum_like_rfast(
        opacity=opacities_big,
        earth=earth,
        df_earth=df_earth,
        opdir='hires_opacities',
        calculation="thermal",
        exclude_mol=exclude_mol,
        cloud_frac=0.0
    )

# %%
fig, ax = plt.subplots(1, 1, figsize=[6, 4.5])
wno, fpfs, thermal = res_thermal['all']

ax.plot(1e4/wno, thermal/1e7, c='k', lw=1, label='Modern Earth')

# Plot molecular contributions
for key in res_thermal:
    if key in ['all']:
        continue
    _, fpfs1, thermal1 = res_thermal[key]
    ax.fill_between(1e4/wno, thermal/1e7, thermal1/1e7, label=key, alpha=0.5)

ax.set_xlim(5, 30.0)
ax.set_ylim(0, 30.0)
ax.set_ylabel('Thermal flux (W/m$^2$/micron)')
ax.set_xlabel('Wavelength (microns)')
ax.legend()

plt.show()

# %% [markdown]
# ## Transmission
#
# Finally, we reproduce their Figure 6, which is a cloud-free tranmission spectrum of Earth. We do not include any approximation for refraction.

# %%
earth = initialize_earth(opacities_big)

exclude_mols = [None, 'H2O', 'CO2', 'CH4', 'O2', 'O3']

res_transit = {}
for exclude_mol in exclude_mols:
    key = exclude_mol
    if key is None:
        key = 'all'
        
    res_transit[key] = earth_spectrum_like_rfast(
        opacity=opacities_big,
        earth=earth,
        df_earth=df_earth,
        opdir='hires_opacities',
        calculation="transmission",
        exclude_mol=exclude_mol,
        cloud_frac=0.0
    )

# %%
Rs = constants.R_sun.to(u.km).value
Rp = constants.R_earth.to(u.km).value

fig, ax = plt.subplots(1, 1, figsize=[6, 4.5])
wno, rprs2, _ = res_transit['all']
z_eff = Rs * np.sqrt(rprs2) - Rp
ax.plot(1e4/wno, z_eff, c='k', lw=1, label='Modern Earth')

# Plot molecular contributions
for key in res_transit:
    if key in ['all']:
        continue
    _, rprs21, _ = res_transit[key]
    z_eff1 = Rs * np.sqrt(rprs21) - Rp
    ax.fill_between(1e4/wno, z_eff, z_eff1, label=key, alpha=0.5)

ax.set_xlim(0.5, 20.0)
ax.set_ylim(0, 60.0)
ax.set_xscale('log')
ax.set_ylabel('Effective transit altitude (km)')
ax.set_xlabel('Wavelength (microns)')
ax.legend()

plt.show()
