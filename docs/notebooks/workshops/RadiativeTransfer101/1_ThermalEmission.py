# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Thermal Emission: Building Intuition from Limiting Cases
#
# A radiative transfer code turns a temperature profile and optical properties into the light
# that leaves the planet. Before trusting it on a real atmosphere, it helps to ask what it should
# give in situations where we already know the answer. Each of those limiting cases teaches a piece
# of the physics, and together they tell us the code is working.
#
# This lesson calls PICASO's thermal solver, `picaso.fluxes.get_thermal_1d`, directly. It solves the
# two-stream equations with the source function technique of
# [Toon et al. (1989)](https://ui.adsabs.harvard.edu/abs/1989JGR....9416287T/abstract), the same
# routine used for `spectrum(calculation='thermal')` and inside every climate model iteration.
# We give it small made-up atmospheres, so no opacity data is needed.
#
# **What you will learn**
#
# 1. Why an isothermal atmosphere always looks like a blackbody
# 2. How a surface's reflectivity sets how much it emits
# 3. How a temperature gradient creates absorption features, limb darkening and brightness temperature
# 4. Why scattering makes an atmosphere emit less than a blackbody
#
# Every section ends with a check (`assert`). The same checks run automatically as PICASO's unit
# tests in `tests/test_thermal_rt.py`, so if the code ever stops obeying this physics, both fail.

# %%
import numpy as np
import matplotlib.pyplot as plt

from picaso.fluxes import blackbody, blackbody_integrated, get_thermal_1d

# colors used throughout
BLUE, ORANGE, AQUA, YELLOW, INK = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#52514e'

# %% [markdown]
# ## Setting up a toy atmosphere
#
# `get_thermal_1d` works on **levels** (boundaries between layers, where temperature is defined) and
# **layers** (slabs between levels, which carry the optical properties):
#
# * `tlevel` : temperature at each of the `nlevel` levels, top to bottom (K)
# * `dtau` : optical depth of each of the `nlevel - 1` layers, at each wavenumber
# * `w0` : single scattering albedo, the fraction of extinction that is scattering rather than absorption
# * `cosb` : asymmetry parameter, how forward-peaked the scattering is
# * `ubar1` : cosine of the angle we look at the planet from (1 = straight down, 0 = grazing the limb)
#
# It returns the outgoing flux at the top for every viewing angle and wavenumber. The units are
# 2&pi; times intensity, so **a blackbody at temperature T comes out as 2&pi;B(T)**. (PICASO's
# `disco` module later integrates over the disk.) We wrap the call in a small helper:

# %%
WNO = np.linspace(200, 6000, 300)  # wavenumber grid, cm-1 (1.7 - 50 micron)
DWNO = np.gradient(WNO)            # bin widths, only used by the climate version of the Planck function
UBAR1 = np.array([[0.1], [0.5], [0.95]])  # three viewing angles: near the limb, halfway, disk center
ANGLE_LABELS = ['limb ($\\mu$=0.1)', 'middle ($\\mu$=0.5)', 'center ($\\mu$=0.95)']


def thermal(temperature, dtau, w0=1e-6, cosb=0.0, hard_surface=0, surf_reflect=0.0, integrated_energy=False,
            get_lvl_flux=0):
    """
    Runs get_thermal_1d. dtau can be a number (same for every layer and wavenumber), a per-layer
    array, or a full (nlayer, nwno) array. w0 defaults to almost zero: PICASO always has a little
    Rayleigh scattering, and exactly zero divides by zero inside the solver.
    """
    temperature = np.asarray(temperature, dtype=float)
    nlevel, nwno = len(temperature), len(WNO)
    nlayer = nlevel - 1
    dtau = np.asarray(dtau, dtype=float)
    if dtau.ndim == 1:
        dtau = dtau[:, np.newaxis]
    dtau = np.broadcast_to(dtau, (nlayer, nwno)).copy()
    numg, numt = UBAR1.shape
    flux_at_top, level_fluxes = get_thermal_1d(
        nlevel, WNO, nwno, numg, numt, temperature, dtau,
        np.full((nlayer, nwno), w0), np.full((nlayer, nwno), cosb),
        np.logspace(1, 6, nlevel),  # level pressures (dyne/cm2)
        UBAR1, np.full(nwno, surf_reflect), hard_surface, DWNO, integrated_energy,
        get_lvl_flux=get_lvl_flux)  # level fluxes are only computed when asked for
    return flux_at_top, level_fluxes


def two_pi_planck(temperature):
    """2 pi B(T) in the same units get_thermal_1d returns."""
    return 2 * np.pi * blackbody(np.array([float(temperature)]), 1 / WNO)[0]


def wavelength_axis(ax):
    ax.set_xlabel('Wavelength [$\\mu$m]')
    ax.set_xscale('log')
    ax.set_xticks([2, 3, 5, 10, 20, 50], labels=['2', '3', '5', '10', '20', '50'])
    ax.minorticks_off()


WAVE = 1e4 / WNO  # micron

# %% [markdown]
# ## 1. An isothermal atmosphere is a blackbody, at any optical depth
#
# Along a ray, the intensity obeys $dI/d\tau = I - S$, where the source function $S$ is what the gas
# emits. For gas that only absorbs, $S = B(T)$. If the temperature is the same everywhere, the
# solution through a slab of optical depth $\tau$ is
#
# $$ I_{\rm top} = I_{\rm bottom}\, e^{-\tau} + B(T)\,(1 - e^{-\tau}). $$
#
# Below the limit of what we are modeling in a gas giant's atmosphere there is, of course, more atmosphere. 
# Here we assume it is also at the same temperature, so the light entering from below is also $B(T)$. 
# Then $I_{\rm top} = B(T)$ regardless of **whatever $\tau$
# is**. A thin atmosphere lets us see the deep blackbody, a thick one is its own blackbody, and they
# are the same. Let's check.

# %%
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(WAVE, two_pi_planck(1000), color=INK, linewidth=6, alpha=0.25, label='2$\\pi$B(1000 K)')
for dtau, color, style in zip([1e-4, 1, 100], [BLUE, ORANGE, AQUA], ['--', ':', '-.']):
    flux, _ = thermal(np.full(11, 1000.), dtau)
    ax.plot(WAVE, flux[1, 0], color=color, linewidth=1.5, linestyle=style, label=f'layer $\\tau$ = {dtau:g}')
wavelength_axis(ax)
ax.set_ylabel('Outgoing flux [erg/s/cm$^2$/cm]')
ax.set_title('Isothermal, absorbing atmosphere: all three optical depths overlap')
ax.legend(frameon=False)
plt.show()

# %%
for dtau in [1e-4, 0.1, 1, 10, 100]:
    flux, _ = thermal(np.full(11, 1000.), dtau)
    np.testing.assert_allclose(flux, np.broadcast_to(two_pi_planck(1000), flux.shape), rtol=1e-6)
print('Isothermal atmospheres emit 2 pi B(T) at every optical depth and angle')

# %% [markdown]
# The climate solver uses a slightly different Planck function, averaged over each wavenumber bin
# (`integrated_energy=True`), because it needs the energy in each bin rather than the value at its center.
# The physics is the same:

# %%
flux, _ = thermal(np.full(11, 1000.), 1.0, integrated_energy=True)
expected = 2 * np.pi * blackbody_integrated(np.array([1000.]), WNO, DWNO)[0]
np.testing.assert_allclose(flux, np.broadcast_to(expected, flux.shape), rtol=1e-6)
print('...and so does the bin-integrated version used by the climate code')

# %% [markdown]
# ### Energy balance deep down
#
# If the atmosphere is the same temperature everywhere, nothing should heat or cool. Deep inside a
# thick isothermal atmosphere the upward and downward fluxes should cancel exactly. With
# `get_lvl_flux=1`, the second output of `get_thermal_1d` holds the downward and upward fluxes at
# each level (spectra skip them by default, because only the flux at the top is needed):

# %%
_, (flux_minus, flux_plus, _, _) = thermal(np.full(11, 1000.), 20.0, get_lvl_flux=1)
net_deep = flux_plus[:, :, -3, :] - flux_minus[:, :, -3, :]
np.testing.assert_allclose(net_deep / two_pi_planck(1000), 0, atol=1e-10)
print(f'largest net flux deep in the atmosphere: {np.abs(net_deep).max():.1e} (zero)')

# %% [markdown]
# ## 2. A transparent atmosphere over a surface: Kirchhoff's law
#
# A rocky planet has a hard surface (`hard_surface=1`). A surface that reflects a fraction $r$ of
# light can only emit a fraction $1 - r$ of a blackbody; its **emissivity** is $\epsilon = 1 - r$
# (Kirchhoff's law). If the atmosphere above is transparent, we should see exactly
# $(1 - r)\,B(T_{\rm surface})$.

# %%
temperature = np.linspace(600, 1200, 11)  # surface (bottom level) at 1200 K
fig, ax = plt.subplots(figsize=(7, 4))
for surf_reflect, color in zip([0.0, 0.3, 0.6], [BLUE, ORANGE, AQUA]):
    flux, _ = thermal(temperature, 1e-7, hard_surface=1, surf_reflect=surf_reflect)
    expected = (1 - surf_reflect) * two_pi_planck(1200)
    ax.plot(WAVE, expected, color=color, linewidth=6, alpha=0.25)
    ax.plot(WAVE, flux[1, 0], color=color, linewidth=1.5, linestyle='--', label=f'reflectivity {surf_reflect}')
    np.testing.assert_allclose(flux, np.broadcast_to(expected, flux.shape), rtol=1e-4)
wavelength_axis(ax)
ax.set_ylabel('Outgoing flux [erg/s/cm$^2$/cm]')
ax.set_title('Transparent atmosphere over a 1200 K surface\n(thick lines: (1 - r) 2$\\pi$B, dashed: PICASO)')
ax.legend(frameon=False)
plt.show()

# %% [markdown]
# ## 3. Temperature gradients: where spectral features come from
#
# Real atmospheres typically get hotter with depth (unless there is a thermal inversion). 
# Now the light we receive comes mostly from around optical
# depth $\tau \approx \mu$ along our line of sight, so:
#
# * The emission must lie between $B(T_{\rm top})$ and $B(T_{\rm bottom})$.
# * Looking straight down ($\mu$ near 1) we see deeper, hotter gas than looking near the limb, so the
#   disk is brighter in the center: **limb darkening**.
# * Where the gas absorbs strongly (a molecular band), $\tau \approx 1$ is reached higher up where it
#   is cooler, so the band looks **dark**. This is why molecules show up as absorption features in
#   emission spectra when temperature increases with depth.
#
# A handy way to see this is the **brightness temperature**, the temperature of the blackbody that
# would give the same flux. It tells us roughly which temperature, and so which altitude, we are seeing.

# %%
def brightness_temperature(flux):
    """Inverts F = 2 pi B(T) at each wavenumber (blackbody is per unit wavelength in cm)."""
    h, c, k = 6.62607004e-27, 2.99792458e+10, 1.38064852e-16
    wave_cm = 1 / WNO
    planck = flux / (2 * np.pi)
    return (h * c / (wave_cm * k)) / np.log(1 + 2 * h * c**2 / (wave_cm**5 * planck))


temperature = np.linspace(600, 1200, 11)
continuum = 0.2
band = 30 * np.exp(-0.5 * ((WNO - 1600) / 150)**2)  # a strong absorption band near 6 micron
dtau = np.broadcast_to(continuum + band, (10, len(WNO)))

flux, _ = thermal(temperature, dtau)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
for i, color in enumerate([BLUE, ORANGE, AQUA]):
    ax1.plot(WAVE, brightness_temperature(flux[i, 0]), color=color, label=ANGLE_LABELS[i])
ax1.axhline(600, color=INK, linestyle=':', linewidth=1)
ax1.axhline(1200, color=INK, linestyle=':', linewidth=1)
ax1.text(WAVE.min(), 615, 'T top', color=INK)
ax1.text(WAVE.min(), 1160, 'T bottom', color=INK)
wavelength_axis(ax1)
ax1.set_ylabel('Brightness temperature [K]')
ax1.set_title('A band seen in emission')
ax1.legend(frameon=False, loc='upper right')

ax2.plot(temperature, np.arange(11), color=INK)
ax2.invert_yaxis()
ax2.set_xlabel('Temperature [K]')
ax2.set_ylabel('Level (0 = top)')
ax2.set_title('Temperature profile')
fig.tight_layout()
plt.show()

# %% [markdown]
# The band is darker (cooler) because we see higher up there, and every curve is brighter the more
# directly we look down. The same checks, on a band-free version:

# %%
flux, _ = thermal(temperature, 2.0)
assert np.all(flux > two_pi_planck(temperature[0])) and np.all(flux < two_pi_planck(temperature[-1]))
limb, middle, center = flux[:, 0, :]
assert np.all(limb < middle) and np.all(middle < center), 'no limb darkening'
hotter, _ = thermal(temperature + 100, 2.0)
assert np.all(hotter > flux)
print('Emission is bounded by the top and bottom temperatures, limb darkened, and brighter when hotter')

# %% [markdown]
# **Try it:** make the temperature *decrease* with depth (a thermal inversion, like Earth's
# stratosphere or a strongly irradiated hot Jupiter) by using `np.linspace(1200, 600, 11)`. What
# happens to the band?

# %% [markdown]
# ## 4. Scattering makes an atmosphere emit less than a blackbody
#
# If some of the extinction is scattering ($w_0 > 0$), photons emitted deep down get bounced around
# and some are scattered back down, while the gas emits less because it absorbs less (Kirchhoff
# again: emission goes with absorption). An isothermal, scattering atmosphere therefore emits **less**
# than $B(T)$, and less the more it scatters. For isotropic scattering
# [Chandrasekhar (1960)](https://ui.adsabs.harvard.edu/abs/1960ratr.book.....C/abstract) solved this
# exactly: $I(\mu) = \sqrt{1 - w_0}\, H(\mu)\, B$, with $H \ge 1$, so the emission can never drop
# below $\sqrt{1 - w_0}\, B$.
#
# One PICASO detail matters here. Above the top level, `get_thermal_1d` imagines the atmosphere
# continuing upward at the same temperature, with an optical depth set by the top layer. This keeps
# the climate solver's energy balance smooth at the top. If the top layer were thick, that imagined
# gas would shine $B(T)$ back down and hide the effect. So we make the top layers thin, as they are
# in a real atmosphere:

# %%
thin_top_dtau = np.logspace(-5, 2, 10)  # per-layer optical depth, thin at the top, thick below
w0_values = np.array([1e-6, 0.2, 0.5, 0.7, 0.9, 0.95, 0.99])
ratios = np.array([thermal(np.full(11, 1000.), thin_top_dtau, w0=w0)[0][:, 0, 0] / two_pi_planck(1000)[0]
                   for w0 in w0_values])

fig, ax = plt.subplots(figsize=(7, 4))
for i, color in enumerate([BLUE, ORANGE, AQUA]):
    ax.plot(w0_values, ratios[:, i], 'o-', color=color, label=ANGLE_LABELS[i])
w0_fine = np.linspace(0, 0.99, 100)
ax.plot(w0_fine, np.sqrt(1 - w0_fine), color=INK, linestyle=':', label='$\\sqrt{1 - w_0}$ (lower bound)')
ax.set_xlabel('Single scattering albedo $w_0$')
ax.set_ylabel('Emission / blackbody')
ax.set_title('Isothermal scattering atmosphere')
ax.legend(frameon=False)
plt.show()

# %%
assert np.all(np.diff(ratios, axis=0) < 0), 'more scattering should mean less emission'
assert np.all(ratios[1:] > np.sqrt(1 - w0_values[1:])[:, np.newaxis]), 'emission below sqrt(1 - w0) B'
print('Scattering lowers emission, and never below the analytic sqrt(1 - w0) bound')

# %% [markdown]
# Notice that scattering also brightens the disk center relative to the limb, even with no
# temperature gradient: near the limb we see the top of the atmosphere, where photons escape most
# easily, so less is left to come out.
#
# ## Summary
#
# | Limiting case | What PICASO gives | Physics |
# |---|---|---|
# | Isothermal, absorbing | $2\pi B(T)$ at any $\tau$ | source function is $B(T)$ everywhere |
# | Transparent over a surface | $(1-r)\,2\pi B(T_s)$ | Kirchhoff's law, emissivity $= 1 - r$ |
# | Temperature increasing with depth | between $B(T_{\rm top})$ and $B(T_{\rm bottom})$, limb darkened | we see $\tau \approx \mu$ |
# | Absorption band | lower brightness temperature in the band | band reaches $\tau = 1$ higher, where it is cooler |
# | Scattering, isothermal | below $B$, above $\sqrt{1-w_0}\,B$ | emission goes with absorption |
#
# Next in this course: the same approach for {doc}`reflected light </notebooks/workshops/RadiativeTransfer101/2_ReflectedLight>`, where the limits are a black planet,
# a perfectly reflecting cloud deck that conserves energy, and a bare Lambertian surface.
