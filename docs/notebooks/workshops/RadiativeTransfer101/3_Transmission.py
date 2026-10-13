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
# # Transmission: Building Intuition from Limiting Cases
#
# When a planet crosses its star, starlight filters through the thin ring of atmosphere around the
# planet's edge (the limb). Wavelengths where the gas absorbs strongly are blocked higher up, so the
# planet looks slightly bigger there. The **transit depth** $(R_p/R_\star)^2$ at each wavelength is a
# transmission spectrum.
#
# PICASO computes it with `picaso.fluxes.get_transit_1d`, following
# [Brown (2001)](https://ui.adsabs.harvard.edu/abs/2001ApJ...553.1006B/abstract). This is what
# `spectrum(calculation='transmission')` uses. As in the
# {doc}`thermal </notebooks/workshops/RadiativeTransfer101/1_ThermalEmission>` and {doc}`reflected light </notebooks/workshops/RadiativeTransfer101/2_ReflectedLight>` lessons, we test it on
# simple atmospheres where we know the answer. No opacity data is needed.
#
# **What you will learn**
#
# 1. How the transit depth is built from rays (chords) passing through the atmosphere
# 2. Two limits: a transparent and an opaque atmosphere, and what a cloud deck does
# 3. The scale height: why e times more opacity makes the planet one scale height bigger
# 4. Where the transit radius actually is (optical depth ~0.56 along the grazing ray)
# 5. The Rayleigh slope, and why heavy atmospheres have much smaller spectral features
#
# Every section ends with a check. The same checks run as PICASO's unit tests in
# `tests/test_transit_rt.py`.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from picaso.fluxes import get_transit_1d

BLUE, ORANGE, AQUA, YELLOW, INK = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#52514e'

# %% [markdown]
# ## Setting up a toy atmosphere
#
# In hydrostatic equilibrium, pressure falls by a factor of e every **scale height**
#
# $$ H = \frac{k_B T}{\mu\, m_u\, g}, $$
#
# where $\mu$ is the mean molecular weight. For a hot Jupiter ($T$ = 1000 K, $\mu$ = 2.3, $g$ = 25
# m/s$^2$), $H \approx 145$ km. We build an isothermal atmosphere on a pressure grid that is evenly
# spaced in $\ln P$, so each level sits at exactly $z = R_p + H \ln(P_{\rm bottom}/P)$.
#
# `get_transit_1d` takes:
#
# * `z`, `dz` : altitude of each level (top first) and the spacing between levels (cm)
# * the pressure and temperature of each level, to get the number density $n = P/k_BT$
# * `DTAU` : the optical depth of each layer, from which it recovers the **cross section per
#   particle** $\sigma$ by dividing out column density and molecular weight
#
# It then sends a ray (chord) past every level, adds up $\sigma n$ along the chord's path through
# each shell to get the transmission $\mathcal{T}$ along it, and sums the blocked starlight from each
# annulus:
#
# $$ \left(\frac{R_p}{R_\star}\right)^2 = \frac{z_{\rm bottom}^2 + 2 \sum_i (1 - \mathcal{T}_i)\, z_i\, dz_i}{R_\star^2}. $$
#
# In our helper we hand it the column density times molecular weight, so `DTAU` is just $\sigma$.

# %%
K_B, AMU = 1.38064852e-16, 1.66053904e-24  # cgs
R_PLANET, R_STAR = 7e9, 7e10                 # cm: a Jupiter around a Sun-like star


def scale_height(temperature=1000., mu=2.3, gravity=2500.):
    return K_B * temperature / (mu * AMU * gravity)


def atmosphere(nlevel=200, p_top_bar=1e-8, p_bottom_bar=1e2, temperature=1000., mu=2.3, gravity=2500.):
    """Isothermal levels from the top down: pressure (dyne/cm2), altitude z (cm), spacing dz (cm)."""
    H = scale_height(temperature, mu, gravity)
    pressure = np.logspace(np.log10(p_top_bar), np.log10(p_bottom_bar), nlevel) * 1e6
    z = R_PLANET + H * np.log(pressure[-1] / pressure)
    dz = np.full(nlevel, H * np.log(pressure[1] / pressure[0]))
    return dict(pressure=pressure, z=z, dz=dz, temperature=temperature, mu=mu, H=H)


def effective_radius(cross_section, atm):
    """
    Transit radius R_star * sqrt(depth). cross_section (cm2 per particle) can be a number, a per-layer
    array, or an (nlayer, nwavelength) array for a spectrum.
    """
    nlevel = len(atm['pressure'])
    nlayer = nlevel - 1
    cross_section = np.asarray(cross_section, dtype=float)
    if cross_section.ndim == 0:
        cross_section = np.full((nlayer, 1), cross_section)
    elif cross_section.ndim == 1:
        cross_section = cross_section[:, np.newaxis]
    mmw = np.full(nlayer, atm['mu'])
    depth = get_transit_1d(atm['z'], atm['dz'], nlevel, cross_section.shape[1], R_STAR, mmw, K_B, AMU,
                           atm['pressure'], np.full(nlevel, atm['temperature']), mmw * AMU, cross_section.copy())
    return R_STAR * np.sqrt(depth)


atm = atmosphere()
H = atm['H']
print(f'scale height H = {H / 1e5:.0f} km; the grid spans {(atm["z"][0] - R_PLANET) / H:.0f} scale heights')

# %% [markdown]
# ## 1. Transparent and opaque atmospheres, and a cloud deck
#
# If the atmosphere absorbs nothing, the star sees only the solid (or deep) planet, $R_p$. If every
# layer is opaque, starlight is blocked all the way to the top of the atmosphere. And if there is an
# opaque cloud deck under clear gas, the planet looks exactly as big as the top of the deck.
# Clouds flatten transmission spectra for this reason: everything below the deck is hidden.

# %%
transparent = effective_radius(0.0, atm)[0]
opaque = effective_radius(1e10, atm)[0]
print(f'transparent: R = R_p + {(transparent - R_PLANET) / H:.3f} H')
print(f'opaque:      R = top of atmosphere + {(opaque - atm["z"][0]) / atm["dz"][0]:+.3f} layers')
np.testing.assert_allclose(transparent, R_PLANET, rtol=1e-12)
assert abs(opaque - atm['z'][0]) < 0.1 * atm['dz'][0]

deck_levels = np.arange(10, 199, 10)
deck_radius = []
for level in deck_levels:
    cross_section = np.zeros(len(atm['pressure']) - 1)
    cross_section[level:] = 1e10  # opaque from this level down
    deck_radius.append(effective_radius(cross_section, atm)[0])
deck_radius = np.array(deck_radius)
deck_pressure = atm['pressure'][deck_levels] / 1e6

fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
ax.plot(deck_pressure, (atm['z'][deck_levels] - R_PLANET) / H, color=INK, linewidth=6, alpha=0.25,
        label='altitude of the deck top')
ax.plot(deck_pressure, (deck_radius - R_PLANET) / H, 'o', color=BLUE, label='PICASO transit radius')
ax.set_xscale('log')
ax.invert_xaxis()
ax.set_xlabel('Cloud deck top pressure [bar]')
ax.set_ylabel('Height above $R_p$ [scale heights]')
ax.set_title('An opaque cloud deck sets the transit radius')
ax.legend(frameon=False)
plt.show()

np.testing.assert_allclose(deck_radius, atm['z'][deck_levels], atol=0.1 * atm['dz'][0])

# %% [markdown]
# ## 2. The scale height: e times more opacity, one scale height higher
#
# Density falls by a factor of e every scale height. So if a gas absorbs e times more per particle,
# the level where the atmosphere becomes opaque moves up by exactly one scale height, whatever the
# starting opacity. The transit radius therefore grows with the **logarithm** of the opacity:
#
# $$ \frac{dR}{d\ln\sigma} = H. $$
#
# This is the most useful rule in transmission spectroscopy: a feature 100 times stronger than its
# surroundings is only $\ln 100 \approx 4.6$ scale heights taller.

# %%
log_sigma = np.linspace(np.log(1e-28), np.log(1e-20), 33)
radius = np.array([effective_radius(np.exp(ls), atm)[0] for ls in log_sigma])

fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
ax.plot(log_sigma, (radius - R_PLANET) / H, 'o', color=BLUE, label='PICASO')
ref = np.argmin(np.abs(log_sigma - np.log(1e-25)))
ax.plot(log_sigma, (radius[ref] - R_PLANET) / H + (log_sigma - log_sigma[ref]), color=INK, linestyle=':',
        label='slope 1 (one H per factor of e)')
ax.axhline((atm['z'][0] - R_PLANET) / H, color=ORANGE, linewidth=1, label='top of the model atmosphere')
ax.set_xlabel('ln(cross section per particle / cm$^2$)')
ax.set_ylabel('Transit radius above $R_p$ [scale heights]')
ax.legend(frameon=False)
plt.show()

# %% [markdown]
# The slope is 1 until the radius approaches the top of the model atmosphere (orange line), where
# there is no more gas above to absorb. **Make sure your pressure grid extends well above where
# your strongest features form.** Away from the top, the rule holds very precisely:

# %%
for sigma in [1e-26, 1e-25, 1e-24]:
    shift = effective_radius(np.e * sigma, atm)[0] - effective_radius(sigma, atm)[0]
    print(f'sigma = {sigma:g} cm2: e times more opacity raises the radius by {shift / H:.4f} H')
    assert abs(shift / H - 1) < 0.01

# %% [markdown]
# ## 3. Where exactly is the transit radius?
#
# A ray grazing the planet at radius $R$ passes through a path of length about $\sqrt{2\pi R H}$ of
# gas near that radius, so its optical depth is
#
# $$ \tau_{\rm chord}(R) = \sigma\, n(R) \sqrt{2 \pi R H}. $$
#
# [Lecavelier des Etangs et al. (2008)](https://ui.adsabs.harvard.edu/abs/2008A%26A...481L..83L/abstract)
# showed that the transit radius is where $\tau_{\rm chord} \approx 0.56$: not where the gas becomes
# fully opaque, because the partially transparent rays above and the partially opaque ones below
# balance out there.

# %%
n_bottom = atm['pressure'][-1] / (K_B * atm['temperature'])
heights = np.linspace(-2, 25, 400)  # scale heights above R_p
fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
for sigma, color in zip([1e-26, 1e-24], [BLUE, ORANGE]):
    radii = R_PLANET + heights * H
    tau_chord = sigma * n_bottom * np.exp(-heights) * np.sqrt(2 * np.pi * radii * H)
    picaso_radius = effective_radius(sigma, atm)[0]
    expected = brentq(lambda r: sigma * n_bottom * np.exp(-(r - R_PLANET) / H) * np.sqrt(2 * np.pi * r * H) - 0.56,
                      R_PLANET - 50 * H, R_PLANET + 200 * H)
    ax.plot(tau_chord, heights, color=color, label=f'$\\sigma$ = {sigma:g} cm$^2$')
    ax.axhline((picaso_radius - R_PLANET) / H, color=color, linestyle='--', linewidth=1)
    print(f'sigma = {sigma:g}: PICASO radius is {(picaso_radius - expected) / H:+.3f} H from the tau = 0.56 level')
    assert abs(picaso_radius - expected) < 0.05 * H
ax.axvline(0.56, color=INK, linestyle=':', label='$\\tau_{\\rm chord}$ = 0.56')
ax.set_xscale('log')
ax.set_xlim(1e-3, 1e3)
ax.set_xlabel('Optical depth along the grazing ray')
ax.set_ylabel('Height above $R_p$ [scale heights]')
ax.set_title('Dashed: PICASO transit radius')
ax.legend(frameon=False)
plt.show()

# %% [markdown]
# ## 4. The Rayleigh slope and the size of spectral features
#
# Small molecules scatter blue light more than red, with $\sigma \propto \lambda^{-4}$ (Rayleigh
# scattering). Combining that with $dR/d\ln\sigma = H$ gives a straight line in $R$ versus
# $\ln\lambda$ with slope
#
# $$ \frac{dR}{d\ln\lambda} = -4H, $$
#
# so the blue slope of a clear planet measures its scale height, and from it the temperature
# ([Lecavelier des Etangs et al. 2008](https://ui.adsabs.harvard.edu/abs/2008A%26A...481L..83L/abstract)).

# %%
wavelength = np.logspace(np.log10(0.3), np.log10(1.0), 30)  # micron
rayleigh = 1e-26 * (wavelength / 0.5)**-4
nlayer = len(atm['pressure']) - 1

fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
for temperature, color in zip([500., 1000., 2000.], [BLUE, ORANGE, AQUA]):
    hot = atmosphere(temperature=temperature)
    radius = effective_radius(np.broadcast_to(rayleigh, (nlayer, len(wavelength))), hot)
    slope = np.polyfit(np.log(wavelength), radius, 1)[0]
    ax.plot(wavelength, (radius - radius[-1]) / 1e5, color=color,
            label=f'T = {temperature:.0f} K: slope = {slope / hot["H"]:.2f} H')
    assert abs(slope / hot['H'] + 4) < 0.05
ax.set_xscale('log')
ax.set_xticks([0.3, 0.5, 0.7, 1.0], labels=['0.3', '0.5', '0.7', '1.0'])
ax.minorticks_off()
ax.set_xlabel('Wavelength [$\\mu$m]')
ax.set_ylabel('Radius relative to 1 $\\mu$m [km]')
ax.set_title('Rayleigh scattering: dR/dln$\\lambda$ = -4H')
ax.legend(frameon=False)
plt.show()

# %% [markdown]
# ### Heavy atmospheres have small features
#
# Because every feature is measured in scale heights, and $H \propto 1/\mu$, the same molecule
# produces features about **12 times smaller** in an N$_2$ atmosphere ($\mu$ = 28) than in an
# H$_2$-dominated one ($\mu$ = 2.3). This is why terrestrial planets are so much harder to
# characterize in transmission than gas giants. In transit depth, a feature that is
# $\Delta R$ tall changes the depth by about $2 R_p \Delta R / R_\star^2$:

# %%
band = 1e-26 + 1e-23 * np.exp(-0.5 * ((wavelength - 0.6) / 0.05)**2)  # a gray continuum plus a band
fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
amplitudes = {}
for mu, color, label in [(2.3, BLUE, 'H$_2$-rich, $\\mu$ = 2.3'), (28.0, ORANGE, 'N$_2$, $\\mu$ = 28')]:
    gas = atmosphere(mu=mu)
    depth = (effective_radius(np.broadcast_to(band, (nlayer, len(wavelength))), gas) / R_STAR)**2
    amplitudes[mu] = depth.max() - depth.min()
    ax.plot(wavelength, (depth - depth.min()) * 1e6, color=color, label=label)
ax.set_xscale('log')
ax.set_xticks([0.3, 0.5, 0.7, 1.0], labels=['0.3', '0.5', '0.7', '1.0'])
ax.minorticks_off()
ax.set_xlabel('Wavelength [$\\mu$m]')
ax.set_ylabel('Transit depth above minimum [ppm]')
ax.set_title('Same absorber, different mean molecular weight')
ax.legend(frameon=False)
plt.show()

ratio = amplitudes[2.3] / amplitudes[28.0]
print(f'the feature is {ratio:.1f} times larger in the H2-rich atmosphere (28 / 2.3 = {28 / 2.3:.1f})')
assert abs(ratio / (28 / 2.3) - 1) < 0.05

# %% [markdown]
# ## Summary
#
# | Limiting case | What PICASO gives | Physics |
# |---|---|---|
# | Transparent | $(R_p/R_\star)^2$ exactly | nothing absorbs |
# | Opaque | radius = top of the atmosphere | everything absorbs |
# | Opaque cloud deck | radius = top of the deck | clouds hide what is below and flatten spectra |
# | e times more opacity | radius up by one $H$ | density falls by e per scale height |
# | Absolute radius | where $\tau_{\rm chord} \approx 0.56$ | balance of partly transparent and opaque rays |
# | Rayleigh scattering | $dR/d\ln\lambda = -4H$ | $\sigma \propto \lambda^{-4}$ |
# | Heavier gas | features smaller by $\mu$ | $H \propto 1/\mu$ |
#
# **Try it:** put a cloud deck at 1 mbar under the Rayleigh atmosphere from section 4. What happens
# to the slope at the reddest wavelengths, and why?
