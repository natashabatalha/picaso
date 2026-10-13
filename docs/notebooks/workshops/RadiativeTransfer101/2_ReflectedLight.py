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
# # Reflected Light: Building Intuition from Limiting Cases
#
# In the {doc}`thermal emission lesson </notebooks/workshops/RadiativeTransfer101/1_ThermalEmission>` the planet was the light source. Here the star
# is: starlight enters the atmosphere, gets scattered or absorbed, and some of it comes back out
# toward us. As before, we test PICASO's solver on simple atmospheres where we know the answer.
#
# We call `picaso.fluxes.get_reflected_1d` directly. It is the two-stream method of
# [Toon et al. (1989)](https://ui.adsabs.harvard.edu/abs/1989JGR....9416287T/abstract) with an exact
# treatment of light that scatters only once, and it is what `spectrum(calculation='reflected')`
# uses. No opacity data is needed.
#
# **What you will learn**
#
# 1. What geometric albedo is and how PICASO adds up light from across the planet's disk
# 2. Two exact limits: a black planet and a bare Lambertian surface
# 3. Single scattering, the phase function, and why thin layers are hard to integrate over the disk
# 4. Multiple scattering and conservation of energy
# 5. How PICASO compares with a published benchmark, and why the choice of phase function matters
#
# Every section ends with a check. The same checks run as PICASO's unit tests in
# `tests/test_reflected_rt.py`.

# %%
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from picaso.disco import compress_disco, compute_disco, get_angles_1d
from picaso.fluxes import get_reflected_1d

BLUE, ORANGE, AQUA, YELLOW, INK = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#52514e'

# %% [markdown]
# ## Geometry: adding up light from the whole disk
#
# At full phase (star behind us) we see the whole day side. Each point on the disk is lit from angle
# $\mu_0$ (cosine of the angle from the star) and seen at angle $\mu$; at full phase $\mu_0 = \mu$,
# from 1 at the disk center to 0 at the limb.
#
# PICASO solves the radiative transfer at a handful of these angles and then adds them up with
# Gauss quadrature weights to get the **geometric albedo** $A_g$: how bright the planet is at full
# phase compared with a flat, perfectly white (Lambertian) disk of the same size. `get_angles_1d`
# gives the angles and weights, `compute_disco` turns them into $\mu_0$ and $\mu$, and
# `compress_disco` does the sum. At each angle the reflectance is $I/F = \pi\,x_{\rm int}/F_0\pi$,
# where $x_{\rm int}$ is what `get_reflected_1d` returns.

# %%
NWNO = 1
WNO = np.array([1e4])  # 1 micron; these toy atmospheres are gray so the wavelength does not matter
F0PI = np.ones(NWNO)   # incoming stellar flux, normalized
TTHG = (1., -1., 2., -0.5, 1.)  # PICASO's default two-term Henyey-Greenstein parameters
OTHG, TTHG_RAY = 1, 3  # single scattering phase function options
N2, N1 = 0, 1          # multiple scattering phase function options


def geometry(ngauss=8):
    gangle, gweight, tangle, tweight = get_angles_1d(ngauss)
    ubar0, ubar1, cos_theta, _, _ = compute_disco(len(gangle), len(tangle), gangle, tangle, 0.0)
    return ubar0, ubar1, cos_theta, gweight, tweight


def reflected(dtau, w0, cosb=0.0, rayleigh=False, surf_reflect=0.0, nlayer=20, ngauss=8,
              single_phase=TTHG_RAY, multi_phase=N2, delta_eddington=False, get_lvl_flux=0):
    """
    Reflected light from a uniform atmosphere at full phase.
    dtau : total optical depth, or an array of per-layer optical depths
    w0 : single scattering albedo; cosb : asymmetry of cloud scattering
    rayleigh : True for Rayleigh scattering instead of a cloud
    Returns (geometric albedo, I/F at each angle, level fluxes)
    """
    ubar0, ubar1, cos_theta, gweight, tweight = geometry(ngauss)
    per_layer = np.atleast_1d(dtau) if np.ndim(dtau) else np.full(nlayer, dtau / nlayer)
    nlevel = len(per_layer) + 1
    dtau = np.repeat(per_layer[:, np.newaxis], NWNO, axis=1)
    w0, cosb = np.full_like(dtau, w0), np.full_like(dtau, cosb)
    if rayleigh:
        ftau_cld, ftau_ray, gcos2 = np.zeros_like(w0), np.ones_like(w0), np.full_like(w0, 0.5)
    else:
        ftau_cld, ftau_ray, gcos2 = np.ones_like(w0), np.zeros_like(w0), np.zeros_like(w0)

    def cumulative(layer_dtau):
        tau = np.zeros((nlevel, NWNO))
        tau[1:] = np.cumsum(layer_dtau, axis=0)
        return tau

    if delta_eddington:  # remove the forward peak from the multiple scattering part (see section 5)
        f = cosb**2
        w0_ms, cosb_ms, dtau_ms = w0 * (1 - f) / (1 - w0 * f), (cosb - f) / (1 - f), dtau * (1 - w0 * f)
    else:
        w0_ms, cosb_ms, dtau_ms = w0, cosb, dtau
    xint, level_fluxes = get_reflected_1d(
        nlevel, WNO, NWNO, *ubar0.shape, dtau_ms, cumulative(dtau_ms), w0_ms, cosb_ms, gcos2, ftau_cld, ftau_ray,
        dtau, cumulative(dtau), w0, cosb,
        np.full(NWNO, surf_reflect), ubar0, ubar1, cos_theta, F0PI, single_phase, multi_phase, *TTHG,
        get_toa_intensity=1, get_lvl_flux=get_lvl_flux, toon_coefficients=0, b_top=0.)
    albedo = compress_disco(NWNO, cos_theta, xint, gweight, tweight, F0PI)[0]
    return albedo, np.pi * xint[:, 0, 0] / F0PI[0], level_fluxes


W0_CONSERVATIVE = 0.999999  # "no absorption"; exactly 1 divides by zero (PICASO caps w0 just below 1)
W0_ABSORBING = 1e-6

# %% [markdown]
# ## 1. A black planet reflects nothing
#
# If the atmosphere only absorbs ($w_0 \approx 0$) and is thick, no light comes back.

# %%
albedo, _, _ = reflected(100.0, W0_ABSORBING)
print(f'geometric albedo of a black planet: {albedo:.1e}')
assert albedo < 1e-6

# %% [markdown]
# ## 2. A bare Lambertian surface: $A_g = \frac{2}{3} r$
#
# A Lambertian surface scatters light equally in all directions, so its brightness only depends on
# how directly it is lit ($\propto \mu_0$). Adding that up over a sphere gives the classic result that
# a Lambertian sphere with reflectivity $r$ has geometric albedo $\frac{2}{3}r$, not $r$: the edges of
# the disk are lit at a slant and look darker. We make the atmosphere transparent and change the
# surface reflectivity.

# %%
reflectivity = np.linspace(0, 1, 11)
albedos = np.array([reflected(1e-8, W0_ABSORBING, surf_reflect=r)[0] for r in reflectivity])

fig, ax = plt.subplots(figsize=(5, 4), layout='constrained')
ax.plot(reflectivity, 2 / 3 * reflectivity, color=INK, linewidth=6, alpha=0.25, label='$\\frac{2}{3}r$')
ax.plot(reflectivity, albedos, 'o', color=BLUE, label='PICASO')
ax.set_xlabel('Surface reflectivity $r$')
ax.set_ylabel('Geometric albedo $A_g$')
ax.set_title('Bare Lambertian surface')
ax.legend(frameon=False)
plt.show()

np.testing.assert_allclose(albedos[1:], 2 / 3 * reflectivity[1:], rtol=2e-3)

# %% [markdown]
# ## 3. Single scattering and the phase function
#
# In a very thin layer, each photon scatters at most once. How much light goes back toward us
# depends on the **phase function** $P(\Theta)$, the probability of scattering by angle $\Theta$.
# At full phase we look straight back along the incoming light, so what matters is the
# backscattering value $P(180^\circ)$: 1 for isotropic scattering and 1.5 for Rayleigh scattering
# (molecules scatter forward and backward more than sideways). For one scattering the exact answer is
#
# $$ \frac{I}{F} = \frac{w_0 P(180^\circ)}{4} \, \frac{\mu_0}{\mu_0 + \mu} \left(1 - e^{-\tau (1/\mu_0 + 1/\mu)}\right). $$

# %%
tau = 1e-4
mu = geometry()[1][:, 0]
exact_shape = 0.5 * (1 - np.exp(-2 * tau / mu)) / 4

fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
for rayleigh, phase_at_180, color, label in [(True, 1.5, BLUE, 'Rayleigh'), (False, 1.0, ORANGE, 'isotropic')]:
    _, i_over_f, _ = reflected(tau, W0_CONSERVATIVE, rayleigh=rayleigh)
    exact = W0_CONSERVATIVE * phase_at_180 * exact_shape
    ax.plot(mu, exact, color=color, linewidth=6, alpha=0.25)
    ax.plot(mu, i_over_f, 'o', color=color, label=f'{label}, P(180$^\\circ$) = {phase_at_180}')
    np.testing.assert_allclose(i_over_f, exact, rtol=1e-3)
ax.set_xlabel('$\\mu$ (1 = disk center, 0 = limb)')
ax.set_ylabel('I/F')
ax.set_yscale('log')
ax.set_title(f'Thin layer ($\\tau$ = {tau:g}); thick lines: exact single scattering')
ax.legend(frameon=False)
plt.show()

# %% [markdown]
# PICASO matches the exact answer at every angle. Notice something else: the thin layer is much
# brighter **at the limb**, because there we look through it at a slant and see more of it
# ($I/F \propto \tau/\mu$).
#
# ### Why that matters for disk integration
#
# For a thin layer the geometric albedo is exactly $A_g = w_0 P(180^\circ)\,\tau/2$, so $0.75\,\tau$
# for Rayleigh. But the brightness piles up at the limb, where there are few quadrature angles, so
# the disk sum converges slowly. PICASO's 1D spectra use between 5 and 8 angles (set by `num_gangle`
# in `phase_angle`, default 5):

# %%
ngauss_values = [5, 6, 7, 8]
albedo_over_tau = [reflected(tau, W0_CONSERVATIVE, rayleigh=True, ngauss=n)[0] / tau for n in ngauss_values]
for n, value in zip(ngauss_values, albedo_over_tau):
    print(f'{n} angles: A_g / tau = {value:.3f}   ({100 * (value / 0.75 - 1):+.1f}% from the exact 0.75)')
assert np.all(np.diff(albedo_over_tau) > 0) and albedo_over_tau[-1] < 0.75

# %% [markdown]
# The answer approaches 0.75 as the angles increase. For thick or cloudy atmospheres the limb is not
# special and a few angles are plenty (the Lambertian surface above was accurate to 0.1%), but for
# very thin, clear atmospheres it is worth increasing `num_gangle` in `phase_angle`.

# %% [markdown]
# ## 4. Multiple scattering and conservation of energy
#
# When nothing is absorbed ($w_0 = 1$), every photon ends up either reflected back to space or
# transmitted through the bottom. So the reflected fraction $R$ and transmitted fraction $T$ must add
# to 1, and the net (down minus up) flux must be the same at every level. We can check this with the
# level fluxes (`get_lvl_flux=1`), which the climate code uses for heating rates.

# %%
taus = np.logspace(-2, 1, 13)  # beyond tau ~ 10 even w0 = 0.999999 absorbs a measurable amount
R, T = [], []
for total_tau in taus:
    _, _, (flux_minus, flux_plus, _, _) = reflected(total_tau, W0_CONSERVATIVE, get_lvl_flux=1)
    net = flux_minus - flux_plus
    np.testing.assert_allclose(net, np.broadcast_to(net[:, :, :1, :], net.shape), rtol=1e-3)
    center = 3  # one viewing angle is enough to show it
    R.append(flux_plus[center, 0, 0, 0] / flux_minus[center, 0, 0, 0])
    T.append(net[center, 0, -1, 0] / flux_minus[center, 0, 0, 0])
R, T = np.array(R), np.array(T)

fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
ax.plot(taus, R, color=BLUE, label='reflected R')
ax.plot(taus, T, color=ORANGE, label='transmitted T')
ax.plot(taus, R + T, color=INK, linestyle=':', label='R + T')
ax.set_xscale('log')
ax.set_xlabel('Total optical depth $\\tau$')
ax.set_ylabel('Fraction of incoming flux')
ax.set_title('Non-absorbing cloud layer over a black surface')
ax.legend(frameon=False)
plt.show()

np.testing.assert_allclose(R + T, 1, rtol=1e-3)
print('R + T = 1 at every optical depth: energy is conserved')

# %% [markdown]
# Thicker layers reflect more and transmit less. Once a little absorption is added, $R + T < 1$ and
# the missing energy heats the atmosphere. More absorption (smaller $w_0$) makes a darker planet:

# %%
w0_values = np.array([0.3, 0.6, 0.8, 0.9, 0.95, 0.99, 0.999])
deep_albedo = [reflected(1000.0, w0)[0] for w0 in w0_values]
assert np.all(np.diff(deep_albedo) > 0)
for w0, a in zip(w0_values, deep_albedo):
    print(f'w0 = {w0:<6}  A_g = {a:.3f}')

# %% [markdown]
# Even a tiny amount of absorption matters a lot: going from $w_0 = 0.999$ to $0.99$ drops the
# albedo noticeably, because in a deep atmosphere photons scatter many times before escaping, and
# each scattering is another chance to be absorbed.
#
# ## 5. A published benchmark, and why the phase function matters
#
# For a deep (semi-infinite) atmosphere there is no simple formula, but
# [Dlugach & Yanovitskij (1974)](https://ui.adsabs.harvard.edu/abs/1974Icar...22...66D/abstract)
# computed geometric albedos exactly for Rayleigh, isotropic and Henyey-Greenstein (HG) scattering
# with asymmetry $g$. Their table ships with PICASO and was used to benchmark it in
# [Batalha et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019ApJ...878...70B/abstract).
#
# Two choices matter:
#
# * **Phase function.** PICASO's default for clouds is a *two-term* HG function that adds a small
#   back-scattering lobe, because real cloud particles scatter some light straight back. The table is
#   for a *one-term* HG function, so for a fair comparison we switch to one-term HG (`OTHG`).
# * **Delta-Eddington.** Strongly forward scattering particles send much of their light almost
#   straight ahead, which a two-stream method handles poorly. The delta-Eddington approximation
#   treats that forward peak as unscattered light. PICASO applies it by default.

# %%
# the table ships with PICASO's reference data
dlugach = pd.read_csv(os.path.join(os.environ['picaso_refdata'], 'base_cases', 'testing', 'DLUGACH_TEST.csv'))
dlugach = dlugach.set_index('asy')
w0_table = np.array([float(c) for c in dlugach.columns])
deep_layers = 10**np.linspace(-5, 3, 59)  # per-layer optical depths: semi-infinite in practice


def deep_albedo_curve(asy, **kwargs):
    return np.array([reflected(deep_layers, min(w0, W0_CONSERVATIVE), **kwargs)[0] for w0 in w0_table])


cases = {
    'Rayleigh': dict(row='Ray', kwargs=dict(rayleigh=True), color=BLUE),
    'isotropic': dict(row='0', kwargs=dict(), color=ORANGE),
    'HG g = 0.75 (one-term HG)': dict(row='0.75', kwargs=dict(cosb=0.75, single_phase=OTHG, multi_phase=N1,
                                                            delta_eddington=True), color=AQUA),
}
# plot against the absorption per scattering, 1 - w0, so the points near w0 = 1 spread out
absorption = 1 - np.minimum(w0_table, W0_CONSERVATIVE)
fig, ax = plt.subplots(figsize=(7, 4.5), layout='constrained')
for label, case in cases.items():
    picaso = deep_albedo_curve(case['row'], **case['kwargs'])
    ax.plot(absorption, dlugach.loc[case['row']].values, color=case['color'], linewidth=6, alpha=0.25)
    ax.plot(absorption, picaso, 'o', color=case['color'], label=label)
    case['picaso'] = picaso
default_tthg = deep_albedo_curve('0.75', cosb=0.75, delta_eddington=True)
ax.plot(absorption, default_tthg, 'x', color=YELLOW, label='HG g = 0.75 with default two-term HG')
ax.set_xscale('log')
ax.invert_xaxis()
ax.set_xlabel('Absorption per scattering, $1 - w_0$ (w$_0$ = 1 plotted at 10$^{-6}$)')
ax.set_ylabel('Geometric albedo $A_g$')
ax.set_title('Deep atmosphere: PICASO (points) vs Dlugach & Yanovitskij (thick lines)')
ax.legend(frameon=False)
plt.show()

# %%
for label, case in cases.items():
    error = 100 * (case['picaso'] / dlugach.loc[case['row']].values - 1)
    print(f'{label:28s} largest difference from the table: {np.abs(error).max():.1f}%')
np.testing.assert_allclose(cases['Rayleigh']['picaso'], dlugach.loc['Ray'].values, rtol=0.03)
np.testing.assert_allclose(cases['isotropic']['picaso'], dlugach.loc['0'].values, rtol=0.03)
np.testing.assert_allclose(cases['HG g = 0.75 (one-term HG)']['picaso'], dlugach.loc['0.75'].values, rtol=0.15)

# %% [markdown]
# Rayleigh and isotropic scattering agree with the exact answers to a few percent. For forward
# scattering clouds the two-stream method is good to ~10%, typical of two-stream methods (see
# [Batalha et al. 2019](https://ui.adsabs.harvard.edu/abs/2019ApJ...878...70B/abstract), and the
# {doc}`spherical harmonics tutorial </notebooks/H_radiativetransfer/2_AnalyzingApproximationsReflectedLightSH>`
# for a more accurate method). The yellow crosses show how much the phase function matters: the same
# particles with the default two-term HG function look much brighter at full phase, because of their
# back-scattering lobe. That is physics, not error, and a reminder to choose the phase function that
# matches your particles.
#
# ## Summary
#
# | Limiting case | What PICASO gives | Physics |
# |---|---|---|
# | Thick, absorbing | $A_g = 0$ | nothing comes back |
# | Bare Lambertian surface | $A_g = \frac{2}{3} r$ | slanted lighting near the limb |
# | Thin layer, per angle | exact single scattering | $I/F \propto w_0 P(180^\circ)\,\tau/\mu$ |
# | Thin layer, whole disk | $A_g \to 0.75\tau$ (Rayleigh) as angles increase | limb brightening needs more angles |
# | No absorption | $R + T = 1$ at any $\tau$ | conservation of energy |
# | Deep atmosphere | within 3% (Rayleigh, isotropic), ~10% (forward scattering) of exact | two-stream accuracy |
#
# **Try it:** put a bright surface ($r = 0.8$) under a thin Rayleigh atmosphere and increase $\tau$.
# When does the atmosphere hide the surface?
#
# Next in this course: {doc}`transmission </notebooks/workshops/RadiativeTransfer101/3_Transmission>`, where starlight passes through the planet's limb.
