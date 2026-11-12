# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: pic312
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Geometric, Spherical and Bond Albedos
#
# `PICASO` returns several albedos. This notebook explains how they differ and how to compute a
# Bond albedo that includes the phase integral, following Cahoy, Marley & Fortney (2010; ApJ 724, 189).
#
# | Quantity | Symbol | Definition | How to get it |
# |---|---|---|---|
# | Albedo at phase $\alpha$ | $A(\lambda,\alpha)$ | $F_p/F_\star = A(\lambda,\alpha)\,(R_p/a)^2$ (Cahoy Eq. 3) | `spectrum()['albedo']` at the phase set by `phase_angle` |
# | Geometric albedo | $A_g(\lambda)$ | $A(\lambda, 0)$ (Cahoy Eq. 2) | `spectrum()['albedo']` with `phase_angle(0)` |
# | Phase function | $\Phi(\lambda,\alpha)$ | $A(\lambda,\alpha)/A_g(\lambda)$ | |
# | Phase integral | $q(\lambda)$ | $2\int_0^\pi \Phi(\lambda,\alpha)\sin\alpha\,d\alpha$ (Cahoy Eq. 4) | `bond_albedo()['phase_integral']` |
# | Spherical albedo | $A_s(\lambda)$ | $A_g(\lambda)\,q(\lambda) = 2\int_0^\pi A(\lambda,\alpha)\sin\alpha\,d\alpha$ (Cahoy Eq. 5) | `bond_albedo()['spherical_albedo']` |
# | Bond albedo | $A_B$ | $\int A_s F_\star\,d\lambda \,/ \int F_\star\,d\lambda$ (Cahoy Eq. 6) | `bond_albedo()['bond_albedo']` |
#
# The spherical albedo is the fraction of light at wavelength $\lambda$ that the planet scatters
# in *all* directions. The Bond albedo is its average weighted by the stellar spectrum. This is
# the quantity that sets the planet's energy budget.
#
# **The `bond_albedo` key returned by `spectrum()` is not the true Bond albedo.** It is the
# stellar-flux-weighted albedo at whatever phase you requested (eq. 18 of Batalha et al. 2019),
# which is equivalent to assuming $q = 1$. Cahoy et al. (2010, Table 5) find $q \approx 1.2$-$1.8$ for
# Jupiter and Neptune analogs, so this underestimates $A_B$. Use `inputs.bond_albedo()` instead.

# %%
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from picaso import justdoit as jdi

# %% [markdown]
# ## How the phase integral is computed
#
# Substituting $\mu = \cos\alpha$ gives $A_s = 2\int_{-1}^{1} A(\mu)\,d\mu$. `bond_albedo()`
# therefore runs reflected-light spectra at Gauss-Legendre nodes in $\mu$ (`n_phase` of them,
# default 12) plus zero phase. The nodes never land on $\alpha = 180^\circ$, where the
# reflected-light geometry is singular.
#
# In phase angle the nodes are nearly evenly spaced (about 11 to 169 degrees for `n_phase = 12`); the
# Gauss-Legendre weights, rather than the spacing, are what make the integral accurate. For a Lambert
# sphere, whose phase function is $\Phi_L = [\sin\alpha + (\pi-\alpha)\cos\alpha]/\pi$ (Cahoy Eq. 1), this
# recovers the exact phase integral $q = 3/2$ (Cahoy Sec. 4.4) to better than $10^{-5}$ with 12 nodes.
# To use your own phase grid instead, pass `phase_grid` (see the notes at the end).

# %% [markdown]
# ## A cloud-free Rayleigh-scattering atmosphere
#
# Set up a case exactly as you would for `spectrum`. Here we switch off Raman scattering so that the
# blue end of the spectrum is dominated by Rayleigh scattering. For comparison, a semi-infinite pure
# Rayleigh atmosphere has $A_g = 3/4$ (Cahoy Sec. 2.2) and a Rayleigh phase function has $q = 4/3$
# (Cahoy Sec. 4.4).

# %%
opa = jdi.opannection(wave_range=[0.3, 1])

case = jdi.inputs()
case.phase_angle(0)
case.gravity(gravity=25, gravity_unit=u.Unit('m/(s**2)'))
case.star(opa, 6000, 0.0122, 4.437)
case.atmosphere(filename=jdi.jupiter_pt(), sep=r'\s+')
case.approx(raman='none')

clear_out = case.bond_albedo(opa, n_phase=8)

# %% [markdown]
# `bond_albedo` costs `n_phase + 1` reflected-light calculations and does not change the case's own
# `phase_angle` settings. Compare its result with the $q = 1$ value from `spectrum`:

# %%
q1_bond_albedo = case.spectrum(opa)['bond_albedo']
print(f"spectrum()['bond_albedo'] (q = 1)           : {q1_bond_albedo:.4f}")
print(f"bond_albedo()['bond_albedo']                : {clear_out['bond_albedo']:.4f}")
print(f"flux-weighted phase integral A_B/<A_g>      : {clear_out['phase_integral_flux_weighted']:.4f}")

# %%
wavelength_um = 1e4 / clear_out['wavenumber']
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
ax[0].plot(wavelength_um, clear_out['geometric_albedo'], label='$A_g$ (geometric)')
ax[0].plot(wavelength_um, clear_out['spherical_albedo'], label='$A_s$ (spherical)')
ax[0].set_ylabel('Albedo')
ax[0].legend()
ax[1].plot(wavelength_um, clear_out['phase_integral'])
ax[1].axhline(1.5, ls='--', c='k', lw=1, label='Lambert, 3/2')
ax[1].set_ylabel('Phase integral $q$')
ax[1].set_xlabel(r'Wavelength [$\mu$m]')
ax[1].legend()

# %% [markdown]
# In strong absorption bands $A_g$ becomes very small. $A_s$ is integrated directly from the albedo
# at each phase, so it stays well behaved there, and $q = A_s/A_g$ is set to NaN where $A_g = 0$.
#
# ## Adding clouds
#
# Forward-scattering clouds and hazes change the phase function, so the cloudy planet's $q$
# differs from the clear_out one even where $A_g$ is similar.

# %%
case.clouds(filename=jdi.jupiter_cld(), sep=r'\s+')
cloudy_out = case.bond_albedo(opa, n_phase=8)
print(f"cloudy A_B = {cloudy_out['bond_albedo']:.4f}, "
      f"flux-weighted q = {cloudy_out['phase_integral_flux_weighted']:.4f}")

# %%
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(wavelength_um, clear_out['phase_integral'], label='clear_out')
ax.plot(wavelength_um, cloudy_out['phase_integral'], label='cloudy')
ax.set_ylabel('Phase integral $q$')
ax.set_xlabel(r'Wavelength [$\mu$m]')
ax.legend()

# %% [markdown]
# ## Practical notes
#
# - **Accuracy.** For the clear_out case above, `n_phase = 12` agrees with a 2-degree trapezoid grid to
#   0.02% in $A_B$. Check with a larger value for strongly forward-scattering particles.
# - **Your own phase grid.** Pass `phase_grid` (radians, including 0, below $\pi$) to integrate with
#   the trapezoid rule instead. For example, `np.radians(list(range(0, 180, 12)) + [179])`.
# - **Wavelength coverage.** $A_B$ only includes stellar flux inside the opacity database's
#   `wave_range`. For a true bolometric Bond albedo, choose a range that covers most of the star's
#   emission (Cahoy et al. integrate 0.35-2.5 um and find longer wavelengths add less than 5%).
# - **Angles.** Nonzero phases need more Chebyshev angles. The defaults are `num_gangle = 8` and
#   `num_tangle = 8`.
# - **Parallelization.** `n_cpu > 1` runs phases in separate processes. Each process holds a copy of
#   the opacity class, so memory use grows with `n_cpu`.
#
# **References:** Cahoy, Marley & Fortney (2010), ApJ 724, 189; Sobolev (1975), *Light Scattering in
# Planetary Atmospheres*; Batalha et al. (2019).
