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
# # Level Fluxes and the Energy Budget of an Atmosphere
#
# A climate model converges a temperature profile until the atmosphere is in radiative-convective equilibrium (RCE) for a chosen internal temperature, $T_{\rm int}$. Many profiles do not come from a climate model, though: they are retrieved from data, taken from a parameterization, or adjusted by hand. This notebook shows how to use level fluxes to check how close any profile is to RCE.
#
# The idea is simple. In RCE, the net upward flux through every radiative layer is the flux the interior has to lose:
#
# $$F_{\rm net}(P) = F_{\rm thermal}^{\uparrow} - F_{\rm thermal}^{\downarrow} + r \left(F_{\star}^{\uparrow} - F_{\star}^{\downarrow}\right) = \sigma T_{\rm int}^4$$
#
# where $r$ is the redistribution factor for starlight (`rfacv` in the climate code: 0.5 for full day-night redistribution, 1 for the dayside). In convective layers, convection carries part of the flux, so the radiative $F_{\rm net}$ falls below $\sigma T_{\rm int}^4$ there. Where $F_{\rm net}$ is larger the layer loses energy and would cool; where it is smaller the layer gains energy and would warm.
#
# ### Two settings: `get_lvl_flux` and `integrated_energy`
#
# - `case.approx(get_lvl_flux=True)` makes `spectrum()` also return the upward (`flux_plus`) and downward (`flux_minus`) fluxes at every level, and at each layer midpoint (`flux_plus_mdpt`, `flux_minus_mdpt`). They are in the full output under `['full_output']['level']['thermal_fluxes']` and `['full_output']['level']['reflected_fluxes']`, with one row per level and one column per wavenumber.
# - `case.approx(integrated_energy=True)` makes every level-flux value the **energy in its wavenumber bin** (erg/cm$^2$/s), as the climate code computes it: the thermal fluxes use the blackbody integrated over each bin, and `star()` integrates the stellar flux over each bin. Summing over wavenumber then gives the total flux at each level, over the wavelength range of the opacities. Without it, level fluxes are monochromatic, in the same units as the spectrum.
#
# A climate calculation always uses both, and so does an energy budget. Neither changes the spectrum itself; the level fluxes only cost extra time and memory, which is why they are off by default.
#
# ### Thermal and reflected light need different geometries
#
# The climate code computes thermal emission with the usual Gauss angles across the disk, but it puts starlight into a single plane-parallel column lit at $\mu_0 = 0.5$, the dayside average. To reproduce its energy budget we do the same, with two runs:
#
# - **Thermal:** `case.phase_angle(0)` (the default Gauss angles) and `calculation='thermal'`.
# - **Reflected:** `case.phase_angle(0, num_gangle=1, num_tangle=1)`, the single column at $\mu_0 = 0.5$, and `calculation='reflected'`. The downward stellar flux at the top is then half the incident flux. Multiply the stellar fluxes by the redistribution factor to compare them with the thermal fluxes.
#
# Why not one run? With the usual disk geometry, the reflected level fluxes are an average over facets that are each lit at their own angle. That is a valid average, but it absorbs starlight at slightly different depths than the single column the climate code uses, and starlight is hundreds of times $\sigma T_{\rm int}^4$, so that small difference swamps the budget. The single column cannot be used for thermal emission either: the thermal source function is singular at $\mu = 0.5$, and PICASO raises an error if you try.
#
# Because starlight is usually hundreds of times larger than $\sigma T_{\rm int}^4$ for irradiated planets, $F_{\rm net}$ is a small difference between large fluxes. To compare with a climate model, use the same geometry as above and the same opacities, redistribution factor, and Raman setting (the climate code turns Raman scattering off, `raman='none'`).

# %%
import os
import numpy as np
import astropy.units as u
from picaso import justdoit as jdi
from picaso import justplotit as jpi
plt = jpi.plt

# %% [markdown]
# ## Planet, star, and opacities
#
# We use the WASP-39 b setup from the climate tutorial (D_climate/2_Exoplanet_PreW): 10x solar metallicity, solar C/O, and the preweighted correlated-k tables.

# %%
mh = '1.0'  # log metallicity, 10x solar
CtoO = '0.46'  # solar C/O
ck_db = os.path.join(os.getenv('picaso_refdata'), 'opacities', 'preweighted', f'sonora_2121grid_feh{mh}_co{CtoO}.hdf5')
opacity_ck = jdi.opannection(ck_db=ck_db, method='preweighted')

tint = 200  # internal temperature (K)
rfacv = 0.5  # redistribution factor for starlight: full day-night redistribution
grav = 4.5  # m/s2
star = dict(temp=5326.6, metal=-0.03, logg=4.38933, radius=0.932, radius_unit=u.R_sun,
            semi_major=0.0486, semi_major_unit=u.AU, database='phoenix')
sigma = 5.67e-5  # Stefan-Boltzmann constant (cgs)

# %% [markdown]
# ## 1. A profile in equilibrium: run the climate model
#
# This takes a minute or two. The Guillot profile we start from will also serve as an example of a profile that is not in equilibrium.

# %%
cl_run = jdi.inputs(calculation='planet', climate=True)
cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)'))
cl_run.effective_temp(tint)
cl_run.star(opacity_ck, **star)

guess = cl_run.guillot_pt(1000, nlevel=91, T_int=tint, p_bottom=2, p_top=-6)
cl_run.inputs_climate(temp_guess=guess['temperature'].values, pressure=guess['pressure'].values,
                      rcb_guess=85, rfacv=rfacv)
out = cl_run.climate(opacity_ck, save_all_profiles=False, with_spec=False, verbose=False)
converged = out['ptchem_df']

# %% [markdown]
# ## 2. The energy budget of any profile
#
# This function runs the two spectra described above and sums the level fluxes over wavenumber. Upward is positive, so the stellar term is negative: starlight goes down.

# %%
def energy_budget(profile, opacity):
    case = jdi.inputs()
    # level fluxes as energies per wavenumber bin, with the same Raman setting as the climate code
    case.approx(get_lvl_flux=True, integrated_energy=True, raman='none')
    case.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)'))
    case.star(opacity, **star)
    case.atmosphere(df=profile)
    # thermal emission: the usual Gauss angles across the disk
    case.phase_angle(0)
    level = case.spectrum(opacity, calculation='thermal', full_output=True)['full_output']['level']
    thermal = level['thermal_fluxes']
    # starlight: one column lit at mu0 = 0.5, the dayside average, as in the climate code
    case.phase_angle(0, num_gangle=1, num_tangle=1)
    stellar = case.spectrum(opacity, calculation='reflected', full_output=True)['full_output']['level']['reflected_fluxes']
    thermal_net = np.sum(thermal['flux_plus'] - thermal['flux_minus'], axis=1)
    stellar_net = rfacv * np.sum(stellar['flux_plus'] - stellar['flux_minus'], axis=1)
    net = thermal_net + stellar_net
    return dict(pressure=np.asarray(level['pressure']), thermal_net=thermal_net, stellar_net=stellar_net,
                net=net,
                # the internal temperature each level implies: sign(F_net) (|F_net| / sigma)^(1/4)
                tint_implied=np.sign(net) * (np.abs(net) / sigma) ** 0.25)

budget = {'converged climate profile': energy_budget(converged, opacity_ck)}

# a profile that is not in equilibrium: the same chemistry with the Guillot starting guess
not_converged = converged.copy()
not_converged['temperature'] = guess['temperature'].values
budget['Guillot guess (not in equilibrium)'] = energy_budget(not_converged, opacity_ck)

b = budget['converged climate profile']
print(f'starlight absorbed at the top / sigma Tint^4 = {abs(b["stellar_net"][0]) / (sigma * tint**4):.0f}')

# %% [markdown]
# ## 3. Is the profile in equilibrium?
#
# The left panel shows the two profiles. The right panel shows the internal temperature that the net flux at each level implies. In equilibrium it equals the target $T_{\rm int}$ through the radiative zone, and falls below it in the convective zone (shaded), where convection carries the rest of the flux.

# %%
convective_top = out['pressure'][out['cvz_locs'][1]]  # top of the convective zone (bar)
fig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
ax_t.semilogy(converged['temperature'], converged['pressure'], color='black', label='converged climate profile')
ax_t.semilogy(not_converged['temperature'], not_converged['pressure'], color='orange', label='Guillot guess (not in equilibrium)')
ax_t.set_xlabel('Temperature (K)')
ax_t.set_ylabel('Pressure (bar)')
ax_t.legend(loc='upper right')

for (name, b), color in zip(budget.items(), ['black', 'orange']):
    ax_f.semilogy(b['tint_implied'], b['pressure'], color=color, label=name)
ax_f.axvline(tint, color='grey', linestyle='dashed', label=f'target $T_{{\\rm int}}$ = {tint} K')
ax_f.axvline(0, color='grey', linewidth=0.5)
ax_f.set_xlabel(r'Internal temperature implied by the net flux, sign($F_{\rm net}$) $(|F_{\rm net}|/\sigma)^{1/4}$ (K)')
for ax in (ax_t, ax_f):
    ax.axhspan(convective_top, out['pressure'].max(), color='grey', alpha=0.15)
ax_f.text(0.02, 0.03, 'convective zone', transform=ax_f.transAxes, color='dimgrey')
ax_f.invert_yaxis()
ax_f.legend(loc='upper right')
plt.tight_layout()

# %% [markdown]
# Here is the same comparison as numbers. For the converged profile the net flux equals $\sigma T_{\rm int}^4$ to within a few percent over most of the radiative zone (median 1.00). Near 1 bar it dips to about a quarter of $\sigma T_{\rm int}^4$: that is the climate solution's own convergence residual, and the climate code reports the same numbers in `out['flux_balance']`. For the Guillot guess the net flux is off by large factors, and negative in places (those layers absorb more than they emit and would heat up).

# %%
radiative = budget['converged climate profile']['pressure'] < convective_top
for name, b in budget.items():
    ratio = b['net'][radiative] / (sigma * tint**4)
    print(f'{name:36s} F_net / sigma Tint^4 in the radiative zone: median {np.median(ratio):7.2f}, '
          f'range [{ratio.min():8.2f}, {ratio.max():8.2f}]')

# %% [markdown]
# ## Using this on your own profile
#
# Replace `converged` with your own profile (a dataframe with pressure, temperature, and abundances, as for `case.atmosphere`), and set `tint`, `rfacv`, the opacities, and the star to the assumptions you want to test. Then compare the implied internal temperature with what you expect:
#
# - Close to the target through the radiative zone (as for the converged profile above): the profile is close to radiative-convective equilibrium.
# - Off by hundreds of kelvin, or negative (as for the Guillot guess): some layers gain or lose far more energy than the interior supplies, so the profile is far from equilibrium.
# - Below the target only in the deep atmosphere: those layers may be convective, where convection carries the rest of the flux.
#
# For a brown dwarf, start from `jdi.inputs(calculation='browndwarf')`, skip `star()`, run `spectrum(calculation='thermal')`, and drop the stellar term: the net thermal flux should then equal $\sigma T_{\rm eff}^4$.
