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
# # Level Fluxes and the Energy Budget of an Atmosphere
#
# A climate model converges a temperature profile until the atmosphere is in radiative-convective equilibrium (RCE). Many profiles do not come from a climate model, though: they are retrieved from data or taken from a parameterization. This notebook shows how to use level fluxes to check how close any profile is to RCE.
#
# In RCE, the net upward flux through every radiative layer is the flux the interior has to lose:
#
# $$F_{\rm net}(P) = F_{\rm thermal}^{\uparrow} - F_{\rm thermal}^{\downarrow} + r \left(F_{\star}^{\uparrow} - F_{\star}^{\downarrow}\right) = \sigma T_{\rm int}^4$$
#
# where $r$ is the redistribution factor for starlight (`rfacv` in the climate code: 0.5 for full day-night redistribution, 1 for the dayside). In convective layers, convection carries part of the flux, so the radiative $F_{\rm net}$ falls below $\sigma T_{\rm int}^4$ there. Where $F_{\rm net}$ is larger the layer loses energy and would cool; where it is smaller the layer gains energy and would warm.
#
# ### Two settings: `get_lvl_flux` and `integrated_energy`
#
# - `case.approx(get_lvl_flux=True)` makes the function `spectrum()` also return the upward (`flux_plus`) and downward (`flux_minus`) fluxes at every level, and at each layer midpoint (`flux_plus_mdpt`, `flux_minus_mdpt`). They are in the full output under `['full_output']['level']['thermal_fluxes']` and `['full_output']['level']['reflected_fluxes']`, with one row per level and one column per wavenumber.
# - `case.approx(integrated_energy=True)` makes every level-flux value the **energy in its wavenumber bin** (erg/cm$^2$/s), as the climate code computes it: the thermal fluxes use the blackbody integrated over each bin, and `star()` integrates the stellar flux over each bin. Summing over wavenumber then gives the total flux at each level, over the wavelength range of the opacities. Without it, level fluxes are monochromatic, in the same units as the spectrum.
#
# A climate calculation always uses both, and so does an energy budget. Neither changes the spectrum itself; the level fluxes only cost extra time and memory, which is why they are off by default.
#
# ### Thermal and reflected light need different geometries for flux balance
#
# The climate code computes thermal emission with the usual Gauss angles across the disk, but it puts starlight into a single plane-parallel column lit at $\mu_0 = 0.5$, the dayside average. To reproduce its energy budget, `inputs.energy_budget` (used below) does the same, with two runs:
#
# - **Thermal:** `case.phase_angle(0)` (the default Gauss angles) and `calculation='thermal'`.
# - **Reflected:** `case.phase_angle(0, num_gangle=1, num_tangle=1)`, the single column at $\mu_0 = 0.5$, and `calculation='reflected'`. The downward stellar flux at the top is then half the incident flux. Multiply the stellar fluxes by the redistribution factor to compare them with the thermal fluxes.
#
# Why not one run? With the usual disk geometry, the reflected level fluxes are an average over facets that are each lit at their own angle. That is a valid average, but it absorbs starlight at slightly different depths than the single column the climate code uses, and starlight is hundreds of times $\sigma T_{\rm int}^4$, so that small difference swamps the budget. The single column cannot be used for thermal emission either: the thermal source function is singular at $\mu = 0.5$, and PICASO raises an error if you try.
#
# `energy_budget` takes all of these from the bundle you call it on, so calling it on the climate bundle gives exactly the climate code's setup.

# %%
import os
import numpy as np
import astropy.units as u
import astropy.constants as c
from picaso import justdoit as jdi
from picaso import justplotit as jpi

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

cl_run = jdi.inputs(calculation='planet', climate=True)
cl_run.gravity(gravity=4.5, gravity_unit=u.Unit('m/(s**2)'))
cl_run.effective_temp(tint)
cl_run.star(opacity_ck, temp=5326.6, metal=-0.03, logg=4.38933, radius=0.932, radius_unit=u.R_sun,
            semi_major=0.0486, semi_major_unit=u.AU, database='phoenix')

# %% [markdown]
# ## 1. A profile in equilibrium: run the climate model
#
# This takes a minute or two. The Guillot profile we start from will also serve as an example of a profile that is not in equilibrium.

# %%
guess = cl_run.guillot_pt(1000, nlevel=91, T_int=tint, p_bottom=2, p_top=-6) #this guess is probably not in RCE!
cl_run.inputs_climate(temp_guess=guess['temperature'].values, pressure=guess['pressure'].values,
                      rcb_guess=85, rfacv=rfacv)
out = cl_run.climate(opacity_ck, verbose=False)

# %% [markdown]
# ## 2. The energy budget of any profile
#
# `cl_run.energy_budget` runs the two spectra described above (thermal with the usual Gauss angles, starlight in a single column at $\mu_0 = 0.5$) and sums the level fluxes over wavenumber. It uses the atmosphere currently in `cl_run`, with the same opacities, star, redistribution factor and setting as the climate code, and it does not change `cl_run`. To choose the profile, set it first with `cl_run.atmosphere(df=...)`: a dataframe with pressure, temperature, and abundances, such as the converged climate profile `out['ptchem_df']`.
#
# Among other things (see the docstring), it returns the net upward radiative flux at each level, `flux_net_radiative` $= F_{\rm thermal}^{\uparrow} - F_{\rm thermal}^{\downarrow} + r (F_{\star}^{\uparrow} - F_{\star}^{\downarrow})$, and the same flux expressed as a temperature, `tint_computed` $= {\rm sign}(F_{\rm net}) (|F_{\rm net}|/\sigma)^{1/4}$, to compare with $T_{\rm int}$. The output has the same format as a climate output, so it can also be passed to `jpi.flux_analyzer` (see the [climate tutorials](https://natashabatalha.github.io/picaso/notebooks/D_climate/10_Exoplanet_ExcludeMol.html)).
#
# For comparison with a profile that is not in equilibrium, we take the converged chemistry with the Guillot starting guess for the temperature.

# %%
not_converged = out['ptchem_df'].copy()
not_converged['temperature'] = guess['temperature'].values

budget = {}
cl_run.atmosphere(df=out['ptchem_df'])
budget['converged climate profile'] = cl_run.energy_budget(opacity_ck)
cl_run.atmosphere(df=not_converged)
budget['Guillot guess (not in equilibrium)'] = cl_run.energy_budget(opacity_ck)

# %% [markdown]
# ## 3. Is the profile in equilibrium?
#
# `jpi.equilibrium_check` plots the profiles (left) and, for each profile, the internal temperature computed from its net flux at each level, `tint_computed` (right). All profiles are compared with one target $T_{\rm int}$ (dashed line). By default the target is computed from the first energy budget (its `internal_heat` $= \sigma T_{\rm int}^4$, from `cl_run.effective_temp`). Here we give it ourselves with `target_tint`, as a sanity check: it should be the same as the one the energy budget computes (`equilibrium_check` warns if they differ by more than 1%).

# %%
b = budget['converged climate profile']
print(f"T_int we set: {tint} K, T_int computed from the energy budget: {(b['internal_heat']/c.sigma_sb.cgs.value)**0.25:.1f} K")

jpi.equilibrium_check(list(budget.values()), labels=list(budget.keys()), colors=['black', 'orange'],
                      cvz_locs=out['cvz_locs'], target_tint=tint, backend='plotly')

# %% [markdown]
# **How to read the right panel.** Through the radiative zone, the net radiative flux at each level should carry exactly the internal heat, so the computed $T_{\rm int}$ should sit on the target:
#
# - **On the target**: the profile is in radiative equilibrium at that level (the converged profile, in black).
# - **Above the target**: more energy leaves through that level than the interior supplies, so the layers below it lose energy and would cool.
# - **Below the target**: less energy leaves than the interior supplies, so the layers below it gain energy and would warm.
# - **Negative**: the extreme case of "below the target". The net radiative flux points *down*: more energy flows down into the layers below (here, absorbed starlight) than comes up out of them. A temperature cannot be negative, so read this as a flux: we keep the sign of $F_{\rm net}$ and plot ${\rm sign}(F_{\rm net})(|F_{\rm net}|/\sigma)^{1/4}$, so that a downward flux is not mistaken for an upward one of the same size.
#
# In the convective zone (shaded, from the climate output `out['cvz_locs']`) the computed value falls below the target even in equilibrium, because convection carries the rest of the flux.
#
# The Guillot guess (orange) is far off: it is above the target in some layers and negative in others.

# %% [markdown]
# Here is the same comparison as numbers. For the converged profile the net flux equals $\sigma T_{\rm int}^4$ to within a few percent over most of the radiative zone (median 1.01). Near 1 bar it dips to about a quarter of $\sigma T_{\rm int}^4$: that is the climate solution's own convergence residual, and the climate code reports the same numbers in `out['flux_balance']`. For the Guillot guess the net flux is off by large factors, and negative in places (those layers absorb more than they emit and would heat up).

# %%
radiative = out['pressure'] < out['pressure'][out['cvz_locs'][1]]  # above the top of the convective zone
for name, b in budget.items():
    ratio = b['flux_net_radiative'][radiative] / b['internal_heat']
    print(f'{name:36s} F_net / sigma Tint^4 in the radiative zone: median {np.median(ratio):7.2f}, '
          f'range [{ratio.min():8.2f}, {ratio.max():8.2f}]')

# %% [markdown]
# ## Using this on your own profile
#
# Replace `not_converged` with your own profile (a dataframe with pressure, temperature, and abundances, as for `case.atmosphere`), and set `tint`, `rfacv`, the opacities, and the star to the assumptions you want to test. Then compare the computed internal temperature with what you expect:
#
# - Close to the target through the radiative zone (as for the converged profile above): the profile is close to radiative-convective equilibrium.
# - Off by hundreds of kelvin, or negative (as for the Guillot guess): some layers gain or lose far more energy than the interior supplies, so the profile is far from equilibrium. Pass the budget to `jpi.flux_analyzer` to see the heating and cooling rates of each layer.
# - Below the target only in the deep atmosphere: those layers may be convective, where convection carries the rest of the flux.
#
# You do not need to run a climate model first: `energy_budget` only needs a bundle with the planet, the star, an atmosphere, and `inputs_climate` (for the redistribution factor), or you can pass `rfacv=` yourself. Without a climate output there is no convective zone to shade (leave out `cvz_locs`).
#
# For a brown dwarf, start from `jdi.inputs(calculation='browndwarf', climate=True)` and skip `star()`: `energy_budget` then has no stellar term, and the net thermal flux should equal $\sigma T_{\rm eff}^4$: the target is $T_{\rm eff}$. With no starlight, a negative computed value needs thermal emission from hotter layers above to exceed the emission from below (a strong temperature inversion), so it is rare for brown dwarfs.

# %%
