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
# # One-Dimensional Climate Models: How Does a Single Molecule Heat or Cool a Planet?
#
# You may have already used `exclude_mol` to understand which molecules shape a spectrum (e.g. [identifying molecular features](https://natashabatalha.github.io/picaso/notebooks/A_basics/1_GetStarted.html)). The same keyword can be used in a **climate** calculation. Removing the opacity of a single molecule and re-converging the model tells you how much that molecule heats or cools the atmosphere, and where.
#
# In this tutorial you will learn:
#
# 1. How to run a climate model with and without the opacity of CO$_2$ using `exclude_mol`
# 2. How to use `jpi.flux_analyzer` to diagnose the energy budget of a climate model (net fluxes, radiative heating/cooling rates and outgoing thermal emission), and to compare two models
#
# What you should already be familiar with:
#
# - [how to compute a planet 1D climate model with on-the-fly opacity mixing](https://natashabatalha.github.io/picaso/notebooks/D_climate/2b_Exoplanet-ResortRebin-Chemeq.html)
# - [basics of running/analyzing thermal spectra](https://natashabatalha.github.io/picaso/notebooks/A_basics/5_AddingThermalFlux.html)
#
# **Important:** `exclude_mol` only removes **opacity**. It does NOT change the chemistry, the mean molecular weight, or the density of the atmosphere. Think of it as making a molecule "invisible" to radiation while keeping everything else the same. This is exactly what we want here: an isolated measure of the radiative role of a molecule.
#
# **Important:** Excluding line opacity requires the individual molecular opacities to be mixed on-the-fly (`method='resortrebin'`). It will not work with premixed/preweighted ck tables, since the molecules have already been combined.

# %%
from picaso import justplotit as jpi
from picaso import justdoit as jdi
import warnings
warnings.filterwarnings('ignore')
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# %%
# exclude_mol needs separable opacities so that individual molecules can be removed
opacity_ck = jdi.opannection(method='resortrebin')# grab your opacities

# %% [markdown]
# ## Set up the planet
#
# We will use the exact same WASP-39 b-like setup as the [planet climate tutorial](https://natashabatalha.github.io/picaso/notebooks/D_climate/2b_Exoplanet-ResortRebin-Chemeq.html).

# %%
cl_run = jdi.inputs(calculation="planet", climate = True) # start a calculation

tint= 200 # Intrinsic Temperature of your Planet in K
grav = 4.5 # Gravity of your Planet in m/s/s

cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity
cl_run.effective_temp(tint) # input effective temperature

T_star =5326.6 # K, star effective temperature
logg =4.38933 #logg , cgs
metal =-0.03 # metallicity of star
r_star = 0.932 # solar radius
semi_major = 0.0486 # star planet distance, AU

cl_run.star(opacity_ck, temp =T_star,metal =metal, logg =logg, radius = r_star,
            radius_unit=u.R_sun,semi_major= semi_major , semi_major_unit = u.AU)

# %% [markdown]
# ## Initial T(P) and convective zone guess
#
# We will use the same initial guess for both models so that the only difference between them is the CO$_2$ opacity.

# %%
nlevel = 91 # number of plane-parallel levels in your code
Teq=1000 # planet equilibrium temperature
pt = cl_run.guillot_pt(Teq, nlevel=nlevel, T_int = tint, p_bottom=2, p_top=-6)
temp_guess = pt['temperature'].values
pressure = pt['pressure'].values

rcb_guess = 85 # top most level of guessed convective zone
rfacv = 0.5 # planet-wide average

mh=10#10xsolar
cto_relative = 1 #1xsolar

# %% [markdown]
# ## Run 1: Climate model with all molecules
#
# This is our reference model. We do not specify `exclude_mol`, so every molecule contributes to the opacity.

# %%
cl_run.inputs_climate(temp_guess= temp_guess, pressure= pressure,
                      rcb_guess=rcb_guess, rfacv = rfacv)
cl_run.atmosphere(mh=mh, cto_relative=cto_relative, chem_method='visscher')

out_all = cl_run.climate(opacity_ck, save_all_profiles=True, with_spec=True)

# %% [markdown]
# ## Run 2: Climate model without CO$_2$ opacity
#
# Now we rerun `cl_run.atmosphere` exactly as before, adding `exclude_mol='CO2'`. We also call `inputs_climate` again to reset the initial guess. The previous climate run replaced the atmosphere profile in `cl_run` with its converged solution, and we want both runs to start from the same place.
#
# `exclude_mol` can take a few forms:
#
# - `exclude_mol='CO2'` : remove all CO$_2$ opacity (line, continuum, and Rayleigh)
# - `exclude_mol=['CO2','H2O']` : remove all opacity from several molecules
# - `exclude_mol={'CO2':['line']}` : remove only specific types of opacity (`line`, `continuum`, `rayleigh`, or `all`)

# %%
cl_run.inputs_climate(temp_guess= temp_guess, pressure= pressure,
                      rcb_guess=rcb_guess, rfacv = rfacv)
cl_run.atmosphere(mh=mh, cto_relative=cto_relative, chem_method='visscher',
                  exclude_mol='CO2')

out_noco2 = cl_run.climate(opacity_ck, save_all_profiles=True, with_spec=True)

# %% [markdown]
# Let's verify what we said above: CO$_2$ is still present in the chemistry of the "No CO$_2$" model. Only its opacity was removed.
#
# The abundances are not exactly identical. Each model converges to a different T(P) profile, and the equilibrium chemistry responds to the temperature, so the CO$_2$ abundance differs wherever the temperatures differ (here, the upper atmosphere).

# %%
plt.figure(figsize=(6,6))
plt.loglog(out_all['ptchem_df']['CO2'], out_all['ptchem_df']['pressure'], color='k', lw=3, label='All molecules')
plt.loglog(out_noco2['ptchem_df']['CO2'], out_noco2['ptchem_df']['pressure'], color='tab:red', ls='--', lw=3, label='No CO$_2$ opacity')
plt.ylim(1e2,1e-6)
plt.xlabel('CO$_2$ volume mixing ratio')
plt.ylabel('Pressure [bars]')
plt.legend()

# %% [markdown]
# ## Compare the T(P) profiles
#
# First, the simplest diagnostic: how does the converged temperature structure change?

# %%
plt.figure(figsize=(6,6))
plt.semilogy(out_all['temperature'],out_all['pressure'],color="k",linewidth=3,label="All molecules")
plt.semilogy(out_noco2['temperature'],out_noco2['pressure'],color="tab:red",linewidth=3,label="No CO$_2$ opacity")
plt.ylim(100,1e-6)
plt.ylabel("Pressure [Bars]", fontsize=20)
plt.xlabel('Temperature [K]', fontsize=20)
plt.legend(fontsize=12)

# %% [markdown]
# ## Understanding *why* the temperature changed: Analyzing Level-Fluxes
#
# The T(P) profile tells us *what* happened, but not *why*. The climate output contains the net fluxes at every level of the atmosphere in `out['flux_balance']`:
#
# - `flux_net_v` : net stellar (visible) flux. This is negative because it travels downward into the atmosphere. Its decrease in magnitude with depth tells you where starlight is absorbed.
# - `flux_net_ir` : net upward thermal (IR) flux. Its increase with altitude tells you where the atmosphere is losing energy to space.
#
# The difference in net flux across a layer tells us how much energy that layer gains or loses, and gives a radiative heating rate:
#
# $\dfrac{dT}{dt} = \dfrac{g}{c_p}\dfrac{dF_{net}}{dP}$
#
# In a converged radiative zone, the stellar heating and the thermal cooling of every layer balance each other. Removing a molecule's opacity upsets this balance, and the atmosphere adjusts its temperature until a new balance is reached.
#
# `jpi.flux_analyzer` computes and plots these quantities for a climate output:
#
# 1. Net absorbed stellar and net upward thermal flux at each level
# 2. Radiative heating (stellar) and cooling (thermal) rates
# 3. The outgoing thermal flux spectrum at the top of the atmosphere (`out['flux_ir_attop']`)
#
# Convective zones are shaded. Supplying `gravity` converts flux divergences into heating rates in K/s. Supplying `opacityclass` gives exact wavenumber bin widths for the outgoing flux spectrum.
#
# ### The energy budget of a single climate model
#
# Let's first look at our reference model on its own.

# %%
diag_all, fig = jpi.flux_analyzer(out_all, labels=('All molecules',),
                                  gravity=grav*u.Unit('m/(s**2)'),
                                  opacityclass=opacity_ck)

# %% [markdown]
# **Net fluxes:** Starlight (dashed) is absorbed gradually as it travels down, and almost none of it reaches below about 1 bar. The net upward thermal flux (solid) tracks it closely: in the radiative region, the energy absorbed below each level is re-emitted upward. Deep in the atmosphere, the net thermal flux settles to the internal heat flux, $\sigma T_{int}^4$ (~$10^5$ erg/cm$^2$/s for $T_{int}$=200 K).
#
# **Heating and cooling rates:** Stellar heating (dashed, positive) and thermal cooling (solid, negative) mirror each other throughout the radiative region, which is what a converged model should look like. Both rates become very large at low pressures, because there is very little mass there to heat or cool.
#
# **Thermal emission:** The outgoing thermal spectrum. Its integral (OLR, in the legend) is the total energy the planet radiates.

# %% [markdown]
# ### Comparing the models with and without CO$_2$
#
# If we pass a second climate output, `flux_analyzer` plots both models on the top row and adds a second row with the difference of each quantity (second model minus first).

# %%
diagnostics, fig = jpi.flux_analyzer(out_all, out_noco2,
                                     labels=('All molecules','No CO$_2$'),
                                     gravity=grav*u.Unit('m/(s**2)'),
                                     opacityclass=opacity_ck,
                                     title='Radiative impact of CO$_2$')

# %% [markdown]
# ### How to read these plots
#
# **Temperature change (T(P) plot above):** Look for the regions where the two profiles diverge. Where the "No CO$_2$" model is **warmer**, CO$_2$ was acting as a **coolant**: it was efficiently radiating energy to space. Where the "No CO$_2$" model is **cooler**, CO$_2$ was **warming** the atmosphere, either by absorbing starlight or by trapping outgoing thermal radiation (a greenhouse effect).
#
# **Change in net fluxes (left column):** Without CO$_2$, the atmosphere is more transparent at the wavelengths of the CO$_2$ bands. Look at the change in the net stellar flux: a positive change means more starlight reaches that level instead of being absorbed above it.
#
# **Change in heating and cooling rates (middle column):** For each model, the stellar heating (dashed) and thermal cooling (solid) mirror each other in the radiative region because the model is converged. The change in these rates shows which layers had their energy budget rearranged by the removal of CO$_2$.
#
# **Change in thermal emission (right column):** This shows *where in wavelength* the change happens. Without CO$_2$ there is a clear increase in emission in the 4.3 $\mu$m band (and the weaker 2.7 $\mu$m band). Thermal radiation that used to be absorbed by CO$_2$ now escapes from deeper, hotter layers. The bolometric outgoing flux (OLR, shown in the legend) is nearly identical between the two models. This is expected: a converged model must emit the energy it absorbs plus the internal heat, so the atmosphere rearranges its temperature (and emission at other wavelengths) to compensate.

# %% [markdown]
# ### Using the returned diagnostics
#
# `flux_analyzer` also returns all of the computed quantities (a dictionary for one model, or a list of two dictionaries when comparing) so that you can make your own plots or quantify the effect. For example, let's find the maximum warming and cooling caused by removing CO$_2$ and where they occur, and compare the total outgoing thermal flux.

# %%
dT = out_noco2['temperature'] - out_all['temperature']
p = out_all['pressure']
print(f"Max warming without CO2: {dT.max():.1f} K at {p[np.argmax(dT)]:.2e} bars")
print(f"Max cooling without CO2: {dT.min():.1f} K at {p[np.argmin(dT)]:.2e} bars")
print(f"Outgoing thermal flux, all molecules: {diagnostics[0]['olr']:.4e} erg/cm2/s")
print(f"Outgoing thermal flux, no CO2       : {diagnostics[1]['olr']:.4e} erg/cm2/s")

# %% [markdown]
# ## Compare the brightness temperature
#
# Finally, the brightness temperature spectra (see the [planet climate tutorial](https://natashabatalha.github.io/picaso/notebooks/D_climate/2b_Exoplanet-ResortRebin-Chemeq.html)) show the pressures we see at each wavelength. Since we ran both climate models with `with_spec=True`, we already have the spectra. Without CO$_2$ opacity, the 4.3 $\mu$m and 15 $\mu$m bands probe deeper into the atmosphere.

# %%
bt_all = jpi.brightness_temperature(out_all['spectrum_output'], plot=False)
bt_noco2 = jpi.brightness_temperature(out_noco2['spectrum_output'], plot=False)
wave = 1e4/out_all['spectrum_output']['wavenumber']

plt.figure(figsize=(12,6))
plt.semilogx(wave, bt_all, color='k', label='All molecules')
plt.semilogx(wave, bt_noco2, color='tab:red', label='No CO$_2$ opacity')
plt.xlim(1,20)
plt.ylim(500,2500)
plt.xlabel('Wavelength [$\\mu$m]', fontsize=16)
plt.ylabel('Brightness Temperature [K]', fontsize=16)
plt.legend(fontsize=13)

# %% [markdown]
# ## Try it yourself
#
# - Repeat this for H$_2$O, CH$_4$, or CO. Which molecule dominates the cooling of the upper atmosphere?
# - Use the dictionary form, e.g. `exclude_mol={'H2O':['line']}`, to separate the role of line opacity from continuum opacity
# - Change the metallicity (`mh`). Does CO$_2$ become more or less important for the climate as metallicity increases?
