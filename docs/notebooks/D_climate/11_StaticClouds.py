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
# # Static Clouds in Climate Models: What Is the Radiative Effect of a Cloud?
#
# In the [cloudy brown dwarf tutorial](https://natashabatalha.github.io/picaso/notebooks/D_climate/5_CloudyBrownDwarf_PreW.html) the cloud is computed with `virga` and recomputed every time the temperature changes, so the cloud and the climate converge together. Sometimes you want the opposite: a cloud that you choose at the start and that **stays fixed** while the climate model finds the temperature. This is useful when:
#
# - you want a controlled experiment that isolates what a cloud does to the climate (this tutorial)
# - your cloud comes from somewhere else (a GCM, a microphysics model, a retrieval) and you want the temperature structure consistent with it
# - you want to test how sensitive your climate is to cloud properties without changing anything else
#
# In this tutorial you will learn:
#
# 1. How to add a static (fixed) cloud to a climate model
# 2. How to use the energy budget of the atmosphere (level fluxes and radiative heating/cooling rates) to see *why* the cloud changes the climate
# 3. What the radiative effect of a cloud is: the competition between reflecting starlight (cooling), absorbing starlight high up (shielding the layers below), and trapping thermal radiation (a greenhouse effect)
#
# What you should already be familiar with:
#
# - [how to compute a planet 1D climate model](https://natashabatalha.github.io/picaso/notebooks/D_climate/2_Exoplanet_PreW.html) (we use the exact same WASP-39 b-like setup)
# - [how to analyze the energy budget of a climate model with flux_analyzer](https://natashabatalha.github.io/picaso/notebooks/D_climate/10_Exoplanet_ExcludeMol.html)
# - [how clouds are specified in PICASO](https://natashabatalha.github.io/picaso/notebooks/C_clouds/1_PairingPICASOToVIRGA.html)
#
# What you will need to download to use this tutorial: the same files as the [planet climate tutorial](https://natashabatalha.github.io/picaso/notebooks/D_climate/2_Exoplanet_PreW.html) (the 10x solar preweighted correlated-k table and the PHOENIX stellar models).

# %%
from picaso import justplotit as jpi
from picaso import justdoit as jdi
import os
import warnings
warnings.filterwarnings('ignore')
import astropy.units as u
import numpy as np
# %matplotlib inline

# %%
mh = '1.0' #log metallicity, 10xSolar
CtoO = '0.46' # CtoO ratio, Solar C/O
ck_db = os.path.join(os.getenv('picaso_refdata'),'opacities', 'preweighted', f'sonora_2121grid_feh{mh}_co{CtoO}.hdf5')
opacity_ck = jdi.opannection(ck_db=ck_db,method='preweighted') # grab your opacities

# %% [markdown]
# ## Set up the planet
#
# This is the same WASP-39 b-like planet as the [planet climate tutorial](https://natashabatalha.github.io/picaso/notebooks/D_climate/2_Exoplanet_PreW.html). All of our models start from the same initial guess, so that the only difference between them is the cloud.

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
cl_run.star(opacity_ck, temp =T_star,metal =metal, logg =logg, radius = r_star,database='phoenix',
            radius_unit=u.R_sun,semi_major= semi_major , semi_major_unit = u.AU)

nlevel = 91 # number of plane-parallel levels in your code
Teq=1000 # planet equilibrium temperature
pt = cl_run.guillot_pt(Teq, nlevel=nlevel, T_int = tint, p_bottom=2, p_top=-6)
temp_guess = pt['temperature'].values
pressure = pt['pressure'].values

rcb_guess = 85 # top most level of guessed convective zone
rfacv = 0.5 # planet-wide average

# %% [markdown]
# ## Run 1: Cloud-free climate model
#
# This is our reference model, identical to the [planet climate tutorial](https://natashabatalha.github.io/picaso/notebooks/D_climate/2_Exoplanet_PreW.html).

# %%
cl_run.inputs_climate(temp_guess= temp_guess, pressure= pressure,
                      rcb_guess=rcb_guess, rfacv = rfacv)
out_clear = cl_run.climate(opacity_ck, with_spec=True, verbose=False)

# %% [markdown]
# ## Run 2: Climate model with a static cloud
#
# A static cloud is added with the same `clouds` function you would use for a spectrum. There is only one rule: **call `clouds()` after `inputs_climate()`**, so that the cloud is made on the climate pressure grid. If you do not also call `virga()`, the climate code holds the cloud fixed. Its opacity is included at every step, but it is never recomputed as the temperature changes.
#
# We will use the simplest possible cloud, a grey slab defined by:
#
# - `p` : log10 of the pressure at the **bottom** of the cloud (bars). `p=0` puts the cloud base at 1 bar.
# - `dp` : thickness of the cloud in log10 pressure. `dp=2` extends the cloud up to 0.01 bar.
# - `opd` : extinction optical depth of **each layer** inside the cloud
# - `w0` : single scattering albedo. `w0=1` means the particles only scatter light, `w0=0` means they only absorb it.
# - `g0` : asymmetry parameter. `g0=0` scatters equally forward and backward, `g0=1` scatters straight forward.
#
# We start with a bright, mostly scattering cloud: `w0=0.9`, `g0=0.8`. This cloud is grey: it has the same properties at every wavelength. Real clouds are not grey (see the `virga` tutorials), but a grey cloud makes it easier to build intuition.
#
# We turn on the printout this time (`verbose=True`): the header now says `clouds: fixed`. The run takes about as long as the cloud-free one, since `virga` is never called.

# %%
cloud_bright = dict(p=[0.0], dp=[2.0], opd=[0.5], g0=[0.8], w0=[0.9])

cl_run.inputs_climate(temp_guess= temp_guess, pressure= pressure,
                      rcb_guess=rcb_guess, rfacv = rfacv)
cl_run.clouds(**cloud_bright)
out_bright = cl_run.climate(opacity_ck, with_spec=True, verbose=True)

# %% [markdown]
# The output contains the cloud that was used (`out['cld_df']`), in the same format as a `virga` run, so you can save it alongside the profile or use it to compute spectra later. Since we used `with_spec=True`, we can also look at the cloud with `jpi.cloud`, which plots the cloud properties at each pressure and wavelength. The cloud is grey, so the maps are uniform in wavelength.

# %%
cld_df = out_bright['cld_df']
in_cloud = cld_df.loc[cld_df['opd'] > 0, 'pressure']
print(f"The cloud fills the layers between {in_cloud.min():.3f} and {in_cloud.max():.2f} bars")
print(f"Total optical depth of the cloud: {cld_df.groupby('wavenumber')['opd'].sum().max():.1f}")

jpi.cloud(out_bright['spectrum_output']['full_output'], backend='plotly')

# %% [markdown]
# ## How did the cloud change the climate?

# %%
jpi.pt([out_clear, out_bright], legend=['Cloud-free', 'Bright static cloud'], backend='plotly',
       y_range=[100, 1e-6], width=600, height=500)

# %% [markdown]
# Two things happened:
#
# 1. The atmosphere **at and just above the top of the cloud got warmer**.
# 2. The atmosphere **in the lower part of the cloud and below it got much colder**, by hundreds of Kelvin.
#
# The rest of this tutorial explains why, using the radiative fluxes.

# %% [markdown]
# ## The energy budget at equilibrium: flux_analyzer
#
# As in the [CO$_2$ tutorial](https://natashabatalha.github.io/picaso/notebooks/D_climate/10_Exoplanet_ExcludeMol.html), `jpi.flux_analyzer` uses the net fluxes that the climate model converged on (`out['flux_balance']`) to show where starlight is absorbed and where the atmosphere loses energy, and turns them into radiative heating and cooling rates with
#
# $\dfrac{dT}{dt} = \dfrac{g}{c_p}\dfrac{dF_{net}}{dP}$
#
# Since the cloudy output has a cloud (`cld_df`), the cloud layers are hatched.

# %%
diag_eq, fig = jpi.flux_analyzer(out_clear, out_bright,
                                 labels=('Cloud-free','Bright static cloud'),
                                 colors=('k','tab:blue'),
                                 gravity=grav*u.Unit('m/(s**2)'),
                                 opacityclass=opacity_ck,
                                 title='Cloud-free vs. static cloud (both converged)')

# %% [markdown]
# **Net fluxes (left):** In the cloud-free model, starlight (dashed) reaches about 1 bar before it is used up. With the cloud, almost no starlight gets below the cloud base. The net absorbed stellar flux at the top is also smaller: the cloud reflects some of the starlight back to space before it can be absorbed.
#
# **Heating/cooling rates (middle):** The model is converged, so in every radiative layer the stellar heating (dashed) and thermal cooling (solid) still balance each other. The cloud has moved *where* that balance happens: much more of the starlight is now absorbed in the upper part of the cloud, and much less below it.
#
# **Thermal emission (right):** The total outgoing thermal flux (OLR, in the legend) went down. A converged model emits what it absorbs (plus its internal heat), and the cloudy planet absorbs less starlight.
#
# This tells us how the atmosphere looks *after* it has adjusted to the cloud. To understand *what pushed it there*, we need to look at the moment the cloud appears, before the temperature has had time to change.

# %% [markdown]
# ## The instantaneous effect of the cloud
#
# A good way to understand "what pushed the climate there" or the forcing is to take your cloud-free profile, add the cloud, and compute the energy budget without fully converging the profile. this exercise might also be useful for a retrieved run that isnt necessarily in equillibrium to begin with. 
#
# `cl_run.energy_budget` does exactly this. It computes the level fluxes (`approx(get_lvl_flux=True)`) of an atmosphere at a fixed temperature:
#
# - the starlight, in a single column lit at $\mu_0$=0.5, which is the geometry our climate code uses
# - the thermal emission
#
# It returns them in the same format as a climate output, so everything we did above with `flux_analyzer` works on it too.
#
# `energy_budget` uses the atmosphere and the cloud that are currently in the bundle, and does not change them. To choose what it computes, we set them with the usual functions: `atmosphere(df=...)` sets the temperature profile (here, a converged climate profile), and `clouds()` / `clouds_reset()` add or remove the cloud. Right now `cl_run` still has the bright cloud from Run 2, so we compute three budgets:

# %%
#1. the converged cloudy model
cl_run.atmosphere(df=out_bright['ptchem_df'])
budget_bright = cl_run.energy_budget(opacity_ck)

#2. the converged cloud-free profile, with the cloud added
cl_run.atmosphere(df=out_clear['ptchem_df'])
budget_instant = cl_run.energy_budget(opacity_ck)

#3. the converged cloud-free profile, without the cloud (clouds_reset sets its opacity to zero)
cl_run.clouds_reset()
budget_clear = cl_run.energy_budget(opacity_ck)

# %% [markdown]
# ### The energy budget at the top of the atmosphere
#
# At the top of the atmosphere, a planet in equilibrium absorbs as much energy as it emits:
#
# $\underbrace{r_{st} (F_{\star}^{\downarrow} - F_{\star}^{\uparrow})}_{\rm absorbed\ starlight} + \underbrace{\sigma T_{int}^4}_{\rm internal\ heat} = \underbrace{F_{th}^{\uparrow} - F_{th}^{\downarrow}}_{\rm outgoing\ thermal}$
#
# `energy_budget` returns each of these terms, and the `imbalance` between them.

# %%
print(f"{'':28s} {'albedo':>7s} {'absorbed':>11s} {'outgoing':>11s} {'imbalance':>11s}   [erg/cm2/s]")
for name, b in [('Cloud-free', budget_clear), ('Cloud added, same T(P)', budget_instant),
                ('Cloudy, converged', budget_bright)]:
    print(f"{name:28s} {b['bond_albedo']:7.3f} {b['absorbed_stellar']:11.4e} {b['outgoing_thermal']:11.4e} {b['imbalance']:11.4e}")

# %% [markdown]
# The cloud-free and converged cloudy models are both balanced: their imbalance is tiny compared to fluxes of ~10$^6$ erg/cm$^2$/s. The moment the cloud appears, the balance is broken. The **cloud radiative effect** is this change, split in two parts and, by convention, positive when it warms the planet:
#
# - **Shortwave (stellar) effect** = change in absorbed starlight. A cloud that is brighter than the clear atmosphere reflects starlight back to space before it can be absorbed, so this is negative (cooling). This is the *albedo effect*.
# - **Longwave (thermal) effect** = decrease in the outgoing thermal flux. The cloud is opaque to the thermal radiation coming from the hot layers below, and emits from its own colder top instead, so less energy escapes. This is positive (warming). This is the *greenhouse effect* of the cloud.

# %%
sw_effect = budget_instant['absorbed_stellar'] - budget_clear['absorbed_stellar']
lw_effect = budget_clear['outgoing_thermal'] - budget_instant['outgoing_thermal']
print(f"Shortwave (albedo) effect   : {sw_effect:+.3e} erg/cm2/s ({sw_effect/budget_clear['absorbed_stellar']:+.1%} of the absorbed starlight)")
print(f"Longwave (greenhouse) effect: {lw_effect:+.3e} erg/cm2/s ({lw_effect/budget_clear['absorbed_stellar']:+.1%})")
print(f"Net cloud radiative effect  : {sw_effect+lw_effect:+.3e} erg/cm2/s")

# %% [markdown]
# - The instant the cloud appears, the planet absorbs about 10% less starlight (albedo effect), but it also emits about 18% less thermal radiation, because the hot layers below the cloud are hidden behind it (greenhouse effect).
# - The greenhouse effect wins, so the net effect is **positive**: the atmosphere now gains more energy than it loses, and must warm until it emits enough again. This is the warming we saw at the top of the cloud.
# - Once it has converged, the cloudy planet emits *less* than the cloud-free one. It is in balance with less absorbed starlight. The net cloud radiative effect tells you which way the atmosphere has to move from the moment the cloud appears. It does not compare the two converged models.
#
# The top-of-atmosphere budget tells us about the planet as a whole, not *where* in wavelength or in the atmosphere the energy is gained or lost. For that we need the fluxes at every level.

# %% [markdown]
# ### Where in wavelength does the cloud act?
#
# `energy_budget` also returns the level fluxes in every wavenumber bin (`budget['level_fluxes']`): the downward and upward starlight and thermal radiation at each level. Let's compare the fraction of the incident starlight that reaches the cloud base, and the fraction that is reflected back to space (the albedo spectrum).

# %%
lf_clear, lf_cloud = budget_clear['level_fluxes'], budget_instant['level_fluxes']
i_base = np.argmin(np.abs(budget_clear['pressure'] - in_cloud.max()))

jpi.spectrum(lf_clear['wavenumber'],
             [lf['stellar_down'][i_base]/lf['stellar_down'][0] for lf in (lf_clear, lf_cloud)],
             legend=['Cloud-free', 'Cloud added, same T(P)'], backend='plotly', x_axis_type='log', x_range=[0.3, 20],
             y_axis_label='Fraction of starlight', title='Starlight reaching the cloud base (1 bar)')

# %%
jpi.spectrum(lf_clear['wavenumber'],
             [lf['stellar_up'][0]/lf['stellar_down'][0] for lf in (lf_clear, lf_cloud)],
             legend=['Cloud-free', 'Cloud added, same T(P)'], backend='plotly', x_axis_type='log', x_range=[0.3, 20],
             y_axis_label='Fraction of starlight', title='Starlight reflected to space (albedo spectrum)')

# %% [markdown]
# **Without the cloud**, the only starlight that reaches 1 bar is blue light between about 0.4 and 0.55 $\mu$m. At shorter wavelengths Rayleigh scattering sends the light back up, at longer optical wavelengths the alkali metals (Na, K) absorb it, and in the infrared the molecular bands (H$_2$O, CH$_4$, CO) absorb it. Rayleigh scattering is also the only thing that reflects light, so the cloud-free planet is only reflective in the blue.
#
# **With the cloud**, even the blue light no longer reaches the cloud base. The albedo spectrum changes completely: the planet is now reflective in the red and near-infrared, *between* the molecular bands, where the light can travel down to the cloud and back out. Inside the bands, the gas above the cloud absorbs the light before it reaches the cloud. In the blue, the albedo actually goes *down*: each time a cloud particle scatters light it absorbs 10% of it (`w0=0.9`), so the cloud is darker than pure Rayleigh scattering.

# %% [markdown]
# ### Where in the atmosphere does the cloud heat and cool?
#
# Since the energy budget has the same format as a climate output, we can pass the cloud-free budget and the "cloud added" budget to `flux_analyzer`. Both have the same temperature, so the bottom row now shows how the cloud changes the heating and cooling **the moment it appears**.

# %%
diag_instant, fig = jpi.flux_analyzer(budget_clear, budget_instant,
                                      labels=('Cloud-free','Cloud added, same T(P)'),
                                      colors=('k','tab:blue'),
                                      gravity=grav*u.Unit('m/(s**2)'),
                                      opacityclass=opacity_ck,
                                      title='The instantaneous effect of the cloud')

# %% [markdown]
# **Change in net fluxes (bottom left):** The starlight (dashed) absorbed below each level drops everywhere. At the top of the atmosphere this is the albedo effect, and below the cloud nothing is left: the cloud **shields** the layers beneath it. The net upward thermal flux (solid) drops too, most of all inside the cloud. This is the greenhouse effect.
#
# **Change in heating and cooling (bottom middle):**
#
# - *Stellar heating (dashed):* the top of the cloud absorbs much more starlight than the gas did. The particles absorb some of the light, and the scattering makes the light bounce around, so the gas there has more chances to absorb it. Deeper in the cloud and below it there is less starlight left, so the stellar heating goes down.
# - *Thermal cooling (solid):* the cloud top now radiates to space much more (more cooling). Deeper inside the cloud, the layers can no longer radiate their energy to space through the cloud above them, so they cool less (positive change).
#
# The net result is a strong heating at the top of the cloud, and a net cooling in the lower part of the cloud and just below it: those layers lost more stellar heating than the greenhouse effect gave back. The converged temperature change follows this pattern: the cloud top warms and the lower cloud cools.
#
# Two regions show that the instantaneous change is only the first push:
#
# - The layers just *above* the cloud cool more than they are heated at first, yet they end up slightly warmer. Once the cloud top has warmed, its thermal emission heats them.
# - Below about 3 bar, the instantaneous change is ~0, since almost no starlight reaches these layers even without a cloud. They cool anyway: the deep radiative zone only carries the small internal heat flux upward, so its temperature is tied to that of the layers above it. When the bottom of the cloud cools, the deep atmosphere follows.
#
# This is sometimes called an **anti-greenhouse effect**. The cloud deposits starlight high in the atmosphere, where it is easily radiated back to space, so the deep atmosphere is shielded and cools. On an irradiated planet, *where* starlight is absorbed matters as much as *how much* is absorbed.

# %% [markdown]
# ## Run 3: Bright vs. dark clouds
#
# How much of this is due to the cloud *reflecting* starlight, and how much to the cloud *absorbing* it? Let's run a third model with a dark cloud that absorbs half the light it intercepts (`w0=0.5`). Everything else is the same.

# %%
cloud_dark = dict(cloud_bright, w0=[0.5])

cl_run.inputs_climate(temp_guess= temp_guess, pressure= pressure,
                      rcb_guess=rcb_guess, rfacv = rfacv)
cl_run.clouds(**cloud_dark)
out_dark = cl_run.climate(opacity_ck, with_spec=True, verbose=False)

#the dark cloud on the cloud-free profile
cl_run.atmosphere(df=out_clear['ptchem_df'])
budget_instant_dark = cl_run.energy_budget(opacity_ck)

# %%
jpi.pt([out_clear, out_bright, out_dark], backend='plotly',
       legend=['Cloud-free', 'Bright cloud (w0=0.9)', 'Dark cloud (w0=0.5)'],
       y_range=[100, 1e-6], width=600, height=500)

# %%
print(f"{'':24s} {'albedo':>7s} {'SW effect':>11s} {'LW effect':>11s} {'net effect':>11s}   [erg/cm2/s]")
for name, b in [('Bright cloud (w0=0.9)', budget_instant), ('Dark cloud (w0=0.5)', budget_instant_dark)]:
    sw = b['absorbed_stellar'] - budget_clear['absorbed_stellar']
    lw = budget_clear['outgoing_thermal'] - b['outgoing_thermal']
    print(f"{name:24s} {b['bond_albedo']:7.3f} {sw:+11.3e} {lw:+11.3e} {sw+lw:+11.3e}")

# %% [markdown]
# **Albedo:** the dark cloud reflects *less* starlight than the cloud-free atmosphere (albedo 0.035 vs. 0.055). It absorbs the blue light that Rayleigh scattering would otherwise have sent back to space. Its shortwave effect is therefore positive: a dark cloud warms the planet in both the shortwave and the longwave.
#
# **Where the starlight goes:** let's compare the instantaneous effect of the two clouds directly (dark minus bright, both on the cloud-free temperature profile).

# %%
diag_dark, fig = jpi.flux_analyzer(budget_instant, budget_instant_dark,
                                   labels=('Bright cloud, same T(P)','Dark cloud, same T(P)'),
                                   colors=('tab:blue','tab:brown'),
                                   gravity=grav*u.Unit('m/(s**2)'),
                                   opacityclass=opacity_ck,
                                   title='Dark vs. bright cloud: the instantaneous effect')

# %% [markdown]
# The dark cloud absorbs more of the starlight right at its top, instead of letting the scattered light spread deeper into the cloud (positive change in stellar heating at the cloud top, negative below it). Its top warms about twice as much as the top of the bright cloud. The lower part of the dark cloud receives even less starlight and cools more, and so does the deep atmosphere below it.
#
# So the *amount* of starlight a cloud reflects decides how much energy the planet gets in total. *Where* the cloud makes the starlight be absorbed decides which layers warm and which cool.

# %% [markdown]
# ## Summary: the radiative effect of a cloud
#
# A cloud changes the climate of an irradiated planet in three ways:
#
# 1. **Albedo effect:** a cloud that is brighter than the clear atmosphere sends starlight back to space before it can be absorbed, so less energy enters the planet (cooling). A dark cloud can do the opposite.
# 2. **Greenhouse effect (warming):** the cloud is opaque to the thermal radiation coming from below and emits from its own colder top, so less energy escapes to space. In both of our clouds this was larger than the albedo effect.
# 3. **Shielding (anti-greenhouse) effect:** the cloud makes starlight be absorbed high in the atmosphere, at the cloud top. The cloud top warms. The layers below lose their stellar heating and cool, and the deep atmosphere follows.
#
# `energy_budget` (the level fluxes at a fixed temperature) and `flux_analyzer` (heating and cooling rates) tell you where in wavelength and where in the atmosphere each of these happens, before and after the climate responds.
#
# On a brown dwarf there is no starlight, so only the greenhouse effect is left, and clouds warm the atmosphere below them (compare with the [cloudy brown dwarf tutorial](https://natashabatalha.github.io/picaso/notebooks/D_climate/5_CloudyBrownDwarf_PreW.html)).
#
# Keep in mind that a static cloud is a **controlled experiment, not a self-consistent model**. In a real atmosphere the cloud would move and change as the temperature changes: here the atmosphere below the cloud cooled by hundreds of Kelvin, which would change where the cloud species condense. For self-consistent clouds, use `virga` in the climate model (see the cloudy brown dwarf tutorials).

# %% [markdown]
# ## Fixing a cloud from virga
#
# Your static cloud does not have to be grey. Any cloud in the PICASO format (columns `pressure`, `wavenumber`, `opd`, `w0`, `g0`) can be passed with `clouds(df=...)`. For example, you can compute a `virga` cloud **once** on the cloud-free profile and keep it fixed during the climate run. This needs the `virga` Mie files (`virga_dir`) and a Kzz profile:
#
# ```python
# import virga.justdoit as vj
# cloud = vj.Atmosphere(['MgSiO3'], fsed=1, mh=10, mmw=2.3)
# cloud.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)'))
# cloud.ptk(df=jdi.pd.DataFrame({'pressure':out_clear['pressure'], 'temperature':out_clear['temperature'], 'kz':1e9}),
#           kz_min=1e5)
# cld = vj.compute(cloud, as_dict=True, directory=virga_dir)
#
# cl_run.inputs_climate(temp_guess= temp_guess, pressure= pressure, rcb_guess=rcb_guess, rfacv = rfacv)
# cl_run.clouds(df=vj.picaso_format(cld['opd_per_layer'], cld['single_scattering'], cld['asymmetry'],
#                                   pressure=cld['pressure'], wavenumber=1e4/cld['wave']))
# out_virga = cl_run.climate(opacity_ck, with_spec=True)
# ```
#
# Note that `cl_run.virga(...)` would instead turn on the **self-consistent** cloud, which is recomputed at every step.
#
# **Removing the cloud:** a cloud added with `clouds()` stays in `cl_run` for later climate runs and energy budgets. This is why we ran the cloud-free model first. Use `cl_run.clouds_reset()` to remove it (it sets the cloud opacity to zero), as we did for the cloud-free energy budget.

# %% [markdown]
# ## Try it yourself
#
# - Move the cloud deeper (e.g. `p=[1.0]`, a cloud base at 10 bars). Does it still cool the deep atmosphere?
# - Make the cloud optically thin (e.g. `opd=[0.01]`). Which of the three effects disappears first?
# - Change `g0`. Strongly forward-scattering particles reflect less light: how does this change the albedo effect?
# - Remove the star (`rfacv=0`). You now have a self-luminous object. Is the cloud warming or cooling the atmosphere below it?
