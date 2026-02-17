# ---
# jupyter:
#   jupytext:
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
# # Intuition Building: Young Planet Spectroscopy
#
# What you will learn
#
# 1. What do formation models predict for the effective temperatures of young planets across different masses?
# 2. Given identical mass and age, what might two different formation scenarios lead the spectra to look like?
# 3. How do we dissect spectroscopy of planet atmospheres in order to infer atmospheric physical properties such as abundance and climate profiles?
#
# What you should already know:
#
# 1. Complete all [Installation instructions](https://natashabatalha.github.io/picaso/installation.html)
#     - This involves downloading two files, one of which is large (6 Gig). So plan accordingly!
#
# 2. Download and untar the [Sonora Grid of Models](https://zenodo.org/record/1309035#.YJov_2ZKh26). They do not need to go in any specific folder. As long as you correctly point to them below when you define `sonora_profile_db`.
#
#
# *Optional: look through [The Basics of Thermal Emission](https://natashabatalha.github.io/picaso/tutorials.html#basics-of-thermal-emission). This tutorial walks you through computing a thermal emission spectrum. If you have never done so, this may be an helpful extra tutorial. However, you can complete this tutorial without it.*
#
#
# **Questions?** [Submit an Issue to PICASO Github](https://github.com/natashabatalha/picaso/issues) with any issues you are experiencing. Don't be shy! Others are likely experiencing similar problems

# %%
import warnings
warnings.filterwarnings(action='ignore')
import os
import pandas as pd
import numpy as np
#point to your sonora profile grid that you untared (see above cell #2)
# sonora_profile_db = '/data/sonora_profile/'
sonora_profile_db = os.path.join(os.getenv('picaso_refdata'),'sonora_grids','bobcat')

# %%
#if you are having trouble with environment variables
#os.environ['picaso_refdata'] = 'path/to/picaso_refdata'

# %% [markdown]
# Here are the two main `PICASO` functions you will be exploring:
#
# `justdoit` contains all the spectroscopic modeling functionality you will need in these exercises.
#
# `justplotit` contains all the of the plotting functionality you will need in these exercises.
#
# Tip if you are not familiar with Python or `jupyter notebooks`:
# - In any cell you can write `help(INSERT_FUNCTION)` and it will give you documentation on the input/output
# - If you type `jdi.` followed by "tab" a box will pop up with all the available functions in `jdi`. This applies to any python function (e.g. `np`, `pd`)

# %%
import picaso.justdoit as jdi
import picaso.justplotit as jpi
jpi.output_notebook()

# %%
help(jpi.spectrum)

# %% [markdown]
# ## Planet Evolution Tracks
#
# First we will use this `picaso` function to load some evolution tracks computed using the [Marley et al. 2007](#References) methodology.
#
# ### Basics of Two Evolution Tracks
#
# Stellar-like planet formation models typically start with a hot, ~2 Jupiter radius initial planet that quickly cools off. This is the type of planet that might result from fragmentation or gravitational instability. These have been termed “hot start” models.
#
# Core accretion planet formation models first produce a rocky/icy core which eventually grows massive enough to pull in gas from the nearby nebula. The rapidly infalling gas accretes through a shock and, in doing so, radiates away a great deal of their gravitational potential energy. The result is an initially cooler, smaller, less luminous planet. This type of formation pathway has been deemed “cold start”. There is likely a continuum of formation pathways between cold and hot start, but here we will just consider two extreme cases.

# %%
evo = jdi.evolution_track(mass='all',age='all')
#parse out the two formation model tracks
evo_hot = evo['hot']
evo_cold = evo['cold']

# %% [markdown]
# Since we have selected `all` for `mass` and `age`, we have the following columns:
# - age_years : Age of planet(years)
# - Teff `X` Mj : Effective temperature of the planet (Kelvin) where `X` is the mass of the planet in Jupiter masses, as computed by the evolution model
# - grav_cgs`X`Mj : Gravity of planet (cm/s^2) where `X` is the mass of the planet in Jupiter masses
# - logL`X`Mj : Log luminosity of the planet (cm/s^2) where `X` is the mass of the planet in Jupiter masses

# %%
evo_cold.keys()

# %% [markdown]
# Let's visualize this data to get a sense for the different tracks. We will first make our own version of the plot. Then we will use the built in `picaso` function which adds a color bar and hover tools.
#
# *You can try creating your own version of this plot by plotting each of the `Teff` columns as a function of `age_years`*

# %%
f = jpi.figure(x_axis_type='log',height=400, width=500,y_axis_label='Teff',x_axis_label='Age Years')
#makes a color scale
colors = jpi.pals.viridis(10)
for i, ikey in enumerate(list(evo_hot.keys())[1:]):
    if 'Teff' in ikey:
            #pull out the mass integer from the column name
            mass = int(ikey[ikey.rfind('Teff'[-1])+1:ikey.find('M')])
            #there are about 10 masses so we can use the mass index for the color index
            icolor = mass -1
            f.line(x=evo_hot['age_years'],y=evo_hot[ikey],line_width=2,
                   color=colors[icolor]
                  ,legend_label='Hot Start')
            f.line(x=evo_hot['age_years'],y=evo_cold[ikey],line_width=2,
                   color=colors[icolor],line_dash='dashed'
                  ,legend_label='Cold Start')
jpi.show(f)

# %% [markdown]
# Now you basically know what the `picaso` function does to make this plot. If you are interested you can look at the source code to see how to add hover tools and a color bar.

# %%
jpi.show(jpi.plot_evolution(evo))

# %% [markdown]
# ## Create and analyze the spectrum of a planet along this evolution track
#
# Before we get into subtleties, let's understand how to compute and dissect a planet spectrum.
#
# For now, we will pick one age and one mass and analyze their differences. After you go through this once, feel free to choose different masses, and ages and repeat the exercise.
#

# %%
case_study = jdi.evolution_track(mass=8,age=1e7)
cold = case_study['cold']
hot = case_study['hot']

# %%
hot,cold

# %%
wave_range = [1,5]
opa = jdi.opannection(wave_range=wave_range)

# %% [markdown]
# The only difference in the code blocks below is the gravity and the effective temperature, which we can pull from the planet evolution tracks. For now, we will focus on absolute flux from the planet (as opposed to contrast, the ratio of planet to stellar flux). Therefore, we are relatively agnostic to the stellar spectrum.
#
# A quick refresher in running the `jdi.inputs` function:
#
# 1. First define an empty class by running `jdi.inputs`
# 2. Set the stellar parameters : `star(opacityclass, Teff, M/H, logg, radius, radius_unit)`
# 3. Set the `gravity` of the planet. In this case we have this information from evolution models.
# 4. Set the chemistry and pressure-temperature using the `sonora` grid 1D models that you downloaded.
# 5. Finally, compute the spectrum with calculation set to `thermal` for thermal emission (other options include `reflected` and `transmission`).
#

# %%
#HOT START
yph = jdi.inputs()
yph.star(opa, 5000,0,4.0,radius=1, radius_unit=jdi.u.Unit('R_sun'))
yph.gravity(gravity=hot['grav_cgs'] , gravity_unit=jdi.u.Unit('cm/s**2'))
yph.sonora(sonora_profile_db,  hot['Teff'])
hot_case = yph.spectrum(opa,calculation='thermal', full_output=True)

#COLD START
ypc = jdi.inputs()
ypc.star(opa, 5000,0,4.0,radius=1, radius_unit=jdi.u.Unit('R_sun'))
ypc.gravity(gravity=cold['grav_cgs'] , gravity_unit=jdi.u.Unit('cm/s**2'))
ypc.sonora(sonora_profile_db,  cold['Teff'])
cold_case = ypc.spectrum(opa,calculation='thermal', full_output=True)


# %% [markdown]
# Now we can use our first `PICASO` plotting function: `jpi.spectrum`. More plotting functions will follow

# %%
wno,spec=[],[]
for i in [cold_case, hot_case]:
    x,y = jdi.mean_regrid(i['wavenumber'],i['thermal'], R=100) #ADD UNITS, ? resolution , do not exceed 150
    wno+=[x]
    spec+=[y]
jpi.show(jpi.spectrum(wno,spec,legend=['Cold','Hot'], y_axis_type='log',
                     plot_width=500))

# %% [markdown]
# ## How to dissect a planetary spectrum
#
# It is great to be able to produce spectra. But let's dig into what is going on in these atmospheres that give the spectra their distinct bumps and wiggles. Specifically, we want to understand:
#
# 1. What chemical species are dominating each spectra in both regimes?
# 2. What chemical species are minor, but still present a visible influence on the spectrum (note we say "visible" not "observable" -- which we will get into in the last tutorial)
# 3. What is the approximate pressure-temperature profile of the planet?

# %% [markdown]
# ### Step 1: Check inputs to make sure input chemistry and climate are as expected
#
# For both the cold and hot start cases, let's inspect your input in order to make sure that it follows your intuition regarding the effective temperature, gravity that you have inputed.

# %%
#everything will be aggregated here
cold_case['full_output'].keys()

# %%
cold_case['full_output']['layer'].keys()

# %%
#see raw PT profile info
P = cold_case['full_output']['layer']['pressure']
T = cold_case['full_output']['layer']['temperature']

# %%
#pressure temperature profiles
jpi.show(jpi.row(
    [jpi.pt(cold_case['full_output'], title='Cold Start'),
     jpi.pt(hot_case['full_output'], title='Hot Start')]))

# %%
#all chemistry can be found here - mixing ratios is synonymous with chemical abundance
cold_case['full_output']['layer']['mixingratios'].head()

# %%
#chemistry profiles
jpi.show(jpi.row(
    [jpi.mixing_ratio(cold_case['full_output'],limit=15,
                      title='Cold Start',plot_height=400),
     jpi.mixing_ratio(hot_case['full_output'],limit=15,
                      title='Hot Start',plot_height=400)]))

# %% [markdown]
# *Right now I have limited to the highest abundance 15 molecules. However, with the full `pandas` dataframe above, you should be able to explore other trace species that were included in the final model.*
#
# Note the color ordering of the mixing ratio figure is returned as a function of decreasing mixing ratio abundance. You are looking at the highest 15 (or whatever you specified for `limit`) abundant molecules. Are they the same for each case? You should archive these high abundance molecules in your mind so that you can properly analyze your spectra in Step 2 and 3.
#

# %% [markdown]
# ### Step 2: Determine what the major molecular contributors are

# %%
cold_cont = jdi.get_contribution(ypc, opa, at_tau=1)
hot_cont = jdi.get_contribution(yph, opa, at_tau=1)

# %% [markdown]
# This output consists of three important items:
#
# `taus_per_layer`
# - Each dictionary entry is a nlayer x nwave that represents the per layer optical depth for that molecule.
#
# `cumsum_taus`
# - Each dictionary entry is a nlevel x nwave that represents the cumulative summed opacity for that molecule.
#
# `tau_p_surface`
# - Each dictionary entry is a nwave array that represents the pressure level where the cumulative opacity reaches the value specified by the user through `at_tau`.

# %%
#explore the output
hot_cont['tau_p_surface'].keys()

# %% [markdown]
# Let's take a look at the last one, optical depth ~ 1 surface, as it will give us the best global view of what is going on.
#
# Need to gain more intuition for optical depth? Play around with increasing and decrease the `at_tau` parameter in the `get_contribution` function.

# %%
figs=[]
for i,it in zip([cold_cont['tau_p_surface'], hot_cont['tau_p_surface']],['Cold Start','Hot Start']):
    wno=[]
    spec=[]
    labels=[]
    for j in i.keys():
        x,y = jdi.mean_regrid(opa.wno, i[j],R=100)
        if np.min(y)<5: # Bars
            wno+=[x]
            spec+=[y]
            labels +=[j]
    fig = jpi.spectrum(wno,spec,plot_width=600,plot_height=350,y_axis_label='Tau~1 Pressure (bars)',
                       y_axis_type='log',x_range=[1,5],
                         y_range=[1e2,1e-4],legend=labels)
    fig.title.text=it
    figs+=[fig]
jpi.show(jpi.column(figs))

# %% [markdown]
# Let's think through these main points:
#
# 1. Is there a difference between the continuum species? Does that make sense given your intuition of the temperature pressure profiles? *insert more reading on basic hydrogen continuum*
#
# 2. How has the interplay of H2O/CO2/CH4/CO changed between the two models? Does this check out given chemical equilibrium tables. [Moses et al. 2016 Fig. 8](#References) will help strengthen your intuition.
#
# 3. Focus on the CH4 $\tau$~1 pressure curve for both cases. Without looking at the abundance plots, what can you deduce about the vertical abundance profile of CH4 in the hot start case? Does it increase or decrease with pressure? *Hint: What distinguishes this CH4 curve from H2O, for instance?* At what pressure is this transition occurring? If you happen to know something about carbon-chemistry you might be able to surmise the approximate temperature associated with the pressure you have identified. If not, not to worry!!! We will explore this further in the next notebook.
#

# %% [markdown]
# ### Step 3: Compare the flux of the system with basic blackbody curves to build understanding of the climate structure
#
# Flux units are hard, but blackbodies are your friend. When producing emission spectra, it's helpful to produce your flux output with blackbodies. The `PICASO` function, `jpi.flux_at_top` takes in an array of pressures. With the pressures, it looks at your pressure-temperature profile, to determine what the temperature is at each pressure. Then, it computes a blackbody at each of those temperatures. Given this flux at the top output, you should be able to reproduce rough sketch of the pressure temperature profile.
#
# *Bonus*: Another good exercise is to use the Planck function to transform the flux out the top of the atmosphere, to a "brightness temperature". Students are encouraged to explore this and create the plot.
#

# %%
figs =[]
for title,data in zip(['Cold Start','Hot Start'],[cold_case, hot_case]):
    fig = jpi.flux_at_top(data, pressures=[10,1,0.1],R=100,title=title)
    fig.legend.location='bottom_right'
    figs+=[fig]
jpi.show(jpi.row(figs))

# %% [markdown]
# What can you take away from this plot?
#
# 1. What opacity contribution is absent at 1 micron from the cold start case that gives you access to comparatively high pressures?
#
#
# 2. Where is the flux emanating from across the 4-5 micron region for these two cases? How are they different? Referring back to your opacity contribution plot, what is causing this?
#
# 3. What is the range of pressures you will be sensitive to if conducting a 1-5 micron spectrum?
#
# 4. J, H, and K bands are popular bands for spectroscopy. What pressure ranges each of these bands sensitive to? Cross reference this against your opacity contribution plot. What molecules are dominating these different regions?
#
# 5. If you were limited to differential photometry (e.g. J-H, J-K, H-K) what two bands might you pick to maximize information from this system?
#
# 6. In addition to the two photometric bands you've chosen, what third ~1 micron in width spectroscopic band might you choose in this wavelength region? Assume there are no observational constraints across this 1-5 micron region.
#
# Wrap up:
#
# In this exercise we started with a mass and an age. In reality, we start with **luminosity and age** and try to infer formation and mass. We will explore this in the next exercise.

# %% [markdown]
# ## References
#
# [Marley, Mark S., et al. "On the luminosity of young Jupiters." The Astrophysical Journal 655.1 (2007): 541.](https://ui.adsabs.harvard.edu/abs/2007ApJ...655..541M/abstract)
#
# [Moses, Julianne I., et al. "On the composition of young, directly imaged giant planets." The Astrophysical Journal 829.2 (2016): 66.](https://ui.adsabs.harvard.edu/abs/2016ApJ...829...66M/abstract)

# %%
