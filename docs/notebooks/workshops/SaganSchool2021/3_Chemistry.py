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
# # Chemistry: Young Planet Spectroscopy
#
# What you will learn:
#
# 1. What happens to the spectra of planet atmospheres as they cool?
# 2. What is happening from a chemical standpoint to cause these spectral features?
# 3. What molecules are good temperature probes?
#
# What you should know:
#
# 1. What do formation models predict for the effective temperatures of young planets across different masses?
# 2. Given identical luminosity and age, can formation scenarios and mass be determined?
# 3. How do we dissect spectroscopy of planet atmospheres in order to infer atmospheric physical properties such as abundance and climate profiles?
#

# %%
import warnings
warnings.filterwarnings(action='ignore')
import picaso.justdoit as jdi
import picaso.justplotit as jpi
jpi.output_notebook()
import os
import pandas as pd
import numpy as np
#point to your sonora profile grid that you untared (see above cell #2)
# sonora_profile_db = '/data/sonora_profile/'
sonora_profile_db = os.path.join(os.getenv('picaso_refdata'),'sonora_grids','bobcat')

# %%
wave_range = [0.6,5] #don't worry we will play around with this more later
opa = jdi.opannection(wave_range=wave_range)

# %% [markdown]
# ## What happens to an observable atmospheric spectrum as the planet ages
#
# In the previous workbook, we learned how to analyze spectra. Now we will compare a sequence of spectra as a function of age in order to gain an intuition for major transitions expected as a function of age (and by proxy, temperature and gravity).

# %%
case_study = jdi.evolution_track(mass=8, age='all')

# %% [markdown]
# There are a lot of grid points on this evolutionary track (150!). Let's pick the "hot" start case as it offers a more dramatic cooling event. This will enable us to learn about the chemical transitions that happen as a planet cools.

# %%
case_study['hot']

# %% [markdown]
# Let's take a feasible subset of these. I will choose ten, though if you are curious, or want to do the full track go for it!

# %%
i_to_compute = case_study['hot'].index[0::15]#take every 15th value

# %% [markdown]
# Let's run PICASO in a loop with the different effective temperatures and gravities

# %%
yph = jdi.inputs()
#let's keep the star fixed
yph.star(opa, 5000,0,4.0,radius=1, radius_unit=jdi.u.Unit('R_sun'))
yph.phase_angle(0)

#Let's stick the loop in right here!
hot_output={} #easy mechanism to save all the output
for i in i_to_compute:
    Teff = case_study['hot'].loc[i,'Teff']
    grav = case_study['hot'].loc[i,'grav_cgs']
    yph.gravity(gravity= grav,
                gravity_unit=jdi.u.Unit('cm/s**2'))
    yph.sonora(sonora_profile_db,  Teff)
    hot_case = yph.spectrum(opa,calculation='thermal', full_output=True)
    hot_output[f'{Teff}_{grav}'] = hot_case

# %% [markdown]
# Let's plot the sequence!!

# %%
wno,spec=[],[]
fig = jpi.figure(height=500,width=600, y_axis_type='log',
                 x_axis_label='Wavelength(um)',y_axis_label='Flux (erg/s/cm2/cm)')
for i,ikey in enumerate(hot_output.keys()):
    x,y = jdi.mean_regrid(hot_output[ikey]['wavenumber'],
                          hot_output[ikey]['thermal'], R=150)

    t,g=tuple(ikey.split('_'));g=int(np.log10(float(g))*1000)/1000
    a=fig.line(1e4/x,y,color=jpi.pals.Spectral11[::-1][i],line_width=3,
               legend_label=f'Teff={t}K,logg={g}')
fig.legend.location='bottom_right'

jpi.show(fig)

# %% [markdown]
# There is rich information encoded in these spectra. In order to fully grasp what is going on, it is important to understand the chemistry.

# %% [markdown]
# ## What molecules are most important to planetary spectroscopy?
#
# In the previous exercise we focused on look at the specific molecular contributions of two distinct cases. Therefore, we were focused on abundances as a function of pressure. Here we want you to get a handle on bulk abundance properties as a function of effective temperature. So we are going to **collapse** the pressure axis by taking the "median" value of each abundance array. By doing so, we want to see what the ~10 most abundant molecules are in each of these 10 spectra.

# %%
#remember the mixing ratios (or abundances) exist in this pandas dataframe
hot_output[ikey]['full_output']['layer']['mixingratios'].head()
#but this is too many molecules to keep track of for every single spectrum

# %%
relevant_molecules=[]
for i,ikey in enumerate(hot_output.keys()):
    abundances = hot_output[ikey]['full_output']['layer']['mixingratios']

    #first let's get the top 10 most abundance species in each model bundle we ran
    median_top_10 = abundances.median().sort_values(ascending=False)[0:10]
    relevant_molecules += list(median_top_10.keys())

#taking the unique of  relevant_molecules will give us the molecules we want to track
relevant_molecules = np.unique(relevant_molecules)

print(relevant_molecules)

# %% [markdown]
# Now that we have condensed this to a meaningful set of molecules, we can proceed to plot the sequence
#
#
# *Side note: You might try to see if the technique of taking the "median" yields the same results as "max" or "mean". This gives some insight into how dynamic molecular abundances are as a function of pressure*

# %% [markdown]
# ## Where in temperature space do chemical transitions seem to take place?

# %%
fig = jpi.figure(height=500,width=700, y_axis_type='log',
                 y_range=[1e-15,1],x_range=[200,2600],
                 x_axis_label='Planet Effective Temperature',y_axis_label='Abundance')

#now let's go back through our models and plot the abundances as a function of teff
relevant_molecules={i:[] for i in relevant_molecules}
for i,ikey in enumerate(hot_output.keys()):
    abundances = hot_output[ikey]['full_output']['layer']['mixingratios'].median()

    #save each abundance
    for i in relevant_molecules.keys():
        relevant_molecules[i] += [abundances[i]]

#last loop to plot each line
for i,ikey in enumerate( relevant_molecules.keys()):
    fig.line(case_study['hot'].loc[i_to_compute,'Teff'], relevant_molecules[ikey],
               color=jpi.pals.Category20[20][i],line_width=3,legend_label=ikey)
fig.legend.location='bottom_right'
jpi.show(fig)

# %% [markdown]
# There is a lot happening but let's break it down in very broad digestible categories. I will ask you to look back to the spectra that you made in the first tutorial. However in some cases, those spectra might not be computed at high enough effective temperatures to explore the molecular contribution. In those cases, use the techniques you learned from the previous exercise (`jdi.get_contribution`) to answer the questions below:
#
# #### Universally abundant molecules:
# - Which are the few highest abundance molecules/elements that exist across all temperature?
# - In what ways do these molecules/elements contribute to planetary spectra?
#
# #### Carbon-bearing species (CO2, CH4, CO, C2H6):
# - Which molecules are good temperature indicators, meaning they only exist in certain temperature regimes?
# - For the molecules that are good temperature indicators, where do their transitions occur? Keep these numbers archived in the back of your brain as they are great to have for intuition
# - Do these molecules make a significant contribution to the spectra? Are they hard or easy to detect? At what wavelengths?
#
# #### Besides Carbon, what other non-metal-based molecules are dominant?
# - Are any of them indicators of high or low temperature?
# - Do any of them exhibit an interplay that is similar to that of the CH4/CO transition?
# - Do these molecules make a significant contribution to the spectra? Are they hard or easy to detect? At what wavelengths?
#
# #### What Alkali-based molecules/elements are dominant?
# - At what temperatures do these molecules/elements begin to appear?
# - Do these molecules make a significant contribution to the spectra? Are they hard or easy to detect? At what wavelengths?
#
# #### What Metal-based species are dominant?
# - At what temperatures do these molecules/elements begin to appear?
# - Do these molecules make a significant contribution to the spectra? Are they hard or easy to detect? At what wavelengths?
#
# #### SYNTHESIZE:
# - Across all these molecules, what are the few most critical temperature transitions? According to our two formation scenarios, what planet age does these effective temperatures correspond to?
