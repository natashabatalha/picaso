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
# # Can formation scenarios and mass be determined with age and thermal spectroscopy
#
# What you will learn
#
# 1. If we only know the age of a young exoplanet, can we infer both the mass and the birth mechanism (hot vs. cold) just from its spectrum?
#
# What you should already know:
#
# 1. What do formation models predict for the effective temperatures of young planets across different masses?
# 2. Given identical mass and age, what might two different formation scenarios lead the spectra to look like?
# 3. How do we dissect spectroscopy of planet atmospheres in order to infer atmospheric physical properties such as abundance and climate profiles?
#
#
# **Questions?** [Submit an Issue to PICASO Github](https://github.com/natashabatalha/picaso/issues) with any issues you are experiencing. Don't be shy! Others are likely experiencing similar problems

# %%
import warnings
warnings.filterwarnings(action='ignore')
import os
import pandas as pd
import numpy as np

import picaso.justdoit as jdi
import picaso.justplotit as jpi
jpi.output_notebook()

#point to your sonora profile grid that you untared (see above cell #2)
# sonora_profile_db = '/data/sonora_profile/'
sonora_profile_db = os.path.join(os.getenv('picaso_refdata'),'sonora_grids','bobcat')

# %% [markdown]
# ## Planet Evolution Tracks in the Context of Planet Discoveries
#
# How do these stack up against real data. Bolometric luminosities and ages of nearly all young planets and brown dwarfs were compiled from [Zhang et al. 2020](#References) (Tables 3-4) and [Zhang et al. 2021](#References) (Tables 4-5). The remaining 2 objects, beta Pic c & YSES-1c, are from [Nowak et al. 2020](#References) & [Bohn et al. 2020](#References), respectively. Lastly, there is one brand new object, COCONUTS-2b from [Zhang et al. 2021](#References).

# %%
#load curves again
evo = jdi.evolution_track(mass='all',age='all')

#load table from ZJ Zhang
data = jdi.young_planets()
data.head()

# %% [markdown]
# The data are in luminosity. So we need to change our evolution tracks.

# %%
data.keys()

# %%
fig = jpi.plot_evolution(evo, y = 'logL',
                         y_range=[26.5,30],x_range=[1e6,1e9],
                         plot_height=400, plot_width=500,
                         title='Thermal Evolution Against Data')

jpi.plot_multierror(data['age_Gyr']*1e9, data['log_lbol'] + np.log10(3.839e33),
                    fig,
                    dx_low = 1e9*data['age_Gyr_low_err'],
                    dx_up = 1e9*data['age_Gyr_upp_err'],
                    dy_low = data['log_lbol_low_err'],
                    dy_up = data['log_lbol_upp_err'],
                    error_kwargs={'line_width':1.5,'color':'black'},
                    point_kwargs={'line_color':'red','color':'white','size':6})

fig.legend.location='bottom_left'
jpi.show(fig)


# %% [markdown]
# 1. Which of these planets/brown dwarfs would have to be hot start?
# 2. Which of these planets/brown dwarfs would have to be cold start?
# 3. Which could be either?

# %% [markdown]
# ## Analyze the spectra of two planets with same age and luminosity
#
# Let's pick an ambiguous location along these cold/hot start cases. For example, the 10 Mj cold start curve crosses the 4 Mj hot start curve at an age of ~3.2e7 years. Let's take a look to see if we can differentiate these scenarios.

# %%
cold = jdi.evolution_track(mass=10,age=3.2e7)['cold'] #cold start, higher mass
hot = jdi.evolution_track(mass=4,age=3.2e7)['hot'] #hot start, lower mass

# %%
hot,cold

# %%
wave_range = [0.8,14]
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
    x,y = jdi.mean_regrid(i['wavenumber'],i['thermal'], R=100)
    wno+=[x]
    spec+=[y]
jpi.show(jpi.spectrum(wno,spec,legend=['Cold','Hot'], y_axis_type='log',
                     plot_width=500))

# %% [markdown]
# As you can immediately see, it is a lot more complicated to differentiate these!! Let's see if we can pick apart any differences

# %% [markdown]
# ## Application of Spectroscopy Analysis Skills
#
# In the previous exercise we went through these steps to analyze a spectrum:
#
# 1. Assess chemistry, pressure-temperature input
# 2. Assess contribution function of opacity
# 3. Assess "flux at top" in comparison with black body functions or brightness temperature
#
# We will focus on #2 in this demo.

# %%
cold_cont = jdi.get_contribution(ypc, opa, at_tau=1)
hot_cont = jdi.get_contribution(yph, opa, at_tau=1)

# %% [markdown]
# As a reminder, this output consists of three important items:
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
# Let's take a look at the last one, optical depth ~ 1 surface, as it will give us the best global view of what is going on

# %%
figs=[]
for i,it in zip([cold_cont['tau_p_surface'], hot_cont['tau_p_surface']],['Cold Start','Hot Start']):
    wno=[]
    spec=[]
    labels=[]
    for j in i.keys():
        x,y = jdi.mean_regrid(opa.wno, i[j],R=100)
        if np.min(y)<5:
            wno+=[x]
            spec+=[y]
            labels +=[j]
    fig = jpi.spectrum(wno,spec,plot_width=600,plot_height=350,y_axis_label='Tau~1 Pressure (bars)',
                       y_axis_type='log',x_range=[1,6],
                         y_range=[1e2,1e-4],legend=labels)
    fig.title.text=it
    figs+=[fig]
jpi.show(jpi.column(figs))

# %% [markdown]
# Though these two cases look nearly identical, what is the main difference that is ultimately visible in the spectra?
#
# Any other insight we can glean form the the flux plot?

# %%
figs =[]
for title,data in zip(['Cold Start','Hot Start'],[cold_case, hot_case]):
    fig = jpi.flux_at_top(data, pressures=[10,1,0.1],R=100,title=title)
    fig.legend.location='bottom_right'
    figs+=[fig]
jpi.show(jpi.row(figs))

# %% [markdown]
# Revisit questions concerning observables. Would any of your answers change?
#
# 1. What do each of the spectroscopic bands provide you? J, H and K? What do the JWST modes get you? You can use [the PandExo graphic for guidance](https://exoctk.stsci.edu/pandexo/calculation/new)
# 2. If you were limited to differential photometry (e.g. J-H, J-K, H-K) what two bands might you pick to maximize information from this system? Does photometry help at all?
# 3. In addition to the two photometric bands you've chosen, what third 1 micron in width spectroscopic band might you choose in this wavelength region? Assume there are no observational constraints across this 1-14 micron region.
#
# Then move to discuss:
#
# 1. If photometry is not suitable for this problem, what spectroscopic bands are most suitable for differentiating formation scenarios?
#
# Final discussion:
#
# 1. If we only know the age of a young exoplanet, can we infer both the mass and the birth mechanism (hot vs. cold) just from its spectrum? What aspects have we not considered? What could help? What could complicate things further?

# %% [markdown]
# ## References
#
# [Bohn, Alexander J., et al. "Two Directly Imaged, Wide-orbit Giant Planets around the Young, Solar Analog TYC 8998-760-1." The Astrophysical Journal Letters 898.1 (2020): L16.](https://ui.adsabs.harvard.edu/abs/2020ApJ...898L..16B/abstract)
#
# [Nowak, Mathias, et al. "Direct confirmation of the radial-velocity planet β Pictoris c." Astronomy & Astrophysics 642 (2020): L2.](https://ui.adsabs.harvard.edu/abs/2020A%26A...642L...2N/abstract)
#
# [Zhang, Zhoujian, et al. "COol Companions ON Ultrawide orbiTS (COCONUTS). I. A High-gravity T4 Benchmark around an Old White Dwarf and a Re-examination of the Surface-gravity Dependence of the L/T Transition." The Astrophysical Journal 891.2 (2020): 171.](https://ui.adsabs.harvard.edu/abs/2020ApJ...891..171Z/abstract)
#
# [Zhang, Zhoujian, et al. "The Hawaii Infrared Parallax Program. V. New T-dwarf Members and Candidate Members of Nearby Young Moving Groups." The Astrophysical Journal 911.1 (2021): 7.](https://ui.adsabs.harvard.edu/abs/2021ApJ...911....7Z/abstract)
#
# [Zhang, Zhoujian, et al. "The Second Discovery from the COol Companions ON Ultrawide orbiTS (COCONUTS) Program: A Cold Wide-Orbit Exoplanet around a Young Field M Dwarf at 10.9 pc." arXiv preprint arXiv:2107.02805 (2021).](https://ui.adsabs.harvard.edu/abs/2021arXiv210702805Z/abstract)

# %%
