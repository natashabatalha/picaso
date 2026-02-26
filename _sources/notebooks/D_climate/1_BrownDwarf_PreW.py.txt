# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # One-Dimensional Climate Models: The Basics of Brown Dwarfs
#
# In this tutorial you will learn the very basics of running 1D climate runs. For a more in depth look at the climate code check out [Mukherjee et al. 2022](https://ui.adsabs.harvard.edu/abs/2022arXiv220807836M/abstract) (note this should also be cited if using this code/tutorial).
#
# What you should already be familiar with:
#
# - [basics of running/analyzing thermal spectra](https://natashabatalha.github.io/picaso/tutorials.html#basics-of-thermal-emission)
# - [how to analyze thermal emission spectra](https://natashabatalha.github.io/picaso/notebooks/workshops/ERS2021/ThermalEmissionTutorial.html)
#
# What you will need to download to use this tutorial:
#
# 1. [Download](https://doi.org/10.5281/zenodo.18636725) New 1460 PT, 196 wno Correlated-k Tables to be used by the climate code for opacity
# 2. [Download](https://zenodo.org/record/5063476/files/structures_m%2B0.0.tar.gz?download=1) the sonora bobcat cloud free `structures_` file so that you can validate your model run
#
#
# You can use the `data.get_data` helper function to get these files and add them to the default picaso location: `reference/opaities/preweighted` for the opacities and `reference/sonora/bobcat` for the structure files
#  >> import picaso.data as d
#
#  >>d.get_data(category_download='ck_tables',target_download='by-molecule')
#
#   >>d.get_data(category_download='sonora_grids',target_download='bobcat')
#

# %%
import os
import warnings
warnings.filterwarnings('ignore')
import picaso.justdoit as jdi
import picaso.justplotit as jpi
jpi.output_notebook()
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# %% [markdown]
# ## What does a climate model solve for?

# %% [markdown]
# 1D Radiative-Convective Equilibrium Models solve for atmospheric structures of brown dwarfs and exoplanets, which includes:
#
# 1\. The Temperature Structure (T(P) profile)
#
# 2\. The Chemical Structure
#
# 3\. Energy Transport in the atmosphere
#
# But these physical components are not independent of each other. For example, the chemistry is dependent on the T(P) profile, the radiative transfer is dependent on clouds and the chemistry and so on.
#
# `PICASO` tries to find the atmospheric state of your object by taking care of all of these processes and their interconnections self-consistently and iteratively. Therefore, you will find that the climate portion of `PICASO` is slower than running a single forward model evaluation.

# %% [markdown]
# ## Starting up the Run

# %% [markdown]
# You will notice that starting a run is nearly identical as running a spectrum. However, how we will add `climate=True` to our inputs flag. We will also specify `browndwarf` in this case, which will turn off the irradiation the object is receiving.
#
# New Parameter: **Effective Temperature**. This excerpt from [Modeling Exoplanetary Atmospheres (Fortney et al)](https://arxiv.org/pdf/1804.08149.pdf) provides a thorough description and more reading, if you are interested.
#
# >If the effective temperature, $T_{eff}$, is defined as the temperature of a blackbody of
# the same radius that would emit the equivalent flux as the real planet, $T_{eff}$ and $T_{eq}$
# can be simply related. This relation requires the inclusion of a third temperature,
# $T_{int}$, the intrinsic effective temperature, that describes the flux from the planet’s
# interior. These temperatures are related by:"
#
# >$T_{eff}^4 =  T_{int}^4 + T_{eq}^4$
#
# >We then recover our limiting cases: if a planet is self-luminous (like a young giant
# planet) and far from its parent star, $T_{eff} \approx  T_{int}$; for most rocky planets, or any
# planets under extreme stellar irradiation, $T_{eff} \approx T_{eq}$.

# %%
cl_run = jdi.inputs(calculation="browndwarf", climate = True) # start a calculation

#note you need to put the climate keyword to be True in order to do so
# now you need to add these parameters to your calculation

teff= 1000 # Effective Temperature of your Brown Dwarf in K
grav = 1000 # Gravity of your brown dwarf in m/s/s

cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity
cl_run.effective_temp(teff) # input effective temperature

# %% [markdown]
# Let's now grab our gaseous opacities, whose path we have already defined above. Again, this code uses a correlated-k approach for accurately capturing opacities (see [section 2.1.4; Mukerjee et al 2022](https://ui.adsabs.harvard.edu/abs/2022arXiv220807836M/abstract)).

# %%
mh = '0.0'#'+0.0' #log metallicity
CtoO = '0.46'# # CtoO absolute ratio
ck_db = os.path.join(os.getenv('picaso_refdata'),'opacities', 'preweighted', f'sonora_2121grid_feh{mh}_co{CtoO}.hdf5')

sonora_profile_db = os.path.join(os.getenv('picaso_refdata'),'sonora_grids','bobcat')

# and not the line by line opacities
opacity_ck = jdi.opannection(ck_db=ck_db,method='preweighted') # grab your opacities

# %% [markdown]
# ## Initial T(P)  Guess
#
# Every calculation requires an initial guess of the pressure temperature profile. The code will iterate from there to find the correct solution. A few tips:
#
# 1. We recommend **using typically 51-91 atmospheric pressure levels**. Too many pressure layers increases the computational time required for convergence. Too little layers makes the atmospheric grid too coarse for an accurate calculation.
#
# 2. Start with **a guess that is close to your expected solution**. We will show an example using an isothermal P(T) profile below so you can see the iterative process. Later though, we recommend leveraging pre-computed grids (e.g. Sonora) as a starting guess for Brown Dwarfs.
#

# %%
nlevel = 91 # number of plane-parallel levels in your code

#Lets set the max and min at 1e-4 bars and 500 bars

Pmin = 1e-4 #bars
Pmax = 500 #bars
pressure=np.logspace(np.log10(Pmin),np.log10(Pmax),nlevel) # set your pressure grid

temp_guess = np.zeros(shape=(nlevel)) + 500 # K , isothermal atmosphere guess

# %% [markdown]
# ## Initial Convective Zone Guess
#
# You also need to have a crude guess of the convective zone of your atmosphere. Generally the deeper atmosphere is always convective. Again a good guess is always the published SONORA grid of models for this. But lets assume that the bottom 7 levels of the atmosphere is convective.
#
# **New Parameters:**
#
# 1. `rcb_guess` : this defines the top most level of your guessed convective zone. If you don't have a clue where your convective zone might end be choose a number that is $\sim$nlevel-5 (a few pressure levels away from the very bottom of your grid)
# 2. `rfacv`: (See [Mukherjee et al. 2022](https://ui.adsabs.harvard.edu/abs/2022arXiv220807836M/abstract) Eqn. 20 `r_st`)
#
# Non-zero values of rst (aka "rfacv" legacy terminology) is only relevant when the external irradiation on the atmosphere is non-zero. In the scenario when a user is computing a planet-wide average T(P) profile, the stellar irradiation is contributing to 50% (one hemisphere) of the planet and as a result rst = 0.5. If instead the goal is to compute a night-side average atmospheric state, rst is set to be 0. On the other extreme, to compute the day-side atmospheric state of a tidally locked planet rst should be set at 1.

# %%
rcb_guess = 83 # top most level of guessed convective zone

# Here are some other parameters needed for the code.
rfacv = 0.0 #we are focused on a brown dwarf so let's keep this as is

# %% [markdown]
# Now we would use the inputs_climate function to input everything together to our cl_run we started.

# %%
cl_run.inputs_climate(temp_guess=temp_guess, pressure=pressure,
                      rcb_guess=rcb_guess, rfacv=rfacv)

# %% [markdown]
# ## Run the Climate Code

# %% [markdown]
#  The actual climate code can be run with the cl_run.run command. The save_all_profiles is set to True to save the T(P) profile at all steps. The code will now iterate from your guess to reach the correct atmospheric solution for your brown dwarf of interest.
#
#

# %%
out = cl_run.climate(opacity_ck, save_all_profiles=True, with_spec=True)

# %% [markdown]
# ## `xarray` model storage
#

# %%
preserve_clima = jdi.output_xarray(out, cl_run)

# %%
preserve_clima

# %% [markdown]
# ## Benchmark with Sonora Bobcat

# %%
pressure_bobcat,temp_bobcat = np.loadtxt(jdi.os.path.join(
                            sonora_profile_db,f"t{teff}g{grav}nc_m0.0.cmp.gz"),
                            usecols=[1,2],unpack=True, skiprows = 1)

plt.figure(figsize=(10,10))
plt.ylabel("Pressure [Bars]", fontsize=25)
plt.xlabel('Temperature [K]', fontsize=25)
plt.ylim(500,1e-4)
plt.xlim(200,3000)

plt.semilogy(out['temperature'],out['pressure'],color="r",linewidth=3,label="Our Run")

plt.semilogy(temp_bobcat,pressure_bobcat,color="k",linestyle="--",linewidth=3,label="Sonora Bobcat")


plt.minorticks_on()
plt.tick_params(axis='both',which='major',length =30, width=2,direction='in',labelsize=23)
plt.tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=23)

plt.legend(fontsize=15)

plt.title(r"T$_{\rm eff}$= 1000 K, log(g)=5.0",fontsize=25)



# %% [markdown]
# ## Climate Plots and Animations

# %% [markdown]
# ### Animate Convergence

# %% [markdown]
# We can also try to see how our initial guess of an isothermal atmosphere was changed by the code to reach the converged solution

# %%
import importlib;importlib.reload(jpi)

# %%
ani = jpi.animate_convergence(out, cl_run, opacity_ck,
    molecules=['H2O','CH4','CO','NH3'])

# %%
ani

# %% [markdown]
# ### Brightness Temperature

# %% [markdown]
# Checking the brightness temperature serves many useful purposes:
#
# 1. Intuition building. Allows you to see what corresponding temperature are you sensitive to at each wavelength
#
# Note that this temperature doesn't need to be the physical temperature of your atmosphere but if you can find the physical converged atmospheric temperature closest to this brightness temperature you can also get an idea of the atmospheric pressure from where the flux you are seeing is originating from.
#
# 2. Determining if your choice in bottom boundary pressure grid was correct.
#
# If your brightness temperature is such that you bottom out at the temperature corresponding to the highest pressure, you have not extended your grid to high enough pressures.
#
# Brightness Temperature Equation:
#
# $T_{\rm bright}=\dfrac{a}{{\lambda}log\left(\dfrac{{b}}{F(\lambda){\lambda}^5}+1\right)}$
#
# where a = 1.43877735x$10^{-2}$ m.K and b = 11.91042952x$10^{-17}$ m$^4$kg/s$^3$
#
# Let's calculate the brightness temperature of our current run and check if our pressure grid was okay.

# %%
brightness_temp, figure= jpi.brightness_temperature(out['spectrum_output'])

# %% [markdown]
# In the above plot you can see that your brightness temperature is nicely bound between the minimum and maximum temperature. Your run is good and your choice of pressure grid is also great. Well done team!
#
#
# ## Selecting an Adequate Pressure Grid
#
# For understanding and intuition building, let's do out a run where we purposely choose an incomplete pressure grid. Let's do the same run by the max pressure set at only 3 bars instead of 500 bars.

# %%
cl_bad_pres = jdi.inputs(calculation="brown", climate = True)
cl_bad_pres.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity
cl_bad_pres.effective_temp(teff)

nlevel = 91 # number of plane-parallel levels in your code
Pmin = 1e-4 #bars
Pmax = 3 #bars
pressure=np.logspace(np.log10(Pmin),np.log10(Pmax),nlevel) # set your pressure grid

temp_guess = np.zeros(shape=(nlevel)) + 500 # K , isothermal atmosphere guess

rcb_guess = 83 # top most level of guessed convective zone

rfacv = 0.0 #we are focused on a brown dwarf so let's keep this as is

cl_bad_pres.inputs_climate(temp_guess= temp_guess, pressure= pressure,
                      rcb_guess=rcb_guess, rfacv = rfacv)
out_bad_pres = cl_bad_pres.climate(opacity_ck, save_all_profiles=True,with_spec=True)

# %% [markdown]
# Lets plot the profile from our new run and check it against our old run.

# %%
plt.figure(figsize=(10,10))
plt.ylabel("Pressure [Bars]", fontsize=25)
plt.xlabel('Temperature [K]', fontsize=25)
plt.ylim(30,1e-4)
plt.xlim(200,1200)

plt.semilogy(out['temperature'],out['pressure'],color="r",linewidth=3,label="Good Run")
plt.semilogy(out_bad_pres['temperature'],out_bad_pres['pressure'],color="b",linewidth=3,label="Bad Pressure Run")

plt.semilogy(temp_bobcat,pressure_bobcat,color="k",linestyle="--",linewidth=3,label="Sonora Bobcat")


plt.minorticks_on()
plt.tick_params(axis='both',which='major',length =30, width=2,direction='in',labelsize=23)
plt.tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=23)

plt.legend(fontsize=15)

plt.title(r"T$_{\rm eff}$= 1000 K, log(g)=5.0",fontsize=25)



# %% [markdown]
# This new profile is slightly off from our run and also the sonora bobcat run. Lets look at its brightness temperature as a function of wavelength and check if it matches well with our previous run.

# %%
brightness_temp_bad, figure_bad= jpi.brightness_temperature(
    out_bad_pres['spectrum_output'])

# %% [markdown]
# See how the brightness temperature from this new run is different from our previous succesful run. The brightness temperatures infact goes over the maximum temperature achieved by the model. Therefore the pressure grid used for this run is incorrect because one can look through the atmosphere to the bottom of the grid at most wavelengths which is not good and the resultant "converged" T(P) profile is also visibly inaccurate as a result as well.

# %% [markdown]
# ## Post Process High Resolution Spectrum
# We can quickly do this by resetting the opannection to not use the ck database and use the `ptchem_df` DataFrame as input for the atmosphere
#
# This is also the point where you could post-process clouds using `virga` or a `box model` as seen in these tutorials here:
# 1. [Adding clouds with virga](https://natashabatalha.github.io/picaso/notebooks/7_PairingPICASOToVIRGA.html)
# 2. [Adding box model clouds](https://natashabatalha.github.io/picaso/notebooks/5_AddingTransitSpectrum.html#Adding-Grey-Cloud)

# %%
opa_mon = jdi.opannection()

hi_res = jdi.inputs(calculation="browndwarf") # start a calculation
teff= 1000 # Effective Temperature of your Brown Dwarf in K
grav = 1000 # Gravity of your brown dwarf in m/s/s

hi_res.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity

hi_res.atmosphere(df=out['ptchem_df'])
df_spec = hi_res.spectrum(opa_mon, calculation='thermal', full_output=True)
w,f = jdi.mean_regrid(df_spec['wavenumber'],df_spec['thermal'], R=100)

# %%
preserve_clima

# %%
opa_mon = jdi.opannection()
hi_res = jdi.input_xarray(preserve_clima,opa_mon, calculation='browndwarf')
df_spec = hi_res.spectrum(opa_mon, calculation='thermal', full_output=True)

# %%
preserve_hires = jdi.output_xarray(df_spec, hi_res)

# %%
w,f = jdi.mean_regrid(df_spec['wavenumber'],df_spec['thermal'], R=100)

# %%
jpi.show(jpi.spectrum(w,f,x_axis_type='log',y_axis_type='log'))

# %% [markdown]
# ## `xarray` model storage with post-processed models

# %%
preserve_hires = jdi.output_xarray(df_spec, hi_res)

# %%
preserve_hires

# %%
