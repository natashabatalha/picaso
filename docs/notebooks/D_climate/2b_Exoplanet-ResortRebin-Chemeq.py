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
# # One-Dimensional Climate Models: The Basics of Planets
#
# In this tutorial you will learn the very basics of running 1D climate runs. For a more in depth look at the climate code check out [Mukherjee et al. 2023](https://ui.adsabs.harvard.edu/abs/2023ApJ...942...71M/abstract) (note this should also be cited if using this code/tutorial).
#
# What you should already be familiar with:
#
# - [basics of running/analyzing thermal spectra](https://natashabatalha.github.io/picaso/notebooks/A_basics/5_AddingThermalFlux.html)
# - [how to analyze thermal emission spectra](https://natashabatalha.github.io/picaso/notebooks/workshops/ERS2021/ThermalEmissionTutorial.html)
# - [how to compute a Brown Dwarf 1D climate model](https://natashabatalha.github.io/picaso/notebooks/D_climate/1_BrownDwarf_PreW.html)

# %%
import warnings
warnings.filterwarnings('ignore')
import picaso.justdoit as jdi
import picaso.justplotit as jpi
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# %%
# In this case, since we're not using the pre-weighted grid, we don't need to set a ck_db and we can just set the method to 'resortrebin'
opacity_ck = jdi.opannection(method='resortrebin')# grab your opacities

# %% [markdown]
# ## Starting up the Run
#

# %% [markdown]
# You will notice that starting a run is nearly identical as running a spectrum and brown dwarf climate model. However, how we will add `climate=True` to our inputs flag. We will also specify `planet` in this case, which will turn on the irradiation the object is receiving.
#
# New Parameter (was also used in the Brown Dwarf tutorial): **Effective Temperature**. This excerpt from [Modeling Exoplanetary Atmospheres (Fortney et al)](https://arxiv.org/pdf/1804.08149.pdf) provides a thorough description and more reading, if you are interested.
#
# > If the effective temperature, $T_{eff}$, is defined as the temperature of a blackbody of
# the same radius that would emit the equivalent flux as the real planet, $T_{eff}$ and $T_{eq}$
# can be simply related. This relation requires the inclusion of a third temperature,
# $T_{int}$, the intrinsic effective temperature, that describes the flux from the planet’s
# interior. These temperatures are related by:
#
# > $T_{eff}^4 =  T_{int}^4 + T_{eq}^4$
#
# > We then recover our limiting cases: if a planet is self-luminous (like a young giant
# planet) and far from its parent star, $T_{eff} \approx  T_{int}$; for most rocky planets, or any
# planets under extreme stellar irradiation, $T_{eff} \approx T_{eq}$.

# %%
cl_run = jdi.inputs(calculation="planet", climate = True) # start a calculation

#note you need to put the climate keyword to be True in order to do so
# now you need to add these parameters to your calculation


tint= 200 # Intrinsic Temperature of your Planet in K
grav = 4.5 # Gravity of your Planet in m/s/s

cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity
cl_run.effective_temp(tint) # input effective temperature

# %% [markdown]
# Let's now input the host-star properties

# %%
T_star =5326.6 # K, star effective temperature
logg =4.38933 #logg , cgs
metal =-0.03 # metallicity of star
r_star = 0.932 # solar radius
semi_major = 0.0486 # star planet distance, AU

cl_run.star(opacity_ck, temp =T_star,metal =metal, logg =logg, radius = r_star,
            radius_unit=u.R_sun,semi_major= semi_major , semi_major_unit = u.AU)#opacity db, pysynphot database, temp, metallicity, logg


# %% [markdown]
# ## Initial T(P)  Guess
#
# Every calculation requires an initial guess of the pressure temperature profile. The code will iterate from there to find the correct solution. A few tips:
#
# 1. We recommend **using typically 51-91 atmospheric pressure levels**. Too many pressure layers increases the computational time required for convergence. Too little layers makes the atmospheric grid too coarse for an accurate calculation.
#
# 2. Start with **a guess that is close to your expected solution**. One easy way to get fairly close is by using the Guillot et al 2010 temperature-pressure profile approximation
#

# %%
nlevel = 91 # number of plane-parallel levels in your code

#Lets set the max and min at 1e-4 bars and 500 bars
Teq=1000 # planet equilibrium temperature
pt = cl_run.guillot_pt(Teq, nlevel=nlevel, T_int = tint, p_bottom=2, p_top=-6)
temp_guess = pt['temperature'].values
pressure = pt['pressure'].values

# %% [markdown]
# ## Initial Convective Zone Guess
#
# You also need to have a crude guess of the convective zone of your atmosphere. Generally the deeper atmosphere is always convective. Again a good guess is always the published SONORA grid of models for this. But lets assume that the bottom 7 levels of the atmosphere is convective.
#
# **New Parameters:**
#
# 1. `rfacv`: (See [Mukherjee et al. 2023](https://ui.adsabs.harvard.edu/abs/2023ApJ...942...71M/abstract) Eqn. 20 `r_st`)
#
# Non-zero values of rst (aka "rfacv" legacy terminology) is only relevant when the external irradiation on the atmosphere is non-zero. In the scenario when a user is computing a planet-wide average T(P) profile, the stellar irradiation is contributing to 50% (one hemisphere) of the planet and as a result rst = 0.5. If instead the goal is to compute a night-side average atmospheric state, rst is set to be 0. On the other extreme, to compute the day-side atmospheric state of a tidally locked planet rst should be set at 1.

# %%
rcb_guess = 85 # top most level of guessed convective zone

# Here are some other parameters needed for the code.
rfacv = 0.5 #we are focused on a brown dwarf so let's keep this as is

# %% [markdown]
# Now we would use the inputs_climate function to input everything together to our cl_run we started.

# %%
cl_run.inputs_climate(temp_guess= temp_guess, pressure= pressure,
                      rcb_guess=rcb_guess, rfacv = rfacv)
mh=10#10xsolar
cto_relative = 1 #1xsolar
#now that we are not using preweighted ck tables we need to tell picaso how to compute chemistry on the fly
cl_run.atmosphere(mh=mh, cto_relative=cto_relative, chem_method='visscher')

# %% [markdown]
# ## Run the Climate Code

# %% [markdown]
#  The actual climate code can be run with the cl_run.run command. The save_all_profiles is set to True to save the T(P) profile at all steps. The code will now iterate from your guess to reach the correct atmospheric solution for your brown dwarf of interest.
#
#

# %%
out = cl_run.climate(opacity_ck, save_all_profiles=True,with_spec=True)

# %%
base_case = jdi.pd.read_csv(jdi.HJ_pt(), sep=r'\s+')

plt.figure(figsize=(10,10))
plt.ylabel("Pressure [Bars]", fontsize=25)
plt.xlabel('Temperature [K]', fontsize=25)
plt.ylim(500,1e-4)
plt.xlim(200,3000)

plt.semilogy(out['temperature'],out['pressure'],color="r",linewidth=3,label="Our Run")

plt.semilogy(base_case['temperature'],base_case['pressure'],color="k",linestyle="--",linewidth=3,label="WASP-39 b ERS Run")


plt.minorticks_on()
plt.tick_params(axis='both',which='major',length =30, width=2,direction='in',labelsize=23)
plt.tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=23)

plt.legend(fontsize=15)

plt.title(r"T$_{\rm eff}$= 1000 K, log(g)=5.0",fontsize=25)



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
# where a = 1.43877735$\times$10$^{-2}$ m.K and b = 11.91042952$\times$10$^{-17}$ m$^4$kg/s$^3$
#
# Let's calculate the brightness temperature of our current run and check if our pressure grid was okay.

# %%
brightness_temp, figure= jpi.brightness_temperature(out['spectrum_output'])

# %% [markdown]
# ### Check Adiabat
#
# This plot and datareturn is helpful to check that there have been no issues with where the code has found the location of the convective zone(s). See below, dTdP never exceeds the adiabat.

# %%
cp, adiabat, dtdp, pressure = jpi.pt_adiabat(out, cl_run, opacity_ck, plot=True)

# %%
jdi.output_xarray(out,cl_run,savefile='w39.nc')

# %%
