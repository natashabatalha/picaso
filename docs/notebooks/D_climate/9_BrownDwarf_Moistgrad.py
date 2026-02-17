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
# # Brown Dwarf Climate Model with Moist Adiabat
#
# In this tutorial we will test another parameter space using a moist adiabt which takes into consideration water latent heat release. You can read more details about the effects of this in [Tang et al. 2021](https://iopscience.iop.org/article/10.3847/1538-4357/ac1e90) for clear and partly cloudy atmospheres, and [Mang et al. 2024](https://iopscience.iop.org/article/10.3847/1538-4357/ad6c4c) for fully cloudy atmospheres.
#
# Here we're just going to test running the model for a clear atmosphere
#
# What you should already be familiar with:
# [basics of running/analyzing thermal spectra](https://natashabatalha.github.io/picaso/tutorials.html#basics-of-thermal-emission)

# %%
import sys
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

# %%
cl_run = jdi.inputs(calculation="browndwarf", climate = True) # start a calculation

#note you need to put the climate keyword to be True in order to do so
# now you need to add these parameters to your calculation

teff= 350 # Effective Temperature of your Brown Dwarf in K
grav = 100 # Gravity of your brown dwarf in m/s/s

cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity
cl_run.effective_temp(teff) # input effective temperature

# %%
# Here we're going to run a higher metallicity model since the effect of the latent heat is more pronounced at higher metallicities

mh = '+100' #log metallicity
CtoO = '100'# CtoO ratio relative to solar
ck_db = os.path.join(os.getenv('picaso_refdata'),'opacities', 'preweighted', f'sonora_2020_feh{mh}_co_{CtoO}.data.196') # recommended download #1 above
sonora_profile_db = os.path.join(os.getenv('picaso_refdata'),'sonora_grids','bobcat')

# and not the line by line opacities
opacity_ck = jdi.opannection(ck_db=ck_db,method='preweighted') # grab your opacities

# %%
nlevel = 91 # number of plane-parallel levels in your code

# let's start with a Sonora Bobcat profile
pressure_bobcat,temp_bobcat = np.loadtxt(jdi.os.path.join(
                            sonora_profile_db,f"t{teff}g{grav}nc_m0.0.cmp.gz"),
                            usecols=[1,2],unpack=True, skiprows = 1)

# %%
rcb_guess = 40 # top most level of guessed convective zone

# Here are some other parameters needed for the code.
rfacv = 0.0 #we are focused on a brown dwarf so let's keep this as is

# %% [markdown]
# Now we would use the inputs_climate function to input everything together to our cl_run we started. Here we'll make sure to set `moistgrad = True`

# %%
cl_run.inputs_climate(temp_guess=temp_bobcat, pressure=pressure_bobcat,
                      rcb_guess=rcb_guess, rfacv=rfacv, moistgrad=True)

# %%
out = cl_run.climate(opacity_ck, save_all_profiles=True, with_spec=True)

# %% [markdown]
# Now let's compare the PT profile with the one generated in Tang et al. 2021 to benchmark our model

# %%
pressure_tang, temp_tang = np.loadtxt("t350g100nc_m1.0.cmp.gz",
                            usecols=[1,3],unpack=True, skiprows = 1)

plt.figure(figsize=(10,10))
plt.ylabel("Pressure [Bars]", fontsize=25)
plt.xlabel('Temperature [K]', fontsize=25)
plt.ylim(500,1e-4)
plt.xlim(0,3000)

# plt.semilogy(temp_bobcat,pressure_bobcat,color="k",linestyle="--",linewidth=3,label="Sonora Bobcat, Dry Adiabat ")
plt.semilogy(temp_tang,pressure_tang,color="b",linestyle="--",linewidth=3,label="Tang et al. 2021 Model, Moist Adiabat")

plt.semilogy(out['temperature'],out['pressure'],color="r",linewidth=3,label="Our Run")

plt.minorticks_on()
plt.tick_params(axis='both',which='major',length =30, width=2,direction='in',labelsize=23)
plt.tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=23)

plt.legend(fontsize=15)

plt.title(r"T$_{\rm eff}$= 350 K, log(g)=4.0",fontsize=25)

# %% [markdown]
# Overall it looks pretty good! Slight differences are likely due to the difference in the number of layers used (Tang 2021 models use 68 layers and we have 91) as well as the fact that we're using updated ck tables.

# %% [markdown]
#
