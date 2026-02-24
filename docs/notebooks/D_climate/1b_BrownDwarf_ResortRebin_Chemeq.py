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
# # One-Dimensional Climate Models: Brown Dwarfs w/ Chemical Equilibrium and Resort-Rebin
#
# In this tutorial you will learn how to run 1d climate models with chemical equilibrium but using resort rebin instead of the pre-weighted chemeq tables.
#
#
# What you should already be familiar with:
#
# - [basics of running/analyzing thermal spectra](https://natashabatalha.github.io/picaso/tutorials.html#basics-of-thermal-emission)
# - [how to analyze thermal emission spectra](https://natashabatalha.github.io/picaso/notebooks/workshops/ERS2021/ThermalEmissionTutorial.html)
# - [how to run a basic 1d brown dwarf tutorial](https://natashabatalha.github.io/picaso/notebooks/climate/12a_BrownDwarf.html)
#
# What you should have downloaded:
#
# Use the `data.get_data` helper function to get resortrebin files and add them to the default picaso location: `reference/opaities/resortrebin`
#  >> import picaso.data as d
#
#  >>d.get_data(category_download='ck_tables',target_download='by-molecule')
#
# You should also already have the bobcat structure files:
#
#  >>d.get_data(category_download='sonora_grids',target_download='bobcat')
#

# %%
import os
import warnings
warnings.filterwarnings('ignore')
import picaso.justdoit as jdi
import picaso.justplotit as jpi
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
# #%matplotlib inline
from astropy import constants as const
from astropy import units as u
import sys
import pandas as pd

# %% [markdown]
# ## Setting up Initial Run (highlighting main differences for resort-rebin)
#
# In this case becuase we're mixing the gases on the fly, we don't need to define the correlated-k database/file. As you can see here in opannection, the method is now set to `resortrebin`. We've defined a list of gases to mix here but if you don't define anything all gasses will be mixed. For more information about the difference between these two methods, you can look at the [Fun with Chem notebook]()

# %%
sonora_profile_db = os.path.join(os.getenv('picaso_refdata'),'sonora_grids','bobcat')

gases_fly = ['CO','CH4','H2O','NH3','CO2','N2','HCN','H2','He','PH3','C2H2','Na','K','TiO','VO','FeH']

#change opacity connection
opacity_ck = jdi.opannection(method='resortrebin',preload_gases=gases_fly) # grab your opacities


# %%
cl_run = jdi.inputs(calculation="browndwarf", climate = True) # start a calculation


tint= 700
grav = 316 # Gravity of your Planet in m/s/s

cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity
cl_run.effective_temp(tint) # input effective temperature

nlevel = 91


# %% [markdown]
# We recommend starting with Sonora-Bobcat models as an initial guess.

# %%
pressure,temp_guess = np.loadtxt(jdi.os.path.join(
                            sonora_profile_db,f"t{tint}g{grav}nc_m0.0.cmp.gz"),
                            usecols=[1,2],unpack=True, skiprows = 1)

rcb_guess = 79 # top most level of guessed convective zone

# Here are some other parameters needed for the code.
rfacv = 0.0 #we are focused on a brown dwarf so let's keep this as is

# %% [markdown]
# In comparison to when running climate models with the preweighted ck tables, those we already defined the metallicity and C/O ratio in the format of the table files. Here since we're not using those tables, we need to define the metallicity and C/O ratio along with the type of chemistry we want to do.
#
# **New Parameters**
#
# `mh` : this is the metallicity **relative** to solar
#
# `cto_relative` : C/O **relative** to solar OR `cto_absolute` which makes it the actual C/O value
#
# `chem_method` : This will tell PICASO what kind of chemistry you want. The 4 options are `visscher`, `visscher_1060`, `on-the-fly`, and `photochem`.
#
# The difference between `visscher` and `visscher_1060` are the grid points used with `visscher` having the most updated chemistry as well. The `on-the-fly` option computes equilibrium chemistry at runtime using the equilibrium solver. We will get into `photochem` in another notebook doing photochemistry.
#
# You **MUST** include `mh` and one of the `cto...` inputs when using `visscher`, `visscher_1060`, `on-the-fly`, or `photochem` (`photochem` also needs `photochem_init_args`).

# %%
cl_run.inputs_climate(temp_guess= temp_guess, pressure= pressure,
                        rcb_guess=rcb_guess, rfacv = rfacv)

mh=1
cto_relative = 1

#now that we are not using preweighted ck tables we need to tell picaso how to compute chemistry on the fly
cl_run.atmosphere(mh=mh, cto_relative=cto_relative, chem_method='visscher')

# %%
out = cl_run.climate(opacity_ck, save_all_profiles = True, with_spec=True)

# %% [markdown]
# ## Compare Resort-rebin and Pre-weighted CK table derived Climate Profiles
#
# Now we can compare how these two methods perform against one another

# %%
plt.ylim(200,1.7e-4)
plt.semilogy(out['temperature'],out['pressure'],"r", label='Resort-Rebin, Chemical Equilibrium')
plt.semilogy(temp_guess,pressure,color="k",linestyle="--", label='Pre-weighted CK, Chemical Equilibrium')
plt.legend()
plt.ylabel('Pressure [bars]')
plt.xlabel('Temperature [K]')

# %%
