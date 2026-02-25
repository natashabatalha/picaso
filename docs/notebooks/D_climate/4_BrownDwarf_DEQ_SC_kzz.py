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
# # One-Dimensional Climate Models: Brown Dwarfs w/ Disequilibrium Chemistry with Self-Consistent Kzz
#
# In this tutorial you will learn how to run 1d climate models with the effects of disequilibrium chemistry as was done for the Elf-OWL Grid [Mukherjee et al. 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv240200756M/abstract) (note this should also be cited along with [PICASO 4.0]() if using this code/tutorial).
#
# What you should already be familiar with:
#
# - [basics of running/analyzing thermal spectra](https://natashabatalha.github.io/picaso/tutorials.html#basics-of-thermal-emission)
# - [how to analyze thermal emission spectra](https://natashabatalha.github.io/picaso/notebooks/workshops/ERS2021/ThermalEmissionTutorial.html)
# - [how to run a basic 1d brown dwarf tutorial](https://natashabatalha.github.io/picaso/notebooks/climate/12a_BrownDwarf.html)
#
# What you should have downloaded:
#
# 1. [Download](https://doi.org/10.5281/zenodo.18644980) New 1460, 661 wno Correlated-k Tables to be used by the climate code for opacity by individual molecule
#
# You can easily do this with the `get_data` function:
#
# Use the `data.get_data` helper function to get resortrebin files and add them to the default picaso location: `reference/opaities/resortrebin`
#  >> import picaso.data as d
#
#  >>d.get_data(category_download='ck_tables',target_download='by-molecule')
#

# %% [markdown]
# ### First, check that you have downloaded and placed the correlated-k files in the correct folder

# %%
import os
os.listdir(os.path.join(os.getenv('picaso_refdata'),'opacities','resortrebin')) #should show you a list of files

# %%
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
# ## Setting up Initial Run (highlighting main differences for disequilibrium)

# %%
# sonora_profile_db = '/Users/nbatalh1/Documents/data/sonora_bobcat/structures_m+0.0' #recommended download #2 above
sonora_profile_db = os.path.join(os.getenv('picaso_refdata'),'sonora_grids','bobcat')

gases_fly = ['CO','CH4','H2O','NH3','CO2','N2','HCN','H2','He','PH3','C2H2','Na','K','TiO','VO','FeH']

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
# ### Setting K$_{zz}$
#
# We will add one more concept which is the addition of  K$_{zz}$ [cm$^2$/s]. K$_{zz}$ is the eddy diffusion constant, which sets the strength of vertical mixing. In `PICASO` we have two options for  K$_{zz}$:
#
#  1. Constant value: sets a constant at every atmospheric layer
#  2. Self consistent (see Eqn. 27 and 28 in [Mukherjee et al 2022](https://arxiv.org/pdf/2208.07836.pdf))
#
#
# **New code parameters**:
#
# 1. `diseq_chem=True` : Turns on disequilibrium chemistry
# 2. `self_consistent_kzz` : (True/False) This solves self consistently for
# 3. `save_all_kzz` : (True/False) Similar to `save_all_profiles` this saves your intermediate k_zz values if you are trying to solve for a `self_consistent_kzz=True`.
# 4. `kz` : constant value if `self_consistent_kzz=False`
#
# **Which of those 6 do I need change change**
#
# Likely you will only be changing `kz` and/or, for example, playing around with a `self_consistent_kzz` vs a `constant profile`. Unless you are certain, we recommend the following set of `gases_fly` to remain unchanged.
#
# In this case, we are going to use a self-consistent kzz. Because of this we don't need to define `kz`, we just need to make sure that `self_consistent_kzz = True`.
#

# %%
cl_run.inputs_climate(temp_guess= temp_guess, pressure= pressure,
                        rcb_guess=rcb_guess, rfacv = rfacv)
mh=1#NOT LOG
cto_relative = 1 #relative to solar

#now that we are not using preweighted ck tables we need to tell picaso how to compute chemistry on the fly
cl_run.atmosphere(mh=mh, cto_relative=cto_relative, chem_method='visscher', quench=True)

out = cl_run.climate(opacity_ck, save_all_profiles = True, with_spec=True,
        save_all_kzz = False, self_consistent_kzz=True,diseq_chem = True)


# %% [markdown]
# ## Compare Diseq and Chemeq Climate Profile
#
# For the case we chose to do a self-consistent kzz instead of a low, constant kzz. We also use the resort-rebin chemistry method compared to the pre-weighted CK tables. For more information about the difference between these, you can look at the [Fun with Chem notebook]()

# %%
plt.ylim(200,1.7e-4)
plt.semilogy(out['temperature'],out['pressure'],"r", label='Resort-Rebin, Chemical Disequilibrium, Self Consistent Kzz')
plt.semilogy(temp_guess,pressure,color="k",linestyle="--", label='Pre-weighted CK, Chemical Equilibrium')
