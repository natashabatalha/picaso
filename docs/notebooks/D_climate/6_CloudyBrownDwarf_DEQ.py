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
# # One-Dimensional Climate Models: Brown Dwarfs w/ Clouds in Chemical Disequilibrium
#
# In this tutorial you will learn how to run 1-D climate models for brown dwarfs with Virga clouds and chemical disequilibrium. For a more in depth look at the climate-cloud code check out [Mang et al. 2026]() (note this should also be cited if using this code/tutorial).
#
# You should already be familiar with running 1-D climate models with running a [simple clear brown dwarf model](https://natashabatalha.github.io/picaso/notebooks/climate/12a_BrownDwarf.html) and running clouds in [equilibrium]()
#
# What you need to have downloaded for clouds to work:
#
# [Virga](https://natashabatalha.github.io/virga/installation.html)
#

# %%
import os
import warnings
warnings.filterwarnings('ignore')
import picaso.justdoit as jdi
import picaso.justplotit as jpi
import virga.justdoit as vj
import virga.justplotit as cldplt
jpi.output_notebook()
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from bokeh.plotting import show, figure
import xarray as xr
import pickle

# %%
# #sonora bobcat cloud free structures file
# sonora_profile_db = '/data/sonora_bobcat/structure/structures_m+0.0'
# sonora_profile_db = '/Users/nbatalh1/Documents/data/sonora_bobcat/structures_m+0.0'
sonora_profile_db = os.path.join(os.getenv('picaso_refdata'),'sonora_grids','bobcat')

# %% [markdown]
# We are going to initialize our climate run like other disequilibrium runs without using the pre-weighted ck tables

# %%
cl_run = jdi.inputs(calculation="browndwarf", climate = True) # start a calculation

#note you need to put the climate keyword to be True in order to do so
# now you need to add these parameters to your calculation

teff= 400 # Effective Temperature of your Brown Dwarf in K
grav = 1000 # Gravity of your brown dwarf in m/s/s

cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity
cl_run.effective_temp(teff) # input effective temperature

gases_fly = ['CO','CH4','H2O','NH3','CO2','N2','HCN','H2','PH3','C2H2','Na','K','TiO','VO','FeH']
opacity_ck =  jdi.opannection(method='resortrebin',preload_gases=gases_fly)# grab your opacities

# %%
nlevel = 91 # number of plane-parallel levels in your code

# Here we're going to start with a cloudfree Sonora Elf Owl model
pressure,temp_guess = np.loadtxt("profilegrid_kz_1d9_qt_onfly_400_grav_1000_mh_+0.0_cto_1.0.dat",
                                usecols=[1,2],unpack=True, skiprows = 1)

# %%
rcb_guess = 45 # top most level of guessed convective zone
# for the sake of time of this tutorial, I set it to 40 because I know where it should be in this case. In general for clouds it is better
# to start deeper in the atmosphere and work your way up. It just takes more time.

# Here are some other parameters needed for the code.
rfacv = 0.0 #we are focused on a brown dwarf so let's keep this as is

# this is the Kzz parameter
kzval = pressure*0+1e9

# %%
virga_dir = os.path.join(os.getenv('picaso_refdata'),'virga') # path to virga directory with the Mie scattering files

# %% [markdown]
# All requirements needed for this notebook will be met from properly following the previous notebook for clouds in equilibrium chemistry
#
# **New PICASO code parameters**:
# We've introduced other optional chemical treatments into the disequilibrium runs that should be included in the `atmosphere()` function
#
# 1. `vol_rainout` : (True/False) Follows rainout chemistry (ie. Sonora Bobcat) even in disequilibrium. Default = False
# 2. `cold_trap` : (True/False) If True, the abundances of volatile species like H2O, NH3, and CH4 will not be allowed to increase after they begin to rainout. Default = False
# 3. `no_ph3` :  (True/False) If True, completely remove PH3 from the atmosphere. Default = False
#
# These parameters are included to increase flexibility for you when you want to generate your own model! To play around with these different parameters you can look at the [Fun with Chem notebook]()
#
# *Note* that all disequilibrium models default to follow the kinetic CO2 prescription described in [Zahnle & Marley 2014](https://ui.adsabs.harvard.edu/abs/2014ApJ...797...41Z/abstract) where the abundance quenches with respect to CO in disequilibrium. This is a correction applied to [Sonora Elf Owl v2](https://ui.adsabs.harvard.edu/abs/2025RNAAS...9..108W/abstract)
#
#
# Our recommendation to follow for the latest release of the Sonora family of models are as follows:
# 1. `vol_rainout` = True
# 2. `cold_trap` = True
# 3. `no_ph3` = True

# %%
cl_run.inputs_climate(temp_guess= temp_guess, pressure= pressure,
                      rcb_guess=rcb_guess, rfacv = rfacv)

cl_run.atmosphere(mh=1, cto_relative=1, chem_method='visscher',
                quench=True, cold_trap = True, no_ph3 = True, vol_rainout=True)#

cl_run.inputs['atmosphere']['profile']['kz']=kzval

# Let's add in a few different cloud species that we expect to condense in the deeper parts of the atmosphere of this 400 K brown dwarf
df_cld_guess=cl_run.virga(condensates=['Cr', 'MnS', 'Na2S', 'ZnS', 'KCl'], fsed=2.0, directory=virga_dir, latent_heat=True)

# %% [markdown]
# In this case, with a good starting radiative-convective boundary guess and a good starting PT profile, this run should take ~10 minutes to compute. If it takes significantly longer longer there might be something wrong. Most times though, cloudy runs do take much longer on timescales of 30 minutes to an hour depending on how deep your intial radiaitve-convective zone boundary is.

# %%
out = cl_run.climate(opacity_ck, save_all_profiles = True, with_spec=True,
        diseq_chem = True, self_consistent_kzz =False)

# %% [markdown]
# ## Plot the P-T Profile

# %% [markdown]
# Now we can plot the results, first let's grab the condensation curve for all our cloud species

# %%
kcl_cond_p, kcl_cond_t = vj.condensation_t('KCl', 1, 2.2, pressure = out['pressure'])
na2s_cond_p, na2s_cond_t = vj.condensation_t('Na2S', 1, 2.2, pressure = out['pressure'])
mns_cond_p, mns_cond_t = vj.condensation_t('MnS', 1, 2.2, pressure = out['pressure'])
zns_cond_p, zns_cond_t = vj.condensation_t('ZnS', 1, 2.2, pressure = out['pressure'])
cr_cond_p, cr_cond_t = vj.condensation_t('Cr', 1, 2.2, pressure = out['pressure'])

# %%
pressure_bobcat,temp_bobcat = np.loadtxt(jdi.os.path.join(
                            sonora_profile_db,f"t{teff}g{grav}nc_m0.0.cmp.gz"),
                            usecols=[1,2],unpack=True, skiprows = 1)

pressure_elfowl,temp_elfowl = np.loadtxt(f"profilegrid_kz_1d9_qt_onfly_400_grav_1000_mh_+0.0_cto_1.0.dat",
                                usecols=[1,2],unpack=True, skiprows = 1)

plt.figure(figsize=(10,8))
plt.ylabel("Pressure [Bars]")
plt.xlabel('Temperature [K]')
plt.xlim(0,max(out['temperature'])+50)
plt.ylim(3e3,1e-3)

plt.semilogy(temp_bobcat,pressure_bobcat,color="k",linestyle="--",linewidth=3,label="Sonora Bobcat")
plt.semilogy(temp_elfowl,pressure_elfowl,color='r',linewidth=3, label="Sonora Elf Owl")
plt.semilogy(out['temperature'],out['pressure'],label="Our Cloudy Run")

plt.plot(kcl_cond_t,kcl_cond_p, color ='gray', label = 'KCl Condensation Curve')
plt.plot(na2s_cond_t,na2s_cond_p, color ='orange', label = 'Na2S Condensation Curve')
plt.plot(mns_cond_t,mns_cond_p, color ='purple', label = 'MnS Condensation Curve')
plt.plot(zns_cond_t,zns_cond_p, color ='green', label = 'ZnS Condensation Curve')
plt.plot(cr_cond_t,cr_cond_p, color ='blue', label = 'Cr Condensation Curve')

plt.legend()
plt.tight_layout()
plt.show()

# %%
# once again we can do a quick sanity check to make sure a cloud is present
show(jpi.mixing_ratio(out['spectrum_output']['full_output'], limit=14, height=600, width=600))

# %%
# once again we can do a quick sanity check to make sure a cloud is present
show(cldplt.all_optics_1d(out['virga_output'], wave_range=[1,2]))

# %% [markdown]
# ## Cloudy vs Clear Spectra

# %%
opa_mon = jdi.opannection(wave_range=[0.3,15])

hi_res = jdi.inputs(calculation="browndwarf") # start a calculation
teff= 400 # Effective Temperature of your Brown Dwarf in K
grav = 1000 # Gravity of your brown dwarf in m/s/s
hi_res.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity

hi_res.atmosphere(df=out['ptchem_df'])

# now let's add the cloud optical properties
hi_res.clouds(df=out['cld_df'])

df_spec = hi_res.spectrum(opa_mon, calculation='thermal',full_output=True)

wno, fp = df_spec['wavenumber'], df_spec['thermal'] #erg/cm2/s/cm
wno,fp = jdi.mean_regrid(wno,fp, R =200)

# %%
jpi.show(jpi.spectrum(wno,fp,x_axis_type='log',y_axis_type='log'))

# %% [markdown]
# This next spectrum is from the Sonora Elf Owl grid of models which are cloud-free to use as reference. To download the rest of the Sonora Elf Owl grid you can find them separated by [L](https://zenodo.org/records/10385987), [T](https://zenodo.org/records/10385821), and [Y](https://zenodo.org/records/10381250) dwarfs.

# %%
ds_elfowl = xr.load_dataset("spectra_logzz_9.0_teff_400.0_grav_1000.0_mh_0.0_co_1.0.nc")

# %% [markdown]
# Now let's regrid the spectra to R=200 to make sure the two spectra are on the same grid

# %%
wno_elfowl, fp_elfowl = jdi.mean_regrid(1e4/ds_elfowl['wavelength'].values,ds_elfowl['flux'].values,R=200)

# %%
fig = plt.figure(figsize=(12,6))
plt.loglog(1e4/wno_elfowl,fp_elfowl, 'k', label = 'Sonora Elf Owl')
plt.loglog(1e4/wno,fp, label = 'Our Cloudy Run')
plt.xlabel('Wavelength [micron]')
plt.ylabel('F$_\\nu$ [erg/cm$^2$/s/Hz]')
plt.xlim(0.5,12)
plt.legend()
plt.show()

# %% [markdown]
# For these cloud species you can see their features in the Y and J bands for example in comparison to the cloud-free Sonora Elf Owl model
