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
# # Terrestrial Planets: Adding a Surface
#
# Three major changes differences you need to make note of for terrestrial planets in both reflected, emission and transmission geometry:
#
# 1. Adding a surface reflectivity
# 2. Ensuring the code knows that your lower boundary condition is a hard surface
# 3. Ensuring that the reference pressure is lower than the lower pressure of your surface. `PICASO` default is 1 bar, which is often times too high.

# %%
import pandas as pd
import numpy as np
#picaso
from picaso import justdoit as jdi
from picaso import justplotit as jpi
#plotting
jpi.output_notebook()

# %% [markdown]
# ### Connect to Opacity Database

# %%
opacity = jdi.opannection(wave_range=[0.3,1])

# %% [markdown]
# ### Load blank slate

# %%
sum_planet = jdi.inputs()

# %% [markdown]
# ### Set Planet & Star Properties

# %%
#phase angle
sum_planet.phase_angle(0) #radians

#define gravity
sum_planet.gravity(gravity=9.8, gravity_unit=jdi.u.Unit('m/(s**2)')) #any astropy units available

#define star
sum_planet.star(opacity,5000,0,4.0) #pysynphot database, temp, metallicity, logg

# %% [markdown]
# ## Add Surface Reflectivity
#
# For reflected light it is important to specify the surface albedo in case there are wavelength bands that are optically thin from molecular absorption. In those cases you will see the surface reflection in your spectra.
#
# For emission, it is important the surface reflectivity routine also tells the code that you have a "hard surface" boundary condition (opposed to a high pressure surface). Therefore if you are running reflected light, or thermal emission you will have to run the `surface_reflect` routine.
#
# **Therefore, it is critical that you run this routine for all terrestrial cases (reflected light and thermal emission**

# %%
sum_planet.atmosphere(df=pd.DataFrame({'pressure':np.logspace(-6,0,40),
                         'temperature':np.linspace(100,270,40), #very fake temperature profile with a 270 K surface
                         'H2O':np.zeros(40)+0.01,
                         'CO2':np.zeros(40)+1-0.01}))

sum_planet.surface_reflect(0.3,opacity.wno) #this inputs the surface reflectivity and tells the code
                                            #there is a hard surface


#can also input a wavelength dependent quantity here
#fake_surface= np.linspace(0.01,0.9, opacity.nwno)
#sum_planet.surface_reflect(fake_surface,opacity.wno)

# %% [markdown]
# Note you can turn off these printout messages with `verbose=False` in the `atmosphere` function

# %% [markdown]
# ## Create 1D Albedo Spectrum

# %%
df = sum_planet.spectrum(opacity)

# %%
wno, alb, fpfs = df['wavenumber'],df['albedo'],df['fpfs_reflected']
wno, alb = jpi.mean_regrid(wno, alb, R=150)

# %%
jpi.show(jpi.spectrum([wno], [alb], plot_width=500))

# %% [markdown]
# What about contrast units?

# %%
fpfs

# %% [markdown]
# ### Get Contrast Units
#
# In order to get contrast units we have to make sure to give the `gravity` and `star` functions semi major axis, mass and radius.

# %%
sum_planet.star(opacity,5000,0,4.0,semi_major=1, semi_major_unit=jdi.u.Unit('au'))
sum_planet.gravity(radius=1.,radius_unit=jdi.u.Unit('R_earth'),
             mass=1,mass_unit=jdi.u.Unit('M_earth'))
df = sum_planet.spectrum(opacity)
wno, alb, fpfs = df['wavenumber'],df['albedo'],df['fpfs_reflected']
wno, fpfs = jdi.mean_regrid(wno, fpfs , R=150)


# %%
jpi.show(jpi.spectrum([wno], [fpfs*1e6], plot_width=500))
