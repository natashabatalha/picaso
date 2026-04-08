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
# # Model Storage: Preservation and Reuse
#
# Storing data is often tricky. You may be wondering what output to save, what output to throw away. You may also have questions surround what formats to use to store your data.
#
# PICASO's recommended model output is `xarray`. There are several `xarray` tutorials throughout the picaso documentation. [The most thorough is in the data uniformity tutorial](https://natashabatalha.github.io/picaso/notebooks/codehelp/data_uniformity_tutorial.html).
#
# Here you will learn how to use a nifty picaso function that will store your the **minimum data needed to reproduce your 1D spectrum**. The one caveat is that we will not store opacity data. However, you can always get your reference opacity data using the [auto citation tools](https://natashabatalha.github.io/picaso/notebooks/References.html).
#
# For this tutorial we assume you already know the basics of running 1D models. Note that we have still not enabled this functionality for 3D and 4D runs. Please contact the developers if you are interested in this functionality.

# %%
import picaso.justdoit as jdi
import picaso.justplotit as jpi
u = jdi.u #astropy units

# %% [markdown]
# ## Storing your run with xarray
#
# Here we show a simple example of how to use the `output_xarray` function for preservation

# %%
opa = jdi.opannection(wave_range=[0.3,1])
pl = jdi.inputs()#calculation='brown')
pl.gravity(radius=1, radius_unit=u.Unit('R_jup'),
           mass=1, mass_unit=u.Unit('M_jup'))
pl.atmosphere(filename=jdi.jupiter_pt(), sep='\s+')
pl.phase_angle(0)
pl.clouds(filename=jdi.jupiter_cld(), sep='\s+')
pl.star(opa, 5000,0,4, radius=1, radius_unit=u.Unit("R_sun"), semi_major=1, semi_major_unit=1*u.AU)
#MUST USE full output=True for this functionality
df= pl.spectrum(opa, calculation='reflected', full_output=True)


# %%
preserve = jdi.output_xarray(df,pl,savefile='/data/picaso_dbs/model.nc')

# %%
preserve

# %% [markdown]
# ## Reusing your run with xarray
#
# We often revisit models time and time again. Maybe you want to change your wavelength range, or observing geometry (e.g. transit vs emission).
#
# Here we show a simple example of how to use the `input_xarray` function for reuse. In this simple example, we will take our previous input but instead of computing reflected light, add thermal emission and transmission

# %%
opa = jdi.opannection(wave_range=[0.3,14])
ds = jdi.xr.load_dataset('/data/picaso_dbs/model.nc')
reuse = jdi.input_xarray(ds, opa)
new_model = reuse.spectrum(opa, calculation='reflected+thermal', full_output=True)

# %%
new_output= jdi.output_xarray(new_model, reuse)

# %%
new_output

# %% [markdown]
# ## Adding meta data
#
# We have a tutorial that shows what some recommended meta data might be. `output_xarray` has a field `add_output` that allows you to add extra arguments to the `xarray.Dataset.attrs`. However, it has to be in a very specific format.
#
# You can see the basic format by running this function:
#
#

# %%
jdi.standard_metadata()

# %%
opa = jdi.opannection(wave_range=[0.3,1])
pl = jdi.inputs()#calculation='brown')
pl.gravity(radius=1, radius_unit=u.Unit('R_jup'),
           mass=1, mass_unit=u.Unit('M_jup'))
mh = 0
cto = 0.55 #solar value

pl.atmosphere(filename=jdi.jupiter_pt(), sep='\s+')
pl.chemeq_visscher_2121(cto, mh)
pl.phase_angle(0)
pl.clouds(filename=jdi.jupiter_cld(), sep='\s+')
pl.star(opa, 5000,0,4, radius=1, radius_unit=u.Unit("R_sun"), semi_major=1, semi_major_unit=1*u.AU)
#MUST USE full output=True for this functionality
df= pl.spectrum(opa, calculation='reflected', full_output=True)

preserve = jdi.output_xarray(df,pl,add_output={
    'author':"Awesome Scientist",
    'contact' : "awesomescientist@universe.com",
    'code' : "picaso",
    'planet_params':{'mh':mh, 'co':cto},
    'cloud_params':{'fsed':3}
})


# %%
preserve

# %%
