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
# # Querying Opacities
#
# ``PICASO`` currently comes with single opacity database that includes both the continuum and molecular opacity. The opacity file that comes included with has a 0.3-14 $\mu$m opacity grid sampled at R=15000. This is why we consistently bin our spectra to R~100. Contact us with opacity needs.
#
# We chose those use `sqlite3` for our database because of it's 1) user-friendliness, 2) speed, 3) scalability, 4) compatibility with parallel programming. In 2019, we tried out various other methods as well-- `json`, `hdf5`, `ascii`, `sqlalchemy`-- but `sqlite3` was truly better for this specific problem. Having revisited this problem recently `hdf5` has now outpaced `sqlite3` in terms of storage and performance. Therefore in future PICASO v5 we will move away from `sqlite3` and toward `hdf5` which has a much larger user base now. 
#
# In this tutorial you wil learn how to quickly grab any opacity data

# %%
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import picaso.opacity_factory as opa
#plotting
import picaso.justplotit as jpi
import picaso.justdoit as jdi
jpi.output_notebook()

# %% [markdown]
# ## General Sqlite3 and how our database is structured
#
# - 3 Tables: `header`, `continuum`, and `molecular`
# - header: contains units and the wavenumber grid
# - continuum: contains a grid continuum opacity and is temperature dependent
# - molecular: contains all molecular opacity and is pressure-temperature dependent.

# %% [markdown]
# ## How to Query the Database

# %%
opanxn = jdi.opannection()
#this is where your opacity file should be located if you've set your environments correctly
db_filename = opanxn.db_filename

# %% [markdown]
# ### What version db are you using ?
#
# Note this only works with PICASO Opacities v3 on Zenodo and above

# %%
opa.get_metadata_item(db_filename, 'version'),opa.get_metadata_item(db_filename, 'zenodo')

# %% [markdown]
# ### What Species, Pressures and Temperatures are Available

# %%
molecules, pt_pairs = opa.molecular_avail(db_filename)

# %%
print(molecules)

# %%
pt_pairs[0:10] #full list of all the pt pairs. the first value is just an index for bookkeeping

# %%
species_to_get = ['H2O']
temperature = list(np.arange(400,1000,10)) #this will find the nearest point for you in the db
pressure = [1]*len(temperature) #in bars

#NOTE: temperature and pressure lists are defined as pairs. The list above is
#grabbing temperatures from 400-1000 all at 1 bar.

data  = opa.get_molecular(db_filename, species_to_get, temperature,pressure)

# %%
data.keys()

# %%
#note these temperatures might be different from your input
#if there isn't an exact point matching your input
data['H2O'].keys()

# %%
data['H2O'][400.0].keys() #in this case we had the exact point (1.0 bar)

# %%
Spectral11  = jpi.pals.Spectral11
f = jpi.figure(y_axis_type='log',height=500,y_range=[1e-29,1e-17],
          y_axis_label='Cross Section (cm2/species)', x_axis_label='Micron')
for T , C in zip(data['H2O'].keys(), Spectral11):
    x,y = jdi.mean_regrid(data['wavenumber'],data['H2O'][T][1.0], R=200)
    f.line(1e4/x,y, color=C,legend_label=str(T))
jpi.show(f)


# %% [markdown]
# ### Get Continuum Opacity
#
# Very similar to molecular opacity, except not pressure dependent

# %%
molecules, temperatures = opa.continuum_avail(db_filename)

# %%
print(molecules)

# %% [markdown]
# Note: The continuum is perhaps the most "hard-coded" aspect of this code. If you want to add new continuum molecules you must also update this part of `picaso`:
#
# https://github.com/natashabatalha/picaso/blob/8ae68dfb0bd7c876f5f4fe7ee1d2ac00366d15ef/picaso/atmsetup.py#L313

# %%
data  = opa.get_continuum(db_filename, ['H2He'], list(temperature))

# %%
data.keys()

# %%
data['H2He'].keys()

# %%

f = jpi.figure(y_axis_type='log',height=300,y_range=[1e-15,1e-5],
          y_axis_label='Cross Section (cm-1 amagat-2)', x_axis_label='Micron')
for T , C in zip(list(data['H2He'].keys())[::2], Spectral11):
    x,y = jdi.mean_regrid(data['wavenumber'],data['H2He'][T], R=200)
    f.line(1e4/x,y, color=C,legend_label=str(T))

jpi.show(f)

# %%
