# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # HDF5 Opacity Database Suport
#
# ``PICASO`` is moving away from SQLite and toward HDF5. Currently, this notebook tutorial shows users how to: 
# - transfer their existing SQLite database to HDF5 (the support for doing this includes options to decrease the float precision of the data to drastically reduce memory by ~factor of 2). 
# - perform queries on the new HDF5 formatted data 

# %%
from picaso import justplotit as jpi
from picaso import justdoit as jdi
import numpy as np
import picaso.opacity_factory as opa
#plotting
jpi.output_notebook()

# %% [markdown]
# ## Transfer SQLite data to HDF5 format
#
# Some new parameters to consider for the function below: 
#
# - `compression` and `shuffle`: When lzf + shuffle compression is used, this decreases resampled opacity file sizes by a factor of 10.
# - `storage_format`: HDF5 files can store opacities in two different formats: log10_uint16 and log10_float32. The log10_uint16 approach saves log10 opacities using unsigned 16 bit integers which will degrade accuracy by  < 0.04% in core tests (reflected + transmission + thermal). We do not guarantee that as the ceiling. We encourage users to test for their case of interest. 

# %%
#lets convert your default opacity file (note this will not delete or overwrite your default)
opanxn = jdi.opannection()
db_filename = opanxn.db_filename
new_hdf5_filename = jdi.os.path.join(jdi.os.getcwd(),'test.hdf5')#'/data/picaso_dbs'
opa.convert_sqlite_to_hdf5(
    input_db=db_filename,#lets convert our current default to HDF5
    output_hdf5=new_hdf5_filename,
    compression='lzf',
    shuffle=True,
    storage_format="log10_uint16",
    chunks=(1, 4096),
    molecular_log10_floor=1e-50,
    continuum_log10_floor=1e-100,
    verbose=True,
)

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
molecules, pt_pairs = opa.molecular_avail(new)

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
