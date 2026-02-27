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
# # Model Formatting Overview
#
# This notebook was created to enable common formatting for the Early Release Science modeling initiatives. We will review:
#
# 1. Variable terminology
# 2. File naming schemes
# 3. Data formatting
# 4. Physical unit archiving
#
# This is namely for the booking of the following model types:
#
# 1. 1D-3D climate (spatial- and/or altitude- dependent climate)
# 2. 1D-3D chemistry (spatial- and/or altitude- dependent atmospheric composition)
# 3. 1D-3D cloud (spatial- and/or altitude- dependent single scattering, asymmetries, cloud optical depth)
# 4. Spectroscopy (flux, transit depth as a function of wavelength)
#
# However, it can be applied to other modeling products (e.g. mixing profiles).
#
# ## Variable Terminology
#
# All file names and meta data should conform to the following variable names. Note, that these will not apply to all models. This is just an initial list. Please shoot me a DM, or slack for additional parameters to add (natasha.e.batalha@nasa.gov, or ERS slack channel)
#
# ### Planet parameters (`planet_params`)
#
# 1. `rp`: planet radius
# 2. `mp`: planet mass
# 3. `tint`: object internal temperature
# 4. `heat_redis`: heat redistribution (only relevant for irradiated objects)
# 5. `p_reference`: reference pressure radius
# 6. `pteff`: planetary effective temperature
# 7. `mh` : metallicity
# 8. `cto` : carbon to oxygen ratio
# 9. `logkzz` : log of the kzz eddy diffusion
#
# ### Stellar parameters (`stellar_params`)
# 1. `logg` : gravity
# 2. `feh` : stellar metallicity
# 3. `steff` : stellar effective temperature
# 4. `rs` : stellar radius
# 5. `ms` : stellar mass
#
# ### Orbital parameters (`orbit_params`)
# 1. `sma` : semi-major axis
#
# ### Cloud parameters (`cld_params`)
# 1. `opd` : extinction optical depth
# 2. `ssa` : single scattering albedo
# 3. `asy` : asymmetry parameter
# 4. `fsed` : cloud sedimentation efficiency parameter
#
# ### Model Gridding ( `coords`)
# 1. `pressure`: pressure grid
# 2. `wavelength`: wavelength grid
# 3. `wno`: wavenumber grid
# 4. `lat`: latitude grid
# 5. `lon`: longitude grid
#
# ### Model Output ( `data_vars`)
#
# There are SO many different model outputs users will want to pass. For the purposes of ERS, we will focus on these, but feel free to send recommendations for more. Note that in your xarray file there will not be a separation between these categories. They will all be lumped into data_vars. However their coordinate systems will be different! The beauty of xarray!
#
# #### Spectrum
#
# 1. `transit_depth` : transmission spectrum reported as unitless depth (rp/rs)^2. This way it can be directly compared to data.
# 2. `fpfs_emission` : relative emission spectrum (unitless)
# 3. `fpfs_reflection` relative reflected light spectrum (unitless)
# 4. `flux_emission` : thermal emission in raw flux units
# 5. `albedo` : albedo spectrum
#
# #### Chemistry
#
# 6. case sensitive molecule names (e.g. Na, H2O, TiO) for each chemical abundance (either 1d or 3d). This means your mixing ratio profile for TiO would not be TIO. Or, for example the chemical profile for sodium would be "Na" not NA
#
# #### Climate
#
# 7. `temperature`: computed temperature profile either 1d or 3d
#
# #### Cloud
#
# 8. `opd` : extinction optical depth
# 9. `ssa` : single scattering albedo
# 10. `asy` : asymmetry parameter
#
# #### Retrieval output
#
# 11. Coming soon.
#

# %% [markdown]
# ## Specifying units
#
# We should be able to convert all units to `astropy.units`. For unitless parameters (e.g. single scattering albedo, optical depth) unitless designation should be provided. See example:
#

# %%
import astropy.units as u

# %%
#examples of valid units

u.Unit('cm') #Valid
#u.Unit('CM') #NOT valid
u.Unit("R_jup")#Valid
u.Unit("R_jupiter")#Valid
#u.Unit("R_Jupiter")#NOT Valid

# %%
unit = 'cm'
#doing it this away enables easy conversions. for example:
(1*u.Unit('R_jup')).to('cm')

# %% [markdown]
# ## Data Types: Using `xarray`
#
# [xarray: N-D labeled arrays and datasets in Python](https://docs.xarray.dev/en/stable/): From their website: "array introduces labels in the form of dimensions, coordinates and attributes on top of raw NumPy-like arrays, which allows for a more intuitive, more concise, and less error-prone developer experience. The package includes a large and growing library of domain-agnostic functions for advanced analytics and visualization with these data structures."
#
# Xarray is your friend and will make it very easy for other folks to use your data. Let's build some simple examples.
#

# %% [markdown]
# ### `attrs` in `xarray`
#
# This is how we make sure we credit the appropriate people for their work, and understand the meta data surrounding the model runs.
#
# ### Required `attrs`
#
# 1. author: Author or author list
# 2. contact email: point of contact
# 3. code used: (can be a dictionary. e.g. {'chemistry':'vulcan', 'cloud':'virga', 'spectra':'chimera'}
#
# ### Optional  `attrs`
# 1. doi (str): made sure to include if you want your work referenced!
# 2. planet_params (json dict) : with dict keys defined in section 1
# 3. stellar_params (json dict): with dict keys defined in section 1
# 4. orbit_params (json dict) : with dict keys defined in section 1
# 5. cld_params (json dict): with dict keys defined in section 1
# 6. model_notes (str) : any additional modeling notes that you want the user to be aware of

# %% [markdown]
# ## Easy Example: 1D data: e.g. P-T profiles, chemistry
#
# Here we will show an example with `pressure` as the dependent variable. Spectra, which are on a wavelength or wavenumber grid, can also be stored similarly.

# %%
import numpy as np
import xarray as xr
from astropy.utils.misc import JsonCustomEncoder
import json #we will use this to dump model parameters into an attribute

# %%
#fake pressure temperature profile at 1000 K
pressure = np.logspace(-6,2,50)
temperature = np.logspace(-6,2,50)*0 + 1000

# %% [markdown]
# Practice convert to `xarray`. In this case we are storing `temperature` data, labeled with `unit` "Kelvin" that is on a grid of `pressure` with units of "bar"

# %%
# put data into a dataset where each
ds = xr.Dataset(
    data_vars=dict(
        temperature=(["pressure"], temperature,{'units': 'Kelvin'})#, required
    ),
    coords=dict(
        pressure=(["pressure"], pressure,{'units': 'bar'})#required*
    ),
    attrs=dict(author="NE Batalha",#required
               contact="natasha.e.batalha@nasa.gov",#required
               code="numpy", #required, in this case I used numpy to make my fake model.
               doi="add your paper here",#optional if there is a citation to reference
               planet_params=json.dumps({'rp':1*u.Unit('R_jup'), 'mp':1*u.Unit('M_jup')},
                                        cls=JsonCustomEncoder), #optional in accordance with model runs
               stellar_params=json.dumps({'rs':1*u.Unit('R_sun'), 'ms':1*u.Unit('M_sun')},
                                         cls=JsonCustomEncoder) #optional in accordance with model runs
              ),
)

# %%
#printing is easy
ds

# %%
#plotting is easy
ds['temperature'].plot(y='pressure')

# %% [markdown]
# ### Looping to add many variables to `data_vars`
#
# If you have a big table of many parameters, e.g. molecules you can pre-make `data_vars` as a dictionary then add it like this:

# %%
fake_chemistry = {i:np.random.randn(len(pressure)) for i in ['H2O','CH4','CO2','CO',"H2S"]}

data_vars=dict(
        temperature=(["pressure"], temperature,{'units': 'Kelvin'})#, required
    )
for i in fake_chemistry.keys():
    data_vars[i] = (["pressure"], fake_chemistry[i],{'units': 'v/v'})#volume mixing ratio units

# put data into a dataset where each
ds = xr.Dataset(
    data_vars=data_vars,
    coords=dict(
        pressure=(["pressure"], pressure,{'units': 'bar'})#required*
    ),
    attrs=dict(author="NE Batalha",#required
               contact="natasha.e.batalha@nasa.gov",#required
               code=json.dumps({'climate':"numpy","chemistry":'numpy'}), #required
               doi="",#optional if there is a citation to reference
               planet_params=json.dumps({'rp':1*u.Unit('R_jup'), 'mp':1*u.Unit('M_jup')},
                                        cls=JsonCustomEncoder), #optional in accordance with model runs
               stellar_params=json.dumps({'rs':1*u.Unit('R_sun'), 'ms':1*u.Unit('M_sun')},
                                         cls=JsonCustomEncoder) #optional in accordance with model runs #optional in accordance with model runs
              ),
)

# %% [markdown]
# ## 2D data: e.g. cloud profiles with pressure vs wavenumber

# %%
wno = np.linspace(10,10000,400)#fake wavenumber array
opd = np.zeros((len(pressure), len(wno))) + 1
ssa = np.zeros((len(pressure), len(wno))) + 0.9
asy = np.zeros((len(pressure), len(wno))) + 0.8

# put data into a dataset where each
ds = xr.Dataset(
    #now data is a function of two dimensions
    data_vars=dict(opd=(["pressure","wno"], opd,{'units': 'unitless per layer'}),
                   ssa=(["pressure","wno"], ssa,{'units': 'unitless'}),
                   asy=(["pressure","wno"], asy,{'units': 'unitless'}),
                  ),
    coords=dict(
        pressure=(["pressure"], pressure,{'units': 'bar'}),#required
        wno=(["wno"], wno,{'units': 'cm**(-1)'})#required
    ),
    attrs=dict(author="NE Batalha",#required
               contact="natasha.e.batalha@nasa.gov",#required
               code='numpy', #required
               doi="",#optional if there is a citation to reference
               planet_params=json.dumps({'rp':1*u.Unit('R_jup'), 'mp':1*u.Unit('M_jup')},
                                        cls=JsonCustomEncoder), #optional in accordance with model runs
               stellar_params=json.dumps({'rs':1*u.Unit('R_sun'), 'ms':1*u.Unit('M_sun')},
                                         cls=JsonCustomEncoder) #optional in accordance with model runs #optional in accordance with model runs
              ),
)

# %% [markdown]
# ## 3D data: e.g. GCM pressure grid
#
# `xarray` is incredibly useful for GCM work. [Here is an example of how picaso uses xarray and xesfm to do regridding and 3d calculations](https://natashabatalha.github.io/picaso/notebooks/9a_3DInputsWithPICASOandXarray.html).
#
# Here we'll just show how to create a 3D gridded dataset.

# %%
lat = np.linspace(-90,85,40)#fake latitude array
lon = np.linspace(-180,175,100)#fake latitude array
temperature_3d = np.random.randn(len(lat),len(lon),len(pressure))

# put data into a dataset where each
ds = xr.Dataset(
    #now data is a function of two dimensions
    data_vars=dict(temperature=(["lat","lon","pressure"],temperature_3d ,{'units': 'Kelvin'})
                  ),
    coords=dict(
        pressure=(["pressure"], pressure,{'units': 'bar'}),#required
        lat=(["lat"], lat,{'units': 'degree'}),#required
        lon=(["lon"], lon,{'units': 'degree'}),#required
    ),
    attrs=dict(author="NE Batalha",#required
               contact="natasha.e.batalha@nasa.gov",#required
               code='numpy', #required
               doi="",#optional if there is a citation to reference
               planet_params=json.dumps({'rp':1*u.Unit('R_jup'), 'mp':1*u.Unit('M_jup')},
                                        cls=JsonCustomEncoder), #optional in accordance with model runs
               stellar_params=json.dumps({'rs':1*u.Unit('R_sun'), 'ms':1*u.Unit('M_sun')},
                                         cls=JsonCustomEncoder) #optional in accordance with model runs #optional in accordance with model runs
              ),
)

# %%
#easy plotting
ds['temperature'].isel(pressure=10).plot()

# %% [markdown]
# ## Storing `xarray` data
#
# ### Filenaming
#
# We usually rely on a long filename to give us information about the model. If we properly use `attrs` then filenaming does not matter. However, friendly filenames are always appreciated by people using your models. We suggest the following naming convention.
#
# Given independent variables (x,y,z): `tag_x{x}_y{y}_z{z}.nc`
#
# For example: `jupiter_mh1_teff1000_tint100.nc`
#
# ### Using `netcdf`
#
# "The recommended way to store xarray data structures is netCDF, which is a binary file format for self-described datasets that originated in the geosciences. Xarray is based on the netCDF data model, so netCDF files on disk directly correspond to Dataset objects (more accurately, a group in a netCDF file directly corresponds to a Dataset object. See Groups for more.)" - [Quoted from xarray website](https://docs.xarray.dev/en/stable/user-guide/io.html)

# %%
ds.to_netcdf("/data/picaso_dbs/fakeplanet_1000teq.nc")

# %% [markdown]
# ### Using `pickle`

# %%
import pickle as pk
pk.dump(ds, open("/data/picaso_dbs/fakeplanet_1000teq.pk",'wb'))

# %% [markdown]
# ## Reading/interpreting an `xarray` file
#
# First, make sure you have installed [netCDF4](https://github.com/Unidata/netcdf4-python) and [h5netcdf](https://github.com/h5netcdf/h5netcdf) :
#
# ```
# pip install netCDF4
# pip install h5netcdf
# ```
# or if you prefer conda
# ```
# conda install -c conda-forge netCDF4
# ```

# %%
ds_sm = xr.open_dataset("profile_eq_planet_300_grav_4.5_mh_+2.0_CO_2.0_sm_0.0486_v_0.5_.nc")

# %% [markdown]
# Look at all the information we can glean from this

# %%
ds_sm #39 data variables

# %%
ds_sm['wavelength']#data operates very similarly to pandas, note we can see the unit of the coordinate system

# %%
ds_sm['wavelength'].values #same as pandas!

# %%
ds_sm['temperature']

# %% [markdown]
# How to get attributes from string dictionary

# %%
json.loads(ds_sm.attrs['planet_params'])

# %%
json.loads(ds_sm.attrs['planet_params'])['rp'] #radius used


# %% [markdown]
# ## Checking your data is in compliance
#
# TLDR: this function will check that your data can be properly interpreted

# %%
def data_check(usr_xa):
    """This function will check that all the requirements have been met"""

    #step 1: check that required attributes are present
    assert 'author' in usr_xa.attrs ,'No author information in attrs'
    assert 'contact' in usr_xa.attrs ,'No contact information in attrs'
    assert 'code' in usr_xa.attrs , 'Code used was not specified in attrs'

    #step 2: check that all coordinates have units
    try:
        for i in usr_xa.coords.keys(): test= usr_xa[i].units
    except AttributeError:
        print(f'Missing unit for {i} coords')

    #step 2: check that all coordinates have units
    try:
        for i in usr_xa.data_vars.keys(): test=usr_xa[i].units
    except AttributeError:
        print(f'Missing unit for {i} data_var')

    #step 3: check that some attrs is a proper dictionary
    try :
        for i in usr_xa.attrs:
            #these need to be dictionaries to be interpretable
            if i in ['planet_params','stellar_params','cld_params','orbit_params']:
                json.loads(usr_xa.attrs[i])
    except ValueError:
        print(f"Was not able to read attr for {i}. This means that you did not properly define a dictionary with json and a dict."," For example: json.dumps({'mp':1,'rp':1})")

    #step 4: hurray if you have made it to here this is great
    #last thing is the least important -- to make sure that we agree on terminology
    for i in usr_xa.attrs:
        if i == 'planet_params':
            for model_key in json.loads(usr_xa.attrs[i]).keys():
                assert model_key in ['rp', 'mp', 'tint', 'heat_redis', 'p_reference','rainout','p_quench',
                'pteff', 'mh' , 'cto' , 'logkzz'], f'Could not find {model_key} in listed planet_params attr. This might be because we havent added it yet! Check your terms and contact us if this is the case'

        elif  i == 'stellar_params':
            for model_key in json.loads(usr_xa.attrs[i]).keys():
                assert model_key in ['logg', 'feh', 'steff', 'rs', 'ms',
                ], f'Could not find {model_key} in listed stellar_params attr. This might be because we havent added it yet! Check your terms and contact us if this is the case'

        elif  i == 'orbit_params':
            for model_key in json.loads(usr_xa.attrs[i]).keys():
                assert model_key in ['sma',
                ], f'Could not find {model_key} in listed orbit_params attr. This might be because we havent added it yet! Check your terms and contact us if this is the case'

        elif  i == 'cld_params':
            for model_key in json.loads(usr_xa.attrs[i]).keys():
                assert model_key  in ['opd','ssa','asy','fsed','p_cloud','haze_effec',
                ], f'Could not find {model_key} in listed cld_params attr. This might be because we havent added it yet! Check your terms and contact us if this is the case'

    print('SUCCESS!')



# %%
ds_sm = xr.open_dataset("profile_eq_planet_300_grav_4.5_mh_+2.0_CO_2.0_sm_0.0486_v_0.5_.nc")
data_check(ds_sm)

# %%
