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
# # Creating a Grid of Models to use with `GridFitter`

# %% [markdown]
# You should already be familiar with:
# - how to output create a climate model
# - how to post process a high resolution spectral model from that climate model
# - xarray file formats
# - It is **very important you already understand how to analyze and diagnose the valididy of climate models**. Here we will use a streamlined function to trigger the creation of many models. However, there is always a chance that some of the climate runs do not converge. Therefore, it is important to you go back and assess each of these models.
#
# The main goal of this notebook will be to create a mini mini grid of models that is on a function of :
#
# - Metallicity (mh)
# - Carbon to Oxygen Ratio (cto)
# - Heat redistribution (heat_redis)
# - Internal Heat (tint)
#
# Take a look at some papers which have created climate grids for recommendation on how to determine what values for each of these. Here in this notebook we will simply run two of each, just to show how the function works.
#
# - WASP-39 b (e.g. [Alderson et al 2023](https://ui.adsabs.harvard.edu/abs/2023Natur.614..664A/abstract))
# - WASP-17 b (e.g. [Grant et al 2023](https://ui.adsabs.harvard.edu/abs/2023ApJ...956L..32G/abstract))
#
#

# %% [markdown]
# ## Create function to run climates
#
# If you dont understand these inputs we encourage you to complete the basic climate tutorials first!

# %%
import os
from picaso import justdoit as jdi
from picaso import justplotit as jpi
import numpy as np


# %%
def run_climate(base_case_name, array_save_dir, mh, CtoO,
                t_int, star_temp, star_metal, star_radius,
                star_logg, semi_major, nlevel, rcb_guess,
                rfacv, opacity_ph, teq,
                planet_mass, planet_radius,
                resume=False):
    #check if done already and want to skip:
    #create file name
    print("Running climate")

    #let's create a nice naming scheme based on our parameters
    savefile=base_case_name+"_climate_tint"+str(t_int)+"_rfacv"+str(rfacv)+"_mh"+mh+"_cto"+CtoO+".nc"

    #let's make sure we dont repeat ourselves if a file already exists
    if (resume & os.path.exists(os.path.join(array_save_dir,'climate',savefile))):
        print('Skipping',savefile,', file already exists')
        cl_xarray = jdi.xr.load_dataset(os.path.join(array_save_dir,'climate',savefile))
        return savefile,cl_xarray
    else:

        #This is setting up the problem for a planet
        case_name = jdi.inputs(calculation="planet", climate = True)
        case_name.star(opacity_ph, temp=star_temp,metal=star_metal,logg=star_logg,
                        radius=star_radius, semi_major= semi_major ,
                        database='phoenix', radius_unit = jdi.u.Unit('R_sun'), semi_major_unit = jdi.u.AU)
        case_name.gravity(mass=planet_mass, mass_unit=jdi.u.Unit('M_jup'),
                      radius=planet_radius, radius_unit=jdi.u.Unit('R_jup'))
        case_name.effective_temp(t_int)

        #setup mechanism to create initial guesses
        pt = case_name.guillot_pt(teq, nlevel=nlevel, T_int = t_int, p_bottom=2, p_top=-6)
        temp_guess = pt['temperature'].values
        pressure = pt['pressure'].values

        case_name.inputs_climate(temp_guess=temp_guess, pressure=pressure,
                          rcb_guess=rcb_guess, rfacv=rfacv)

        #here we are setting with_spec to True just so we can use the output_xarray function
        out = case_name.climate(opacity_ph, save_all_profiles=True, with_spec=True,verbose=False)

        #make folder if needed
        print("Saving xarray: " + savefile)
        #in add output you can add any extra meta data fields you want to have in your xarray file
        climate_xarray = jdi.output_xarray(out, case_name, add_output={'author': 'Awesome Scientist',
                                                                      'planet_params':{
                                                                          'mh':mh,
                                                                          'cto':CtoO,
                                                                          'heat_redis':rfacv,
                                                                          'tint':t_int,
                                                                      }},
                          savefile=os.path.join(array_save_dir,"climate",savefile))

        return savefile,climate_xarray



# %% [markdown]
# ### An important note about `add_output` kwarg in `jdi.output_xarray`
#
# Later, you might want to use your xarray grid as input for picaso's `GridFitting` function. `GridFitting` is discussed in the next set of tutorials. Without going into depth, the main thing to note at thi stage is that `GridFitting` looks at your xarray attributes to understand what parameters your grid is computed for. For example, if `GridFitter` encoutered 8 different xarray files and the attributes included varying M/H and C/O then it would conclude your grid has been created as a function of those two parameters.
#
# SO, **if you are building a grid for use with `GridFitter` you must**:
# - Use picaso's naming scheme for variables (which you can see with picaso.analyze.possible_params see example below)
# - Use picaso's categories (for example planet metallicity `mh` would be stored under `planet_params` -- as is done in the example above -- stellar metallicity `feh` would be stored under `stellar_params`)
# - Make sure these are included in your spectra `xarray` files (as is done in this example)

# %% [markdown]
# ## Create Function to Post-Process High Res Spectra

# %%
def hi_res_spec_xarr(savefile, climate_xarray, array_save_dir, opacity_highres,add_xarray_output,resume=False):

    #make sure we have the same naming system for this file
    spec_savefile = savefile.replace('climate','spectra')

    #let's make sure we dont repeat ourselves if a file already exists
    if (resume & os.path.exists(os.path.join(array_save_dir,'spectra',spec_savefile))):
        print('Skipping',spec_savefile,', file already exists')
        return
    else:
        #we can directly input our climate xarray
        hi_res = jdi.input_xarray(climate_xarray,opacity_highres, calculation='planet')
        #at this point you could consider post processing clouds if you wish!
        #hi_res.clouds(... )
        df_spec = hi_res.spectrum(opacity_highres, calculation='thermal+transmission',
                                  full_output=True)
        preserve_hires = jdi.output_xarray(df_spec, hi_res,
                                  savefile=os.path.join(array_save_dir,"spectra",spec_savefile),
                                          add_output=add_xarray_output)
        print('Done!', spec_savefile)


# %% [markdown]
# ## Set Inputs and Trigger Loop!

# %%
correlated_k_dir = os.path.join(os.getenv('picaso_refdata'),'opacities', 'preweighted')
array_save_dir ='/data/test/tutorial'
mh_list = ['0.0', '1.0']
CtoO_list = ['0.46', '0.27']
heat_redis_list=[0.5]
t_int_list=[200, 300]

#continue where left off? Or restart?
resume=False

#other variables
base_case_name='wasp-39b' #tag name for filenames
R=100 #resolution power

#wav range for high res spectra
min_wav=3
max_wav=15

nlevel = 91 # number of atmospheric layers we want
rcb_guess = 85 #upper bound on first radiative convective boundary

#planet/star params
star_temp = 5400
star_metal = 0.01
star_radius = 0.9
star_logg = 4.45

wasp_mass=0.28
wasp_radius=1.27
teq=1120.55
semi_major=0.0486

#define what resampled opacities you want to use
opacity_hires = jdi.opannection(wave_range=[min_wav,max_wav])


#Trigger mega loop !
for mh in mh_list:
    for CtoO in CtoO_list:
        for rfacv in heat_redis_list:
            for t_int in t_int_list:
                #grab k table opacities
                print(mh, CtoO,rfacv,t_int)
                ph_db = os.path.join(
                    correlated_k_dir,f'sonora_2121grid_feh{mh}_co{CtoO}.hdf5')
                print(ph_db)
                opacity_ph = jdi.opannection(ck_db=ph_db,method='preweighted')
                #get climate xarray
                savefile, climate_xarray = run_climate(base_case_name, array_save_dir,
                                                       mh, CtoO, t_int, star_temp, star_metal,
                            star_radius, star_logg, semi_major, nlevel, rcb_guess, rfacv,
                                                       opacity_ph, teq=teq,
                            planet_mass=wasp_mass, planet_radius=wasp_radius, resume=resume)
                #get hi res xarray
                hi_res_spec_xarr(savefile,climate_xarray, array_save_dir,opacity_hires,
                                 add_xarray_output={'author': 'Awesome Scientist',
                                                              'planet_params':{
                                                                  'mh':mh,
                                                                  'cto':CtoO,
                                                                  'heat_redis':rfacv,
                                                                  'tint':t_int,
                                                              }},resume=resume)

# %% [markdown]
# ## Now you are ready for grid fitting (next tutorial)
#
# Now that you have a grid of xarrays you are ready for grid fitting using `picaso`'s grid fitting tools. The next set of tutorials will teach you how to use these skills.

# %%
import picaso.analyze as lyz

# %%
grid_name = 'mygrid'

fitting = lyz.GridFitter(grid_name, model_dir = os.path.join(array_save_dir,'spectra'),
                         to_fit='transit_depth')

# %% [markdown]
# All we have done is point GridFitter to the directory and it has detected a grid that is a function of our two metallicities, internatl temperatures, and C/Os. This was possible because we gave it the correct naming parameters.
#
#
# These are the possible parameters that `GridFitter` searches for. Make sure to follow the example above to add them to the xarray output fields.

# %%
lyz.possible_params

# %%
