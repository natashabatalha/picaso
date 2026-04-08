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
# # How to Fit Grid Models to Data
#
# In this notebook, we show how to use the `PICASO`-formatted grid models to interpret data. We will use the results of the [JWST Transiting Exoplanet Community Early Release Science Team's](https://arxiv.org/pdf/2208.11692.pdf) first look analysis of WASP-39 b.
#
# **Helpful knowledge before running this notebook:**
#
# - [How to use xarray files](https://natashabatalha.github.io/picaso/notebooks/codehelp/data_uniformity_tutorial.html)
# - [Basic PICASO knowledge of how to compute transit spectra](https://natashabatalha.github.io/picaso/notebooks/workshops/ESO2021/ESO_Tutorial.html)
#
# **Need to do before running this notebook**
#
# In order to use this notebook you will have to:
#
# - [Download and unpack the zenodo grid models](https://doi.org/10.5281/zenodo.7236759)
# - [Download the final planet spectrum](https://zenodo.org/record/6959427#.Y1M0U-zMLvU)

# %%
import numpy as np
import os

import picaso.justdoit as jdi
import picaso.justplotit as jpi
import picaso.analyze as lyz
jpi.output_notebook()

# %% [markdown]
# ## Define Paths To Data and Models
#
# You should have four folders in your `model_dir`:
#
# 1. `RCTE_cloud_free/`: 192 models
# 2. `RCTE_cloudy/`: 3840 models
# 3. `photochem_cloud_free/`: 116 models
# 4. `photochem_cloudy/`: 580 models
#

# %%
#should have sub folders similar to above
#agnostic to where it is, just make sure you point to the right data file
model_dir = "/data2/models/WASP-39B/xarray/"

#downloaded and unzipped from Zenodo
data_dir = '/data2/observations/WASP-39b/ZENODO/TRANSMISSION_SPECTRA_DATA/'
#for this tutorial let's grab the firely reduction
data_file = os.path.join(data_dir,"FIREFLY_REDUCTION.txt")

wlgrid_center,rprs_data2,wlgrid_width, e_rprs2 = np.loadtxt(data_file,usecols=[0,1,2,3],unpack=True,skiprows=1)

#for now, we are only going to fit 3-5 um
wh =  np.where(wlgrid_center < 3.0)
wlgrid_center = np.delete(wlgrid_center,wh[0])
wlgrid_width = np.delete(wlgrid_width,wh[0])
rprs_data2 = np.delete(rprs_data2,wh[0])
e_rprs2 = np.delete(e_rprs2,wh[0])
reduction_name = "Firefly"

# %%
f=jpi.plot_errorbar(wlgrid_center, rprs_data2,e_rprs2, plot_type='matplotlib',
                   plot_kwargs={'ylabel':r'(R$_p$/R$_*$)$^2$'})#plot_type='bokeh' also available
#jpi.show(f) #if using bokeh (note if using bokeh need those key words (e.g. y_axis_label instead of ylabel))

# %% [markdown]
# ## Add Available Grids to Test

# %% [markdown]
# First step will be to load your first grid into the `GridFitter` class. You can do this easily by supplying the function a directory location, and grid name (`grid_name`).
#
# The only purpose of `grid_name` is in case you add more grids to your `GridFitter` function, it will be easy to keep track of what parameters go with what grid.

# %%
grid_name = "picaso_cld_free"
location = os.path.join(model_dir,"RCTE_cloud_free")
fitter = lyz.GridFitter(grid_name,location, verbose=True)

# %% [markdown]
# This shows you what parameters the grid was created over

# %%
fitter.grid_params['picaso_cld_free']['planet_params'].keys()

# %%
location = os.path.join(model_dir,"RCTE_cloudy")
fitter.add_grid('picaso_cldy', location)#

# %% [markdown]
# ### Explore the parameters of the grid

# %% [markdown]
# You can see what grids you have loaded

# %%
fitter.grids #what grids exist

# %% [markdown]
# You can also see what the top level information about your grid is

# %%
fitter.overview['picaso_cld_free']#top level info from the attrs

# %% [markdown]
# The full list of planet parameters can also be cross referenced against the full list of file names so you can easily plot of different models.

# %%
print(fitter.grid_params['picaso_cld_free']['planet_params']['tint'][0],
#this full list can be cross referened against the file list
fitter.list_of_files['picaso_cld_free'][0])
#in this case we can verify against the filename

# %% [markdown]
# ## Add Datasets to Explore
#
# Though the models are interesting, what we are really after is which is most representative of the data. So now let's add some datasets to explore.

# %%
fitter.add_data('firefly',wlgrid_center, wlgrid_width, rprs_data2, e_rprs2)

# %% [markdown]
# ## Compute $\chi_{red}^2$/N and Retrieve Single Best Fit
#
# In this analysis we used the reduced chi sq per data point as a metric to fit the grid. This fitter function will go through your whole grid and compute cross reference the chi sq compared to your data.

# %%
fitter.fit_grid('picaso_cld_free','firefly')
fitter.fit_grid('picaso_cldy','firefly')

# %% [markdown]
# Now that we have accumulated results let's turn this into a dictionary to easily see what we've done

# %%
out = fitter.as_dict()#allows you to easily grab data
out.keys()

# %% [markdown]
# We are most interested in the models with the best reduced chi sq. We can use our ranked order to get the models that best fit the data.

# %%
### Use rank order to get the top best fit or other parameters
#top 5 best fit models metallicities for the cloud free grid
print("cld free",np.array(out['grid_params']['picaso_cld_free']['planet_params']['mh']
        )[out['rank_order']['picaso_cld_free']['firefly']][0:5])

#top 5 best fit models metallicities for the cloudy grid
print("cldy",np.array(out['grid_params']['picaso_cldy']['planet_params']['mh']
        )[out['rank_order']['picaso_cldy']['firefly']][0:5])

# %% [markdown]
# Interesting! We are already seeing interesting information. Without clouds our model predicts higher metallicity than when we add clouds. Let's look at the associated chi square values.

# %%
#top 5 best fit chi sqs for the cloud free grid
print("cld free", np.array(out['chi_sqs']['picaso_cld_free']['firefly']
        )[out['rank_order']['picaso_cld_free']['firefly']][0:5])

#top 5 best fit chi sq for the cloudy grid
print("cldy", np.array(out['chi_sqs']['picaso_cldy']['firefly']
        )[out['rank_order']['picaso_cldy']['firefly']][0:5])

# %% [markdown]
# The cloudy grid is giving lower chi square giving us clues that this planet likely has clouds affecting the spectrum.

# %% [markdown]
# ## Analyze Single Best Fits
#
# Let's analyze the single best fits in order to compare the spectrum with the data

# %%
fig,ax = fitter.plot_best_fit(['picaso_cld_free','picaso_cldy'],'firefly')

# %% [markdown]
# By-eye, our cloudy grid is giving a much better representation of the data. Let's look at what physical parameters are associated with this.

# %%
best_fit = fitter.print_best_fit('picaso_cldy','firefly')

# %% [markdown]
# You can see these same parameters reported in original Nature paper: https://arxiv.org/pdf/2208.11692.pdf

# %% [markdown]
# ## Estimated Posteriors
#
# It is also helpful to get an idea of what the probability is for each grid parameter in your model. This will give you a better representation of degeneracies that exist with your data and each of your physical parameters.

# %%
posterior_chance_dict, fig = fitter.plot_chi_posteriors(['picaso_cldy', 'picaso_cld_free'],
                                                        'firefly', max_row=3, max_col=2, input_parameters='all')

# %%
posterior_chance_dict

# %% [markdown]
# What can you take away from this plot?
# 1. Cloudy models reduce the number of models that can be fit to the data with high metallicity
# 2. Internal temperature cannot be constrained by the data
# 3. C/O ratios greater than ~0.8 can be ruled out by the data

# %% [markdown]
# ## Interpret Best Fit
#
# Now that we are happy with the best-fitting model, we can load in that data and post process some plots in order to gain better understanding of our results.
#
# We can use `PICASO`'s `xarray` loader to quickly load in one of our models.

# %%
#grab top model
top_model_file  = np.array(out['list_of_files']['picaso_cldy']
        )[out['rank_order']['picaso_cldy']['firefly']][0]

xr_usr = jdi.xr.load_dataset(top_model_file)
#take a look at the Xarray file
xr_usr

# %%
opa = jdi.opannection(wave_range=[3,5])
case = jdi.input_xarray(xr_usr, opa)
#if you need to rerun your spectrum
#out = case.spectrum(opa,calculation='transmisson')

# %% [markdown]
# ### See Contribution From Each Molecule
#
# One of the most common plots that was also used in the original paper is the "leave one out" method to see how each molecule is affecting our spectrum.

# %%
# #copy atmo before modifying and rerunning picaso
og_atmo = jdi.copy.deepcopy(case.inputs['atmosphere']['profile'])
#atmo
w,f,l =[],[],[]
for iex in ['CH4','H2O','CO2',None]:
    case.atmosphere(df = og_atmo,exclude_mol=iex, sep=r'\s+')
    df= case.spectrum(opa, full_output=True,calculation='transmission') #note the new last key
    wno, rprs2  = df['wavenumber'] , df['transit_depth']
    wno, rprs2 = jdi.mean_regrid(wno, rprs2, R=150)
    w +=[wno]
    f+=[rprs2]
    if iex==None:
        leg='all'
    else:
        leg = f'No {iex}'
    l+=[leg]
jpi.show(jpi.spectrum(w,f,legend=l))

# %% [markdown]
# ## Quantify Molecular Detection using Gaussian Fitting
#
#
# For very gaussian shaped molecules (like CO2 in this case), we can use a simple Gaussian fitting technique to quantify the significance of our detection. Note this ONLY works in cases where the shape of the molecule is gaussian with a single peak and well-shaped wings.

# %%
#grab file to test
top_model_file  = np.array(out['list_of_files']['picaso_cldy']
        )[out['rank_order']['picaso_cldy']['firefly']][0]

min_wave = 3 #min wave to search for gauss peak
max_wave = 5 #max wave to search for gauss peak
out = lyz.detection_test(fitter,'CO2',min_wave,max_wave,'picaso_cldy','firefly',
                     top_model_file,
                     #opa_kwargs={wave_range=[]}#this is where you input arguments for opannection
                     plot=True)

# %% [markdown]
# By comparing the line fit to the single Gaussian fit, we can use the methodology of [Trotta 2008](https://ui.adsabs.harvard.edu/abs/2008ConPh..49...71T/abstract) to get out a sigma detection significance. In this case we can see that the single gaussian fit is preferred over the line model at 26 sigma.

# %%
out['sigma_single_v_line']
