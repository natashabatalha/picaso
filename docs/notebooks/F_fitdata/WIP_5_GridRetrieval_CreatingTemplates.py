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
# # Creating Retrieval Scripts
#
# If this is your first crack at fitting parameters with PICASO, we strong encourage you to look at the first three retrieval tutorial. This notebooks on creating scripts with multiple models to test.
#
# **What you will learn:**
#
# 1. All the template scripts that are available to aid in retrieval analysis
# 2. How to create a template for a grid + flex cloud retrieval
#
# **What you should know:**
#
# 1. `picaso`'s retrieval class structure (model_set, prior_set, param_set)
# 2. `picaso.retrieval.create_template` function to generate template scripts (covered in Retrieval Tutorial 1 & 2)
# 3. `picaso.analyze.prep_gridtrieval` function to simply generate and interpolate on a spectrum
# 4. Running `virga` cloud models `picaso` forward modeling (e.g. computing transmission spectra with `justdoit.inputs`)
#
# ## Basics of Retrieval Templates
#
# We have introduced these templates in previous tutorials. Here will show a few more examples of template creation.
#
# **How should I use these templates??**
#
# These templates are supposed to offer you a headstart to building retrievals yourself. You are encouraged to write your own retrieval functions using these are **starting points**.

# %%
import picaso.retrieval as pr

# %% [markdown]
# ## Gridtrieval + Postprocessed Chemistry

# %%
rtype='grid_addchem' #first lets specify the retrieval type 'grid'
sript_name='run_test.py' #speciy a script name
sampler_output='/data/test/ultranest/grid_virga'

grid_location = '/data2/models/WASP-17b/spec/zenodo/v1' # should ultimately point to location of all .nc files
grid_name = 'cldfree' #for your own book-keeping
to_fit = 'transit_depth' #this is based on what you want to fit in the xarray files s

#opacity and chemistry
opacity_filename_db = '/data2/picaso_dbs/R15000/all_opacities_0.3_15_R15000.db'
molecules = ['H2','He','H2O','CO','CO2','CH4','H2O']

#bonus chemistry
overwrite_molecule = 'SO2' #in this simple example we will add SO2

#new
grid_kwargs={'grid_location':grid_location,'grid_name':grid_name,'to_fit':to_fit,
            'opacity_filename_db':opacity_filename_db, 'molecules':molecules,
            'overwrite_molecule':overwrite_molecule}

pr.create_template(rtype,sript_name,sampler_output,grid_kwargs=grid_kwargs)

# %% [markdown]
# ## Gridtrieval + Flex Cloud

# %%
rtype='grid_flexcloud' #first lets specify the retrieval type 'grid'
sript_name='run_test.py' #specify a script name
sampler_output='/data/test/ultranest/grid_flexcloud'

grid_location = '/data2/models/WASP-17b/spec/zenodo/v1' # should ultimately point to location of all .nc files
grid_name = 'cldfree' #for your own book-keeping
to_fit = 'transit_depth' #this is based on what you want to fit in the xarray files s

#opacity and chemistry
opacity_filename_db = '/data2/picaso_dbs/R15000/all_opacities_0.3_15_R15000.db'
molecules = ['H2','He','H2O','CO','CO2','CH4','H2O']

#cloud things
virga_mieff_files = '/data/virga_dbs/virga_0,3_15_R300/'
cloud_species = ['SiO2']

#new
grid_kwargs={'grid_location':grid_location,'grid_name':grid_name,'to_fit':to_fit,
            'opacity_filename_db':opacity_filename_db, 'molecules':molecules,
            'virga_mieff_dir':virga_mieff_files,'cloud_species':cloud_species}

pr.create_template(rtype,sript_name,sampler_output,grid_kwargs=grid_kwargs)

# %% [markdown]
# ## Gridtrieval + Virga

# %%
rtype='grid_virga' #first lets specify the retrieval type 'grid'
sript_name='run_test.py' #specify a script name
sampler_output='/data/test/ultranest/grid_virga'

grid_location = '/data2/models/WASP-17b/spec/zenodo/v1' # should ultimately point to location of all .nc files
grid_name = 'cldfree' #for your own book-keeping
to_fit = 'transit_depth' #this is based on what you want to fit in the xarray files s

#opacity and chemistry
opacity_filename_db = '/data2/picaso_dbs/R15000/all_opacities_0.3_15_R15000.db'
molecules = ['H2','He','H2O','CO','CO2','CH4','H2O']

#cloud things
virga_mieff_files = '/data/virga_dbs/virga_0,3_15_R300/'
cloud_species = ['SiO2','Al2O3']

#new
grid_kwargs={'grid_location':grid_location,'grid_name':grid_name,'to_fit':to_fit,
            'opacity_filename_db':opacity_filename_db, 'molecules':molecules,
            'virga_mieff_dir':virga_mieff_files,'cloud_species':cloud_species}

pr.create_template(rtype,sript_name,sampler_output,grid_kwargs=grid_kwargs)


# %% [markdown]
# ## Create One Template with Multiple Models
#
# Once you build your templates, it's recommended creating a single script with all the functions you would like to test for one single data set. For example, you can imagine your class `param_set` looking like this:

# %%
class param_set:
    grid = ['mh','cto','tint','heat_redis','offset']
    grid_virga = ['mh','cto','tint','heat_redis','xrp','logfsed','logkzz']
    grid_flexcloud = ['mh','cto','tint','heat_redis','xrp','logcldbar','logfsed','logndz','sigma','lograd']

# %% [markdown]
# Why organize things this way?
#
# 1. All models are can be imported with one script and called with e.g.
#
# `MODEL=getattr(model_set, 'grid')`
#
# 2. Using the techniques outlined in the previous notebooks you can easily debug each of your built functions in a loop
#
# ```
# for test_model in ['grid','grid_virga','grid_flexcloud']:
#     GUESS = getattr(guesses_set, test_model)
#     MODEL = getattr(model_set, test_model)
#     x,y,inf,err = MODEL(GUESS)
#     plt.plot(x,y,label=test_model)
# ```
#
# 3. Sending retrieval scripts to others for reproducibility is easier as all code is contained into a single script.
# 4. Publishing retrieval code on Zenodo is easier as models are kept in one file with their respective priors, enabling reproducibility.
