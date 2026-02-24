# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     notebook_metadata_filter: nbsphinx
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
#   nbsphinx:
#     execute: never
# ---

# %% [markdown]
# # Quickstart
#
# Here is the quickstart to getting up and running with `PICASO` using `python` to set environment variables and `get_data` to get everything you need.
#
# Be sure to have followed the [installation instructions](https://github.com/James-Mang/dev_picaso/blob/9278a1df9bab306b9e03a2a427a59e7ddaaa7e09/docs/installation.rst) and follow the steps below depending on how you installed PICASO.
#
# This notebook is organized as follows:
# 1. One additional step **only those who used pip or conda to install PICASO**
# 2. Run PICASO environment checker
# 2. Additional data that you might need **regardless of installation method**
# 3. Quickstart for quick simple examples and workshops 

# %% [markdown]
# ## Create `picaso_refdata` environment variable
#
# We give [three different ways of setting environment variables here](https://natashabatalha.github.io/picaso/installation.html#create-environment-variable). Setting them with `os` is perfectly fine though some users like setting them system wide so that they do not have to constantly set paths.
#
#

# %%
import os
print(os.environ['picaso_refdata'] ) #should return a path

#does it not?? Lets make sure you have it set in your environment:
os.environ['picaso_refdata'] = 'YOUR_PATH/picaso/reference/'
os.environ['PYSYN_CDBS'] = os.path.join(os.environ['picaso_refdata'],'stellar_grids')

# %% [markdown]
# Note what we did above sets the environment variable which is totally okay but this way you will need to add this to the top of all your future notebooks **before you import picaso** if you haven't set it in your bash file

# %% [markdown]
# # 1. For those who pip or conda installed PICASO:

# %% [markdown]
# ## Download reference data directory
#
# We need this basic directory: https://github.com/natashabatalha/picaso/tree/master/reference
#
# If you installed through pip or conda then you will need to do this step. If you did a git clone then you should already have this and just need to point `picaso_refdata` to the directory `reference`.

# %%
import picaso.data as data

# %% [markdown]
# This next cell you can run to download **ALL** the reference data. You only need to run this next cell **ONCE**
# **WARNING**: if you run it again, this will download all the reference data again and overwrite anything you have in there.

# %%
data.get_reference(os.environ['picaso_refdata']) #only ever need to do one time

# %% [markdown]
# # 2. Run PICASO Environment Checker

# %%
data.check_environ()

# %% [markdown]
# You might see that we still need to download the resampled opacity file.

# %% [markdown]
# # 3. Additional reference data you'll need for PICASO

# %% [markdown]
# ## Downloading the resampled opacity file
#
# If you want to make this the picaso default then you should put it here:
#
# - $picaso_refdata/opacities
#
# The following function will do that for you if you have set up your path correctly.
#
# **Note**: This is ~ 5GB file so please make sure that you have a reliable internet connection before trying to download this file, otherwise you might encounter a timeout error.

# %%
#commented out for docs build
data.get_data(category_download='resampled_opacity',target_download='default')


# %% [markdown]
# ## Download the stellar grids needed for exoplanet modeling (optional)
#
# If you want to use these stellar files they will need to be accessed by pysynphot package which checks for them here:
#
# - $PYSYN_CDBS/grid
#
# You will be asked if you want to download phoenix or ck04models. we recommend ck04models as a default, but phoenix for mid-IR spectra and climate modeling.
#

# %%
#now let's make sure we have the pysynphot environment variable set
data.get_data(category_download='stellar_grids')

# %% [markdown]
# ## Want anything else? Use `get_data` function
#
#
# PICASO relies on lots of different kinds of data. However you might not need all of it depending on what you are working on. For example, if you are only working on substellar objects, you do not need to download stellar spectra.
#
# | Data Type                        | Req? | What it is primarily used for | Where it should go                                          |
# |----------------------------------|------|-------------------------------|-------------------------------------------------------------|
# | Reference                        | Yes  | everything                    | $picaso_refdata                                             |
# | Resampled Opacities              | Yes  | Spectroscopic modeling        | $picaso_refdata/opacities/opacities*.db                     |
# | Stellar Database                 | No   | Exoplanet modeling            | $PYSYN_CDBS/grid                                            |
# | Pre-weighted correlated-K Tables | No   | Chemical equilibrium climate  | Your choice (default=$picaso_refdata/opacities/preweighted) |
# | By molecule correlated-K Tables  | No   | Disequilibrium climate        | Your choice (default=$picaso_refdata/opacities/resortrebin) |
# | Sonora grid models               | No   | Initial guess/grid fitting    | Your choice (default=$picaso_refdata/sonora_grids)          |
# | Virga Mieff files                | No   | Virga cloud modeling          | Your choice (default=$picaso_refdata/virga)                 |
#
#
# ### Examples using get data in interactive mode
# ```
# data.get_data()
# What data can I help you download? Options include:
# ['resampled_opacity', 'stellar_grids', 'sonora_grids', 'ck_tables']
# >> sonora_grids
# Great. I found these options for sonora_grids. Select one:
# 0 - 'elfowl-Ytype': The models between Teff of 275 to 550 K (applicable to Y-type objects). Total: ~40 Gb.
# 1 - 'elfowl-Ttype': The models for Teff between 575 to 1200 K (applicable for T-type objects). Total: ~40 Gb.
# 2 - 'elfowl-Ltype': Models for Teff between 1300 to 2400 K (applicable for L-type objects). Total: ~40 Gb.
# 3 - 'bobcat': Sonora bobcat pressure-temperature profiles
# 4 - 'diamondback':
# >> elfowl-Ytype
# No destination has been specified. Let me help put this in the right place.
# When running the code you will have to point to this directory. Therefore, keep it somewhere you will remember. My suggestion would be something like /Users/myaccount/Documents/data/picaso_data/sonora_grids. Please enter a path:
# /Users/nbatalh1/Documents/data/sonora_grids/elfowl
# ```

# %%
data.get_data()

# %% [markdown]
# # 4. Quickstart for Students & Learning
#
# Here is the quickstart to getting up and running with `PICASO` using `python` to set environment variables and `get_data` to get everything you need for PICASO-lite

# %%
#install the code
# !pip install picaso

# %%
import picaso.data as d

#pick a path to download all the reference data
d.os.environ['picaso_refdata'] = '/data/reference_data/picaso/reftest'

#get all needed data
d.get_data(category_download='picaso-lite', target_download='tutorial_sagan23',final_destination_dir=d.os.environ['picaso_refdata'] )


#add this to the top of any picaso notebook that models exoplanets
d.os.environ['PYSYN_CDBS'] = d.os.path.join(d.os.environ['picaso_refdata'],'stellar_grids')


#use this for any virga related notebook
mieff_dir = d.os.path.join(d.os.environ['picaso_refdata'],'virga')
#use this for any climate modeling (this must includes a few metallicities and c/o ratios so you can get an idea of how to do climate modeling)
ck_dir = d.os.path.join(d.os.environ['picaso_refdata'],'opacities', 'preweighted')
