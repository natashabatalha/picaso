# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: pic312
#     language: python
#     name: python3
# ---

# %% [markdown]
# # PICASO Retrieval Setup
#
# By using the parameterizations shown in the previous tutorial, a user is free to setup their likelihood functions and retrieval parameterizations. However, sometimes it is benefitial to use an input file system for run submission. 
#
# PICASO retrieval runs are triggered through creating a `toml` file, similar to what we did in the `Driver Tutorial`. Below we outline two ways to get this toml file spun up (manual, and via our app). We highly recommend the `picaso-app` method as it let's you vet and test your setup before running full retrievals. Once you have your `.toml` configuration file we show how to initiate those runs at the end of this tutorial. 
#
# ## Retrieval Setup Option 1
#
# The easiest way to setup a retrieval toml file is to go through the `streamlit` PICASO app. You can launch this app by typing: 
#
# >> picaso-app
#
# which will open up a web browser. Proceed through the retrieval setup page and download a `configuation.toml` file. 
#
# ## Retreival Setup Option 2 
#
# The other way to setup a retrieval toml file is to do so manually:
#
# - create a copy of the master toml file located in `reference/input_tomls/driver.toml` 
# - edit the entries of the `toml` file as we learned in the Driver Tutorial 
# - indicate in the retrieval section, which parameters are free and add their prior range information 
#
# `reference/input_tomls/driver.toml`  includes note descriptions for each of the input blocks. 
#
# ## Running a Retrieval with a `toml` file
#
# Now that the toml is in hand there are a few ways to submit the job. 
#
# First create a script that runs the retrieval: 
#
# ```python run.py
# import picaso.driver as go
# go.retrieve(driver_file = 'wasp-17-lrs-example.toml')
# ```
#
# ### Using `UltraNest` that uses `openmpi`
#
# Run your script with appropriate `mpiexec` command and number of processors `np`. Note that `np` should be no more than you number of live points.
#
# ```bash
# #!/bin/bash
# mpirun -np 200 python run.py 
# ```
#
# ### Using `dynesty` that uses a `Pool`
#
# Run your script with python 
# ```bash
# #!/bin/bash
# python run.py 
# ```
