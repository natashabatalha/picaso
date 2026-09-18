# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
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
# # Retrieval Analysis Tutorial
#
# In this tutorial, you will learn how to analyze the results of a retrieval run (using either `dynesty` or `ultranest` nested samplers) within Python.
#
# We will cover:
#
# 1. Loading the configuration TOML file used in the retrieval.
# 2. Extracting parameter names using `prior_finder`.
# 3. Loading the retrieval outputs (samples, likelihoods, summary statistics) using `read_retrievals`.
# 4. Generating a corner plot from the samples using `plot_pair`.
# 5. Extracting the maximum log-likelihood point, generating its corresponding model spectrum via `check_model_samples`, and calculating the chi-squared statistic.

# %%
import os
import toml
import numpy as np
import matplotlib.pyplot as plt
import picaso.driver as go
import picaso.retrieval as ret 

# %% [markdown]
# ## 1. Load the Configuration File
#
# We'll load the TOML configuration file. This file contains metadata about the retrieval, the parameters being retrieved, and the location of the retrieval output directory.

# %%
# Path to your configuration file (similar to driver.toml)
config_path = "/Users/nbatalh1/Documents/research/WASP17b/test_driver_retrievals/wasp-17-lrs-example.toml"

config = go.load_config(config_path)

# Ensure we have a valid path for retrieval_output in InputOutput
os.path.isdir(config['InputOutput']['retrieval_output'] )

# %% [markdown]
# ## 2. Identify the Retrieved Parameters
#
# We use the `prior_finder` helper from `picaso.driver` to find all parameters being fit under the `[retrieval]` section.

# %%
fitpars = go.prior_finder(config['retrieval'])
params = list(fitpars.keys())
print("Retrieved parameters found in configuration:")
print(params)

# %% [markdown]
# ## 3. Read Retrieval Outputs
#
# The `read_retrievals` function in `picaso.retrieval` automatically detects if the nested sampling output is from `dynesty` (looking for a `dynesty.save` checkpoint file) or from `ultranest`. It then loads equal-weighted samples, the maximum log-likelihood value, and summary statistics.

# %%
retrieval_dir = config['InputOutput']['retrieval_output']

info = ret.read_retrievals(retrieval_dir, params)

# For the sake of this tutorial, we describe the dictionary returned by read_retrievals:
# - info['samples_equal']: numpy array of equally-weighted samples (n_samples, n_params)
# - info['max_logl']: maximum log-likelihood value
# - info['max_logl_point']: parameters at the maximum likelihood point (array of length n_params)
# - info['med_intervals']: DataFrame of parameter statistics/intervals
# - info['param_names']: list of parameter names

# %% [markdown]
# ## 4. Display Corner Plot
#
# We can easily plot the parameter correlations and distributions using the `plot_pair` function.

# %%
fig, ax = ret.plot_pair(info['samples_equal'], info['param_names'])
plt.show()

# %% [markdown]
# ## 5. Model the Maximum Likelihood Spectrum and Compute Chi-Squared
#
# Using the maximum log-likelihood point `max_logl_point`, we can call `check_model_samples` with `N=1` to generate the corresponding model spectrum on the data grid. We then calculate the chi-squared statistic and overlay the data and model.

# %%
max_logl_point = info['max_logl_point']
out = go.check_model_samples(
    config, N=1, 
    samples=np.atleast_2d(max_logl_point),
    full_likelihood=True
    )

# %%
# Plot the results
wavelength = 1e4 / out['xdata']
plt.figure(figsize=(8, 5))
plt.errorbar(wavelength, out['ydata'][0],
              yerr=out['edata'][0], 
              fmt='o', color='black', label='Data')
plt.plot(wavelength, out['ymodel'][0], color='red', label='Max LogL Model')
chi2=out['chi_sq_per_pt'][0]
plt.title(f"(Chi-sq = {chi2:.2f})")
plt.xlabel("Wavelength (micron)")
plt.ylabel("Flux / Transit Depth")
plt.legend()
plt.show()

# %% [markdown]
# ## 6. Banded Plots for Profiles and Spectra
#
# Now let's get the sigma intervals for chemistry/temperature. We will rerun the same function but now use the samples.

# %%
returns = ret.get_bands(config, info,
                    pressure_bands=
                    ['temperature','H2O','CO2'],eval_maxlogl=True)

f_chem,a_chem = ret.plot_pressure_bands(returns)
f_spec,a_spec = ret.plot_spectra_bands(returns)

# %% [markdown]
# ## 7. Save and Bundle Results 
#

# %%
filename = '/Users/nbatalh1/Documents/research/WASP17b/test_driver_retrievals/results/w17'
xarray_bundle =ret.retrieval_results(returns, 
    info, filename,round=3,
    return_samples=True,
    spectrum_tag='transit_depth',
    spectrum_unit='cm**2/cm**2',
    author="",contact="",model_description="",
    code="PICASO")

# %%
