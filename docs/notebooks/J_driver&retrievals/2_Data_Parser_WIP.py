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
# # PICASO Observational Data Parse Function
#
# This function allows you to easily parse data and convert to PICASO's standard units for flux and wavelength. 

# %%
import picaso.driver as go
import os

# %% [markdown]
# Using the parse data function without going through the driver function 

# %%
#first lets make some sample data 
go.pd.DataFrame(dict(
    x = go.np.linspace(1,10,100),
    flux = go.np.zeros(100)+1e-8, 
    flux_error = go.np.zeros(100)+1e-10
)).to_csv('test_ascii.csv')

#definte the units
flux_unit = 'erg*cm^(-2)*s^(-1)*Hz^(-1)'

#run the data parser 
out = go.parse_data('test_ascii.csv', 'x', 'flux',
                    'flux_error', coord_unit = 'cm^(-1)', data_unit = flux_unit)
# returns a dictionary with ordered wavenumber, flux (in picaso units or erg/cm3/s), error
