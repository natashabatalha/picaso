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
# # WIP Using the Driver Function to Compute Jacobians 
#
# In this notebook we will create a simple driver setup of a Jupiter-like case and use it to compute the jacobian for a set of parameters. 
#
# You should already be familiar with: 
#
# - how to compute reflected light spectrum
# - how to use and edit a driver.toml file
#

# %%
from picaso import justdoit as jdi
from picaso import information_content as ic
from picaso import justplotit as jpi
jpi.output_notebook()

# %%
_default_ = jdi.__refdata__ 

simple_input = {
    'OpticalProperties': {'opacity_file': f'{_default_}/opacities/opacities.db',
                       'opacity_kwargs': {'wave_range':[0.3,2.0]},
                       'opacity_method': 'resampled',
                       'virga_mieff': f'{_default_}/virga/'},
    'calc_type': 'spectrum',
    'irradiated': True,
    'geometry': {'phase': {'unit': 'radian', 'value': 0}},
    'object': {'distance': {'unit': 'parsec', 'value': 8.3},
            'gravity': {'unit': 'cm/s**2', 'value': 100000.0},
            'mass': {'unit': 'Mjup', 'value': 1.2},
            'radius': {'unit': 'Rjup', 'value': 1.2},
            'teff': {'unit': 'Kelvin', 'value': 5400},
            'teq': {'unit': 'Kelvin', 'value': 500}},
    'observation_type': 'reflected',   
    'star': {'grid': {'database': 'ck04models',
                   'feh': 0,
                   'logg': 4,
                   'teff': 5400},
          'radius': {'unit': 'Rsun', 'value': 1},
          'semi_major': {'unit': 'AU', 'value': 200}
    },
    'temperature':{
        'profile':'guillot',
        'pressure':{
            'reference': {'value': 1e1, 'unit': 'bar'},
            'min': {'value': 1e-5, 'unit': 'bar'},
            'max': {'value': 1e3, 'unit': 'bar'},
            'nlevel': 60,
            'spacing': 'log'
        },
        'guillot': {'T_int': 100,
                             'Teq': 200,
                             'alpha': 0.5,
                             'logKir': -1.5,
                             'logg1': -1}
    }, 
    'chemistry':{
            'method': 'visscher',
            'visscher': {'cto_absolute': 0.55, 'log_mh': 1.7},
    },
    'clouds':{
        'cloud1_type': 'virga',
        'cloud1': {'virga':
                    {'condensates': ['H2O'],
                        'fsed': 2,
                        'kzz': 1e8,
                        'mh': 100,
                        'mmw': 2.2,
                        'sig': 2
                        }
                    },
        }
    } 

# %%
spectrum = ic.run(driver_dict=simple_input)

# %%
x,y = jdi.mean_regrid(spectrum['wavenumber'],
                      spectrum['albedo'],R=300)

jpi.show(jpi.spectrum(x,y, plot_width=500))

# %%
jac_params = ['']

# %%
