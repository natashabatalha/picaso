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
# # Computing Jacobians and IC Statistics 
#
# In this notebook we will create a simple driver setup of a Jupiter-like case and use it to compute the jacobian for a set of parameters. 
# You will also learn how to do the same thing through a picaso class that will directly perturb your pressure dependent arrays
#
# You should already be familiar with: 
#
# - how to compute a spectrum
# - how to use and edit a driver.toml file
#

# %%
from picaso import justdoit as jdi
np = jdi.np
from picaso import information_content as ic
from picaso import justplotit as jpi
jpi.output_notebook()

# %% [markdown]
# ## Using a Driver File to Compute Jacobian

# %%
_default_ = jdi.__refdata__ 

simple_input = {
    'OpticalProperties': {'opacity_file': f'{_default_}/opacities/opacities.db',
                       'opacity_kwargs': {'wave_range':[0.3,2.0]},
                       'opacity_method': 'resampled',
                       'virga_mieff': f'{_default_}/virga/'},
    'calc_type': 'spectrum',
    'irradiated': True,
    'geometry': {'phase': {'unit': 'radian', 'value': jdi.np.pi/2}, 
                 'phase_kwargs':{'num_tangle':6, 'num_gangle':6}},
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
            'method': 'chemeq_on_the_fly',
            'chemeq_on_the_fly': {'cto_absolute': 0.55, 'log_mh': 2},
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

# %% [markdown]
# ### Compute Initial State Vector 

# %%
spectrum,cl = ic.run(driver_dict=simple_input,return_class=True)

# %%
x,y = jdi.mean_regrid(spectrum['wavenumber'],
                      spectrum['albedo'],R=300)

jpi.show(jpi.spectrum(x,y, plot_width=500))

# %% [markdown]
# ### Compute Jacobian Given a Parameter Set

# %%
jac_params = ['cto_absolute','log_mh','fsed','Teq','phase']

# %%
jac_mat = ic.jacobian(driver_dict = simple_input, params = jac_params)

# %%
import matplotlib.pyplot as plt
K_rebin = []
for i, ip in enumerate(jac_params):
    x,y = jdi.mean_regrid(spectrum['wavenumber'], jac_mat[:,i],R=200)
    K_rebin += [y]
    plt.plot(1e4/x,jdi.np.abs(y/jdi.np.max(jdi.np.abs(y))), label=ip)
plt.legend()

# %% [markdown]
# ## Compute Information Statistics 

# %%
error = 0.01 #+- on abledo itself 
IC_analyzer = ic.Analyze(spectrum['wavenumber'], jac_mat, error , R=200)
DOF_SVD = IC_analyzer.degrees_of_freedom_svd()
prior = [1, 2, 10, 500, np.pi]
SIC = IC_analyzer.shannon_ic(prior)

# %% [markdown]
# ## Using a PICASO Class to Compute Jacobian and IC Statistics 
#
# Using the same PICASO class that was setup above, lets compute the jacobian on directly the atmosphere parameters using the picaso class we setup

# %%
jac_params = ['atmosphere.profile.H2O','atmosphere.profile.CH4','atmosphere.profile.CO2','atmosphere.profile.temperature']
is_log = [False,False,False,False]

opacityclass =jdi.opannection(filename_db= simple_input['OpticalProperties']['opacity_file'], **simple_input['OpticalProperties']['opacity_kwargs'])
calculation = simple_input['observation_type']

jac_mat_class = ic.jacobian(picaso_class = cl, params = jac_params, is_log=is_log, calculation=calculation, opacityclass=opacityclass) 


# %%
import matplotlib.pyplot as plt
K_rebin = []
for i, ip in enumerate(jac_params):
    x,y = jdi.mean_regrid(spectrum['wavenumber'], jac_mat_class[:,i],R=200)
    K_rebin += [y]
    plt.plot(1e4/x,jdi.np.abs(y/jdi.np.max(jdi.np.abs(y))), label=ip)
plt.legend()

# %%
error = 0.01 #+- on abledo itself 
IC_analyzer_cl = ic.Analyze(spectrum['wavenumber'], jac_mat_class, error , R=200)
DOF_SVD = IC_analyzer_cl.degrees_of_freedom_svd()
prior = [1, 1, 1, 500]
SIC = IC_analyzer_cl.shannon_ic(prior)


# %% [markdown]
# ## Save your Jacobian data to use with the PICASO UI

# %%
np.savez('jacobian_data.npz', jacobian=jac_mat, wno=spectrum['wavenumber'], params=jac_params)
