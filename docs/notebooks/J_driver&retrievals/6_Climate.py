# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     formats: ipynb,py:percent
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
# # Running Climate Models with Configuration Files (climate.toml)
#
# Just like spectra can be run from a `driver.toml` file (see the [Driver Tutorial](1_Driver_Tutorial.html)), 1D radiative-convective climate models can be run entirely from a single TOML configuration file, typically named `climate.toml`. Its structure mirrors `driver.toml`, and wherever the two overlap (e.g., `[OpticalProperties]`, `[object]`, `[star]`, `[temperature]`, `[chemistry]`) the inputs are identical.
#
# In this tutorial you will learn:
#
# 1. How to run a climate model from a configuration file using `picaso.driver.run_climate`.
# 2. What options exist in the climate TOML configuration.
# 3. How to set up (but not run) a climate class with `picaso.driver.setup_climate_class`.
# 4. How to reproduce the brown dwarf and exoplanet climate tutorials by changing a few config values.
#
# What you should already be familiar with:
#
# - [One-Dimensional Climate Models: The Basics of Brown Dwarfs](../D_climate/1_BrownDwarf_PreW.html)
#
# What you will need to download to use this tutorial:
#
# 1. [Download](https://doi.org/10.5281/zenodo.18636725) the Correlated-k Tables used by the climate code for opacity
# 2. [Download](https://zenodo.org/record/5063476/files/structures_m%2B0.0.tar.gz?download=1) the sonora bobcat cloud free `structures_` file so that you can validate your model run
#
# You can use the `data.get_data` helper function to get these files and add them to the default picaso location:
#
#  >> import picaso.data as d
#
#  >> d.get_data(category_download='ck_tables',target_download='by-molecule')
#
#  >> d.get_data(category_download='sonora_grids',target_download='bobcat')

# %% [markdown]
# ## Quickstart: Running a Climate Model with picaso.driver.run_climate
#
# The `picaso.driver` module exposes a `run_climate` function which accepts either a file path to your climate TOML configuration or a pre-loaded Python dictionary. `picaso.driver.run` also works: if `calc_type='climate'` it hands the config to `run_climate`.

# %%
import picaso.driver as go
from picaso import justdoit as jdi
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Under normal usage, you would specify the path to your toml file:
# out = go.run_climate(driver_file="my_climate.toml")

# %%
# Get the path to the default climate.toml in the picaso_refdata directory
refdata_dir = os.getenv("picaso_refdata")
master_toml_path = os.path.join(refdata_dir, "input_tomls", "climate.toml")
print(f"Master climate TOML is located at: {master_toml_path}")

# %% [markdown]
# Let's load the configuration and look at its top-level keys. `load_climate_template_config` loads the master file, removes the `_options` lists (which are only there to tell you what the options are) and replaces the `_default_` placeholders with your reference data directory.

# %%
config = go.load_climate_template_config()
print("Top-level keys in configuration:")
print(list(config.keys()))

# %% [markdown]
# ## Complete Options Reference
#
# ### 1. Global / Top-level Options
#
# * `irradiated` (bool): `true` for a planet irradiated by a star (requires the `[star]` section and usually `climate.rfacv>0`), `false` for self-luminous objects like brown dwarfs.
# * `calc_type` (str): `'climate'`. This lets `picaso.driver.run` recognize the file as a climate config.
#
# ### 2. InputOutput Section
#
# * `[InputOutput]`
#   * `climate_output` (str): If specified, the climate output is saved to this filename as an xarray netcdf file (via `justdoit.output_xarray`). Requires `climate.with_spec=true`.
#
# ### 3. OpticalProperties Section
#
# Climate calculations use correlated-k opacities:
#
# * `[OpticalProperties]`
#   * `opacity_method` (str): `'preweighted'` or `'resortrebin'`.
#   * `opacity_file` (str): For `preweighted`, the preweighted ck table hdf5 file. The chemistry of these tables is fixed by the file (e.g., `feh0.0_co0.46` = solar metallicity and C/O=0.46). For `resortrebin`, the directory of by-molecule ck tables (leave `''` to use the picaso default).
#   * `opacity_kwargs` (dict): Additional inputs for `justdoit.opannection` (e.g., `preload_gases` for `resortrebin`).
#
# ### 4. Object Section
#
# * `[object]`
#   * `gravity` (dict): Surface gravity, e.g. `{value=1000, unit='m/s**2'}`.
#   * `radius`, `mass` (dict): If both are given, gravity is computed from them instead.
#   * `teff` (dict): Effective temperature of a brown dwarf, or the intrinsic temperature of an irradiated planet, e.g. `{value=1000, unit='K'}`.
#
# ### 5. Star Section
#
# Identical to `driver.toml`, and only used when `irradiated = true`: `radius`, `semi_major`, `type` (`'grid'` or `'userfile'`), `[star.grid]` (`teff`, `logg`, `feh`, `database`) and `[star.userfile]` (`filename`, `w_unit`, `f_unit`).
#
# ### 6. Climate Section
#
# * `[climate]`
#   * `rcb_guess` (int): Index of the level at the top of the guessed convective zone. Must be less than the number of pressure levels minus one.
#   * `rfacv` (float): Fraction of the stellar flux in the energy balance: 0 = no irradiation (brown dwarfs), 0.5 = full day-night redistribution, 1 = dayside.
#   * `rfaci` (float): Fraction of the thermal flux in the energy balance (usually kept at 1).
#   * `moistgrad` (bool): Use the moist adiabatic gradient.
#   * `save_all_profiles` (bool): Save and return every iteration of the temperature profile.
#   * `with_spec` (bool): Compute the spectrum of the converged profile.
#   * `verbose` (bool): Print progress of the climate calculation.
#
# ### 7. Temperature Section
#
# In a climate config, `[temperature]` sets the **initial guess** of the temperature profile and the pressure grid of the climate model. It uses the same blocks as `driver.toml`:
#
# * `profile` (str): One of `go.climate_pt_options`, e.g. `'isothermal'` (as in the brown dwarf tutorial), `'guillot'` (as in the exoplanet tutorial), `'sonora_bobcat'` (as in the resort-rebin tutorial), or `'userfile'`.
# * `[temperature.pressure]`: Pressure grid (`min`, `max`, `nlevel`, `spacing`) for parameterized profiles. `userfile` and `sonora_bobcat` guesses bring their own pressure grid.
#
# ### 8. Chemistry Section
#
# * `method` (str):
#   * `'preweighted'`: Chemistry is set by the preweighted `opacity_file` (requires `opacity_method='preweighted'`).
#   * `'visscher'`: Chemical equilibrium from the Visscher tables, recomputed every iteration (requires `opacity_method='resortrebin'`). Block `[chemistry.visscher]` with `log_mh` and `cto_absolute` (or `cto_relative`), identical to `driver.toml`.
#   * `'chemeq_on_the_fly'`: Chemical equilibrium computed on the fly with photochem's equilibrium solver (requires `opacity_method='resortrebin'`). Block `[chemistry.chemeq_on_the_fly]` with `log_mh` and `cto_absolute`.
# * `no_ph3`, `cold_trap`, `vol_rainout` (bool): Climate chemistry options for `visscher` and `chemeq_on_the_fly` (see `justdoit.inputs.atmosphere`).

# %%
print('Initial guess:', config['temperature']['profile'], config['temperature']['isothermal'])
print('Climate settings:', config['climate'])

# %% [markdown]
# ## Brown Dwarf with Preweighted Opacities
#
# The default `climate.toml` reproduces the [brown dwarf climate tutorial](../D_climate/1_BrownDwarf_PreW.html): a Teff=1000 K, g=1000 m/s$^2$ brown dwarf with solar metallicity preweighted ck tables, starting from a 500 K isothermal guess.

# %%
out, climate_class = go.run_climate(driver_file=master_toml_path, return_class=True)

# %% [markdown]
# Let's compare our run with the Sonora Bobcat model of the same object:

# %%
sonora_profile_db = os.path.join(refdata_dir, 'sonora_grids', 'bobcat')
pressure_bobcat, temp_bobcat = np.loadtxt(os.path.join(sonora_profile_db, "t1000g1000nc_m0.0.cmp.gz"),
                                          usecols=[1,2], unpack=True, skiprows=1)

plt.figure(figsize=(6,6))
plt.semilogy(out['temperature'], out['pressure'], color="r", linewidth=3, label="climate.toml run")
plt.semilogy(temp_bobcat, pressure_bobcat, color="k", linestyle="--", linewidth=3, label="Sonora Bobcat")
plt.ylim(500,1e-4)
plt.xlim(200,3000)
plt.ylabel("Pressure [Bars]")
plt.xlabel('Temperature [K]')
plt.legend()

# %% [markdown]
# ## Setting Up the Climate Class Without Running It
#
# `setup_climate_class` does every step of `run_climate` except the climate calculation itself. This is useful if you want to inspect the inputs, or add something to the climate class (e.g., energy injection) before running it.

# %%
opacity_ck = go.climate_opacity(config)
climate_class = go.setup_climate_class(config, opacity_ck)

guess = climate_class.inputs['climate']
plt.figure(figsize=(6,6))
plt.semilogy(guess['guess_temp'], guess['pressure'])
plt.ylim(500,1e-4)
plt.ylabel("Pressure [Bars]")
plt.xlabel('Temperature [K] (initial guess)')

# %% [markdown]
# To finish the calculation, run the climate function of the class as usual:
#
# >> out = climate_class.climate(opacity_ck, save_all_profiles=True, with_spec=True)

# %% [markdown]
# ## Brown Dwarf with Chemistry Computed On the Fly (Resort-Rebin)
#
# This reproduces the [resort-rebin chemical equilibrium tutorial](../D_climate/1b_BrownDwarf_ResortRebin_Chemeq.html). Instead of preweighted ck tables, the opacities of each molecule are mixed on the fly with chemical equilibrium recomputed every iteration. We start from the Sonora Bobcat profile of a Teff=700 K, g=316 m/s$^2$ brown dwarf.

# %%
config_rr = go.load_climate_template_config()
config_rr['OpticalProperties']['opacity_method'] = 'resortrebin'
config_rr['OpticalProperties']['opacity_file'] = '' #use the default by-molecule directory
config_rr['OpticalProperties']['opacity_kwargs'] = {'preload_gases':['CO','CH4','H2O','NH3','CO2','N2','HCN','H2','He',
                                                                     'PH3','C2H2','Na','K','TiO','VO','FeH']}
config_rr['object']['gravity'] = {'value':316, 'unit':'m/s**2'}
config_rr['object']['teff'] = {'value':700, 'unit':'K'}
config_rr['temperature']['profile'] = 'sonora_bobcat'
config_rr['temperature']['sonora_bobcat']['teff'] = 700
config_rr['climate']['rcb_guess'] = 79
config_rr['chemistry']['method'] = 'visscher'
config_rr['chemistry']['visscher'] = {'log_mh':0.0, 'cto_relative':1.0} #solar metallicity and C/O

out_rr = go.run_climate(driver_dict=config_rr)

# %%
plt.figure(figsize=(6,6))
plt.semilogy(out_rr['temperature'], out_rr['pressure'], "r", label='Resort-Rebin, Chemical Equilibrium')
pressure_bobcat, temp_bobcat = np.loadtxt(os.path.join(sonora_profile_db, "t700g316nc_m0.0.cmp.gz"),
                                          usecols=[1,2], unpack=True, skiprows=1)
plt.semilogy(temp_bobcat, pressure_bobcat, color="k", linestyle="--", label='Sonora Bobcat')
plt.ylim(200,1.7e-4)
plt.ylabel('Pressure [bars]')
plt.xlabel('Temperature [K]')
plt.legend()

# %% [markdown]
# ## Irradiated Exoplanet
#
# This reproduces the [exoplanet climate tutorial](../D_climate/2_Exoplanet_PreW.html) of a WASP-39 b like planet. We need to:
#
# 1. set `irradiated=true` (the default `[star]` section is already set up for WASP-39)
# 2. turn on the stellar flux with `rfacv=0.5` (full day-night heat redistribution)
# 3. use `teff` as the intrinsic temperature of the planet
# 4. start from a Guillot profile on a grid from 1e-6 to 100 bars
#
# We also use the 10x solar metallicity preweighted ck tables and save the output to a netcdf file.

# %%
config_pl = go.load_climate_template_config()
config_pl['irradiated'] = True
config_pl['OpticalProperties']['opacity_file'] = os.path.join(refdata_dir, 'opacities', 'preweighted',
                                                              'sonora_2121grid_feh1.0_co0.46.hdf5')
config_pl['object']['gravity'] = {'value':4.5, 'unit':'m/s**2'}
config_pl['object']['teff'] = {'value':200, 'unit':'K'} #intrinsic temperature
config_pl['temperature']['profile'] = 'guillot'
config_pl['temperature']['guillot']['Teq'] = 1000
config_pl['temperature']['guillot']['T_int'] = 200
config_pl['temperature']['pressure']['min'] = {'value':1e-6, 'unit':'bar'}
config_pl['temperature']['pressure']['max'] = {'value':1e2, 'unit':'bar'}
config_pl['climate']['rcb_guess'] = 85
config_pl['climate']['rfacv'] = 0.5
config_pl['InputOutput']['climate_output'] = 'w39_climate.nc'

out_pl = go.run_climate(driver_dict=config_pl)

# %%
base_case = jdi.pd.read_csv(jdi.HJ_pt(), sep=r'\s+')
plt.figure(figsize=(6,6))
plt.semilogy(out_pl['temperature'], out_pl['pressure'], color="r", linewidth=3, label="climate.toml run")
plt.semilogy(base_case['temperature'], base_case['pressure'], color="k", linestyle="--", linewidth=3, label="WASP-39 b ERS Run")
plt.ylim(100,1e-6)
plt.xlim(200,3000)
plt.ylabel("Pressure [Bars]")
plt.xlabel('Temperature [K]')
plt.legend()

# %% [markdown]
# The saved file can be reloaded with xarray, and used to rerun higher resolution spectra with `jdi.input_xarray` (as shown at the end of the [brown dwarf climate tutorial](../D_climate/1_BrownDwarf_PreW.html)):

# %%
preserved = jdi.xr.load_dataset('w39_climate.nc')
preserved
