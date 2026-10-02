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
# # Running Spectra with Configuration Files (driver.toml)
#
# While PICASO can be run interactively using the standard class APIs, it can also be run entirely from a single TOML configuration file (typically named `driver.toml`). This allows for highly reproducible runs, cleaner scripting, and seamless integration with PICASO's automated spectra calculation, climate, and retrieval pipelines.
#
# In this tutorial, you will learn:
#
# 1. How to load and run a spectrum via the configuration file using `picaso.driver.run`.
# 2. What options exist in the TOML configuration.
# 3. What each option does, with links directly to their corresponding implementations in `parameterizations.py`.

# %% [markdown]
# ## Quickstart: Running a Spectrum with picaso.driver.run
#
# Executing PICASO from a configuration file is incredibly straightforward. The `picaso.driver` module exposes a `run` function which accepts either a file path to your TOML configuration or a pre-loaded Python dictionary.
#
# Here is the basic usage:

# %%
import picaso.driver as go
import os

# Under normal usage, you would specify the path to your toml file:
# output = go.run(driver_file="my_driver.toml")

# Let's inspect the options by loading the master config provided in the reference package data
# %%
# Get the path to the default driver.toml in the picaso_refdata directory
refdata_dir = os.getenv("picaso_refdata")
if refdata_dir:
    master_toml_path = os.path.join(refdata_dir, "input_tomls", "driver.toml")
    print(f"Master TOML is located at: {master_toml_path}")
else:
    print("Please make sure your picaso_refdata environment variable is configured!")

# %% [markdown]
# Let's see how we can load, examine, and run this configuration directly in Python:

# %%
# Standard loading using tomllib/toml
if refdata_dir:
    import tomllib
    with open(master_toml_path, "rb") as f:
        config = tomllib.load(f)
        
    # We can inspect the top-level keys
    print("Top-level keys in configuration:")
    print(list(config.keys()))

# you can also load a slightly cleaned up version with a go function call 

config = go.load_template_config()

# %% [markdown]
# The master `driver.toml` in reference data has examples of every input option. Not every single block needs to be filled out. For example, let's inspect `config['temperature]`

# %%
#this config element contains an example of all the available types of TP profiles
config['temperature'].keys()

# %%
print('This config will use this type of profile',config['temperature']['profile']) 

print('With the following inputs',config['temperature']['userfile']) 

# %% [markdown]
# If we wanted to change the type of PT profile specified we would simply change "profile" and ensure the corresponding block had the inputs we wanted. E.g.,  

# %%
config['temperature']['profile'] = 'knots'
print('Here are some default knot inputs',config['temperature']['knots'])

# %% [markdown]
# ## Complete Options Reference and Parameterizations Mapping
#
# Below is a detailed breakdown of all the sections in `driver.toml` and what each option represents.
#
# ---
#
# ### 1. Global / Top-level Options
#
# * `observation_type` (str): Specifies the kind of calculation/observation to perform.
#   * Options: `'thermal'`, `'fpfs_thermal'`, `'transit_depth'`, `'albedo'`, `'fpfs_reflected'`.
# * `irradiated` (bool): `true` if the planet is irradiated by a star (requires star options to be filled), `false` for self-luminous bodies like brown dwarfs.
# * `calc_type` (str):
#   * `'spectrum'`: Calculate a single forward spectrum.
#   * `'retrieval'`: Run a parameter retrieval.
#   * `'climate'`: Undergo a self-consistent climate calculation (WIP -- not yet implemented).
#
# ---
#
# ### 2. InputOutput Section
#
# * `[InputOutput]`
#   * `retrieval_output` (str): Path to write retrieval outputs.
#   * `spectrum_output` (str): Path to write spectrum outputs.
#
# ---
#
# ### 3. ObservationData Section
#
# Used primarily for retrievals to fit observed data:
#
# * `[ObservationData]`
#   * `filenames` (list of str): List of observational dataset file paths.
#   * `data` (str): Column/variable name in files representing the data values.
#   * `data_unit` (str): Astropy-compatible unit of data values.
#   * `coord` (str): Column/variable name representing coordinates (e.g. wavelength/wavenumber).
#   * `coord_unit` (str): Astropy-compatible unit of coordinates.
#   * `error` (str): Column/variable name representing observational uncertainties.
#   * `instruments` (list of str): Instruments used (e.g., `["jwst nirspec prism"]`) Can run driver.get_instrument_options to see what jwst instruments are available.
#
# ---
#
# ### 4. OpticalProperties Section
#
# Handles the opacity tables and mie scattering properties:
#
# * `[OpticalProperties]`
#   * `opacity_file` (str): Path to the SQLite or HDF5 opacity database.
#   * `opacity_method` (str): `'resampled'` (default), `'preweighted'`, or `'resortrebin'`.
#   * `opacity_kwargs` (dict): Additional parameters for PICASO opacity setup.
#   * `virga_mieff` (str): Directory containing Virga Mie scattering efficiency files.
#
# ---
#
# ### 5. Object Section
#
# Represents the exoplanet / brown dwarf physical properties:
#
# * `[object]`
#   * `radius` (dict): Planetary radius, e.g. `{value=1.2, unit='Rjup'}`.
#   * `mass` (dict): Planetary mass, e.g. `{value=1.2, unit='Mjup'}`.
#   * `gravity` (dict): Surface gravity, e.g. `{value=1e5, unit='cm/s**2'}`.
#   * `distance` (dict): Distance to system, e.g. `{value=8.3, unit='parsec'}`.
#   * `teff` (dict): Effective temperature, e.g. `{value=540, unit='Kelvin'}`.
#   * `teq` (dict): Equilibrium temperature, e.g. `{value=500, unit='Kelvin'}`.
#   * `vrot` (dict): Rotational velocity, e.g. `{value=0, unit='km/s'}`.
#   * `RV` (dict): Doppler velocity, e.g. `{value=0, unit='km/s'}`.
# ---
#
# ### 6. Geometry Section
#
# Viewing geometry for reflected light:
#
# * `[geometry]`
#   * `phase` (dict): Orbital phase angle of the planet, e.g. `{value=0, unit='radian'}`.
#   * `phase_kwargs` (dict): Extra keywords passed to the phase function.
#
# ---
#
# ### 7. Star Section
#
# Required when `irradiated = true`:
#
# * `[star]`
#   * `radius` (dict): Stellar radius, e.g. `{value=1, unit='Rsun'}`.
#   * `semi_major` (dict): Semi-major axis of planet's orbit, e.g. `{value=200, unit='AU'}`.
#   * `type` (str): `'grid'` (stellar model grid) or `'userfile'` (custom file).
# * `[star.grid]`
#   * `teff` (float): Stellar effective temperature (K).
#   * `logg` (float): Stellar surface gravity ($\log g$).
#   * `feh` (float): Stellar metallicity ($\text{[Fe/H]}$).
#   * `database` (str): phoenix, ck04, or other pysynphot databases.
# * `[star.userfile]`
#   * `filename` (str): Custom spectrum file path (column 1: wavelength, column 2: flux).
#   * `w_unit` (str): Wavelength unit.
#   * `f_unit` (str): Flux unit.
#
# ---
#
# ### 8. Temperature Section
#
# Specifies how the atmospheric temperature-pressure (P-T) profile is modeled:
#
# * `[temperature]`
#   * `profile` (str): The method used to model the P-T profile (see options below).
#
# Depending on the chosen profile type under `[temperature]`, PICASO will call the corresponding method in `parameterizations.py`. Here are the available P-T profile options:
#
# * **`'isothermal'`**
#   * Dict block: `[temperature.isothermal]`
#   * Parameters: `T` (isothermal temperature in K)
#   * API Link: See [Parameterize.pt_isothermal](../../picaso.html#picaso.parameterizations.Parameterize.pt_isothermal) or Sphinx role: :meth:`picaso.parameterizations.Parameterize.pt_isothermal`
#
# * **`'knots'`**
#   * Dict block: `[temperature.knots]`
#   * Parameters: `P_knots` (list of pressure knots), `T_knots` (list of temperature knots), `interpolation` (type of interpolation)
#   * API Link: See [Parameterize.pt_knots](../../picaso.html#picaso.parameterizations.Parameterize.pt_knots) or Sphinx role: :meth:`picaso.parameterizations.Parameterize.pt_knots`
#
# * **`'guillot'`**
#   * Dict block: `[temperature.guillot]`
#   * Parameters: `Teq`, `T_int`, `logg1`, `logKir`, `alpha`
#   * API Link: See [Parameterize.pt_guillot](../../picaso.html#picaso.parameterizations.Parameterize.pt_guillot) or Sphinx role: :meth:`picaso.parameterizations.Parameterize.pt_guillot`
#
# * **`'madhu_seager_09_inversion'`**
#   * Dict block: `[temperature.madhu_seager_09_inversion]`
#   * Parameters: `P_1`, `P_2`, `P_3`, `T_3`, `alpha_1`, `alpha_2`, `beta`
#   * API Link: See [Parameterize.pt_madhu_seager_09_inversion](../../picaso.html#picaso.parameterizations.Parameterize.pt_madhu_seager_09_inversion) or Sphinx role: :meth:`picaso.parameterizations.Parameterize.pt_madhu_seager_09_inversion`
#
# * **`'madhu_seager_09_noinversion'`**
#   * Dict block: `[temperature.madhu_seager_09_noinversion]`
#   * Parameters: `P_1`, `P_3`, `T_3`, `alpha_1`, `alpha_2`, `beta`
#   * API Link: See [Parameterize.pt_madhu_seager_09_noinversion](../../picaso.html#picaso.parameterizations.Parameterize.pt_madhu_seager_09_noinversion) or Sphinx role: :meth:`picaso.parameterizations.Parameterize.pt_madhu_seager_09_noinversion`
#
# * **`'zj24'`**
#   * Dict block: `[temperature.zj24]`
#   * Parameters: `pressures`, `dTs`, `Tbottom` (based on Zhang+24 parameterization)
#   * API Link: See [Parameterize.pt_zj24](../../picaso.html#picaso.parameterizations.Parameterize.pt_zj24) or Sphinx role: :meth:`picaso.parameterizations.Parameterize.pt_zj24`
#
# * **`'sonora_bobcat'`**
#   * Dict block: `[temperature.sonora_bobcat]`
#   * Parameters: `sonora_path`, `teff` (retrieves Bobcat P-T profile from grid)
#
# * **`'userfile'`**
#   * Dict block: `[temperature.userfile]`
#   * Parameters: `filename` (path to a custom P-T file), `pd_kwargs` (pandas csv reader options)
#
# ---
#
# ### 9. Chemistry Section
#
# Controls how the molecular and atomic abundances are parameterized:
#
# * `[chemistry]`
#   * `method` (str): Chemical method to use. See available options below.
#
# Depending on the chosen chemistry method, PICASO maps to the corresponding method in `parameterizations.py`:
#
# * **`'free'`** (Free chemical retrieval with custom parameterized abundance profiles):
#   * Dict block: `[chemistry.free]`
#   * Configures custom molecular profiles such as constant, knots, or two-gradient models. See examples for how to specify each molecules inputs.
#   * API Link: See [Parameterize.chem_free](../../picaso.html#picaso.parameterizations.Parameterize.chem_free) or Sphinx role: :meth:`picaso.parameterizations.Parameterize.chem_free`
#   * Sub-methods within free chemistry:
#     * `constant`: Links to constant abundance.
#     * `2gradients`: Links to a gradient decay to simulate either quenching or rainout: See [Parameterize.vmr_2gradients](../../picaso.html#picaso.parameterizations.Parameterize.vmr_2gradients) / :meth:`picaso.parameterizations.Parameterize.vmr_2gradients`
#     * `knots`: flexible spline approach. See [Parameterize.vmr_knots](../../picaso.html#picaso.parameterizations.Parameterize.vmr_knots) / :meth:`picaso.parameterizations.Parameterize.vmr_knots`
#
# * **`'visscher'`** (Equilibrium chemistry calculation based on Visscher analytical model):
#   * Dict block: `[chemistry.visscher]`
#   * Parameters: `cto_absolute` (C/O ratio), `log_mh` (metallicity)
#   * API Link: See [Parameterize.chem_visscher](../../picaso.html#picaso.parameterizations.Parameterize.chem_visscher) or Sphinx role: :meth:`picaso.parameterizations.Parameterize.chem_visscher`
#
# * **`'chemeq_on_the_fly'`** (On-the-fly equilibrium calculation):
#   * Dict block: `[chemistry.chemeq_on_the_fly]`
#   * Parameters: `cto_absolute`, `log_mh`
#   * API Link: See [Parameterize.chem_chemeq_on_the_fly](../../picaso.html#picaso.parameterizations.Parameterize.chem_chemeq_on_the_fly) or Sphinx role: :meth:`picaso.parameterizations.Parameterize.chem_chemeq_on_the_fly`
#
# * **`'userfile'`** (User-provided chemistry profiles):
#   * Dict block: `[chemistry.userfile]`
#   * Parameters: `filename` (path to CSV containing chemical abundances), `pd_kwargs` (pandas options)
#
# ---
#
# ### 10. Clouds Section
#
# Custom parameterized aerosols. You can configure multiple cloud decks, e.g. `cloud1`, `cloud2`:
#
# * `[clouds]`
#   * `cloud1_type` (str): Specifies the model type for `cloud1`.
#
# Mapped cloud types and their corresponding functions in `parameterizations.py`:
#
# * **`'virga'`** (Ackerman & Marley physical cloud models):
#   * Dict block: `[clouds.cloud1.virga]`
#   * Parameters: `mh`, `condensates`, `fsed`, `kzz`, `mmw`, `sig`
#   * API Link: See [Parameterize.cloud_virga](../../picaso.html#picaso.parameterizations.Parameterize.cloud_virga) or Sphinx role: :meth:`picaso.parameterizations.Parameterize.cloud_virga`
#
# * **`'flex_fsed'`** (Sedimenting exponential cloud profile):
#   * Dict block: `[clouds.cloud1.flex_fsed]`
#   * Parameters: `condensate`, `base_pressure`, `ndz`, `fsed`, `distribution`, `lognorm_kwargs`, `hansen_kwargs`
#   * API Link: See [Parameterize.cloud_flex_fsed](../../picaso.html#picaso.parameterizations.Parameterize.cloud_flex_fsed) or Sphinx role: :meth:`picaso.parameterizations.Parameterize.cloud_flex_fsed` (also aliased as `flex_cloud`)
#
# * **`'brewster_mie'`** (Mie scattering cloud decks based on Brewster+ parameterization):
#   * Dict block: `[clouds.cloud1.brewster_mie]`
#   * Parameters: `condensate`, `distribution`, `decay_type`, `lognorm_kwargs`, `hansen_kwargs`, `slab_kwargs`, `deck_kwargs`
#   * API Link: See [Parameterize.cloud_brewster_mie](../../picaso.html#picaso.parameterizations.Parameterize.cloud_brewster_mie) or Sphinx role: :meth:`picaso.parameterizations.Parameterize.cloud_brewster_mie`
#
# * **`'brewster_grey'`** (Grey wavelength-scaling cloud model):
#   * Dict block: `[clouds.cloud1.brewster_grey]`
#   * Parameters: `decay_type`, `alpha`, `ssa`, `reference_wave`, `slab_kwargs`, `deck_kwargs`
#   * API Link: See [Parameterize.cloud_brewster_grey](../../picaso.html#picaso.parameterizations.Parameterize.cloud_brewster_grey) or Sphinx role: :meth:`picaso.parameterizations.Parameterize.cloud_brewster_grey`
#
# * **`'hard_grey'`** (Simple constant grey cloud layer):
#   * Dict block: `[clouds.cloud1.hard_grey]`
#   * Parameters: `g0` (asymmetry), `w0` (single scattering albedo), `opd` (optical depth), `p` (bottom pressure), `dp` (thickness)
#   * API Link: See [Parameterize.cloud_hard_grey](../../picaso.html#picaso.parameterizations.Parameterize.cloud_hard_grey) or Sphinx role: :meth:`picaso.parameterizations.Parameterize.cloud_hard_grey`
#
# * **`'userfile'`** (Custom cloud profile loaded from a file):
#   * Dict block: `[clouds.cloud1.userfile]`
#   * Parameters: `filename` (path to file), `pd_kwargs` (pandas reading options)
#
# ---
#
# ### 11. Doppler Shift & Rotational Broadening
#
# Adds physical velocity effects to the calculated spectra:
#
# * `[doppler_shift]`
#   * `doppler_v` (float): RV doppler velocity shift in km/s.
# * `[rotational_broadening]`
#   * `rotational_v` (float): Projected rotational velocity ($v \sin i$) in km/s.
#   * `eps` (float): Limb darkening coefficient.
#   * `nr` (int): Number of radial bins.
#   * `ntheta` (int): Number of azimuthal bins.
#   * `dif` (float): Differential rotation coefficient.
#
# ---
#
# ### 12. Retrieval Section
#
# Used exclusively when `calc_type = 'retrieval'`. Configures parameter priors and sampler options:
#
# * `[retrieval]`
#   * Set parameters like object radius prior, chemistry priors, cloud parameter priors, etc.
# * `[retrieval.sampler]`
#   * `code` (str): Sampler engine to use, e.g. `'dynesty'` or `'ultranest'`.
#   * `sampler_kwargs` (dict): Keyword arguments passed to the sampler (such as `live_points`).
#   * `resume` (bool): Whether to resume a previously saved retrieval.
#   * `run_kwargs` (dict): Additional runtime settings for the sampler.

# %% [markdown]
# ## Running through all PICASO options
#
# ### All Pressure Temperature Profile Options

# %%
#do all the PT options work?
for i in go.pt_options:
    print(i)
    config = go.load_template_config()
    config['temperature']['profile'] = i
    test1 = go.run(driver_dict =config)

# %% [markdown]
# ### All Chemistry Options

# %%
#do all the chem options work?
for i in go.chem_options:
    config = go.load_template_config()
    config['chemistry']['method']=i
    print(i)
    test1 = go.run(driver_dict =config)

# %% [markdown]
# ### All Cloud Options

# %%
#do all the cloud options work?
for i in go.cloud_options:#running everything except userfile since the pressures won't exaclty line up with this flex pressure option
    config = go.load_template_config()
    config['clouds']['cloud1_type']=i
    print(i)
    if i=='userfile':config['temperature']['profile']='userfile' #just to make sure that pressure grid matches
    test1 = go.run(driver_dict =config)

# %%
