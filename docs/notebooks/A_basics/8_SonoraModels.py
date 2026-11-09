# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Sonora Models: Bobcat, Elf Owl, Diamondback and Flame Skimmer
#
# Author:  Caroline Morley "Which Sonora model should I use?" (Exoclimes VI)
#
# The **Sonora** models are grids of brown dwarf and giant planet atmospheres: pressure-temperature profiles, chemical abundances, spectra and (for some) evolution models, all computed assuming radiative-convective equilibrium. PICASO can load any of them as the starting point for your own calculation.
#
# In this tutorial you will learn:
#
# 1. Which Sonora grid to use for your object
# 2. How to download each grid with `picaso.data.get_data`, and how big each one is
# 3. How to load a grid model with `inputs.sonora()`, compute its spectrum with PICASO and compare it with the published spectrum, for each grid: **Bobcat**, **Elf Owl**, **Diamondback** and **Flame Skimmer**
# 4. How to use a Sonora model in a driver (TOML) run
#
# ## Sonora or PICASO?
#
# PICASO is the name of an open-source atmosphere modeling *code*, with legacy in the Fortran EGP code ([Marley et al. 1999](https://ui.adsabs.harvard.edu/abs/1999ApJ...513..879M/abstract)). **Sonora** is the name of the *models* themselves (pressure-temperature profiles, spectra, ...), regardless of the code used to make them: Bobcat and Diamondback were computed with the legacy EGP Fortran code, while Elf Owl and Flame Skimmer were computed with PICASO.
#
# ## Which Sonora model should I use?
#
# This guidance follows the poster "Which Sonora model should I use?" (C. Morley et al; Exoclimes VI).
#
# | Grid | Paper | Teff (K) | Clouds | Chemistry | Evolution models | Use for |
# |---|---|---|---|---|---|---|
# | **Bobcat** | [Marley et al. 2021](https://ui.adsabs.harvard.edu/abs/2021ApJ...920...85M/abstract) | 200-2400 | no | equilibrium | yes | cloud-free objects where disequilibrium chemistry isn't important, or when you need self-consistent evolution models |
# | **Diamondback** | [Morley et al. 2024](https://ui.adsabs.harvard.edu/abs/2024ApJ...975...59M/abstract) | 900-2400 | yes (Al2O3, Fe, Mg2SiO4, MgSiO3; fsed=1-8) | equilibrium | yes, including the clouds | cloudy L and T dwarfs where disequilibrium chemistry isn't important |
# | **Elf Owl v1** | [Mukherjee et al. 2024](https://ui.adsabs.harvard.edu/abs/2024ApJ...963...73M/abstract) | 275-2400 | no | disequilibrium (Kzz) | no | cloud-free objects where disequilibrium chemistry is important. Use Bobcat or Diamondback evolution with Elf Owl spectra |
# | **Elf Owl v2** | [Wogan et al. 2025](https://ui.adsabs.harvard.edu/abs/2025RNAAS...9..108W/abstract) | 275-2400 | no | disequilibrium (Kzz) | no | save as above but with corrected CO2 quencing |
# | **Flame Skimmer** | [Mang et al. 2026](https://ui.adsabs.harvard.edu/abs/2026ApJ..1008..254M/abstract) | 50-2400 | no | equilibrium and disequilibrium | yes | the coldest objects (50-500 K) and low gravities (log g = 2-3.5), e.g. cold directly imaged planets |
#
# A few notes from the poster:
#
# * **Bobcat** is the original Sonora grid. It describes the overall methodology, opacities and chemistry, so please consider citing it whatever grid you choose. It has metallicities of -0.5, 0.0 and +0.5 and C/O of 0.5, 1.0 and 1.5x solar (the profiles that `picaso-data` downloads are the solar ones).
# * **Diamondback** is the first Sonora grid with clouds ([Ackerman & Marley 2001](https://ui.adsabs.harvard.edu/abs/2001ApJ...556..872A/abstract) framework) and includes metallicities of -0.5, 0.0 and +0.5. Its cloudy evolution models give a hydrogen-burning minimum mass of 70.2 Jupiter masses, close to the empirical limit of 70 MJ.
# * **Elf Owl** has more than 10,000 models with [M/H] from -1 to +1 and C/O from 0.5 to 2.5x solar, with disequilibrium chemistry coupled to the radiative-convective calculation. A key result is that brown dwarf atmospheres mix less efficiently than previously thought (Kzz ~ 10$^1$-10$^4$ cm$^2$/s), because the chemistry quenches in their radiative layers. Version 2 of the grid corrects the quenched CO$_2$ abundances and removes PH$_3$; use v2.
# * **Flame Skimmer** extends the framework to colder temperatures (50-250 K) and lower gravities. It is cloud-free; its cloudy companion **Gila Monster** (Mang et al., in prep.) will add clouds, including water clouds.
# * **Cholla** ([Karalidi et al. 2021](https://ui.adsabs.harvard.edu/abs/2021ApJ...923..269K/abstract)) was the first grid with disequilibrium chemistry coupled to the radiative-convective calculation. It has been replaced by Elf Owl, which covers a broader range of metallicities, C/O ratios and temperatures, so PICASO does not load it.

# %%
import os
import numpy as np
import pandas as pd
import astropy.units as u

from picaso import justdoit as jdi
from picaso import justplotit as jpi
from picaso import data

# %% [markdown]
# ## Downloading the grids
#
# The full grids are big, so download only what you need. Each line below is a `picaso.data.get_data` target in the `sonora_grids` category; the size is the download size. The same downloads are available from the command line (`picaso-data`) and from the "Setup reference data" page of `picaso-app`.

# %%
_, data_config = data.get_data_config()
pd.DataFrame([{'target': name, 'size (GB)': info['size_gb'], 'files': len(info['url']), 'description': info['description']}
              for name, info in data_config['sonora_grids'].items()])

# %% [markdown]
# For this tutorial you need:
#
# * `bobcat` (profiles, 0.002 GB). The published Bobcat spectrum used below already comes with this tutorial (`sp_t900g100nc_m0.0`); `bobcat-spectra` has all of them.
# * one Elf Owl file, e.g. `elfowl-Ytype-v2` (38 GB, 275-550 K)
# * `diamondback` (profiles and clouds, 0.3 GB) and `diamondback-spectra` (8.6 GB) for the published spectra
# * `flameskimmer-quickstart` (8.7 GB), a sparse version of the full Flame Skimmer grid
#
# By default each grid goes in its own folder in `$picaso_refdata/sonora_grids`:
#
# ```python
# data.get_data(category_download='sonora_grids', target_download='bobcat')
# data.get_data(category_download='sonora_grids', target_download='elfowl-Ytype-v2')
# data.get_data(category_download='sonora_grids', target_download='diamondback')
# data.get_data(category_download='sonora_grids', target_download='diamondback-spectra')
# data.get_data(category_download='sonora_grids', target_download='flameskimmer-quickstart')
# ```
#
# If you keep the grids somewhere else, point `sonora_grids` to that folder. `inputs.sonora` looks for `<sonora_path>/<grid>`, so it can be the folder that holds one folder per grid, or the folder of a single grid.

# %%
sonora_grids = os.path.join(os.getenv('picaso_refdata'), 'sonora_grids')

# %% [markdown]
# ## Setting up
#
# A brown dwarf has no star (`calculation='browndwarf'`), and the gravity you set is used to pick the grid model. We will compute every spectrum with the default opacity database (`full_output=True`, so that we can plot the atmosphere PICASO used) and compare it with the published spectrum at R=200. All the figures are made with `justplotit` and its interactive `plotly` backend (see the Plotting Backends tutorial).
#
# PICASO only includes the molecules that are in your opacity database; the others in the model are ignored. Check which of the main brown dwarf absorbers are missing (see the end of this tutorial for what that does to the spectra):

# %%
opa = jdi.opannection(wave_range=[0.6, 15])

def new_brown_dwarf(gravity):
    """Brown dwarf with gravity in m/s2 (the grids are labeled in m/s2)"""
    bd = jdi.inputs(calculation='browndwarf')
    bd.gravity(gravity=gravity, gravity_unit=u.Unit('m/s**2'))
    return bd

def picaso_vs_published(runs, R=200, ymin=None, **kwargs):
    """
    Plots PICASO (solid) and published (dashed) spectra in erg/cm2/s/cm at resolution R.
    runs : list of (label, output of bd.spectrum(..., full_output=True), published spectrum file)
    """
    xs, ys, legend, palette = [], [], [], []
    for i, (label, out, published) in enumerate(runs):
        x, y = jdi.mean_regrid(out['wavenumber'], out['thermal'], R=R)
        wave_pub, flux_pub = jdi.read_sonora_spectrum(published)
        _, y_pub = jdi.mean_regrid(1e4/wave_pub, flux_pub, newx=x)
        xs += [x, x]; ys += [y, y_pub]
        legend += [f'PICASO {label}', f'published {label}']
        palette += [jpi.Colorblind8[i]]*2
    if ymin is not None: kwargs['y_range'] = [ymin, 2*np.nanmax(np.concatenate(ys))]
    fig = jpi.spectrum(xs, ys, legend=legend, palette=palette, backend='plotly', y_axis_type='log',
                       y_axis_label='Flux [erg/cm²/s/cm]', **kwargs)
    for trace in fig.data[1::2]: trace.line.dash = 'dash'
    return fig

# %% [markdown]
# ## 1. Sonora Bobcat
#
# Bobcat profile files only hold the pressure-temperature profile, so `sonora` adds chemistry for you. The default (`chem='low'`) uses a precomputed solar-composition chemical equilibrium table; `chem='grid'` computes chemical equilibrium on the fly. `sonora` finds the model nearest to your Teff and gravity.

# %%
bobcat = new_brown_dwarf(gravity=100)
bobcat.sonora(sonora_grids, teff=900, grid='bobcat')
bobcat_out = bobcat.spectrum(opa, calculation='thermal', full_output=True)
bobcat.inputs['atmosphere']['sonora_filename']

# %%
jpi.pt(bobcat_out['full_output'], backend='plotly', title='Bobcat, Teff=900 K, g=100 m/s²')

# %%
jpi.mixing_ratio(bobcat_out['full_output'], molecules=['H2O', 'CH4', 'CO', 'NH3'], backend='plotly')

# %% [markdown]
# The published Bobcat spectra are in the `bobcat-spectra` download. `jdi.find_sonora_spectrum` finds the published spectrum of the model that was loaded; if you did not download them, we use the one that comes with this tutorial.

# %%
bobcat_model = os.path.join(sonora_grids, 'bobcat', bobcat.inputs['atmosphere']['sonora_filename'])
bobcat_published = jdi.find_sonora_spectrum(bobcat_model) or 'sp_t900g100nc_m0.0'
picaso_vs_published([('Bobcat 900 K', bobcat_out, bobcat_published)],
                    title='Sonora Bobcat, Teff=900 K, g=100 m/s²')

# %% [markdown]
# ## 2. Sonora Elf Owl
#
# Elf Owl models save the full chemistry (with disequilibrium quenching) with the profile, and `sonora` uses it by default (`chem='model'`). Elf Owl only has disequilibrium chemistry models, so you also choose the vertical mixing, `logkzz` (log10 Kzz in cm$^2$/s). Metallicity (`log_mh`, log10 relative to solar) and C/O (`cto_relative`, relative to solar = 0.458) are optional.
#
# `jdi.sonora_grid_options` shows which models you have on disk:

# %%
elfowl_models = jdi.sonora_grid_options(os.path.join(sonora_grids, 'elfowl'))
print(len(elfowl_models), 'Elf Owl models')
elfowl_models.drop(columns='filename').describe().loc[['min', 'max']]

# %%
elfowl = new_brown_dwarf(gravity=1000)
elfowl.sonora(sonora_grids, teff=275, grid='elfowl', logkzz=2, log_mh=0.0, cto_relative=1.0)
elfowl_out = elfowl.spectrum(opa, calculation='thermal', full_output=True)
elfowl.inputs['atmosphere']['sonora_params']

# %%
jpi.pt(elfowl_out['full_output'], backend='plotly', title='Elf Owl, Teff=275 K, g=1000 m/s², log Kzz=2')

# %%
jpi.mixing_ratio(elfowl_out['full_output'], molecules=['H2O', 'CH4', 'CO', 'CO2', 'NH3', 'PH3'], backend='plotly')

# %% [markdown]
# For Elf Owl (and Flame Skimmer) the published spectrum is saved in the same file as the profile:

# %%
elfowl_published = jdi.find_sonora_spectrum(elfowl_models.loc[
    (elfowl_models['filename'].str.endswith(elfowl.inputs['atmosphere']['sonora_filename'])), 'filename'].values[0])
picaso_vs_published([('Elf Owl 275 K', elfowl_out, elfowl_published)], ymin=1e3,
                    title='Sonora Elf Owl, Teff=275 K, g=1000 m/s², log Kzz=2')

# %% [markdown]
# ## 3. Sonora Diamondback
#
# Diamondback models are cloudy. Choose the sedimentation efficiency with `fsed` (1, 2, 3, 4 or 8; smaller fsed means thicker, more extended clouds), or leave `fsed=None` for the cloud-free models. For a cloudy model `sonora` also loads the cloud optical properties (optical depth, single scattering albedo and asymmetry, from the `cloud_optical_properties` files), so the PICASO spectrum includes the cloud. Diamondback profiles also save the chemistry of the main gases, which `sonora` uses by default.

# %%
diamondback = new_brown_dwarf(gravity=100)
diamondback.sonora(sonora_grids, teff=1500, grid='diamondback', fsed=1, log_mh=0.0)
diamondback_out = diamondback.spectrum(opa, calculation='thermal', full_output=True)
diamondback.inputs['atmosphere']['sonora_filename']

# %%
jpi.pt(diamondback_out['full_output'], backend='plotly', title='Diamondback, Teff=1500 K, g=100 m/s², fsed=1')

# %%
jpi.mixing_ratio(diamondback_out['full_output'], molecules=['H2O', 'CH4', 'CO', 'Na', 'K'], backend='plotly')

# %% [markdown]
# `jpi.cloud` shows the cloud that `sonora` loaded: the single scattering albedo, the optical depth of each layer and the asymmetry parameter, as a function of wavelength and pressure.

# %%
jpi.cloud(diamondback_out['full_output'], backend='plotly')

# %% [markdown]
# The published Diamondback spectra are the `.spec` files in the `diamondback-spectra` download. Below we compare the cloud-free model with fsed=1 and fsed=3.

# %%
runs = []
for fsed in [None, 3, 1]:
    bd = new_brown_dwarf(gravity=100)
    bd.sonora(sonora_grids, teff=1500, grid='diamondback', fsed=fsed)
    model = os.path.join(sonora_grids, 'diamondback', bd.inputs['atmosphere']['sonora_filename'])
    published = jdi.find_sonora_spectrum(model, os.path.join(sonora_grids, 'diamondback'))
    if published is None:
        print('Download diamondback-spectra to compare with the published spectrum')
        continue
    out = bd.spectrum(opa, calculation='thermal', full_output=True)
    runs.append(('cloud-free' if fsed is None else f'fsed={fsed}', out, published))
picaso_vs_published(runs, x_range=[0.8, 5], ymin=1e10, title='Sonora Diamondback, Teff=1500 K, g=100 m/s²')

# %% [markdown]
# ## 4. Sonora Flame Skimmer
#
# Flame Skimmer has both chemical equilibrium and disequilibrium models. Leave `logkzz=None` for chemical equilibrium, or set `logkzz` for disequilibrium. The `flameskimmer-quickstart` download is a sparse version of the full grid at R=3000: Teff = 50-2400 K, log g = 2-5.5, [M/H] = -1 to +2, C/O = 0.5-2.5x solar and log Kzz = 2, 4, 7, 8, 9. The full grid (R=30,000) is split into the `flameskimmer-eq` and `flameskimmer-deq-*` downloads; if you have both, `sonora` uses the higher resolution file.
#
# At 300 K, faster mixing (larger Kzz) brings up CO and CO$_2$ from the deep atmosphere and lowers NH$_3$, which changes the 4-5 $\mu$m spectrum:

# %%
flameskimmer = {}
for logkzz in [None, 2, 7]:
    bd = new_brown_dwarf(gravity=100)
    bd.sonora(sonora_grids, teff=300, grid='flameskimmer', logkzz=logkzz, log_mh=0.0, cto_relative=1.0)
    label = 'equilibrium' if logkzz is None else f'log Kzz={logkzz}'
    flameskimmer[label] = (bd, bd.spectrum(opa, calculation='thermal', full_output=True))

# %% [markdown]
# `jpi.pt` and `jpi.mixing_ratio` return plotly figures, so we can combine the three models into one figure of each:

# %%
pt_fig = None
for label, (bd, out) in flameskimmer.items():
    fig = jpi.pt(out['full_output'], backend='plotly', title='Flame Skimmer, Teff=300 K, g=100 m/s²')
    fig.update_traces(name=label, showlegend=True)
    if pt_fig is None: pt_fig = fig
    else: pt_fig.add_traces(fig.data)
pt_fig

# %%
chem_fig = None
for (label, (bd, out)), dash in zip(flameskimmer.items(), ['solid', 'dash', 'dot']):
    fig = jpi.mixing_ratio(out['full_output'], molecules=['CO', 'CO2', 'NH3'], backend='plotly',
                           x_range=[1e-12, 1e-3])
    for trace in fig.data: trace.update(name=f'{trace.name} {label}', line_dash=dash)
    if chem_fig is None: chem_fig = fig
    else: chem_fig.add_traces(fig.data)
chem_fig

# %%
models = jdi.sonora_grid_options(os.path.join(sonora_grids, 'flameskimmer'))
runs = []
for label in ['equilibrium', 'log Kzz=7']:
    bd, out = flameskimmer[label]
    model = models.loc[models['filename'].str.endswith(bd.inputs['atmosphere']['sonora_filename']), 'filename'].values[0]
    runs.append((label, out, jdi.find_sonora_spectrum(model)))
picaso_vs_published(runs, x_range=[1, 15], ymin=1e5, title='Sonora Flame Skimmer, Teff=300 K, g=100 m/s²')

# %% [markdown]
# ## Why doesn't the PICASO spectrum match the published one exactly?
#
# The spectra above use the same atmospheric structure as the published models, but they are not identical calculations:
#
# * **Missing molecules.** The largest differences come from molecules that are not in your opacity database. The default database (`opacities_0.3_15_R15000.db`, v3.3) does not include NH$_3$, PH$_3$, FeH, CrH, TiO or VO, so the PICASO spectra above miss the NH$_3$ bands at 8-15 $\mu$m (very clear in the cold Elf Owl and Flame Skimmer models, and the 10.5 $\mu$m band of Bobcat) and the FeH/CrH absorption near 1 $\mu$m in warmer objects. For these, use a database that includes them, e.g. `data.get_data('resampled_opacity', 'R20000,4.8-15um')` (7 GB) or `'R60000,0.6-6um'` (38 GB), and pass it with `jdi.opannection(filename_db=...)`.
# * **Opacity versions and resolution.** The published models used the opacities and code version of their release (EGP for Bobcat and Diamondback), while the spectra above use your PICASO version and its opacities resampled at R=15,000.
#
# Differences of tens of percent in individual bands are typical even with the same molecules, and are largest where the flux is small. If you want the exact published spectrum, read it with `jdi.read_sonora_spectrum`.
#
# Where PICASO is useful is in changing the model: e.g. adding or removing a molecule, changing the clouds, or using the profile as the starting guess of a climate calculation.
#
# ## Using a Sonora model in a driver (TOML) run
#
# In a driver file, `temperature.profile = 'sonora'` loads a grid model (the gravity is taken from `[object]`). `chemistry.method = 'sonora'` keeps the model's own abundances; any other chemistry method replaces them. A Diamondback cloud is kept unless your file also has a `[clouds]` section.
#
# ```toml
# [temperature]
# profile = 'sonora'
#
# [temperature.sonora]
# sonora_path = '_default_/sonora_grids'
# grid = 'diamondback'
# teff = 1500.0
# log_mh = 0.0
# cto_relative = 1.0
# logkzz = '' # elfowl, flameskimmer
# fsed = 1    # diamondback; '' for cloud-free
# chem = ''   # '' uses the model's abundances ('low' for bobcat)
#
# [chemistry]
# method = 'sonora'
# ```
#
# In a climate run (`go.run_climate`), `temperature.profile = 'sonora'` sets the initial guess of the temperature profile.
#
# ## Citing the Sonora models
#
# If you use these models, please cite the paper of the grid you used, and consider citing [Marley et al. 2021](https://ui.adsabs.harvard.edu/abs/2021ApJ...920...85M/abstract) (Bobcat), which describes the overall methodology:
#
# * Bobcat: [Marley et al. 2021](https://ui.adsabs.harvard.edu/abs/2021ApJ...920...85M/abstract)
# * Diamondback: [Morley et al. 2024](https://ui.adsabs.harvard.edu/abs/2024ApJ...975...59M/abstract)
# * Elf Owl: [Mukherjee et al. 2024](https://ui.adsabs.harvard.edu/abs/2024ApJ...963...73M/abstract) (and the v2 CO$_2$ correction research note by [Wogan et al. 2025](https://ui.adsabs.harvard.edu/abs/2025RNAAS...9..108W/abstract))
# * Flame Skimmer: [Mang et al. 2026](https://ui.adsabs.harvard.edu/abs/2026ApJ..1008..254M/abstract)

# %%
