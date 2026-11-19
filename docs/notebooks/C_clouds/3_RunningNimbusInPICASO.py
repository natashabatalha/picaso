# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
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
# # Running Nimbus in PICASO
#
# Here we will explore a different cloud model. This is [Nimbus](https://nimbus-kiefersv.readthedocs.io/en/latest/), an open-source efficient microphysical cloud model recently developed in [Kiefer et al. 2026](https://iopscience.iop.org/article/10.3847/1538-4357/ae5101). In this notebook we will run Nimbus through PICASO and compare it to the Virga cloud profile. They will use the same Jupiter pressure–temperature profile, constant mixing, and H2O cloud, then replace Virga with Nimbus and compare their reflected-light spectra and cloud optical properties.
#
# **Virga vs Nimubs?**
#
# In short: Virga assumes the condensing gas is in phase equilibrium and uses a log-normal particle-size distribution set by `sig` and `fsed`, while Nimbus computes nucleation and accretion rates with single-size particles, which costs ~10–20 s per run compared with ~1 s for Virga ([Kiefer et al. 2026](https://arxiv.org/abs/2603.13167), Sec. II). So in short Nimbus offers more physics at a computational cost of around ~10x Virga.
#
# What you will learn:
#
# 1. How to run an efficient microphysical cloud model through PICASO
#
# What you should already know:
#
# 1. How to run `Virga` through PICASO via [this tutorial](https://natashabatalha.github.io/picaso/notebooks/C_clouds/1_PairingPICASOToVIRGA.html).
#
# **Optional dependencies that need to be installed**
#
#
# You need to install [Nimbus](https://nimbus-kiefersv.readthedocs.io/en/latest/Install_And_Quick_Start.html) and [MieNet](https://mienet.readthedocs.io/en/latest/Install_And_Quick_Start.html) for this tutorial. It is not included as part of picaso regular isntall. 
#
# Nimbus can be installed through pip, as an optional dependency of picaso
#
# `pip install "picaso[nimbus]"` (or directly with `pip install nimbus-exo`)
#
# And MieNet through git 
#
# `pip install git+https://github.com/MieNet-org/mienet.git`
#
#

# %%
import astropy.units as u
import numpy as np
import pandas as pd

from picaso import justdoit as jdi
from picaso import justplotit as jpi
import os

from bokeh.plotting import show
from bokeh.io import output_notebook
output_notebook()


# %% [markdown]
# ## Set up the same atmosphere
#
# We include the visible wavelengths from the Virga tutorial and extend the
# range through 4 µm for the optical-property comparison. Both models use the
# same stellar spectrum and PICASO reflected-light setup.

# %%
opacity = jdi.opannection(wave_range=[0.3, 15.0])
atmosphere = pd.read_csv(jdi.jupiter_pt(), sep=r"\s+")
atmosphere["kz"] = 1e9  # cm²/s, at every pressure level

metallicity = 1  # relative to Solar
mean_molecular_weight = 2.2

# note that Nimbus does not need an fsed since it is a microphysical cloud model and not a parametertized cloud model. Here we set the fsed for the Virga cloud comparison later.
fsed = 1

# %% [markdown]
# ## Submit a H2O cloud Nimbus run through PICASO
#
# Nimbus needs the deep H2O **mass** mixing ratio (MMR) explicitly, which is related to the number mixing ratio (VMR):
# $$ MMR = VMR ~\frac{\mu_\mathrm{H_2O}}{\mu_\mathrm{MMW}}$$
# The value below is a Solar-composition estimate (H2O number mixing ratio 7.54e-4 converted with molecular weights 18 and 2.34). This is taken directly from Virga. The returned dictionary contains layer optical depth, single-scattering albedo, asymmetry, pressure, and wavelength.

# %%
nimbus_planet = jdi.inputs()
nimbus_planet.phase_angle(0)
nimbus_planet.gravity(gravity=25, gravity_unit=u.m / u.s**2)
nimbus_planet.star(opacity, 5000, 0, 4.0)
nimbus_planet.atmosphere(df=atmosphere.copy())

solar_h2o_mmr = 7.54e-4 * 18.0 / 2.34
nimbus_cloud = nimbus_planet.nimbus(
    ["H2O"], mh=metallicity, mmw=mean_molecular_weight,
    gas_mmr={"H2O": solar_h2o_mmr}, verbose=True,
)
nimbus_spectrum = nimbus_planet.spectrum(opacity, calculation="reflected", full_output=True)


# %% [markdown]
# ### Inspect the Nimbus cloud at 4 µm
#
# The optical depth shown is the optical depth of each layer, rather than the cumulative optical depth from the top of the atmosphere. The other two quantities are the single-scattering albedo (ω₀) and asymmetry factor (g). `jpi.all_optics_1d` averages each quantity over the wavelength range we give it, here a narrow window around 4 µm.

# %%
wave_range = [3.9, 4.1]  # µm, window around 4 µm

show(jpi.all_optics_1d(nimbus_spectrum["full_output"], wave_range=wave_range))


# %% [markdown]
# ### Run the Virga H2O cloud from the previous tutorial
#
# This uses its original `fsed`, metallicity, mean molecular weight, and PICASO Virga reference directory just so that we can plot up some comparisons between the two models.

# %%
virga_planet = jdi.inputs()
virga_planet.phase_angle(0)
virga_planet.gravity(gravity=25, gravity_unit=u.m / u.s**2)
virga_planet.star(opacity, 5000, 0, 4.0)
virga_planet.atmosphere(df=atmosphere.copy())

virga_directory = os.path.join(os.environ["picaso_refdata"], "virga")
virga_cloud = virga_planet.virga(
    ["H2O"], virga_directory, fsed=fsed, mh=metallicity,
    mmw=mean_molecular_weight, verbose=False,
)
virga_spectrum = virga_planet.spectrum(opacity, calculation="reflected", full_output=True)


# %% [markdown]
# ### Compare reflected-light spectra
#
# Regrid both spectra to the same resolving power, as in the Virga tutorial. Any difference now comes from the cloud calculations, since the atmosphere, star, gravity, and PICASO radiative-transfer settings are the same.

# %%
wavenumbers, albedos = [], []
for spectrum in (nimbus_spectrum, virga_spectrum):
    wavenumber, albedo = jdi.mean_regrid(
        spectrum["wavenumber"], spectrum["albedo"], R=150
    )
    wavenumbers.append(wavenumber)
    albedos.append(albedo)

show(jpi.spectrum(wavenumbers, albedos, legend=["Nimbus", "Virga"],
                  plot_width=600, plot_height=400,
                  y_axis_label="Geometric albedo", x_range=[0.3, 5]))


# %% [markdown]
# ### Compare cloud optical properties at 4 µm

# %%
show(jpi.all_optics_1d(
    [nimbus_spectrum["full_output"], virga_spectrum["full_output"]],
    wave_range=wave_range, legend=["Nimbus", "Virga"],
))

# %% [markdown]
# ## Use Nimbus outside PICASO for more control
#
# Nimbus offeres additional flexibility and control over the microphysics and numerical set-up which are not available when run through PICASO. Here we show an example of how an external Nimbus run can be integrated into PICASO. For further infromation on how to use Nimbus see the [documentation](https://nimbus-kiefersv.readthedocs.io/en/latest/). 

# %%
from nimbus import Nimbus

# define temperature [K] and pressure [bar] structure
temperature = np.asarray([554, 572, 607, 653, 775, 951, 1073, 1111, 1540, 2654, 3000])
pressure = np.asarray([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4])

# set up Nimbus object
obj = Nimbus()
obj.set_up_atmosphere(
    temperature = temperature,
    pressure = pressure,
    kzz = np.ones_like(pressure) * 1e9,
    mmw = 2.34,
    gravity = 10**2.49,
    species = 'SiO',
    deep_mmr = 10**-3,
)
obj.set_up_solver()

# compute the cloud structure
obj.compute()

# get optical properties
df_nimbus = obj.picaso_formater()

df_nimbus

# %% [markdown]
# And now you have a dataframe that is ready to use when generating a PICASO spectrum as you would with any other Virga or standalone cloud profile to be processed!

