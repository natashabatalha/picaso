# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Running Nimbus in PICASO
#
# Here we will explore a different cloud model. This is [Nimbus](https://nimbus-kiefersv.readthedocs.io/en/latest/), an open-source efficient microphysical cloud model recently developed in [Kiefer et al. 2026](https://iopscience.iop.org/article/10.3847/1538-4357/ae5101). In this notebook we will run Nimbus through PICASO and compare it to the Virga cloud profile. They will use the same Jupiter pressure–temperature profile, constant mixing, and H2O cloud, then replace Virga with Nimbus and compare their reflected-light spectra and cloud optical properties.
#
# What you will learn:
#
# 1. How to run an efficient microphysical cloud model through PICASO
#
# What you should already know:
#
# 1. How to run `Virga` through PICASO via [this tutorial](https://natashabatalha.github.io/picaso/notebooks/C_Clouds/1_PairingPICASOToVIRGA.html).
#
# Note: you should have [Nimbus](https://nimbus-kiefersv.readthedocs.io/en/latest/Install_And_Quick_Start.html) and [MieNet](https://mienet.readthedocs.io/en/latest/Install_And_Quick_Start.html) already installed in your environment for this tutorial to run

# %%
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from picaso import justdoit as jdi


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
    gas_mmr={"H2O": solar_h2o_mmr}, verbose=False,
)
nimbus_spectrum = nimbus_planet.spectrum(opacity, calculation="reflected")


# %% [markdown]
# ## Inspect the Nimbus cloud at 4 µm
#
# `opd_per_layer` is the optical depth of each layer, rather than the cumulative optical depth from the top of the atmosphere. The other two quantities are the single-scattering albedo (ω₀) and asymmetry factor (g). We select the nearest wavelength in Nimbus's cloud grid just as part of a visualization.

# %%
target_wave = 4.0  # µm

# find the index for the 4µm optical properties within Nimbus
def cloud_column(cloud, wavelength):
    wave = np.asarray(cloud["wave"])
    index = np.argmin(np.abs(wave - wavelength))
    return index, float(wave[index])


nimbus_index, nimbus_wave = cloud_column(nimbus_cloud, target_wave)
print(f"Nimbus cloud wavelength nearest 4 µm: {nimbus_wave:.3f} µm")

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for ax, key, label in zip(
    axes,
    ("opd_per_layer", "single_scattering", "asymmetry"),
    ("Optical depth", "Single-scattering albedo", "Asymmetry"),
):
    ax.plot(np.asarray(nimbus_cloud[key])[:, nimbus_index], nimbus_cloud["pressure"])
    ax.set_xlabel(label)
    ax.grid(alpha=0.3)
axes[0].set_xscale("symlog", linthresh=1e-4)
axes[0].set_ylabel("Pressure [bar]")
axes[0].set_yscale("log")
axes[0].invert_yaxis()
fig.suptitle(f"Nimbus H2O cloud at {nimbus_wave:.3f} µm")
fig.tight_layout()
plt.show()


# %% [markdown]
# ## Run the Virga H2O cloud from the previous tutorial
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
virga_spectrum = virga_planet.spectrum(opacity, calculation="reflected")


# %% [markdown]
# ## Compare reflected-light spectra
#
# Regrid both spectra to the same resolving power, as in the Virga tutorial. Any difference now comes from the cloud calculations, since the atmosphere, star, gravity, and PICASO radiative-transfer settings are the same.

# %%
fig, ax = plt.subplots(figsize=(8, 6))
for spectrum, label in ((nimbus_spectrum, "Nimbus"), (virga_spectrum, "Virga")):
    wavenumber, albedo = jdi.mean_regrid(
        spectrum["wavenumber"], spectrum["albedo"], R=150
    )
    ax.plot(1e4 / wavenumber, albedo, label=label)
ax.set(xlabel="Wavelength [µm]", ylabel="Geometric albedo", xlim=(0.3, 5))
ax.legend()
fig.tight_layout()
plt.show()


# %% [markdown]
# ## Compare cloud optical properties at 4 µm.

# %%
virga_index, virga_wave = cloud_column(virga_cloud, target_wave)
print(f"Virga cloud wavelength nearest 4 µm: {virga_wave:.3f} µm")

fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
for ax, key, label in zip(
    axes,
    ("opd_per_layer", "single_scattering", "asymmetry"),
    ("Optical depth", "Single-scattering albedo", "Asymmetry"),
):
    for cloud, index, name in (
        (nimbus_cloud, nimbus_index, "Nimbus"),
        (virga_cloud, virga_index, "Virga"),
    ):
        ax.plot(np.asarray(cloud[key])[:, index], cloud["pressure"], label=name)
    ax.set_xlabel(label)
    ax.grid(alpha=0.3)
axes[0].set_xscale("symlog", linthresh=1e-4)
axes[0].set_ylabel("Pressure [bar]")
axes[0].set_yscale("log")
axes[0].invert_yaxis()
axes[0].legend()
fig.suptitle("H2O cloud optical properties near 4 µm")
fig.tight_layout()
plt.show()

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

