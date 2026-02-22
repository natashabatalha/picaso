# ---
# jupyter:
#   jupytext:
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
# # Approximations for Spherical Harmonics Radiative Transfer in Reflected Light
#
# In [Rooney et al 2023](add-link) we rigorously derive the spherical harmonics method for reflected light and benchmark the 4-term method (SH4) against [Toon et al. 1989](https://ui.adsabs.harvard.edu/abs/1989JGR....9416287T/abstract) and two independent methods. Here, we provide the code to reproduce the analysis that compares Toon89 with the higher fidelity 4-term spherical harmonics method for reflected light calculations.

# %%
import numpy as np
import pandas as pd

#picaso
from picaso import justdoit as jdi
from picaso import justplotit as jpi
import picaso.model_compare as ptest

jpi.output_notebook()

# %% [markdown]
# ## Benchmark Jupiter-like case profile
#
# Here we use the same profile explored in the two-stream radiative transfer tutorial.

# %%
opa = jdi.opannection(wave_range=[0.3,1])#, resample=100)
case1 = jdi.inputs()
#phase
case1.phase_angle(0) #radians

#gravity
case1.gravity(gravity = 25, gravity_unit=jdi.u.Unit('m/(s**2)'))

#star
case1.star(opa, 6000,0.0122,4.437) #kelvin, log metal, log cgs

#atmosphere
case1.atmosphere(filename= jdi.jupiter_pt(), sep=r'\s+')

#set model clouds
case1.clouds( filename= jdi.jupiter_cld(), sep=r'\s+')

# %% [markdown]
# ## Setup comparison with two-stream radiative transfer

# %%
#we'll use these labels to keep track of the cases we have created
labels = []
albs = []
#run

multi_phase = 'N=2' #two legendre polynomials (toon default)
single_phase='TTHG_ray' #two term HG with rayleigh (toon default)
DE = True #delta eddington correction, (toon default)
raman='none' #lets turn raman off, for clarity in this benchmark case

#set all approximatinos
case1.approx(toon_coefficients='quadrature',
             multi_phase=multi_phase,
             single_phase=single_phase,
             delta_eddington=DE, raman=raman)
df = case1.spectrum(opa)
wno, alb, fpfs = df['wavenumber'] , df['albedo'] , df['fpfs_reflected']
wno, alb = jdi.mean_regrid(wno, alb, R=150)

labels+=['Toon89']
albs+=[alb]

# %% [markdown]
# ## Setting up Spherical Harmonics Approximations to compare with Toon
#
# Spherical harmonics allows us to remain consistent with scattering functions throughout the methodology. In Toon when calculation the two stream solution for multiple layers, the phase functions are hard-coded set to be one term HG. However, when implementing the source function technique to derive the outgoing intensity we introduce a two-term HG for direct-scattering beam in attempt to capture the back-scattering radiation observed on Neptune. These details are further discussed in Cahoy et al 2010, Batalha et al. 2019, Feng et al. 2017.
#
# In spherical harmonics, we can simply consider a two-term HG phase function throughout the calculation. In order to better compare with Toon however (for historical purposes), we made sure that our SH routines could also match that Toon approach. Therefore, below you see we have **two** different single scattering forms. Note that this is only for comparison with Toon89. The default for SH is to use a TTHG phase function with Rayleigh throughout the radiative transfer calculation.
#
# Further details can be found in Rooney et al. 2023.

# %%
# NOT SH default; enforcing for toon comparison which cannot handle
# pre-processed two term phase functions
w_single_form='OTHG'; w_single_rayleigh='off'
w_multi_form='OTHG'; w_multi_rayleigh='on'

#defaults in SH technique
psingle_form='TTHG'; psingle_rayleigh='on'
#2 or 4 stream (SH routines handles both though 4 is the default)
stream=4

case1.approx(rt_method='SH', stream=stream,
              w_single_form=w_single_form, w_single_rayleigh=w_single_rayleigh,
              w_multi_form=w_multi_form, w_multi_rayleigh=w_multi_rayleigh,
             psingle_form=psingle_form, psingle_rayleigh=psingle_rayleigh,
             delta_eddington=DE, raman=raman)
# case1.approx(rt_method='SH', stream=4, raman='none')
df3 = case1.spectrum(opa)
wno3, alb3, fpfs3 = df3['wavenumber'] , df3['albedo'] , df3['fpfs_reflected']
wno3, alb3 = jdi.mean_regrid(wno3, alb3, R=150)
labels+=['SH4 (OTHG multi)']
albs+=[alb3]

# %% [markdown]
# Here we relax the constraint for OTHG to see how the multile scattering approximation in Toon impacts the spectrum.

# %%
w_multi_form='TTHG'; w_multi_rayleigh='on'

case1.approx(rt_method='SH', stream=4,
              w_single_form=w_single_form, w_single_rayleigh=w_single_rayleigh,
              w_multi_form=w_multi_form, w_multi_rayleigh=w_multi_rayleigh,
             psingle_form=psingle_form, psingle_rayleigh=psingle_rayleigh,
             delta_eddington=DE, raman='none')
# case1.approx(rt_method='SH', stream=4, raman='none')
df5 = case1.spectrum(opa)
wno5, alb5, fpfs5 = df5['wavenumber'] , df5['albedo'] , df5['fpfs_reflected']
wno5, alb5 = jdi.mean_regrid(wno5, alb5, R=150)
labels+=['SH4 (TTHG multi)']
albs+=[alb5]

# %% [markdown]
# ## Reproducing Figure 5a from Rooney et al. 2023

# %%
jpi.show(jpi.spectrum([wno]*3, albs, labels, width=700))

# %% [markdown]
# ## Reproducing Figure 5b from Rooney et al. 2023
#
# Here we reproduce the same procedure from above, but with a different cloud setup.

# %%
#set model clouds, note these are lists since you can specify multiple cloud layers
case1.clouds( g0=[0.9], w0=[0.8], opd=[0.5], p = [0.0], dp=[1.0])  # Slab cloud from 1.0 bar up to 0.1 bar
labels = []
albs = []

# %%
multi_phase = 'N=2'
single_phase='TTHG_ray'
DE = True

case1.approx(multi_phase=multi_phase,single_phase=single_phase,
             delta_eddington=DE, raman='none')
df = case1.spectrum(opa)
wno, alb, fpfs = df['wavenumber'] , df['albedo'] , df['fpfs_reflected']
wno, alb = jdi.mean_regrid(wno, alb, R=150)
labels+=['Toon89']
albs+=[alb]

# %%
#run SH4
w_single_form='OTHG'; w_single_rayleigh='off'
w_multi_form='OTHG'; w_multi_rayleigh='on'
psingle_form='TTHG'; psingle_rayleigh='on'

case1.approx(rt_method='SH', stream=4,
              w_single_form=w_single_form, w_single_rayleigh=w_single_rayleigh,
              w_multi_form=w_multi_form, w_multi_rayleigh=w_multi_rayleigh,
             psingle_form=psingle_form, psingle_rayleigh=psingle_rayleigh,
             delta_eddington=DE, raman='none')
# case1.approx(rt_method='SH', stream=4, delta_eddington=DE, raman='none')
df2 = case1.spectrum(opa)
wno2, alb2, fpfs2 = df2['wavenumber'] , df2['albedo'] , df2['fpfs_reflected']
wno2, alb2 = jdi.mean_regrid(wno2, alb2, R=150)
labels+=['SH4 (OTHG single)']
albs+=[alb2]

# %%
#run SH2
w_multi_form='TTHG'; w_multi_rayleigh='on'

case1.approx(rt_method='SH', stream=4,
              w_single_form=w_single_form, w_single_rayleigh=w_single_rayleigh,
              w_multi_form=w_multi_form, w_multi_rayleigh=w_multi_rayleigh,
             psingle_form=psingle_form, psingle_rayleigh=psingle_rayleigh,
             delta_eddington=DE, raman='none')
df3 = case1.spectrum(opa)
wno3, alb3, fpfs3 = df3['wavenumber'] , df3['albedo'] , df3['fpfs_reflected']
wno3, alb3 = jdi.mean_regrid(wno3, alb3, R=150)
labels+=['SH4 (TTHG multi)']
albs+=[alb3]

# %%
jpi.show(jpi.spectrum([wno]*3, albs,labels, width=700))
