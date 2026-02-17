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
# # Approximations for Spherical Harmonics Radiative Transfer in Thermal Emission
#
# In [Rooney et al 2023](add-link) we rigorously derive the spherical harmonics method for thermal emission and benchmark the 2-term and 4-term method (SH4) against [Toon et al. 1989](https://ui.adsabs.harvard.edu/abs/1989JGR....9416287T/abstract) and CDISORT. Here, we provide the code to reproduce the analysis that compares Toon89 with the higher fidelity 4-term spherical harmonics method for thermal emission spectroscopy.
#
# Note that all comparisons with `CDISORT` are precomputed following Rooney et al's calculations, which used [V1 opacities](https://zenodo.org/record/3759675#.Y_aJROzMI8Y).

# %%
import numpy as np
import pandas as pd
import astropy.units as u

#picaso
from picaso import justdoit as jdi
from picaso import justplotit as jpi

jpi.output_notebook()

# %% [markdown]
#
# ## Setting up Brown Dwarf Comparison
#
# Within the PICASO repository there exists a simple benchmark brown dwarf case that we used in the paper to compare the code. We will start by setting that up.

# %%
wave_range = [.7,14]
opa = jdi.opannection(wave_range=wave_range)#, resample=100)
bd = jdi.inputs(calculation='browndwarf')

bd.phase_angle(0)
grav = 200
bd.gravity(gravity=grav , gravity_unit=u.Unit('m/s**2'))
bd.surface_reflect(0,opa.wno)

#brown dwarf PT and CLD provide a pressure-temperature profile and cloud profile
#from a standard brown dwarf case with Teff~1270 K and fsed = 1 courtesy of C. Morley @ UT Austin
bd.atmosphere(filename=jdi.brown_dwarf_pt(), sep=r'\s+')
bd.clouds(filename=jdi.brown_dwarf_cld(), sep=r'\s+')


# %% [markdown]
# ### Using the `jdi.approx` function to setup different test cases
#
# Similar to the reflected light problem, we can use the `approx` key to setup different methods of computing the thermal radiative transfer

# %%
dfs = []; labels = [];
# PICASO Original Methodology using Toon source function technique
dfs += [bd.spectrum(opa, full_output=True,
                    calculation='thermal')
       ]; labels += ["PICASO Toon89"]

# 2-term Spherical harmonics
bd.approx(rt_method = 'SH', stream=2)
two_lin = bd.spectrum(opa, full_output=True,
                      calculation='thermal')
dfs += [two_lin]; labels += ["SH2"]

# 4-term Spherical Harmonics
bd.approx(rt_method = 'SH', stream=4)
four_lin = bd.spectrum(opa, full_output=True,
                       calculation='thermal')
dfs += [four_lin]; labels += ["SH4"]


# %% [markdown]
# Simple regridding and plotting, as we normally do

# %%
xs = []; ys = []; labs = [];
for df, i in zip(dfs, range(len(dfs))):
    x,y = df['wavenumber'], df['thermal'] #units of erg/cm2/s/cm
    xflux,yflux = jdi.mean_regrid(x, y, R=150)
    xs += [xflux]
    ys += [yflux]
    labs += [labels[i]]

# %% [markdown]
# ### External Testing with `CDISROT` for Validation
#
# Reproduction of Figure 2b, Rooney et al.
#
# In Rooney et al., we tested `PICASO` against a higher order code, `CDISORT`. Here is a code snippet to grab it from the code base

# %%
# # read in my disort profile
disort_output = jdi.os.path.join(jdi.__refdata__, 'base_cases','testing','cdisort_output_1270_cloudy.spec')
son = pd.read_csv(disort_output,
                    sep=r'\s+', skiprows=2,header=None,names=['1','2','3'])
sonx, sony, flx =  np.array(1e4/son['1']), np.array(son['2']), np.array(son['3'])
sonx_,sony_ = jdi.mean_regrid(sonx, sony*1e1, newx=xs[0])
xs += [sonx_]; ys += [sony_]; labs += ['DISORT16']


# %%
fig = jpi.spectrum(xs,ys,legend=labs
                  ,plot_width=800,x_range=wave_range,x_axis_type='log')
jpi.show(fig)


# %% [markdown]
# ## Setting up Jupiter-like Case
#
# Within the PICASO repository there exists a simple benchmark code jupiter-like case that we used in the paper to compare the code. We will start by setting that up.

# %%
wave_range = [5,14]
opa = jdi.opannection(wave_range=wave_range)#, resample=100)
planet = jdi.inputs()

planet.phase_angle(0)
grav = 25
planet.gravity(gravity=grav , gravity_unit=u.Unit('m/s**2'))
# bd.surface_reflect(0,opa.wno)
planet.star(opa, 5000,0,4.0) #opacity db, pysynphot database, temp, metallicity, logg

planet.atmosphere(filename= jdi.jupiter_pt(), sep=r'\s+')

planet.clouds( filename= jdi.jupiter_cld(), sep=r'\s+')


# %% [markdown]
# ### Using the `jdi.approx` function to setup different test cases
#
# Similar to the reflected light problem, we can use the `approx` key to setup different methods of computing the thermal radiative transfer

# %%
dfs = []; labels = [];
# PICASO
dfs += [planet.spectrum(opa, full_output=True, calculation='thermal')
                        ]; labels += ["PICASO"]

# 2-stream
planet.approx(rt_method = 'SH', stream=2)
two_lin = planet.spectrum(opa, full_output=True, calculation='thermal')
dfs += [two_lin]; labels += ["SH2"]

# 4-stream
planet.approx(rt_method = 'SH', stream=4)
four_lin = planet.spectrum(opa, full_output=True, calculation='thermal')
dfs += [four_lin]; labels += ["SH4"]

# %%
xs = []; ys = []; labs = [];
for df, i in zip(dfs, range(len(dfs))):
    x,y = df['wavenumber'], df['thermal'] #units of erg/cm2/s/cm
    xflux,yflux = jdi.mean_regrid(x, y, R=150)
    xs += [xflux]
    ys += [yflux]
    labs += [labels[i]]

# %% [markdown]
# ### External Testing with `CDISROT` for Validation
#
# Reproduction of Figure bb, Rooney et al.
#
# In Rooney et al., we tested `PICASO` against a higher order code, `CDISORT`. Here is a code snippet to grab it from the code base

# %%
# # read in my disort profile
disort_output = jdi.os.path.join(jdi.__refdata__, 'base_cases','testing','cdisort_output_jupiter_cloudy.spec')
son = pd.read_csv(disort_output,
                    sep=r'\s+', skiprows=2,header=None,names=['1','2','3'])
sonx, sony, flx =  np.array(1e4/son['1']), np.array(son['2']), np.array(son['3'])
sonx_,sony_ = jdi.mean_regrid(sonx, sony*1e1, newx=xs[0])
xs += [sonx_]; ys += [sony_]; labs += ['DISORT16']

# %%
fig = jpi.spectrum(xs,ys,legend=labs
                  ,plot_width=800,x_range=wave_range,x_axis_type='log')
jpi.show(fig)

# %% [markdown]
# We can directly compare all of them. Interestingly enough, this figure shows better agreement between the methods (compared to the Brown Dwarf case). As we explain in Rooney, this is because the Toon89 methodology is better suited for scattering regimes in the limit of single scattering -> 1 and -> 0. We will explore this further below.

# %% [markdown]
# # Dependence of Radiative Transfer Method on Scattering Parameters
#
# As we alluded to above, there is a large accuracy dependence on single scattering and on single scattering and asymmetry. We can run `PICASO` Toon89 methodology, along with SH2 and SH4 and compare with a precomputed cdisort 32-stream calculation.
#
# For this we will use a pre-built test function included in `picaso.test` (this will take some time as it is computing many radiative transfer calculations.

# %%
import picaso.model_compare as ptest

# %%
# Toon quadrature
Toon_quad = ptest.thermal_sh_test(method="toon",
                                  tau=0.2)
# SH 2-term
SH2 = ptest.thermal_sh_test(method="SH",
                            stream=2, tau=0.2)

# SH 4-term
SH4 = ptest.thermal_sh_test(method='SH',
                            stream=4, tau=0.2)

# %% [markdown]
# Let's load the cdisort pre computed test file that we have in the `PICASO` reference files. Note that since this is precomputed, minor differences might occur based on what it is in the Rooney paper.

# %%
data_disort32 = jdi.pd.read_csv(jdi.os.path.join(jdi.__refdata__, 'base_cases','testing',
                                         'cdisort32str_1270K_tau02.csv'),index_col=0)

# %%
compare_picaso_disort32 = (data_disort32-Toon_quad)/data_disort32*100
compare_SH4_disort32 = (data_disort32-SH4)/data_disort32*100
compare_SH2_disort32 = (data_disort32-SH2)/data_disort32*100

# %% [markdown]
# ## Plot heatmap comparing radiative transfer methods
#
# Reproduce Figure 6 in Rooney et al. 2023 Part II Thermal.
#
# The efficacy of Toon large depends on the strength of the scattering. When single scattering approaches 1 or 0, Toon89 can be a very effective RT method. However, for intermediate values large errors (>10%) are visible when compared to DISORT 32 stream.

# %%
toon89_fig = jpi.rt_heatmap(
    compare_picaso_disort32.iloc[:-1,[0,1,3,5,6,7,8,9,10,11,12,13,14]],
    cmap_kwargs={'palette':jpi.pals.plasma(11)[::-1], 'low':-5,'high':60},
    figure_kwargs={'title':'Toon89'}
)

sh4_fig = jpi.rt_heatmap(
    compare_SH4_disort32.iloc[:-1,[0,1,3,5,6,7,8,9,10,11,12,13,14]],
    cmap_kwargs={'palette':jpi.pals.viridis(11), 'low':-6,'high':2},
    figure_kwargs={'title':'SH4'}
)

diff_fig = jpi.rt_heatmap((compare_picaso_disort32.iloc[:-1,[0,1,3,5,6,7,8,9,10,11,12,13,14]] -
                    compare_SH4_disort32.iloc[:-1,[0,1,3,5,6,7,8,9,10,11,12,13,14]]
                           ),
    cmap_kwargs={'palette':jpi.pals.RdBu[11], 'low':-60,'high':60},
    figure_kwargs={'title':'Diff between Toon89 and SH4'}
)

jpi.show(jpi.row([toon89_fig, sh4_fig, diff_fig]))

# %%
