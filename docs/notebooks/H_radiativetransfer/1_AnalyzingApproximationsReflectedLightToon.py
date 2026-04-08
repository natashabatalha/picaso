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
# # Approximations for Toon89 Two-Stream Radiative Transfer in Reflected Light
#
# Like any code, there are several approximations that go into computing intensity from various phase functions. In reflected light models, some of these approximations drastically change the output spectra.
#
# In this notebook you will:
#
# - learn how to use the `approx` method to access different ways of computing radiative transfer
# - focusing on Toon et al 1989 methodology, learn how each approximation affects reflected light spectrum
# - how to run run a benchmark test against reference data within the PICASO code (leverages data from [Dlugach & Yanovitskij (1974)](https://ui.adsabs.harvard.edu/abs/1974Icar...22...66D/abstract) to replicates the study of [Batalha et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019ApJ...878...70B/abstract) )

# %%
import warnings
warnings.filterwarnings('ignore')
#picaso
from picaso import justdoit as jdi
from picaso import justplotit as jpi
from bokeh.plotting import show, figure
from bokeh.layouts import column
from bokeh.palettes import Colorblind8
from bokeh.io import output_notebook
import astropy.units as u
output_notebook()

# %% [markdown]
# ## Using the `approx` key

# %%
opa = jdi.opannection(wave_range=[0.3,1])
cloud_free = jdi.inputs()

# %% [markdown]
# Notice that all the `approx` keys have predefined inputs. These are our recommendations for how to run the code. But, users should always be weary of these and test their sensitivity to your results.

# %% [markdown]
# ## How to see what radiative transfer scattering options exist?
#
# All of the approximations have to do with how scattering is handled.

# %%
print('Options for Direct Scattring Phase: ', jdi.single_phase_options())

print('Options for Multiple Scattring Phase: ', jdi.multi_phase_options())

print('Options for Raman Scattering: ', jdi.raman_options())

print('Options for Toon89 Coefficients: ', jdi.toon_phase_coefficients())

# %% [markdown]
# ### Set inputs normally

# %%
cloud_free.phase_angle(0) #phase in radians

cloud_free.gravity(gravity=25, gravity_unit=u.Unit('m/(s**2)' ))

#set star
cloud_free.star(opa, 6000, 0.0122, 4.437)

#set atmosphere comp and temp
cloud_free.atmosphere(filename=jdi.jupiter_pt(),sep=r'\s+')

#make a copy to have a separate cloud input dict
from copy import deepcopy
cloudy=deepcopy(cloud_free)
cloudy.clouds( filename=jdi.jupiter_cld(),sep=r'\s+')

# %% [markdown]
# ## Toon Coefficients
#
# Though the default is Quadrature, Table 1 of [Toon et al. 1989]() gives two options for scattering phase functions for solar, reflected light.

# %%
#let's make two different figures for this
fig_cloudy = figure(x_range=[0.3,1],y_range=[0,1.2], width=500, height=300
                    ,x_axis_label='Wavelength [μm]', y_axis_label='Geometric Albedo',)
fig_no_cloud = figure(x_range=[0.3,1],y_range=[0,1.2], width=500, height=300
                      ,x_axis_label='Wavelength [μm]', y_axis_label='Geometric Albedo',)

#define our options
options = jdi.toon_phase_coefficients()
colors = Colorblind8[0:len(options)]

#loop through all approximations
for approx, c in zip(options, colors):
    #set approximations
    cloud_free.approx(toon_coefficients = approx)
    cloudy.approx(toon_coefficients = approx)
    df = cloud_free.spectrum(opa)
    wno_nc, alb_nc = jdi.mean_regrid(df['wavenumber'] , df['albedo'],R=150)
    df = cloudy.spectrum(opa)
    wno_c, alb_c = jdi.mean_regrid(df['wavenumber'] , df['albedo'],R=150)
    fig_no_cloud.line(1e4/wno_nc, alb_nc, legend_label=approx, color=c, line_width=3)
    fig_cloudy.line(1e4/wno_c, alb_c,  color=c, line_width=3)
jpi.plot_format(fig_cloudy)
jpi.plot_format(fig_no_cloud)
show(column(fig_no_cloud,fig_cloudy ))

# %% [markdown]
# ## Direct Scattering Approximation
#
# The [derivation documentation](https://natashabatalha.github.io/picaso_dev#slide02) has a full description of these direct scattering approximations. Briefly I'll describe them here:
#
# At the center of each is the [One Term Henyey-Greenstein Phase Function (OTHG)](http://adsabs.harvard.edu/abs/1941ApJ....93...70H) and the [Two Term HG Phase Function (TTHG)](http://adsabs.harvard.edu/abs/1965ApJ...142.1563I).
#
# We also know that planet atmospheres have high degrees of Rayleigh scattering. [Cahoy+2010](http://adsabs.harvard.edu/abs/2010ApJ...724..189C) developed a methodology for incorporating Rayleigh into the direct scattering component.
#
# A more robust way of dealing with Rayleigh is to directly fold it's phase function into the TTHG phase function (TTHG_Ray).
#
# We'll run each case with and without a cloud so you can see what happens in both regimes

# %%
#let's make two different figures for this
fig_cloudy = figure(x_range=[0.3,1],y_range=[0,1.2], width=500, height=300
                    ,x_axis_label='Wavelength [μm]', y_axis_label='Geometric Albedo',)
fig_no_cloud = figure(x_range=[0.3,1],y_range=[0,1.2], width=500, height=300
                      ,x_axis_label='Wavelength [μm]', y_axis_label='Geometric Albedo',)

#define our options
options = jdi.single_phase_options()
colors = Colorblind8[0:len(options)]

#loop through all approximations
for approx, c in zip(options, colors):
    #set approximations
    cloud_free.approx(single_phase = approx,raman='pollack')
    cloudy.approx(single_phase = approx,raman='pollack')
    df = cloud_free.spectrum(opa)
    wno_nc, alb_nc = jdi.mean_regrid(df['wavenumber'] , df['albedo'],R=150)
    df = cloudy.spectrum(opa)
    wno_c, alb_c = jdi.mean_regrid(df['wavenumber'] , df['albedo'],R=150)
    fig_no_cloud.line(1e4/wno_nc, alb_nc, legend_label=approx, color=c, line_width=3)
    fig_cloudy.line(1e4/wno_c, alb_c,  color=c, line_width=3)
jpi.plot_format(fig_cloudy)
jpi.plot_format(fig_no_cloud)
show(column(fig_no_cloud,fig_cloudy ))

# %% [markdown]
# ## Multiple Scattering Approximations
#
# Again, [derivation documentation](https://natashabatalha.github.io/picaso_dev#slide03) has a full description of these multiple scattering approximations.
#
# To complete the multiple scattering integration over all _diffuse angles_, we have to use some mathematical tricks. [Legendre Polynomials](http://mathworld.wolfram.com/LegendrePolynomial.html) are often used to complete this integration to varying degrees. For Solar System/Exoplanet papers, we often stop the expansion at either `N=1` or `N=2`. Our standard will be to run with `N=2`, but below we show how to run each.

# %%
fig = figure(x_range=[0.3,1],y_range=[0,1.2], width=500, height=300,
             x_axis_label='Wavelength [μm]', y_axis_label='Geometric Albedo',)

options = jdi.multi_phase_options()
colors = Colorblind8[0:len(options)*2]

for approx, c1,c2 in zip(options, colors[0:2], colors[2:]):
    cloud_free.approx(multi_phase= approx)
    cloudy.approx(multi_phase = approx)
    df = cloud_free.spectrum(opa)
    wno_nc, alb_nc =jdi.mean_regrid( df['wavenumber'] , df['albedo'],R=150)
    df = cloudy.spectrum(opa)
    wno_c, alb_c =jdi.mean_regrid( df['wavenumber'] , df['albedo'],R=150)
    fig.line(1e4/wno_nc, alb_nc, color=c1, line_width=3)
    fig.line(1e4/wno_c, alb_c,  color=c2, line_width=3)
jpi.plot_format(fig)
show(fig)

# %% [markdown]
# ## Raman Scattering Approximations
#
# We all know the importance of Rayleigh scattering in planetary atmospheres. Raman scattering also has important implications for our spectra (these features have been observed in Solar System planets). In particular, at short wavelengths, Raman scattering imprints molecular features from the star on the planetary spectrum.
#
# The most complete analysis of all Raman approximations is in [Sromosvky+2005](http://adsabs.harvard.edu/abs/2005Icar..173..254S). From these, we use the _Pollack Approximation_ that was used in [Cahoy+2010](http://adsabs.harvard.edu/abs/2010ApJ...724..189C) and others.
#
# We include the original Pollack methodology, but also include a modified version with [Oklopcic et al 2018](http://iopscience.iop.org/article/10.3847/0004-637X/832/1/30/meta) cross sections and updated methodology to include effects of stellar spectrum.

# %%
fig_cloudy = figure(x_range=[0.3,1],y_range=[0,1.2], width=500, height=300
                    ,x_axis_label='Wavelength [μm]', y_axis_label='Geometric Albedo',)
fig_no_cloud = figure(x_range=[0.3,1],y_range=[0,1.2], width=500, height=300
                      ,x_axis_label='Wavelength [μm]', y_axis_label='Geometric Albedo',)

options = jdi.raman_options()
colors = Colorblind8[0:len(options)]

for approx, c in zip(options, colors):
    cloud_free.approx(raman = approx)
    cloud_free.star(opa, 6000, 0.0122, 4.437)
    cloudy.approx(raman = approx)
    cloudy.star(opa, 6000, 0.0122, 4.437)
    df = cloud_free.spectrum(opa)
    wno_nc, alb_nc = jdi.mean_regrid(df['wavenumber'] , df['albedo'],R=150)
    df = cloudy.spectrum(opa)
    wno_c, alb_c = jdi.mean_regrid(df['wavenumber'] , df['albedo'],R=150)
    fig_no_cloud.line(1e4/wno_nc, alb_nc, legend_label=approx, color=c, line_width=3)
    fig_cloudy.line(1e4/wno_c, alb_c,  color=c, line_width=3)

jpi.plot_format(fig_cloudy)
jpi.plot_format(fig_no_cloud)
show(column(fig_no_cloud,fig_cloudy ))

# %% [markdown]
# ## The Effect of Stellar Spectrum
#
# With the updated Raman scattering approximation, you will notice imprints of the stellar spectrum in the planet reflected light spectrum. Take a look below.

# %%
fig_cloudy = figure(x_range=[0.3,1],y_range=[0,1.2], width=500, height=300
                    ,x_axis_label='Wavelength [μm]', y_axis_label='Geometric Albedo',)
fig_no_cloud = figure(x_range=[0.3,1],y_range=[0,1.2], width=500, height=300
                      ,x_axis_label='Wavelength [μm]', y_axis_label='Geometric Albedo',)

#lets play around with different stellar spectra Teff
stellar_teff = [6000,4000,3500]

colors = Colorblind8[0:len(options)]

cloud_free.approx(raman = 'oklopcic')
cloudy.approx(raman = 'oklopcic')

for approx, c in zip(stellar_teff, colors):
    cloud_free.star(opa, approx, 0.0122, 4.437)
    cloudy.star(opa, approx ,0.0122, 4.437)
    df = cloud_free.spectrum(opa)
    wno_nc, alb_nc = jdi.mean_regrid(df['wavenumber'] , df['albedo'],R=150)
    df = cloudy.spectrum(opa)
    wno_c, alb_c = jdi.mean_regrid(df['wavenumber'] , df['albedo'],R=150)
    fig_no_cloud.line(1e4/wno_nc, alb_nc, legend_label=str(approx), color=c, line_width=3)
    fig_cloudy.line(1e4/wno_c, alb_c,  color=c, line_width=3)

cloud_free.approx(raman = 'pollack')
cloudy.approx(raman = 'pollack')

df = cloud_free.spectrum(opa)
wno_nc, alb_nc = jdi.mean_regrid(df['wavenumber'] , df['albedo'],R=150)
df = cloudy.spectrum(opa)
wno_c, alb_c =jdi.mean_regrid( df['wavenumber'] , df['albedo'],R=150)
fig_no_cloud.line(1e4/wno_nc, alb_nc, legend_label='Pollack', color='black', line_width=2, line_dash='dashed')
fig_cloudy.line(1e4/wno_c, alb_c,  color='black', line_width=2, line_dash='dashed')


jpi.plot_format(fig_cloudy)
jpi.plot_format(fig_no_cloud)
show(column(fig_no_cloud,fig_cloudy ))

# %% [markdown]
# # Benchmark Toon89 w/ Dlugach & Yanovitskij (1974)
#
# This replicates Figure 9 from Batalha et al. 2019, which is a benchmark study with [Dlugach & Yanovitskij (1974)](https://ui.adsabs.harvard.edu/abs/1974Icar...22...66D/abstract).

# %%
import picaso.model_compare as ptest

# %%
Dlugach, Toon89= ptest.dlugach_test(delta_eddington=True,toon_coefficients='quadrature',opd=0.5)

# %%
FigToon = jpi.rt_heatmap(Toon89,
                        cmap_kwargs={'palette':jpi.pals.viridis(11),
                                    'low':0,'high':0.8},
                        figure_kwargs={'title':'Toon89'})
FigDlugach = jpi.rt_heatmap(Dlugach,
                        cmap_kwargs={'palette':jpi.pals.viridis(11),
                                    'low':0,'high':0.8} ,
                        figure_kwargs={'title':'Dlugach'})
FigDiff = jpi.rt_heatmap((Toon89-Dlugach)/Dlugach,
                        cmap_kwargs={'palette':jpi.pals.RdGy11,
                                    'low':-0.4,'high':0.4} ,
                        figure_kwargs={'title':'% Diff'})
jpi.show(jpi.row([FigToon, FigDlugach,FigDiff ]))

# %% [markdown]
# A second way to visualize the albedo from using Toon, and Dlugach

# %%
f = jpi.figure(x_range=[0.7,1], y_axis_label='Albedo',x_axis_label='Single Scattering Albedo',
              height=300)
for i,c in enumerate(Toon89.index):
    f.line(Toon89.columns.astype(float), Toon89.loc[c,:],
           color=jpi.pals.Spectral7[i],line_width=3)
    f.line(Dlugach.columns.astype(float), Dlugach.loc[c,:], line_dash='dashed',
           color=jpi.pals.Spectral7[i],line_width=3)
jpi.show(f)

# %%
