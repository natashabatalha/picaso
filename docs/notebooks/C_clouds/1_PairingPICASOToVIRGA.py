# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # How clouds affect albedo spectroscopy
#
# `PICASO` is used to compute reflected light and thermal emission of exoplanets. It is not a `virga` dependency though. This tutorial is mostly used as a module to help people gain an intuition for how various cloud outputs will affect reflected light observations.
#
# What you will learn:
#
# 1. How different cloud models affect reflected light spectra
#
# What you should already know:
#
# 1. How to run `PICASO` via [these tutorials](https://natashabatalha.github.io/picaso/notebooks/1_GetStarted.html).
#

# %%
import os
from picaso import justdoit as pj
from virga import justdoit as vj
#plot tools
from picaso import justplotit as picplt
from virga import justplotit as cldplt


import astropy.units as u
import pandas as pd
from bokeh.plotting import show, figure
from bokeh.io import output_notebook
output_notebook()

# %% [markdown]
# Let's set up a basic spectrum plot

# %%
opacity = pj.opannection(wave_range=[0.3,1])

# %%
sum_planet = pj.inputs()
sum_planet.phase_angle(0) #radians
sum_planet.gravity(gravity=25, gravity_unit=u.Unit('m/(s**2)')) #any astropy units available
sum_planet.star(opacity, 5000,0,4.0) #opacity db, pysynphot database, temp, metallicity, logg


# %% [markdown]
# Let's do simple cloud run with the `jupiter_pt` profile and constant $K_z$

# %%
df_atmo = pd.read_csv(pj.jupiter_pt(), sep='\s+')
#you will have to add kz to the picaso profile
df_atmo['kz'] = [1e9]*df_atmo.shape[0]

# %%
#business as usual
sum_planet.atmosphere(df=df_atmo)

#let's get the cloud free spectrum for reference
cloud_free = sum_planet.spectrum(opacity)

x_cld_free, y_cld_free = pj.mean_regrid(cloud_free['wavenumber'], cloud_free['albedo'], R=150)


# %% [markdown]
# ## Submit `virga` run within `picaso`
#
# You will be able to run `virga` runs directly through `picaso` so you don't have to go through all the steps twice.

# %%
metallicity = 1 #atmospheric metallicity relative to Solar
mean_molecular_weight = 2.2 # atmospheric mean molecular weight
# directory ='/data/virga/'
directory = os.path.join(os.getenv('picaso_refdata'), 'virga')

#we can get the same full output from the virga run
cld_out = sum_planet.virga(['H2O'],directory, fsed=1,mh=metallicity,
                 mmw = mean_molecular_weight)
out = sum_planet.spectrum(opacity, full_output=True)

x_cldy, y_cldy = pj.mean_regrid(out['wavenumber'], out['albedo'], R=150)

# %% [markdown]
# ## Analyzing Cloudy versus Cloud-free
#
# Let's take a moment to understand the difference between the cloudy and the cloud free case

# %%
show(picplt.spectrum([x_cld_free, x_cldy],
                     [y_cld_free, y_cldy],plot_width=500, plot_height=300,
                  legend=['Cloud Free','Cloudy']))


# %% [markdown]
# ### Brighter or Darker?
#
# Our spectra clearly got much brighter towards 1 micron. Let's explore why this is by looking at the optical properties and the photon attenuation plot.
#
# #### Photon Attenuation Depth
#
# This will tell us at what pressure we attain $\tau \sim 1$ optical depth for the molecular opacity, cloud opacity, and rayleigh opacity.

# %%
show(picplt.photon_attenuation(out['full_output'],
                            plot_width=500, plot_height=300))


# %% [markdown]
# If there were no cloud present, in this case Rayleigh scattering would dominate at the bluer wavelengths and gas opacity at the redder. But we see that the cloud reaches a significant optical depth well above the other two, so it will certainly influence the computed spectrum. Therefore, we know that our spectrum is going to be heavily influenced by this particular cloud model. **But now we need to know what the optical properties of the cloud are at those pressures**
#
# Let's ask ourselves the following questions:
#
# 1. What are the particle radii at 0.1 bar where the cloud opacity is sitting?
# 2. What are the Mie properties (asymmetry and single scattering) or the main condensate at that particle radii?
#
# #### 1) Look at particle radii
#

# %%
fig, dndr = cldplt.radii(cld_out,at_pressure=0.1)
show(fig)

# %% [markdown]
# Eyeballing the plot, it looks like our mean particle radius is about 30 microns. Let's pull our Mie parameters and see what the single scattering and asymmetry look like at that point.
#
# #### 2) Look at Mie parameters
#
# The Mie parameters describe how to translate our cloud solution to optical properties that can be used by radiative transfer. We'll extract $Q_\text{ext}$, the efficiency of extinction; $Q_\text{scat}$, the efficiency of scattering; and $g$, the Mie asymmetry parameter, which encodes the directional efficiency of scattering. These can be translated to the optical depth, single-scattering albedo, and asymmetry required by PICASO to compute a reflected light spectrum.

# %%
#grab your mie parameters
qext, qscat, g_qscat, nwave,radii,wave = vj.get_mie('H2O',directory)

# %%
from bokeh.layouts import row,column
ind = cldplt.find_nearest_1d(radii,30e-4) #remember the radii are in cm

qfig = figure(width=300, height=300,
              x_axis_type='log',y_axis_label ='Asymmetry',
              x_axis_label='Wavelength(um)')

wfig = figure(width=300, height=300,
              x_axis_type='log',y_axis_label ='Qscat/Qext',
              x_axis_label='Wavelength(um)')

qfig.line(wave[:,0], g_qscat[:,ind]/qscat[:,ind])
wfig.line(wave[:,0], qscat[:,ind]/qext[:,ind])

show(row(qfig, wfig))

# %% [markdown]
# This plot covers a large wavelength range. Zoom in on 0.3,1 micron, where we are computing our spectrum. **What asymmetry and single scattering values do you observe?**.
#
# Single scattering is essentially 1!! Water clouds are highly scattering! The asymmetry is around 0.8, which is quite forward scattering. However, the **high single scattering** creates a very bright cloud deck.

# %% [markdown]
# ## How varying cloud species affects reflectance spectrum
#
# Let's ramp up the temperature a bit and move away from H2O clouds to Na2S and ZnS. **WARNING**: We are about to artificially increase the temperature. Don't try this at home kids. IRL I would have to change all my chemistry as well. But here I am just making a point about how cloud composition/temp affects the reflectance

# %%
hot_atmo = df_atmo
hot_atmo['temperature'] = hot_atmo['temperature'] + 600

#remember we can use recommend_gas function to look at what the condensation curves look like
recommended = vj.recommend_gas(hot_atmo['pressure'], hot_atmo['temperature'], metallicity,mean_molecular_weight,
                #Turn on plotting and add kwargs for bokeh.figure
                 plot=True, y_axis_type='log',y_range=[1e2,1e-3],
                               plot_height=400, plot_width=600,
                  y_axis_label='Pressure(bars)',x_axis_label='Temperature (K)')


# %% [markdown]
# ### Run `picaso` and `virga` with different cloud species

# %%
#business as usual
sum_planet.atmosphere(df=hot_atmo)

#make sure clouds are turned off
sum_planet.clouds_reset()

#let's get the cloud free spectrum for reference
cloud_free = sum_planet.spectrum(opacity)
x_cld_free, y_cld_free = pj.mean_regrid(cloud_free['wavenumber'], cloud_free['albedo'], R=150)

#now the cloudy runs
cld_out = sum_planet.virga(['Na2S','ZnS'],directory, fsed=1,mh=metallicity,
                 mmw = mean_molecular_weight)

out = sum_planet.spectrum(opacity, full_output=True)
x_cld, y_cld = pj.mean_regrid(out['wavenumber'], out['albedo'], R=150)



# %% [markdown]
# ### Compare Na2S/ZnS against H2O clouds
#
# Look at the huge difference here. Our Na2S/ZnS did not have the effect the water clouds did. Let's look at photon attenuation, particle radii and Mie parameters to see if we can deduce why.
#

# %%
w = [x_cld_free, x_cld]
a = [y_cld_free, y_cld]
show(picplt.spectrum(w,a,plot_width=500, plot_height=300,
                  legend=['Cloud Free','Cloudy']))


# %% [markdown]
#
# #### Is the cloud opacity at a different pressure than the water cloud?

# %%
show(picplt.photon_attenuation(out['full_output'],
                            plot_width=500, plot_height=300))


# %% [markdown]
# Yes! The cloud opacity is quite a bit lower than our H2O run.
#
# #### Are the particle radii different?

# %%
fig, dndr = cldplt.radii(cld_out,at_pressure=0.5)
show(fig)

# %% [markdown]
# Not drastically different. We have particles that are approximately 10 microns instead of 30 microns. This is probably not a huge driver.
#
# #### Are the optical properties different? Run this script for Na2S and ZnS

# %%
#grab your mie parameters
gas_name = 'Na2S' #ZnS
qext, qscat, g_qscat, nwave,radii,wave = vj.get_mie(gas_name,directory)
ind = cldplt.find_nearest_1d(radii,10e-4) #remember the radii are in cm

qfig = figure(width=300, height=300,
              x_axis_type='log',y_axis_label ='Asymmetry',
              x_axis_label='Wavelength(um)')

wfig = figure(width=300, height=300,
              x_axis_type='log',y_axis_label ='Qscat/Qext',
              x_axis_label='Wavelength(um)')

qfig.line(wave[:,0], g_qscat[:,ind]/qscat[:,ind])
wfig.line(wave[:,0], qscat[:,ind]/qext[:,ind])

show(row(qfig, wfig))

# %% [markdown]
# Yes!! What did you notice about the optical properties between 0.3-1 micron? While these condensates are significantly less forward scattering, the single scattering albedos are much lower than one.
#
# **In conclusion! Your reflectance spectra are going to drastically change depending on what condensate species you are running!!!** For this case the higher pressure and lower single scattering albedos were the culprit in decreasing the overall brightness of the spectrum.

# %% [markdown]
# ## How sedimentation efficiency affects albedo spectrum?
#
# Let's go back to your normal cool case with H2O clouds.

# %%
df_atmo = pd.read_csv(pj.jupiter_pt(), sep="\s+")
df_atmo['kz'] = [1e10]*df_atmo.shape[0]

sum_planet.atmosphere(df = df_atmo)

all_fseds =  [1, 6, 10]
w = []
a = []
all_outs,cld_outs=[],[]
for fs in all_fseds:
    cld = sum_planet.virga(['H2O'],directory, fsed=fs,mh=metallicity,
                 mmw = mean_molecular_weight)
    cld_outs += [cld]
    out = sum_planet.spectrum(opacity,full_output=True)
    x,y = pj.mean_regrid(out['wavenumber'], out['albedo'], R=150)
    w += [x]
    a += [y]
    all_outs += [out['full_output']]


# %%
show(picplt.spectrum(w,a,plot_width=500, plot_height=300,
                     legend=['fs= '+str(i) for i in all_fseds]))

# %% [markdown]
# ### For H2O, why does lower $f_{sed}$ lead to higher albedo ?
#
# We can use the same exercises that we have used for the previous exercise in order to assess what is going on.
#
# For the sake of less code, let's just compare the $f_{sed}$ 0.1 and 6 case.
#
# ### How $f_{sed}$ affects cloud optical depth

# %%
show(column(picplt.photon_attenuation(all_outs[0],title='fs=0.1',
                            plot_width=500, plot_height=300),
   picplt.photon_attenuation(all_outs[2],title='fs=6',
                            plot_width=500, plot_height=300)))

# %% [markdown]
# Zoom out so you can see where the cloud opacity is at low fsed! The cloud opacity at low fsed is about an order of magnitude lower pressure than the high fsed case. In the high fsed case, at longer wavelengths, the molecular opacity starts competing with the cloud opacity.
#
# ### How $f_{sed}$ affects particle size

# %%
#low fsed case
fig, dndr = cldplt.radii(cld_outs[0],at_pressure=1e-3)
show(fig)

# %%
#high fsed case
fig, dndr = cldplt.radii(cld_outs[2],at_pressure=1)
show(fig)

# %% [markdown]
# Mean particle sizes where the cloud is most optically thick have changed significantly (almost two orders of magnitude). Therefore, even though we have the same cloud species, the Mie parameters will be different!

# %%
#grab your mie parameters
gas_name = 'H2O'
qext, qscat, g_qscat, nwave,radii,wave = vj.get_mie(gas_name,directory)

ind_low = cldplt.find_nearest_1d(radii,1e-4) #look at the plots to get these numberss
ind_high = cldplt.find_nearest_1d(radii,50e-4)#remember radii is in cm

qfig = figure(width=300, height=300,
              x_axis_type='log',y_axis_label ='Asymmetry',
              x_axis_label='Wavelength(um)')

wfig = figure(width=300, height=300,
              x_axis_type='log',y_axis_label ='Qscat/Qext',
              x_axis_label='Wavelength(um)')

for ind,l,c in zip([ind_low,ind_high],['fsed=0.1','fsed=6'],['red','blue']):
    qfig.line(wave[:,0], g_qscat[:,ind]/qscat[:,ind],legend_label=l,color=c)
    wfig.line(wave[:,0], qscat[:,ind]/qext[:,ind],legend_label=l,color=c)

show(row(qfig, wfig))

# %% [markdown]
# ## How $K_{z}$ affects albedo spectrum?
#
# Lastly, we can do the same thing with K_zz, the mixing parameter.

# %%
df_atmo = pd.read_csv(pj.jupiter_pt(), sep="\s+")

all_kzz = [1e6, 1e8, 1e10]
w = []
a = []
all_outs,cld_outs=[],[]
for kz in all_kzz:
    df_atmo['kz'] = [kz]*df_atmo.shape[0]
    sum_planet.atmosphere(df = df_atmo)
    cld = sum_planet.virga(['H2O'],directory, fsed=2,mh=metallicity,
                 mmw = mean_molecular_weight)
    cld_outs += [cld]
    out = sum_planet.spectrum(opacity,full_output=True)
    x,y = pj.mean_regrid(out['wavenumber'], out['albedo'], R=150)
    w += [x]
    a += [y]
    all_outs += [out['full_output']]


# %% [markdown]
# By looking at the spectra, you might suspect that increasing K_zz simply decreases the optical thickness of the cloud. Let's take a look at what else is changing in the cloud model.

# %%
f = picplt.spectrum(w,a,plot_width=500, plot_height=300,
                     legend=['Kzz= '+str(i) for i in all_kzz])
f.legend.location = 'bottom_left'
show(f)

# %% [markdown]
# From the lowest to the highest case, things don't immediately look very different. Close inspection of the photon attenuation plot reveals that the $\tau\sim$1 level of the highest mixing case is indeeed a bit lower than the lower mixing mixing case.

# %%
show(column(picplt.photon_attenuation(all_outs[0],title='kz=1e6',
                            plot_width=500, plot_height=300),
   picplt.photon_attenuation(all_outs[2],title='kz=1e10',
                            plot_width=500, plot_height=300)))

# %% [markdown]
# Lastly, inspecting the particle size distributions is where we see the largest differences. From the lowest, to highest K_zz cases we ran, we see an increase in 4 orders of magnitude in particle size!

# %%
fig, dndr = cldplt.radii(cld_outs[0],at_pressure=1e-2)
print('kz=1e6')
show(fig)


# %%
fig, dndr = cldplt.radii(cld_outs[2],at_pressure=1e-1)
print('kz=1e10')
show(fig)

# %%
