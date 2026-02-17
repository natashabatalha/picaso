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
# # Getting Started : Basic Inputs and Outputs
#
# If you are here then you have already successfully
#
# 1) Installed the code
#
# 2) Downloaded the necessary reference data
#
# 3) Added environment variables
#
# If you have not done these things, please return to [Installation Guilde](https://natashabatalha.github.io/picaso/installation.html)
#
# Reminder you can always run these to check your environment:
#
# >> import picaso.data as data
#
# >> data.check_environ()

# %%
#picaso
from picaso import justdoit as jdi
from picaso import justplotit as jpi
import numpy as np

jpi.output_notebook()

# %%
#double check that your reference file path has been set
import os
refdata = os.getenv("picaso_refdata")
print(refdata)
#if you are having trouble setting this you can do it right here in the command line
#os.environ["picaso_refdata"]= add your path here AND COPY AND PASTE ABOVE
#IT WILL NEED TO GO ABOVE YOUR PICASO IMPORT

# %% [markdown]
# ## Connect to Opacity Database
#
# There is a full notebook in the tutorials devoted to learning how to make opacity databases in our format. If you cloned from `github`, there should also be an opacity database there called `opacity.db`.

# %%
help(jdi.opannection)

# %% [markdown]
# `opannection` has a few default parameters. A few notes:
#
#    1) `wave_range` can be used to select a subset of the full opacity database you are using
#
#    2) `filename_db` is a sqlite database that should have been downloaded and stored with your reference data (see [Installation Documentation](https://natashabatalha.github.io/picaso/installation.html))
#
#    3) `resample` should be used only if necessary to down sample the opacity database from its original value

# %%
opacity = jdi.opannection(wave_range=[0.3,1],verbose=True) #lets just use all defaults
#can also see molecules by doing:
#print(opacity.molecules )
#can see wavelength solution by doing:
#print(opacity.wno)

# %% [markdown]
# ## Load blank slate

# %%
start_case = jdi.inputs()

# %% [markdown]
# In order to run the code for reflected light we need (at the minimum) specific info about the:
#
# - **phase angle** (not needed for transmission or thermal calculations)
#
# - **planet** : gravity or mass/radius
#
# - **star** : temperature, metallicity, gravity and (if needed) stellar radius and semi major axis
#
# - **atmosphere** : P-T profile, chemical composition
#
# Some things (for example) `phase_angle` are only needed if you are running reflected light, but would not be needed if you were running `transmission`, for instance.

# %% [markdown]
# ## Set Planet & Star Properties

# %%
#phase angle
start_case.phase_angle(0) #radians

#define gravity
start_case.gravity(gravity=25, gravity_unit=jdi.u.Unit('m/(s**2)')) #any astropy units available

#define star
start_case.star(opacity, 5000,0,4.0) #opacity db, pysynphot database, temp, metallicity, logg

# %% [markdown]
# ## Set Atmospheric Composition
#
# There are different options for setting atmospheric composition.
#
#     1) Specifying a file path to model run
#     2) Give arbitrary pressure, temperature and composition directly as a dictionary input
#

# %% [markdown]
# ### Option 1) Specify file path
#
# Below, I am loading in a profile path for Jupiter that should be included in your reference data

# %%
print(jdi.jupiter_pt()) #should return the path to your reference data

# %%
start_case.atmosphere(filename=jdi.jupiter_pt(), sep=r'\s+')

# %% [markdown]
# ### File format
#
# 1) Must have **pressure**(bars), **temperature**(Kelvin), and `case sensitive` molecule names (e.g. TiO, Na, H2O, etc) for mixing ratios (in no particular order)
#
# 2) Can specify any necessary key word arguments for pd.read_csv at the end
#
# **PICASO will auto-compute mixing ratios, determine what CIA is neceesary and compute mean molecular weight based on these headers. Take at the preloaded example below**

# %%
#to give you an idea
comp_file = jdi.pd.read_csv(jdi.jupiter_pt(), sep=r'\s+')
#see example below
comp_file.head()

# %% [markdown]
# ## Create 1D Spectrum
#
# Let's create our first spectrum of Jupiter's reflected light at full phase

# %%
df = start_case.spectrum(opacity,calculation='reflected')#other options: transmission, thermal or a combo of many e.g., "reflected+transmission"

# %% [markdown]
# Checkout out what was returned (Note this is a change in v1.0). This dictionary will change based on what you have requested to be computed. For example, for reflected light you will see albedo, bond_albedo, fpfs_reflected. For thermal you will see

# %%
df.keys()

# %% [markdown]
# ## Regrid Opacities to Constant Resolution
#

# %%
wno, alb, fpfs = df['wavenumber'] , df['albedo'] , df['fpfs_reflected']
wno, alb = jdi.mean_regrid(wno, alb , R=150)

# %%
jpi.show(jpi.spectrum(wno, alb, plot_width=500,x_range=[0.3,1]))

# %% [markdown]
# FpFs is the relative flux of the planet and star or :
#
# $\frac{f_p}{f_s} = a_\lambda \left( \frac{ r_p}{a} \right) ^2$
#
# where $a$ is the semi-major axis. You may have noticed that **we did not supply a radius or semi-major axis in the above code**. Therefore, if you print out $\frac{f_p}{f_s}$ you will see this:

# %%
fpfs

# %% [markdown]
# Let's add this to the star function so we can get the relative flux as well..

# %%
start_case.star(opacity, 5000,0,4.0,semi_major=1, semi_major_unit=jdi.u.Unit('au'))
start_case.gravity(radius=1, radius_unit=jdi.u.Unit('R_jup'),
                   mass = 1, mass_unit=jdi.u.Unit('M_jup'))
df = start_case.spectrum(opacity, calculation='reflected',full_output=True)
wno, alb, fpfs = df['wavenumber'] , df['albedo'] , df['fpfs_reflected']

# %%
fpfs

# %% [markdown]
# ## Full Output
#
# Now we have specified that full_output=True so your dictionary will include an additional field that has more information:
#

# %%
df['full_output'].keys()

# %%
df['full_output']['layer'].keys()#includes pressure depenedent information from your calculation

# %%
df['full_output']['warnings']

# %% [markdown]
# ### Option 2) Arbitrary PT and Chemistry
#
# Sometimes for testing (or for atmospheres that we don't fully understand) an isothermal, well-mixed profile is sufficient. If we don't want to load in a full profile, we can give it a simple DataFrame with the info we need.
#

# %%
start_case.atmosphere( df = jdi.pd.DataFrame({'pressure':np.logspace(-6,2,60),
                                                 'temperature':np.logspace(-6,2,60)*0+200,
                                                 "H2":np.logspace(-6,2,60)*0+0.837,
                                                 "He":np.logspace(-6,2,60)*0+0.163,
                                                 "CH4":np.logspace(-6,2,60)*0+0.000466})
                     )

# %%
df = start_case.spectrum(opacity)
wno_ch4, alb_ch4, fpfs = df['wavenumber'] , df['albedo'] , df['fpfs_reflected']
wno_ch4, alb_ch4 = jdi.mean_regrid(wno_ch4, alb_ch4 , R=150)

# %%
jpi.show(jpi.spectrum(wno_ch4, alb_ch4, plot_width=500,x_range=[0.3,1]))

# %% [markdown]
# See how the plot above is much easier to interpret than the one with the full set of molecular input. Here we can clearly see the effects of methane opacity, raman and rayleigh scattering (the next notebook will include a tutorial for more diagnostic plotting)

# %% [markdown]
# ### Diagnostic help: Sometimes it helps to exclude molecule to see how it is influencing the spectrum
#
# Take a look below

# %%
start_case.atmosphere(filename=jdi.jupiter_pt(), exclude_mol='H2O', sep=r'\s+')

df = start_case.spectrum(opacity)
wno_nowater, alb_nowater, fpfs = df['wavenumber'] , df['albedo'] , df['fpfs_reflected']
wno_nowater, alb_nowater= jdi.mean_regrid(wno_nowater, alb_nowater , R=150)
fig = jpi.spectrum(wno, alb, plot_width=500)
fig.line(1e4/wno_nowater, alb_nowater, line_width=2, color='red')
jpi.show(fig)
