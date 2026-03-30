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
#     display_name: pic312
#     language: python
#     name: python3
# ---

# %% [markdown] id="un2r-8IJ4Fos"
# # Basics of analyzing transmission spectra with JWST
#
# You should have already installed PICASO (which can be found [here](https://natashabatalha.github.io/picaso/installation.html)). In this tutorial you will learn how to model the transmission spectrum of Wasp-39b. This entails learning the basics of:
#
# 1. Transit transmission spectra modeling
# 2. Data-to-model comparison via chi-square statistic
# 3. Climate modeling
# 4. Cloud modeling
#
# **NOTE: This tutorial is aimed at both beginner and advanced levels.** It is comprehensive and includes various locations to check understanding.

# %% [markdown] id="6b13e969-791b-4a38-885a-82732ba5627d"
# # Check PICASO Imports

# %% [markdown] id="0a481e95-1809-43e4-8485-6ce5652ddc58"
# Here are the two main PICASO functions you will be exploring:
#
# `justdoit` contains all the spectroscopic modeling functionality you will need in these exercises.
#
# `justplotit` contains all the of the plotting functionality you will need in these exercises.
#
# Tips if you are not familiar with Python or `jupyter notebooks`:
#
# - Run a cell by clicking shift-enter. You can always go back and edit cells. But, make sure to rerun them if you edit it. You can check the order in which you have run your cells by looking at the bracket numbers (e.g. [1]) next to each cell.
#
# - In any cell you can write `help(INSERT_FUNCTION)` and it will give you documentation on the input/output
#
# - If you type `jdi.` followed by "tab" a box will pop up with all the available functions in `jdi`. This applies to any python function (e.g. `numpy`, `pandas`)
#
# - If you type class.function?, for example `jdi.mean_regrid`, it will describe the function and its parameters/returns. This also applies to any class with a function.

# %% [markdown] id="b9ba3b9f"
# ## Make sure we have the right data
#
# 1. Are you a student and want to quickly run this without going through full PICASO data install setup? **PROCEED TO A do not edit B**
#
# 2. Have you already installed picaso, set reference variables, and have an understanding of how to get new data products associated with PICASO? **PROCEED TO edit B.**

# %%
import picaso.data as d

picaso_refdata = '/data/test/tutorial/picaso-lite-reference' #change to where you want this to live

"""
A) Uncomment if you are need the picaso-lite data
"""
d.os.environ['picaso_refdata'] = picaso_refdata

#if you do not yet have the picaso reference data complete this step below
#d.get_data(category_download='picaso-lite', target_download='tutorial_sagan23',final_destination_dir=picaso_refdata)

"""
B) Edit accordingly if needed (no need to edit if completed "A" above)
"""
#picaso ref data, if need to set it
d.os.environ['picaso_refdata'] = picaso_refdata
#stellar data environment
d.os.environ['PYSYN_CDBS'] = d.os.path.join(d.os.environ['picaso_refdata'],'stellar_grids')
#path to virga files
mieff_dir = d.os.path.join(d.os.environ['picaso_refdata'],'virga')
#path to correlated k preweighted files
ck_dir = d.os.path.join(d.os.environ['picaso_refdata'],'opacities', 'preweighted')
#"""

#lets check your environment
d.check_environ()
#dont have the data? return to the step A above to use the get_data function

# %% id="P_mk7nKI78GM"
import os
# Check you have picaso
from picaso import justdoit as jdi
from picaso import justplotit as jpi
import picaso.opacity_factory as op
import numpy as np

jpi.output_notebook()

# %% [markdown] id="9698f8d6-21c1-4230-a9a9-253232319d49"
# # Observed Spectrum
#
# Before we start modeling a planet to match WASP-39 b, let's get hold of WASP-39 b's actual observed spectrum in data format, so we can plot it next to our modeled ones. If you did [part 1 of this tutorial](https://github.com/Kappibw/JWST/blob/main/2_retrieving_jwst_spectra.ipynb), you already downloaded the data prepared by the scientists who wrote [the CO2 discovery paper](https://arxiv.org/pdf/2208.11692.pdf).
#
# Let's practice download the data from [Zenodo which is a popular place to host reduced data](https://zenodo.org/record/6959427#.Yx936-zMJqv). Download the .zip, and then look for `ZENODO/TRANSMISSION_SPECTRA_DATA/EUREKA_REDUCTION.txt`.
#

# %% id="ed9d8643-e8b9-43a3-a933-d347b6110a65"
eureka_reduction_path = '/data2/observations/WASP-39b/ZENODO/TRANSMISSION_SPECTRA_DATA/EUREKA_REDUCTION.txt'

# %% id="qdLa7P4lqUJD"
# Import ascii
from astropy.io import ascii

# Confirm can read the file
observed_data = ascii.read(eureka_reduction_path)

# %% id="353ce54e-3a05-4f42-b570-ef1769a51a89"
observed_data.colnames

# %% [markdown]
# Let's plot the observed data and see what we are working with!

# %% id="c12047e4-1d2a-4132-8c65-644d463d9ba3"
plt=jpi.plt
plt.figure(figsize=(12,4))
plt.errorbar(observed_data['wavelength'], observed_data['tr_depth'],
             [observed_data['tr_depth_errneg'], observed_data['tr_depth_errpos']],fmt='o')
plt.title('WASP-39 b Observed Spectrum')
plt.yscale('log')
plt.ylabel('Transit Depth')
plt.xlabel('Wavelength (micrometers)')
plt.show()

# %% [markdown] id="6d21b8cc-f53a-4bd4-85fc-6db71f9309af"
# # Spectra Ingredients
#
# Now what? Let's slowly build up a model that can match this data. What do we need?
#
# ## PICASO Basics
#
#
# ### List of what you will need before getting started
#
# 1. Planet properties
#
# - planet radius
# - planet mass
# - planet's equilibrium temperature
#
# 2. Stellar properties
#
# - stellar log gravity
# - stellar effective temperature
# - stellar metallicity
#
# 3. Opacities (how do molecules and atoms absorb light under different pressure/temperature conditions)
#
#

# %% [markdown] id="1e0d9c4c"
# # Basic Inputs
#
# ## Cross Section Connection
#
# All rapid radiative transfer codes rely on a database of pre-computed cross sections. Cross sections are computed by using line lists in combination with critical molecular pressure broadening parameters. Both can either be derived from theoretical first principles (e.g., [UCL ExoMol's line lists](https://www.exomol.com/)), measured in a lab, and/or some combination thereof (e.g., [HITRAN/HITEMP line lists](https://hitran.org/)).
#
# When cross sections are initially computed, a resolution ($\lambda/\Delta \lambda$) is assumed. Cross sections are computed on a line-by-line nature and therefore usually computed for R~1e6. For JWST we are often interested in large bands (e.g. 1-14 $\mu$m). Therefore we need creative ways to speed up these runs. You will usually find one of two methods: correlated-k tables and resampled cross sections. [Garland et al. 2019](https://arxiv.org/pdf/1903.03997.pdf) is a good resource on the differences between these two.
#
# For this demonstration we will use the resampled cross section method. **The major thing to note about using resampled cross sections** is that you have to compute your model at ~100x higher resolution than your data, but then must bin it down to a comparable resolution as your data so you can compare them. You will note that the opacity file you downloaded is resampled at R=10,000. Therefore you will note that **in this tutorial we will always bin the model down to R=100** before comparing with the data.
#
# **The overall idea is that we are just simply initializing PICASO to operate, create a "connection" to the models/spectral tools, in that wavelength (in microns) with the respective opacity range.**

# %% id="1c4f1eb8-1721-4617-bec3-073a665fab3a"
opa = jdi.opannection(wave_range=[2.7,6])

# %% [markdown] id="a13f6558"
# ## Set Basic Planet and Stellar Inputs
#
# Second step is to define the basic planet parameters. Depending on the kind of model you want to compute (transmission vs. emission vs. reflected light), there are different requirements for the minimum set of information you need to include.
#
# For WASP-39 b, since we have planet mass, radius, and all the necessary stellar specifications, we will be thorough in our inputs and include all parameters.

# %% [markdown]
# We can create an object called w39 that represents the planet WASP-39 b. We then assign values to that object. We already know the temperature, radius, metallicity, and surface gravity of the host star, so we can assign those values to the parent star under w39.star. We also know the mass and radius of the planet, so we can assign those values to the planet under w39.gravity (named thus because mass and radius will be used to compute the planet's surface gravity). As we move through the tutorial and make climate models etc., we will be doing these calculations on the object w39.

# %% id="ce1ce8d8"
# Create that object or "case"
w39 = jdi.inputs()

# Describe the star
w39.star(opa, temp=5400 , database='phoenix',
         metal=0.01, logg=4.45, radius=0.9, radius_unit=jdi.u.R_sun )

# Describe the planet
w39.gravity(mass=0.28, mass_unit=jdi.u.M_jup,
             radius=1.27, radius_unit=jdi.u.R_jup)


# %% id="8eaf1915"
# To get information about a function, you can use '?'
help(w39.star)

# %%
# want to see the spectrum you input? it lives here
w39.inputs['star']

# %% [markdown] id="cbd735be"
# ## Set Climate and Chemistry
#
# We now need to think about how we can model the climate and chemistry of this system. For the sake of this tutorial we will start really simple, then move forward to something more complex.
#
# **For climate**, we will explore these levels in this tutorial
#
# 1. Isothermal (now)
# 2. Radiative-convective (next section)
#
# **For chemistry**, we will explore these levels in this tutorial
#
# 1. Chemical Equilibrium
#

# %% [markdown] id="ec4bd5a4-9dc7-498a-8db9-e7e3c392509b"
# ### Pressure
# If we imagine our "nlevels" as equally spaced altitude bands on the planet, then we will assign pressures to decrease logarithmically as altitude increases.
#
# Gas is compressible and tends to behave in that way in planetary atmospheres (including on Earth).

# %% id="774e33dc-80fd-4a99-9a2f-c8b611c301b9"
nlevels = 50
# Logspace goes from base^(start) to base^(end)
# so here we are going from 10^-6 to 10^2, which is
# 1 microbar to 100 bars of pressure.
# This is an arbitrarily chosen range, but this is the most common.
pressure = np.logspace(-6,2,nlevels)

# %%
print(pressure)

# %% [markdown] id="a1df5148-9cc7-49ff-b19e-3ffeb57d5750"
# ### Isothermal Temperature

# %% [markdown]
# Next we need to decide how the temperature of the atmosphere varies with pressure. On Earth, temperature generally drops as you travel further from the Earth's surface, i.e. higher in altitude and lower in pressure. But that is not always the case. For simplicity, we'll start by assuming the temperature of WASP-39 b's atmosphere is constant with pressure, which is called "isothermal." As our models get more sophisticated, we'll be able to refer back to this simple case.

# %% id="e57e4846-fe43-4731-8cf6-978672da70d2"
# We can see from exo.MAST that the equilibrium temp
# of WASP 39 b is 1120 kelvin, so let's use a scale
# of temperatures based on that.
equilibrium_temperature = 1120.55
isothermal_temperature = np.zeros(nlevels) + equilibrium_temperature

# %%
print(isothermal_temperature)

# %% [markdown] id="05e13ad1-1728-4dfb-9638-05b761608172"
# #### Setting the Atmosphere in PICASO

# %% [markdown]
# So far we have described the parent star and the planet. Now let's define the planet's atmosphere. Note, this is a common workflow, where you start by creating an object (w39) and then slowly add parameters (information) as you go.

# %% id="475d82ed-51c6-45c3-8f12-3d8472ee3ea2"
w39.atmosphere(df = jdi.pd.DataFrame({
                'pressure':pressure,
                'temperature':isothermal_temperature}))

# %%
# want to see the input added to the class
w39.inputs['atmosphere']['profile']

# %% [markdown] id="f5016fc9-1275-4a55-b9ec-adccceae4921"
# ### Chemistry
#
# Now we need to add the chemistry! PICASO has a prebuilt chemistry table that was computed by Channon Visscher. You can use it by adding it to your input case. Two more chemistry parameters are now going to be introduced:
#
# 1. M/H: Atmospheric metallicity
# 2. C/O ratio: Elemental carbon to oxygen ratio
#

# %% [markdown] id="bb981137-ccbc-4f77-9cad-e7fb35d397ea"
# #### Metallicity
#
# <img src="https://stellarplanetorg.files.wordpress.com/2020/04/wakeforddalba2020_rs_mass_metallicity_v1.jpg?w=736" width="800">
#
# Looking at a mass-metallicity plot (compiled in [Wakeford & Dalba 2020](https://ui.adsabs.harvard.edu/abs/2020RSPTA.37800054W/abstract)) might offer a good starting point to decide what the M/H of your planet might be. Here we can see WASP-39 b HST observations led to the inference of ~100xM/H. One tactic might be to start from that estimate. Another might be to use the Solar System extrapolated value (gray dashed line) as a first pass. Let's start with the latter as a first guess.

# %% id="9bc13061-939e-4cf5-b7fb-081e35d60100"
log_mh = 1.0 # log relative to solar
# so a value of 1 here represents 10^1 = 10x solar

# %% [markdown] id="12494d50-6908-4ee4-85c4-5ce31fd580ec"
# #### C/O Ratio
#
# The elemental ratio of carbon to oxygen controls the dominant carbon-bearing species. For instance, take a look at Figure 1 from the paper [C/O RATIO AS A DIMENSION FOR CHARACTERIZING EXOPLANETARY ATMOSPHERES](https://iopscience.iop.org/article/10.1088/0004-637X/758/1/36/pdf).
#
# At low C/O, we see CO$_2$ and CO as the dominant form of carbon, and at high C/O we see CH$_4$ and CO as the dominant form of carbon.
#
# C/O is given in PICASO in units relative to solar C/O, which is worth noting because you're not giving it the actual ratio of carbon to oxygen, but rather the ratio relative to the Solar C/O. Solar C/O is ~0.5 ([Asplund et al. 2021](https://ui.adsabs.harvard.edu/abs/2021A%26A...653A.141A/abstract)), and let's set our value to be the same as Solar (i.e. 1 relative to Solar).

# %% id="27069c4b-0a92-45f6-a21b-ed56682aef57"
c_o = 0.55 # absolute solar

# %% [markdown] id="afc5200e-7afe-4366-bc8c-cb132951885c"
# Now we can ask PICASO to make us a mixture of molecules consistent with the relative M/H metallicity and relative C/O ratio we set up, and we can take a look at what it creates. We can look at the "atmosphere profile" which shows us the abundances of various molecules at different levels of our atmosphere (levels being places where temperature and pressure differ in the manner we defined above).
#
# This can help us find out which molecules we should really focus on during our analysis, and others that may be harder or redundant to look for based off of our C/O ratio and metallicity.

# %% id="c618536e-3990-48b0-bb7e-516d1ab83223"
w39.chemeq_visscher_2121(c_o, log_mh)

# %% [markdown] id="80b1b45d-eb0a-4373-8f83-165807d47e92"
# Running this function sets a dataframe in your inputs dictionary, which you can access now with `w39.inputs['atmosphere']['profile']`. You can see that PICASO has given us loads of different molecules to work with, but many have miniscule abundances (note some e-38 values in there).

# %%
w39.inputs['atmosphere']['profile'].head()

# %% [markdown] id="98aeb8fc-4b53-4457-870a-a899affc9a9c"
# ### Reference Pressure
#
# Lastly, we need to decide on a "reference pressure." If our planet was terrestrial, this would be the pressure at the surface, and therefore also the pressure corresponding to the radius of the planet. For gas giants like WASP-39 b, this is a bit more complicated -- there is no "surface," so we need to pick a pressure that corresponds to our planet's "radius" or surface, so that PICASO can calculate gravity as a function of altitude from that level.
#
# We are selecting a pressure that we are essentially calling the bottom.
#
# As we've discussed, a planet's radius changes depending on the wavelength at which you observe it -- it's that change that we are measuring with our spectra. But when we input the planet radius above, we picked a single number -- 1.27 `r_jup`. That number is an average calculated over a band of wavelengths. And so when we pick a reference pressure, we estimate roughly what level of the planet's atmosphere that averaged radius corresponds to (where, from the high pressure deep inside, to the low pressure at the exterior, does the chosen radius fall?).
#
# PICASO suggests a reference pressure of 10 bar for gas giants, so we will start with that:

# %% id="7a1b1cc7-37ef-4960-8512-f4d77f326fd4"
w39.approx(p_reference=10)

# %% [markdown] id="969c933b"
# ### Want to check your inputs so far?
#
# If you want you can consult `w39.inputs` to check or reset inputs. Let's see how our WASP-39 b object is holding up!

# %% id="3dc8c591"
w39.inputs.keys()

# %% [markdown]
# An atmosphere profile table shows the elemental abundances at each pressure level in our model atmosphere.

# %% id="7be17b68"
w39.inputs['atmosphere']['profile'].head()

# %% [markdown]
# We can look at the abundances for a single molecule, say CO$_2$.

# %% id="d62cbd84"
# Grab CO2 array, for instance
w39.inputs['atmosphere']['profile']['CO2'].values

# %% [markdown]
# We can check our inputs for the planet and the host star.

# %% id="61388471"
w39.inputs['planet'], w39.inputs['star'] # All your inputs have been archived!

# %% [markdown] id="4ed9002e-85a3-4b3b-a20f-2750e7004847"
# # Creating a Transmission Spectrum
#
# Now that we have set up PICASO with everything it needs, and we understand the components needed to model an exoplanet, let's ask PICASO to output a transmission spectrum for our WASP-39 b.
#
# ## First Run PICASO
#
# We can use the <code>.spectrum</code> function to do so.
#
# We need to give the function a connection to the opacity database we used earlier to look up the absorption spectrum for water, as well as an instruction to create a "transmission" spectrum (as opposed to, for example, a reflected light spectrum of star light bouncing off the planet when it's almost behind its star).

# %% id="bcf63c09-64b4-4122-b56e-24445bb489ce"
model_iso = w39.spectrum(opa,
                       # Other options are "thermal" or "reflected"
                       # or a combination of two e.g. "transmission+thermal"
                       calculation='transmission',
                       full_output=True)


# %% [markdown] id="ba6bba7f-3eb9-4a8a-8c5c-e88c7c371f7d"
# ## Our Spectra
#
# Let's set up a function to display the spectrum in our first output dictionary (first one is called `model_iso`). Moving forward  we will create more models and we want a way to easily display them.

# %% id="42dd80a1-e613-4038-a71a-f16b76c9cf11"
def show_spectra(output, x_range_min=3.0, x_range_max=5.5):
    # Step 1) Regrid model to be on the same x axis as the data
    wnos, transit_depth = jdi.mean_regrid(output['wavenumber'],
                                         output['transit_depth'],
                                         newx=sorted(1e4/observed_data['wavelength']))

    # Step 2) Use PICASO function to create a plot with the models
    fig = jpi.spectrum(wnos, transit_depth,
                       plot_width=800,y_axis_label='Absolute (Rp/Rs)^2',
                       x_range=(x_range_min,x_range_max))

    # Step 3) Use PICASO function to add the data so we can compare
    jpi.plot_multierror(observed_data['wavelength'], observed_data['tr_depth'], fig,
                        dy_low=observed_data['tr_depth_errneg'], dy_up=observed_data['tr_depth_errpos'],
                        dx_low=[0],dx_up=[0], point_kwargs={'color':'black'})
    jpi.show(fig)


# %% id="fe9cb93d-69af-4436-ac40-77c1b5e358df"
# Display our first spectrum!
show_spectra(model_iso)


# %% [markdown]
# Sweet! We have a spectrum that looks the right-ish shape, even though it isn't quite in the right place. Let's take a minute to work that out.

# %% [markdown] id="bd05513b-9193-4e3c-b39e-74a8c877d446"
# ### Transit Depth Offsets
#
# Remember how we guessed what the reference pressure was? Well, it looks like we are a little off. That is okay! When fitting for transit spectra, we introduce a factor to account for this. In the retrieval tutorial, you will fit for a factor of the radius. For now, let's "mean subtract" our data so that the model and data lie on top of one another.

# %% id="ea7f587f"
# Let's mean subtract these
def show_spectra(output, x_range_min=3.0, x_range_max=5.5):
    # Step 1) Regrid model to be on the same x axis as the data
    wnos, transit_depth = jdi.mean_regrid(output['wavenumber'],
                                         output['transit_depth'],
                                         newx=sorted(1e4/observed_data['wavelength']))

    # Step 2) Use PICASO function to create a plot with the models
    fig = jpi.spectrum(wnos, transit_depth-np.mean(transit_depth),
                       plot_width=800, y_axis_label='Relative (Rp/Rs)^2',
                       x_range=(x_range_min,x_range_max))

    # Step 3) Use PICASO function to add the data so we can compare
    jpi.plot_multierror(observed_data['wavelength'],
                        observed_data['tr_depth'] - np.mean(observed_data['tr_depth']),
                        fig,
                        dy_low=observed_data['tr_depth_errneg'], dy_up=observed_data['tr_depth_errpos'],
                        dx_low=[0],dx_up=[0], point_kwargs={'color':'black'})
    jpi.show(fig)


# %% id="d8525969-4251-4c5b-8bf9-17a71c3e8116"
show_spectra(model_iso)

# %% [markdown] id="56c298ba"
# ## Model Investigation
#
# Before trying to improve the complexity of your model, let's make sure you know how to analyze the inputs.
#
# ### Identifying molecular features with optical depth contribution plot
#
# `taus_per_layer` - Each dictionary entry is a nlayer x nwave that represents the per layer optical depth for that molecule.
#
# `cumsum_taus` - Each dictionary entry is a nlevel x nwave that represents the cumulative summed opacity for that molecule.
#
# `tau_p_surface` - Each dictionary entry is a nwave array that represents the pressure level where the cumulative opacity reaches the value specified by the user through at_tau.
#
#

# %% [markdown]
# Let's create a molecule contribution plot. At any given wavelength, this will show us at which pressure layer (`tau_p_surface`) each molecule experiences an optical depth of $\tau$. Below, we will set $\tau = 1$. This is useful because whenever we see a spectral feature, that light is originating from a depth in the atmosphere corresponding to the $\tau=1$ surface for that molecule.

# %% id="0c06856a"
molecule_contribution = jdi.get_contribution(w39, opa, at_tau=1)

# %% id="4639440f"
jpi.show(jpi.molecule_contribution(molecule_contribution,
                                   opa, plot_width=700, x_axis_type='log'))

# %% [markdown] id="8d9d34fc"
# ### Identifying molecular features with "leave-one-out" method
#
# Another option for investigating model output is to remove the contribution of one gas from the model to see if it affects our spectrum. CO$_2$ was a fairly obvious feature for the 4.3$\mu$m. But what about H$_2$O and CO from 4.4-6$\mu$m. In this region there is no distinct "feature" in the spectrum. How sure are we that H$_2$O and CO are really there? We can use the "leave-one-out" method to see how individual molecules are shaping each part of our spectrum.
#
# We can use PICASO's `exclude_mol` key to exclude the optical contribution from one molecule at a time. In the code below lets do this for CO2, H2O, and CO. We will loop through one by one and create a spectra without each of these. The result will show you where their contribution to the spectra is most important.

# %% id="a457832c"
w,f,l =[],[],[]
df_og = jdi.copy.deepcopy(w39.inputs['atmosphere']['profile'])
for iex in ['CO2','H2O','CO',None]:
    w39.atmosphere(df=df_og,exclude_mol=iex, sep=r'\s+')
    df= w39.spectrum(opa, full_output=True,calculation='transmission') #note the new last key
    wno, rprs2  = df['wavenumber'] , df['transit_depth']
    wno, rprs2 = jdi.mean_regrid(wno, rprs2, R=150)
    w +=[wno]
    f +=[rprs2]
    if iex==None:
        leg='all'
    else:
        leg = f'No {iex}'
    l+=[leg]
jpi.show(jpi.spectrum(w,f,legend=l))


# %% [markdown]
# We can see clearly that when H$_2$O or CO$_2$ is not present, a completely different spectrum is created that veers far from our model. Therefore we can feel confident that H$_2$O or CO$_2$ are truly present. When we try leaving out CO, however, the spectrum changes more modestly; it appears that including CO improves the spectrum somewhat, but we are less certain about its presence.

# %% [markdown] id="b1aab798-3d33-4582-9952-94e45e686474"
# # Increasing Model Complexity to Improve Fit: Chemistry, Climate, Clouds
#
# In the next sections we will try to improve our model fit. However, before we continue we need a way to quantify our "goodness of fit."

# %% [markdown] id="8dbbe171-a38f-4bee-8b59-93e3108d6ca4"
# ## Define Goodness of Fit
#
# Let's implement a simple measurement of error called a "chi-squared test." This is a commonly used method to measure how well you are fitting data with a model, and sums up the distance between your model's output and the observed data at each data point.
#
# The standard chi-squared test formula is $\chi^2 = \sum \frac{(O_i - E_i)^2}{(\sigma_i)^2}$ where $O$ is the observed data, $E$ is the expected value (i.e. our model), and $\sigma$ is error.
#
# In this notebook, we'll compute chi-squared *per data point*, which means we are normalizing the standard chi-square by dividing by number of data points. The chi-squared per data point formula is $\frac{\chi^2}{N}$ where $N$ is the number of data points.
#
# Note, another common way to report chi-squared is the *reduced* chi-squared which considers the degrees of freedom $(\nu)$, where $\nu$ is the number of data points ($N$) minus the number of fitted parameters.
#
# Regardless of the version of chi-squared you use, if the result is close to 1, that is considered a good fit.

# %% id="9054e287-dcbb-45c6-a21c-db2fcd9828ce"
def chisqr(model):
    # Step 1) Regrid model to be on the same x axis as the data
    wnos, model_binned = jdi.mean_regrid(model['wavenumber'],
                                         model['transit_depth'],
                                         newx=sorted(1e4/observed_data['wavelength']))
    # Step 2) Flip model so that it is increasing with wavelength, not wavenumber
    model_binned = model_binned[::-1]-np.mean(model_binned)

    # Step 3) Compute chi sq with mean subtraction
    observed = observed_data['tr_depth'] - np.mean(observed_data['tr_depth'])
    error = (observed_data['tr_depth_errneg'] + observed_data['tr_depth_errpos'])/2
    return np.sum(((observed-model_binned)/error)**2 ) / len(model_binned)


# %% id="76d4f75c"
print('Simple First Guess', chisqr(model_iso))

# %% [markdown] id="72a5a724"
# Not great! Let's make it better.

# %% [markdown] id="1fb463d1"
# ## Revisiting Chemistry Assumption
#
# Earlier, we assumed M/H and C/O values. Let's loop through a few M/H and C/O values to see if any of these seem to improve our model fit. Why? These values are faster to assess than climate or clouds, and they're easier to loop through while you're building intuition. We want to find the best fit combination of variables such as C/O and M/H that will gives us the closest fit to the data.

# %% id="ae270058"
mh_grid_vals = [1, 10, 100]
co_grid_vals = [0.55, 1.3] #absolute c/o ratios

chemistry_grid = {}
for imh in mh_grid_vals:
    for ico in co_grid_vals:
        w39.chemeq_visscher_2121(ico, np.log10(imh))
        chemistry_grid[f'M/H={imh},C/O={ico}'] = w39.spectrum(opa, calculation='transmission', full_output=True)
        print(f'M/H={imh},C/O={ico}', chisqr(chemistry_grid[f'M/H={imh},C/O={ico}'] ))


# %% [markdown]
# We see with a M/H of 100 and C/O of 1, we get the best fit where the chi-squared is ~4. Let's plot all these and visually confirm.

# %% id="0f5bf193"
# Let's edit our function once again to allow for multiple model inputs
def show_spectra(output, x_range_min=3.0, x_range_max=5.5):
    #Step 1) Regrid model to be on the same x axis as the data
    wnos, transit_depth = zip(*[jdi.mean_regrid(output[x]['wavenumber'], output[x]['transit_depth'],
                        newx=sorted(1e4/observed_data['wavelength']))
                        for x in output])
    transit_depth = [x-np.min(x) for x in transit_depth]
    wnos = [x for x in wnos]
    legends = [i for i in output]

    # Step 2) Use picaso function to create a plot with the models
    fig = jpi.spectrum(wnos, transit_depth, legend=legends,
                       plot_width=800,y_axis_label='Relative (Rp/Rs)^2',
                       x_range=(x_range_min,x_range_max))

    # Step 3) Use picaso function to add the data so we can compare
    jpi.plot_multierror(observed_data['wavelength'],
                        observed_data['tr_depth'] - np.min(observed_data['tr_depth']),
                        fig,
                        dy_low=observed_data['tr_depth_errneg'], dy_up=observed_data['tr_depth_errpos'],
                        dx_low=[0], dx_up=[0], point_kwargs={'color':'black'})
    jpi.show(fig)


# %% id="25737413"
show_spectra(chemistry_grid)


# %% [markdown]
# ### Check your understanding
#
#
# Let's investigate how our chemistry choices affect our model spectrum. Explore the figure above by clicking on the legend to remove different lines.
#
# 1. Try looking only at lines that keep C/O steady while varying M/H from 1 to 10 to 100. What impact does the change in M/H ratio have on the spectrum?
#
# 2. You'll notice that as M/H increases, the CO$_2$ bump near 4.4 $\mu$m initially grows and then shrinks. Why might this be?
#
# 3. Now look at lines that keep M/H steady while varying C/O from 1 to 2.5. What impact does this change in C/O ratio have on the spectrum?
#
# 4. You'll notice that as C/O increases, the CO$_2$ bump near 4.4 $\mu$m disappears while a new CH$_4$ line appears near 3.35 $\mu$m. Why might this be?
#
# Some more targeted questions to help with the above:
#
# 1. Figure out what the top 4 most abundant oxygen- and carbon-bearing species are by looking at your chemistry grid.
# 2. How does their abundance change when you vary the value of M/H?
# 3. How does their abundance change when we increase C/O from 1 to 2.5?
# 4. How does mean the atmosphere's molecular weight change when you vary M/H?
# 5. Can you see that reflected in the spectrum?
#

# %% [markdown]
# <div class="alert alert-block alert-info">
# Solution 1: Figure out what the top 4 most abundant oxygen- and carbon-bearing species are by looking at your chemistry grid.
# </div>
#
# Let's write a simple function to find the most abundance O and C bearing species

# %%
# Write a simple function to identify the top most abundant species

def find_top(grid, mh=1, co=0.55, n=4, method='average', more=False):
    """
    Find the n most abundant oxygen- and carbon-bearing molecules for a given M/H and C/O ratio.

    Parameters
    ----------
    grid: dict
        chemistry grid"
    mh : int (optional)
        M/H ratio relative to Solar
    co : int (optional)
        C/O ratio relative to Solar
    n : int (optional)
        number of molecules you want to identify
    method : str (optional)
        how you will calculate the abundance of a given species; options are 'sum', 'average', 'median'
    more : boolean (optional)
        if True, will print results of each step

    Returns
    -------
    top_dict
        dictionary of the most abundant C- and O-bearing molecules and their total abundances
    """

    # What are all the available species? (elements start in fourth column)
    cols = grid[f'M/H={mh},C/O={co}']['full_output']['layer']['mixingratios'].columns.tolist()[3::]
    if more == True:
        print("All available species:\n", cols, "\n")

    # Grab only the carbon- and/or oxygen-bearing molecules
    filtered_cols = [mol for mol in cols if ('C' in mol or 'O' in mol) and 'Cs' not in mol]
    if more == True:
        print("Carbon- and/or oxygen-bearing molecules:\n", filtered_cols, "\n")

    # Choose a value to represent the abundance for each molecule
    tot_abun = {}
    for mol in filtered_cols:
        if method == 'sum': # Sum the abundance across all pressure layers
            tot_abun[mol] = np.sum(grid[f'M/H={mh},C/O={co}']['full_output']['layer']['mixingratios'][mol].values)
        elif method == 'average': # Choose the average abundance value
            tot_abun[mol] = np.average(grid[f'M/H={mh},C/O={co}']['full_output']['layer']['mixingratios'][mol].values)
        elif method == 'median': # Choose the median abundance value
            tot_abun[mol] = np.median(grid[f'M/H={mh},C/O={co}']['full_output']['layer']['mixingratios'][mol].values)
    if more == True:
        print("Total abundances:\n", tot_abun, "\n")

    # What are the 4 highest abundances?
    top = sorted(tot_abun, key=tot_abun.get, reverse=True)[:n]
    top_dict = {k: tot_abun[k] for k in top}
    if more == True:
        print(f"Top {n} carbon- and/or oxygen-bearing molecules:\n", top)

    return top_dict

# Uncomment to test
find_top(chemistry_grid, method='average', more=True)

# %% [markdown]
# <div class="alert alert-block alert-info">
# Solution 2: How do abundances change when varying M/H?
# </div>

# %%
# Visualize how the abundance of each molecule changes as M/H increases

# Initialize an empty dictionary to store abundances
abundances = {}

# Loop through the M/H grid values
for i, mh in enumerate(mh_grid_vals):
    # Calculate total abundance of top 4 molecules for a given M/H value
    top_dict = find_top(chemistry_grid, mh=mh, co=0.55, n=5, method='sum')
    print(f'M/H={mh}, C/O=1:', top_dict)

    # Add molecule abundances dynamically and insert at the correct position
    for molecule, abundance in top_dict.items():
        if molecule not in abundances:
            abundances[molecule] = [None] * len(mh_grid_vals)  # Initialize the list with None
        abundances[molecule][i] = abundance  # Insert the abundance at the correct index

# Plot abundance vs. M/H for top 4 molecules
plt.figure(figsize=(8, 4), dpi=100)
for molecule in abundances:
    plt.plot(mh_grid_vals, abundances[molecule], marker='o', label=molecule)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('M/H (xSolar)')
plt.ylabel('Total Abundance')
plt.legend()
plt.show()

# %% [markdown]
# <div class="alert alert-block alert-info">
# We see that most species increase in abundance as metallicity increases, and they do so at different rates.
# </div>

# %% [markdown]
# <div class="alert alert-block alert-info">
# Solution 3: How do abundances change when varying C/O?
# </div>

# %%
# Visualize how the abundance of each molecule changes as C/O increases

# Initialize an empty dictionary to store abundances
abundances = {}

# Loop through the C/O grid values
for i, co in enumerate(co_grid_vals):
    # Calculate total abundance of top 4 molecules for a given C/O value
    top_dict = find_top(chemistry_grid, mh=1, co=co, method='sum')
    print(f'M/H=1, C/O={co}:', top_dict)

    # Add molecule abundances dynamically and insert at the correct position
    for molecule, abundance in top_dict.items():
        if molecule not in abundances:
            abundances[molecule] = [None] * len(co_grid_vals)  # Initialize the list with None
        abundances[molecule][i] = abundance  # Insert the abundance at the correct index

# Plot abundance vs. C/O for top 4 molecules
plt.figure(figsize=(8, 4), dpi=100)
for molecule in abundances:
    plt.plot(co_grid_vals, abundances[molecule], marker='o', label=molecule)
plt.xlabel("C/O (xSolar)")
plt.ylabel("Total Abundance")
plt.legend()
plt.show()

# %% [markdown]
# <div class="alert alert-block alert-info">
# We see that as the ratio of carbon to oxygen increases, the abundance of carbon-bearing species (CH$_4$ and CO) grows while the abundance of H$_2$O declines.
# </div>

# %% [markdown]
# <div class="alert alert-block alert-info">
# Solution 4: How does mean molecular weight vary with M/H?
# </div>

# %%
# Visualize how mean molecular weight changes with M/H

# Get the average mean molecular weight of the atmosphere at each M/H value
mmws = []
for mh in mh_grid_vals:
    mmws.append(np.mean(chemistry_grid[f'M/H={mh},C/O=0.55']['full_output']['layer']['mmw']))

# Plot how the mean molecular weight of each molecules changes with M/H
plt.figure(figsize=(8, 4), dpi=100)
plt.plot(mh_grid_vals, mmws, marker='o')
#plt.xscale('log')
plt.xlabel('M/H (xSolar)')
plt.ylabel('Mean $\\mu$')
plt.show()

# %% [markdown]
# <div class="alert alert-block alert-info">
# We see that mean molecular weight increases linearly as M/H increases.
# </div>

# %% [markdown]
# <div class="alert alert-block alert-info">
# Step 5: Can you see this reflected in the spectrum?
# <br>
# <br>
# What impact does the change in M/H ratio have on the spectrum?
#
# <details>
#   <summary><b>Click to expand solution</b></summary>
# <i>Why a feature might grow:</i> The strength of a spectral feature is impacted by a molecule's abundance. When we increase atmospheric metallicity, we are increasing the raw abundance of all elements heavier than hydrogen. Complicated chemical processes ensue, resulting in a new distribution of molecules constructed from those raw elements. Some molecules (e.g., CO$_2$) are more sensitive to changes in metallicity than others (e.g., H$_2$O), so although the abundance of all molecules will increase as metallicity increases, they will do so at different rates.
# <br><br>
# <i>Why a feature might shrink:</i> The strength of all spectral features is tied to the scale height of the atmosphere. Scale height is proportional to temperature and inversely proportional to mean molecular weight ($\mu$). If the atmosphere is heavier (i.e., bigger M/H, bigger $\mu$), its scale height will decrease (i.e. it will cling closer to the planet), and consequently all absorption features will decrease in amplitude.
# <br><br>
# The overall strength of any feature is related both to the molecule's abundance (which is increasing as M/H increases) and the atmosphere's scale height (which is decreasing as M/H increases).
# <br><br>
# <i>What's going on with CO$_2$ while M/H changes (at low C/O):</i> When C/O was held constant at a low value of 1, and M/H was varied from 1 to 10 to 100, we saw the CO$_2$ feature at 4.4 $\mu$m increased then decreased. We might infer that it crossed a certain critical M/H threshold, at which point the damping impact of the decreasing scale height overpowered the amplifying impact of the rising abundance.
# <br><br>
# <i>What's going on with CH$_4$ while M/H changes (at high C/O):</i> When C/O was held constant at a high value of 2.5, and M/H was varied from 1 to 10 to 100, we saw the CH$_4$ feature (embedded within H$_2$O continuum) near 3.35 $\mu$m shrink smaller and smaller. Our explorations showed that H$_2$O and CH$_4$ both increase in abundance at the same rate when M/H is increased, thus their features <i>relative to one another</i> should not change as M/H changes. However, as M/H increases, so does the mean molecular weight, and thus the scale height shrinks, causing all features to be damped. This damping explains why the CH$_4$ feature and the surrounding H$_2$O background both diminish in tandem as M/H grows.
# </details>
#
# <br>
# What impact does the change in C/O ratio have on the spectrum?
# <details>
#   <summary><b>Click to expand  solution</b></summary>
# As noted above, we saw different spectral features at low vs. high C/O. We noticed when C/O is low that there is a prominent CO$_2$ feature, whereas when C/O is high we see a prominent CH$_4$ feature. As the C/O ratio increases, there is proportionally more carbon in the atmosphere, facilitating reactions that alter the relative abundance of various carbon-bearing species. One of the most important reactions that takes place is CO$_2$ + H$_2$ $\rightarrow$ CH$_4$ + H$_2$O. Generally, when C/O increases from $<$ to $>$ 2 x Solar, you go from a carbon dioxide dominated system to one dominated by methane.
#
# </details>
# </div>

# %% [markdown] id="dd746bf8-c73e-4406-ae58-1b5286723437"
# ## Revisiting Climate Assumption
#
# 1D Radiative-Convective Equilibrium Models solve for atmospheric structures of brown dwarfs and exoplanets, which includes:
#
# 1\. The Temperature Structure (T(P) profile)
#
# 2\. The Chemical Structure
#
#
# But these physical components are not independent of each other. For example, the chemistry is dependent on the T(P) profile.
#
# `PICASO` tries to find the atmospheric state of your object by taking care of all of these processes and their interconnections self-consistently and iteratively. Therefore, you will find that the climate portion of `PICASO` is slower than running a single forward model evaluation.
#
# ### Modeling the full temperature-pressure profile
#
# Remember our previous assumption of setting the isothermal profile to the equilibrium temperature? Let's improve that by modeling the one-dimensional temperature-pressure structure
#
# #### Correlated-K Tables where to download them?
#
# Earlier when we created a model atmosphere run we needed to edit our call to `opannection`. For climate calculations we need to make sure that we are covering the full range of the planetary energy distribution, which usually amounts to ~0.3-300 $\mu$m. This would be prohibitively slow if we used our same monochromatic opacities. Therefore, we instead use correlated-k tables which are on a very low resolution grid with only 196 wavelength points spanning 0.3-300 $\mu$m. We compute these correlated-K tables **as a function of M/H and C/O**. Therefore, unlike before, we are setting the chemistry of our calculation up front by specifying the correlated-K table.
#

# %%
mh = '1.0'#'+0.0' #log metallicity
CtoO = '0.46'# # CtoO absolute ratio
ck_db = os.path.join(os.getenv('picaso_refdata'),'opacities', 'preweighted', f'sonora_2121grid_feh{mh}_co{CtoO}_NoTiOVO.hdf5')
opacity_ck = jdi.opannection(ck_db=ck_db,method='preweighted') # grab your opacities


# %% [markdown] id="16ae3740"
# #### Effective and Intrinsic Temperatures
#
# You will notice that starting a run is nearly identical as running a spectrum. However, how we will add `climate=True` to our inputs flag, telling PICASO to create a climate. Let's create a new object or "case" where we are running the exact same parameters of the star, planet but now we are going to end with the goal of having a climate model.
#
# New Parameter: **Effective Temperature**. This excerpt from [Modeling Exoplanetary Atmospheres (Fortney et al)](https://arxiv.org/pdf/1804.08149.pdf) provides a thorough description and more reading, if you are interested.
#
# >If the effective temperature, $T_{eff}$, is defined as the temperature of a blackbody of
# the same radius that would emit the equivalent flux as the real planet, $T_{eff}$ and $T_{eq}$
# can be simply related. This relation requires the inclusion of a third temperature,
# $T_{int}$, the intrinsic effective temperature, that describes the flux from the planet’s
# interior. These temperatures are related by:
#
# >$T_{eff} =  T_{int} + T_{eq}$
#
# >We then recover our limiting cases: if a planet is self-luminous (like a young giant
# planet) and far from its parent star, $T_{eff} \approx  T_{int}$; for most rocky planets, or any
# planets under extreme stellar irradiation, $T_{eff} \approx T_{eq}$.
#

# %% id="9560151e"
cl_run = jdi.inputs(calculation="planet", climate = True) # start a calculation

semi_major = 0.0486 # star planet distance, AU
cl_run.star(opacity_ck, temp=5400 , database='phoenix',
         metal=0.01, logg=4.45, radius=0.9, radius_unit=jdi.u.R_sun,
            #note, now we need to know the planet-star separation!
         semi_major= semi_major, semi_major_unit = jdi.u.AU)


cl_run.gravity(mass=0.28, mass_unit=jdi.u.M_jup,
             radius=1.27, radius_unit=jdi.u.R_jup)

# Intrinsic temperature of your planet in K
# Let's keep this fixed for now
tint= 200
cl_run.effective_temp(tint) # input intrinsic temperature


# %% [markdown] id="a67de9ff"
# ### Initial T(P)  Guess
#
# Every calculation requires an initial guess of the pressure temperature profile. The code will iterate from there to find the correct solution. A few tips:
#
# 1. We recommend **using typically 51-91 atmospheric pressure levels**. Too many pressure layers increases the computational time required for convergence. Too few layers makes the atmospheric grid too coarse for an accurate calculation.
#
# 2. Start with **a guess that is close to your expected solution**. One easy way to get fairly close is by using the Guillot et al 2010 temperature-pressure profile approximation.
#

# %% id="d0a47c74"
nlevel = 91 # number of plane-parallel levels in your code

# Let's set the max and min at 1e-4 bars and 500 bars (note below is in log units)
pt = cl_run.guillot_pt(equilibrium_temperature, nlevel=nlevel, T_int = tint, p_bottom=2.5, p_top=-4)
temp_guess = pt['temperature'].values
pressure = pt['pressure'].values

# %% [markdown] id="c4132d98"
# ### Initial Convective Zone Guess
#
# You also need to have a crude guess of the convective zone of your atmosphere. Generally the deeper atmosphere is always convective. Lets crudely assume that the bottom 7 levels of the atmosphere is convective.

# %% [markdown]
# **New Parameters:**
#
# 1. `rfacv`: (See Mukherjee et al Eqn. 20 `r_st`) https://arxiv.org/pdf/2208.07836.pdf
#
# Non-zero values of rst (aka "rfacv" legacy terminology) is only relevant when the external irradiation on the atmosphere is non-zero. In the scenario when a user is computing a planet-wide average T(P) profile, the stellar irradiation is contributing to 50% (one hemisphere) of the planet and as a result rfacv = 0.5. If instead the goal is to compute a night-side average atmospheric state, rfacv is set to be 0. On the other extreme, to compute the day-side atmospheric state of a tidally locked planet rfacv should be set at 1.

# %% id="30cc2bf3"
rcb_guess = 85 # Top most level of guessed convective zone.

# Here are some other parameters needed for the code.
rfacv = 0.5 # Let's assume perfect heat redistribution.

# %% [markdown] id="249a2e67"
# Now we would use the inputs_climate function to input everything together to our cl_run we started.

# %% id="aede1b78"
cl_run.inputs_climate(temp_guess = temp_guess, pressure = pressure,
                      rcb_guess=rcb_guess, rfacv = rfacv)

# %% [markdown] id="ac19b705"
# ### Run the Climate Code
#
#  The actual climate code can be run with the cl_run.run command. The save_all_profiles is set to True to save the T(P) profile at all steps. The code will now iterate from your guess to reach the correct atmospheric solution for your exoplanet.
#
# This will take a few minutes (~3 to 10 min)

# %% id="b3293289"
clima_out = cl_run.climate(opacity_ck, save_all_profiles=True, with_spec=True)

# %% [markdown] id="70c22504"
# Let's visually confirm the convergence through a plot animation. What you should see is the pressure-temperature evolving towards a solution that is in "radiative-convective equilibrium". If we were close to the initial guess then you shouldnt see much movement at all. If you were far away from your initial guess then you should see the pressure-temperature profile wiggle around until it settles on an appropriate solution.

# %% id="ce5b3b97"
ani = jpi.animate_convergence(clima_out, cl_run, opacity_ck,
                              calculation='transmission',
    molecules=['H2O','CH4','CO','CO2'])

# %% id="f4f1bec6"
ani

# %% [markdown]
# <div class="alert alert-block alert-info">
# How are the animations showing a convergence? What is converging? I'm not sure what I'm supposed to get out of these plots.
# </div>

# %% [markdown] id="1ce54cda"
# Nice! Our initial parameterized guess wasn't too far from the final converged solution. Let's add this climate pressure-temperature chemical information we just made back to our original WASP-39b object or "case", and see how it affects our spectrum.

# %% id="fa39d584"
# Let's go back to our initial object or "case" that we were using for our transit spectra.
w39.atmosphere(df=clima_out['ptchem_df'])
df_spec = w39.spectrum(opa, calculation='transmission', full_output=True)
show_spectra({'clima':df_spec})

print(f'Converged Climate Model Chi sq=', chisqr(df_spec ))

# %% [markdown]
# ## Revisiting Cloudless assumption
#
# So far, we've been pretending that WASP-39 b is cloudless. Let's change that by building off our previous `w39` model with the converged climate solution for M/H=10xSolar, C/O=1.
#
# ### First: Simple gray opacity source
#
# We can simply define the slab of a cloud in picaso by using the `.clouds` routine. You will need to learn a few extra parameters
#
# - `g0`: the asymmetry parameter which describes how forward scattering your parameter is. [See this animation here](https://natashabatalha.github.io/picaso_dev#slide02)
# - `w0` : single scattering albedo which describes the magnitude of the scattering source
# - `opd` : the total extinction or optical depth in a layer dp
# - `p` : the bottom location of a cloud deck in LOG10
# - `dp` : the total thickness of the cloud deck above p (in LOG10). Cloud will span 10**(np.log10(p-dp)) and dp should never be negative

# %%
cloud_models = {}
cloud_models['thin_cloud'] = jdi.copy.deepcopy(w39)
cloud_models['thick_cloud'] = jdi.copy.deepcopy(w39)

cloud_models['thin_cloud'].clouds(p=[0.1], dp=[2.5], opd=[1], g0=[0,0.8,1], w0=[0,0.5,1])
cloud_models['thick_cloud'].clouds(p=[1], dp=[4.5], opd=[1], g0=[0], w0=[0])

# %% [markdown]
# We can have a look at the clouds we've created using `justplotit`, but note that the Y axis here is atmosphere `layer`. We set our model up ages ago to have 50 layers in the atmosphere, and the higher layers correspond to lower pressures and higher altitudes. You can see that we've basically added a band of uniform cloud coverage of two different sizes, starting at different altitudes.

# %%
nwno = 196 #this is just the default number
nlayer = cloud_models['thin_cloud'].nlevel-1 #one less than the number of PT points in our input
print('Thin cloud')
fig = jpi.plot_cld_input(nwno, nlayer,df=cloud_models['thin_cloud'].inputs['clouds']['profile'])
print('Thick cloud')
fig = jpi.plot_cld_input(nwno, nlayer,df=cloud_models['thick_cloud'].inputs['clouds']['profile'])

# %%
models = {}
for i in cloud_models:
    models[i] = cloud_models[i].spectrum(opa, calculation='transmission',full_output=True)
    print(f'Chi-Sq of {i}:' ,chisqr(models[i]))
models['cld_free']=df_spec
show_spectra(models)

# %% [markdown]
# Ooh wow! That made a difference on our chi sq. Overall the clouds work to mute the spectral features. W39b needed a little muting to get the features correct.

# %% [markdown] id="4c32f3c9"
# # Mystery absorbery : How to diagnose unfit spectral regions
#
# There is one last feature that we haven't quite got yet in our model! Let's explore what it could be.
#
# ## Explore opacity database
#
# To plot the raw opacities we can use a new picaso function called the `opacity_factory`

# %%
import picaso.opacity_factory as opaf
mols, pts = opaf.molecular_avail(opa.db_filename)

# %% [markdown]
# Let's see everything that we have available to see what could be absorbing at 4 microns.

# %%
t = [1000]#Kelvin
p = [0.1]#bars
data  = opaf.get_molecular(opa.db_filename, mols, t,p)


#plot
Spectral11  = jpi.pals.Spectral11
f = jpi.figure(y_axis_type='log',height=500,y_range=[1e-23,1e-17],
          y_axis_label='Cross Section (cm2/species)', x_axis_label='Micron')
for imol,C in zip(mols,Spectral11):
    x,y = jdi.mean_regrid(data['wavenumber'],data[imol][t[0]][p[0]], R=200)
    f.line(1e4/x,y, color=C,legend_label=imol,line_width=2)
jpi.show(f)

# %% [markdown]
# Check Understanding
#
# Which molecules look like they could be responsible for the 4 micron feature?
#

# %% [markdown]
# ## Create Model with Fixed Chemistry

# %% [markdown]
# Let's create an object or "case" where SO$_2$ is present and see if that will create a model with a better fit to the data.

# %% id="e036f764"
so2_case = jdi.copy.deepcopy(w39)
#add in the cloud we liked
so2_case.clouds(p=[0.1], dp=[2.5], opd=[1], g0=[0,0.8,1], w0=[0,0.5,1])
# Let's explore
so2_abundance = [1e-6,5e-6,1e-5]
so2_models = {}
for iso2 in so2_abundance:
    so2_case.inputs['atmosphere']['profile']['SO2']=iso2
    df_spec_so2 = so2_case.spectrum(opa, calculation='transmission',full_output=True)
    so2_models[f'{iso2}']=df_spec_so2
    print(f'Chi-Sq of SO2={iso2}:' ,chisqr(df_spec_so2))
show_spectra(so2_models)

# %% [markdown]
# <div class="alert alert-block alert-info">
# Looks like adding sulfur dioxide improved our fit!
# </div>

# %% [markdown] id="c7a9e749"
# # Xarray output (.nc) the model

# %% [markdown]
# Now that we have our models created and all done, let's save our climate model as an [Xarray](https://docs.xarray.dev/en/stable/). This is a great tool to neatly wrap all of your models. This is very useful because we can continue our analysis from low-fidelty PICASO to the high fidelty bayesian fitting (which is the next tutorial)! It also allows anyone else to take a look and analyze the models you have created very easily.

# %% [markdown]
# ## Save W39 case w/o SO2

# %%
savefile="W39b_climate.nc"
xarr_no_so2=jdi.output_xarray(df_spec, w39, savefile=savefile)

# %%
xarr_no_so2

# %% [markdown]
# ### Access Xarray values similarly to a regular array

# %%
xarr_no_so2.keys()

# %%
xarr_no_so2['temperature']

# %% [markdown]
# ## Save SO2 case

# %%
savefile="W39b_climate_so2.nc"
xarr_so2=jdi.output_xarray(df_spec_so2, so2_case, savefile=savefile)

# %%
xarr_so2

# %%
xarr_so2['pressure']

# %% [markdown]
# Note that this has been a simplfied analysis of WASP-39b. A further analysis would include thick/thin clouds and photochemistry, which is below!

# %% [markdown]
# # Cloud Modeling with Virga

# %% [markdown]
# Unfortunately, not every analysis will be this simple! In the case of WASP-39b, there may be some extra chemicals in the atmosphere (like SO2 we just added!) or even clouds we will have to "inject" into the model after the core parameters. In this section, we will be focusing on the basics of `virga` and how clouds are introduced and affect the complexity of your model.

# %% [markdown]
# You should already have [virga](https://natashabatalha.github.io/virga/installation.html) installed from the notebook `AnalyzeExoplanet_1_picaso_setup`. Let's first begin by seeing what gas condensates we can work with.

# %% [markdown]
# Let's import virga and re-open that PICASO analysis we just did above.

# %%
# Here is virga
import virga.justdoit as vjdi
import virga.justplotit as vjpi

# %%
xarr_no_so2 = jdi.xr.open_dataset("W39b_climate.nc")
xarr_no_so2

# %% [markdown]
# ### Choosing a Gas Condensate

# %%
# Check gas species to choose from
vjdi.available()

# %% [markdown]
# We have a lot to choose from! If you already know which you want present, then you can skip past to the next section. Otherwise, let's see what recommended gas condensates virga believes we should look into more.

# %%
pressure = xarr_no_so2['pressure'].values
temperature = xarr_no_so2['temperature'].values
metallicity = 1 # Atmospheric metallicity relative to Solar (for anything above 1, you will need to do your own chemical equilbrium)
mean_molecular_weight = 2.2 # Atmospheric mean molecular weight

# Get virga recommendation for which gases to run
recommended = vjdi.recommend_gas(pressure, temperature, metallicity, mean_molecular_weight,plot=True)
# Print the results
print(recommended)

# %% [markdown]
# Virga recommends that we use the following gases above. Although Virga recommends it, it doesn't mean you need to have it or it is even there physically. It is simply stating that these gases can physically exist, although they may not be present. For the sakes of this tutorial, we will be choosing *MnS, Na2S, MgSi03* to replicate what was done in the paper.

# %% [markdown]
# ### Running Virga

# %% [markdown]
# To run Virga we need two additional parameters: fsed and kzz.
#
# fsed describes the vertical extend of the cloud. Small fsed will create small particles that are lofted through large vertical extents. Large fseds will rainout particles at a faster rate creating large particles and vertically thin clouds.
#
# kzz is the vertical mixing parameter and describes the velocity of vertical mixing. The particle radii are largely determined through how rapid this mixing parameter is. Low kzz create "stagnant" atmospheres (effectively no wind) with small particles and large kzzs create large particles.

# %%
#we cam use the picaso input xarray function to setup our case with what we did previously
w39_clouds = jdi.input_xarray(xarr_no_so2,opa)

#the main thing we need to add is a parameter called kzz
w39_clouds.inputs['atmosphere']['profile']['kz'] = 1e10

#now lets add Virga
all_out = w39_clouds.virga(condensates=['MnS', 'Na2S', 'MgSiO3'],directory = mieff_dir,
                  fsed=1, mh=metallicity,
                  mmw = mean_molecular_weight)


# %% [markdown]
# Now that we have run the code and computed the cloud modeling, there is many different plots we can call from the `virga.justplotit` class to see what we are looking at exactly! But first, let's get a read of what variables and information we have access to.

# %%
all_out.keys()

# %% [markdown]
# Lot's of choices to index to and see, but let's visually plot and gain more information of what the clouds will look like.

# %% [markdown]
# ### Checking and Proving Gas Condensate

# %% [markdown]
# The first plot we can create is our P-T profile with all of the chemicals respective P-T profiles. The thick lines are the gasses that we defined two cells above to be present in our cloud modeling. Anything to the right of the black dotted line, the user input, is a gas that more than likely can condensate in the atmosphere.
#
# As we can see, *MgSi02* and *MnS* condensates, as their P-T profiles are to the right of ours. But, we can see that *Na2s* is to the **left** of ours, which means that it is **not** condensating in the cloud mixture.

# %%
jpi.show(vjpi.pt(all_out))

# %% [markdown]
# <div class="alert alert-block alert-info">
# Each solid line represents the condensation temperature profile for a given molecule or element. Take the thick red MgSiO3 line as an example. If an atmosphere's temperature pressure conditions fall to the right of the red line (higher temp or lower pressure), any MgSiO3 present in the atmosphere will be in its gaseous state. If atmospheric temperature pressure conditions fall to the left of the red line (lower temp or higher pressure), MgSiO3 will be in its liquid state (condensate). The dotted black line represents the conditions of WASP-39 b's atmosphere. We can infer that at high alitudes (lower pressure), WASP-39 b's atmospheric conditions are left (colder) than the red line, thus MgSiO3 may condensate. At lower altitudes (higher pressure), WASP-39 b's atmospheric conditions are right (warmer) than the red line, thus MgSiO3 will remain gaseous.
# </div>

# %% [markdown]
# We can see a breakdown of each of the cloud optical depth by species. Note Na2S never condensed, as we expected from the above plot.

# %%
jpi.show(vjpi.opd_by_gas(all_out))

# %% [markdown]
# Another way to look at this is through the condensate mass mixing ratios.

# %%
jpi.show(vjpi.condensate_mmr(all_out))

# %% [markdown]
# ### Analyzing Cloud Interactions

# %% [markdown]
# Next, we can take visualizing the single scattering albedo, optical depth, and the asymmetry. Let's dissect each of these and see what they mean exactly.
#
# - Single Scattering Albedo
#
# This plots the pressure against the wavelength to find where the albedo is strongest (i.e. where in the atmosphere higher or lower fractions of light are being reflected). We see that near the top, the albedo is close to or is 1. This means that most of the light absorbed at this layer is reflected, which makes sense! Planets, specifically puffy planets like WASP-39 b, will reflect a lot of light near the top. As we go deeper, we can see that below 1.4E-3 bars the albedo begins to drop a bit. As the light drops deeper into the atmosphere, it is a lower fractional reflection. Moving from left to right can also tell us how much light is being reflected in the infrared, visual and ultraviolet!
#
#
# - Cloud Optical Depth
#
# Optical depth refers to how optically thick the cloud is. For instance, on this middle plot it can be seen at the lower pressures near the top there is a lot of light being let in. As we go deeper in the atmosphere, the clouds begin to block more and more light.
#
# - Asymmetry Parameter
#
# Describes the scattering direction of the particles. Particles which high asymmetry scatter light in the forward directly. Particles with asymmetry near 0 scatter in all directions equally.

# %%
jpi.show(vjpi.all_optics(all_out))

# %% [markdown]
# Awesome! This tells us so much about the interaction of clouds between the atmosphere, and is a much different analysis than a cloud free model. This is just a brief overview of what virga can do and the amount of science that can be retrieved from just one set of JWST data. For more information, please feel free to visit the [virga](https://natashabatalha.github.io/virga/index.html) documentation!
