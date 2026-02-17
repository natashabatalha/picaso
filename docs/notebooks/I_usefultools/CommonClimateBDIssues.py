# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Climate Model Common Issues
# In this tutorial, we're going to highlight some of the most common issues that you could potentially encounter when running 1D climate models and what are the general recommendations to fix it! **You don't neccesarily need to rerun this notebook**, this is more for you to look at to see what issues might look like in your profiles and diagnostic plots.
#
# Now let's take a look at what weird things might happen in your climate models!

# %%
import os
import warnings
warnings.filterwarnings('ignore')
import picaso.justdoit as jdi
import picaso.justplotit as jpi
import virga.justdoit as vj
import virga.justplotit as cldplt
jpi.output_notebook()
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import xarray
from bokeh.plotting import show, figure

# %% [markdown]
# ## Starting Too High of Radiative-Convective Boundary Guess
#
# Now let's say you tried starting with a rc boundary guess higher up in the profile since you don't want it to be too deep in the atmosphere. If you start with a guess too high up in the atmosphere you'll notice some odd behaviours as well. This is very common in both clear and cloudy models but fortunately it's very easy to visually catch this issue majority of the time. Let's look at this with a clear atmospheric model.

# %%
#1 ck tables from roxana
mh = '+000'#'+0.0' #log metallicity
CtoO = '100'#'1.0' # CtoO ratio

ck_db = f"/Users/jjm6243/Documents/freedman/sonora_2020_feh{mh}_co_{CtoO}.data.196"

# ck_db = os.path.join(os.getenv('picaso_refdata'),'opacities', 'preweighted',f'sonora_2020_feh{mh}_co_{CtoO}.data.196')

# %%
cl_run = jdi.inputs(calculation="browndwarf", climate = True) # start a calculation

teff= 200 # Effective Temperature of your Brown Dwarf in K
grav = 1000 # Gravity of your brown dwarf in m/s/s

cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity
cl_run.effective_temp(teff) # input effective temperature

opacity_ck = jdi.opannection(ck_db=ck_db, method='preweighted') # grab your opacities

# %%
nlevel = 91 # number of plane-parallel levels in your code

pressure_bobcat,temp_bobcat = np.loadtxt(jdi.os.path.join(
                            sonora_profile_db,f"t{teff}g{grav}nc_m0.0.cmp.gz"),
                            usecols=[1,2],unpack=True, skiprows = 1)

# %% [markdown]
# Maybe you didn't want to wait forever for your model to run so you didn't want to start with a very deep `nstr_upper`, what happens if we start with a guess at let's say layer 45 (reminder in these tutorials we have 91 layers in our atmosphere)

# %%
rcb_guess = 45 # top most level of guessed convective zone

# Here are some other parameters needed for the code.
rfacv = 0.0 #we are focused on a brown dwarf so let's keep this as is

# %%
cl_run.inputs_climate(temp_guess= temp_bobcat, pressure= pressure_bobcat,
                      rcb_guess=rcb_guess, rfacv = rfacv)

# %%
out = cl_run.climate(opacity_ck, save_all_profiles=True,with_spec=True)

# %%
pressure_bobcat,temp_bobcat = np.loadtxt(jdi.os.path.join(
                            sonora_profile_db,f"t{teff}g{grav}nc_m0.0.cmp.gz"),
                            usecols=[1,2],unpack=True, skiprows = 1)
plt.figure(figsize=(8,6))
plt.ylabel("Pressure [Bars]")
plt.xlabel('Temperature [K]')
plt.xlim(0,max(out['temperature'])+50)
plt.ylim(3e3,1e-3)

plt.semilogy(temp_bobcat,pressure_bobcat,color="k",linestyle="--",linewidth=3,label="Sonora Bobcat")

plt.semilogy(out['temperature'],out['pressure'],label="Our Climate Run")

plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# So as you can see here there's this large jump in the PT profile where the profile is following the adiabat too far up. This is a clear visual clue that you need to start with a deeper `nstr_upper` initial guess, even though it says this climate model ended up converging.
#
# Sometimes it's not as visually noticeable as this so a good way to tell if these models are good or not is by looking at dT/dP

# %%
cp, grad, dtdp, layer_p= jpi.pt_adiabat(out,cl_run,opacity_ck)

# %% [markdown]
# So here in this case, you can see that dT/dP actually goes super negative. Anytime you see these going really far negative, it usually points to a poorly converged model.
#
# Sometimes if you start way too high, the convective zone will reach the top of the atmosphere, PICASO will send you an error and the run will fail if that occurs.
#
# **Recommendation**:
# Generally, a conservative starting rc boundary guess is 83 (if doing 91 levels). The actual layer of where the rc boundary is going to vary depending on the surface gravity, whether there are clouds or not, the temperature of the object, etc.

# %% [markdown]
# ### Solution:
#
# Now let's fix this by running a case where we have a better rc boundary guess that isn't too high and we can see what good diagnostic plots should look like

# %%
#1 ck tables from roxana
mh = '+000'#'+0.0' #log metallicity
CtoO = '100'#'1.0' # CtoO ratio

# ck_db = f'/Users/nbatalh1/Documents/data/kcoeff_asci/sonora_2020_feh{mh}_co_{CtoO}.data.196'
ck_db = os.path.join(os.getenv('picaso_refdata'),'opacities', 'preweighted',f'sonora_2020_feh{mh}_co_{CtoO}.data.196')

# %%
cl_run = jdi.inputs(calculation="browndwarf", climate = True) # start a calculation

teff= 200 # Effective Temperature of your Brown Dwarf in K
grav = 1000 # Gravity of your brown dwarf in m/s/s

cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity
cl_run.effective_temp(teff) # input effective temperature

opacity_ck = jdi.opannection(ck_db=ck_db, method='preweighted') # grab your opacities

# %%
nlevel = 91 # number of plane-parallel levels in your code

pressure_bobcat,temp_bobcat = np.loadtxt(jdi.os.path.join(
                            sonora_profile_db,f"t{teff}g{grav}nc_m0.0.cmp.gz"),
                            usecols=[1,2],unpack=True, skiprows = 1)

# %%
rcb_guess = 60 # top most level of guessed convective zone

# Here are some other parameters needed for the code.
rfacv = 0.0 #we are focused on a brown dwarf so let's keep this as is

# %%
cl_run.inputs_climate(temp_guess= temp_bobcat, pressure= pressure_bobcat,
                      rcb_guess=rcb_guess, rfacv = rfacv)

# %%
out = cl_run.climate(opacity_ck, save_all_profiles=True,with_spec=True)

# %%
pressure_bobcat,temp_bobcat = np.loadtxt(jdi.os.path.join(
                            sonora_profile_db,f"t{teff}g{grav}nc_m0.0.cmp.gz"),
                            usecols=[1,2],unpack=True, skiprows = 1)
plt.figure(figsize=(8,6))
plt.ylabel("Pressure [Bars]")
plt.xlabel('Temperature [K]')
plt.xlim(0,max(out['temperature'])+50)
plt.ylim(3e3,1e-3)

plt.semilogy(temp_bobcat,pressure_bobcat,color="k",linestyle="--",linewidth=3,label="Sonora Bobcat")

plt.semilogy(out['temperature'],out['pressure'],label="Our Climate Run")

plt.legend()
plt.tight_layout()
plt.show()

# %%
cp, grad, dtdp, layer_p= jpi.pt_adiabat(out,cl_run,opacity_ck)

# %% [markdown]
# ### Checking if You Reached Radiative-Convective Equilibrium (RCE)
#
# Another indicator, on top of the lapse rate figure, that indicates your model isn't well converged even though it says `YAY ENDING WITH CONVERGENCE` is to look at the F_{net}/F_{IR} output. This will indicate to us that your model might not be in radiative-convective equilibrium (RCE).
#
# Let's check whether we achieved RCE in this case. In the convective regions for these climate models, the flux coming out of the convective layers should be in the IR. This isn't the case for the radiative layers. So when we look at the Fnet/F_IR plot, you should see a chair like behaviour where at the top of the atmosphere, there are low values of Fnet/F_IR (~1e-3 to 1e-5). Then, near the rc boundary, it should sharply increase since this is the beginning of the region where convection dominates.

# %%
plt.figure(figsize=(8,6))
plt.ylabel("Pressure [bar]")
plt.xlabel(r'F$_{\rm net}$/F$_{\rm net IR}$')
plt.ylim(1e4,1e-4)

plt.loglog(abs(out['fnet/fnetir']),out['pressure'], label = "Our Climate Run")
plt.axhline(y=pressure_bobcat[out['cvz_locs'][1]],color="k",linestyle="--",linewidth=3, label = 'RC Boundary')

plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# So here we see the really nice behavior we expected in a well converged model in the Fnet/Fnet-IR diagnostic plots. Keep these two diagnostic plots in mind when running models for a sanity check!
#
# Since you can see there's still some interesting behaviour at the top of the atmosphere both in the PT profile and the adiabat, sometimes you might have to restart with the resultant profile and start with an rc boundary guess a couple layers deeper in the atmosphere to reconverge the profile. Other things to tune within the code include the `egp_stepmax` temperature threshold, but consult with your Sonora team member to discuss this further.
