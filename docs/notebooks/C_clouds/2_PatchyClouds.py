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
# # Patchy Clouds
#
# We've shown how to use Virga, and how to input simple box clouds. In this tutorial you will learn how to do a simple patchy cloud for thermal, transmission, and/or reflected light.
#
# The way our formalism works is that to run a patchy cloud, you turn on clouds as normal. Then you define "holes" of clearness. The clear hole can either be a fully clear patch, or it can be a partially cloudy patch.
#
# ## Patchy Cloud Parameters for `clouds` and `virga` functions
#
# - `do_holes` : Turns on patchiness
# - `fhole` [0-1] : defines the fraction of the cloud hole. 0 would be a fully cloudy atmosphere. 1 would be one giant hole.
# - `fthin_cld` : this is going to scale the "hole" region. If you just want a purely clear hole then you would do fthin_cld=0.
#
# So in practice let's review some simple cases:
#
# - Inputs for a 50/50 cloudy, clear: do_holes=True, fhole=0.5, fthin_cld=0
# - Inputs for a 50% cloudy, 50% thinned cloud by 30%:  do_holes=True, fhole=0.5, fthin_cld=0.3
#

# %%
import numpy as np
import pandas as pd

#picaso
from picaso import justdoit as jdi
from picaso import justplotit as jpi

#plotting
jpi.output_notebook()


# %% [markdown]
# Let's do some calculations for practice

# %%
opa = jdi.opannection(wave_range=[0.6,5])

case1 = jdi.inputs()

case1.phase_angle(0)

case1.approx(p_reference=10)
#here we are going to have to specify gravity through R and M since we need it in the Flux calc
case1.gravity(mass=1, mass_unit=jdi.u.Unit('M_jup'),
              radius=1.2, radius_unit=jdi.u.Unit('R_jup'))

#here we are going to have to specify R as well
case1.star(opa, 4000,0.0122,4.437,radius=0.7, radius_unit = jdi.u.Unit('R_sun') )

#atmo
case1.atmosphere(filename = jdi.HJ_pt(), sep=r'\s+')

#clear transmission, thermal emission, and reflected light
df_clear= case1.spectrum(opa, full_output=True,calculation='transmission+thermal+reflected')

# %% [markdown]
# ## Patchy Clouds
#
# Now lets do a 50/50 patchy cloud and compare to a fully cloudy run

# %%
case1.clouds(p = [1], dp=[4], opd=[1], g0=[0], w0=[0],do_holes=False)
df_full_cld = case1.spectrum(opa, full_output=True,calculation='transmission+thermal+reflected') #note the new last key


case1.clouds(p = [1], dp=[4], opd=[1], g0=[0], w0=[0],do_holes=True, fhole=0.5,fthin_cld=0)
df_50p_cld= case1.spectrum(opa, full_output=True,calculation='transmission+thermal+reflected') #note the new last key

# %% [markdown]
# ### Compare Transit Depth

# %%
to_compare = dict(clear=df_clear, cldy_full = df_full_cld, cldy_50 = df_50p_cld)
wnos=[];trs=[]
for i in to_compare.keys():
    wno,r =jdi.mean_regrid(to_compare[i]['wavenumber'], to_compare[i]['transit_depth'], R=150)
    r = 1e6*(r-np.mean(r))
    wnos+=[wno]; trs+=[r]

jpi.show(jpi.spectrum(wnos,trs,
                  legend=to_compare.keys(),
                  plot_width=500))

# %% [markdown]
# ### Compare Albedo Spectra

# %%
to_compare = dict(clear=df_clear, cldy_full = df_full_cld, cldy_50 = df_50p_cld)
wnos=[];albs=[]
for i in to_compare.keys():
    wno,r =jdi.mean_regrid(to_compare[i]['wavenumber'], to_compare[i]['albedo'], R=150)
    wnos+=[wno]; albs+=[r]

jpi.show(jpi.spectrum(wnos,albs,
                  legend=to_compare.keys(),x_range=[0.6,2],
                  plot_width=500))

# %% [markdown]
# ### Compare Thermal Spectra

# %%
to_compare = dict(clear=df_clear, cldy_full = df_full_cld, cldy_50 = df_50p_cld)
wnos=[];sps=[]
for i in to_compare.keys():
    wno,r =jdi.mean_regrid(to_compare[i]['wavenumber'], to_compare[i]['thermal'], R=150)
    wnos+=[wno]; sps+=[r]

jpi.show(jpi.spectrum(wnos,sps,
                  legend=to_compare.keys(),
                  plot_width=500))
