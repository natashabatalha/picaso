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

# %% [markdown]
# # Adding your Own Chemistry in PICASO
#
# In this tutorial you will learn how ro run simple chemical equilibrium using our pre-computed grids. Plus add additional disequilibrium 'hacks' that are used within our climate code. 
#

# %%
import os

import warnings
warnings.filterwarnings('ignore')
import picaso.justdoit as jdi
import picaso.justplotit as jpi
jpi.output_notebook()
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# %% [markdown]
# ## Chemical equilibrium by interpolating the Visscher pre-computed grid
#
# The code below demonstrates how to set the atmospheric composition to chemical equilibrium by interpolating the Visscher grid of pre-computed equilibrium abundances.

# %%
opacity = jdi.opannection()

chem_example = jdi.inputs(calculation='browndwarf')
chem_example.gravity(gravity=1000, gravity_unit=u.Unit('m/(s**2)')) # input gravity
chem_example.guillot_pt(Teq=1200) # Add a P-T profile with a simple parameterization

# Compute chemical equilibrium by interpolating the visscher grid
chem_example.chemeq_visscher_2121(log_mh=0, cto_absolute=0.549) 

# Compute spectrum
full_out = chem_example.spectrum(opacity,calculation='thermal',full_output=True)

jpi.show(jpi.mixing_ratio(full_out['full_output'], limit=15)) # plot top 15 chemical species

# %% [markdown]
# ## Chemical equilibrium with on-the-fly calculations
#
# The code below demonstrates how to set the atmospheric composition to chemical equilibrium by computing it on-the-fly with the equilibrium chemistry solver in the `photochem` package (if needed install `photochem` with `conda install -c conda-forge photochem`). By default, the solver uses thermodynamics that approximate the full set of thermodynamics in Channon Visscher's Sonora grid. The results from the on-the-fly calculation should be close to the result from `chemeq_visscher_2121`.
#

# %%
opacity = jdi.opannection()

chem_example = jdi.inputs(calculation='browndwarf')
chem_example.gravity(gravity=1000, gravity_unit=u.Unit('m/(s**2)')) # input gravity
chem_example.guillot_pt(Teq=1200) # Add a P-T profile with a simple parameterization

# Compute chemical equilibrium on the fly
chem_example.chemeq_on_the_fly(log_mh=0, cto_absolute=0.549, method='sonora-approx') # method='sonora-approx' is default

# Compute spectrum
full_out = chem_example.spectrum(opacity,calculation='thermal',full_output=True)

jpi.show(jpi.mixing_ratio(full_out['full_output'], limit=15)) # plot top 15 chemical species

# %% [markdown]
# # Comparing Other Ways to get Chemical Equilibrium incl w/ Disequilibrium Hacks
#
# For climate calculations especially, we often are using chemical equilibrium from pre-computed correlated k tables. This next section relies on you have downloaded the `preweighted` correlated k tables and `resortrebin` files. 

# %%
#1 ck tables from roxana
mh = '0.0'#'+0.0' #log metallicity
CtoO = '0.55'# # CtoO absolute ratio
ck_db = os.path.join(os.getenv('picaso_refdata'),'opacities', 'preweighted', f'sonora_2121grid_feh{mh}_co{CtoO}.hdf5')

# %% [markdown]
# Let's create two different opacity connections so that we can look at the chemical equilibrium that is encoded within the preweighted CK tables as well as the chemistry that is loaded from the chemistry tables.

# %%
cl_run = jdi.inputs(calculation="browndwarf", climate = True) # start a calculation

#note you need to put the climate keyword to be True in order to do so
# now you need to add these parameters to your calculation

teff= 400 # Effective Temperature of your Brown Dwarf in K
grav = 1000 # Gravity of your brown dwarf in m/s/s

cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity
cl_run.effective_temp(teff) # input effective temperature

opacity_preweighted =  jdi.opannection(method='preweighted',ck_db=ck_db)

# %%
nlevel = 91 # number of plane-parallel levels in your code

# Here we're going to start with a cloudfree Sonora Elf Owl model
pressure,temp = np.loadtxt("profilegrid_kz_1d9_qt_onfly_400_grav_1000_mh_+0.0_cto_1.0.dat",
                                usecols=[1,2],unpack=True, skiprows = 1)
#get our PT profile sorted 
cl_run.add_pt(P= pressure, T= temp)


# %% [markdown]
# ## Option 1: Chemistry from the Pre-Weighted CK Tables
#
# Dont need to run this cell if you haven't downloaded the preweighted correlated k files

# %%
cl_run.premix_atmosphere(opa=opacity_preweighted)
df_pre = jdi.copy.deepcopy(cl_run.inputs['atmosphere']['profile'])

# %% [markdown]
# ## Option 2: Chemistry from Chemeq Visscher Functions

# %%
cl_run.chemeq_visscher_2121(log_mh=0, cto_absolute=0.549)#absolute c/o
df_2121 = jdi.copy.deepcopy(cl_run.inputs['atmosphere']['profile'])

#or older visscher data 
cl_run.chemeq_visscher_1060(c_o=1, log_mh=0)#relative c/o!!!
df_1060 = jdi.copy.deepcopy(cl_run.inputs['atmosphere']['profile'])

# %% [markdown]
# ## Option 3: Chemistry from on-the-fly equilibrium calculations

# %%
cl_run.chemeq_on_the_fly(log_mh=0, cto_absolute=0.549)
df_onfly = jdi.copy.deepcopy(cl_run.inputs['atmosphere']['profile'])

# %% [markdown]
# ## Option 4: Chemistry from Atmosphere Function with Disequilibrium Hacks
#
# Note the use of chem_method below: "visscher" defaults to the newest visscher grid which is the 2121 grid. 'visscher_1060's is the older 1060 grid. 

# %%
#no hacks turned on (should be identical to option 2)
cl_run.atmosphere(mh=1, cto_relative=1, chem_method='visscher',
                  quench=False, cold_trap = False, no_ph3 = False, vol_rainout=False)
quench_level = {'CO-CH4-H2O': np.int64(73), 'CO2': np.int64(66), 'NH3-N2': np.int64(74), 'HCN': np.int64(73), 'PH3': np.int64(71)}
cl_run.premix_atmosphere(opa=None,quench_levels=quench_level,verbose=True)
df_none = jdi.copy.deepcopy(cl_run.inputs['atmosphere']['profile'])

#turn on quenching
cl_run.atmosphere(mh=1, cto_relative=1, chem_method='visscher',
                  quench=True, cold_trap = False, no_ph3 = False, vol_rainout=False)
quench_level = {'CO-CH4-H2O': np.int64(73), 'CO2': np.int64(66), 'NH3-N2': np.int64(74), 'HCN': np.int64(73), 'PH3': np.int64(71)}
cl_run.premix_atmosphere(opa=None,quench_levels=quench_level,verbose=True)
df_quench_only = jdi.copy.deepcopy(cl_run.inputs['atmosphere']['profile'])

#turn on quenching and volatile rainout water 
cl_run.atmosphere(mh=1, cto_relative=1, chem_method='visscher',
                  quench=True, cold_trap = False, no_ph3 = True, vol_rainout=True)
quench_level = {'CO-CH4-H2O': np.int64(73), 'CO2': np.int64(66), 'NH3-N2': np.int64(74), 'HCN': np.int64(73), 'PH3': np.int64(71)}
cl_run.premix_atmosphere(opa=None,quench_levels=quench_level,verbose=True)
df_quench_rain= jdi.copy.deepcopy(cl_run.inputs['atmosphere']['profile'])

#turn on quenching and volatile rainout water and cold trap water 
cl_run.atmosphere(mh=1, cto_relative=1, chem_method='visscher',
                  quench=True, cold_trap = True, no_ph3 = True, vol_rainout=True)
quench_level = {'CO-CH4-H2O': np.int64(73), 'CO2': np.int64(66), 'NH3-N2': np.int64(74), 'HCN': np.int64(73), 'PH3': np.int64(71)}
cl_run.premix_atmosphere(opa=None,quench_levels=quench_level,verbose=True)
df_all= jdi.copy.deepcopy(cl_run.inputs['atmosphere']['profile'])

# %%
colors = jpi.pals.Light8
FIGURES = []
f = jpi.figure(y_range=[5000,1e-3], y_axis_type='log',x_axis_label='temperature',y_axis_label='pressure')
f.line(temp, pressure,color='black', line_width=4)
jpi.plot_format(f)
FIGURES+=[f]

for mol in ['NH3', 'H2O','CH4','CO']:
    f = jpi.figure(y_range=[5000,1e-3], y_axis_type='log',x_axis_type='log',x_axis_label=mol,y_axis_label='pressure')
    linwidth=8
    ii=0
    for i ,il in zip([df_none, df_quench_only, df_quench_rain, df_all,df_pre,df_2121, df_1060, df_onfly],['chemeq', 'quench_only', 'quench+rain','quench+rain+cold trap','preweighted','new_visscher_Lo20','visscher_1060', 'on-the-fly']):
        f.line(i[mol],i['pressure'],legend_label=il,line_width=linwidth, color=colors[ii]);linwidth-=1;ii+=1
    jpi.plot_format(f)
    FIGURES+=[f]
jpi.show(jpi.gridplot([FIGURES[0:1],FIGURES[1:3],FIGURES[3:]])
         )

# %%
