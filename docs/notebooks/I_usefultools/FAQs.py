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

# %%
import warnings
warnings.filterwarnings('ignore')
import numpy as np
#picaso
from picaso import justdoit as jdi
from picaso import justplotit as jpi
from bokeh.io import output_notebook
output_notebook()

# %% [markdown]
# # How do I load in target properties from Exo.Mast?

# %%
nexsci = jdi.all_planets()

# %%
nexsci.head()

# %%
nexsci.loc[nexsci['hostname']=='HAT-P-26']

# %%
#first isolate the row
hatp26_row = nexsci.loc[nexsci['hostname']=='HAT-P-26']

#add it to load planet function with opacity
opa = jdi.opannection(wave_range=[1,5])
hatp26 = jdi.load_planet(hatp26_row,opa , st_metfe=0)#hatp26 is misst st_metfe so we must add it as a keyword

# %% [markdown]
# Load planet function adds in:
#     - planet properties and stellar properties
#     - parameterized PT profile
#
# That means you still need to add in the chemistry and cloud parameters

# %%
absolute_co=0.55#solar
logfeh = 1.5 #10^1.5 ~ 30x solar
hatp26.chemeq_visscher_2121(absolute_co,logfeh) #adds in template chemistry from sonora
df = hatp26.spectrum(opa, calculation='transmission+thermal')

# %%
x,y = jdi.mean_regrid(df['wavenumber'], df['fpfs_thermal'],R=150)
plot = [jpi.spectrum(x,y,plot_width=400, y_axis_type='log',y_axis_label='FpFs')]
x,y = jdi.mean_regrid(df['wavenumber'], df['transit_depth'],R=150)
plot += [jpi.spectrum(x,y,plot_width=400,y_axis_label='(Rp/Rs)^2')]

jpi.show(jpi.row(plot))

# %% [markdown]
# # How do I access the pressure-temperature profile parameterizations?

# %%
#start by loading in some template properties
opa = jdi.opannection(wave_range=[1,5])
hatp26_row = nexsci.loc[nexsci['hostname']=='HAT-P-26']
hatp26 = jdi.load_planet(hatp26_row,opa, st_metfe=0)

# %% [markdown]
# ### How do the input parameters effect the parameterization?

# %%
fig = [jpi.figure(width=300, height=300, y_axis_type='log', y_range=[100,1e-6],x_range=[400,1700],
                  y_axis_label='Pressure(bars)',x_axis_label='Temperature (K)',
                 title='Effect of logg1')]
colors = jpi.pals.magma(10)
for i, logg1 in enumerate(np.linspace(-2,2,10)):
    hatp26.guillot_pt(576.17, T_int=100, logg1=logg1, logKir=-1.5)
    fig[0].line(hatp26.inputs['atmosphere']['profile']['temperature'],
            hatp26.inputs['atmosphere']['profile']['pressure'],color=colors[i])

fig += [jpi.figure(width=300, height=300, y_axis_type='log', y_range=[100,1e-6],x_range=[400,1700],
                 x_axis_label='Temperature (K)',title='Effect of LogKir')]
for i,logKir in enumerate(np.linspace(-2,2,10)):
    hatp26.guillot_pt(576.17, T_int=100, logg1=-2, logKir=logKir)
    fig[1].line(hatp26.inputs['atmosphere']['profile']['temperature'],
            hatp26.inputs['atmosphere']['profile']['pressure'], color=colors[i])

fig += [jpi.figure(width=300, height=300, y_axis_type='log', y_range=[100,1e-6],x_range=[400,1700],
                 x_axis_label='Temperature (K)',title='Effect of Tint')]
for i,T_int in enumerate(np.linspace(50,300,10)):
    hatp26.guillot_pt(576.17, T_int=T_int, logg1=-2, logKir=-2)
    fig[2].line(hatp26.inputs['atmosphere']['profile']['temperature'],
            hatp26.inputs['atmosphere']['profile']['pressure'], color=colors[i])

jpi.show(jpi.row(fig))


# %% [markdown]
# # How do I return the output from photon attenuation plot?

# %%
#first isolate the row
hatp26_row = nexsci.loc[nexsci['hostname']=='HAT-P-26']

#add it to load planet function with opacity
opa = jdi.opannection(wave_range=[0.3,1])
hatp26 = jdi.load_planet(hatp26_row,opa ,st_metfe=0)
co_absolute = 0.55
logfeh=2
hatp26.chemeq_visscher_2121(co_absolute,logfeh)
df = hatp26.spectrum(opa, calculation='reflected',full_output=True)

# %%
plot,wave,at_pressures_gas,at_pressures_cld,at_pressures_ray = jpi.photon_attenuation(df['full_output'],
                                                                                     return_output=True)

# %%
#now you can make your own plot!
fig = jpi.figure(y_axis_type='log',y_range=[10,1e-3],height=250, width=500)
for i,iy,ilab in zip([0,1],[at_pressures_gas,at_pressures_ray ], ['Molecular','Rayleigh']):
    x,y = jpi.mean_regrid(1e4/wave,iy, R=150)
    fig.line(1e4/x,y,legend_label=ilab, line_width=4, color=jpi.Colorblind8[i])
jpi.show(fig)

# %% [markdown]
# # Can I use the phase angle function to specify a non-zero phase for thermal emission?
#
# The phase angle function computes the incoming and outgoing angles. However, non-zero phase functionality is specific to __reflected light observations only__. This might be confusing because of course it is possible to observe thermal emission of planet at non-zero phase. However, unlike reflected light, thermal emission radiates from the planet in all directions (regardless of phase angle).
#
# __Do not force the code to run at non-zero phase angles for thermal emission.__

# %% [markdown]
# # I'm confused about what opacity file to use
#
# Question: I'm confused about the opacity files. Version 1.0 (opacity.db) is the low res version that covers the "useful" wavelengths (0.3 to 14 micron) while version 2.0 has two files that cover the 0.6-6 micron at higher resolution and 4.8-15 micron. What should I download? And once I download it, where do I put them? If it's just the one file that seems straightforward in terms of where to put it but what happens when there are two files (or all 3) in the same folder?
#
# Answer:
#
# **What do I download?**
#
# The [low sampled files across a large wavelength range](https://zenodo.org/record/3759675#.Y8HVkezMLvU), which is on Zenodo as V1 is great for quick calculations that don't necessarily need to be accurate. For example: proposals, example models, any testing, retrievals on fake data.
#
# However, when comparing to real data, it's important to use higher sampling. [This tutorial](https://natashabatalha.github.io/picaso/notebooks/10_ResamplingOpacities.html) shows users estimated sampling errors. Therefore, it is important to have higher sampling files as well. The [higher sampling files are located here under V2](https://zenodo.org/record/6928501#.Y8HWH-zMLvU).
#
# So depending on your use case, the answer might be: download both!
#
# **Where do I put all the files?**
#
# `PICASO` uses the function [`justdoit.opannection`](https://natashabatalha.github.io/picaso/picaso.html#picaso.justdoit.opannection) to grab the opacity file located in the reference directory [`opacities`](https://github.com/natashabatalha/picaso/tree/master/reference/opacities). In the installation instructions you will notice there is a step to place the zenodo file here. Just for completeness, internally, we specify the name of this [file here](https://github.com/natashabatalha/picaso/blob/891343fcc41faa345f8b85aaa8d50c4939c421a3/reference/config.json#L91).
#
# The general recommendation is to keep one "default" file in your `reference/opacities` folder so that you do not need to worry about always specifying a file when running the code. Then assign one place, easy to locate, where you include the rest of the files. In order to access these will need to point to this file path  using the `filename_db` keyword in `opannection`.
#

# %% [markdown]
# # Has your question not been answered? Feel free to contact us!
#
# Submit an issue on Github: https://github.com/natashabatalha/picaso/issues

# %%
