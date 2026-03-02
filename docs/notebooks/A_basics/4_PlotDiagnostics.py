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
# # Using Full Output for Diagnostics
#
# Sometimes it helps to have a bigger picture of what the full output is doing.
#
# These steps will guide you through how to get out gain more intuition from your runs.

# %%
#picaso
from picaso import justdoit as jdi
from picaso import justplotit as jpi
jpi.output_notebook()

# %% [markdown]
# We will use a cloudy Jupiter again to guide us through the exercise.

# %%
opa = jdi.opannection(wave_range=[0.3,1])

case1 = jdi.inputs()

case1.phase_angle(0)

case1.gravity(gravity=25, gravity_unit=jdi.u.Unit('m/(s**2)'))

case1.star(opa, 6000,0.0122,4.437)

case1.atmosphere(filename = jdi.jupiter_pt(), sep=r'\s+')
case1.clouds(filename = jdi.jupiter_cld(), sep=r'\s+')

# %% [markdown]
# ## Return ``PICASO`` Full Ouput

# %%
df = case1.spectrum(opa, full_output=True) #note the new last key
wno, alb, full_output = df['wavenumber'] , df['albedo'] , df['full_output']

# %% [markdown]
# ## Visualizing Full Output
#
# ### Mixing Ratios

# %%
jpi.show(jpi.mixing_ratio(full_output))
#can also input any key word argument acceptable for bokeh.figure:
#show(jpi.mixing_ratio(full_output, plot_width=500, y_axis_type='linear',y_range=[10,1e-3]))

# %% [markdown]
# ### Cloud Profile
#
# Depending on your wavelength grid, you might exceed ``Jupyter Notebook's`` data rage limit. You can fix this by initiating jupyter notebook with a higher data rate limit.
#
# ``jupyter notebook --NotebookApp.iopub_data_rate_limit=10000000``

# %%
fig = jpi.cloud(full_output)

# %% [markdown]
# ### Pressure-Tempertaure Profile

# %%
jpi.show(jpi.pt(full_output))

# %% [markdown]
# ### Heatmap of Molecular, Cloud, and Rayleigh Scattering Opacity
#
# This will show you where your molecular, cloud and rayleigh scattering opacity become optically thick ($\tau$~1). Blue regions of the plot are optically thin, while red are optically thick.

# %%
jpi.heatmap_taus(df)

# %% [markdown]
# Sometimes, we just want to compare the $\tau\sim 1$ surface for each component. This is effectively the plot below, the "photon attenuation".

# %% [markdown]
# ### Photon Attenuation Depth
#
# This is a useful plot to see the interplay between scattering and absorbing sources. It should explain why you are getting bright versus dark reflectivity.
#
# To make this plot, we take data that is in previous plots (taugas, taucld, tauray), integrate it starting at the top of the atmosphere, then determine where a certain opacity hits the `at_tau` value.

# %%
jpi.show(jpi.photon_attenuation(full_output, at_tau=0.1, plot_width=500))

# %% [markdown]
# Compare it to the spectrum and you can see right away what is driving the overall shape of your spectrum

# %%
jpi.show(jpi.spectrum(wno,alb,plot_width=500))

# %% [markdown]
# ### Disecting Full Output

# %%
full_output.keys()

# %%
full_output['layer'].keys()

# %%
taugas = full_output['taugas'] #matrix that is nlevel x nwvno
taucld = full_output['taucld'] #matrix that is nlevel x nwvno
taugas = full_output['taugas'] #matrix that is nlevel x nwvno

# %%
