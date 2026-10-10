# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Choosing a Plotting Backend: Bokeh, Matplotlib, or Plotly
#
# By default, the plotting functions in `justplotit` return interactive [bokeh](https://bokeh.org/) figures. Many of them can also return [matplotlib](https://matplotlib.org/) or [plotly](https://plotly.com/python/) figures, so you can use whichever library fits your workflow:
#
# - **bokeh** (default): interactive, and what all of the other tutorials use
# - **matplotlib**: static figures that are easy to customize and save for papers
# - **plotly**: interactive, with true 3D figures. This is what the `picaso-app` uses. Plotly is optional, so you may need to `pip install plotly`
#
# There are two ways to switch:
#
# 1. Add the `backend` keyword to a `justplotit` function, e.g. `jpi.spectrum(wno, alb, backend='matplotlib')`
# 2. Use `jpi.Analyzer`, which holds your output and a default backend for all of its plots
#
# Nothing changes if you don't use `backend`. All of your existing code will still return bokeh figures.
#
# We will make the same diagnostic plots as in [Using Full Output for Diagnostics](../A_basics/4_PlotDiagnostics.html).

# %%
#picaso
from picaso import justplotit as jpi
from picaso import justdoit as jdi
#tells bokeh to plot in the notebook
jpi.output_notebook()
#so that plotly figures display both in jupyter and in the html documentation
import plotly.io as pio 
pio.renderers.default = 'plotly_mimetype+notebook_connected'

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

# %%
df = case1.spectrum(opa, full_output=True)
wno, alb, full_output = df['wavenumber'] , df['albedo'] , df['full_output']

# %% [markdown]
# ## Option 1: The `backend` keyword
#
# Here is the same spectrum with each backend. The only difference is how you display the figure:
#
# - **bokeh**: wrap it in `jpi.show()`
# - **matplotlib**: it displays on its own. Assign it to a variable (`fig = ...`) so it does not show up twice
# - **plotly**: make it the last line of a cell, or call `fig.show()`

# %%
jpi.show(jpi.spectrum(wno, alb, plot_width=500, backend='bokeh'))

# %%
fig = jpi.spectrum(wno, alb, plot_width=500, backend='matplotlib')

# %%
jpi.spectrum(wno, alb, plot_width=500, backend='plotly')

# %% [markdown]
# Keywords such as `plot_width`, `height`, `title`, `x_axis_label`, `y_axis_type='log'`, and `x_range` work with every backend. Any other keyword is passed to `bokeh.plotting.figure()` when you use bokeh, and is ignored by matplotlib and plotly.
#
# Each backend returns its own figure object, so you can keep customizing it with that library:

# %%
fig = jpi.spectrum(wno, alb, plot_width=500, backend='matplotlib')
ax = fig.axes[0]
ax.axvspan(0.6, 0.65, color='grey', alpha=0.3)
ax.set_ylabel('Geometric Albedo');
#fig.savefig('albedo.pdf')

# %%
fig = jpi.spectrum(wno, alb, plot_width=500, backend='plotly')
fig.add_vrect(x0=0.6, x1=0.65, fillcolor='grey', opacity=0.3, line_width=0)
fig.update_yaxes(title_text='Geometric Albedo')

# %% [markdown]
# ## Option 2: The `Analyzer`
#
# If you are making several diagnostic plots of the same run, `jpi.Analyzer` saves you from passing `full_output` and `backend` to every function. You can give it either the whole dictionary returned by `spectrum(opa, full_output=True)` or `df['full_output']`.

# %%
analyze = jpi.Analyzer(df, backend='plotly')

# %% [markdown]
# ### Mixing Ratios

# %%
analyze.mixing_ratio(limit=10)

# %% [markdown]
# Every `Analyzer` method also takes `backend`, which overrides the default for just that plot:

# %%
fig = analyze.mixing_ratio(limit=10, backend='matplotlib')

# %% [markdown]
# ### Cloud Profile
#
# The interactive versions of this plot show at most 500 wavelength points (`jpi.MAX_HEATMAP_COLUMNS`) so that your notebook does not get too large. The matplotlib version uses every point.

# %%
analyze.cloud()

# %%
jpi.show(analyze.cloud(backend='bokeh'))

# %% [markdown]
# ### Pressure-Temperature Profile
#
# You can also change the default backend at any time:

# %%
analyze.backend = 'matplotlib'
fig = analyze.pt()

# %% [markdown]
# ### Photon Attenuation Depth
#
# `return_output=True` returns the same data with every backend, so you can always make your own plot:

# %%
fig, wave, at_pressures_gas, at_pressures_cld, at_pressures_ray = analyze.photon_attenuation(at_tau=0.1,
                                                                         plot_width=700, return_output=True)

# %% [markdown]
# ## Which functions support `backend`?
#
# | Function | Default backend | Notes |
# |---|---|---|
# | `spectrum`, `plot_errorbar` | bokeh | `plot_errorbar` can add points to an existing bokeh or plotly figure with `plot=fig` |
# | `mixing_ratio`, `pt`, `photon_attenuation`, `flux_at_top`, `all_optics_1d` | bokeh | `mixing_ratio` also accepts the older `plot_type` keyword |
# | `cloud`, `thermal_contribution`, `transmission_contribution` | matplotlib | |
# | `disco`, `map`, `taumap` | matplotlib | Plotly makes 3D spheres. Bokeh makes flat latitude-longitude maps |
#
# Every function in this table is also a method of `jpi.Analyzer`. The `Analyzer` uses bokeh as its default unless you choose another backend. `flux_at_top` is the only method that needs the whole dictionary from `spectrum(opa, full_output=True)` rather than just `full_output`.
#
# All other `justplotit` functions return the same figures that they always have.
