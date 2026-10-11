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
# # 3D Transmission Spectra: Benchmark with TRIDENT
#
# `PICASO` computes 3D transmission spectra following the path distribution approach of TRIDENT. The atmosphere is divided into columns: each column is a **zenith slice** (angle from the terminator plane, from the dayside to the nightside) in an **azimuthal sector** (angle around the terminator, measured from the north pole). Each column has its own temperature, chemistry, clouds and therefore its own radial grid.
#
# Here you will learn:
#
# 1. How to set up the geometry of a 3D transmission calculation with `transit_geometry`
# 2. How to define a parametric 3D atmosphere directly on the transit columns
# 3. How to visualize the geometry you have specified with `jpi.transit_geometry`
# 4. How a day-night temperature gradient changes a transmission spectrum
#
# We reproduce the ultra-hot Jupiter benchmark from Section 4.2 of MacDonald & Lewis (2022), which compares TRIDENT against the 3D radiative transfer code Pytmosph3R (Figures 8 and 9).
#
# **Citation: [MacDonald & Lewis 2022 "TRIDENT: A Rapid 3D Radiative Transfer Model for Exoplanet Transmission Spectra" ApJ](https://ui.adsabs.harvard.edu/abs/2022ApJ...929...20M/abstract)**

# %%
from picaso import justplotit as jpi
from picaso import justdoit as jdi
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
u = jdi.u

jpi.output_notebook()

# %% [markdown]
# ## The Benchmark Model
#
# The model comes from the Pytmosph3R documentation and is described in Section 4.2 of MacDonald & Lewis (2022). It is a 2D atmosphere: properties vary with zenith angle (day to night) but not with azimuthal angle.
#
# | Parameter | Value |
# |---|---|
# | Stellar radius | 1.458 $R_\odot$ |
# | Planet radius (at 10 bar) | 1.7670 $R_J$ |
# | Planet mass | 1.1829 $M_J$ |
# | Dayside temperature | 3300 K (isothermal) |
# | Nightside temperature | 500 K (isothermal) |
# | Deep temperature ($P>0.1$ bar) | 2500 K (everywhere) |
# | Nightside H$_2$O | $5.0119\times10^{-3}$ |
# | Dayside H$_2$O | 10$\times$ less than nightside |
# | CO | $4.4\times10^{-4}$ (everywhere) |
# | Background gas | H$_2$ (no He) |
# | Day-night transition width, $\beta$ | 10$^\circ$ |
# | Pressure grid | 100 layers, $10^{-9}$ to 10 bar |
# | Surface | 0.1 bar |
#
# Across the day-night transition, the temperature and the log of the mixing ratios vary linearly with zenith angle (MacDonald & Lewis 2022, Section 3.2).

# %%
beta = np.radians(10) #width of day-night transition
p_surface = 0.1 #bars, see note on the surface below
pressure = np.logspace(-9, 1, 101) #101 levels = 100 layers

def benchmark_atmosphere(theta, pressure, beta, T_day=3300, T_night=500, T_deep=2500, p_deep=0.1,
                         h2o_night=5.0119e-3, h2o_dex_day=-1, co=4.4e-4):
    """
    TRIDENT/Pytmosph3R benchmark atmosphere as a function of zenith angle (radians)
    and pressure (bars). Returns temperature and mixing ratios with dimensions (ntheta, npressure)
    """
    #0 on the dayside, 1 on the nightside, linear in between
    night_frac = np.clip((theta + beta/2)/beta, 0, 1)[:, np.newaxis]
    temperature = T_day + (T_night - T_day)*night_frac + 0*pressure[np.newaxis,:]
    temperature[:, pressure > p_deep] = T_deep
    h2o = 10**(np.log10(h2o_night) + h2o_dex_day*(1-night_frac)) + 0*pressure[np.newaxis,:]
    co = co + 0*h2o
    return temperature, {'H2O':h2o, 'CO':co, 'H2':1-h2o-co}

# %% [markdown]
# ## Opacities
#
# The benchmark spectra were computed from 0.4–10 $\mu$m with opacity sampling at R=10,000 and binned to R=50. Here we use the default `PICASO` opacity database across the same wavelength range.

# %%
opacity = jdi.opannection(wave_range=[0.4, 10])

# %% [markdown]
# ## Define the Transit Geometry
#
# `transit_geometry` replaces `phase_angle` (used in emission and reflected light phase curves) for 3D transmission calculations. It controls:
#
# - **Zenith slices** (`num_zenith` or `zenith_edges`): the slices from the dayside ($\theta=-90^\circ$) to the nightside ($\theta=+90^\circ$). The terminator plane ($\theta=0$) must be a slice boundary.
# - **Azimuthal sectors** (`num_sector` or `sector_edges`): the sectors around the terminator ring.
# - **Inclination**: tilts the terminator relative to the planet's equator (only matters if your atmosphere varies with latitude).
#
# To match TRIDENT's benchmark setup we use $N_{\rm zenith}=10$: one dayside slice, 8 slices evenly spaced across the $\beta=10^\circ$ transition, and one nightside slice. Since the atmosphere is 2D, a single azimuthal sector is enough ($N_{\rm sector}=1$).
#
# We also set the reference pressure to the bottom of our grid (10 bar). In 3D, every column has its own radial grid, so `PICASO` assumes that all columns share a common radius at `p_reference`. 

# %%


#defin zenith edges 
zenith_edges = np.concatenate([[-np.pi/2], np.linspace(-beta/2, beta/2, 9), [np.pi/2]])
print('Zenith edges (deg):', np.round(np.degrees(zenith_edges), 2))

def setup_case(beta, surface=True, **atmosphere_kwargs): #see note on surface below

    #regular picaso things
    case = jdi.inputs()
    case.gravity(mass=1.1829, mass_unit=u.Unit('M_jup'),
                 radius=1.7670, radius_unit=u.Unit('R_jup'))
    #only the stellar radius matters for transmission
    case.star(opacity, 6460, 0.13, 4.24, radius=1.458, radius_unit=u.Unit('R_sun'))
    case.approx(p_reference=10)
    
    #new transit 3d function!
    case.transit_geometry(zenith_edges=zenith_edges, num_sector=1, inclination=np.pi/2)


    #build the atmosphere directly on the transit columns (see below)
    geom = case.inputs['disco']
    theta, phi = geom['longitude'], geom['latitude']
    temperature, mixingratios = benchmark_atmosphere(theta, pressure, beta, **atmosphere_kwargs)
    nphi = len(phi)
    dims = ['lon', 'lat', 'pressure']
    coords = dict(lon=('lon', np.degrees(theta), {'units':'degrees'}),
                  lat=('lat', np.degrees(phi), {'units':'degrees'}),
                  pressure=('pressure', pressure, {'units':'bar'}))
    to3d = lambda x: np.repeat(x[:, np.newaxis, :], nphi, axis=1)
    data = {'temperature':(dims, to3d(temperature))}
    data.update({mol:(dims, to3d(mixingratios[mol])) for mol in mixingratios})
    case.atmosphere_3d(xr.Dataset(data, coords=coords), regrid=False, plot=False, verbose=False)

    #note this is just a fake surface to simulate the Mac and Lewis benchmark case
    if surface:
        #grey, completely opaque cloud deck below the surface
        player = np.sqrt(pressure[1:]*pressure[:-1])
        wno = np.array([1e2, 1e5])
        opd = np.zeros((len(theta), nphi, len(player), len(wno)))
        opd[:, :, player > p_surface, :] = 1e3
        dims4 = ['lon', 'lat', 'pressure', 'wno']
        cld_coords = dict(coords)
        cld_coords['pressure'] = ('pressure', player, {'units':'bar'})
        cld_coords['wno'] = ('wno', wno, {'units':'cm^(-1)'})
        #This is not a real surface -- we are using a cloud deck here to simulate the Mac and Lewis benchmark case. 
        #of course if you want to add a cloud this is also how you would do it! 
        case.clouds_3d(xr.Dataset(dict(opd=(dims4, opd), g0=(dims4, 0*opd), w0=(dims4, 0*opd)),
                       coords=cld_coords), regrid=False, plot=False, verbose=False)
    return case

# %% [markdown]
# ### Defining an atmosphere directly on the transit columns
#
# If you have a GCM, you can supply it on its native latitude-longitude grid with `atmosphere_3d(ds, regrid=True)` (see the earlier 3D tutorials). `PICASO` will interpolate it onto the transit columns.
#
# For a parametric model like this one it is more accurate to evaluate the model directly on the columns. After `transit_geometry`, the xarray coordinates are relabeled so that:
#
# - `lon` is the zenith angle of each column: `case.inputs['disco']['longitude']` (radians)
# - `lat` is the azimuthal angle of each column: `case.inputs['disco']['latitude']` (radians)
#
# The true planet latitude and longitude of each column are stored in `case.inputs['disco']['transit_latitude']` and `['transit_longitude']`. Supply your xarray with these coordinates (in degrees) and `regrid=False`, as is done in `setup_case` above.
#
# ### A note on the surface
#
# MacDonald & Lewis (2022) found that they had to place a surface at 0.1 bar (where the P-T profile jumps to the deep isotherm) to match Pytmosph3R. `PICASO` does not have a solid surface for transmission, so we mimic one with a grey, completely opaque cloud deck below 0.1 bar. The planet radius is still defined at 10 bar. You can turn this off with `surface=False` to see how much it matters. For this model it changes the R=50 spectrum by up to ~100 ppm, which is larger than the 29 ppm agreement between TRIDENT and Pytmosph3R.

# %%
case_2d = setup_case(beta, surface=True)
out_2d = case_2d.spectrum(opacity, calculation='transmission', dimension='3d', full_output=True)

# %% [markdown]
# ## Check the Geometry
#
# Before analyzing the spectrum, it is good practice to check that the geometry is what you intended. `jpi.transit_geometry` plots each column to scale, colored by temperature (or any molecule's mixing ratio).
#
# - `plane='day-night'` shows the slice containing the observer's line of sight. The star is to the left and the observer is to the right. This reproduces the right panel of Figure 9 in MacDonald & Lewis (2022).
# - `plane='terminator'` shows the planet as the observer sees it, for a single zenith slice. This is more useful for atmospheres with morning-evening differences.
#
# Notice how much more extended the dayside is than the nightside! A ray passing through the terminator samples the puffy dayside on its way in, and the compact nightside on its way out.

# %%
full_output = out_2d['full_output']
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
jpi.transit_geometry(full_output, plane='day-night', ax=ax[0])
jpi.transit_geometry(full_output, plane='day-night', plot='H2O', cmap='viridis', ax=ax[1])
plt.show()

# %% [markdown]
# Zoom in to see the day-night transition region

# %%
jpi.transit_geometry(full_output, plane='day-night', x_range=[-0.6, 0.6], y_range=[1.6, 2.4])

# %% [markdown]
# We can also reproduce the left panel of Figure 9, the P-T profiles of the dayside, nightside, and the columns on either side of the terminator plane.

# %%
theta_edges = full_output['transit']['theta_edges']
theta = 0.5*(theta_edges[1:] + theta_edges[:-1])
player = full_output['layer']['pressure'][:, 0, 0]
fig, ax = plt.subplots(figsize=(5, 5))
for k, name in [(0, 'Dayside'), (len(theta)//2-1, 'Terminator (day side)'),
                (len(theta)//2, 'Terminator (night side)'), (len(theta)-1, 'Nightside')]:
    ax.plot(full_output['layer']['temperature'][:, k, 0], player,
            label=f'{name}, $\\theta$={np.degrees(theta[k]):.1f}$^\\circ$')
ax.set_yscale('log'); ax.invert_yaxis()
ax.set_xlabel('Temperature (K)'); ax.set_ylabel('Pressure (bar)')
ax.legend(fontsize=8)
plt.show()

# %% [markdown]
# ## Analyze the 3D Spectrum
#
# Now we compare to the benchmark spectra in Figure 8 (bottom panel) of MacDonald & Lewis (2022), binned to R=50.
#
# The benchmark data must be supplied separately. Save the R=50 TRIDENT and/or Pytmosph3R spectra as a csv with columns `wavelength` (microns) and `transit_depth`, then point `benchmark_files` at them.

# %%
R = 50
wno_2d, depth_2d = jdi.mean_regrid(out_2d['wavenumber'], out_2d['transit_depth'], R=R)

x, y, legend = [wno_2d], [depth_2d*100], ['PICASO']

jpi.show(jpi.spectrum(x, y, legend=legend, plot_width=800, x_axis_type='log',
                      y_axis_label='Transit Depth (%)'))

# %% [markdown]
# ## How Much Does the Day-Night Gradient Matter?
#
# To build intuition, let's compare the 2D spectrum with spectra where the whole planet has the dayside or the nightside properties. These are equivalent to 1D calculations, but we run them through the same 3D machinery by setting the transition so that every column has the same properties.

# %%
#entire planet has dayside properties
case_day = setup_case(beta, T_night=3300, h2o_dex_day=0, h2o_night=5.0119e-4)
out_day = case_day.spectrum(opacity, calculation='transmission', dimension='3d')
#entire planet has nightside properties
case_night = setup_case(beta, T_day=500, h2o_dex_day=0)
out_night = case_night.spectrum(opacity, calculation='transmission', dimension='3d')

wno_day, depth_day = jdi.mean_regrid(out_day['wavenumber'], out_day['transit_depth'], R=R)
wno_night, depth_night = jdi.mean_regrid(out_night['wavenumber'], out_night['transit_depth'], R=R)

jpi.show(jpi.spectrum([wno_day, wno_2d, wno_night], [depth_day*100, depth_2d*100, depth_night*100],
                      legend=['Uniform Dayside', '2D Day-Night', 'Uniform Nightside'],
                      plot_width=800, x_axis_type='log', y_axis_label='Transit Depth (%)'))

# %% [markdown]
# The 2D spectrum is much closer to the uniform dayside spectrum than to the uniform nightside spectrum, even though half of every ray's path is on the nightside. Rays entering the hot, extended dayside travel long paths through it, which effectively increases the scale height of the terminator region (MacDonald & Lewis 2022, Section 5.1.1). A 1D model fit to this spectrum would infer a temperature much warmer than the true terminator temperature (Caldas et al. 2019). 
#
# ## Things to Try
#
# 1. Change `beta` to see how the width of the transition region changes the spectrum (their Figure 10).
# 2. Use more azimuthal sectors (`num_sector`) with a morning-evening gradient, and view it with `plane='terminator'`.
# 3. Supply a GCM on a latitude-longitude grid with `atmosphere_3d(ds, regrid=True)` after running `transit_geometry`.
