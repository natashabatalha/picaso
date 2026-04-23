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
# # Using PICASO in GPU mode
#
# In the following tutorial we will switch the hardware in picaso from CPU to GPU and test the timing. We assume you are already well versed in running PICASO. 

# %%
import os 

os.environ['picaso_hardware']='cpu'

#picaso
from picaso import justdoit as jdi
from picaso import justplotit as jpi
import numpy as np

import matplotlib.pyplot as plt

jpi.output_notebook()

# %% [markdown]
# ## Setup Basic PICASO Run 

# %%
opacity = jdi.opannection(wave_range=[0.3,1],verbose=False) 
start_case = jdi.inputs()
#phase angle
start_case.phase_angle(0) #radians
#define gravity
start_case.gravity(radius = 1, radius_unit = jdi.u.Unit('R_jup'),
                   mass = 1,mass_unit = jdi.u.Unit('M_jup')) #any astropy units available

#define star
start_case.star(opacity, 5000,0,4.0, radius=1, radius_unit = jdi.u.Unit('R_sun')) #opacity db, pysynphot database, temp, metallicity, logg
start_case.atmosphere(filename=jdi.jupiter_pt(), sep=r'\s+')


# %% [markdown]
# ## Run Reflected, Thermal, and Transit on CPU

# %%
df_cpu_ref = start_case.spectrum(opacity,calculation='reflected')#other options: transmission, thermal or a combo of many e.g., "reflected+transmission"
# ref_time_cpu = %timeit -o start_case.spectrum(opacity,calculation='reflected')

# %%
df_cpu_th = start_case.spectrum(opacity,calculation='thermal')
# th_time_cpu = %timeit -o start_case.spectrum(opacity,calculation='thermal')

# %%
df_cpu_td = start_case.spectrum(opacity,calculation='transmission')
# td_time_cpu = %timeit -o start_case.spectrum(opacity,calculation='transmission')

# %% [markdown]
# ## Run Reflected, Thermal, and Transit on GPU

# %%
os.environ['picaso_hardware']='gpu'
import importlib;importlib.reload(jdi) #environment variable change means we need to reload justdoit

# %%
opacity = jdi.opannection(wave_range=[0.3,1],verbose=False) 
start_case = jdi.inputs()
#phase angle
start_case.phase_angle(0) #radians
#define gravity
start_case.gravity(radius = 1, radius_unit = jdi.u.Unit('R_jup'),
                   mass = 1,mass_unit = jdi.u.Unit('M_jup')) #any astropy units available

#define star
start_case.star(opacity, 5000,0,4.0, radius=1, radius_unit = jdi.u.Unit('R_sun')) #opacity db, pysynphot database, temp, metallicity, logg
start_case.atmosphere(filename=jdi.jupiter_pt(), sep=r'\s+')


# %%
df_gpu_ref = start_case.spectrum(opacity,calculation='reflected')#other options: transmission, thermal or a combo of many e.g., "reflected+transmission"
# ref_time_gpu = %timeit -o start_case.spectrum(opacity,calculation='reflected')

# %%
df_gpu_th = start_case.spectrum(opacity,calculation='thermal')
# th_time_gpu = %timeit -o start_case.spectrum(opacity,calculation='thermal')

# %%
df_gpu_td = start_case.spectrum(opacity,calculation='transmission')
# td_time_gpu = %timeit -o start_case.spectrum(opacity,calculation='transmission')

# %%
plt.plot(1e4/df_cpu_ref['wavenumber'],df_cpu_ref['albedo'],label=rf'cpu avg time = {ref_time_cpu.average}')
plt.plot(1e4/df_gpu_ref['wavenumber'],df_gpu_ref['albedo'],label=rf'gpu avg time = {ref_time_gpu.average}')
passing= np.allclose(df_gpu_ref['albedo'],df_cpu_ref['albedo'])
max_per_diff = 100*np.max((np.abs(df_gpu_ref['albedo']-df_cpu_ref['albedo']))/df_cpu_ref['albedo'])
plt.title(rf'Albedo CPU=GPU? {passing}. Max % diff={max_per_diff}')
plt.legend()

# %%
plt.semilogy(1e4/df_cpu_th['wavenumber'],df_cpu_th['thermal'],label=rf'cpu avg time = {th_time_cpu.average}')
plt.semilogy(1e4/df_gpu_th['wavenumber'],df_gpu_th['thermal'],label=rf'gpu avg time = {th_time_gpu.average}')
passing= np.allclose(df_gpu_th['thermal'],df_cpu_th['thermal'])
max_per_diff = 100*np.max((np.abs(df_gpu_th['thermal']-df_cpu_th['thermal']))/df_cpu_th['thermal'])
plt.title(rf'Thermal Flux CPU=GPU? {passing}. Max % diff={max_per_diff}')
plt.legend()

# %%
plt.semilogy(1e4/df_cpu_td['wavenumber'],df_cpu_td['transit_depth'],label=rf'cpu avg time = {td_time_cpu.average}')
plt.semilogy(1e4/df_gpu_td['wavenumber'],df_gpu_td['transit_depth'],label=rf'gpu avg time = {td_time_gpu.average}')
passing= np.allclose(df_gpu_td['transit_depth'],df_cpu_td['transit_depth'])
max_per_diff = 100*np.max((np.abs(df_gpu_td['transit_depth']-df_cpu_td['transit_depth']))/df_cpu_td['transit_depth'])
plt.title(rf'Transit Depth CPU=GPU? {passing}. Max % diff={max_per_diff}')
plt.legend()


# %% [markdown]
# ## Manual GPU Memory Management
#
# By default, PICASO will allocate and deallocate GPU memory for every single spectrum calculation. 
# While this is convenient, it introduces significant overhead, especially when running many spectra (e.g., in a retrieval or a phase curve).
#
# To improve performance, we also offer manual management outside of the `spectrum` call. 
# This allows you to allocate the necessary buffers once at the beginning of your session and free them once at the end.
#
# ### Why is this important?
# GPU memory allocation is a relatively slow operation compared to the actual computation. 
# By allocating buffers once, we avoid this overhead in every iteration. 
# This is particularly important for GPU performance where the goal is to minimize host-to-device communication and synchronization.
#
# ### How to use it
# You need to know the dimensions of your problem (number of levels, number of wavenumbers, number of Gauss angles, and number of Tchebychev angles).
#
# Here is how you can manually manage memory for reflected, thermal, and transit calculations:

# %%
# Define dimensions (must match your actual run)
nlevel = start_case.nlevel
nwno = opacity.nwno
ng = start_case.inputs['disco']['num_gangle']
nt = start_case.inputs['disco']['num_tangle']
get_lvl_flux = start_case.inputs['approx'].get('get_lvl_flux', False)

# Allocate buffers
jdi.get_reflected_1d_allocate_buffers(nlevel, nwno, ng, nt)
jdi.get_thermal_1d_allocate_buffers(nlevel, nwno, ng, nt, 1 if get_lvl_flux else 0)
jdi.get_transit_1d_allocate_buffers(nlevel, nwno)

# %% [markdown]
# Now you can run your spectra multiple times without the allocation overhead:

# %%
# Run multiple spectra
# %timeit df_gpu_ref = start_case.spectrum(opacity, calculation='reflected')

# %% [markdown]
# Finally, don't forget to free the memory at the end of your session:

# %%
# Free buffers
jdi.get_reflected_1d_free()
jdi.get_thermal_1d_free(1 if get_lvl_flux else 0)
jdi.get_transit_1d_free()
