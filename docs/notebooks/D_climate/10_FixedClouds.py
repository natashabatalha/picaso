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
import xarray as xr
from bokeh.plotting import show, figure
# %%
#1 ck tables from roxana
mh = '0.0'#'+0.0' #log metallicity
CtoO = '0.46'# # CtoO absolute ratio
ck_db = os.path.join(os.getenv('picaso_refdata'),'opacities', 'preweighted', f'sonora_2121grid_feh{mh}_co{CtoO}.hdf5')

# #sonora bobcat cloud free structures file
sonora_profile_db = os.path.join(os.getenv('picaso_refdata'),'sonora_grids','bobcat')

# %%
cl_run = jdi.inputs(calculation="browndwarf", climate = True) # start a calculation

#note you need to put the climate keyword to be True in order to do so
# now you need to add these parameters to your calculation

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
rcb_guess = 80 # top most level of guessed convective zone

# Here are some other parameters needed for the code.
rfacv = 0.0 #we are focused on a brown dwarf so let's keep this as is
# %%
virga_dir = os.path.join(os.getenv('picaso_refdata'),'virga') # path to virga directory with the Mie scattering files

# %%
cl_run.inputs_climate(temp_guess= temp_bobcat, pressure= pressure_bobcat,
                      rcb_guess=rcb_guess, rfacv = rfacv)
#cl_run.atmosphere(cold_trap=True)
virga_planet = vj.Atmosphere(['H2O'], fsed=8, mh=1, mmw=2.2)
virga_planet.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)'))
virga_planet.ptk(df = pd.DataFrame({'pressure':pressure_bobcat, 'temperature': temp_bobcat, 'kz':  np.ones_like(pressure_bobcat) * 1e10}), kz_min=1e5, latent_heat=True)
v_out = vj.compute(virga_planet, as_dict=True, directory=virga_dir)
cl_run.fix_virga_clouds(v_out)
out = cl_run.climate(opacity_ck, save_all_profiles=True,with_spec=False)

