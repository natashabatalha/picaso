# %%
import sys
import os
import warnings
warnings.filterwarnings('ignore')
import picaso.justdoit as jdi
import picaso.justplotit as jpi
import virga.justdoit as vj
import virga.justplotit as cldplt
import astropy.units as u
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import pickle
import xarray
from copy import deepcopy
from bokeh.plotting import show
import h5py

#1 ck tables from roxana
mh = '+000'#'+0.0' #log metallicity
CtoO = '100'#'1.0' # CtoO ratio

ck_db = f"../data/kcoeff_2020/sonora_2020_feh{mh}_co_{CtoO}.data.196"

#sonora bobcat cloud free structures file
sonora_profile_db = '../data/sonora_bobcat/structures_m+0.0'
sonora_dat_db = '../data/sonora_bobcat/structures_m+0.0'

teff = 1500
print(f"Effective temperature = {teff} K")
cl_run = jdi.inputs(calculation="browndwarf", climate = True) # start a calculation
grav = 100 # Gravity of your brown dwarf in m/s/s
cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity
cl_run.effective_temp(teff) # input effective temperature

opacity_ck = jdi.opannection(ck_db=ck_db) # grab your opacities

nlevel = 91 # number of plane-parallel levels in your code
pressure_bobcat,temp_bobcat = np.loadtxt(jdi.os.path.join(
                            sonora_profile_db,f"t{teff}g{grav}nc_m0.0.dat"),
                            usecols=[1,2],unpack=True, skiprows = 1)

nofczns = 1 # number of convective zones initially
nstr_upper = 65 # top most level of guessed convective zone
nstr_deep = nlevel - 2 # this is always the case. Dont change this
nstr = np.array([0,nstr_upper,nstr_deep,0,0,0])
rfacv = 0.5

# All default from 12b notebook
T_star = 5326.6 # K, star effective temperature
logg = 4.38933 #logg , cgs
metal = -0.03 # metallicity of star
r_star = 0.932 # solar radius
semi_major = 0.0486 # star planet distance, AU

cl_run.star(opacity_ck, temp =T_star,metal =metal, logg =logg, radius = r_star, 
            radius_unit=u.R_sun,semi_major= semi_major , semi_major_unit = u.AU)#opacity db, pysynphot database, temp, metallicity, logg

print("Self-consistent run")
cl_run.inputs_climate(temp_guess=deepcopy(temp_bobcat), pressure= pressure_bobcat,
                    nstr = nstr, nofczns = nofczns , rfacv = rfacv, cloudy = "selfconsistent", mh = '0.0', 
                    CtoO = '1.0',species = ['MgSiO3'], fsed = 8.0, beta = 0.1, virga_param = 'const',
                    mieff_dir = "~/projects/clouds/virga/refrind", do_holes = False, fhole = 0.5, fthin_cld = 0.9, moistgrad = False,
                    )
out_selfconsistent = deepcopy(cl_run.climate(opacity_ck, save_all_profiles=True,with_spec=True))

df_selfconsistent = out_selfconsistent['cld_output_picaso']
all_wavenumbers = np.unique(out_selfconsistent['cld_output_picaso'].wavenumber.values)
opd_selfconsistent = df_selfconsistent[df_selfconsistent.wavenumber == all_wavenumbers[150]].opd.values

# %%
p_star_off, t_star_off = None, None
with h5py.File("../data/four_clouds_testing/fsed8/teff_1500_MgSiO3_browndwarf.h5") as f:
    p_star_off = np.array(f["pressure"])
    t_star_off = np.array(f["temp_selfconsistent"])

# %%
plt.semilogy(out_selfconsistent["temperature"], out_selfconsistent["pressure"], label="Star on")
plt.semilogy(t_star_off, p_star_off, label="Star off")
plt.gca().invert_yaxis()
plt.legend()
plt.xlabel("Temperature (K)")
plt.ylabel("Pressure (bar)")
# %%
