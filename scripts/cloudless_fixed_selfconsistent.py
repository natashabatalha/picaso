# %%
import os
import warnings
import astropy.units as u
import numpy as np
from matplotlib import pyplot as plt
import picaso.justdoit as jdi
warnings.filterwarnings('ignore')

#1 ck tables from roxana
mh = '+000'#'+0.0' #log metallicity
CtoO = '100'#'1.0' # CtoO ratio

ck_db = os.path.join(os.getenv('picaso_refdata'),'opacities', 'preweighted', f'sonora_2020_feh{mh}_co_{CtoO}.data.196')

# #sonora bobcat cloud free structures file
sonora_profile_db = os.path.join(os.getenv('picaso_refdata'),'sonora_grids','bobcat', 'structures_m+0.0')

#note you need to put the climate keyword to be True in order to do so
# now you need to add these parameters to your calculation

teff = 200 # Effective Temperature of your Brown Dwarf in K
grav = 1000 # Gravity of your brown dwarf in m/s/s
opacity_ck = jdi.opannection(ck_db=ck_db, method='preweighted') # grab your opacities
pressure_bobcat,temp_bobcat = np.loadtxt(jdi.os.path.join(
                            sonora_profile_db,f"t{teff}g{grav}nc_m0.0.dat"),
                            usecols=[1,2],unpack=True, skiprows = 1)

rcb_guess = 52 # top most level of guessed convective zone 
virga_dir = os.path.join("/Users", "adityasengupta",'virga', "refrind") # path to virga directory with the Mie scattering files

runs = {}
runmodes = ["cloudless", "fixed", "selfconsistent"]
for runmode in runmodes:
    cl_run = jdi.inputs(calculation="browndwarf", climate = True) # start a calculation
    cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity
    cl_run.effective_temp(teff) # input effective temperature
    cl_run.inputs_climate(temp_guess=temp_bobcat, pressure=pressure_bobcat,
                        rcb_guess=52, rfacv=0.0)
    cl_run.virga(condensates = ['H2O'],directory = virga_dir, runmode=runmode, mh=1,fsed = 8.0, latent_heat=True)
    runs[runmode] = cl_run.climate(opacity_ck, save_all_profiles=True,with_spec=False)
    
# %%
for runmode in runmodes:
    plt.semilogy(runs[runmode]["temperature"], runs[runmode]["pressure"], label=runmode)
plt.gca().invert_yaxis()
plt.xlabel("Temperature (K)")
plt.ylabel("Pressure (bar)")
plt.legend()
# %%
