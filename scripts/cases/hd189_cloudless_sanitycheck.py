# %%
import warnings
warnings.filterwarnings('ignore')
import picaso.justdoit as jdi
import picaso.justplotit as jpi
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

#1 ck tables from roxana
mh = '+000'#'+1.0' #log metallicity, 10xSolar
CtoO = '100'#'1.0' # CtoO ratio, Solar C/O

ck_db = f'/Users/adityasengupta/picaso/reference/kcoeff_2020/sonora_2020_feh{mh}_co_{CtoO}.data.196'

# Notice The keyword ck is set to True because you want to use the correlated-k opacities for your calculation
# and not the line by line opacities
opacity_ck = jdi.opannection(ck_db=ck_db) # grab your opacities

cl_run = jdi.inputs(calculation="planet", climate = True) # start a calculation 

#note you need to put the climate keyword to be True in order to do so
# now you need to add these parameters to your calculation  


tint= 200 # interior Temperature of your Planet in K
grav = 23.08420403 # Gravity of your Planet in m/s/s

cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity
cl_run.effective_temp(tint) # input effective temperature

# pulled from Anna's spreadsheet
T_star = 5012.5 # K, star effective temperature
logg = 4.57 #logg , cgs
metal = 0.04 # metallicity of star
r_star = 0.782 # solar radius
semi_major = 0.03106 # star planet distance, AU

cl_run.star(opacity_ck, temp =T_star,metal =metal, logg =logg, radius = r_star, 
            radius_unit=u.R_sun,semi_major= semi_major , semi_major_unit = u.AU)#opacity db, pysynphot database, temp, metallicity, logg

nlevel = 91 # number of plane-parallel levels in your code

#Lets set the max and min at 1e-4 bars and 500 bars
# replace this with Natasha's 

nofczns = 1
nstr = [0, 85, 89, 0, 0, 0]

rfacv = 0.8
hd189_pressure = np.logspace(-6, 2, 91)
hd189_temperature = np.load("../data/silicate_test_cases/HD189_temperature.npy")
cl_run.inputs_climate(temp_guess= np.copy(hd189_temperature), pressure= np.copy(hd189_pressure), nstr = nstr, nofczns = nofczns , rfacv = rfacv, cloudy="cloudless")

out = cl_run.climate(opacity_ck, save_all_profiles=True,with_spec=True)

# %%
plt.semilogy(hd189_temperature, hd189_pressure, label="HD189 starting profile")
plt.semilogy(out["temperature"], out["pressure"], label="New PICASO run")
plt.gca().invert_yaxis()
plt.legend()
# %%
grid_temp_structure = np.genfromtxt("../data/hd189_temperature_structures/tp_feh+000_tint200_co100_rfacv0.5.txt")
hd189_pressure = grid_temp_structure[:, 0]
hd189_temperature = grid_temp_structure[:, 1]
# %%
