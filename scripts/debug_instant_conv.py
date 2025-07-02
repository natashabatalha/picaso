# %%
import os
from os.path import join, dirname
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')
from copy import deepcopy
import astropy.units as u
import numpy as np
import h5py
import picaso.justdoit as jdi
import virga

picaso_root = dirname(dirname(jdi.__file__))
virga_root = dirname(dirname(virga.__file__))

cloud_species = ["MgSiO3", "Mg2SiO4", "Fe", "Al2O3"]

run_name = os.path.basename(__file__)[:-3]
this_root = dirname(dirname(__file__))

data_path = os.path.join(this_root, "data", run_name)
if not os.path.exists(data_path):
    os.mkdir(data_path)

picaso_root = dirname(dirname(jdi.__file__))

#1 ck tables from roxana
mh = '+000'#'+0.0' #log metallicity
CtoO = '100'#'1.0' # CtoO ratio
ck_db = join(picaso_root, f"data/kcoeff_2020/sonora_2020_feh{mh}_co_{CtoO}.data.196")
# ck_db = join(picaso_root, f"data/kcoeff_1060/m+0.0_co1.0.data.196")
opacity_ck = jdi.opannection(ck_db=ck_db) # grab your opacities
nlevel = 91 # number of plane-parallel levels in your code
nofczns = 1 # number of convective zones initially
nstr_upper = 88 # top most level of guessed convective zone
nstr_deep = nlevel - 2 # this is always the case. Dont change this
nstr = np.array([0,nstr_upper,nstr_deep,0,0,0])
nstr_start = np.copy(nstr)
rfacv = 0.5
tint = 500
grav_ms2 = 1
cl_run = jdi.inputs(calculation="planet", climate = True) # start a calculation
cl_run.star(opacity_ck, filename="../../runpicaso/data/solspec_picaso.dat", w_unit="um", f_unit="FLAM", semi_major=0.02, semi_major_unit=u.AU, radius=1.0, radius_unit=u.R_sun)
# cl_run.star(opacity_ck, temp=5772.0,metal=0.0, logg=4.43767, radius=1.0, radius_unit=u.R_sun,semi_major=0.02, semi_major_unit = u.AU)

r_star = cl_run.inputs['star']['radius'] 
r_star_unit = cl_run.inputs['star']['radius_unit'] 
semi_major = cl_run.inputs['star']['semi_major']
semi_major_unit = cl_run.inputs['star']['semi_major_unit'] 
fine_flux_star  = cl_run.inputs['star']['flux']  # erg/s/cm^2
FOPI = fine_flux_star * ((r_star/semi_major)**2)

cl_run.gravity(gravity=grav_ms2, gravity_unit=u.Unit('m/(s**2)')) # input gravity
cl_run.effective_temp(tint) # input effective temperature
# temp_guess = np.minimum(kazumasa_hj_grid_interpolation(1.0, semi_major, tint, grav_ms2), 5199.0)
temp_guess = np.load(join(picaso_root, "data/silicate_test_cases/HD189_temperature.npy"))
pressure_grid = np.logspace(-6, 2, 91)
cl_run.inputs_climate(temp_guess=np.copy(temp_guess), pressure=pressure_grid, nstr = nstr, nofczns = nofczns , rfacv = rfacv, cloudy = "cloudless", mh = '0.0', CtoO = '1.0',species = cloud_species, beta = 0.1, virga_param = 'const', mieff_dir = join(virga_root, "refrind"), do_holes = False, fhole = 0.5, fthin_cld = 0.9, moistgrad = False,
)
out_cloudless = deepcopy(cl_run.climate(opacity_ck))

# %%
