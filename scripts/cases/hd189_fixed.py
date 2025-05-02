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
import matplotlib
matplotlib.rcParams['animation.embed_limit'] = 2**128
import xarray
from copy import deepcopy

#1 ck tables from roxana
mh = '+000'#'+0.0' #log metallicity
CtoO = '100'#'1.0' # CtoO ratio

ck_db = f"../data/kcoeff_2020/sonora_2020_feh{mh}_co_{CtoO}.data.196"

fsed = 3.311

#sonora bobcat cloud free structures file
sonora_profile_db = '../data/sonora_bobcat/structures_m+0.0'
sonora_dat_db = '../data/sonora_bobcat/structures_m+0.0'

cl_run = jdi.inputs(calculation="planet", climate = True) # start a calculation
# pulled from Anna's spreadsheet
T_star = 5012.5 # K, star effective temperature
logg = 4.57 #logg , cgs
metal = 0.04 # metallicity of star
r_star = 0.782 # solar radius
semi_major = 0.03106 # star planet distance, AU
#1 ck tables from roxana
mh = '+000'#'+1.0' #log metallicity, 10xSolar
CtoO = '100'#'1.0' # CtoO ratio, Solar C/O

ck_db = f'/Users/adityasengupta/projects/clouds/picaso/data/kcoeff_2020/sonora_2020_feh{mh}_co_{CtoO}.data.196'
opacity_ck = jdi.opannection(ck_db=ck_db) # grab your opacities
cl_run.star(opacity_ck, temp =T_star,metal =metal, logg =logg, radius = r_star, 
            radius_unit=u.R_sun,semi_major= semi_major , semi_major_unit = u.AU)#opacity db, pysynphot database, temp, metallicity, logg

tint = 1800
grav = 100

cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity
cl_run.effective_temp(200) # input effective temperature

opacity_ck = jdi.opannection(ck_db=ck_db) # grab your opacities

nlevel = 91 # number of plane-parallel levels in your code
# hd189_pressure = np.load("../data/silicate_test_cases/hd189_pressure.npy")
hd189_pressure = np.logspace(-6, 2, 91)
hd189_temperature = np.load("../../data/silicate_test_cases/HD189_temperature.npy")
nofczns = 1 # number of convective zones initially
nstr_upper = 60 # top most level of guessed convective zone
nstr_deep = nlevel - 2 # this is always the case. Dont change this
nstr = np.array([0,nstr_upper,nstr_deep,0,0,0])
rfacv = 0.5 #we are focused on a brown dwarf so let's keep this as is

def twod_to_threed(arr, reps=4):
    """
    Takes in a 2D array of size (r, c) and repeats it along the last axis to make an array of size (r, c, reps).
    """
    return np.repeat(arr[:, :, np.newaxis], reps, axis=2)

cl_run.inputs_climate(temp_guess= np.copy(hd189_temperature), pressure= hd189_pressure,
                      nstr = nstr, nofczns = nofczns , rfacv = rfacv, cloudy = "cloudless", mh = '0.0', 
                      CtoO = '1.0',species = ['SiO2'], fsed = fsed, beta = 0.1, virga_param = 'const',
                      mieff_dir = "~/projects/clouds/virga/refrind", do_holes = False, fhole = 0.5, fthin_cld = 0.9, moistgrad = False,
                      )

# Restart in order to make a fixed cloud profile
bundle = jdi.inputs(calculation='planet')
bundle.phase_angle(0)
bundle.gravity(gravity=grav, gravity_unit=u.Unit('m/s**2'))
temp, pressure = hd189_temperature, hd189_pressure
bundle.add_pt(temp, pressure)
bundle.premix_atmosphere(opacity_ck, df = bundle.inputs['atmosphere']['profile'].loc[:,['pressure','temperature']])
DTAU, TAU, W0, COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, \
                    W0_no_raman, surf_reflect, ubar0,ubar1,cos_theta, single_phase,multi_phase, \
                    frac_a,frac_b,frac_c,constant_back,constant_forward, \
                    wno,nwno,ng,nt, nlevel, ngauss, gauss_wts, mmw, gweight,tweight = jdi.calculate_atm(bundle, opacity_ck)
bb, y2, tp = 0, 0, 0
FOPI = np.zeros(opacity_ck.nwno) + 1.0
Teff = cl_run.inputs["planet"]["T_eff"]
min_temp = min(opacity_ck.temps)
max_temp = max(opacity_ck.temps)
if Teff > 300:
    tmin = min_temp
else:
    tmin = 10
tmax = max_temp*(1.3)
dt = bundle.inputs['climate']['dt_bb_grid']
ntmps = int((tmax-tmin)/dt)
flux_net_v_layer_full, flux_net_v_full, flux_plus_v_full, flux_minus_v_full , flux_net_ir_layer_full, flux_net_ir_full, flux_plus_ir_full, flux_minus_ir_full = jdi.get_fluxes(pressure, temp, opacity_ck.delta_wno, bb , y2, tp, tmin, tmax, DTAU, TAU, W0, 
                    COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman, surf_reflect, 
                    ubar0,ubar1,cos_theta, FOPI, single_phase,multi_phase,frac_a,frac_b,frac_c,constant_back,constant_forward, 
                    wno,nwno,ng,nt,gweight,tweight, nlevel, ngauss, gauss_wts,True, True)#True for reflected, True for thermal
mean_molecular_weight = np.mean(mmw)
sigma_sb = 0.56687e-4
tidal = np.zeros_like(pressure) - sigma_sb *(Teff**4)
grad = bundle.inputs["climate"]["grad"]
cp = bundle.inputs['climate']['cp']
nstr = [0, 85, 89, 0, 0, 0]
kzz = jdi.get_kzz(
    hd189_temperature, hd189_pressure,grav,mmw,tidal,flux_net_ir_layer_full, flux_net_ir_layer_full,
    cl_run.inputs["climate"]["t_table"], cl_run.inputs["climate"]["p_table"], cl_run.inputs["climate"]["grad"], cl_run.inputs["climate"]["cp"],
    0,list(nstr),bundle.inputs['atmosphere']['profile'].T.values
)
# this virga run will use the cloudless-converged temperature profile
# and the associated Kzz
bundle.inputs['atmosphere']['profile']['temperature'] = temp
bundle.inputs['atmosphere']['profile']['kz'] = kzz
postproc_cld_out = bundle.virga(["SiO2"],"~/projects/clouds/virga/refrind", fsed=fsed,mh=1.0,mmw = mean_molecular_weight, b = 0.1, param = 'const')
postproc_cld_df = vj.picaso_format(postproc_cld_out["opd_per_layer"], postproc_cld_out["single_scattering"], postproc_cld_out["asymmetry"], postproc_cld_out["pressure"], 1e4 / postproc_cld_out["wave"])

"""cl_run.inputs["climate"]["cloudy"] = "fixed"
cl_run.inputs["climate"]["opd_climate"] = twod_to_threed(postproc_cld_out["opd_per_layer"])
cl_run.inputs["climate"]["w0_climate"] = twod_to_threed(postproc_cld_out["single_scattering"])
cl_run.inputs["climate"]["g0_climate"] = twod_to_threed(postproc_cld_out["asymmetry"])"""

# print("Fixed run")
out_fixed = deepcopy(cl_run.climate(opacity_ck, save_all_profiles=True,with_spec=True))

# %%
plt.semilogy(hd189_temperature, hd189_pressure, label="HD189 starting profile")
plt.semilogy(out_fixed["temperature"], out_fixed["pressure"], label="Fixed run")
plt.gca().invert_yaxis()
plt.legend()
# %%
plt.semilogy(postproc_cld_out["opd_per_layer"][:,98], out_fixed["pressure"][:-1])
plt.xlabel("OPD")
plt.ylabel("Pressure")
plt.gca().invert_yaxis()
# %%
