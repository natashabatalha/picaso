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
from matplotlib import animation

cloud_species = ["MgSiO3", "Mg2SiO4", "Fe", "Al2O3"]
# %%

#1 ck tables from roxana
mh = '+000'#'+0.0' #log metallicity
CtoO = '100'#'1.0' # CtoO ratio

ck_db = f"../data/kcoeff_2020/sonora_2020_feh{mh}_co_{CtoO}.data.196"

teff = 1100
for fsed in [1]:
    print(f"fsed = {fsed}, effective temperature = {teff} K")
    cl_run = jdi.inputs(calculation="browndwarf", climate = True) # start a calculation - need to not have "brown" in `calculation`. BD almost always means free-floating.

    grav = 3160 # Gravity of your brown dwarf in m/s/s
    cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity
    cl_run.effective_temp(teff) # input effective temperature

    opacity_ck = jdi.opannection(ck_db=ck_db) # grab your opacities

    nlevel = 91 # number of plane-parallel levels in your code
    nofczns = 1 # number of convective zones initially
    nstr_upper = 88 # top most level of guessed convective zone
    nstr_deep = nlevel - 2 # this is always the case. Dont change this
    nstr = np.array([0,nstr_upper,nstr_deep,0,0,0])
    rfacv = 0.5
    
    # sonora_df_cloudless = pd.read_csv(f"../data/sonora_diamondback/t{teff}g{grav}nc_m0.0_co1.0.pt", sep="\s+", skiprows=[1])
    sonora_df_cloudy = pd.read_csv(f"../data/sonora_diamondback/t{teff}g{grav}f{fsed}_m0.0_co1.0.pt", sep="\s+", skiprows=[1])

    # temp_guess = np.array(sonora_df_cloudless["T"])
    # print("Setting up atmosphere for cloudless run")
    pressure_grid = np.logspace(-6,2,91)

    """cl_run.inputs_climate(temp_guess= temp_guess, pressure=pressure_grid,
                        nstr = nstr, nofczns = nofczns , rfacv = rfacv, cloudy = "cloudless", mh = '0.0', 
                        CtoO = '1.0',species = cloud_species, fsed = fsed, beta = 0.1, virga_param = 'const',
                        mieff_dir = "~/projects/clouds/virga/refrind", do_holes = False, fhole = 0.5, fthin_cld = 0.9, moistgrad = False,
                        )

    out_cloudless = deepcopy(cl_run.climate(opacity_ck, save_all_profiles=True))"""

    print("Self-consistent run")

    # temp_guess = out_cloudless["temperature"]
    temp_guess = np.array(sonora_df_cloudy["T"])
    cl_run.inputs_climate(temp_guess=temp_guess, pressure=pressure_grid,
                        nstr = np.array([0,nstr_upper,nstr_deep,0,0,0]), nofczns = nofczns , rfacv = rfacv, cloudy = "selfconsistent", mh = '0.0', 
                        CtoO = '1.0',species = cloud_species, fsed = fsed, beta = 0.1, virga_param = 'const',
                        mieff_dir = "~/projects/clouds/virga/refrind", do_holes = False, fhole = 0.5, fthin_cld = 0.9, moistgrad = False,
                        )

    out_selfconsistent = deepcopy(cl_run.climate(opacity_ck, save_all_profiles=True,with_spec=True))
    np.save(f"../data/convergence_checking/fsed{fsed}_teff{teff}_all_opd.npy", out_selfconsistent["all_opd"])
    np.save(f"../data/convergence_checking/fsed{fsed}_teff{teff}_all_profiles.npy", out_selfconsistent["all_profiles"])
    with open(f"../data/convergence_checking/fsed{fsed}_teff{teff}_all_cloud.pkl", "wb") as f:
        pickle.dump(out_selfconsistent["all_cld_out"], f)

# %%

