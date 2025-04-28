import warnings
warnings.filterwarnings('ignore')

import os
from os.path import join
import picaso.justdoit as jdi
import virga
import astropy.units as u
import numpy as np
import h5py
from copy import deepcopy
from datetime import datetime

picaso_root = os.path.dirname(os.path.dirname(jdi.__file__))
virga_root = os.path.dirname(os.path.dirname(virga.__file__))

cloud_species = ["MgSiO3", "Mg2SiO4", "Fe", "Al2O3"]

#1 ck tables from roxana
mh = '+000'#'+0.0' #log metallicity
CtoO = '100'#'1.0' # CtoO ratio

ck_db = join(picaso_root, f"data/kcoeff_2020/sonora_2020_feh{mh}_co_{CtoO}.data.196")

for teff in [1100, 1600, 2200]:
    for grav_ms2 in [316, 3160]:
        for fsed in [1, 8]:
            print(f"effective temperature = {teff} K, gravity = {grav_ms2}, fsed = {fsed}")
            nstr_upper = 88
            cl_run = jdi.inputs(calculation="browndwarf", climate = True) # start a calculation - need to not have "brown" in `calculation`. BD almost always means free-floating.
            cl_run.gravity(gravity=grav_ms2, gravity_unit=u.Unit('m/(s**2)')) # input gravity
            cl_run.effective_temp(teff) # input effective temperature
            opacity_ck = jdi.opannection(ck_db=ck_db) # grab your opacities
            nlevel = 91 # number of plane-parallel levels in your code
            nofczns = 1 # number of convective zones initially
            nstr_deep = nlevel - 2 # this is always the case. Dont change this
            nstr = np.array([0,nstr_upper,nstr_deep,0,0,0])
            nstr_start = np.copy(nstr)
            rfacv = 0.5
            
            with h5py.File(join(picaso_root, f"data/wrongside/wrongside_guesses_teff{teff}_grav{grav_ms2}_fsed{fsed}.h5")) as f:
                pressure_grid = np.array(f["pressure"])
                temp_guesses = [np.array(f["temperature_guess_"+x]) for x in ["nc", "radiative", "convective"]]
                nc_levels = [f.attrs[x] for x in ["nc", "nc_radiative", "nc_convective"]]
            
            for (nc, temp_guess) in zip(nc_levels, temp_guesses):
                cl_run.inputs_climate(temp_guess=temp_guess, pressure=pressure_grid,
                                    nstr = np.array([0,nstr_upper,nstr_deep,0,0,0]), nofczns = nofczns , rfacv = rfacv, cloudy = "selfconsistent", mh = '0.0', 
                                    CtoO = '1.0',species = cloud_species, fsed = fsed, beta = 0.1, virga_param = 'const',
                                    mieff_dir = join(virga_root, "refrind"), do_holes = False, fhole = 0.5, fthin_cld = 0.9, moistgrad = False,
                                    )
                
                try:
                    out_selfconsistent = deepcopy(cl_run.climate(opacity_ck, save_all_profiles=True,with_spec=True))
                    
                    cloud_outputs = {x: [] for x in ["temperature", "condensate_mmr", "cond_plus_gas_mmr", "cloud_deck"]}
                    for k in cloud_outputs:
                        cloud_outputs[k] = np.array([x[k] for x in out_selfconsistent["cld"]])

                    tstamp = datetime.now().isoformat().replace(":", ".")
                    with h5py.File(join(picaso_root, f"data/wrong_side_results/wrongside_selfconsistent_teff{teff}_gravms2{grav_ms2}_fsed{fsed}_nc{nc}_dt{tstamp}.h5", "w")) as f:
                        p = f.create_dataset("pressure", data=out_selfconsistent["pressure"])
                        f.create_dataset("nstrs", data=np.array(out_selfconsistent["nstr"]))
                        for k in cloud_outputs:
                            f.create_dataset(k, data=cloud_outputs[k])
                        p.attrs["fsed"] = fsed
                        p.attrs["teff"] = teff
                        p.attrs["cloud_species"] = cloud_species
                        p.attrs["nstr_start"] = nstr_start
                        t = f.create_dataset("temperature_picaso", data=out_selfconsistent["temperature"])
                except ValueError:
                    with open(join(picaso_root, f"data/wrong_side_results/wrongside_teff{teff}_gravms2{grav_ms2}_fsed{fsed}_nc{nc}_cloudmodeselfconsistent_dt{tstamp}.txt")) as f:
                        f.write("inf or NaN error")

