# %%
import warnings
warnings.filterwarnings('ignore')

import picaso.justdoit as jdi
import astropy.units as u
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from copy import deepcopy
from datetime import datetime

cloud_species = ["MgSiO3", "Mg2SiO4", "Fe", "Al2O3"]

#1 ck tables from roxana
mh = '+000'#'+0.0' #log metallicity
CtoO = '100'#'1.0' # CtoO ratio

ck_db = f"../data/kcoeff_2020/sonora_2020_feh{mh}_co_{CtoO}.data.196"

# %%
teff = 300
for nstr_upper in [88, 77]:
    for semi_major in [0.03, 0.05, 0.07]:
        for fsed in [1, 2, 3, 4]:
            print(f"fsed = {fsed}, semi major = {semi_major} au, nstr_upper = {nstr_upper}")
            cl_run = jdi.inputs(calculation="planet", climate = True) # start a calculation - need to not have "brown" in `calculation`. BD almost always means free-floating.

            grav = 25 # Gravity of your brown dwarf in m/s/s
            cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity
            cl_run.effective_temp(teff) # input effective temperature

            opacity_ck = jdi.opannection(ck_db=ck_db) # grab your opacities

            nlevel = 91 # number of plane-parallel levels in your code
            nofczns = 1 # number of convective zones initially
            nstr_deep = nlevel - 2 # this is always the case. Dont change this
            nstr = np.array([0,nstr_upper,nstr_deep,0,0,0])
            nstr_start = np.copy(nstr)
            rfacv = 0.5
            
            T_star = 5326.6 # K, star effective temperature
            logg = 4.38933 #logg , cgs
            metal = -0.03 # metallicity of star
            r_star = 0.932 # solar radius

            cl_run.star(opacity_ck, temp =T_star,metal =metal, logg =logg, radius = r_star, 
                        radius_unit=u.R_sun,semi_major= semi_major , semi_major_unit = u.AU)#opacity db, pysynphot database, temp, metallicity, logg
                
            cl_run.guillot_pt(1000, T_int=300, nlevel=91, p_bottom=2, p_top=-6)
            pt_df = cl_run.inputs["atmosphere"]["profile"]
            temp_guess, pressure_grid = np.array(pt_df["temperature"]), np.array(pt_df["pressure"])

            cl_run.inputs_climate(temp_guess=temp_guess, pressure=pressure_grid,
                                nstr = np.array([0,nstr_upper,nstr_deep,0,0,0]), nofczns = nofczns , rfacv = rfacv, cloudy = "selfconsistent", mh = '0.0', 
                                CtoO = '1.0',species = cloud_species, fsed = fsed, beta = 0.1, virga_param = 'const',
                                mieff_dir = "~/projects/clouds/virga/refrind", do_holes = False, fhole = 0.5, fthin_cld = 0.9, moistgrad = False,
                                )
            
            try:
                out_selfconsistent = deepcopy(cl_run.climate(opacity_ck, save_all_profiles=True,with_spec=True))
                
                cloud_outputs = {x: [] for x in ["temperature", "condensate_mmr", "cond_plus_gas_mmr", "cloud_deck", "altitude", "opd_by_gas"]}
                for k in cloud_outputs:
                    cloud_outputs[k] = np.array([x[k] for x in out_selfconsistent["cld"]])

                tstamp = datetime.now().isoformat().replace(":", ".")
                with h5py.File(f"../data/convh5_hotjupiter/convergence_fsed{fsed}_semimajor_{semi_major}_nstrupper{nstr_upper}_dt{tstamp}.h5", "w") as f:
                    p = f.create_dataset("pressure", data=out_selfconsistent["pressure"])
                    f.create_dataset("nstrs", data=np.array(out_selfconsistent["nstr"]))
                    for k in cloud_outputs:
                        f.create_dataset(k, data=cloud_outputs[k])
                    p.attrs["fsed"] = fsed
                    p.attrs["tint"] = teff
                    p.attrs["cloud_species"] = cloud_species
                    p.attrs["nstr_start"] = nstr_start
            except ValueError:
                with open(f"../data/convh5_hotjupiter/fsed{fsed}_semimajor_{semi_major}_nstrupper{nstr_upper}.txt") as f:
                    f.write(f"inf or NaN error at fsed = {fsed}, semi_major = {semi_major}, nstr_upper start = {nstr_upper}")

    # %%
