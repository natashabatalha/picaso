import warnings
warnings.filterwarnings('ignore')
fsed = 1
cloudmode = "fixed"

import picaso.justdoit as jdi
import virga.justdoit as vj
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

ck_db = f"reference/kcoeff_2020/sonora_2020_feh{mh}_co_{CtoO}.data.196"

def twod_to_threed(arr, reps=4):
    """
    Takes in a 2D array of size (r, c) and repeats it along the last axis to make an array of size (r, c, reps).
    """
    return np.repeat(arr[:, :, np.newaxis], reps, axis=2)

for nstr_upper in [66]:
    for teff in [900, 1200, 1500, 1800, 2100]:
        print(f"fsed = {fsed}, effective temperature = {teff} K, nstr_upper = {nstr_upper}")
        cl_run = jdi.inputs(calculation="browndwarf", climate = True) # start a calculation - need to not have "brown" in `calculation`. BD almost always means free-floating.

        grav = 3160 # Gravity of your brown dwarf in m/s/s
        cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity
        cl_run.effective_temp(teff) # input effective temperature

        opacity_ck = jdi.opannection(ck_db=ck_db) # grab your opacities

        nlevel = 91 # number of plane-parallel levels in your code
        nofczns = 1 # number of convective zones initially
        nstr_deep = nlevel - 2 # this is always the case. Dont change this
        nstr = np.array([0,nstr_upper,nstr_deep,0,0,0])
        nstr_start = np.copy(nstr)
        rfacv = 0.5
        
        sonora_df_cloudy = pd.read_csv(f"./data/sonora_diamondback/t{teff}g{grav}f{fsed}_m0.0_co1.0.pt", sep=r"\s+", skiprows=[1])

        pressure_grid = np.logspace(-6,2,91)

        temp_guess = np.array(sonora_df_cloudy["T"])
        cl_run.inputs_climate(temp_guess=temp_guess, pressure=pressure_grid,
                            nstr = np.array([0,nstr_upper,nstr_deep,0,0,0]), nofczns = nofczns , rfacv = rfacv, cloudy = cloudmode, mh = '0.0', 
                            CtoO = '1.0',species = cloud_species, fsed = fsed, beta = 0.1, virga_param = 'const',
                            mieff_dir = "~/virga/refrind", do_holes = False, fhole = 0.5, fthin_cld = 0.9, moistgrad = False,
                            )

        # Restart in order to make a postprocessed/fixed cloud profile
        bundle = jdi.inputs(calculation='brown')
        bundle.phase_angle(0)
        bundle.gravity(gravity=grav, gravity_unit=u.Unit('m/s**2'))
        bundle.add_pt(temp_guess, pressure_grid)
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
        flux_net_v_layer_full, flux_net_v_full, flux_plus_v_full, flux_minus_v_full , flux_net_ir_layer_full, flux_net_ir_full, flux_plus_ir_full, flux_minus_ir_full = jdi.get_fluxes(pressure_grid, temp_guess, opacity_ck.delta_wno, bb , y2, tp, tmin, tmax, DTAU, TAU, W0, 
                            COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman, surf_reflect, 
                            ubar0,ubar1,cos_theta, FOPI, single_phase,multi_phase,frac_a,frac_b,frac_c,constant_back,constant_forward, 
                            wno,nwno,ng,nt,gweight,tweight, nlevel, ngauss, gauss_wts,True, True)#True for reflected, True for thermal
        mean_molecular_weight = np.mean(mmw)
        sigma_sb = 0.56687e-4
        tidal = np.zeros_like(pressure_grid) - sigma_sb *(Teff**4)
        grad = bundle.inputs["climate"]["grad"]
        cp = bundle.inputs['climate']['cp']
        kzz = jdi.get_kzz(
            pressure_grid, temp_guess,grav,mmw,tidal,flux_net_ir_layer_full, flux_net_ir_layer_full,
            cl_run.inputs["climate"]["t_table"], cl_run.inputs["climate"]["p_table"], cl_run.inputs["climate"]["grad"], cl_run.inputs["climate"]["cp"],
            0,list(nstr_start),bundle.inputs['atmosphere']['profile'].T.values
        )
        # this virga run will use the cloudless-converged temperature profile
        # and the associated Kzz
        bundle.inputs['atmosphere']['profile']['temperature'] = temp_guess
        bundle.inputs['atmosphere']['profile']['kz'] = kzz

        print("Making clouds off cloudless run for post-processed/fixed")
        postproc_cld_out = bundle.virga(cloud_species,"~/virga/refrind", fsed=fsed,mh=1.0,mmw = mean_molecular_weight, b = 0.1, param = 'const')
        postproc_cld_df = vj.picaso_format(postproc_cld_out["opd_per_layer"], postproc_cld_out["single_scattering"], postproc_cld_out["asymmetry"], postproc_cld_out["pressure"], 1e4 / postproc_cld_out["wave"])
        cl_run.inputs["climate"]["cloudy"] = "fixed"
        cl_run.inputs["climate"]["opd_climate"] = twod_to_threed(postproc_cld_out["opd_per_layer"])
        cl_run.inputs["climate"]["w0_climate"] = twod_to_threed(postproc_cld_out["single_scattering"])
        cl_run.inputs["climate"]["g0_climate"] = twod_to_threed(postproc_cld_out["asymmetry"])
                
        try:
            out_fixed = deepcopy(cl_run.climate(opacity_ck, save_all_profiles=True,with_spec=True))
            
            cloud_outputs = {x: [] for x in ["temperature", "condensate_mmr", "cond_plus_gas_mmr", "cloud_deck"]}
            for k in cloud_outputs:
                cloud_outputs[k] = np.array([x[k] for x in [postproc_cld_out]])

            tstamp = datetime.now().isoformat().replace(":", ".")
            with h5py.File(f"data/convh5_fixed/convergence_fsed{fsed}_teff{teff}_nstrupper{nstr_upper}_cloudmode{cloudmode}_dt{tstamp}.h5", "w") as f:
                p_picaso = f.create_dataset("pressure_picaso", data=out_fixed["pressure"])
                p_virga = f.create_dataset("pressure_virga", data=postproc_cld_out["pressure"])
                f.create_dataset("nstrs", data=np.array(out_fixed["nstr"]))
                for k in cloud_outputs:
                    f.create_dataset(k, data=cloud_outputs[k])
                f.create_dataset("altitude_virga", data=postproc_cld_out["altitude"].shape)
                p_virga.attrs["fsed"] = fsed
                p_virga.attrs["teff"] = teff
                p_virga.attrs["cloud_species"] = cloud_species
                p_virga.attrs["nstr_start"] = nstr_start
                t = f.create_dataset("temperature_picaso", data=out_fixed["all_profiles"])
        except ValueError:
            with open(f"./data/convh5_fixed/fsed{fsed}_teff{teff}_nstrupper{nstr_upper}.txt", "w") as f:
                f.write(f"inf or NaN error at fsed = {fsed}, teff = {teff}, nstr_upper start = {nstr_upper}")

