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

for teff in [1300, 1500, 1700, 1900]:
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
    nstr_upper = 60 # top most level of guessed convective zone
    nstr_deep = nlevel - 2 # this is always the case. Dont change this
    nstr = np.array([0,nstr_upper,nstr_deep,0,0,0])
    rfacv = 0.5 #we are focused on a brown dwarf so let's keep this as is

    def twod_to_threed(arr, reps=4):
        """
        Takes in a 2D array of size (r, c) and repeats it along the last axis to make an array of size (r, c, reps).
        """
        return np.repeat(arr[:, :, np.newaxis], reps, axis=2)


    print("Setting up atmosphere for cloudless run")
    cl_run.inputs_climate(temp_guess= temp_bobcat, pressure= pressure_bobcat,
                        nstr = nstr, nofczns = nofczns , rfacv = rfacv, cloudy = "cloudless", mh = '0.0', 
                        CtoO = '1.0',species = ['MgSiO3'], fsed = 1.0, beta = 0.1, virga_param = 'const',
                        mieff_dir = "~/projects/clouds/virga/refrind", do_holes = False, fhole = 0.5, fthin_cld = 0.9, moistgrad = False,
                        )


    out_cloudless = deepcopy(cl_run.climate(opacity_ck, save_all_profiles=True,with_spec=True))

    # Restart in order to make a postprocessed/fixed cloud profile
    bundle = jdi.inputs(calculation='brown')
    bundle.phase_angle(0)
    bundle.gravity(gravity=grav, gravity_unit=u.Unit('m/s**2'))
    temp, pressure = out_cloudless["temperature"], out_cloudless["pressure"]
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
    nstr = out_cloudless["cvz_locs"]
    kzz = jdi.get_kzz(
        out_cloudless["pressure"], out_cloudless["temperature"],grav,mmw,tidal,flux_net_ir_layer_full, flux_net_ir_layer_full,
        cl_run.inputs["climate"]["t_table"], cl_run.inputs["climate"]["p_table"], cl_run.inputs["climate"]["grad"], cl_run.inputs["climate"]["cp"],
        0,list(nstr),bundle.inputs['atmosphere']['profile'].T.values
    )
    # this virga run will use the cloudless-converged temperature profile
    # and the associated Kzz
    bundle.inputs['atmosphere']['profile']['temperature'] = temp
    bundle.inputs['atmosphere']['profile']['kz'] = kzz

    print("Making clouds off cloudless run for post-processed/fixed")
    postproc_cld_out = bundle.virga(["MgSiO3"],"~/projects/clouds/virga/refrind", fsed=1.0,mh=1.0,mmw = mean_molecular_weight, b = 0.1, param = 'const')
    postproc_cld_df = vj.picaso_format(postproc_cld_out["opd_per_layer"], postproc_cld_out["single_scattering"], postproc_cld_out["asymmetry"], postproc_cld_out["pressure"], 1e4 / postproc_cld_out["wave"])

    cl_run.inputs["climate"]["cloudy"] = "fixed"
    cl_run.inputs["climate"]["opd_climate"] = twod_to_threed(postproc_cld_out["opd_per_layer"])
    cl_run.inputs["climate"]["w0_climate"] = twod_to_threed(postproc_cld_out["single_scattering"])
    cl_run.inputs["climate"]["g0_climate"] = twod_to_threed(postproc_cld_out["asymmetry"])

    print("Fixed run")
    out_fixed = deepcopy(cl_run.climate(opacity_ck, save_all_profiles=True,with_spec=True))

    print("Self-consistent run")
    cl_run.inputs_climate(temp_guess=deepcopy(out_fixed["temperature"]), pressure= pressure_bobcat,
                        nstr = nstr, nofczns = nofczns , rfacv = rfacv, cloudy = "selfconsistent", mh = '0.0', 
                        CtoO = '1.0',species = ['MgSiO3'], fsed = 1.0, beta = 0.1, virga_param = 'const',
                        mieff_dir = "~/projects/clouds/virga/refrind", do_holes = False, fhole = 0.5, fthin_cld = 0.9, moistgrad = False,
                        )
    cl_run.inputs["climate"]["guess_temp"][np.isnan(out_fixed["temperature"])] = out_fixed["temperature"][0]

    out_selfconsistent = deepcopy(cl_run.climate(opacity_ck, save_all_profiles=True,with_spec=True))

    def calculate_spectrum(out, cld_out):
        opa_mon = jdi.opannection()

        hi_res = jdi.inputs(calculation="browndwarf") # start a calculation
        grav = 100 # Gravity of your brown dwarf in m/s/s
        hi_res.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity

        hi_res.atmosphere(df=out['ptchem_df'])
        if cld_out is not None:
            hi_res.clouds(df=cld_out)

        df_spec1 = hi_res.spectrum(opa_mon, calculation='thermal', full_output= True)

        wno, fp = df_spec1['wavenumber'], df_spec1['thermal'] #erg/cm2/s/cm
        xcm = 1/wno
        return xcm, fp, df_spec1['full_output']

    spectra = {}
    for (out, cld_out, run_name) in zip(
        [out_cloudless, out_cloudless, out_fixed, out_selfconsistent],
        [None, postproc_cld_df, postproc_cld_df, out_selfconsistent['cld_output_picaso']],
        ["cloudless", "postprocessed", "fixed", "selfconsistent"]
    ):
        xcm, fp, spec_out = calculate_spectrum(out, cld_out)
        spectra["x_cm"] = xcm
        spectra[run_name] = fp

    opd_fixed = twod_to_threed(postproc_cld_out["opd_per_layer"])[:,-150,0]
    all_wavenumbers = np.unique(out_selfconsistent['cld_output_picaso'].wavenumber.values)
    df_selfconsistent = out_selfconsistent['cld_output_picaso']
    opd_selfconsistent = df_selfconsistent[df_selfconsistent.wavenumber == all_wavenumbers[150]].opd.values

    with h5py.File(f"../data/four_clouds_testing/teff_{teff}_MgSiO3_browndwarf.h5", "w") as f:
        f.create_dataset("pressure", data=out_cloudless["pressure"])
        f.create_dataset("temp_cloudless", data=out_cloudless["temperature"])
        f.create_dataset("temp_fixed", data=out_fixed["temperature"])
        f.create_dataset("temp_selfconsistent", data=out_selfconsistent["temperature"])
        f.create_dataset("opd_fixed", data=opd_fixed)
        f.create_dataset("opd_selfconsistent", data=opd_selfconsistent)
        for k in spectra:
            f.create_dataset("spectrum_" + k, data=spectra[k])

# %%
