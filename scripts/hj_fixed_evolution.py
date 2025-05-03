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
import virga.justdoit as vj

picaso_root = dirname(dirname(jdi.__file__))
virga_root = dirname(dirname(virga.__file__))

cloud_species = ["MgSiO3", "Mg2SiO4", "Fe", "Al2O3"]

#1 ck tables from roxana
mh = '+000'#'+0.0' #log metallicity
CtoO = '100'#'1.0' # CtoO ratio
ck_db = join(picaso_root, f"data/kcoeff_2020/sonora_2020_feh{mh}_co_{CtoO}.data.196")
opacity_ck = jdi.opannection(ck_db=ck_db) # grab your opacities
nlevel = 91 # number of plane-parallel levels in your code
nofczns = 1 # number of convective zones initially
nstr_upper = 89 # top most level of guessed convective zone
nstr_deep = nlevel - 2 # this is always the case. Dont change this
nstr = np.array([0,nstr_upper,nstr_deep,0,0,0])
nstr_start = np.copy(nstr)
rfacv = 0.5

def twod_to_threed(arr, reps=4):
    """
    Takes in a 2D array of size (r, c) and repeats it along the last axis to make an array of size (r, c, reps).
    """
    return np.repeat(arr[:, :, np.newaxis], reps, axis=2)

def cloudless_and_fixed(tint, grav_ms2, fsed, semi_major):
    cl_run = jdi.inputs(calculation="planet", climate = True) # start a calculation
    cl_run.star(opacity_ck, temp=5772.0,metal=0.0, logg=4.43767, radius=1.0, radius_unit=u.R_sun,semi_major=semi_major, semi_major_unit = u.AU)#opacity db, pysynphot database, temp, metallicity, logg
    cl_run.gravity(gravity=grav_ms2, gravity_unit=u.Unit('m/(s**2)')) # input gravity
    cl_run.effective_temp(tint) # input effective temperature
    temp_guess = np.load(join(picaso_root, "data/silicate_test_cases/HD189_temperature.npy"))
    pressure_grid = np.logspace(-6, 2, 91)
    cl_run.inputs_climate(temp_guess=np.copy(temp_guess), pressure=pressure_grid, nstr = nstr, nofczns = nofczns , rfacv = rfacv, cloudy = "cloudless", mh = '0.0', CtoO = '1.0',species = cloud_species, beta = 0.1, virga_param = 'const', mieff_dir = join(virga_root, "refrind"), do_holes = False, fhole = 0.5, fthin_cld = 0.9, moistgrad = False,
    )
    out_cloudless = deepcopy(cl_run.climate(opacity_ck))
    bundle = jdi.inputs(calculation='brown')
    bundle.phase_angle(0)
    bundle.gravity(gravity=grav_ms2, gravity_unit=u.Unit('m/s**2'))
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
    sigma_sb = 0.56687e-4
    tidal = np.zeros_like(pressure_grid) - sigma_sb *(Teff**4)
    grad = bundle.inputs["climate"]["grad"]
    cp = bundle.inputs['climate']['cp']
    kzz = jdi.get_kzz(
        pressure_grid, temp_guess,grav_ms2,mmw,tidal,flux_net_ir_layer_full, flux_net_ir_layer_full,
        cl_run.inputs["climate"]["t_table"], cl_run.inputs["climate"]["p_table"], cl_run.inputs["climate"]["grad"], cl_run.inputs["climate"]["cp"],
        0,list(nstr_start),bundle.inputs['atmosphere']['profile'].T.values
    )
    # this virga run will use the cloudless-converged temperature profile
    # and the associated Kzz
    bundle.inputs['atmosphere']['profile']['temperature'] = temp_guess
    bundle.inputs['atmosphere']['profile']['kz'] = kzz
    
    postproc_cld_out = bundle.virga(cloud_species, join(virga_root, "refrind"), fsed=fsed, mh=1.0, mmw=2.35, b=0.1, param='const')
    cl_run.inputs["climate"]["cloudy"] = "fixed"
    cl_run.inputs["climate"]["opd_climate"] = twod_to_threed(postproc_cld_out["opd_per_layer"])
    cl_run.inputs["climate"]["w0_climate"] = twod_to_threed(postproc_cld_out["single_scattering"])
    cl_run.inputs["climate"]["g0_climate"] = twod_to_threed(postproc_cld_out["asymmetry"])
    out_fixed = deepcopy(cl_run.climate(opacity_ck))
    tstamp = datetime.now().isoformat().replace(":", ".")
    cloudless_and_fixed_path = join(picaso_root, f"data/cloudless_and_fixed/hj_cloudlessfixed_tint_{tint}_gravms2_{grav_ms2}_fsed_{fsed}_semimajor_{semi_major}_dt{tstamp}.h5")
    try:
        with h5py.File(cloudless_and_fixed_path, "w") as f:
            p = f.create_dataset("pressure", data=out_cloudless["pressure"])
            tcloudless = f.create_dataset("temperature_cloudless", data=out_cloudless["temperature"])
            tfixed = f.create_dataset("temperature_fixed", data=out_fixed["temperature"])
            qcfixed = f.create_dataset("condensate_mmr", data=postproc_cld_out["condensate_mmr"])
    except ValueError:
        with open(join(picaso_root, f"data/cloudless_and_fixed/hj_cloudlessfixed_tint_{tint}_gravms2_{grav_ms2}_fsed_{fsed}_semimajor_{semi_major}_dt{tstamp}.txt")) as f:
            f.write("inf or NaN error")

separations = [0.02, 0.05]
tints = [100, 300, 500, 750, 1000]
g = [1, 5, 10, 30, 50, 75, 100]
fseds = [1, 3, 8]
for tint in tints:
    for grav_ms2 in g:
        for fsed in fseds:
            for semi_major in separations:
                cloudless_and_fixed(tint, grav_ms2, fsed, semi_major)
