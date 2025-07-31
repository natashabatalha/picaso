from .atmsetup import ATMSETUP
from .fluxes import get_reflected_1d, get_reflected_3d , get_thermal_1d, get_thermal_3d, get_reflected_SH, get_thermal_SH,get_transit_1d, tidal_flux

from .climate import  namedtuple,run_chemeq_climate_workflow,run_diseq_climate_workflow


from .wavelength import get_cld_input_grid
from .optics import RetrieveOpacities,compute_opacity,RetrieveCKs
from .disco import get_angles_1d, get_angles_3d, compute_disco, compress_disco, compress_thermal
from .justplotit import numba_cumsum, mean_regrid
from .build_3d_input import regrid_xarray


from virga import justdoit as vj
from scipy.interpolate import UnivariateSpline, interp1d,RegularGridInterpolator
from scipy import special
from numpy import exp, sqrt,log
from numba import jit,njit
from scipy.io import FortranFile
import re

import os
import glob
import pickle as pk
import numpy as np
import pandas as pd
import copy
import json
import warnings

from synphot.models import Empirical1D
from synphot import SourceSpectrum
import stsynphot as sts

import astropy.units as u
import astropy.constants as c
from astropy.utils.misc import JsonCustomEncoder
import math
import xarray as xr
from joblib import Parallel, delayed, cpu_count
import h5py

# #testing error tracker
# from loguru import logger 
__refdata__ = os.environ.get('picaso_refdata')
__version__ = '3.3'


if not os.path.exists(__refdata__): 
    raise Exception("You have not downloaded the PICASO reference data. You can find it on github here: https://github.com/natashabatalha/picaso/tree/master/reference . If you think you have already downloaded it then you likely just need to set your environment variable. See instructions here: https://natashabatalha.github.io/picaso/installation.html#download-and-link-reference-documentation . You can use `os.environ['PYSYN_CDBS']=<yourpath>` directly in python if you run the line of code before you import PICASO.")
else: 
    ref_v = json.load(open(os.path.join(__refdata__,'config.json'))).get('version',2.3)
    
    if __version__ != str(ref_v): 
        warnings.warn(f"Your code version is {__version__} but your reference data version is {ref_v}. For some functionality you may experience Keyword errors. Please download the newest ref version or update your code: https://github.com/natashabatalha/picaso/tree/master/reference")


if not os.path.exists(os.environ.get('PYSYN_CDBS')): 
    warnings.warn("You have not downloaded the Stellar reference data. If you only plan on working on substellar objects that is okay but for exoplanets it will be required. Follow the installation instructions here: https://natashabatalha.github.io/picaso/installation.html#download-and-link-pysynphot-stellar-data. If you think you have already downloaded it then you likely just need to set your environment variable. You can use `os.environ['PYSYN_CDBS']=<yourpath>` directly in python if you run the line of code before you import PICASO.")

#hello peter

def picaso(bundle,opacityclass, dimension = '1d',calculation='reflected', 
    full_output=False, plot_opacity= False, as_dict=True):
    """
    Currently top level program to run albedo code 
    Parameters 
    ----------
    bundle : dict 
        This input dict is built by loading the input = `justdoit.load_inputs()` 
    opacityclass : class
        Opacity class from `justdoit.opannection`
    dimension : str 
        (Optional) Dimensions of the calculation. Default = '1d'. But '3d' is also accepted. 
        In order to run '3d' calculations, user must build 3d input (see tutorials)
    full_output : bool 
        (Optional) Default = False. Returns atmosphere class, which enables several 
        plotting capabilities. 
    plot_opacity : bool 
        (Optional) Default = False, Creates pop up of the weighted opacity
    as_dict : bool 
        (Optional) Default = True. If true, returns a condensed dictionary to the user. 
        If false, returns the atmosphere class, which can be used for debugging. 
        The class is clunky to navigate so if you are consiering navigating through this, ping one of the 
        developers. 
    Return
    ------
    dictionary with albedos or fluxes or both (depending on what calculation type)
    """
    inputs = bundle.inputs

    wno = opacityclass.wno
    nwno = opacityclass.nwno
    ngauss = opacityclass.ngauss
    gauss_wts = opacityclass.gauss_wts #for opacity

    #check to see if we are running in test mode
    test_mode = inputs['test_mode']

    ############# DEFINE ALL APPROXIMATIONS USED IN CALCULATION #############
    #see class `inputs` attribute `approx`
    #what rt method are we using?? 
    rt_method = inputs['approx']['rt_method'] #either toon or spherical harmonics
    
    #USED by all RT
    stream = inputs['approx']['rt_params']['common']['stream']
    #parameters needed for the two term hg phase function. 
    #Defaults are set in config.json
    f = inputs['approx']['rt_params']['common']['TTHG_params']['fraction']
    frac_a = f[0]
    frac_b = f[1]
    frac_c = f[2]
    constant_back = inputs['approx']['rt_params']['common']['TTHG_params']['constant_back']
    constant_forward = inputs['approx']['rt_params']['common']['TTHG_params']['constant_forward']
    raman_approx =inputs['approx']['rt_params']['common']['raman']
    #define delta eddington approximinations 
    delta_eddington = inputs['approx']['rt_params']['common']['delta_eddington']


    #USED in TOON (if being used)
    single_phase = inputs['approx']['rt_params']['toon']['single_phase']
    toon_coefficients = inputs['approx']['rt_params']['toon']['toon_coefficients']
    multi_phase = inputs['approx']['rt_params']['toon']['multi_phase']

    #USED in SH (if being used)
    single_form = inputs['approx']['rt_params']['SH']['single_form']
    w_single_form = inputs['approx']['rt_params']['SH']['w_single_form']
    w_multi_form = inputs['approx']['rt_params']['SH']['w_multi_form']
    psingle_form = inputs['approx']['rt_params']['SH']['psingle_form']
    w_single_rayleigh = inputs['approx']['rt_params']['SH']['w_single_rayleigh']
    w_multi_rayleigh = inputs['approx']['rt_params']['SH']['w_multi_rayleigh']
    psingle_rayleigh = inputs['approx']['rt_params']['SH']['psingle_rayleigh']
    calculate_fluxes = inputs['approx']['rt_params']['SH']['calculate_fluxes']


    #for patchy clouds
    do_holes = inputs['clouds'].get('do_holes',False)
    if do_holes == True:
        fhole = inputs['clouds']['fhole']
        fthin_cld = inputs['clouds']['fthin_cld']

    #save level fluxes in addition to the top of atmosphere fluxes?
    #default is false
    get_lvl_flux = inputs['approx'].get('get_lvl_flux',False)


    #pressure assumption
    p_reference =  inputs['approx']['p_reference']

    ############# DEFINE ALL GEOMETRY USED IN CALCULATION #############
    #see class `inputs` attribute `phase_angle`
    

    #phase angle 
    phase_angle = inputs['phase_angle']
    #get geometry
    geom = inputs['disco']

    ng, nt = geom['num_gangle'], geom['num_tangle']
    gangle,gweight,tangle,tweight = geom['gangle'], geom['gweight'],geom['tangle'], geom['tweight']
    lat, lon = geom['latitude'], geom['longitude']
    cos_theta = geom['cos_theta']
    ubar0, ubar1 = geom['ubar0'], geom['ubar1']
   # ubar1 = np.array([[-.99999],[-0.5],[0.5],[1.]])
   # ubar0 = np.array([[1/np.sqrt(2)],[1/np.sqrt(2)],[1/np.sqrt(2)],[1/np.sqrt(2)]])
   # ng = 4; nt = 1

    #set star parameters
    radius_star = inputs['star']['radius']

    #need to account for case where there is no star
    if 'nostar' in inputs['star']['database']:
        F0PI = np.zeros(opacityclass.nwno) + 1.0
    else:
        F0PI = opacityclass.relative_flux

    b_top = 0.
    #semi major axis
    sa = inputs['star']['semi_major']

    #begin atm setup
    atm = ATMSETUP(inputs)


    #Add inputs to class 
    atm.surf_reflect = inputs['surface_reflect']
    atm.hard_surface = inputs['hard_surface']#0=no hard surface, 1=hard surface
    atm.wavenumber = wno
    atm.planet.gravity = inputs['planet']['gravity']
    atm.planet.radius = inputs['planet']['radius']
    atm.planet.mass = inputs['planet']['mass']
    atm.get_lvl_flux=get_lvl_flux

    if dimension == '1d':
        atm.get_profile()
    elif dimension == '3d':
        atm.get_profile_3d()

    #now can get these 
    atm.get_mmw()
    atm.get_density()
    atm.get_altitude(p_reference = p_reference)#will calculate altitude if r and m are given (opposed to just g)
    atm.get_column_density()

    #gets both continuum and needed rayleigh cross sections 
    #relies on continuum molecules are added into the opacity 
    #database. Rayleigh molecules are all in `rayleigh.py` 
    atm.get_needed_continuum(opacityclass.rayleigh_molecules,
                             opacityclass.avail_continuum)

    #get cloud properties, if there are any and put it on current grid 
    atm.get_clouds(wno)

    #Make sure that all molecules are in opacityclass. If not, remove them and add warning
    no_opacities = [i for i in atm.molecules if i not in opacityclass.molecules]
    atm.add_warnings('No computed opacities for: '+','.join(no_opacities))
    atm.molecules = np.array([ x for x in atm.molecules if x not in no_opacities ])
    
    #opacity assumptions
    exclude_mol = inputs['atmosphere']['exclude_mol']

    get_opacities = opacityclass.get_opacities

    nlevel = atm.c.nlevel
    nlayer = atm.c.nlayer
    

    if dimension == '1d':
        #lastly grab needed opacities for the problem
        get_opacities(atm,exclude_mol=exclude_mol)
        #only need to get opacities for one pt profile

        #There are two sets of dtau,tau,w0,g in the event that the user chooses to use delta-eddington
        #We use HG function for single scattering which gets the forward scattering/back scattering peaks 
        #well. We only really want to use delta-edd for multi scattering legendre polynomials. 

        DTAU, TAU, W0, COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman, f_deltaM= compute_opacity(
            atm, opacityclass, ngauss=ngauss, stream=stream, delta_eddington=delta_eddington,test_mode=test_mode,raman=raman_approx,
            full_output=full_output, plot_opacity=plot_opacity)

        #do we want soem clear patches ? If so, specify the thining through fthin where=0 is a fully clear patch
        if do_holes:
            DTAU_clear, TAU_clear, W0_clear, COSB_clear,ftau_cld_clear, ftau_ray_clear,GCOS2_clear, DTAU_OG_clear, TAU_OG_clear, W0_OG_clear, COSB_OG_clear, \
                W0_no_raman_clear, f_deltaM= compute_opacity(
                atm, opacityclass, ngauss=ngauss, stream=stream, delta_eddington=delta_eddington,test_mode=test_mode,raman=raman_approx,
                full_output=full_output, plot_opacity=plot_opacity, fthin_cld = fthin_cld, do_holes = True)
        
        if  'reflected' in calculation:
            xint_at_top = 0 
            for ig in range(ngauss): # correlated - loop (which is different from gauss-tchevychev angle)
                nlevel = atm.c.nlevel

                if rt_method == 'SH':
                    (xint, flux_out)  = get_reflected_SH(nlevel, nwno, ng, nt, 
                                    DTAU[:,:,ig], TAU[:,:,ig], W0[:,:,ig], COSB[:,:,ig], 
                                    ftau_cld[:,:,ig], ftau_ray[:,:,ig], f_deltaM[:,:,ig],
                                    DTAU_OG[:,:,ig], TAU_OG[:,:,ig], W0_OG[:,:,ig], COSB_OG[:,:,ig], 
                                    atm.surf_reflect, ubar0, ubar1, cos_theta, F0PI, 
                                    w_single_form, w_multi_form, psingle_form, 
                                    w_single_rayleigh, w_multi_rayleigh, psingle_rayleigh,
                                    frac_a, frac_b, frac_c, constant_back, constant_forward, 
                                    stream, b_top=b_top, flx=calculate_fluxes, 
                                    single_form=single_form) 
                #only other rt scheme is toon right now 
                else:
                    if get_lvl_flux: 
                        atm.lvl_output_reflected = dict(flux_minus=0, flux_plus=0, flux_minus_mdpt=0, flux_plus_mdpt=0)
                    
                    xint,lvl_fluxes = get_reflected_1d(nlevel, wno,nwno,ng,nt,
                                    DTAU[:,:,ig], TAU[:,:,ig], W0[:,:,ig], COSB[:,:,ig],
                                    GCOS2[:,:,ig],ftau_cld[:,:,ig],ftau_ray[:,:,ig],
                                    DTAU_OG[:,:,ig], TAU_OG[:,:,ig], W0_OG[:,:,ig], COSB_OG[:,:,ig],
                                    atm.surf_reflect, ubar0,ubar1,cos_theta, F0PI,
                                    single_phase,multi_phase,
                                    frac_a,frac_b,frac_c,constant_back,constant_forward,
                                    get_toa_intensity=1,get_lvl_flux=int(atm.get_lvl_flux),
                                    toon_coefficients=toon_coefficients,b_top=b_top)
                    
                    flux_minus_all_v, flux_plus_all_v, flux_minus_midpt_all_v, flux_plus_midpt_all_v = lvl_fluxes

                    if do_holes == True:
                        xint_clear, out_ref_fluxes_clear = get_reflected_1d(nlevel, wno,nwno,ng,nt,
                                DTAU_clear[:,:,ig], TAU_clear[:,:,ig], W0_clear[:,:,ig], COSB_clear[:,:,ig],
                                GCOS2[:,:,ig],ftau_cld[:,:,ig],ftau_ray[:,:,ig],
                                DTAU_OG_clear[:,:,ig], TAU_OG_clear[:,:,ig], W0_OG_clear[:,:,ig], COSB_OG_clear[:,:,ig],
                                atm.surf_reflect, ubar0,ubar1,cos_theta, F0PI,
                                single_phase,multi_phase,
                                frac_a,frac_b,frac_c,constant_back,constant_forward, 
                                get_toa_intensity=1, get_lvl_flux=int(atm.get_lvl_flux),
                                toon_coefficients=toon_coefficients,b_top=b_top)
                    
                        flux_minus_all_v_clear, flux_plus_all_v_clear, flux_minus_midpt_all_v_clear, flux_plus_midpt_all_v_clear = out_ref_fluxes_clear
                        
                        #weighted average of cloudy and clearsky
                        flux_plus_midpt_all_v = (1.0 - fhole)* flux_plus_midpt_all_v + fhole * flux_plus_midpt_all_v_clear
                        flux_minus_midpt_all_v = (1.0 - fhole)* flux_minus_midpt_all_v + fhole * flux_minus_midpt_all_v_clear
                        flux_plus_all_v = (1.0 - fhole)* flux_plus_all_v + fhole * flux_plus_all_v_clear
                        flux_minus_all_v = (1.0 - fhole)* flux_minus_all_v + fhole * flux_minus_all_v_clear
                        xint = (1.0 - fhole)* xint + fhole * xint_clear

                xint_at_top += xint*gauss_wts[ig]

                if get_lvl_flux: 
                    atm.lvl_output_reflected['flux_minus']+=flux_minus_all_v*gauss_wts[ig]
                    atm.lvl_output_reflected['flux_plus']+=flux_plus_all_v*gauss_wts[ig]
                    atm.lvl_output_reflected['flux_minus_mdpt']+=flux_minus_midpt_all_v*gauss_wts[ig]
                    atm.lvl_output_reflected['flux_plus_mdpt']+=flux_plus_midpt_all_v*gauss_wts[ig]


        
        if 'thermal' in calculation:
            #use toon method (and tridiagonal matrix solver) to get net cumulative fluxes 
            flux_at_top = 0 

            if get_lvl_flux: 
                calc_type=1
                atm.lvl_output_thermal = dict(flux_minus=0, flux_plus=0, flux_minus_mdpt=0, flux_plus_mdpt=0)
            else: 
                calc_type=0

            for ig in range(ngauss): # correlated-k - loop (which is different from gauss-tchevychev angle)
                
                #remember all OG values (e.g. no delta eddington correction) go into thermal as well as 
                #the uncorrected raman single scattering 
                if rt_method == 'toon':
                    #flux  = get_thermal_1d(nlevel, wno,nwno,ng,nt,atm.level['temperature'],
                    #                                    DTAU_OG[:,:,ig], W0_no_raman[:,:,ig], COSB_OG[:,:,ig], 
                    #                                    atm.level['pressure'],ubar1,
                    #                                    atm.surf_reflect, atm.hard_surface, tridiagonal)
                    flux,lvl_fluxes  = get_thermal_1d(nlevel, wno,nwno,ng,nt,atm.level['temperature'],
                                                        DTAU_OG[:,:,ig], W0_no_raman[:,:,ig], COSB_OG[:,:,ig], 
                                                        atm.level['pressure'],ubar1,
                                                        atm.surf_reflect, atm.hard_surface,
                                                        #setting wno to zero since only used for climate, calctype only gets TOA flx 
                                                        wno*0, calc_type=calc_type)
                    
                    flux_minus_all_i, flux_plus_all_i, flux_minus_midpt_all_i, flux_plus_midpt_all_i = lvl_fluxes
                    
                    if do_holes == True:
                        #clearsky case
                        flux_clear,out_therm_fluxes_clear = get_thermal_1d(nlevel, wno,nwno,ng,nt,atm.level['temperature'],
                                                    DTAU_OG_clear[:,:,ig], W0_no_raman_clear[:,:,ig], COSB_OG_clear[:,:,ig], 
                                                    atm.level['pressure'],ubar1,
                                                    atm.surf_reflect, atm.hard_surface,
                                                    wno*0, calc_type=calc_type)
                        
                        flux_minus_all_i_clear, flux_plus_all_i_clear, flux_minus_midpt_all_i_clear, flux_plus_midpt_all_i_clear= out_therm_fluxes_clear
                        
                        #weighted average of cloudy and clearsky
                        flux_plus_midpt_all_i = (1.0 - fhole)* flux_plus_midpt_all_i + fhole * flux_plus_midpt_all_i_clear
                        flux_minus_midpt_all_i = (1.0 - fhole)* flux_minus_midpt_all_i + fhole * flux_minus_midpt_all_i_clear
                        flux_plus_all_i = (1.0 - fhole)* flux_plus_all_i + fhole * flux_plus_all_i_clear
                        flux_minus_all_i = (1.0 - fhole)* flux_minus_all_i + fhole * flux_minus_all_i_clear
                        flux = (1.0 - fhole)*flux + fhole * flux_clear


                elif rt_method == 'SH':
                    flux, flux_layers = get_thermal_SH(nlevel, wno, nwno, ng, nt, atm.level['temperature'],
                                                DTAU[:,:,ig], TAU[:,:,ig], W0[:,:,ig], COSB[:,:,ig], 
                                                DTAU_OG[:,:,ig], TAU_OG[:,:,ig], W0_OG[:,:,ig], 
                                                W0_no_raman[:,:,ig], COSB_OG[:,:,ig], 
                                                atm.level['pressure'], ubar1, 
                                                atm.surf_reflect, stream, atm.hard_surface)

                if ((rt_method == 'toon') & get_lvl_flux): 
                    #ck-table gauss points 
                    atm.lvl_output_thermal['flux_minus']+=flux_minus_all_i*gauss_wts[ig]
                    atm.lvl_output_thermal['flux_plus']+=flux_plus_all_i*gauss_wts[ig]
                    atm.lvl_output_thermal['flux_minus_mdpt']+=flux_minus_midpt_all_i*gauss_wts[ig]
                    atm.lvl_output_thermal['flux_plus_mdpt']+=flux_plus_midpt_all_i*gauss_wts[ig]
                
                #ck-table gauss points 
                flux_at_top += flux*gauss_wts[ig]
                
                
            #if full output is requested add in flux at top for 3d plots
            if full_output: 
                atm.flux_at_top = flux_at_top

        
        if 'transmission' in calculation:
            rprs2 = 0 
            for ig in range(ngauss): # correlated - loop (which is different from gauss-tchevychev angle)

                rprs2_g = get_transit_1d(atm.level['z'],atm.level['dz'],
                                  nlevel, nwno, radius_star, atm.layer['mmw'], 
                                  atm.c.k_b, atm.c.amu, atm.level['pressure'], 
                                  atm.level['temperature'], atm.layer['colden'],
                                  DTAU_OG[:,:,ig])
                
                if do_holes: 
                    rprs2_g_clear = get_transit_1d(atm.level['z'],atm.level['dz'],
                                  nlevel, nwno, radius_star, atm.layer['mmw'], 
                                  atm.c.k_b, atm.c.amu, atm.level['pressure'], 
                                  atm.level['temperature'], atm.layer['colden'],
                                  DTAU_OG_clear[:,:,ig])
                    rprs2_g = (1.0 - fhole)*rprs2_g + fhole * rprs2_g_clear
                
                rprs2 += rprs2_g*gauss_wts[ig]
    elif dimension == '3d':
        #setup zero array to fill with opacities
        TAU_3d = np.zeros((nlevel, nwno, ng, nt, ngauss))
        DTAU_3d = np.zeros((nlayer, nwno, ng, nt, ngauss))
        W0_3d = np.zeros((nlayer, nwno, ng, nt, ngauss))
        COSB_3d = np.zeros((nlayer, nwno, ng, nt, ngauss))
        GCOS2_3d = np.zeros((nlayer, nwno, ng, nt, ngauss))
        FTAU_CLD_3d = np.zeros((nlayer, nwno, ng, nt, ngauss))
        FTAU_RAY_3d = np.zeros((nlayer, nwno, ng, nt, ngauss))

        #these are the unchanged values from delta-eddington
        TAU_OG_3d = np.zeros((nlevel, nwno, ng, nt, ngauss))
        DTAU_OG_3d = np.zeros((nlayer, nwno, ng, nt, ngauss))
        W0_OG_3d = np.zeros((nlayer, nwno, ng, nt, ngauss))
        COSB_OG_3d = np.zeros((nlayer, nwno, ng, nt, ngauss))
        #this is the single scattering without the raman correction 
        #used for the thermal caclulation
        W0_no_raman_3d = np.zeros((nlayer, nwno, ng, nt, ngauss))

        #pressure and temperature 
        TLEVEL_3d = np.zeros((nlevel, ng, nt))
        PLEVEL_3d = np.zeros((nlevel, ng, nt))

        #if users want to retain all the individual opacity info they can here 
        if full_output:
            TAUGAS_3d = np.zeros((nlayer, nwno, ng, nt, ngauss))
            TAUCLD_3d = np.zeros((nlayer, nwno, ng, nt, ngauss))
            TAURAY_3d = np.zeros((nlayer, nwno, ng, nt, ngauss))  

        #get opacities at each facet
        #sample MPI code 
        #from mpi4py improt MPI
        #comm = MPI.COMM_WORLD
        #rank = comm.Get_rank()
        #size = comm.Get_size()
        #idx = rank-1 
        #gts = [[g,t] for g in range(ng) for t in range(nt)]
        for g in range(ng):
            for t in range(nt): 
                #g,t = gts[idx]
                #edit atm class to only have subsection of 3d stuff 
                atm_1d = copy.deepcopy(atm)

                #diesct just a subsection to get the opacity 
                atm_1d.disect(g,t)

                get_opacities(atm_1d)

                dtau, tau, w0, cosb,ftau_cld, ftau_ray, gcos2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, WO_no_raman, f_deltaM = compute_opacity(
                    atm_1d, opacityclass, ngauss=ngauss, stream=stream,delta_eddington=delta_eddington,
                    test_mode=test_mode,raman=raman_approx, full_output=full_output)
                DTAU_3d[:,:,g,t,:] = dtau
                TAU_3d[:,:,g,t,:] = tau
                W0_3d[:,:,g,t,:] = w0 
                COSB_3d[:,:,g,t,:] = cosb
                GCOS2_3d[:,:,g,t,:]= gcos2 
                FTAU_CLD_3d[:,:,g,t,:]= ftau_cld
                FTAU_RAY_3d[:,:,g,t,:]= ftau_ray

                #these are the unchanged values from delta-eddington
                TAU_OG_3d[:,:,g,t,:] = TAU_OG
                DTAU_OG_3d[:,:,g,t,:] = DTAU_OG
                W0_OG_3d[:,:,g,t,:] = W0_OG
                COSB_OG_3d[:,:,g,t,:] = COSB_OG
                W0_no_raman_3d[:,:,g,t,:] = WO_no_raman

                #temp and pressure on 3d grid
                
                TLEVEL_3d[:,g,t] = atm_1d.level['temperature']
                PLEVEL_3d[:,g,t] = atm_1d.level['pressure']

                if full_output:
                    TAUGAS_3d[:,:,g,t,:] = atm_1d.taugas
                    TAUCLD_3d[:,:,g,t,:] = atm_1d.taucld
                    TAURAY_3d[:,:,g,t,:] = atm_1d.tauray

        if full_output:
            atm.taugas = TAUGAS_3d
            atm.taucld = TAUCLD_3d
            atm.tauray = TAURAY_3d

        if  'reflected' in calculation:
            xint_at_top=0
            for ig in range(ngauss): # correlated - loop (which is different from gauss-tchevychev angle)
                #use toon method  to get net cumulative fluxes 
                xint  = get_reflected_3d(nlevel, wno,nwno,ng,nt,
                                                DTAU_3d[:,:,:,:,ig], TAU_3d[:,:,:,:,ig], W0_3d[:,:,:,:,ig], COSB_3d[:,:,:,:,ig],GCOS2_3d[:,:,:,:,ig],
                                                FTAU_CLD_3d[:,:,:,:,ig],FTAU_RAY_3d[:,:,:,:,ig],
                                                DTAU_OG_3d[:,:,:,:,ig], TAU_OG_3d[:,:,:,:,ig], W0_OG_3d[:,:,:,:,ig], COSB_OG_3d[:,:,:,:,ig],
                                                atm.surf_reflect, ubar0,ubar1,cos_theta, F0PI,
                                                single_phase,multi_phase,
                                                frac_a,frac_b,frac_c,constant_back,constant_forward)
                xint_at_top += xint*gauss_wts[ig]

                #if full output is requested add in xint at top for 3d plots
            if full_output: 
                atm.xint_at_top = xint_at_top

        if 'thermal' in calculation:
            flux_at_top=0
            for ig in range(ngauss): # correlated - loop (which is different from gauss-tchevychev angle)
                #remember all OG values (e.g. no delta eddington correction) go into thermal as well as 
                #the uncorrected raman single scattering 
                flux  = get_thermal_3d(nlevel, wno,nwno,ng,nt,TLEVEL_3d,
                                            DTAU_OG_3d[:,:,:,:,ig], W0_no_raman_3d[:,:,:,:,ig], COSB_OG_3d[:,:,:,:,ig], 
                                            PLEVEL_3d,ubar1, atm.surf_reflect, atm.hard_surface)
                flux_at_top += flux*gauss_wts[ig]
            #if full output is requested add in flux at top for 3d plots
            if full_output: 
                atm.flux_at_top = flux_at_top

    #set up initial returns
    returns = {}
    returns['wavenumber'] = wno
    #returns['flux'] = flux
    if 'transmission' in calculation: 
        returns['transit_depth'] = rprs2

    #COMPRESS FULL TANGLE-GANGLE FLUX OUTPUT ONTO 1D FLUX GRID

    #for reflected light use compress_disco routine
    #this takes the intensity as a function of tangle/gangle and creates a 1d spectrum
    if  ('reflected' in calculation):
        albedo = compress_disco(nwno, cos_theta, xint_at_top, gweight, tweight,F0PI)
        returns['albedo'] = albedo 

        # This is attempt to get the compress_disco to return the integrated fluxes
        # However, its mixing an albedo calc and an itegration calc I think

        if ((rt_method == 'toon') & get_lvl_flux): 
            #for i in atm.lvl_output_reflected.keys():
            for key, data in atm.lvl_output_reflected.items():
                #atm.lvl_output_reflected[i] = compress_disco(nwno,cos_theta,atm.lvl_output_reflected[i], gweight, tweight,F0PI)   

                # Get the number of layers to do the layer slicing
                _, _, nlayer, nwno_data = data.shape

                # Integrate each layer
                atm.lvl_output_reflected[key] = np.array([compress_disco(nwno_data,
                                                                         cos_theta,
                                                                         data[:, :, layer_idx, :],
                                                                         gweight, tweight, np.zeros(nwno) + 1.) for layer_idx in range(nlayer)])


        #see equation 18 Batalha+2019 PICASO 
        returns['bond_albedo'] = (np.trapz(x=1/wno, y=albedo*opacityclass.unshifted_stellar_spec)/
                                    np.trapz(x=1/wno, y=opacityclass.unshifted_stellar_spec))

        if ((not np.isnan(sa ) and (not np.isnan(atm.planet.radius))) ):
            returns['fpfs_reflected'] = albedo*(atm.planet.radius/sa)**2.0
        else: 
            returns['fpfs_reflected'] =[]
            if np.isnan(sa ):
                returns['fpfs_reflected'] += ['Semi-major axis not supplied. If you want fpfs, add it to `star` function. ']
            if np.isnan(atm.planet.radius): 
                returns['fpfs_reflected'] += ['Planet Radius not supplied. If you want fpfs, add it to `gravity` function with a mass.']

    #for thermal light use the compress thermal routine
    #this takes the intensity as a functin of tangle/gangle and creates a 1d spectrum
    if ('thermal' in calculation):
        thermal = compress_thermal(nwno,flux_at_top, gweight, tweight)
        returns['thermal'] = thermal
        returns['thermal_unit'] = 'erg/s/(cm^2)/(cm)'#'erg/s/(cm^2)/(cm^(-1))'
        returns['effective_temperature'] = (np.trapz(x=1/wno[::-1], y=thermal[::-1])/5.67e-5)**0.25

        if full_output: 
            atm.thermal_flux_planet = thermal

        if ((rt_method == 'toon') & get_lvl_flux): 
            for i in atm.lvl_output_thermal.keys():
                delta_wno = getattr(opacityclass,'delta_wno', np.concatenate((np.diff(opacityclass.wno),[np.diff(opacityclass.wno)[-1]])))
                disk_integrated_lvl_flux = compress_thermal(nwno,atm.lvl_output_thermal[i], gweight, tweight)  
                energy_per_wave_bin = disk_integrated_lvl_flux*delta_wno 
                atm.lvl_output_thermal[i] = energy_per_wave_bin

        #only need to return relative flux if not a browndwarf calculation
        if radius_star == 'nostar': 
            returns['fpfs_thermal'] = ['No star mode for Brown Dwarfs was used']
        elif ((not np.isnan(atm.planet.radius)) & (not np.isnan(radius_star))) :
            fpfs_thermal = thermal/(opacityclass.unshifted_stellar_spec)*(atm.planet.radius/radius_star)**2.0
            returns['fpfs_thermal'] = fpfs_thermal
        else:
            returns['fpfs_thermal'] =[]
            if np.isnan(atm.planet.radius): 
                returns['fpfs_thermal'] += ['Planet Radius not supplied. If you want fpfs, add it to `gravity` function with radius.']
            if np.isnan(radius_star): 
                returns['fpfs_thermal'] += ['Stellar Radius not supplied. If you want fpfs, add it to `stellar` function.']


    #return total if users have calculated both thermal and reflected 
    if (('fpfs_reflected' in list(returns.keys())) & ('fpfs_thermal' in list(returns.keys()))): 
        if ((not isinstance(returns['fpfs_reflected'],list)) & (not isinstance(returns['fpfs_thermal'],list))) :
            returns['fpfs_total'] = returns['fpfs_thermal'] + returns['fpfs_reflected']

    if full_output: 
        if as_dict:
            returns['full_output'] = atm.as_dict()
            if radius_star != 'nostar':returns['full_output']['star']['flux'] = opacityclass.unshifted_stellar_spec
        else:
            returns['full_output'] = atm

    #if output_dir != None:
    #    filename = output_dir
    #    pk.dump({'pressure': atm.level['pressure'], 'temperature': atm.level['temperature'], 
    #        'nlevel':nlevel, 'wno':wno, 'nwno':nwno, 'ng':ng, 'nt':nt, 
    #        'dtau':DTAU, 'tau':TAU, 'w0':W0, 'cosb':COSB, 'gcos2':GCOS2,'ftcld':ftau_cld,'ftray': ftau_ray,
    #        'dtau_og':DTAU_OG, 'tau_og':TAU_OG, 'w0_og':W0_OG, 'cosb_og':COSB_OG, 
    #        'surf_reflect':atm.surf_reflect, 'ubar0':ubar0, 'ubar1':ubar1, 'costheta':cos_theta, 'F0PI':F0PI, 
    #        'single_phase':single_phase, 'multi_phase':multi_phase, 
    #        'frac_a':frac_a, 'frac_b':frac_b, 'frac_c':frac_c, 'constant_back':constant_back, 
    #        'constant_forward':constant_forward, 'dim':dimension, 'stream':stream,
    #        #'xint_at_top': xint_at_top, 'albedo': albedo, 'flux': flux_out, 'xint': intensity,
    #        'b_top': b_top, 'gweight': gweight, 'tweight': tweight, 'gangle': gangle, 'tangle': tangle}, 
    #        open(filename,'wb'), protocol=2)
    return returns

def _finditem(obj, key):
    if key in obj: return obj[key]
    for k, v in obj.items():
        if isinstance(v,dict):
            item = _finditem(v, key)
            if item is not None:
                return item
def standard_metadata(): 
    return {
    'author':'optional',
    'contact':'optional', 
    'code':'optional', 
    'doi':'optional', 
    'planet_params':{
        'rp':'usually taken from picaso',
        'mp':'usually taken from picaso',
        'mh':'optional',
        'cto':'optional',
        'heat_redis':'optional',
        'p_reference':'usually taken from picaso',
        'tint':'optional'
    },
    'stellar_params':{
         'logg':'usually taken from picaso', 
         'feh': 'usually taken from picaso', 
         'steff':'usually taken from picaso', 
         'rs': 'usually taken from picaso', 
         'ms': 'optional', 
         'database':'usually taken from picaso',
    },
    'orbit_params':{
        'sma':'usually taken from picaso', 
    }
}
def check_units(unit):
    try: 
        return u.Unit(unit)
    except ValueError: 
        #check if real unit
        return None
    
def output_xarray(df, picaso_class, add_output={}, savefile=None): 
    """
    This function converts all picaso output to xarray which is easy to save 
    It is recommended for reuse and reproducibility of models. the returned 
    xarray can be used with input_xarray to rerun models. See Preservation notebook. 
    
    Parameters
    ----------
    df : dict 
        This is the output of your spectrum and must include "full_output=True"
        For example, df = case.spectrum(opa, full_output=True)
        It can also be the output of your climate run. For example, out=case.climate(..., with_spec=True)
        If running climate you must run with_spec=True OR add output of case.spectrum to your output dictionary
        by doing (for example): 
        >> out = case.climate(..., withSpec=False)
        >> #add any post processing steps you desire (e.g. case.atmosphere or case.clouds)
        >> df = case.spectrum(opa, calculation='thermal')
        >> out['spectrum_output'] = df
    picaso_class : picaso.inputs
        This is the original class that you made to do your run. 
        For example, case=jdi.inputs();followed by case.atmosphere(), case.clouds(), etc
    add_output : dict
        These are any additional outputs you want included in your xarray metadata 
        This dictionary has a very specific struture if you want it to work correctly. 
        To see the structure you can run justdoit.standard_metadata() which will 
        return an empty dictionary that you can fill. 
    savefile : str 
        Optional, if string it will save a xarray file for you 
    
    Returns
    -------
    xarray.Dataset
        this xarray dataset can be easily passed to justdoit.input_xarray to run a spectrum 
    
    Todo
    ----
    - figure out why pandas index are being returned for pressure and wavelenth 
    - fix clouds wavenumber_layer which doesnt seem like it would work 
    - add clima inputs : teff (or tint), number of convective zones
    - cvs_locs array should be more clear
    """ 
    attrs = {}
    if not isinstance(_finditem(df, 'full_output'), type(None)): 
        #print("Found full_output!")
        full_output = _finditem(df, 'full_output')
        molecules_included = full_output['weights']
    else: 
        raise Exception("full_output is required. Either you need to run spectrum(opa, full_output=True), climate(..., with_spec=True), or create and add info to spectrum_output")

    if not isinstance(_finditem(df, 'wavenumber'), type(None)):
        wavenumber = _finditem(df, 'wavenumber')
    else: 
        raise Exception("wavenumber is required to be in df somewhere")
        
        
    #if they put in climate data, add it to xarray and create meta data
    if not isinstance(_finditem(df, 'dtdp'), type(None)): 
        attrs['climate_params'] = {}
    
    #is df_atmo ran with climate calculations or hi-res spectrum
    if not isinstance(_finditem(df, 'ptchem_df'), type(None)):
        df_atmo = _finditem(df,'ptchem_df')
    else:
        df_atmo = picaso_class.inputs['atmosphere']['profile']

    #start with simple layer T
    data_vars=dict(temperature = (["pressure"], 
                                  df_atmo['temperature'],
                                  {'units': 'Kelvin'}))

    #add climate data
    if not isinstance(_finditem(df, 'dtdp'), type(None)): 
        dtdp = _finditem(df,'dtdp')
        data_vars['dtdp'] = (["pressure_layer"], dtdp, {'units': 'K/bar'})
   
    if not isinstance(_finditem(df, 'cvz_locs'), type(None)): #for metadata (converged) and other(all_profiles/nlevel)
        cvz_locs = _finditem(df, 'cvz_locs')
        attrs['climate_params']['cvs_locs'] = cvz_locs
        
    if not isinstance(_finditem(df, 'converged'), type(None)):
        converged = _finditem(df,'converged')
        attrs['climate_params']['converged'] = converged
        
    if not isinstance(_finditem(df , 'all_profiles'), type(None)):
        all_profiles = _finditem(df , 'all_profiles')
        nlevel = len(df_atmo['pressure'])
        for i in range((int(len(all_profiles)/nlevel))): 
            #after each nlevel amount (ex: 91), restart guess count
            index_start=nlevel*i
            index_finish=nlevel*(i+1)
            data_vars['guess '+str(i+1)]= (["pressure"], all_profiles[index_start:index_finish],{'units': 'Kelvin'})
            
    #spectral data 
    if not isinstance(_finditem(df, 'thermal'), type(None)):
        thermal = _finditem(df,'thermal')
        data_vars['flux_emission'] = (["wavelength"], thermal,{'units': 'erg/cm**2/s/cm'}) 
    if not isinstance(_finditem(df, 'transit_depth'), type(None)):
        transit_depth = _finditem(df, 'transit_depth')
        data_vars['transit_depth'] = (["wavelength"], transit_depth,{'units': 'R_jup**2/R_jup**2'}) 
    if not isinstance(_finditem(df, 'temp_brightness'), type(None)): 
        temp_brightness = _finditem(df, 'temp_brightness')
        data_vars['temp_brightness'] = (["wavelength"], temp_brightness,{'units': 'Kelvin'})
    if not isinstance(_finditem(df, 'fpfs_thermal'), type(None)):
        fpfs_thermal= _finditem(df, 'fpfs_thermal')
        if isinstance(fpfs_thermal, np.ndarray):
            data_vars['fpfs_emission'] = (["wavelength"], fpfs_thermal,{'units': 'erg/cm**2/s/cm/(erg/cm**2/s/cm)'})
    if not isinstance(_finditem(df, 'albedo'), type(None)): 
        albedo=_finditem(df, 'albedo')
        data_vars['albedo'] = (["wavelength"], albedo,{'units': 'none'})
    if not isinstance(_finditem(df, 'fpfs_reflected'), type(None)): 
        fpfs_reflected=_finditem(df, 'fpfs_reflected')
        if isinstance(fpfs_reflected, np.ndarray): 
            data_vars['fpfs_reflected'] = (["wavelength"], fpfs_reflected,{'units': 'erg/cm**2/s/cm/(erg/cm**2/s/cm)'})


    #atmospheric data data 
    for ikey in molecules_included:
        data_vars[ikey] = (["pressure"], df_atmo[ikey].values,{'units': 'v/v'})
        
    if 'kz' in df_atmo: 
        data_vars['kzz'] = (["pressure"], df_atmo['kz'].values,{'units': 'cm**2/s'})
      
    #clouds if they exist 
    if 'clouds' in picaso_class.inputs: 
        if not isinstance(picaso_class.inputs['clouds']['profile'],type(None)):
            for ikey,lbl in zip( ['opd', 'w0', 'g0'], ['opd','ssa','asy']):
                array = np.reshape(picaso_class.inputs['clouds']['profile'][ikey].values, 
                       (picaso_class.nlevel-1, 
                        len(picaso_class.inputs['clouds']['wavenumber'])))

                data_vars[lbl]=(['pressure_layer','wavenumber_layer'],array,{'units': 'unitless'})
    
    #basic info
    for ikey in ['author','code','doi','contact']:
        if add_output.get(ikey,'optional') != 'optional':
            attrs[ikey] = add_output[ikey]
            
    #planet params 
    planet_params = add_output.get('planet_params',{})
    attrs['planet_params'] = {}
    
    if not isinstance(_finditem(df, 'effective_temperature'), type(None)):
        effective_temp = _finditem(df, 'effective_temperature')
        attrs['planet_params']['effective_temp'] = effective_temp

    #find gravity in picaso
    gravity = picaso_class.inputs['planet'].get('gravity',np.nan)
    if np.isfinite(gravity): 
        gravity = gravity * check_units(picaso_class.inputs['planet']['gravity_unit'])
    #otherwise find gravity from user input
    else: 
        gravity = planet_params.get('gravity', np.nan) 
    
    mp = picaso_class.inputs['planet'].get('mass',np.nan)
    if np.isfinite(mp):
        mp = mp * check_units(picaso_class.inputs['planet']['mass_unit'])
    else: 
        mp = planet_params.get('mp',np.nan) 
        
    rp = picaso_class.inputs['planet'].get('radius',np.nan)
    if np.isfinite(rp):
        rp = rp * check_units(picaso_class.inputs['planet']['radius_unit'])
    else: 
        rp = planet_params.get('rp',np.nan) 
        
    #add required RP/MP or gravity
    if (not np.isnan(gravity)): 
        attrs['planet_params']['gravity'] = gravity
        assert isinstance(attrs['planet_params']['gravity'],u.quantity.Quantity ), "User supplied gravity in planet_params must be an astropy unit: e.g. 1*u.Unit('m/s**2')"
    elif (((not np.isnan(mp)) & (not np.isnan(rp))) & (((not isinstance(mp,str)) & (not isinstance(rp,str))))):
        attrs['planet_params']['mp'] = mp
        attrs['planet_params']['rp'] = rp
        assert isinstance(attrs['planet_params']['mp'],u.quantity.Quantity ), "User supplied mp in planet_params must be an astropy unit: e.g. 1*u.Unit('M_jup')"
        assert isinstance(attrs['planet_params']['rp'],u.quantity.Quantity ), "User supplied rp in planet_params must be an astropy unit: e.g. 1*u.Unit('R_jup')"
    else: 
        print('Mass and Radius, or gravity not provided in add_output, and wasn not found in picaso class')
    
    #add anything else the user had in planet params
    for ikey in planet_params.keys(): 
        if ikey not in ['rp','mp','logg','gravity']:
            if planet_params[ikey]!='optional':attrs['planet_params'][ikey] = planet_params[ikey]

    #find gravity in picaso
    p_reference = picaso_class.inputs['approx'].get('p_reference',np.nan)
    if np.isfinite(p_reference): 
        p_reference = p_reference * u.Unit('bar')
    #otherwise find gravity from user input
    else: 
        p_reference = planet_params.get('p_reference', np.nan) 
    if not np.isnan(p_reference): attrs['planet_params']['p_reference'] = p_reference
        

    attrs['planet_params'] = json.dumps(attrs['planet_params'],cls=JsonCustomEncoder)


    if 'nostar' not in picaso_class.inputs['star']['database']:
        #stellar params 
        stellar_params = add_output.get('stellar_params',{})
        attrs['stellar_params'] = {}
        #must be supplied in picaso
        attrs['stellar_params']['database'] = stellar_params.get('database', picaso_class.inputs['star'].get('database',None)) 
        attrs['stellar_params']['steff'] = stellar_params.get('steff', picaso_class.inputs['star'].get('temp',None)) 
        attrs['stellar_params']['feh'] = stellar_params.get('feh', picaso_class.inputs['star'].get('metal',None)) 
        attrs['stellar_params']['logg'] = stellar_params.get('logg', picaso_class.inputs['star'].get('logg',None)) 
        #optional could be nan
        rs = picaso_class.inputs['star'].get('radius',np.nan)
        if np.isfinite(rs):
            rs = rs * check_units(picaso_class.inputs['star']['radius_unit'])
        else: 
            rs = stellar_params.get('rs',np.nan)
            
        if not np.isnan(rs):
            attrs['stellar_params']['rs'] = rs
            assert isinstance(attrs['stellar_params']['rs'],u.quantity.Quantity ), "User supplied rs in stellar_params must be an astropy unit: e.g. 1*u.Unit('R_sun')"
        
        #perform stellar params checks
        for ikey in attrs['stellar_params'].keys():
            assert not isinstance(attrs['stellar_params'][ikey],type(None)), f"We couldnt find these stellar parameters in add_output or the picaso class {attrs['stellar_params']}" 
        for ikey in stellar_params.keys(): 
            if ikey not in ['database','steff','feh','logg','rs']:
                if stellar_params[ikey]!='optional':attrs['stellar_params'][ikey] = stellar_params[ikey]
                
        #orbit params
        orbit_params = add_output.get('orbit_params',{})
        sma = picaso_class.inputs['star'].get('semi_major',np.nan)
        if np.isfinite(sma):
            sma = sma * check_units(picaso_class.inputs['star']['semi_major_unit'])
        else: 
            sma = orbit_params.get('sma',np.nan)        
        
        if not np.isnan(sma): 
            attrs['orbit_params'] = {}
            attrs['orbit_params']['sma'] = sma
            assert isinstance(attrs['orbit_params']['sma'],u.quantity.Quantity ), "User supplied rs in orbit_params must be an astropy unit: e.g. 1*u.Unit('AU')"
            
            for ikey in orbit_params.keys(): 
                if ikey not in ['sma']:
                    if orbit_params[ikey]!='optional':attrs['orbit_params'][ikey] = orbit_params[ikey] 
                    
            attrs['orbit_params'] = json.dumps(attrs['orbit_params'],cls=JsonCustomEncoder)
            

               
        if 'stellar_params' in attrs.keys(): attrs['stellar_params'] = json.dumps(attrs['stellar_params'],cls=JsonCustomEncoder)
        if 'climate_params' in attrs.keys(): attrs['climate_params'] = json.dumps(attrs['climate_params'],cls=JsonCustomEncoder)
        
        
    #add anything else requested by the user
    for ikey in add_output.keys(): 
        if ikey not in attrs.keys(): 
            if add_output[ikey]!="optional":attrs[ikey] = add_output[ikey]
    
    
    coords=dict(
            pressure=(["pressure"], np.array(df_atmo['pressure'].values),{'units': 'bar'}),#required*
            wavelength=(["wavelength"], np.array(1e4/wavenumber),{'units': 'micron'})
        )
    if 'clouds' in 'opd' in data_vars.keys(): 
        coords['wavenumber_layer'] = (["wavenumber_layer"], picaso_class.inputs['clouds']   ,{'units': 'cm**(-1)'})
        coords['pressure_layer'] = (["pressure_layer"], full_output['layer']['pressure'] ,{'units': full_output['layer']['pressure_unit']})
    if 'dtdp' in data_vars.keys():
        coords['pressure_layer'] = (["pressure_layer"], full_output['layer']['pressure'] ,{'units': full_output['layer']['pressure_unit']})

    # put data into a dataset where each
    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=attrs
    )
    
    if isinstance(savefile, str): ds.to_netcdf(savefile)

    return ds

def input_xarray(xr_usr, opacity,calculation='planet',approx_kwargs={}):
    """
    This takes an input based on these standards and runs: 
    -gravity
    -phase_angle
    -star
    -approx (p_reference=10)
    -atmosphere
    -clouds (if there are any)

    Parameters
    ----------
    xr_usr : xarray
        xarray based on ERS formatting requirements 
    opacity : justdoit.opannection
        opacity connection
    approx_kwargs : dict 
        any key words to pass to the approx class 
    calculation : str 
        'planet' or 'browndwarf'

    Example
    -------
    case = jdi.input_xarray(xr_user)
    case.spectrum(opacity,calculation='transit_depth')
    """
    planet_params = eval(xr_usr.attrs['planet_params'])

    case = inputs(calculation = calculation)
    case.phase_angle(0) #radians

    p_reference_xarray = _finditem(planet_params,'p_reference')
    p_reference = approx_kwargs.get('p_reference',10)

    if (not isinstance(p_reference_xarray, type(None))): 
        p_bar = p_reference_xarray['value']*u.Unit(p_reference_xarray['unit'])
        p_bar = p_bar.to('bar').value
        approx_kwargs['p_reference']=p_bar
    elif (not isinstance(p_reference, type(None))): 
        #is it common to want to change the reference pressure
        approx_kwargs['p_reference']=p_reference
        
    else: 
        raise Exception("p_reference couldnt be found in the xarray, nor was it supplied to this function inputs. Please rerun function with p_reference=10 (or another number in bars).")

    case.approx(**approx_kwargs)

    #define gravity
    
    if 'brown' not in calculation:
        stellar_params = eval(xr_usr.attrs['stellar_params'])
        orbit_params = eval(xr_usr.attrs['orbit_params'])
        steff = _finditem(stellar_params,'steff')
        feh = _finditem(stellar_params,'feh')
        logg = _finditem(stellar_params,'logg')
        database = 'phoenix' if type(_finditem(stellar_params,'database')) == type(None) else _finditem(stellar_params,'database')
        ms = _finditem(stellar_params,'ms')
        rs = _finditem(stellar_params,'rs')
        semi_major = _finditem(orbit_params,'sma')
        case.star(opacity, steff,feh,logg, radius=rs['value'], 
                  radius_unit=u.Unit(rs['unit']), database=database, 
                  semi_major=semi_major['value'],semi_major_unit=u.Unit(semi_major['unit']))

    mp = _finditem(planet_params,'mp')
    rp = _finditem(planet_params,'rp')
    gravity = _finditem(planet_params,'gravity')
    logg = _finditem(planet_params,'logg')
    gravity = _finditem(planet_params,'gravity')

    if ((not isinstance(mp, type(None))) & (not isinstance(rp, type(None)))):
        case.gravity(mass = mp['value'], mass_unit=u.Unit(mp['unit']),
                    radius=rp['value'], radius_unit=u.Unit(rp['unit']))
    elif (not isinstance(gravity, type(None))): 
        case.gravity(gravity = gravity['value'], gravity_unit=u.Unit(gravity['unit']))    
    elif (not isinstance(logg, type(None))): 
        case.gravity(gravity = 10**logg['value'], gravity_unit=u.Unit(logg['unit']))
    elif (not isinstance(gravity, type(None))): 
        case.gravity(gravity = gravity['value'], gravity_unit=u.Unit(gravity['unit']))

    else: 
        print('Mass and Radius or gravity not provided in xarray, user needs to run gravity function')


    df = {'pressure':xr_usr.coords['pressure'].values}
    for i in [i for i in xr_usr.data_vars.keys() if 'transit' not in i]:
        if i not in ['opd','ssa','asy']:
            #only get single coord pressure stuff
            if (len(xr_usr.data_vars[i].values.shape)==1 &
                        ('pressure' in xr_usr.data_vars[i].coords)):
                df[i]=xr_usr.data_vars[i].values
        
    case.atmosphere(df=pd.DataFrame(df))

    if 'opd' in xr_usr.data_vars.keys():
        df_cld = vj.picaso_format(xr_usr['opd'].values, 
                xr_usr['ssa'].values, 
                xr_usr['asy'].values)

        case.clouds(df=df_cld)

    return case

def get_contribution(bundle, opacityclass, at_tau=1, dimension='1d'):
    """
    Currently top level program to run albedo code 

    Parameters 
    ----------
    bundle : dict 
        This input dict is built by loading the input = `justdoit.load_inputs()` 
    opacityclass : class
        Opacity class from `justdoit.opannection`
    at_tau : float 
        (Optional) Default = 1, This is to compute the pressure level at which cumulative opacity reaches 
        at_tau. Usually users want to know when the cumulative opacity reaches a tau of 1. 
    dimension : str 
        (Optional) Default = '1d'. Currently only 1d is supported. 

    Return
    ------
    taus_by_species : dict
        Each dictionary entry is a nlayer x nwave that represents the 
        per layer optical depth for that molecule. If you do not see 
        a molecule that you have added as input, check to make sure it is
        propertly formatted (e.g. Sodium must be Na not NA, Titanium Oxide must be TiO not TIO)
    cumsum_taus : dict
        Each dictionary entry is a nlevel x nwave that represents the cumulative summed opacity
        for that molecule. If you do not see a molecule that you have added as input, check to make sure it is
        propertly formatted (e.g. Sodium must be Na not NA, Titanium Oxide must be TiO not TIO)
    at_pressure_array : dict 
        Each dictionary entry is a nwave array that represents the 
        pressure level where the cumulative opacity reaches the value specieid by the user through `at_tau`.
        If you do not see a molecule that you have added as input, check to make sure it is
        a molecule that you have added as input, check to make sure it is
        propertly formatted (e.g. Sodium must be Na not NA, Titanium Oxide must be TiO not TIO)
        
    """
    inputs = bundle.inputs

    wno = opacityclass.wno
    nwno = opacityclass.nwno
    ngauss = opacityclass.ngauss
    gauss_wts = opacityclass.gauss_wts #for opacity

    #check to see if we are running in test mode
    test_mode = inputs['test_mode']

    ############# DEFINE ALL APPROXIMATIONS USED IN CALCULATION #############
    #see class `inputs` attribute `approx`
    #what rt method are we using?? 
    rt_method = inputs['approx']['rt_method'] #either toon or spherical harmonics
    
    #USED by all RT
    stream = inputs['approx']['rt_params']['common']['stream']
    #parameters needed for the two term hg phase function. 
    #Defaults are set in config.json
    f = inputs['approx']['rt_params']['common']['TTHG_params']['fraction']
    frac_a = f[0]
    frac_b = f[1]
    frac_c = f[2]
    constant_back = inputs['approx']['rt_params']['common']['TTHG_params']['constant_back']
    constant_forward = inputs['approx']['rt_params']['common']['TTHG_params']['constant_forward']
    raman_approx =inputs['approx']['rt_params']['common']['raman']
    #define delta eddington approximinations 
    delta_eddington = inputs['approx']['rt_params']['common']['delta_eddington']


    #USED in TOON (if being used)
    single_phase = inputs['approx']['rt_params']['toon']['single_phase']
    toon_coefficients = inputs['approx']['rt_params']['toon']['toon_coefficients']
    multi_phase = inputs['approx']['rt_params']['toon']['multi_phase']

    #USED in SH (if being used)
    single_form = inputs['approx']['rt_params']['SH']['single_form']
    w_single_form = inputs['approx']['rt_params']['SH']['w_single_form']
    w_multi_form = inputs['approx']['rt_params']['SH']['w_multi_form']
    psingle_form = inputs['approx']['rt_params']['SH']['psingle_form']
    w_single_rayleigh = inputs['approx']['rt_params']['SH']['w_single_rayleigh']
    w_multi_rayleigh = inputs['approx']['rt_params']['SH']['w_multi_rayleigh']
    psingle_rayleigh = inputs['approx']['rt_params']['SH']['psingle_rayleigh']
    calculate_fluxes = inputs['approx']['rt_params']['SH']['calculate_fluxes']


    #for patchy clouds
    do_holes = inputs['clouds'].get('do_holes',False)
    if do_holes == True:
        print('in justdoit picaso turning on patchy clouds')
        fhole = inputs['clouds']['fhole']
        fthin_cld = inputs['clouds']['fthin_cld']

    #save level fluxes in addition to the top of atmosphere fluxes?
    #default is false
    get_lvl_flux = inputs['approx'].get('get_lvl_flux',False)


    #pressure assumption
    p_reference =  inputs['approx']['p_reference']

    ############# DEFINE ALL GEOMETRY USED IN CALCULATION #############
    #see class `inputs` attribute `phase_angle`
    

    #phase angle 
    phase_angle = inputs['phase_angle']
    #get geometry
    geom = inputs['disco']

    ng, nt = geom['num_gangle'], geom['num_tangle']
    gangle,gweight,tangle,tweight = geom['gangle'], geom['gweight'],geom['tangle'], geom['tweight']
    lat, lon = geom['latitude'], geom['longitude']
    cos_theta = geom['cos_theta']
    ubar0, ubar1 = geom['ubar0'], geom['ubar1']
   # ubar1 = np.array([[-.99999],[-0.5],[0.5],[1.]])
   # ubar0 = np.array([[1/np.sqrt(2)],[1/np.sqrt(2)],[1/np.sqrt(2)],[1/np.sqrt(2)]])
   # ng = 4; nt = 1

    #set star parameters
    radius_star = inputs['star']['radius']

    #need to account for case where there is no star
    if 'nostar' in inputs['star']['database']:
        F0PI = np.zeros(opacityclass.nwno) + 1.0
    else:
        F0PI = opacityclass.relative_flux

    b_top = 0.
    #semi major axis
    sa = inputs['star']['semi_major']

    #begin atm setup
    atm = ATMSETUP(inputs)


    #Add inputs to class 
    atm.surf_reflect = inputs['surface_reflect']
    atm.hard_surface = inputs['hard_surface']#0=no hard surface, 1=hard surface
    atm.wavenumber = wno
    atm.planet.gravity = inputs['planet']['gravity']
    atm.planet.radius = inputs['planet']['radius']
    atm.planet.mass = inputs['planet']['mass']
    atm.get_lvl_flux=get_lvl_flux

    if dimension == '1d':
        atm.get_profile()
    elif dimension == '3d':
        atm.get_profile_3d()

    #now can get these 
    atm.get_mmw()
    atm.get_density()
    atm.get_altitude(p_reference = p_reference)#will calculate altitude if r and m are given (opposed to just g)
    atm.get_column_density()

    #gets both continuum and needed rayleigh cross sections 
    #relies on continuum molecules are added into the opacity 
    #database. Rayleigh molecules are all in `rayleigh.py` 
    atm.get_needed_continuum(opacityclass.rayleigh_molecules,
                             opacityclass.avail_continuum)

    #get cloud properties, if there are any and put it on current grid 
    atm.get_clouds(wno)

    #Make sure that all molecules are in opacityclass. If not, remove them and add warning
    no_opacities = [i for i in atm.molecules if i not in opacityclass.molecules]
    atm.add_warnings('No computed opacities for: '+','.join(no_opacities))
    atm.molecules = np.array([ x for x in atm.molecules if x not in no_opacities ])
    
    #opacity assumptions
    exclude_mol = inputs['atmosphere']['exclude_mol']

    get_opacities = opacityclass.get_opacities

    nlevel = atm.c.nlevel
    nlayer = atm.c.nlayer
    
    #lastly grab needed opacities for the problem
    get_opacities(atm)
    #only need to get opacities for one pt profile

    #There are two sets of dtau,tau,w0,g in the event that the user chooses to use delta-eddington
    #We use HG function for single scattering which gets the forward scattering/back scattering peaks 
    #well. We only really want to use delta-edd for multi scattering legendre polynomials. 
    taus_by_species= compute_opacity(atm, opacityclass, ngauss=ngauss,  stream=stream,raman=raman_approx, return_mode=True)

    cumsum_taus = {}
    for i in taus_by_species.keys(): 
        shape = taus_by_species[i].shape
        taugas = np.zeros((shape[0]+1, shape[1]))
        taugas[1:,:]=numba_cumsum(taus_by_species[i])
        cumsum_taus[i] = taugas

    pressure = atm.level['pressure']/atm.c.pconv

    at_pressure_array = {}
    for i in taus_by_species.keys(): 
        at_pressure_array[i] = find_press(at_tau, cumsum_taus[i], shape[1], pressure)

    return {'taus_per_layer':taus_by_species, 
            'cumsum_taus':cumsum_taus, 
            'tau_p_surface':at_pressure_array}

@njit()
def find_press(at_tau, a, b, c):
    at_press = []
    for iw in range(b): 
        at_press.append(np.interp([at_tau],a[:,iw],c)[0])
    return at_press

def opannection(wave_range = None, filename_db = None, 
                resample=1, method='resampled',
                ck_db=None, raman_db = None, 
                preload_gases='all',
                #deq= False, on_fly=False,
                #gases_fly =None,ck=False,
                verbose=True):
    """
    Sets up database connection to opacities. 

    Parameters
    ----------
    wave_range : list of float 
        Subset of wavelength range for which to run models for 
        Default : None, which pulls entire grid 
    filename_db : str 
        Filename of opacity database to query from 
        Default is none which pulls opacity file that 
        comes with distribution 
    raman_db : str 
        Filename of raman opacity cross section 
        Default is none which pulls opacity file that comes with distribution 
    resample : int 
        Default=1 (no resampling) PROCEED WITH CAUTION!!!!!This will resample your opacites. 
        This effectively takes opacity[::BINS] depending on what the 
        sampling requested is. Consult your local theorist before 
        using this. 
    method : str 
        By default method='resampled'
        Other options include: ['preweighted','resortrebin']
    ck_db : str 
        Can be: 
        - (required if method is preweighted) ASCII dir of ck file
        - (required if method is preweighted) HDF5 filename 
        - (optional) path to HDF5 directory, if none specified then assumed default in climate_INPUTS/ktable_by_molecule folder
    preload_gases : str
        Gases that you want to have mixed on the fly, you can specify them here. Default is 'all'
    verbose : bool 
        Error message to warn users about resampling. Can be turned off by supplying 
        verbose=False
    """
    inputs = json.load(open(os.path.join(__refdata__,'config.json')))

    if ((method == 'resampled') & isinstance(ck_db,type(None))):
        #set raman database 
        if isinstance(raman_db,type(None)): raman_db = os.path.join(__refdata__, inputs['opacities']['files']['raman'])
        
        #get default file if it was supplied
        if isinstance(filename_db,type(None)): 
            filename_db = os.path.join(__refdata__, inputs['opacities']['files']['opacity'])
            if not os.path.isfile(filename_db):
                raise Exception(f'The default opacity file does not exist: {filename_db}. In order to have a default database please download one of the opacity files from Zenodo and place into this folder with the name opacities.db: https://zenodo.org/record/6928501#.Y2w4C-zMI8Y if you dont want a single default file then you just need to point to the opacity db using the keyword filename_db.')
        #if a name is supplied check that it exists 
        elif not isinstance(filename_db,type(None) ): 
            if not os.path.isfile(filename_db):
                raise Exception('The opacity file you have entered does not exist: '  + filename_db)

        #if resampling was entered just warn users
        if resample != 1:
            if verbose:print("YOU ARE REQUESTING RESAMPLING!! This could degrade the precision of your spectral calculations so should be used with caution. If you are unsure check out this tutorial: https://natashabatalha.github.io/picaso/notebooks/10_ResamplingOpacities.html")

        opacityclass=RetrieveOpacities(
                    filename_db, 
                    raman_db,
                    wave_range = wave_range, resample = resample)   
    elif ((method == 'resampled') & isinstance(ck_db,str)):
        raise Exception("ck_db was supplied but method is set to resampled. Change kwarg method='preweighted' to use the preweighted ck tables")
    elif method == 'preweighted':

        #get default continuum if nothing was specified
        if isinstance(filename_db,type(None)): 
            filename_db = os.path.join(__refdata__, inputs['opacities']['files']['ktable_continuum'])
        
        if not os.path.exists(ck_db):
            if ck_db[-1] == '/':ck_db = ck_db[0:-1]
            if os.path.isfile(ck_db+'.tar.gz'): 
                raise Exception('The CK filename that you have selected appears still be .tar.gz. Please unpack and rerun')
            else: 
                raise Exception('The CK filename that you have selected does not exist. Please make sure you have downloaded and unpacked the right CK file.')
            
        opacityclass=RetrieveCKs(
                    ck_db, 
                    filename_db, 
                    method='preweighted')

    elif method == 'resortrebin':
        #get default continuum if nothing was specified
        if isinstance(filename_db,type(None)): 
            #NEB TODO: we really ony need one continuum file, the rest can be interpolated down to the 196 grid. 
            filename_db = os.path.join(__refdata__, inputs['opacities']['files']['ktable_resortrebin'])

        #now we can load the ktables for those gases defined
        if isinstance(ck_db ,type(None) ):
            ck_db  = os.path.join(__refdata__,  inputs['opacities']['files']['ktable_by_molecule'])

        #if preload gases is 'all' then get everything in the ck_db dir    
        if isinstance(preload_gases,str):
            if preload_gases=='all':
                check_hdf5=glob.glob(os.path.join(ck_db ,'*.hdf5'))
                check_npy=glob.glob(os.path.join(ck_db,'*.npy'))
                if len(check_hdf5)>0:
                    preload_gases = [i.split('/')[-1].split('_')[0] for i in check_hdf5]
                elif len(check_npy)>0:
                    preload_gases = [i.split('/')[-1].split('_')[0] for i in check_npy]
                else:
                    raise Exception(f'No .npy or .hdf5 molecule files were found in {ck_db}')
            else: 
                preload_gases = [preload_gases]
        opacityclass=RetrieveCKs(
                    ck_db, 
                    filename_db, 
                    method='resortrebin',
                    preload_gases=preload_gases)
    else: 
        raise Exception("The only available opacity methods are: resortrebin, preweighted, and resampled")
    return opacityclass

class inputs():
    """Class to setup planet to run

    Parameters
    ----------
    calculation: str 
        (Optional) Controls planet or brown dwarf calculation. Default = 'planet'. Other option is "browndwarf".
    climate : bool 
        (Optional) If true, this do a thorough iterative calculation to compute 
        a temperature pressure profile.


    Attributes
    ----------
    phase_angle() : set phase angle
    gravity() : set gravity
    star() : set stellar spec
    atmosphere() : set atmosphere composition and PT
    clouds() : set cloud profile
    approx()  : set approximation
    spectrum() : create spectrum
    """
    def __init__(self, calculation='planet', climate=False):

        self.inputs = json.load(open(os.path.join(__refdata__,'config.json')))
        
        if 'brown' in calculation:
            self.setup_nostar()
        
        if climate: 
            self.setup_climate()

    def phase_angle(self, phase=0,num_gangle=10, num_tangle=1,symmetry=False, 
        phase_grid=None, calculation=None):
        """Define phase angle and number of gauss and tchebychev angles to compute. 
        Controls all geometry of the calculation. Computes latitude and longitudes. 
        Computes cos theta and incoming and outgoing angles. Adds everything to class. 

        Please see geometry notebook for a deep dive into how these angles are incorporated 
        into the calculation.

        Typical numbers:
        - 1D Thermal: num_gangle = 10, num_tangle=1
        - 1D Reflected light with zero phase : num_gangle = 10, num_tangle=1
        - 1D Reflected light with non zero phase : num_gangle = 8, num_tangle=8
        - 3D Thermal or Reflected : angles will depend on 3D features interested in resolving.
        
        Parameters
        ----------
        phase : float,int
            Phase angle in radians 
        num_gangle : int 
            Number of Gauss angles to integrate over facets (Default is 10). This 
            is defined as angles over the full sphere. So 10 as the default compute the RT 
            with 5 points on the west hemisphere and 5 points on the west. 
            Higher numbers will slow down code. 
        num_tangle : int 
            Number of Tchebyshev angles to integrate over facets (Default is 1, which automatically 
            assumes symmetry). 
            Must be even if considering symmetry. 
        symmetry : bool 
            Default is False. Note, if num_tangle=1 you are automatically considering symmetry.
            This will only compute the unique incoming angles so that the calculation 
            isn't doing repetetive models. E.g. if num_tangles=10, num_gangles=10. Instead of running a 10x10=100
            FT calculations, it will only run 5x5 = 25 (or a quarter of the sphere).
        phase_grid : array 
            This is ONLY for computing phase curves and ONLY can be used 
        calculation : str 
            Used for phase curve calculation. Needs to be specified in order 
            to determine integrable angles correctly.
        """
        #phase curve requested?? 
        if not isinstance(phase_grid,type(None)):
            if isinstance(calculation, type(None)):
                raise Exception("Phase curve calculation activated because phase_grid is supplied. However, 'calculation' needs to be specified to either 'reflected' or 'thermal'")
            self.phase_curve_geometry(calculation, phase_grid, num_gangle=num_gangle, num_tangle=num_tangle)
            return #no need to continue if compute a phase curve

        if (phase > 2*np.pi) or (phase<0): raise Exception('Oops! you input a phase angle greater than 2*pi or less than 0. Please make sure your inputs are in radian units: 0<phase<2pi')
        if ((num_tangle==1) or (num_gangle==1)): 
            #this is here so that we can compare to older models 
            #this model only works for full phase calculations

            if phase!=0: raise Exception("""The default PICASO disk integration 
                is to use num_tangle=1 and num_gangle>1. This method is faster because 
                it makes use of symmetry and only computs one fourth of the sphere. 
                However, it looks like you would like to compute non-zero phase functions. 
                In this case, we can no longer utilize symmetry in our disk integration. Therefore, 
                please resubmit phase_angle with num_tange>10 and num_gangle>10.""")
            if num_gangle==1:  raise Exception("""num_gangle cannot be 1. 
                Please resubmit your run with num_tangle=1, and increase number of Gauss points""")

            num_gangle = int(num_gangle/2) #utilizing symmetry

            #currently only supported values
            possible_values = np.array([5,6,7,8])
            idx = (np.abs(possible_values - num_gangle)).argmin() 
            num_gangle = possible_values[idx]


            gangle, gweight, tangle, tweight = get_angles_1d(num_gangle) 
            ng = len(gangle)
            nt = len(tangle)

            ubar0, ubar1, cos_theta,lat,lon = compute_disco(ng, nt, gangle, tangle, phase)

            geom = {}
            #build dictionary
            geom['num_gangle'], geom['num_tangle'] = ng, nt 
            geom['gangle'], geom['gweight'],geom['tangle'], geom['tweight'] = gangle,gweight,tangle,tweight
            geom['latitude'], geom['longitude']  = lat, lon 
            geom['cos_theta'] = 1.0 
            geom['ubar0'], geom['ubar1'] = ubar0, ubar1 
            self.inputs['phase_angle'] = phase
            self.inputs['disco'] = geom
        else: 

            #this is the new unhardcoded way except 
            self.inputs['phase_angle'] = phase
            
            ng = int(num_gangle)
            nt = int(num_tangle)

            gangle,gweight,tangle,tweight = get_angles_3d(ng, nt) 

            unique_geom = {}
            full_geom = {}

            #planet disk is divided into gaussian and chebyshev angles and weights for perfoming the 
            #intensity as a function of planetary phase angle 
            ubar0, ubar1, cos_theta,lat,lon = compute_disco(ng, nt, gangle, tangle, phase)

            #build dictionary
            full_geom['num_gangle'] = ng
            full_geom['num_tangle'] = nt 
            full_geom['gangle'], full_geom['gweight'],full_geom['tangle'], full_geom['tweight'] = gangle,gweight,tangle,tweight
            full_geom['latitude'], full_geom['longitude']  = lat, lon 
            full_geom['cos_theta'] = cos_theta 
            full_geom['ubar0'], full_geom['ubar1'] = ubar0, ubar1 

            #if running symmetric calculations
            if symmetry:
                if (phase!=0): 
                    raise Exception('If phase is non zero then you cannot utilize symmetry to \
                        reduce computation speed.')
                if (((num_gangle==2) | (num_tangle==2)) and symmetry):
                    raise Exception('Youve selected num_tangle or num_gangle of 2 \
                        however for symmetry to be utilized we need at LEAST two points \
                        on either side of the symmetric axis (e.g. num angles >=4 or 1).\
                        Conducting a help(phase_angle) will tell you \
                        Typical num_tangle and num_gangles to use')
                if ((np.mod(num_tangle,2) !=0 ) and (num_tangle !=1 )):
                    raise Exception('Youve selected an odd num_tangle that isnt 1 \
                        however for symmetry to be utilized we need at LEAST two points \
                        on either side of the symmetric axis (e.g. num angles >=4 or 1).\
                        Conducting a help(phase_angle) will tell you \
                        Typical num_tangle and num_gangles to use')
                if ((np.mod(num_gangle,2) !=0 ) and (num_gangle !=1 )):
                    raise Exception('Youve selected an odd num_gangle that isnt 1 \
                        however for symmetry to be utilized we need at LEAST two points \
                        on either side of the symmetric axis (e.g. num angles >=4 or 1).\
                        Conducting a help(phase_angle) will tell you \
                        Typical num_tangle and num_gangles to use')

                unique_geom['symmetry'] = 'true'
                nt_uni = len(np.unique((tweight*1e6).astype(int))) #unique to 1ppm  
                unique_geom['num_tangle'] = nt_uni
                ng_uni = len(np.unique((gweight*1e6).astype(int)))  #unique to 1ppm 
                unique_geom['num_gangle'] = ng_uni
                unique_geom['ubar1'] = ubar1[0:ng_uni, 0:nt_uni]
                unique_geom['ubar0'] = ubar0[0:ng_uni, 0:nt_uni]
                unique_geom['latitude'] = lat[0:nt_uni]
                unique_geom['longitude'] = lon[0:ng_uni] 
                #adjust weights since we will have to multiply everything by 2 
                adjust_gweight = num_tangle/nt_uni
                adjust_tweight = num_gangle/ng_uni
                unique_geom['gangle'], unique_geom['gweight'] = gangle[0:ng_uni] ,adjust_gweight*gweight[0:ng_uni]
                unique_geom['tangle'], unique_geom['tweight'] = tangle[0:nt_uni],adjust_tweight*tweight[0:nt_uni]
                unique_geom['cos_theta'] = cos_theta 
                #add this to disco
                self.inputs['disco'] = unique_geom
                self.inputs['disco']['full_geometry'] = full_geom
            else: 
                full_geom['symmetry'] = 'false'
                self.inputs['disco'] = full_geom

    def phase_curve_geometry(self, calculation, phase_grid, num_gangle=10, num_tangle=10): 
        """
        Geometry setup for phase curve calculation. This computes all the 
        necessary 

        Parameters 
        ----------
        calculation : str 
            'reflected' or 'thermal'
        phase_grid : float 
            phase angle grid to compute phase curve in radians
        num_gangle : int 
            number of gauss angles, equivalent to longitude 
        num_tangle : int 
            number of tchebychev angles, equivalent to latitude 
        """
        if min(phase_grid)<0:raise Exception('Input minimum of phase grid less than 0. Input phase_grid such that there are only values between 0-2pi')
        if max(phase_grid)>np.pi*2:raise Exception('Input maximum of phase grid is greater than 2pi. Input phase_grid such that there are only values between 0-2pi')
        self.inputs['phase_angle'] = phase_grid
        
        ng = int(num_gangle)
        nt = int(num_tangle)

        gangle,gweight,tangle,tweight = get_angles_3d(ng, nt) 
        def compute_angles(phase):
            out = {}
            #planet disk is divided into gaussian and chebyshev angles and weights for perfoming the 
            #intensity as a function of planetary phase angle 
            ubar0, ubar1, cos_theta,lat,lon = compute_disco(ng, nt, gangle, tangle, phase)

            #build dictionary
            out['num_gangle'] = ng
            out['num_tangle'] = nt 
            out['gangle'], out['gweight'],out['tangle'], out['tweight'] = gangle,gweight,tangle,tweight
            out['latitude'], out['longitude']  = lat, lon 
            out['cos_theta'] = cos_theta 
            out['ubar0'], out['ubar1'] = ubar0, ubar1 
            out['symmetry'] = 'false'
            return out 

        full_geom = {}
        for iphase in self.inputs['phase_angle'] :  
            if calculation == 'thermal': 
                #this may seem odd but for thermal emission, flux 
                #emits at all angles. 
                #so for all phases we need the same integrable geometry
                full_geom[iphase] = compute_angles(0)
            elif calculation == 'reflected': 
                full_geom[iphase] = compute_angles(iphase)
            else: 
                raise Exception('Phase curve setup only works for calculation=thermal or reflected')


        self.inputs['disco'] = full_geom
        self.inputs['disco']['calculation'] = calculation
    
    def gravity(self, gravity=None, gravity_unit=None, 
        radius=None, radius_unit=None, mass = None, mass_unit=None):
        """
        Get gravity based on mass and radius, or gravity inputs 

        Parameters
        ----------
        gravity : float 
            (Optional) Gravity of planet 
        gravity_unit : astropy.unit
            (Optional) Unit of Gravity
        radius : float 
            (Optional) radius of planet MUST be specified for thermal emission!
        radius_unit : astropy.unit
            (Optional) Unit of radius
        mass : float 
            (Optional) mass of planet 
        mass_unit : astropy.unit
            (Optional) Unit of mass 
        """
        if (mass is not None) and (radius is not None):
            m = (mass*mass_unit).to(u.g)
            r = (radius*radius_unit).to(u.cm)
            g = (c.G.cgs * m /  (r**2)).value
            self.inputs['planet']['radius'] = r.value
            self.inputs['planet']['radius_unit'] = 'cm'
            self.inputs['planet']['mass'] = m.value
            self.inputs['planet']['mass_unit'] = 'g'
            self.inputs['planet']['gravity'] = g
            self.inputs['planet']['gravity_unit'] = 'cm/(s**2)'
        elif gravity is not None:
            g = (gravity*gravity_unit).to('cm/(s**2)')
            g = g.value
            self.inputs['planet']['gravity'] = g
            self.inputs['planet']['gravity_unit'] = 'cm/(s**2)'
            self.inputs['planet']['radius'] = np.nan
            self.inputs['planet']['radius_unit'] = 'Radius not specified'
            self.inputs['planet']['mass'] = np.nan
            self.inputs['planet']['mass_unit'] = 'Mass not specified'
        else: 
            raise Exception('Need to specify gravity or radius and mass + additional units')
    
    def setup_climate(self):
        """
        Turns off planet specific things, so program can run as usual
        """
        self.inputs['calculation'] ='climate'

        self.inputs['approx']['rt_params']['common']['raman'] = 2 #turning off raman scattering
        #auto turn on zero phase for now there is no use giving users a choice in disk gauss angle 
        self.phase_angle(0,num_gangle=10,num_tangle=1) #auto turn on zero phase

        #set didier raw data -- NEED TO CHECK WHAT THIS IS
        #t_table=np.loadtxt(os.path.join(__refdata__,'climate_INPUTS/tlog'),usecols=[0],unpack=True)
        #p_table=np.loadtxt(os.path.join(__refdata__,'climate_INPUTS/plog'),usecols=[0],unpack=True)

        #grad=np.zeros(shape=(53,26))
        #cp = np.zeros(shape=(53,26))
        
        #grad_inp, i_inp, j_inp = np.loadtxt(os.path.join(__refdata__,'climate_INPUTS/GRAD_FOR_PY_Y28'),usecols=[0,1,2],unpack=True)
        #for i in range(len(grad_inp)):
        #    grad[int(i_inp[i]-1),int(j_inp[i]-1)]=grad_inp[i]
        
        cp_grad = json.load(open(os.path.join(__refdata__,'climate_INPUTS','specific_heat_p_adiabat_grad.json')))

        #log10 base temperature Kelvin 
        self.inputs['climate']['t_table'] = np.array(cp_grad['temperature'])
        #log10 base pressure bars 
        self.inputs['climate']['p_table'] = np.array(cp_grad['pressure'])
        #\nabla_ad = d ln T/ d ln P |_S (at constant entropy)
        self.inputs['climate']['grad'] = np.array(cp_grad['adiabat_grad'])
        #log Cp (erg/g/K);Specific heat at constant pressure for the same H/He 
        self.inputs['climate']['cp'] = np.array(cp_grad['specific_heat'])


    

    def setup_nostar(self):
        """
        Turns off planet specific things, so program can run as usual
        """
        self.inputs['approx']['rt_params']['common']['raman'] = 2 #turning off raman scattering
        self.inputs['star']['database'] = 'nostar'
        self.inputs['star']['temp'] = 'nostar'
        self.inputs['star']['logg'] = 'nostar'
        self.inputs['star']['metal'] = 'nostar'
        self.inputs['star']['radius'] = 'nostar'
        self.inputs['star']['radius_unit'] = 'nostar' 
        self.inputs['star']['flux'] = 'nostar' 
        self.inputs['star']['wno'] = 'nostar' 
        self.inputs['star']['semi_major'] = 'nostar' 
        self.inputs['star']['semi_major_unit'] = 'nostar' 

    def star(self,opannection,temp=None, metal=None, logg=None ,radius = None, radius_unit=None,
        semi_major=None, semi_major_unit = None, #deq = False, 
        database='ck04models',filename=None, w_unit=None, f_unit=None):
        """
        Get the stellar spectrum using stsynphot and interpolate onto a much finer grid than the 
        planet grid. 

        Parameters
        ----------
        opannection : class picaso.RetrieveOpacities
            This is the opacity class and it's needed to get the correct wave info and raman scattering cross sections
        temp : float 
            (Optional) Teff of the stellar model if using the stellar database feature. 
            Not needed for filename option. 
        metal : float 
            (Optional) Metallicity of the stellar model if using the stellar database feature. 
            Not needed for filename option. 
        logg : float 
            (Optional) Logg cgs of the stellar model if using the stellar database feature. 
            Not needed for filename option. 
        radius : float 
            (Optional) Radius of the star. Only needed as input if you want relative flux units (Fp/Fs)
        radius_unit : astropy.unit
            (Optional) Any astropy unit (e.g. `radius_unit=astropy.unit.Unit("R_sun")`)
        semi_major : float 
            (Optional) Semi major axis of the planet. Only needed to compute fp/fs for albedo calculations. 
        semi_major_unit : astropy.unit 
            (Optional) Any astropy unit (e.g. `radius_unit=astropy.unit.Unit("au")`)
        database : str 
            (Optional)The database to pull stellar spectrum from. See documentation for pysynphot. 
            Most popular are 'ck04models', phoenix' and 
        filename : str 
            (Optional) Upload your own stellar spectrum. File format = two column white space (wave, flux). 
            Must specify w_unit and f_unit 
        w_unit : str 
            (Optional) Astropy Unit 
        f_unit : str 
            (Optional) Astrpy Unit 
        """
        #most people will just upload their thing from a database
        if (not isinstance(radius, type(None))):
            r = (radius*radius_unit).to(u.cm).value
            radius_unit='cm'
        else :
            r = np.nan
            radius_unit = "Radius not supplied"

        #add semi major axis if supplied 
        if (not isinstance(semi_major, type(None))):
            semi_major = (semi_major*semi_major_unit).to(u.cm).value
            semi_major_unit='cm'
        else :
            if 'climate' in self.inputs['calculation']:raise Exception('star function requires semi_major and semi_major_input when running climate')
            semi_major = np.nan
            semi_major_unit = "Semi Major axis not supplied"        

        #upload from file  
        if (not isinstance(filename,type(None))):
            star = np.genfromtxt(filename, dtype=(float, float), names='w, f')
            flux = star['f']
            wave = star['w']
            ST_SS = SourceSpectrum(Empirical1D, points=wave*u.Unit(w_unit), lookup_table=flux*u.Unit(f_unit))
        elif ((not isinstance(temp, type(None))) & (not isinstance(metal, type(None))) & (not isinstance(logg, type(None)))):
            ST_SS = sts.grid_to_spec(database, temp, metal, logg) 
        else: 
            raise Exception("Must enter 1) filename,w_unit & f_unit OR 2)temp, metal & logg ")

        wno_star = 1e4/(ST_SS.waveset).to(u.um).value[::-1]
        flux_star = ST_SS(ST_SS.waveset,flux_unit=u.Unit('erg*cm^(-3)*s^(-1)')).value[::-1]
        
        # Get a bool for whether we want level fluxes
        get_lvl_flux = self.inputs['approx'].get('get_lvl_flux', False)

        wno_planet = opannection.wno

        #this adds stellar shifts 'self.raman_stellar_shifts' to the opacity class
        #the cross sections are computed later 
        if self.inputs['approx']['rt_params']['common']['raman'] == 0: 
            max_shift = np.max(wno_planet)+6000 #this 6000 is just the max raman shift we could have 
            min_shift = np.min(wno_planet) -2000 #it is just to make sure we cut off the right wave ranges
            #do a fail safe to make sure that star is on a fine enough grid for planet case 
            fine_wno_star = np.linspace(min_shift, max_shift, len(wno_planet)*5)
            fine_flux_star = np.interp(fine_wno_star,wno_star, flux_star)
            
            opannection.compute_stellar_shits(fine_wno_star, fine_flux_star)
            bin_flux_star = opannection.unshifted_stellar_spec

        elif ('climate' in self.inputs['calculation'] or (get_lvl_flux)):
            if not ((not np.isnan(semi_major)) & (not np.isnan(r))): 
                raise Exception ('semi_major and r parameters are not provided but are needed to compute relative fluxes for climate calculation or when get_lvl_flux are being requested')

            # Ensure valid values for interpolation
            mask_valid = flux_star > 1e-30  
            if not np.all(mask_valid):
                wno_star, flux_star = wno_star[mask_valid], flux_star[mask_valid]

            # Log-space interpolation
            interpolator = interp1d(np.log10(wno_star), np.log10(flux_star), kind='linear',
                                    fill_value='extrapolate', bounds_error=False)
            fine_flux_star = 10**interpolator(np.log10(wno_planet))
            
            # Compute binned flux using trapezoidal integration
            fine_flux_star[:] = [np.trapz(fine_flux_star[(wno_planet >= wno_planet[i]) & 
                                                        (wno_planet <= wno_planet[i+1])], 
                                        x=-1/wno_planet[(wno_planet >= wno_planet[i]) & 
                                                        (wno_planet <= wno_planet[i+1])]) 
                                if i < len(wno_planet) - 1 else 0 for i in range(len(wno_planet))]

            # Linear extrapolation for the last point
            if len(wno_planet) > 2:
                slope = (fine_flux_star[-2] - fine_flux_star[-3]) / (wno_planet[-2] - wno_planet[-3])
                fine_flux_star[-1] = fine_flux_star[-2] + slope * (wno_planet[-1] - wno_planet[-2])

            # Handle NaNs and zeros
            mask = np.logical_or(np.isnan(fine_flux_star), fine_flux_star == 0)
            if np.sum(mask) > 20:
                print(f"Having to replace {len(fine_wno_star[mask])} zeros or nans in stellar spectra")
                non_zero_indices = np.where(~mask)[0]
                fine_flux_star[mask] = np.interp(wno_planet[mask], wno_planet[non_zero_indices], fine_flux_star[non_zero_indices])
            

            opannection.unshifted_stellar_spec = fine_flux_star  
            bin_flux_star = fine_flux_star          
            unit_flux =  'ergs cm^{-2} s^{-1}'
        else :
            flux_star_interp = np.interp(wno_planet, wno_star, flux_star)
            _x,bin_flux_star = mean_regrid(wno_star, flux_star,newx=wno_planet)
            #where the star wasn't high enough resolution  
            idx_nobins = np.where(np.isnan(bin_flux_star))[0]
            #replace no bins with interpolated values 
            bin_flux_star[idx_nobins] = flux_star_interp[idx_nobins]
            opannection.unshifted_stellar_spec =bin_flux_star
            unit_flux =  'ergs cm^{-2} s^{-1} cm^{-1}'
        
        
        #only compute relative flux if stellar radius and semi major axis are provided
        if ((not np.isnan(semi_major)) & (not np.isnan(r))): 
            opannection.relative_flux = bin_flux_star * (r/semi_major)**2
        else: 
            opannection.relative_flux = bin_flux_star*0 + 1 #no semi major supplied and so no relative flux exists 

        self.inputs['star']['database'] = database
        self.inputs['star']['temp'] = temp
        self.inputs['star']['logg'] = logg
        self.inputs['star']['metal'] = metal
        self.inputs['star']['radius'] = r 
        self.inputs['star']['radius_unit'] = radius_unit 
        self.inputs['star']['flux'] = bin_flux_star
        self.inputs['star']['flux_unit'] =unit_flux
        self.inputs['star']['relative_flux'] =opannection.relative_flux
        self.inputs['star']['relative_flux_unit'] = f'(Rs/Sa)^2 * {unit_flux}'
        self.inputs['star']['wno'] = wno_planet
        self.inputs['star']['semi_major'] = semi_major 
        self.inputs['star']['semi_major_unit'] = semi_major_unit    
        self.inputs['star']['filename'] = filename
        self.inputs['star']['w_unit'] = w_unit
        self.inputs['star']['f_unit'] = f_unit 


    def atmosphere(self, df=None, filename=None, exclude_mol=None, 
        mh=None, cto_absolute=None, cto_relative=None, chem_method=None,
        #for now the next line is only climate params 
        quench=False,no_ph3=False,cold_trap=False,vol_rainout=False,
        #these only used for photochem climate 
        photochem_init_args=None,add_visscher_abunds = True,
        **pd_kwargs):
        """
        Builds a dataframe and makes sure that minimum necessary parameters have been suplied.
        Sets number of layers in model.  

        Parameters
        ----------
        df : pandas.DataFrame or dict
            (Optional) Dataframe with volume mixing ratios and pressure, temperature profile. 
            Must contain pressure (bars) at least one molecule
        filename : str 
            (Optional) Filename with pressure, temperature and volume mixing ratios.
            Must contain pressure at least one molecule
        exclude_mol : list of str 
            (Optional) List of molecules to ignore from opacity. It will NOT 
            change other aspects of the calculation like mean molecular weight. 
            This should be used as exploratory ONLY. if you actually want to remove 
            the contribution of a molecule entirely from your profile you should remove 
            it from your input data frame. 
        mh : float 
            Metallicity relative to Solar 
        cto_relative : float 
            Carbon-to-Oxygen Ratio relative to Solar (Solar value is determined by `chem_method`)
        cto_absolute : float
            Carbon-to-Oxygen Ratio in absolute terms (e.g. 0.55)
        chem_method : str 
            Current options: 
            - 'visscher' : uses the 2121 chemical equilibrium tables computed by Channon Visscher via the function `chemeq_visscher`
                if 'visscher' needs to input: mh and cto 
            - 'visscher_1060' : uses the chemical equilibrium tables computed by Channon Visscher on the 1060 grid via the function `chemeq_visscher`
                if 'visscher' needs to input: mh and cto 
            - 'photochem' : users photochem model by Nick Wogan
                if 'photochem' user needs to input photochem_init_args and photochem_TOA_pressure
        quench : bool 
            Climate only, default = False: no quencing
        no_ph3 : bool 
            Climate only chem hack, default=False: True removes any PH3 from the atmosphere 
        cold_trap : bool 
            Climate only chem hack, default=False: Force H2O and NH3 abundances to be cold trapped after condensation.
        vol_rainint : bool ;
            Climate only chem hack, default=False: will rainout volatiles like H2O, CH4 and NH3 in diseq runs as in equilibrium model when applicable
        photochem_init_args : dict
            Dictionary containing initialization arguments for photochem. Should contain the following keys
            - "mechanism_file" : str
                Path to the file describing the reaction mechanism
            - "stellar_flux_file" : str
                Path to the file describing the stellar UV flux.
            - "planet_mass" : float
                Planet mass in grams
            - "planet_radius" : float
                Planet radius in cm
            - "nz" : int, optional
                The number of layers in the photochemical model, by default 100
            - "P_ref" : float, optional
                Pressure level corresponding to the planet_radius, by default 1e6 dynes/cm^2
            - "thermo_file" : str, optional
                Optionally include a dedicated thermodynamic file.
            - "TOA_pressure" : float
            Pressure at the top of the atmosphere for photochem, by default 1e-7 bar. Unit must be in dynes/cm^2
        add_visscher_abunds : bool 
            Default = False; Only used for photochemical results. Adds visscher to fill gaps covered by the photochemical mdoel 
        verbose : bool 
            (Optional) prints out warnings. Default set to True
        pd_kwargs : kwargs 
            Key word arguments for pd.read_csv to read in supplied atmosphere file 
        """        
        
        #if a dataframe was input lets check it out and set nlevels
        if not isinstance(df, type(None)):
            if ((not isinstance(df, dict )) & (not isinstance(df, pd.core.frame.DataFrame ))): 
                raise Exception("df must be pandas DataFrame or dictionary")
            else:
                self.nlevel=df.shape[0] 
        #if a filename was input lets read it and set nlevels
        elif not isinstance(filename, type(None)):
            df = pd.read_csv(filename, **pd_kwargs)
            self.nlevel=df.shape[0] 
        
        #if we already have a dataframe in here let's just define df as is assume the user wants to modify chem with only a PT
        elif isinstance(self.inputs['atmosphere']['profile'] ,pd.core.frame.DataFrame ): 
            df = self.inputs['atmosphere']['profile']
        else:
            if 'climate' in self.inputs['calculation']:
                raise("Could not find a starting dataframe in inputs['atmosphere']['profile']. You are running a climate model so this dataframe is usually initialized in inputs_climate() function that needs a temp_guess and pressure_guess. You can also use the function add_PT() or set it yourself manually. ")
            else:
                raise Exception("Could not find a starting dataframe in inputs['atmosphere']['profile'] and no df or filename were specified")

        #if we dont have pressure in the dataframe its a full stop. 
        if 'pressure' not in df.keys(): 
            raise Exception("Check column names. `pressure` must be included. For climate runs set your initial guess in `inputs_climate` before running atmosphere class to set the chemistry")
        
        #if we dont have temperature that might be okay.. this means its a climate model and we dont have a T
        if ('temperature' not in df.keys()):
            #if its not a climate calculation, then full stop
            if 'climate' not in self.inputs['calculation']:
                raise Exception("`temperature` not specified as a column/key name")

        # if there ar molecules we want to exclude lets make sure they are in list format
        if not isinstance(exclude_mol, type(None)):
            if  isinstance(exclude_mol, str):
                exclude_mol = [exclude_mol]
            
            #now lets transfer to a dictionary for each molecule the user has chosen
            #this way we can flip them on and off individually
            self.inputs['atmosphere']['exclude_mol'] = {i:1 for i in df.keys()}
            for i in exclude_mol: 
                self.inputs['atmosphere']['exclude_mol'][i]=0
        else: 
            self.inputs['atmosphere']['exclude_mol'] = 1

        #sort by pressure to make sure 0 index is low pressure, last index is high pressure
        self.inputs['atmosphere']['profile'] = df.sort_values('pressure').reset_index(drop=True)

        #lastly check to see if the atmosphere is non-H2 dominant. 
        #if it is, let's turn off Raman scattering for the user. 
        if df.shape[1]>2:
            if (("H2" not in df.keys()) and (self.inputs['approx']['rt_params']['common']['raman'] != 2)):
                self.inputs['approx']['rt_params']['common']['raman'] = 2
            elif (("H2" in df.keys()) and (self.inputs['approx']['rt_params']['common']['raman'] != 2)): 
                if df['H2'].min() < 0.7: 
                    self.inputs['approx']['rt_params']['common']['raman'] = 2

        #now, if mh and cto were supplied lets add those to inputs and set the chem method requestd 
        if (mh != None ):
            self.inputs['atmosphere']['mh'] = mh 
            if ((cto_absolute == None) and isinstance(cto_relative, (float,int))): 
                cto_absolute=cto_relative*0.549
            elif (cto_relative ==None and isinstance(cto_absolute, (float,int))): 
                cto_relative = cto_absolute/0.549 #such that if user did c/o=1, then cto=0.549
            elif 'cto' in pd_kwargs: 
                raise Exception('cto is not an acceptance argument. need to input either cto_relative or cto_absolute.')
            else: 
                raise Exception('mh was specified but cto_relative or cto_absolute was not. need to input one of these. ')
            self.inputs['atmosphere']['cto_relative'] = cto_relative 
            self.inputs['atmosphere']['cto_absolute'] = cto_absolute 
            self.inputs['approx']['chem_method'] = chem_method
        
        #add photochem initialization if it exists 
        if photochem_init_args!=None: 
            self.inputs['atmosphere']['photochem_init_args'] = photochem_init_args 
            if add_visscher_abunds: 
                #if we also want visscher then we can make the chem method "photochem+visccher"
                self.inputs['approx']['chem_method'] = self.inputs['approx']['chem_method']+'+visscher'

            # sets chemistry options and runs chemistry if the user has input a PT profile
            # otherwise this just checks for valid inputs 
            self.chemistry_handler()

        #SET ATMOSPHERE APPROXIMATIONS 
        #if this is not a climate calculation and one of the parameters is True, then braek the code 
        #TODO: allow users to run these approx for forward models.
        if (self.inputs['calculation'] != 'climate'):
            if np.any([ quench,  no_ph3,  cold_trap,  vol_rainout]):
                raise Exception (f"'quench','no_ph3','cold_trap','vol_rainout' are a climate kwargs and climate calculation is not specified so this will not do anything to the user input chemistry. Please set to false to not avoid confusion. In a later update we could create a portal to these kwargs for the forward modeling.")
        
        #if we've made it this far lets just save the approximation params in chem_params
        self.inputs['approx']['chem_params']=self.inputs['approx'].get('chem_params',{})
        for ikey,ibool in zip(['quench','no_ph3','cold_trap','vol_rainout'],
                              [ quench,  no_ph3,  cold_trap,  vol_rainout]):
            self.inputs['approx']['chem_params'][ikey]=ibool

    def chemistry_handler(self, chemistry_table = None):
        """
        This function sets the chemistry table that we want to use, whether it is the 1060, 2121 and if we want to 
        do photohchemistry.

        Parameters
        ----------
        chemistry_table : str
            Chemistry table type
        """
        #add default chem method
        chem_method = self.inputs['approx'].get('chem_method',None)
        atmosphere_profile = self.inputs['atmosphere']['profile']
        
        # Are we running chemistry or just setting inputs ?
        # if the user has supplied a T and P we will just assume they want to run chemistry

        if (('temperature' in atmosphere_profile.keys()) & ('pressure' in atmosphere_profile.keys())):
            run = True 
        else: 
            run = False 

        #lets set a bool to see if we find a valid chem method
        found_method = False

        # Option : simplest method where we just grab visscher abundances 
        
        if 'visscher_1060' in str(chem_method):            
            mh = self.inputs['atmosphere']['mh'] 
            cto = self.inputs['atmosphere']['cto_relative']   
            if run: self.chemeq_visscher_1060(cto, np.log10(mh))   
            found_method = True
        elif 'visscher' in str(chem_method):  
            mh = self.inputs['atmosphere']['mh'] 
            cto = self.inputs['atmosphere']['cto_absolute']   
            if run: self.chemeq_visscher_2121(cto, np.log10(mh)) 
            found_method = True

        if (('photochem' in str(chem_method)) and (self.inputs['climate'].get('pc',0)==0)): 
            #initialize photochemistry inputs on first time 
            self.photochem_init()
            found_method = True
        
        # Option : Here the user has supplied a chemistry table and we just need to use the chem_interp function to interpolate on that table
        # Notes : This method inherently assumes mh and cto since the loaded table is for a single mh/co
        if not isinstance(chemistry_table, type(None)): 
            self.inputs['approx']['chem_method'] = 'chemistry table loaded through opannection'
            if run: self.chem_interp(chemistry_table)
            found_method=True
        #Option : No other options so far 
        elif not found_method: 
            raise Exception(f"A chem option {chem_method} is not valid. Likely you specified method='resrotrebin' in opannection but did not run `atmosphere()` function after inputs_climate.") 
    
    def volatile_rainout(self,quench_levels,species_to_consider = ['H2O', 'CH4','NH3']):
        """
        Enforces rainout along pvap. So far these are only H2O, CH4 and NH3, since these are the only major species 
        that are in the abundance tables and would be expected to rainout.

        Parameters
        ----------
        quench_levels :
            Layer in the atmosphere where each of the species is quenched.
        species_to_consider : list of str
            Default is ['H2O', 'CH4','NH3']
        """

        quench_molecules = np.concatenate([i.split('-') for i in quench_levels.keys()])
        quench_by_molecule={}
        
        #Q TO JM: Do we only want to loop through the three below 
        
        species_to_adjust = [i for i in species_to_consider if i in self.inputs['atmosphere']['profile'].keys()]

        #species_to_adjust = [i for i in cld_species if i in self.inputs['atmosphere']['profile'].keys()]
        #species_to_adjust = [i for i in species_to_adjust if i in consider_only_these]
        

        #only consider the three above for now
        
        
        for iq in quench_levels.keys():
            for imol in species_to_adjust: 
                if imol in iq: 
                    if quench_levels[iq] < self.nlevel:
                        quench_by_molecule[imol]=quench_levels[iq]
                    else: #this case is if we extended the atmosphere for a deep quench level, the most bottom level abundance is the quench level
                        quench_by_molecule[imol]=self.nlevel-1

        H2 = self.inputs['atmosphere']['profile']['H2'].values
        for imol in species_to_adjust: 
            old = self.inputs['atmosphere']['profile'].loc[:,imol].values 
            #skip anything we dont have a quench level for
            if imol not in quench_molecules: continue 

            get_pvap = getattr(vj.pvaps,imol,0)
            
            #only proceed if we actually have a saturation vapor pressure curve
            if get_pvap !=0:
                #get the quench abundance so we have a point of comparison
                quench_abundance=self.inputs['atmosphere']['profile'].loc[quench_by_molecule[imol],imol]
                #for above layers to the quench point at depth 
                for i in range(0,quench_by_molecule[imol]+1): 
                    #get the pvap abundance at this temperature 
                    pvap_abundance = get_pvap(self.inputs['atmosphere']['profile'].loc[i,'temperature'])*1e-6
                    #compare and reset the profile abundance 
                    if pvap_abundance < quench_abundance:
                        self.inputs['atmosphere']['profile'].loc[i,imol] = pvap_abundance
            
            new = self.inputs['atmosphere']['profile'].loc[:,imol].values 
            diff = old - new 
            H2 = H2 + diff 

        #reset H2 accordingly
        self.inputs['atmosphere']['profile'].loc[:,'H2'] = H2

    def cold_trap(self,species_to_consider = ['H2O','CH4','NH3']): 
        """
        Enforces cold trapping along the chemeq grid for any species included in the cld_species set. Currently only set
        for H2O, CH4 and NH3. 
        """
        #only adjust species that actually have chem computed 
        
        species_to_adjust = [i for i in species_to_consider if i in self.inputs['atmosphere']['profile'].keys()]
        H2 = self.inputs['atmosphere']['profile']['H2'].values
        for mol in species_to_adjust:
            #can generalize later for other mh and mmw but for now, good enough to gauge where to start coldtrapping
            cond_p, cond_t = vj.condensation_t(mol, 1, 2.2, pressure = self.inputs['atmosphere']['profile']['pressure'])            
            try:
                cond_layer = np.where(cond_t > self.inputs['atmosphere']['profile']['temperature'])[0][-1]
            except IndexError:
                    continue
            
            # need to ignore the bottom 10% of layers to avoid the changes in deep atmosphere to properly identify condensation layer
            #JM cutoff = int(0.1 * self.nlevel)  # Dynamically ignore bottom 10% of layers
            # relevant_layers = inverted[:self.nlevel - cutoff]
            # grad = np.abs(np.gradient(relevant_layers))  # Compute abundance gradient

            # unique_vals, counts = np.unique(inverted, return_counts=True)
            # mode_value = unique_vals[np.argmax(counts)]
            # threshold = mode_value * 0.01 # Define a threshold for significant drop (adjustable)

            # Find the first layer where the abundance starts to fall off
            # cond_idx = np.where(grad > threshold)[0]
            #JMcond_layer = self.nlevel - cutoff #- cond_idx[0]
            old = self.inputs['atmosphere']['profile'].loc[:,mol].values 
            for i in range(cond_layer-1, 0, -1): 
                if self.inputs['atmosphere']['profile'].loc[i,mol] < self.inputs['atmosphere']['profile'].loc[i-1,mol]:
                    self.inputs['atmosphere']['profile'].loc[i-1,mol] = self.inputs['atmosphere']['profile'].loc[i,mol]
            new = self.inputs['atmosphere']['profile'].loc[:,mol].values 
            diff = old - new 
            H2 = H2 + diff 
        
        #reset H2 accordingly
        self.inputs['atmosphere']['profile'].loc[:,'H2'] = H2

    def premix_atmosphere(self, opa=None, quench_levels=None, verbose=True):
        """
        Builds a dataframe and makes sure that minimum necessary parameters have been suplied.
        Sets number of layers in model.  
        Parameters
        ----------
        opa : class 
            Opacity class from opannection : RetrieveCks() 
        quench_levels : dict
            Dictionary with the quench levels for each molecule.
        """

        #get a chemistry table from opa if the user supplied it and it exists
        chemistry_table = getattr(opa, 'full_abunds', None)

        #now chemistry options can be enforced basically doing chemical equilibrium / photochem / only chemeq included
        self.chemistry_handler(chemistry_table=chemistry_table)

        #if a quench level dictionary is provided 
        if self.inputs['approx']['chem_params']['quench'] and isinstance(quench_levels,dict):
            if verbose: print('Quench=True; Adjusting quench chemistry')
            self.adjust_quench_chemistry(quench_levels,chemistry_table)
       
        # volatile rainout 
        if self.inputs['approx']['chem_params']['vol_rainout'] and isinstance(quench_levels,dict): 
            if verbose: print(f'vol_rainout=True; Adjusting volatile rainout')
            self.volatile_rainout(quench_levels)

        # cold trap the condensibles 
        if self.inputs['approx']['chem_params']['cold_trap']: 
            if verbose: print(f'cold_trap=True; Adjusting cold trap')
            self.cold_trap()

        #zero out ph3 if hack requested
        if self.inputs['approx']['chem_params']['no_ph3']: 
            #check to see if its there, and zero out if it is 
            if verbose: print('no_ph3=True; Goodbye PH3!')
            if 'PH3' in self.inputs['atmosphere']['profile'].keys(): 
                self.inputs['atmosphere']['profile']['PH3'] = 0
    
    def premix_atmosphere_photochem(self,quench_levels=None, verbose=True):
        """
        This function runs the photochemistry model, and updates the chemical abundance profiles.

        Parameters
        ----------
        quench_levels : dict
            Dictionary with the quench levels for each molecule.
        verbose : bool
            If True, prints out the progress of the photochemistry model.
        """
        if verbose: print('Running photochem')
        #start by getting chemeq if it has been requested 
        self.chemistry_handler()
        #adjust quenching if we have levels to give us a better inital guess 
        if quench_levels!=None: self.adjust_quench_chemistry(quench_levels)
        
        #set photochem to run 
        pc = self.inputs['climate']['pc']
        #gets whatever kzz is specified either constant or self consistent
        kz = self.find_kzz()
        
        df = pc.run_for_picaso(
                        self.inputs['atmosphere']['profile'], 
                        np.log10(float(self.inputs['atmosphere']['mh'])), 
                        float(self.inputs['atmosphere']['cto_relative']), 
                        kz, 
                        True
                    )
        #reset kz to picaso dataframe to keep track of it
        #neb trying to commect this out as i htink this is not needed anymore 
        #df['kz']=kz
        self.inputs['atmosphere']['profile']  = df 
    
    def find_kzz(self):
        """
        Returns in this order: 
        1) self.inputs['atmosphere']['kzz']['constant_kzz']
        OR
        2) self.inputs['atmosphere']['kzz']['sc_kzz'] 
        OR 
        3) self.inputs['atmosphere']['profile']['kz'] 
        OR 
        None
        """
        kzz_dict = self.inputs['atmosphere'].get('kzz',self.inputs['atmosphere']['profile'])
        kz = kzz_dict.get('constant_kzz', kzz_dict.get('sc_kzz', kzz_dict.get('kz',None)))
        return np.array(kz)
    
    def adjust_quench_chemistry(self, quench_levels,chemistry_table=None):
        """
        Adjusts the abundances of quenched species in the chemistry profile based on the provided quench levels.
        CO2 is defaulted to do the Zahnle and Marley (2014) kinetic fix outlined in Wogan et al. 2025 research note. 

        Parameters
        ----------
        quench_levels : dict
            Dictionary with the quench levels for each molecule.
        """
        extend_deeper = False
        kinetic_CO2=True #since our old way was an "error" it seems like we should include htis as an option to be false
        
        df_atmo_og  = self.inputs['atmosphere']['profile']
        
        #kzz could be in a variety of different places.
        #creates a function to make sure correct kzz (sc or constant or from profile or None )
        #this is just needed for the case where things get extended to deeper layers
        kz = self.find_kzz()


        temperature = df_atmo_og['temperature'].values
        pressure = df_atmo_og['pressure'].values

        # Check if any quench level is deeper than the available pressure grid, if so, extend the pressure grid
        nlevel = len(temperature)
        if any(quench_levels[key] >= nlevel for key in quench_levels):
            extend_deeper = True
            print('Extending atmosphere profile deeper to accommodate deep quench levels.')

            # calculate dtdp to accurately extend profile
            for j in range(nlevel -1):
                dtdp=np.zeros(shape=(nlevel-1))
                dtdp[j] = (log( temperature[j]) - log( temperature[j+1]))/(log(pressure[j]) - log(pressure[j+1]))

            # extend the pressure by 10 layers to 1e6 bars (deep enough for even the coldest cases)
            extended_pressure = np.logspace(np.log10(pressure[-1]+100),6,10)
            pressure = np.append(pressure, extended_pressure)
            for i in np.arange(nlevel, nlevel+10):
                new_temp = np.exp(np.log(temperature[i-1]) - dtdp[-1] * (np.log(pressure[i-1]) - np.log(pressure[i])))
                temperature = np.append(temperature, new_temp)
            nlevel = len(temperature)

            # create new df to have the right shape to be used
            self.inputs['atmosphere']['profile'] = pd.DataFrame({'pressure': pressure, 'temperature': temperature})#, 'kz': kz})
            df_atmo_og = self.inputs['atmosphere']['profile']

            # redo the chemistry to make sure new extended layers have to correct abundances
            if chemistry_table is not None:
                self.chemistry_handler(chemistry_table=chemistry_table)
            else:
                self.chemistry_handler()

        #what order will we quench things 
        quench_sequence  = ['PH3','CO-CH4-H2O','CO2','NH3-N2','HCN']

        # start with the molecules in the quench sequence 
        # anything we quench let's take away/add to H2 
        H2 = df_atmo_og['H2'].values
        for iquench in quench_sequence:
            
            #this defines the exactl quench layer 
            quench_level = quench_levels[iquench]
            
            #now individually loop through the sequence if there are multiple molecules included 
            for imol in iquench.split('-'):
                
                #if the molecule is in the set
                if imol in df_atmo_og.keys():
                    #get the abundance at the quench point
                    quench_abundance = df_atmo_og.loc[quench_level,imol]
                    #Everything above the quench point gets set to the quench abundance 
                    old = df_atmo_og.loc[:,imol].values 
                    df_atmo_og.loc[0:quench_level+1,imol] = quench_abundance
                    new = df_atmo_og.loc[:,imol].values 
                    diff = old - new 
                    #adjust H2 accordingly 
                    H2 = H2 + diff 

        # include new option for CO2 in equilibrium with CO, H2O, H2 from equation 43 in Zahnle and Marley (2014)
        if kinetic_CO2 == True:
            #ADD LINK TO NICK W. AAS NOTE HERE 
            K = 18.3*np.exp(-2376/self.inputs['atmosphere']['profile']['temperature'] - (932/self.inputs['atmosphere']['profile']['temperature'])**2)
            fCO2 = (self.inputs['atmosphere']['profile']['CO']*self.inputs['atmosphere']['profile']['H2O'])/(K*self.inputs['atmosphere']['profile']['H2'])            
            # apply the new quench starting from original CO2 quench point
            fCO2[:quench_levels['CO2']] = fCO2[quench_levels['CO2']]
            #set new value and adjust H2 again 
            old = df_atmo_og.loc[:,'CO2'].values
            df_atmo_og.loc[:,'CO2'] = fCO2[:]
            new = df_atmo_og.loc[:,'CO2'].values 
            diff = old - new 
            #adjust H2 accordingly 
            H2 = H2 + diff 
        
        #reset H2 accordingly
        df_atmo_og.loc[:,'H2'] = H2
        #set new atmosphere 
        if extend_deeper == True: #drop last 10 layers if we extended deeper
            self.inputs['atmosphere']['profile'] = df_atmo_og.iloc[:-10]
            ## this is just adding whatever kz used here back to atmosphere profile but I dont think this has to be done
            #if kz is not None:
            #    self.inputs['atmosphere']['profile']['kz'] = kz
        else:
            self.inputs['atmosphere']['profile'] = df_atmo_og

    def premix_atmosphere_diseq_deprecate(self, opa, quench_levels, teff, t_mix=None, df=None, filename=None, vol_rainout = False, 
                                quench_ph3 = True, kinetic_CO2 = True, no_ph3 = False, cold_trap=False, cld_species = None, **pd_kwargs):
        """
        Builds a dataframe and makes sure that minimum necessary parameters have been suplied.
        Sets number of layers in model.  
        Parameters
        ----------
        opa : class 
            Opacity class from opannection : RetrieveCks() 
        teff : float
            Effective temperature of the object
        df : pandas.DataFrame or dict
            (Optional) Dataframe with volume mixing ratios and pressure, temperature profile. 
            Must contain pressure (bars) at least one molecule
        filename : str 
            (Optional) Filename with pressure, temperature and volume mixing ratios.
            Must contain pressure at least one molecule
        exclude_mol : list of str 
            (Optional) List of molecules to ignore from file
        vol_rainout : bool
            (Optional) If True, will rainout volatiles like H2O, CH4 and NH3 as in equilibrium model when applicable
        cold_trap : bool
            Option to cold trap condensible species, default = False
        cld_species : list of str
            (Optional) List of condensing species
        pd_kwargs : kwargs 
            Key word arguments for pd.read_csv to read in supplied atmosphere file 
        """
        if not isinstance(df, type(None)):
            if ((not isinstance(df, dict )) & (not isinstance(df, pd.core.frame.DataFrame ))): 
                raise Exception("df must be pandas DataFrame or dictionary")
            else:
                self.nlevel=df.shape[0] 
        elif not isinstance(filename, type(None)):
            df = pd.read_csv(filename, **pd_kwargs)
            self.nlevel=df.shape[0] 

        if 'pressure' not in df.keys(): 
            raise Exception("Check column names. `pressure` must be included.")

        if ('temperature' not in df.keys()):
            raise Exception("`temperature` not specified as a column/key name")

        # for super cold cases, most quench points are deep in the atmosphere, we don't want to run all models too deep. Use this
        #   extapolation to temporarily capture the proper chemical abundances calculated but return to original df pressure grid later
        if teff <= 250 and df['pressure'].values[-1] < 1e6:
            #calculate dtdp to use to extrapolate thermal structure deeper
            dtdp=np.zeros(shape=(self.nlevel-1))
            temp = df['temperature'].values
            pressure = df['pressure'].values
            for j in range(self.nlevel -1):
                dtdp[j] = (np.log( temp[j]) - np.log( temp[j+1]))/(np.log(pressure[j]) - np.log(pressure[j+1]))

            # extend pressure down to 1e6 bars
            extended_pressure = np.logspace(np.log10(pressure[-1]+100),6,10)
            pressure = np.append(pressure, extended_pressure)
            for i in np.arange(self.nlevel, self.nlevel+10):
                new_temp = np.exp(np.log(temp[i-1]) - dtdp[-1] * (np.log(pressure[i-1]) - np.log(pressure[i])))
                temp = np.append(temp, new_temp)

            self.inputs['atmosphere']['profile'] = pd.DataFrame({'pressure': pressure, 'temperature': temp})
        else:
            self.inputs['atmosphere']['profile'] = df.sort_values('pressure').reset_index(drop=True)

        #Turn off raman for 196 premix calculations 
        self.inputs['approx']['rt_params']['common']['raman'] = 2

        self.chem_interp(opa.full_abunds)
        
        # first quench PH3 from eq abundances of H2O and H2

        # quenching PH3 now, this will only have effect if everything is mixed on the fly
        # formalism from # https://iopscience.iop.org/article/10.1086/428493/pdf
        if quench_ph3 == True:
            OH = OH_conc(self.inputs['atmosphere']['profile']['temperature'].values,self.inputs['atmosphere']['profile']['pressure'].values,self.inputs['atmosphere']['profile']['H2O'].values,self.inputs['atmosphere']['profile']['H2'].values)
            t_chem_ph3 = 0.19047619047*1e13*np.exp(6013.6/self.inputs['atmosphere']['profile']['temperature'].values)/OH
            quench_levels_ph3 = int(0.0)
            for j in range(len(self.inputs['atmosphere']['profile']['temperature'].values)-1,0,-1):

                if ((t_mix[j-1]/1e15) <=  (t_chem_ph3[j-1]/1e15)) and ((t_mix[j]/1e15) >=  (t_chem_ph3[j]/1e15)):
                    quench_levels_ph3 = j
                    break
            #quench_levels_ph3 =0
            print('PH3 quenched at level', quench_levels_ph3)
            # quench ph3 now

            self.inputs['atmosphere']['profile']['PH3'][0:quench_levels_ph3+1] = self.inputs['atmosphere']['profile']['PH3'][0:quench_levels_ph3+1]*0.0 + self.inputs['atmosphere']['profile']['PH3'][quench_levels_ph3]
        if no_ph3 == True:
            self.inputs['atmosphere']['profile']['PH3'] = self.inputs['atmosphere']['profile']['PH3']*0.0


        
        qvmrs=np.zeros(shape=(5))
        qvmrs2=np.zeros(shape=(3))

        if np.min(quench_levels) >-1 :
        
        ### quench point abundances of each
            qvmrs[0] = self.inputs['atmosphere']['profile']['CH4'][quench_levels[0]]
            qvmrs[1] = self.inputs['atmosphere']['profile']['H2O'][quench_levels[0]]
            qvmrs[2] = self.inputs['atmosphere']['profile']['CO'][quench_levels[0]]
        
            qvmrs2[0] = self.inputs['atmosphere']['profile']['CO2'][quench_levels[1]] #change to quench point of CO-H2O-CH4 *JM( changed back to CO2 quench point for kinetics)

            qvmrs[3] = self.inputs['atmosphere']['profile']['NH3'][quench_levels[2]]
            qvmrs2[1] = self.inputs['atmosphere']['profile']['N2'][quench_levels[2]]

            qvmrs2[2] = self.inputs['atmosphere']['profile']['HCN'][quench_levels[3]]

            qvmrs[4]  =1- np.sum(qvmrs[:4])

            ### difference between equilibrium and quench abundances above quench points
            dq_h2o =  self.inputs['atmosphere']['profile']['H2O'][0:quench_levels[0]+1] - qvmrs[1]
            dq_ch4 =  self.inputs['atmosphere']['profile']['CH4'][0:quench_levels[0]+1] - qvmrs[0]
            dq_co  =  self.inputs['atmosphere']['profile']['CO'][0:quench_levels[0]+1]- qvmrs[2]
            dq_co2 =  self.inputs['atmosphere']['profile']['CO2'][0:quench_levels[1]+1] - qvmrs2[0]
            dq_nh3 =  self.inputs['atmosphere']['profile']['NH3'][0:quench_levels[2]+1] - qvmrs[3]
            dq_n2  =  self.inputs['atmosphere']['profile']['NH3'][0:quench_levels[2]+1] - qvmrs2[1]
            dq_hcn =  self.inputs['atmosphere']['profile']['HCN'][0:quench_levels[3]+1] - qvmrs2[2]
            # first quench ch4/co/h2o
            self.inputs['atmosphere']['profile']['CO'][0:quench_levels[0]+1] = self.inputs['atmosphere']['profile']['CO'][0:quench_levels[0]+1]*0.0 + qvmrs[2]
            self.inputs['atmosphere']['profile']['CH4'][0:quench_levels[0]+1] = self.inputs['atmosphere']['profile']['CH4'][0:quench_levels[0]+1]*0.0 + qvmrs[0]
            self.inputs['atmosphere']['profile']['H2O'][0:quench_levels[0]+1] = self.inputs['atmosphere']['profile']['H2O'][0:quench_levels[0]+1]*0.0 + qvmrs[1]

            # including new option to rainout H2O even with diseq mixing/quenching
            if vol_rainout == True:
                for i in range(0,quench_levels[0]+1): 
                    get_pvap_h2o = getattr(vj.pvaps,'H2O')
                    pvap_h2o = get_pvap_h2o(self.inputs['atmosphere']['profile']['temperature'][i])*1e-6
                    if pvap_h2o < qvmrs[1]:
                    # if dq_h2o[i] > 0:
                        self.inputs['atmosphere']['profile']['H2O'][i] = pvap_h2o

                    get_pvap_ch4 = getattr(vj.pvaps,'CH4')
                    pvap_ch4 = get_pvap_ch4(self.inputs['atmosphere']['profile']['temperature'][i])*1e-6
                    if pvap_ch4 < qvmrs[0]:
                    # if dq_ch4[i] > 0:
                        self.inputs['atmosphere']['profile']['CH4'][i] = pvap_ch4
            
            # then quench co2, changed to CO/CH4/H2O quench point *JM ( changed back to CO2 quench point for kinetics)
            self.inputs['atmosphere']['profile']['CO2'][0:quench_levels[1]+1] = self.inputs['atmosphere']['profile']['CO2'][0:quench_levels[1]+1]*0.0 + qvmrs2[0]

            # then quench nh3 and n2
            self.inputs['atmosphere']['profile']['NH3'][0:quench_levels[2]+1] = self.inputs['atmosphere']['profile']['NH3'][0:quench_levels[2]+1]*0.0 + qvmrs[3]
            
            if vol_rainout == True:
                # bug in pvap for nh3 (in my version still), temporarily using dq_nh3
                for i in range(0,quench_levels[2]+1): 
                    get_pvap_nh3 = getattr(vj.pvaps,'NH3')
                    pvap_nh3 = get_pvap_nh3(self.inputs['atmosphere']['profile']['temperature'][i])*1e-6
                    if pvap_nh3 < qvmrs[3]:
                        self.inputs['atmosphere']['profile']['NH3'][i] = pvap_nh3
                #     else:
                #         self.inputs['atmosphere']['profile']['NH3'][i] = self.inputs['atmosphere']['profile']['NH3'][i]*0.0 + qvmrs[3]
                    # if dq_nh3[i] > 0:
                    #     self.inputs['atmosphere']['profile']['NH3'][i] = self.inputs['atmosphere']['profile']['NH3'][i]*0.0 + qvmrs[3]  
            
            # # cold trap the condensibles
            if cold_trap == True:
                if cld_species is None and vol_rainout == True:
                    # print("Clouds aren't turned on, we will now only cold trap H2O, CH4, and NH3 the same species being rained out") #message got annoying being printed out
                    #H2O and CH4 have the same quench level
                    for i in range(quench_levels[0]-2, 0, -1):
                        for mol in ['H2O', 'CH4']:
                            if self.inputs['atmosphere']['profile'][mol][i] < self.inputs['atmosphere']['profile'][mol][i-1]:
                                self.inputs['atmosphere']['profile'][mol][i-1] = self.inputs['atmosphere']['profile'][mol][i]
                    #NH3 has a different quench level
                    for i in range(quench_levels[2]-2, 0, -1):
                        if self.inputs['atmosphere']['profile']['NH3'][i] < self.inputs['atmosphere']['profile']['NH3'][i-1]:
                            self.inputs['atmosphere']['profile']['NH3'][i-1] = self.inputs['atmosphere']['profile']['NH3'][i] 
                elif cld_species is None:
                    raise Exception("Clouds aren't turned on, and no rainout requested. No cold trapping is occuring")
                else:
                    for mol in cld_species:
                        if mol == 'H2O' or mol == 'CH4':
                            for i in range(quench_levels[1]-2, 0, -1):
                                if self.inputs['atmosphere']['profile'][mol][i] < self.inputs['atmosphere']['profile'][mol][i-1]:
                                    self.inputs['atmosphere']['profile'][mol][i-1] = self.inputs['atmosphere']['profile'][mol][i]
                        elif mol == 'NH3':    
                            #NH3 has a different quench level
                            for i in range(quench_levels[2]-2, 0, -1):
                                if self.inputs['atmosphere']['profile']['NH3'][i] < self.inputs['atmosphere']['profile']['NH3'][i-1]:
                                    self.inputs['atmosphere']['profile']['NH3'][i-1] = self.inputs['atmosphere']['profile']['NH3'][i] 
                        else: #NEEDS TO BE TESTED FOR WARMER CLOUDS 
                            # invert abundance to find first layer of condensation by looking for deviation from constant value
                            # inverted = self.inputs['atmosphere']['profile'][mol][::-1]

                            # need to ignore the bottom 10% of layers to avoid the changes in deep atmosphere to properly identify condensation layer
                            cutoff = int(0.1 * self.nlevel)  # Dynamically ignore bottom 10% of layers
                            # relevant_layers = inverted[:self.nlevel - cutoff]
                            # grad = np.abs(np.gradient(relevant_layers))  # Compute abundance gradient

                            # unique_vals, counts = np.unique(inverted, return_counts=True)
                            # mode_value = unique_vals[np.argmax(counts)]
                            # threshold = mode_value * 0.01 # Define a threshold for significant drop (adjustable)

                            # Find the first layer where the abundance starts to fall off
                            # cond_idx = np.where(grad > threshold)[0]
                            cond_layer = self.nlevel - cutoff #- cond_idx[0]

                            for i in range(cond_layer, 0, -1): 
                                if self.inputs['atmosphere']['profile'][mol][i] < self.inputs['atmosphere']['profile'][mol][i-1]:
                                    self.inputs['atmosphere']['profile'][mol][i-1] = self.inputs['atmosphere']['profile'][mol][i]

            self.inputs['atmosphere']['profile']['N2'][0:quench_levels[2]+1] = self.inputs['atmosphere']['profile']['N2'][0:quench_levels[2]+1]*0.0 + qvmrs2[1]

            # then quench hcn
            self.inputs['atmosphere']['profile']['HCN'][0:quench_levels[3]+1] = self.inputs['atmosphere']['profile']['HCN'][0:quench_levels[3]+1]*0.0 + qvmrs2[2]
            
                        
            # lastly quench H2 accordingly
            self.inputs['atmosphere']['profile']['H2'][0:quench_levels[0]+1] -= (dq_co + dq_ch4 + dq_h2o) 
            self.inputs['atmosphere']['profile']['H2'][0:quench_levels[1]+1] -= (dq_co2)
            self.inputs['atmosphere']['profile']['H2'][0:quench_levels[2]+1] -= (dq_nh3 + dq_n2)
            self.inputs['atmosphere']['profile']['H2'][0:quench_levels[3]+1] -= (dq_hcn)

            # include new option for CO2 in equilibrium with CO, H2O, H2 from equation 43 in Zahnle and Marley (2014)
            if kinetic_CO2 == True:
                K = 18.3*np.exp(-2376/self.inputs['atmosphere']['profile']['temperature'] - (932/self.inputs['atmosphere']['profile']['temperature'])**2)
                fCO2 = (self.inputs['atmosphere']['profile']['CO']*self.inputs['atmosphere']['profile']['H2O'])/(K*self.inputs['atmosphere']['profile']['H2'])
                
                # apply the new quench starting from original CO2 quench point
                fCO2[:quench_levels[1]] = fCO2[quench_levels[1]]
                self.inputs['atmosphere']['profile']['CO2'][:] = fCO2[:]
            
        # drop the last 10 layers that I had added on for cold cases to capture the chemistry to return to the same number of original layers
        if teff <= 250:
            self.inputs['atmosphere']['profile'] = self.inputs['atmosphere']['profile'].iloc[:-10]

            # Check if CO2 is below 1e-10, if so, zero out the values
            if self.inputs['atmosphere']['profile']['CO2'].max() < 1e-15:
                self.inputs['atmosphere']['profile']['CO2'] = self.inputs['atmosphere']['profile']['CO2']*0.0
            if self.inputs['atmosphere']['profile']['CO'].max() < 1e-15:
                self.inputs['atmosphere']['profile']['CO'] = self.inputs['atmosphere']['profile']['CO']*0.0
        #self.inputs['atmosphere']['profile'][species] = pd.DataFrame(abunds)

        return qvmrs, qvmrs2
    
    def premix_atmosphere_photochem_deprecate(self, opa, df=None, filename=None,firsttime=False,
                                    mh_interp=None,cto_interp=None, **pd_kwargs):
        """
        Builds a dataframe and makes sure that minimum necessary parameters have been suplied.
        Sets number of layers in model.  
        Parameters
        ----------
        opa : class 
            Opacity class from opannection : RetrieveCks() 
        df : pandas.DataFrame or dict
            (Optional) Dataframe with volume mixing ratios and pressure, temperature profile. 
            Must contain pressure (bars) at least one molecule
        filename : str 
            (Optional) Filename with pressure, temperature and volume mixing ratios.
            Must contain pressure at least one molecule
        exclude_mol : list of str 
            (Optional) List of molecules to ignore from file
        pd_kwargs : kwargs 
            Key word arguments for pd.read_csv to read in supplied atmosphere file 
        """
        print("PLEASE BEWARE THAT I AM LINEARLY INTERPOLATING BETWEEN SONORA CHEMISTRY TABLES")
        print("THIS IS MAINLY NEEDED FOR CAPTURING THE ALKALI METALS OUTSIDE OF THE 1D PHOTOCHEM NETWORK")

        if firsttime==True:
            print("DOING THIS FOR THE FIRST TIME SO MIGHT TAKE SOME TIME")
            
            mh_target = mh_interp
            cto_target = cto_interp
            path= os.path.join(__refdata__, 'climate_INPUTS','sonora_master_arr.npz')
            sonora_arr = np.load(path)
            mh_arr = sonora_arr['mh']
            cto_arr = sonora_arr['cto']
            sp_arr = sonora_arr['species']
            main_arr = sonora_arr['sonora']
            pressure_sonora= sonora_arr['pressure']
            temp_sonora=sonora_arr['temp']
            
            df_interp_abun=pd.DataFrame(columns=sp_arr,index=np.arange(0,1460,1),dtype='float')

            
            for i in range(len(sp_arr)):
                if np.logical_and(sp_arr[i] != 'pressure',sp_arr[i] != 'temperature'):
                    for ct in range(1460):
                        func = RegularGridInterpolator((mh_arr,cto_arr),np.log10(main_arr[:,:,ct,i]),bounds_error=False, fill_value=None)
                        df_interp_abun[sp_arr[i]][ct]= 10**func((mh_target,cto_target))
                        if df_interp_abun[sp_arr[i]][ct] <0:
                            df_interp_abun[sp_arr[i]][ct]=0.0
            df_interp_abun['pressure']= pressure_sonora
            df_interp_abun['temperature']= temp_sonora

            opa.full_abunds=df_interp_abun

        if not isinstance(df, type(None)):
            if ((not isinstance(df, dict )) & (not isinstance(df, pd.core.frame.DataFrame ))): 
                raise Exception("df must be pandas DataFrame or dictionary")
            else:
                self.nlevel=df.shape[0] 
        elif not isinstance(filename, type(None)):
            df = pd.read_csv(filename, **pd_kwargs)
            self.nlevel=df.shape[0] 

        if 'pressure' not in df.keys(): 
            raise Exception("Check column names. `pressure` must be included.")

        if ('temperature' not in df.keys()):
            raise Exception("`temperature` not specified as a column/key name")
        self.inputs['atmosphere']['profile'] = df.sort_values('pressure').reset_index(drop=True)

        #Turn off raman for 196 premix calculations 
        self.inputs['approx']['rt_params']['common']['raman'] = 2

        self.chem_interp(opa.full_abunds)

    #MAKE THIS THE DEFUALT FOR NOW FOR BACK COMAPTIBILITY
    
    def sonora(self, sonora_path, teff, chem='low'):
        """
        This queries Sonora temperature profile that can be downloaded from profiles.tar on 
        Zenodo: 

            - Bobcat Models: [profile.tar file](https://zenodo.org/record/1309035#.Xo5GbZNKjGJ)
            - Elf OWL Models: [L Type Models](https://zenodo.org/records/10385987), [T Type Models](https://zenodo.org/records/10385821), [Y Type Models](https://zenodo.org/records/10381250)

        Note gravity is not an input because it grabs gravity from self. 

        Parameters
        ----------
        sonora_path : str   
            Path to the untarred profile.tar file from sonora grid 
        teff : float 
            teff to query. does not have to be exact (it will query one the nearest neighbor)
        chem : str 
            Default = 'low'. There are two sonora chemistry grids. the default is use low. 
            There is sublety to this that is going to be explained at length in the sonora grid paper. 
            Until then, ONLY use chem='low' unless you have reached out to one of the developers. 
        """
        try: 
            g = self.inputs['planet']['gravity']/100 #m/s2 for sonora
        except AttributeError : 
            raise Exception('Oops! Looks like gravity has not been set. Can you please \
                run the gravity function to set gravity')

        flist = os.listdir(os.path.join(sonora_path))
        #ignore hidden drive files 
        flist = [i for i in flist if '._' != i[0:2]]


        if ('cmp.gz' in str(flist)):

            flist = [i.split('/')[-1] for i in flist if 'gz' in i]
            ts = [i.split('g')[0][1:] for i in flist if 'gz' in i]
            gs = [i.split('g')[1].split('nc')[0] for i in flist]

            pairs = [[ind, float(i),float(j)] for ind, i, j in zip(range(len(ts)), ts, gs)]
            coordinate = [teff, g]

            get_ind = min(pairs, key=lambda c: math.hypot(c[1]- coordinate[0], c[2]-coordinate[1]))[0]

            build_filename = 't'+ts[get_ind]+'g'+gs[get_ind]+'nc_m0.0.cmp.gz'
            if build_filename not in flist: 
                raise Exception(f"The Sonora file you are looking for {build_filename} does not exist in your specified directory {sonora_path}. Please check that it is in there.")
            ptchem = pd.read_csv(os.path.join(sonora_path,build_filename),sep=r'\s+',compression='gzip')
            ptchem = ptchem.rename(columns={'P(BARS)':'pressure',
                                            'TEMP':'temperature',
                                            'HE':'He'})
            self.nlevel = ptchem.shape[0]

            self.inputs['atmosphere']['profile'] = ptchem.loc[:,['pressure','temperature']]
        elif ('.dat' in str(flist)):
            flist = [i.split('/')[-1] for i in flist if 'dat' in i]
            ts = [i.split('g')[0][1:] for i in flist if 'dat' in i]
            gs = [i.split('g')[1].split('nc')[0] for i in flist]
            pairs = [[ind, float(i),float(j)] for ind, i, j in zip(range(len(ts)), ts, gs)]
            coordinate = [teff, g]

            get_ind = min(pairs, key=lambda c: math.hypot(c[1]- coordinate[0], c[2]-coordinate[1]))[0]
            end_file = 'nc'+flist[0].split('nc')[-1]
            build_filename = 't'+ts[get_ind]+'g'+gs[get_ind]+end_file
            pressure_bobcat,temp_bobcat = np.loadtxt(os.path.join(sonora_path,build_filename),usecols=[1,2],unpack=True, skiprows = 1)
            self.add_pt(temp_bobcat, pressure_bobcat)

        else: 
            raise Exception('Oops! Looks like the sonora path you specified does not contain any files that end in .cmp.gz or .dat. Please either: 1) untar the profile.tar file here https://zenodo.org/record/1309035#.Xo5GbZNKjGJ and point to this file path as your input. There should be around 390 files that end in cmp.gz. No need to unzip then individually. OR, 2) Alternatively you can download the structures files from the Bobcat grid located on zenodo https://zenodo.org/record/5063476#.YwPkduzMI-Q')

        if chem == 'low':
            self.channon_grid_low(filename=os.path.join(__refdata__,'chemistry','visscher_abunds_m+0.0_co1.0' ))
        elif chem=='grid':
            #solar C/O and M/H 
            self.chemeq_visscher(c_o=1.0,log_mh=0.0)
        self.inputs['atmosphere']['sonora_filename'] = build_filename


    def chemeq_visscher_2121(self, cto_absolute, log_mh):#, interp_window = 11, interp_poly=2):
        """
        Author of Data: Channon Visscher

        SONORAPY 2020 CHEMISTRY GRIDS README
        ***BETA VERSION*** Spring 2024
        Channon Visscher


        PYTHON IMPLEMENTATION OF THE SONORA CHEMICAL EQUILIBRIUM ABUNDANCE GRIDS


        Based upon easyCHEM Python package of NASA CEA code, written Elise Lei and Paul Mollière
        See https://easychem.readthedocs.io/en/latest/index.html


        This version prints the "2020" grid, which includes all grid points calculated in prior grids
        2020 data points:
        * 20 pressures, from 1e-06 to to 3e03 bar 
        * 101 temperatures, from 75 K to 6000 K


        ****************************************************************************************
        GRID NOTES
        * the full equilibrium calculation includes numerous additional species; the species
        reported here are a selected output of abundances requested for opacity calculations
        * unless calculated, gas abundances at low temperatures set to 1e-50
        * graphite condensation is included in the calculations; 
        the graphite stability field is also indicated in the contour plots
        * consideration of ion chemistry over all temperatures included
        * PH3 is adopted as the stable low-T P-bearing gas (i.e., JANAF P4O6 data); this can 
        be switched by replacing 'P4O6(JANAF)' with 'P4O6(Gurvich)' in the species list
        * This version may be considered a "minimum working example" to test for consistency
        with previous NASA CEA calculations used in SONORA equilibrium chemistry grids
        
        ****************************************************************************************
        METALLICITY VARIATIONS AND THE CARBON-TO-OXYGEN RATIO


        The C/O ratio is calculated as follows:
        1) all abundances are read-in from abundance database
        2) all elements heavier than helium multiplied by metallicity factor (10^feh)
        where feh is the metallicity in dex (i.e. feh = 1.0 is 10x solar
        3) the C/O factor is defined relative to solar (i.e. co_factor = 1 is bulk solar ratio)
        4) the new C/O ratio adjusted by keeping C + O = constant and adjusting the carbon-to-oxygen
        ratio by the co_factor


        This approach keeps the heavy-element-to-hydrogen ratio (Z/X) constant for a given [Fe/H]


        FE/H: -1.0 -0.7 -0.5 -0.3  0.0  0.5  1.0  1.3  1.7  2.0
        C/O x:
        0.25    X    0    0    0    X    0    X    0    0    0     
        0.5     X    0    0    0    X    0    X    0    0    0     
        1.0     X    0    0    0    S    0    X    0    0    0     
        1.5     X    0    0    0    X    0    X    0    0    0     
        2.0     X    0    0    0    X    0    X    0    0    0     
        2.5     X    0    0    0    X    0    X    0    0    0 




        SOLAR METALLICITY DENOTED BY 'S' ABOVE


        METALLICITY AND C/O RATIO INDICATED BY FILENAME


        output_fehxxx_coyyy.txt


        where
        xxx = feh value in dex (e.g., 0.0 for solar)
        yyy = if "x", co_factor value (e.g., 1.0 for 1x solar C/O)
        yyy = bulk c/o ratio (e.g. 0.549 for solar) - python grids

        Parameters
        ----------
        cto_absolute : int 
            carbon to oxygen ratio absolute units.
            Solar = 0.55
        log_mh : int 
            metallicity (relative to solar)
            Will find the nearest value to 0.0, 0.5, 1.0, 1.5, 1.7, 2.0
            Solar = 0
        """
        target_feh=log_mh
        target_co=cto_absolute
        directory = os.path.join(__refdata__,'chemistry','visscher_grid_2121') 

        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Regex to capture feh and co values from the filename.
        # This pattern robustly matches various float representations like:
        # -0.3, 0.55, .5, -.5, 5, -5, 5., -5., +0.7
        # Breakdown of float_pattern:
        #   [-+]?       : Optional sign (+ or -).
        #   (?:         : Start of a non-capturing group for OR logic.
        #     \d+\.?\d* : Matches numbers like "5", "5.", "5.0" (one or more digits,
        #                 optionally followed by a decimal point and zero or more digits).
        #     |         : OR
        #     \.\d+     : Matches numbers like ".5" (a decimal point followed by one or more digits).
        #   )           : End of the non-capturing group.
        float_pattern_segment = r"[-+]?(?:\d+\.?\d*|\.\d+)"
        # Construct the full filename pattern
        # Need to escape the '.' in '.txt' for regex
        filename_pattern_str = f"sonora_2121grid_feh({float_pattern_segment})_co({float_pattern_segment})\\.txt"
        file_pattern_regex = re.compile(filename_pattern_str)

        candidate_files = []

        for filename in os.listdir(directory):
            match = file_pattern_regex.match(filename)
            if match:
                try:
                    feh_val_str, co_val_str = match.groups()
                    file_feh = float(feh_val_str)
                    file_co = float(co_val_str)
                    candidate_files.append({
                        "filename": filename,
                        "feh": file_feh,
                        "co": file_co
                    })
                except ValueError as e:
                    # This might occur if the regex matches a string that float() cannot parse,
                    # though the float_pattern_segment is designed to be compatible with float().
                    raise Exception(f"Warning: Could not convert extracted feh/co from filename '{filename}' to float. "
                        f"Extracted: feh='{feh_val_str}', co='{co_val_str}'. Error: {e}")
                    continue # Skip this file

        if not candidate_files:
            raise Exception(f"No files matching the pattern '{filename_pattern_str}' found in directory '{directory}'.")

        closest_file_info = None
        min_distance = float('inf')

        for file_info in candidate_files:
            diff_feh = file_info["feh"] - target_feh
            diff_co = file_info["co"] - target_co

            distance = math.sqrt(diff_feh**2 + diff_co**2)

            if distance < min_distance:
                min_distance = distance
                closest_file_info = file_info
            elif distance == min_distance:
                # Tie-breaking logic:
                # If overall distances are equal, prefer the file whose 'feh' value
                # is closer to the target 'feh'.
                # If 'feh' differences are also equal, prefer the one whose 'co' value
                # is closer to the target 'co'.
                # If still a tie, the one encountered first (dependent on os.listdir order) is kept.
                if closest_file_info: # Ensure closest_file_info is not None
                    current_abs_diff_feh = abs(closest_file_info["feh"] - target_feh)
                    new_abs_diff_feh = abs(diff_feh) # This is abs(file_info["feh"] - target_feh)

                    if new_abs_diff_feh < current_abs_diff_feh:
                        closest_file_info = file_info
                    elif new_abs_diff_feh == current_abs_diff_feh:
                        current_abs_diff_co = abs(closest_file_info["co"] - target_co)
                        new_abs_diff_co = abs(diff_co) # This is abs(file_info["co"] - target_co)
                        if new_abs_diff_co < current_abs_diff_co:
                            closest_file_info = file_info

        if closest_file_info is None:
            # This case should ideally not be reached if candidate_files is populated.
            # It implies an issue if all candidates somehow resulted in non-comparable distances.
            raise Exception("Error: Could not determine the closest chemistry file despite having candidates. This is unexpected.")

        # Prepare pandas read_csv arguments
        full_file_path = os.path.join(directory, closest_file_info["filename"])

        try:
            header = pd.read_csv(full_file_path).keys()[0]
            cols = header.replace('T(K)','temperature').replace('P(bar)','pressure').replace('atCs','Cs').split()
            a = pd.read_csv(full_file_path,sep=r'\s+',skiprows=1,header=None, names=cols)
            a['pressure']=10**a['pressure']

        except FileNotFoundError:
            # This specific error should be rare here given the file was just listed by os.listdir,
            # but it's good practice to handle it.
            raise Exception(f"Critical Error: The file '{full_file_path}' was listed but could not be found for reading.")
        except pd.errors.EmptyDataError:
            raise Exception(f"Warning: The file '{full_file_path}' is empty.")
        except Exception as e:
            raise Exception(f"Error reading file '{full_file_path}' using pandas")

        self.chem_interp(a)

    def chemeq_visscher_1060(self, c_o, log_mh):#, interp_window = 11, interp_poly=2):
        """
        Author of Data: Channon Visscher

        Find nearest neighbor from visscher grid
        JUNE 2015
        MODELS BASED ON 1060-POINT MARLEY GRID
        GRAPHITE ACTIVITY ADDED IN TEXT FILES (AFTER OCS)
        "ABUNDANCE" INDICATES CONDENSATION CONDITION (O OR 1)
        CURRENT GRID

        FE/H: 0.0, 0.5, 1.0, 1.5, 1.7, 2.0

        C/O: 0.5X, 1.0X, 1.5X, 2.0X, 2.5X

        The *solar* carbon-to-oxygen ratio is calculated from Lodders (2010):
        
        CARBON = 7.19E6 ATOMS
        OXYGEN = 1.57E7 ATOMS
        
        This gives a "solar" C/O ratio of 0.458
         
        The C/O ratio adjusted by keeping C + O = constant and adjusting the carbon-to-oxygen ratio by a factor relative to the solar value (i.e., a factor of "1" means 1x the solar value, i.e. a C/O ratio of 0.458).
         
        This approach keeps the heavy-element-to-hydrogen ratio (Z/X) constant for a given [Fe/H]
         
        These abundances are then multiplied by the metallicity factor (10**[Fe/H]) along with every other element in the model.
        
        Parameters
        ----------
        co : int 
            carbon to oxygen ratio relative to solar.
            Solar = 1
        log_mh : int 
            metallicity (relative to solar)
            Will find the nearest value to 0.0, 0.5, 1.0, 1.5, 1.7, 2.0
            Solar = 0
        """
        #allowable cos 
        cos = np.array([0.25,0.5,1.0,1.5,2.0,2.5])
        #allowable fehs
        fehs = np.array([-0.3, 0.0,0.3,0.5,0.7,1.0,1.5,1.7,2.0])

        if log_mh > max(fehs): 
            raise Exception('Choose a log metallicity less than 2.0')
        if c_o > max(cos): 
            raise Exception('Choose a C/O less than 2.5xSolar')

        grid_co = cos[np.argmin(np.abs(cos-c_o))]
        grid_feh = fehs[np.argmin(np.abs(fehs-log_mh))]
        str_co = str(grid_co).replace('.','')
        str_fe = str(grid_feh).replace('.','').replace('-','m')

        filename = os.path.join(__refdata__,'chemistry','visscher_grid_1060',
            f'2015_06_1060grid_feh_{str_fe}_co_{str_co}.txt').replace('_m0','m0')

        header = pd.read_csv(filename).keys()[0]
        cols = header.replace('T (K)','temperature').replace('P (bar)','pressure').split()
        a = pd.read_csv(filename,sep=r'\s+',skiprows=1,header=None, names=cols)
        a['pressure']=10**a['pressure']


        self.chem_interp(a)

    chemeq_visscher = chemeq_visscher_1060
    def channon_grid_low(self, filename = None):
        """
        Interpolate from visscher grid
        """
        if isinstance(filename, type(None)):filename= os.path.join(__refdata__,'chemistry','visscher_abunds_m+0.0_co1.0')
        a = pd.read_csv(filename)
        a = a.iloc[:,1:]
        self.chem_interp(a)

    def chem_interp(self, chem_grid):
        """
        Interpolates chemistry based on dataframe input of either 1460 or 1060 grid
        This particular function needs to have all molecules as columns as well as 
        pressure and temperature
        """
        #from user input
        plevel = self.inputs['atmosphere']['profile']['pressure'].values
        tlevel =self.inputs['atmosphere']['profile']['temperature'].values
        t_inv = 1/tlevel
        p_log = np.log10(plevel)

        nc_p = chem_grid.groupby('temperature').size().values
        pressures = chem_grid['pressure'].unique()
        temps = chem_grid['temperature'].unique()
        log_abunds = np.log10(chem_grid.drop(['pressure','temperature'],axis=1))
        species = log_abunds.keys()

        #make sure to interp on log and inv array
        p_log_grid = np.unique(pressures)
        p_log_grid =np.log10(p_log_grid[p_log_grid>0])
        t_inv_grid = 1/np.array(temps)

        #Now for the temp point on either side of our atmo grid
        #first the lower interp temp
        t_low_ind = []
        for i in t_inv:
            find = np.where(t_inv_grid>i)[0]
            if len(find)==0:
                #IF T GOES BELOW THE GRID
                t_low_ind +=[0]
            else:    
                t_low_ind += [find[-1]]
        t_low_ind = np.array(t_low_ind)
        #IF T goes above the grid
        t_low_ind[t_low_ind==(len(t_inv_grid)-1)]=len(t_inv_grid)-2
        #get upper interp temp
        t_hi_ind = t_low_ind + 1 

        #now get associated temps
        t_inv_low =  np.array([t_inv_grid[i] for i in t_low_ind])
        t_inv_hi = np.array([t_inv_grid[i] for i in t_hi_ind])


        #We want the pressure points on either side of our atmo grid point
        #first the lower interp pressure
        p_low_ind = [] 
        for i in p_log:
            find = np.where(p_log_grid<=i)[0]
            if len(find)==0:
                #If P GOES BELOW THE GRID
                p_low_ind += [0]
            else: 
                p_low_ind += [find[-1]]
        p_low_ind = np.array(p_low_ind)

        #IF pressure GOES ABOVE THE GRID
        p_log_low = []
        for i in range(len(p_low_ind)): 
            ilo = p_low_ind[i]
            it = t_hi_ind[i]
            max_avail_p = np.min([ilo, nc_p[it]-3])#3 b/c using len instead of where as was done with t above
            p_low_ind[i] = max_avail_p
            p_log_low += [p_log_grid[max_avail_p]]
            
        p_log_low = np.array(p_log_low)

        #get higher pressure vals
        p_hi_ind = p_low_ind + 1 

        #now get associated pressures 
        #p_log_low =  np.array([p_log_grid[i] for i in p_low_ind])
        p_log_hi = np.array([p_log_grid[i] for i in p_hi_ind])

        #translate to full 1060/1460 account for potentially disparate number of pressures per grid point
        t_low_1060 = np.array([sum(nc_p[0:i]) for i in t_low_ind])
        t_hi_1060 = np.array([sum(nc_p[0:i]) for i in t_hi_ind])

        i_t_low_p_low =  t_low_1060 + p_low_ind #(opa.max_pc*t_low_ind)
        i_t_hi_p_low =  t_hi_1060 + p_low_ind #(opa.max_pc*t_hi_ind)
        i_t_low_p_hi = t_low_1060 + p_hi_ind
        i_t_hi_p_hi = t_hi_1060 + p_hi_ind

        t_interp = ((t_inv - t_inv_low) / (t_inv_hi - t_inv_low))[:,np.newaxis]
        p_interp = ((p_log - p_log_low) / (p_log_hi - p_log_low))[:,np.newaxis]

        log_abunds = log_abunds.values

        abunds = 10**(((1-t_interp)* (1-p_interp) * log_abunds[i_t_low_p_low,:]) +
                     ((t_interp)  * (1-p_interp) * log_abunds[i_t_hi_p_low,:]) + 
                     ((t_interp)  * (p_interp)   * log_abunds[i_t_hi_p_hi,:]) + 
                     ((1-t_interp)* (p_interp)   * log_abunds[i_t_low_p_hi,:]) ) 

        self.inputs['atmosphere']['profile'][species] = pd.DataFrame(abunds)

    def add_pt(self, T=None, P=None):
        """
        Adds temperature pressure profile to atmosphere, keeps kzz if it exists, wipes everything else out. 
        Parameters
        ----------
        T : array
            Temperature Array in Kelbin
        P : array 
            Pressure Array in bars 
        
            
        Returns
        -------
        DataFrame 
            in PICASO format
            also sets the nlevels and nlayers
            temperature : numpy.array 
                Temperature grid in Kelvin
            pressure : numpy.array
                Pressure grid in bars 
                
        """
        
        empty_dict = {}
        if not isinstance(T,type(None)):
            empty_dict['temperature']=T
            self.nlevel=len(T) 
        if not isinstance(P,type(None)):
            empty_dict['pressure']=P
            self.nlevel=len(P) 
        df = pd.DataFrame(empty_dict).sort_values('pressure').reset_index(drop=True)
        #neb trying to comment thsi as kz is tracked elsewhere now 
        #if isinstance(self.inputs['atmosphere']['profile'], pd.core.frame.DataFrame):
        #    if 'kz' in  self.inputs['atmosphere']['profile'].keys(): 
        #        df['kz'] = self.inputs['atmosphere']['profile']['kz'].values
        self.inputs['atmosphere']['profile']  = df
        
        # Return TP profile
        return #self.inputs['atmosphere']['profile'] 

    def guillot_pt(self, Teq, T_int=100, logg1=-1, logKir=-1.5, alpha=0.5,nlevel=61, p_bottom = 1.5, p_top = -6):
        """
        Creates temperature pressure profile given parameterization in Guillot 2010 TP profile
        called in fx()
        Parameters
        ----------
        Teq : float 
            equilibrium temperature 
        T_int : float 
            Internal temperature, if low (100) currently set to 100 for everything  
        kv1 : float 
            see parameterization Guillot 2010 (10.**(logg1+logKir))
        kv2 : float
            see parameterization Guillot 2010 (10.**(logg1+logKir))
        kth : float
            see parameterization Guillot 2010 (10.**logKir)
        alpha : float , optional
            set to 0.5
        nlevel : int, optional
            Number of atmospheric layers
        p_bottom : float, optional 
            Log pressure (bars) of the lower bound pressure 
        p_top : float , optional
            Log pressure (bars) of the TOA 
            
        Returns
        -------
        T : numpy.array 
            Temperature grid 
        P : numpy.array
            Pressure grid
                
        """
        kv1, kv2 =10.**(logg1+logKir),10.**(logg1+logKir)
        kth=10.**logKir

        Teff = T_int
        f = 1.0  # solar re-radiation factor
        A = 0.0  # planetary albedo
        g0 = self.inputs['planet']['gravity']/100.0 #cm/s2 to m/s2

        # Compute equilibrium temperature and set up gamma's
        T0 = Teq
        gamma1 = kv1/kth #Eqn. 25
        gamma2 = kv2/kth

        # Initialize arrays
        logtau =np.arange(-10,20,.1)
        tau =10**logtau

        #computing temperature
        T4ir = 0.75*(Teff**(4.))*(tau+(2.0/3.0))
        f1 = 2.0/3.0 + 2.0/(3.0*gamma1)*(1.+(gamma1*tau/2.0-1.0)*np.exp(-gamma1*tau))+2.0*gamma1/3.0*(1.0-tau**2.0/2.0)*special.expn(2.0,gamma1*tau)
        f2 = 2.0/3.0 + 2.0/(3.0*gamma2)*(1.+(gamma2*tau/2.0-1.0)*np.exp(-gamma2*tau))+2.0*gamma2/3.0*(1.0-tau**2.0/2.0)*special.expn(2.0,gamma2*tau)
        T4v1=f*0.75*T0**4.0*(1.0-alpha)*f1
        T4v2=f*0.75*T0**4.0*alpha*f2
        T=(T4ir+T4v1+T4v2)**(0.25)
        P=tau*g0/(kth*0.1)/1.E5
        self.nlevel=nlevel 
        logP = np.linspace(p_top,p_bottom,nlevel)
        newP = 10.0**logP
        T = np.interp(logP,np.log10(P),T)

        self.inputs['atmosphere']['profile']  = pd.DataFrame({'temperature': T, 'pressure':newP})

        # Return TP profile
        return self.inputs['atmosphere']['profile'] 
    
    def TP_line_earth(self,P,Tsfc=294.0, Psfc=1.0, gam_trop=0.18, Ptrop=0.199, 
        gam_strat=-0.045,Pstrat=0.001,nlevel=150):
        """
        Author: Mike R. Line 
        Estimates Earth's pressure-temperature profile. All default 
        values have been tuned to semi reproduce Earth's temperature 
        pressure profile.
        Parameters
        ----------
        P : array 
            Pressure array usually np.logspace(-6,2,nlevel)
        Tsfc : float,optional 
            Surface Temperature (K). Earth is 294 K 
        Psfc : float ,optional
            Surface Pressure (bar). Earth is 1 bar. 
        gam_trop : float ,optional
            Tropospheric dry lapse rate. Earth is ~0.18 
        Ptrop : float ,optional
            Tropospheric pressure. Earth 0.199 bar. 
        gam_strat : float ,optional
            Stratospheric lapse rate. Earth is -0.045
        Pstrat : float ,optional
            Stratospheric pressure (bars). Earth is 0.001 bar. /
        nlevel : int ,optional
            Number of grid levels 
        Returns 
        -------
        array 
            Temperature array (K)Psfc
        """
        #P = np.logspace(np.log10(1e-6), np.log10(100),nlevel)

        if Ptrop <= P.min(): Ptrop=P.min()
        if Pstrat <= P.min(): Pstrat=P.min()

        T=np.zeros(len(P))

        #troposphere T--adibat

        Ttrop=Tsfc*(P/Psfc)**gam_trop  #P0=sfc p, Trop T
            
        #stratosphere
        Tpause=Ttrop[P <= Ptrop ][-1]  #tropopause Temp
        PPtrop=P[P <= Ptrop ][-1]
        Tstrat=Tpause*(P/PPtrop)**gam_strat

        #merging troposphere and stratosphfere
        T[P >= Ptrop]=Ttrop[P >= Ptrop]
        T[P <= Ptrop]=Tstrat[P <= Ptrop]

        #isothermal below surface (making surface a blackbody)
        T[P >= Psfc]=T[P >= Psfc][0]

        #isothermal above "stratopause" pressure, Pstrat
        T[P<=Pstrat]=T[P<=Pstrat][-1]
        T[T<=10]=10
        T[T>=1000]=1000
         
        self.inputs['atmosphere']['profile']  = pd.DataFrame({'temperature': T, 'pressure':P})

        # Return TP profile
        return self.inputs['atmosphere']['profile'] 

    def atmosphere_3d(self, ds, regrid=True, plot=True, iz_plot=0,verbose=True): 
        """
        Checks your xarray input to make sure the necessary elements are included. If 
        requested, it will regrid your output according to what you have specified in 
        phase_angle() routine. If you have not requested a regrid, it will check to make 
        sure that the latitude/longitude grid that you have specified in your xarray
        is the same one that you have set in the phase_angle() routine. 
        
        Parameters
        ----------
        ds : xarray.DataArray
            xarray input grid (see GCM 3D input tutorials)
        regrid : bool
            If True, this will auto regrid your data, based on the input to the 
            phase_angle function you have supllied
            If False, it will skip regridding. However, this assumes that you have already 
            regridded your data to the necessary gangles and tangles. PICASO will double check 
            for you by comparing latitude/longitudes of what is in your xarray to what was computed 
            in the phase_angle function. 
        plot : bool 
            If True, this will auto output a regridded plot 
        iz_plot : int 
            Altitude index to plot if it is requested
        verbose : bool 
            If True, this will plot out messages, letting you know if your input data is being transformed 
        """
        #check 
        if not isinstance(ds, xr.core.dataset.Dataset): 
            raise Exception('PICASO has moved to only accept xarray input. Please see GCM 3D input tutorials to learn how to reformat your input. ')

        #check for temperature and pressure
        if 'temperature' not in ds: raise Exception('Must include temperature as data component')
        
        #check for pressure and change units if needed
        if 'pressure' not in ds.coords: 
            raise Exception("Must include pressure in coords and units")
        else: 
            self.nlevel = len(ds.coords['pressure'].values)
            #CONVERT PRESSURE UNIT
            unit_old = ds.coords['pressure'].attrs['units'] 
            unit_reqd = 'bar'
            if unit_old != unit_reqd: 
                if verbose: print(f'verbose=True; Converting pressure grid from {unit_old} to required unit of {unit_reqd}.')
                ds.coords['pressure'] = (
                    ds.coords['pressure'].values*u.Unit(
                        unit_old)).to('bar').value

        
        #check for latitude and longitude 
        if (('lat' not in ds.coords) or ('lon' not in ds.coords)): 
            raise Exception("""Must include "lat" and "lon" as coordinates. 
                  Please see GCM 3D input tutorials to learn how to reformat your input.""")
        else :
            lat = ds.coords['lat'].values
            len_lat = len(lat)
            lon = ds.coords['lon'].values
            len_lon = len(lon)
            nt = self.inputs['disco']['num_tangle']
            ng = self.inputs['disco']['num_gangle']
            phase = self.inputs['phase_angle']


        if regrid: 
            #cannot regrid from a course grid to a high one
            assert nt <= len(lat), f'Cannot regrid from a course grid. num_tangle={nt} and input grid has len(lat)={len_lat}'
            assert ng <= len(lon), f'Cannot regrid from a course grid. num_gangle={nt} and input grid has len(lon)={len_lon}'
            #call regridder to get to gauss angle chevychev angle grid
            if verbose: print(f'verbose=True;regrid=True; Regridding 3D output to ngangle={ng}, ntangle={nt}, with phase={phase}.')
            ds = regrid_xarray(ds, num_gangle=ng, num_tangle=nt, phase_angle=phase)
        else: 
            #check lat and lons match up
            assert np.array_equal(self.inputs['disco']['latitude']*180/np.pi,
                lat), f"""Latitudes from the GCM do not match the PICASO grid even 
                          though the number of grid points are the same. 
                          Most likely this could be that the input phase of {phase}, is 
                          different from what the regridder used prior to this function. 
                          A simple fix is to provide this function with the native 
                          GCM xarray, turn regrid=True and it will ensure the grids are 
                          the same."""
            assert np.array_equal(self.inputs['disco']['longitude']*180/np.pi,
                lon), f"""Longitude from the GCM do not match the PICASO grid even 
                          though the number of grid points are the same. 
                          Most likely this could be that the input phase of {phase}, is 
                          different from what the regridder used prior to this function. 
                          A simple fix is to provide this function with the native  
                          GCM xarray, turn regrid=True and it will ensure the grids are 
                          the same."""
        
        #if there is only one data field through a warning to the user 
        #that they need to add in chemistry before running specturm
        if len(ds.keys()) ==1: 
            if verbose: print('verbose=True;Only one data variable included. Make sure to add in chemical abundances before trying to run spectra.')

        if plot: 
            if ((ng>1) & (nt>1)):
                ds['temperature'].isel(pressure=iz_plot).plot(x='lon', y ='lat')
            elif ((ng==1) & (nt>1)):
                ds['temperature'].isel(pressure=iz_plot).plot(y ='lat')
            elif ((ng>1) & (nt==1)):
                ds['temperature'].isel(pressure=iz_plot).plot(x ='lon')

        self.inputs['atmosphere']['profile'] = ds.sortby('pressure') 

    def premix_3d(self, opa, n_cpu=1): 
        """
        You must have already ran atmosphere_3d or pre-defined an xarray gcm 
        before running this function. 

        This function will post-process sonora chemical equillibrium 
        chemistry onto your 3D grid. 

        CURRENT options 
        log m/h: 0.0, 0.5, 1.0, 1.5, 1.7, 2.0
        C/O: 0.5X, 1.0X, 1.5X, 2.0X, 2.5X

        Parameters
        ----------
        c_o : float,optional
            default = 1 (solar), options= 0.5X, 1.0X, 1.5X, 2.0X, 2.5X
        log_mh : float, optional
            default = 0 (solar), options = 0.0, 0.5, 1.0, 1.5, 1.7, 2.0
        n_cpu : int 
            Number of cpu to use for parallelization of chemistry
        """
        not_molecules = ['temperature','pressure','kz']
        pt_3d_ds = self.inputs['atmosphere']['profile'].sortby('pressure') 
        lon = pt_3d_ds.coords['lon'].values
        lat = pt_3d_ds.coords['lat'].values
        nt = len(lat)
        ng = len(lon)

        pres = pt_3d_ds.coords['pressure'].values
        self.nlevel = len(pres)
        def run_chem(ilon,ilat):
            warnings.filterwarnings("ignore")
            df = pt_3d_ds.isel(lon=ilon,lat=ilat).to_pandas(
                    ).reset_index(
                    ).drop(['lat','lon'],axis=1
                    )#.sort_values('pressure')
            #convert to 1d format
            self.inputs['atmosphere']['profile']=df
            #run chemistry, which adds chem to inputs['atmosphere']['profile']
            self.chem_interp(opa.full_abunds)
            df_w_chem = self.inputs['atmosphere']['profile']            
            return df_w_chem

        results = Parallel(n_jobs=n_cpu)(delayed(run_chem)(ilon,ilat) for ilon in range(ng) for ilat in range(nt))
        
        all_out = {imol:np.zeros((ng,nt,self.nlevel)) for imol in results[0].keys() if imol not in not_molecules}

        i = -1
        for ilon in range(ng):
            for ilat in range(nt):
                i+=1
                for imol in all_out.keys():
                    if imol not in not_molecules:
                        all_out[imol][ilon, ilat,:] = results[i][imol].values

        data_vars = {imol:(["lon", "lat","pressure"], all_out[imol],{'units': 'v/v'}) for imol in results[0].keys() if imol not in not_molecules}
        # put data into a dataset
        ds_chem = xr.Dataset(
            data_vars=data_vars,
            coords=dict(
                lon=(["lon"], lon,{'units': 'degrees'}),#required
                lat=(["lat"], lat,{'units': 'degrees'}),#required
                pressure=(["pressure"], pres,{'units': 'bar'})#required*
            ),
            attrs=dict(description="coords with vectors"),
        )

        #append input
        self.inputs['atmosphere']['profile'] = pt_3d_ds.update(ds_chem)

    def chemeq_3d(self,c_o=1.0,log_mh=0.0, n_cpu=1): 
        """
        You must have already ran atmosphere_3d or pre-defined an xarray gcm 
        before running this function. 

        This function will post-process sonora chemical equillibrium 
        chemistry onto your 3D grid. 

        CURRENT options 
        log m/h: 0.0, 0.5, 1.0, 1.5, 1.7, 2.0
        C/O: 0.5X, 1.0X, 1.5X, 2.0X, 2.5X

        Parameters
        ----------
        c_o : float,optional
            default = 1 (solar), options= 0.5X, 1.0X, 1.5X, 2.0X, 2.5X
        log_mh : float, optional
            default = 0 (solar), options = 0.0, 0.5, 1.0, 1.5, 1.7, 2.0
        n_cpu : int 
            Number of cpu to use for parallelization of chemistry
        """
        not_molecules = ['temperature','pressure','kz']
        pt_3d_ds = self.inputs['atmosphere']['profile'].sortby('pressure') 
        lon = pt_3d_ds.coords['lon'].values
        lat = pt_3d_ds.coords['lat'].values
        nt = len(lat)
        ng = len(lon)

        pres = pt_3d_ds.coords['pressure'].values
        self.nlevel = len(pres)
        def run_chem(ilon,ilat):
            warnings.filterwarnings("ignore")
            df = pt_3d_ds.isel(lon=ilon,lat=ilat).to_pandas(
                    ).reset_index(
                    ).drop(['lat','lon'],axis=1
                    )#.sort_values('pressure')
            #convert to 1d format
            self.inputs['atmosphere']['profile']=df
            #run chemistry, which adds chem to inputs['atmosphere']['profile']
            self.chemeq_visscher(c_o=1.0,log_mh=0.0)
            df_w_chem = self.inputs['atmosphere']['profile']            
            return df_w_chem

        results = Parallel(n_jobs=n_cpu)(delayed(run_chem)(ilon,ilat) for ilon in range(ng) for ilat in range(nt))
        
        all_out = {imol:np.zeros((ng,nt,self.nlevel)) for imol in results[0].keys() if imol not in not_molecules}

        i = -1
        for ilon in range(ng):
            for ilat in range(nt):
                i+=1
                for imol in all_out.keys():
                    if imol not in not_molecules:
                        all_out[imol][ilon, ilat,:] = results[i][imol].values


        data_vars = {imol:(["lon", "lat","pressure"], all_out[imol],{'units': 'v/v'}) for imol in results[0].keys() if imol not in not_molecules}
        # put data into a dataset
        ds_chem = xr.Dataset(
            data_vars=data_vars,
            coords=dict(
                lon=(["lon"], lon,{'units': 'degrees'}),#required
                lat=(["lat"], lat,{'units': 'degrees'}),#required
                pressure=(["pressure"], pres,{'units': 'bar'})#required*
            ),
            attrs=dict(description="coords with vectors"),
        )

        #append input
        self.inputs['atmosphere']['profile'] = pt_3d_ds.update(ds_chem)

    def atmosphere_4d(self, ds=None, shift=None, plot=True, iz_plot=0,verbose=True, 
        zero_point='night_transit'): 
        """
        Regrids xarray 
        
        Parameters
        ----------
        ds : xarray.DataArray
            xarray input grid (see GCM 3D input tutorials)
            Only optional if you have already defined your dataframe to 
            self.inputs['atmosphere']['profile'] 
        shift : array 
            Degrees, for each orbital `phase`, `picaso` will rotate the longitude grid `phase_i`+`shift_i`. 
            For example, for tidally locked planets, `shift`=0 at all phase angles. 
            Therefore, `shift` must be input as an array of length `n_phase`, set by phase_angle() routine. 
            Use plot=True to understand how your grid is being shifted.
        plot : bool 
            If True, this will auto output a regridded plot
        iz_plot : bool 
            pressure index to plot  
        verbose : bool 
            If True, this will plot out messages, letting you know if your input data is being transformed
        zero_point : str 
            Is your zero point "night_transit", or "secondary_eclipse"
            Default, "night_transit"
        """ 
        if isinstance(ds, type(None)):
            ds = self.inputs['atmosphere']['profile']
            if isinstance(ds, type(None)):
                raise Exception("Need to submit an xarray.DataArray because there is no input attached to self.inputs['atmosphere']['profile']")
        else: 
            #do a deep copy so that users runs dont get over written 
            ds = copy.deepcopy(ds)

        phases = self.inputs['phase_angle']

        #define shift based on user specified shift, and user specified zero point
        if isinstance(shift, type(None)):
            shift = np.zeros(len(phases))
        
        if zero_point == 'night_transit':   ## does not work for reflected case!
            if 'reflected' in self.inputs['disco']['calculation']:
                if verbose: print('Switching to zero point secondary_eclipse which is required for reflected light')
                shift=shift
            else:
                if verbose: print('The zero_point input will be deprecated in the next PICASO version as it does not work for the reflectd light case. Instead things can be reordered in the phase_curve function in justplotit.phase_curve using reorder_output keyword')                
                shift = shift + 180
        elif zero_point == 'secondary_eclipse':
            shift=shift
        else: 
            raise Exception("Do not recognize input zero point. Please specify: night_transit or secondary_eclipse")

        self.inputs['shift'] = shift

        #make sure order is correct 
        if [i for i in ds.dims] != ["lon", "lat","pressure"]:
            ds = ds.transpose("lon", "lat","pressure")

        if not isinstance(ds, xr.core.dataset.Dataset): 
            raise Exception('PICASO has moved to only accept xarray input. Please see GCM 3D input tutorials to learn how to reformat your input. ')

        #check for temperature and pressure
        if 'temperature' not in ds: raise Exception('Must include temperature as data component')
        
        #check for pressure and change units if needed
        if 'pressure' not in ds.coords: 
            raise Exception("Must include pressure in coords and units")
        else: 
            self.nlevel = len(ds.coords['pressure'].values)
            #CONVERT PRESSURE UNIT
            unit_old = ds.coords['pressure'].attrs['units'] 
            unit_reqd = 'bar'
            if unit_old != unit_reqd: 
                if verbose: print(f'verbose=True; Converting pressure grid from {unit_old} to required unit of {unit_reqd}.')
                ds.coords['pressure'] = (
                    ds.coords['pressure'].values*u.Unit(
                        unit_old)).to('bar').value

        
        #check for latitude and longitude 
        if (('lat' not in ds.coords) or ('lon' not in ds.coords)): 
            raise Exception("""Must include "lat" and "lon" as coordinates. 
                  Please see GCM 3D input tutorials to learn how to reformat your input.""")
        else :
            og_lat = ds.coords['lat'].values #degrees
            og_lon = ds.coords['lon'].values #degrees
            pres = ds.coords['pressure'].values #bars  #adding this so I can add it explicitly to ds_New below
            
        #store so we can rotate
        data_vars_og = {i:ds[i].values for i in ds.keys()}
        #run through phases and regrid each one
        new_lat_totals = []
        new_lon_totals = []
        #new_lat_totals_og = []
        new_lon_totals_og = []
        shifted_grids = {}

        # Add if calculation = reflected here:
        for i,iphase in enumerate(phases): 
            new_lat = self.inputs['disco'][iphase]['latitude']*180/np.pi#to degrees
            new_lon_og = self.inputs['disco'][iphase]['longitude']*180/np.pi#to degrees


            #Reflected case needs a step to ensure that the reflected crescent is at the correct point wrt the substellar point
            #This statement only works for 10x10 cases! I am working on expanding this to all grids soon if possible
            micro_shift = (abs(abs(new_lon_og[-1]) - abs(new_lon_og[-2])) - abs(abs(new_lon_og[0]) - abs(new_lon_og[1]))) / 2   #accounts for the difference in sizes between latxlon bins at phases!=0.
            ng = self.inputs['disco'][iphase]['num_gangle'] # phase curve shift dependent on ngangles. Ng is used below to determine correct shift paramters
            if ng == 6:
                if 68<new_lon_og[-1]< 69 and -69<new_lon_og[0]<-68 or 68<new_lon_og[0]<69 and -69<new_lon_og[-1]<-68:  # at full phase, no need for transfer. 
                    new_lon = new_lon_og
                    shift_back = 0
                elif new_lon_og[-1] > 69 and new_lon_og[0] < 0 : #for first quarter of phases 
                    new_lon_transfer = abs(new_lon_og[-1]) - abs(new_lon_og[0]) # take the difference between the first lon and the last lon at each phase
                    new_lon = new_lon_og - new_lon_transfer - micro_shift # The 'transfer' will then shift each phase to the opposite side of the dayside hemisphere. This is crucial for weighting ng and nt correctly for spectrum.
                    #add total shift statement
                    shift_back = -new_lon_transfer - micro_shift
                elif new_lon_og[-1] > 69 and new_lon_og[0] > 0:  # Second quarter of phases
                    new_lon_transfer = new_lon_og[-1] + new_lon_og[0]
                    new_lon = new_lon_og - new_lon_transfer - micro_shift
                    #print("new_lon 2nd Q", new_lon)
                    shift_back = -new_lon_transfer - micro_shift
                elif new_lon_og[-1] < -69 and new_lon_og[0] < 0: # third quarter of phases
                    new_lon_transfer = new_lon_og[-1] + new_lon_og[0]
                    new_lon = new_lon_og - new_lon_transfer + micro_shift #new_lon_transfer here is negative, so we are adding
                    shift_back = -new_lon_transfer + micro_shift
                elif new_lon_og[-1] < -69 and new_lon_og[0] > 0: #last quarter of phases 
                    new_lon_transfer = abs(new_lon_og[-1]) - abs(new_lon_og[0]) # take the difference between the first lon and the last lon at each phase
                    new_lon = new_lon_og + new_lon_transfer + micro_shift# The 'transfer' will then shift each phase to the opposite side of the dayside hemisphere. This is crucial for weighting ng and nt correctly for spectrum.
                    #add total shift statement
                    shift_back = new_lon_transfer + micro_shift
            if ng >= 10:
                if 76<new_lon_og[-1]< 77 and -77<new_lon_og[0]<-76 or 76<new_lon_og[0]<77 and -77<new_lon_og[-1]<-76:  # at full phase, no need for transfer. 
                    new_lon = new_lon_og
                    shift_back = 0
                elif new_lon_og[-1] > 77 and new_lon_og[0] < 0 : #for first quarter of phases 
                    new_lon_transfer = abs(new_lon_og[-1]) - abs(new_lon_og[0]) # take the difference between the first lon and the last lon at each phase
                    new_lon = new_lon_og - new_lon_transfer - micro_shift # The 'transfer' will then shift each phase to the opposite side of the dayside hemisphere. This is crucial for weighting ng and nt correctly for spectrum.
                    #add total shift statement
                    shift_back = -new_lon_transfer - micro_shift
                elif new_lon_og[-1] > 77 and new_lon_og[0] > 0:  # Second quarter of phases
                    new_lon_transfer = new_lon_og[-1] + new_lon_og[0]
                    new_lon = new_lon_og - new_lon_transfer - micro_shift
                    #print("new_lon 2nd Q", new_lon)
                    shift_back = -new_lon_transfer - micro_shift
                elif new_lon_og[-1] < -77 and new_lon_og[0] < 0: # third quarter of phases
                    new_lon_transfer = new_lon_og[-1] + new_lon_og[0]
                    new_lon = new_lon_og - new_lon_transfer + micro_shift #new_lon_transfer here is negative, so we are adding
                    shift_back = -new_lon_transfer + micro_shift
                elif new_lon_og[-1] < -77 and new_lon_og[0] > 0: #last quarter of phases 
                    new_lon_transfer = abs(new_lon_og[-1]) - abs(new_lon_og[0]) # take the difference between the first lon and the last lon at each phase
                    new_lon = new_lon_og + new_lon_transfer + micro_shift# The 'transfer' will then shift each phase to the opposite side of the dayside hemisphere. This is crucial for weighting ng and nt correctly for spectrum.
                    #add total shift statement
                    shift_back = new_lon_transfer + micro_shift
            
            #append new lons and lats, used to create array below this for loop
            new_lat_totals.append(new_lat)
            new_lon_totals.append(new_lon)
            new_lon_totals_og.append(new_lon_og)
            self.inputs['disco'][iphase]['longitude'] = new_lon * np.pi/180   # changing lon around requires us to re-define self.inputs as well (needed for disco geom and also for clouds_4d)
            total_shift = (iphase*180/np.pi + (shift[i] + shift_back)) % 360
            #total_shift = (iphase*180/np.pi + shift[i]) % 360
            change_zero_pt = og_lon +  total_shift
            change_zero_pt[change_zero_pt>360]=change_zero_pt[change_zero_pt>360]%360 #such that always between -180 and 180
            change_zero_pt[change_zero_pt>180]=change_zero_pt[change_zero_pt>180]%180-180 #such that always between -180 and 180
            split = np.argmin(abs(change_zero_pt + 180)) #find point where we should shift the grid
            for idata in data_vars_og.keys():
                swap1 = data_vars_og[idata][0:split,:,:]
                swap2 = data_vars_og[idata][split:,:,:]
                data = np.concatenate((swap2,swap1))
                ds[idata].values = data
            shifted_grids[iphase] = regrid_xarray(ds, latitude=new_lat, longitude=new_lon)
            
            # we need arrays that are len(phase) x len(lon regrid) as array, not list.
            # These are used to create 'lon2d' and 'lat2d', which are needed for reflected case.
            new_lat_totals_array = np.array(new_lat_totals)
            new_lon_totals_array = np.array(new_lon_totals)
            new_lon_totals_og_array = np.array(new_lon_totals_og)

        # This creates 'phase' as a coord 
        stacked_phase_grid=xr.concat(list(shifted_grids.values()), pd.Index(list(shifted_grids.keys()), name='phase'), join='override')  ## join=override gets rid of errant lon values

        # Here we are manually creating a new xarray from scratch that has 'lon2d', 'lat2d', which have 'phase' as their 2nd dimension (neeeded for reflected case)
        # This is a temporary xarray that will be used to merge data variables (created above) with our new 2d coordinates.
        # We do it this way because xarray does not like when you add dimensions to existing coordinate system. This seems to be the only work around.
        ds_New = xr.Dataset(
            data_vars=dict(
            ),
            coords=dict(
                lon2d=(["phase","lon"], new_lon_totals_array,{'units': 'degrees'}), #required. Errors when named lon
                lat2d=(["phase","lat"], new_lat_totals_array,{'units': 'degrees'}), #required
                lon2d_clouds=(["phase","lon"], new_lon_totals_og_array,{'units': 'degrees'}), #This is the original coord system. We need to conserve this for clouds_4d
                lat2d_clouds=(["phase","lat"], new_lat_totals_array,{'units': 'degrees'}), # lon2d_clouds and lat2d_clouds will be used for shift in clouds_4d. This is the only purpose of these two coords.
                pressure=(["pressure"], pres,{'units': 'bar'}) #required
            ),
            attrs=dict(description="coords with vectors"),
        )
        new_phase_grid = ds_New 
        
        # Lets use merge with compat=override (use data_vars from 1st dataset) and join=right
        # This creates an xarray with all of the variables from stacked_phase_grid (i.e., temperature and chemicals).
        # This also creates an xarray with coords named 'lon2d' and 'lat2d' (as well as 'lon' and 'lat'). 'lon2d' and 'lat2d' have 'phase' as their second dimension, which is needed when we use reflected case.
        new_phase_grid = xr.merge([stacked_phase_grid, new_phase_grid], compat='override', join='right')
    
        if plot: 
            new_phase_grid['temperature'].isel(pressure=iz_plot).plot(x='lon2d', y ='lat2d', col='phase',col_wrap=4)
            #changed lon, lat to lon2d, lat2d
        
        self.inputs['atmosphere']['profile'] = new_phase_grid

    def clouds_4d(self, ds=None, plot=True, iz_plot=0,iw_plot=0,verbose=True, calculation='reflected'): 
        """
        Regrids xarray 
        
        Parameters
        ----------
        ds : xarray.DataArray
            xarray input grid (see GCM 3D input tutorials)
            Only optional if you have already defined your dataframe to 
            self.inputs['clouds']['profile'] 
        plot : bool 
            If True, this will auto output a regridded plot
        iz_plot : bool 
            pressure index to plot  
        iw_plot : bool 
            wavelength index to plot 
        verbose : bool 
            If True, this will plot out messages, letting you know if your input data is being transformed 
        """ 
        phases = self.inputs['phase_angle']

        if isinstance(ds, type(None)):
            ds = self.inputs['clouds']['profile']
            if isinstance(ds, type(None)):
                raise Exception("Need to submit an xarray.DataArray because there is no input attached to self.inputs['clouds']['profile']")
        else: 
            ds = copy.deepcopy(ds)

        if not isinstance(ds, xr.core.dataset.Dataset): 
            raise Exception('PICASO has moved to only accept xarray input. Please see GCM 3D input tutorials to learn how to reformat your input. ')

        if 'shift' in self.inputs: 
            shift =  self.inputs['shift']
        else: 
            raise Exception('Oops! It looks like cloud_4d is being run before atmosphere_4d. Please run atmosphere_4d first so that you can speficy a shift, relative to the phase. This shift will then be used in cloud_4d.')
                
        #check for temperature and pressure
        if 'opd' not in ds: raise Exception('Must include opd as data component')
        if 'g0' not in ds: raise Exception('Must include g0 as data component')
        if 'w0' not in ds: raise Exception('Must include w0 as data component')
        
        #check for pressure and change units if needed
        if 'pressure' not in ds.coords: 
            raise Exception("Must include pressure in coords and units")
        else: 
            self.nlevel = len(ds.coords['pressure'].values)
            #CONVERT PRESSURE UNIT
            unit_old = ds.coords['pressure'].attrs['units'] 
            unit_reqd = 'bar'
            if unit_old != unit_reqd: 
                if verbose: print(f'verbose=True; Converting pressure grid from {unit_old} to required unit of {unit_reqd}.')
                ds.coords['pressure'] = (
                    ds.coords['pressure'].values*u.Unit(
                        unit_old)).to('bar').value

        #check for wavenumber coordinates 
        if 'wno' not in ds.coords: 
            raise Exception("Must include 'wno' (wavenumber) in coords and units")
        else:
            #CONVERT wavenumber UNIT if not the right units
            unit_old = ds.coords['wno'].attrs['units'] 
            unit_reqd = 'cm^(-1)'
            if unit_old != unit_reqd: 
                if verbose: print(f'verbose=True; Converting wno grid from {unit_old} to required unit of {unit_reqd}.')
                ds.coords['wno'] = (
                    ds.coords['wno'].values*u.Unit(
                        unit_old)).to('cm^(-1)').value 

        #check for latitude and longitude 
        if (('lat' not in ds.coords) or ('lon' not in ds.coords)): 
            raise Exception("""Must include "lat" and "lon" as coordinates. 
                  Please see GCM 3D input tutorials to learn how to reformat your input.""")
        else :
            og_lat = ds.coords['lat'].values #degrees
            og_lon = ds.coords['lon'].values #degrees
            pres = ds.coords['pressure'].values #bars  #adding this so I can add it explicitly to ds_New below
        if 'reflected' in calculation:
            #store so we can rotate
            data_vars_og = {i:ds[i].values for i in ds.keys()}
            #run through phases and regrid each one
            new_lat_totals = []
            new_lon_totals = []
            shifted_grids = {}
            for i,iphase in enumerate(phases): 
                new_lat = np.array(self.inputs['atmosphere']['profile']['lat2d_clouds'][i,:])#*180/np.pi
                new_lon_og = np.array(self.inputs['atmosphere']['profile']['lon2d_clouds'][i,:])#*180/np.pi
                micro_shift = (abs(abs(new_lon_og[-1]) - abs(new_lon_og[-2])) - abs(abs(new_lon_og[0]) - abs(new_lon_og[1]))) / 2   #accounts for the difference in sizes between latxlon bins at phases!=0.
                ng = self.inputs['disco'][iphase]['num_gangle'] # phase curve shift dependent on ngangles. Ng is used below to determine correct shift paramters
                if ng == 6:
                    if 68<new_lon_og[-1]< 69 and -69<new_lon_og[0]<-68 or 68<new_lon_og[0]<69 and -69<new_lon_og[-1]<-68:  # at full phase, no need for transfer. 
                        new_lon = new_lon_og
                        shift_back = 0
                    elif new_lon_og[-1] > 69 and new_lon_og[0] < 0 : #for first quarter of phases 
                        new_lon_transfer = abs(new_lon_og[-1]) - abs(new_lon_og[0]) # take the difference between the first lon and the last lon at each phase
                        new_lon = new_lon_og - new_lon_transfer - micro_shift    # The 'transfer' will then shift each phase to the opposite side of the dayside hemisphere. This is crucial for weighting ng and nt correctly for spectrum.
                        #add total shift statement
                        shift_back = -new_lon_transfer - micro_shift
                    elif new_lon_og[-1] > 69 and new_lon_og[0] > 0:  # Second quarter of phases
                        new_lon_transfer = new_lon_og[-1] + new_lon_og[0]
                        new_lon = new_lon_og - new_lon_transfer - micro_shift
                        #print("new_lon 2nd Q", new_lon)
                        shift_back = -new_lon_transfer - micro_shift
                    elif new_lon_og[-1] < -69 and new_lon_og[0] < 0: # third quarter of phases
                        new_lon_transfer = new_lon_og[-1] + new_lon_og[0]
                        new_lon = new_lon_og - new_lon_transfer + micro_shift #new_lon_transfer here is negative, so we are adding
                        shift_back = -new_lon_transfer + micro_shift # - 180
                    elif new_lon_og[-1] < -69 and new_lon_og[0] > 0: #last quarter of phases 
                        new_lon_transfer = abs(new_lon_og[-1]) - abs(new_lon_og[0]) # take the difference between the first lon and the last lon at each phase
                        new_lon = new_lon_og + new_lon_transfer + micro_shift # The 'transfer' will then shift each phase to the opposite side of the dayside hemisphere. This is crucial for weighting ng and nt correctly for spectrum.
                        #add total shift statement
                        shift_back = new_lon_transfer + micro_shift # - 180
                if ng >= 10:
                    if 76<new_lon_og[-1]< 77 and -77<new_lon_og[0]<-76 or 76<new_lon_og[0]<77 and -77<new_lon_og[-1]<-76:  # at full phase, no need for transfer. 
                        new_lon = new_lon_og
                        shift_back = 0
                    elif new_lon_og[-1] > 77 and new_lon_og[0] < 0 : #for first quarter of phases 
                        new_lon_transfer = abs(new_lon_og[-1]) - abs(new_lon_og[0]) # take the difference between the first lon and the last lon at each phase
                        new_lon = new_lon_og - new_lon_transfer - micro_shift # The 'transfer' will then shift each phase to the opposite side of the dayside hemisphere. This is crucial for weighting ng and nt correctly for spectrum.
                        #add total shift statement
                        shift_back = -new_lon_transfer - micro_shift
                    elif new_lon_og[-1] > 77 and new_lon_og[0] > 0:  # Second quarter of phases
                        new_lon_transfer = new_lon_og[-1] + new_lon_og[0]
                        new_lon = new_lon_og - new_lon_transfer - micro_shift
                        #print("new_lon 2nd Q", new_lon)
                        shift_back = -new_lon_transfer - micro_shift
                    elif new_lon_og[-1] < -77 and new_lon_og[0] < 0: # third quarter of phases
                        new_lon_transfer = new_lon_og[-1] + new_lon_og[0]
                        new_lon = new_lon_og - new_lon_transfer + micro_shift #new_lon_transfer here is negative, so we are adding
                        shift_back = -new_lon_transfer + micro_shift - 180
                    elif new_lon_og[-1] < -77 and new_lon_og[0] > 0: #last quarter of phases 
                        new_lon_transfer = abs(new_lon_og[-1]) - abs(new_lon_og[0]) # take the difference between the first lon and the last lon at each phase
                        new_lon = new_lon_og + new_lon_transfer + micro_shift # The 'transfer' will then shift each phase to the opposite side of the dayside hemisphere. This is crucial for weighting ng and nt correctly for spectrum.
                        #add total shift statement
                        shift_back = new_lon_transfer + micro_shift - 180

                new_lat_totals.append(new_lat)
                new_lon_totals.append(new_lon)
                #total_shift = (iphase*180/np.pi + (shift[i] - shift_back)) % 360
                # total_shift = (iphase*180/np.pi + (shift[i] + shift_back)) % 360
                total_shift = (iphase*180/np.pi + shift[i]) % 360 
                change_zero_pt = og_lon +  total_shift + shift_back
                change_zero_pt[change_zero_pt>360]=change_zero_pt[change_zero_pt>360]%360 #such that always between -180 and 180
                change_zero_pt[change_zero_pt>180]=change_zero_pt[change_zero_pt>180]%180-180 #such that always between -180 and 180
                split = np.argmin(abs(change_zero_pt + 180)) #find point where we should shift the grid
                for idata in data_vars_og.keys():
                    swap1 = data_vars_og[idata][0:split,:,:]
                    swap2 = data_vars_og[idata][split:,:,:]
                    data = np.concatenate((swap2,swap1))
                    ds[idata].values = data
                shifted_grids[iphase] = regrid_xarray(ds, latitude=new_lat, longitude=new_lon)

                ## we need a lon_total that is len(phase) x len(lon regrid) as ARRAY, not list
                new_lat_totals_array = np.array(new_lat_totals)
                new_lon_totals_array = np.array(new_lon_totals)

            # creates phase as a coord
            stacked_phase_grid=xr.concat(list(shifted_grids.values()), pd.Index(list(shifted_grids.keys()), name='phase'), join='override')  ## join=override gets rid of errant lon values

            # put data into a dataset
            ds_New = xr.Dataset(
                data_vars=dict(
                ),
                coords=dict(
                    lon2d=(["phase","lon"], new_lon_totals_array,{'units': 'degrees'}), #required. Errors when named lon
                    lat2d=(["phase","lat"], new_lat_totals_array,{'units': 'degrees'}), #required
                    pressure=(["pressure"], pres,{'units': 'bar'})#required*
                ),
                attrs=dict(description="coords with vectors"),
            )
            new_phase_grid = ds_New

            # Now we need to add stacked_phase_grid Data Vars to new_phase_grid, and also add Phase to coords
            
            # Lets use merge with compat=override (use data_vars from 1st dataset)
            # This adds a new, 2D coord named 'lon2d' (not 'lon') and 'lat2d' (not 'lon). Lon2d needs to be specified for phase_curve
            new_phase_grid = xr.merge([stacked_phase_grid, new_phase_grid], compat='override', join='right')

            #print(" Cloud Phase Grid XArray", new_phase_grid)

            if plot: 
                #new_phase_grid['opd'].isel(pressure=iz_plot,wno=iw_plot).plot(x='lon2d', y ='lat2d', col='phase',col_wrap=4)
                new_phase_grid['opd'].isel(pressure=iz_plot,wno=iw_plot).plot(x='lon2d', y ='lat2d', col='phase',col_wrap=4)
            
            self.inputs['clouds']['profile'] = new_phase_grid
            self.inputs['clouds']['wavenumber'] = ds.coords['wno'].values

        elif 'thermal' in calculation: # copy-paste of original clouds_4d
                #store so we can rotate
            data_vars_og = {i:ds[i].values for i in ds.keys()}
            #run through phases and regrid each one
            shifted_grids = {}
            for i,iphase in enumerate(phases): 
                new_lat = self.inputs['disco'][iphase]['latitude']*180/np.pi#to degrees
                new_lon = self.inputs['disco'][iphase]['longitude']*180/np.pi#to degrees
                total_shift = (iphase*180/np.pi + shift[i]) % 360 
                change_zero_pt = og_lon +  total_shift
                change_zero_pt[change_zero_pt>360]=change_zero_pt[change_zero_pt>360]%360 #such that always between -180 and 180
                change_zero_pt[change_zero_pt>180]=change_zero_pt[change_zero_pt>180]%180-180 #such that always between -180 and 180
                #ds.coords['lon'].values = change_zero_pt
                split = np.argmin(abs(change_zero_pt + 180)) #find point where we should shift the grid
                for idata in data_vars_og.keys():
                    swap1 = data_vars_og[idata][0:split,:,:]
                    swap2 = data_vars_og[idata][split:,:,:]
                    data = np.concatenate((swap2,swap1))
                    ds[idata].values = data
                shifted_grids[iphase] = regrid_xarray(ds, latitude=new_lat, longitude=new_lon)
            new_phase_grid=xr.concat(list(shifted_grids.values()), pd.Index(list(shifted_grids.keys()), name='phase'))

            if plot: 
                new_phase_grid['opd'].isel(pressure=iz_plot,wno=iw_plot).plot(x='lon', y ='lat', col='phase',col_wrap=4)
            
            self.inputs['clouds']['profile'] = new_phase_grid
            self.inputs['clouds']['wavenumber'] = ds.coords['wno'].values

        else:
            raise Exception("Must include 'reflected' or 'thermal' in calculation")

    def surface_reflect(self, albedo, wavenumber, old_wavenumber = None):
        """
        Set atmospheric surface reflectivity. This preps the code to run a terrestrial 
        planet. This will automatically change the run to "hardsurface", which alters 
        the lower boundary condition of the thermal_1d flux calculation.
        Parameters
        ----------
        albedo : float
            Set constant albedo for surface reflectivity 
        wavenumber : list
            The desired wavenumber grid (inverse cm) for the albedo
        old_wavenumber : list
            Original wavenumber grid (inverse cm) for the albedo which is used to interpolate onto the new wavenumber grid
        """
        if isinstance(albedo, (float, int)):
            self.inputs['surface_reflect'] = np.array([albedo]*len(wavenumber))
        elif isinstance(albedo, (list, np.ndarray)): 
            if isinstance(old_wavenumber, type(None)):
                self.inputs['surface_reflect'] = albedo
            else: 
                self.inputs['surface_reflect'] = np.interp(wavenumber, old_wavenumber, albedo)
        self.inputs['hard_surface'] = 1 #let's the code know you have a hard surface at depth
    
    def clouds_reset(self):
        """Reset cloud dict to zeros"""
        df = self.inputs['clouds']['profile']
        zeros=np.zeros(196*(self.nlevel-1))

        #add in cloud layers 
        df['g0'] = zeros
        df['w0'] = zeros
        df['opd'] = zeros
        self.inputs['clouds']['profile'] = df
    
    def clouds(self, filename = None, g0=None, w0=None, opd=None,p=None, dp=None,df =None,
               do_holes=False, fhole = None, fthin_cld = None,
               **pd_kwargs):
        """
        Cloud specification for the model. Clouds are parameterized by a single scattering albedo (w0), 
        an assymetry parameter (g0), and a total extinction per layer (opd).
        g0,w0, and opd are both wavelength and pressure dependent. Our cloud models come 
        from eddysed. Their output look something like this where 
        pressure is in bars and wavenumber is inverse cm. We will sort pressure and wavenumber before we reshape
        so the exact order doesn't matter
        pressure wavenumber opd w0 g0
        1.   1.   ... . .
        1.   2.   ... . .
        1.   3.   ... . .
        .     . ... . .
        .     . ... . .
        1.   M.   ... . .
        2.   1.   ... . .
        .     . ... . .
        N.   .  ... . .
        If you are creating your own file you have to make sure that you have a 
        **pressure** (bars) and **wavenumber**(inverse cm) column. We will use this to make sure that your cloud 
        and atmospheric profiles are on the same grid. **If there is no pressure or wavelength parameter
        we will assume that you are on the same grid as your atmospheric input, and on the 
        eddysed wavelength grid! **
        Users can also input their own fixed cloud parameters, by specifying a single value 
        for g0,w0,opd and defining the thickness and location of the cloud. 
        Parameters
        ----------
        filename : str 
            (Optional) Filename with info on the wavelength and pressure-dependent single scattering
            albedo, asymmetry factor, and total extinction per layer. Input associated pd_kwargs 
            so that the resultant output has columns named : `g0`, `w0` and `opd`. If you are not 
            using the eddysed output, you will also need a `wavenumber` and `pressure` column in units 
            of inverse cm, and bars. 
        g0 : float, list of float
            (Optional) Asymmetry factor. Can be a single float for a single cloud. Or a list of floats 
            for two different cloud layers 
        w0 : list of float 
            (Optional) Single Scattering Albedo. Can be a single float for a single cloud. Or a list of floats 
            for two different cloud layers      
        opd : list of float 
            (Optional) Total Extinction in `dp`. Can be a single float for a single cloud. Or a list of floats 
            for two different cloud layers 
        p : list of float 
            (Optional) Bottom location of cloud deck (LOG10 bars). Can be a single float for a single cloud. Or a list of floats 
            for two different cloud layers 
        dp : list of float 
            (Optional) Total thickness cloud deck above p (LOG10 bars). 
            Can be a single float for a single cloud or a list of floats 
            for two different cloud layers 
            Cloud will span 10**(np.log10(p-dp))
        df : pd.DataFrame, dict
            (Optional) Same as what would be included in the file, but in DataFrame or dict form
        do_holes : bool 
            (Optional) The dataframe or cloud that you input becomes patchy with the added two parameters fthin_cld and fhole
        fthin_cld : float  
            (Optional) Used only if do_holes=True; scales the hole (the clear part) such that fthin_cld=0 would simply be a fully clear patch
        fhole : float 
            (Optional) Used only if do_holes=True; the fraction of the clear hole such that 
            spec = (1-fhole) * cloudy_spec + fhole * clear_spec
        """
        assert hasattr(self,'nlevel'), "Please make sure to run `atmosphere` before adding clouds"

        #first complete options if user inputs dataframe or dict 
        if (not isinstance(filename, type(None)) & isinstance(df, type(None))) or (isinstance(filename, type(None)) & (not isinstance(df, type(None)))):

            if not isinstance(filename, type(None)):
                df = pd.read_csv(filename, **pd_kwargs)

            cols = df.keys()

            assert 'g0' in cols, "Please make sure g0 is a named column in cld file"
            assert 'w0' in cols, "Please make sure w0 is a named column in cld file"
            assert 'opd' in cols, "Please make sure opd is a named column in cld file"
            

            #CHECK SIZES

            #if it's a user specified pressure and wavenumber
            if (('pressure' in cols) & ('wavenumber' in cols)):
                df = df.sort_values(['pressure', 'wavenumber']).reset_index(drop=True)
                self.inputs['clouds']['wavenumber'] = df['wavenumber'].unique()
                nwave = len(self.inputs['clouds']['wavenumber'])
                nlayer = len(df['pressure'].unique())
                assert df.shape[0] == (self.nlevel-1)*nwave, "There are {0} rows in the df, which does not equal {1} layers previously specified x {2} wave pts".format(df.shape[0], self.nlevel-1, nwave) 
            
            #if its eddysed, make sure there are 196 wave points 
            else: 
                if df.shape[0] == (self.nlevel-1)*196 :
                    self.inputs['clouds']['wavenumber'] = get_cld_input_grid('wave_EGP.dat')
                elif df.shape[0] == (self.nlevel-1)*661:
                    self.inputs['clouds']['wavenumber'] = get_cld_input_grid('wave_EGP.dat',grid661=True)
                else: 
                    raise Exception( "There are {0} rows in the df, which does not equal {1} layers x 196 or 661 eddysed wave pts".format(df.shape[0], self.nlevel-1) )

            #add it to input
            self.inputs['clouds']['profile'] = df

        #first make sure that all of these have been specified
        elif None in [g0, w0, opd, p,dp]:
            raise Exception("Must either give dataframe/dict, OR a complete set of g0, w0, opd,p,dp to compute cloud profile")
        else:
            pressure_level = self.inputs['atmosphere']['profile']['pressure'].values
            pressure = np.sqrt(pressure_level[1:] * pressure_level[0:-1])#layer

            w = get_cld_input_grid('wave_EGP.dat')

            self.inputs['clouds']['wavenumber'] = w

            pressure_all =[]
            for i in pressure: pressure_all += [i]*len(w)
            wave_all = list(w)*len(pressure)

            df = pd.DataFrame({'pressure':pressure_all,
                                'wavenumber': wave_all })


            zeros=np.zeros(196*(self.nlevel-1))

            #add in cloud layers 
            df['g0'] = zeros
            df['w0'] = zeros
            df['opd'] = zeros
            #loop through all cloud layers and set cloud profile
            for ig, iw, io , ip, idp in zip(g0,w0,opd,p,dp):
                maxp = 10**ip #max pressure is bottom of cloud deck
                minp = 10**(ip-idp) #min pressure 
                df.loc[((df['pressure'] >= minp) & (df['pressure'] <= maxp)),'g0']= ig
                df.loc[((df['pressure'] >= minp) & (df['pressure'] <= maxp)),'w0']= iw
                df.loc[((df['pressure'] >= minp) & (df['pressure'] <= maxp)),'opd']= io

            self.inputs['clouds']['profile'] = df  
    
        # add in input parameters for processing patchy cloud spectra
        self.inputs['clouds']['do_holes'] = do_holes
        if do_holes == True:
            if fhole == None: raise Exception ('fhole must be float 0-1 if do_holes = True')
            if fthin_cld == None: raise Exception ('fhole must be float 0-1 if do_holes = True')
            self.inputs['clouds']['fhole'] = fhole
            self.inputs['clouds']['fthin_cld'] = fthin_cld


    def virga(self, condensates, directory,
        fsed=1, b=1, eps=1e-2, param='const', 
        mh=1, mmw=2.2, kz_min=1e5, sig=2,
        Teff=None, alpha_pressure=None, supsat=0,
        gas_mmr=None, do_virtual=False, verbose=True,do_holes=False,fhole=None,fthin_cld=None): 
        """
        Runs virga cloud code based on the PT and Kzz profiles 
        that have been added to inptus class.
        Parameters
        ----------
        condensates : str 
            Condensates to run in cloud model 
        fsed : float 
            Sedimentation efficiency coefficient
        b : float
            Denominator of exponential in sedimentation efficiency  (if param is 'exp')
        eps: float
            Minimum value of fsed function (if param=exp)
        param : str
            fsed parameterisation
            'const' (constant), 'exp' (exponential density derivation), 'pow' (power-law)
        mh : float 
            Metallicity 
        mmw : float 
            Atmospheric mean molecular weight 
        gas_mmr : dict 
            Gas MMR as a dictionary for individual gases. This allows users to override 
            virga's chemistry. E.g. {'SiO2':1e-6}
        kz_min : float
            Minimum kzz value
        sig : float 
            Width of the log normal distribution for the particle sizes 
        Teff : float, optional
            Effective temperature. If None, Teff set to temperature at 1 bar
        alpha_pressure: float, optional
            Pressure at which we want fsed=alpha for variable fsed calculation.
            If None, pressure set to the top of the atmosphere
        do_virtual : bool 
            Turn on and off the "virtual" cloud which is a cloud that forms below 
            the pressure grid defined by the user. 
        verbose : bool 
            Turn off warnings 
        clouds_kwargs : dict 
            Added so that users can add do_hole=True, with fhole and fthin_cld and it will make the virga cloud 
            patchy
        do_holes : bool
            If True, the clouds will be patchy. This is done by using the fthin_cld and fhole parameters
        fhole : float
            Fraction of the clear hole such that spec = (1-fhole) * cloudy_spec + fhole * clear_spec
        fthin_cld : float
            Scales the hole (the clear part) such that fthin_cld=0 would simply be a fully clear patch
        """
        #stages inputs for cloudy run and also get kwargs for clouds function which we run at the end of this 
        clouds_kwargs=dict(do_holes=do_holes,fhole=fhole,fthin_cld=fthin_cld)
        self.inputs['clouds']['do_holes']=do_holes
        self.inputs['clouds']['fhole']=fhole
        self.inputs['clouds']['fthin_cld']=fthin_cld
        
        if ((('temperature' not in self.inputs['atmosphere']['profile'].keys()) 
            or ('kz' not in self.inputs['atmosphere']['profile'].keys()))
            and ('climate' in self.inputs['calculation'])):
            #if there is no temprature and a user has specified clouds, then assume this is just a setup inputs function 
            #and the user does not want an actual run
            run=False
        else: 
            run=True 
        
        #if this is a climate run lets make sure we have all the right inputs set 
        if 'climate' in self.inputs['calculation']:
            self.inputs['climate']['cloudy'] = True
            virga_kwargs = dict(condensates=condensates, directory=directory,
                                                        fsed=fsed, b=b, eps=eps, param=param, 
                                                        mh=mh, mmw=mmw, kz_min=kz_min, sig=sig,
                                                        Teff=Teff, alpha_pressure=alpha_pressure, supsat=supsat,
                                                        gas_mmr=gas_mmr, do_virtual=do_virtual, verbose=verbose,
                                                        do_holes=do_holes, fthin_cld=fthin_cld,fhole=fhole)

            #passes all the virga params 
            self.inputs['climate']['virga_kwargs'] = virga_kwargs
        
        #if we are all good for a run, run virga and produce output
        if run:     
            cloud_p = vj.Atmosphere(condensates,fsed=fsed,mh=mh,
                    mmw = mmw, sig =sig, b=b, eps=eps, param=param, supsat=supsat,
                    gas_mmr=gas_mmr, verbose=verbose) 
            if 'kz' not in self.inputs['atmosphere']['profile'].keys():
                raise Exception ("Must supply kz to atmosphere/chemistry DataFrame, \
                    if running `virga` through `picaso`. This should go in the \
                    same place that you specified you pressure-temperature profile. \
                    Alternatively, you can manually add it by doing \
                    `case.inputs['atmosphere']['profile']['kz'] = KZ`")
            df = self.inputs['atmosphere']['profile'].loc[:,['pressure','temperature','kz']]
            
            cloud_p.gravity(gravity=self.inputs['planet']['gravity'],
                    gravity_unit=u.Unit(self.inputs['planet']['gravity_unit']))#
            # print('virga temp:', df['temperatures'].values)
            cloud_p.ptk(df =df, kz_min = kz_min, latent_heat = True, Teff = Teff, alpha_pressure = alpha_pressure)
            out = vj.compute(cloud_p, as_dict=True,
                            directory=directory, do_virtual=do_virtual)
            opd, w0, g0 = out['opd_per_layer'],out['single_scattering'],out['asymmetry']
            pres = out['pressure']
            wno = 1e4/out['wave']
            df = vj.picaso_format(opd, w0, g0, pressure = pres, wavenumber=wno)
            #only pass through clouds 1d if clouds are one dimension 
            self.clouds(df=df,**clouds_kwargs)
            return out
    
    def virga_3d(self, condensates, directory,
        fsed=1, mh=1, mmw=2.2,kz_min=1e5,sig=2,
        n_cpu=1,verbose=True,smooth_kz=False,full_output=False):
        """
        Runs virga cloud code based on the PT and Kzz profiles 
        that have been added to inptus class.

        Parameters
        ----------
        condensates : str 
            Condensates to run in cloud model 
        fsed : float 
            Sedimentation efficiency 
        mh : float 
            Metallicity 
        mmw : float 
            Atmospheric mean molecular weight  
        n_cpu : int 
            number cpu to parallelize
        verbose : bool 
            Print statements to help user
        smooth_kz : bool 
            If true, it uses the min_kz value and does a UnivariateSpline
            accross the kz values to smooth out the profile
        full_output : bool  
            Returns full output of virga model run
        """
        lat =self.inputs['atmosphere']['profile'].coords['lat'].values
        lon = self.inputs['atmosphere']['profile'].coords['lon'].values
        nt = len(lat)
        ng = len(lon)
        self.nlevel = len(self.inputs['atmosphere']['profile'].coords['pressure'].values)
        nlayer = self.nlevel-1

        
        
        if 'kz' not in self.inputs['atmosphere']['profile']: 
            raise Exception("Must include 'kzz' (vertical mixing) as data component")
        else:
            #CONVERT wavenumber UNIT if not the right units
            unit_old = self.inputs['atmosphere']['profile']['kz'].units 
            unit_reqd = 'cm^2/s'
            if unit_old != unit_reqd: 
                if verbose: print(f'verbose=True; Converting wno grid from {unit_old} to required unit of {unit_reqd}.')
                self.inputs['atmosphere']['profile']['kz'].values = (
                    self.inputs['atmosphere']['profile']['kz'].values*u.Unit(
                        unit_old)).to('cm^2/s').value
                self.inputs['atmosphere']['profile'].kz.attrs['units'] = unit_reqd

        ptk_3d = self.inputs['atmosphere']['profile'][['temperature','kz']] 
        def run_virga(ilon,ilat): 
            cloud_p = vj.Atmosphere(condensates,fsed=fsed,mh=mh,
                     mmw = mmw, sig =sig,verbose=verbose) 
            cloud_p.gravity(gravity=self.inputs['planet']['gravity'],
                     gravity_unit=u.Unit(self.inputs['planet']['gravity_unit']))#
            df = ptk_3d.isel(lon=ilon, lat=ilat
                            ).to_pandas(
                            ).reset_index(
                            ).drop(
                            ['lat','lon'],axis=1
                            ).sort_values('pressure')
            if smooth_kz: 
                x=np.log10(df['pressure'].values)
                y = np.log10(df['kz'].values)
                x=x[y>np.log10(kz_min)]
                y = y[y>np.log10(kz_min)]
                if len(y)<2: 
                    raise Exception(f'Not enough kz values above kz_min of {kz_min} to perform spline smoothing')
                spl = UnivariateSpline(x, y,ext=3)
                df['kz'] = spl(np.log10(df['pressure'].values))
            cloud_p.ptk(df =df, kz_min = kz_min)
            out = vj.compute(cloud_p, as_dict=True,
                              directory=directory)
            return out 

        results = Parallel(n_jobs=n_cpu)(delayed(run_virga)(ilon,ilat) for ilon in range(ng) for ilat in range(nt))
        
        wno_grid = 1e4/results[0]['wave']
        wno_grid_sorted = sorted(wno_grid)
        nwno = len(wno_grid)
        pres = results[0]['pressure']

        data_vars=dict(
                opd=(["pressure","wno","lon", "lat"], np.zeros((nlayer,nwno,ng,nt)),{'units': 'depth per layer'}),
                g0=(["pressure","wno","lon", "lat"], np.zeros((nlayer,nwno,ng,nt)),{'units': 'none'}),
                w0=(["pressure","wno","lon", "lat"], np.zeros((nlayer,nwno,ng,nt)),{'units': 'none'}),
            )

        i=0
        if full_output: all_out = {f'lat{i}':{} for i in range(ng)}
        for ig in range(ng):
            for it in range(nt):
                out = results[i];i+=1
                data_vars['opd'][1][:,:,ig,it]= out['opd_per_layer']
                data_vars['g0'][1][:,:,ig,it] = out['asymmetry']
                data_vars['w0'][1][:,:,ig,it] = out['single_scattering']
                if full_output: all_out[f'lat{it}'][f'lon{ig}'] = out

        ds_virga= xr.Dataset(
            data_vars=data_vars,
            coords=dict(
                lon=(["lon"], lon,{'units': 'degrees'}),#required
                lat=(["lat"], lat,{'units': 'degrees'}),#required
                pressure=(["pressure"], pres,{'units': 'bar'}),#required
                # wno=(["wno"], wno_grid,{'units': 'cm^(-1)'})#required for clouds
                wno=(["wno"], wno_grid_sorted,{'units': 'cm^(-1)'})#required for clouds
            ),
            attrs=dict(description="coords with vectors"),
        )

        self.inputs['clouds']['profile'] = ds_virga 
        self.inputs['clouds']['wavenumber'] = ds_virga.coords['wno'].values

        if full_output:    return all_out 
    
    def clouds_3d(self, ds, regrid=True, plot=True, iz_plot=0, iw_plot=0,
        verbose=True):
        """
        Checks your cloud xarray input to make sure the necessary elements are included. If 
        requested, it will regrid your output according to what you have specified in 
        phase_angle() routine. If you have not requested a regrid, it will check to make 
        sure that the latitude/longitude grid that you have specified in your xarray
        is the same one that you have set in the phase_angle() routine. 
        
        Parameters
        ----------
        ds : xarray.DataArray
            xarray input grid (see cloud GCM 3D input tutorials)
        regrid : bool
            If True, this will auto regrid your data, based on the input to the 
            phase_angle function you have supllied
            If False, it will skip regridding. However, this assumes that you have already 
            regridded your data to the necessary gangles and tangles. PICASO will double check 
            for you by comparing latitude/longitudes of what is in your xarray to what was computed 
            in the phase_angle function. 
        plot : bool 
            If True, this will auto output a regridded plot 
        iz_plot : int 
            Altitude index to plot if a plot is requested
        iw_plot : int 
            Wavelength index to plot if plot is requested
        verbose : bool 
            If True, this will plot out messages, letting you know if your input data is being transformed 
        """
        #tell program that clouds are 3 dimensional 
        self.inputs['clouds']['dims']='3d'

        #check 
        if not isinstance(ds, xr.core.dataset.Dataset): 
            raise Exception('PICASO has moved to only accept xarray input. Please see GCM 3D input tutorials to learn how to reformat your input. ')

        #check for cloud properties
        if 'opd' not in ds: raise Exception("Must include 'opd' (optical detph) as data component")
        if 'g0' not in ds: raise Exception("Must include 'g0' (assymetry) as data component")
        if 'w0' not in ds: raise Exception("Must include 'w0' (single scattering) as data component")
        
        #check for wavenumber coordinates 
        if 'wno' not in ds.coords: 
            raise Exception("Must include 'wno' (wavenumber) in coords and units")
        else:
            #CONVERT wavenumber UNIT if not the right units
            unit_old = ds.coords['wno'].attrs['units'] 
            unit_reqd = 'cm^(-1)'
            if unit_old != unit_reqd: 
                if verbose: print(f'verbose=True; Converting wno grid from {unit_old} to required unit of {unit_reqd}.')
                ds.coords['wno'] = (
                    ds.coords['wno'].values*u.Unit(
                        unit_old)).to('cm^(-1)').value 

        #check for pressure and change units if needed
        if 'pressure' not in ds.coords: 
            raise Exception("Must include pressure in coords and units")
        else: 
            self.nlevel = len(ds.coords['pressure'].values)
            #CONVERT PRESSURE UNIT
            unit_old = ds.coords['pressure'].attrs['units'] 
            unit_reqd = 'bar'
            if unit_old != unit_reqd: 
                if verbose: print(f'verbose=True; Converting pressure grid from {unit_old} to required unit of {unit_reqd}.')
                ds.coords['pressure'] = (
                    ds.coords['pressure'].values*u.Unit(
                        unit_old)).to('bar').value

        
        #check for latitude and longitude 
        if (('lat' not in ds.coords) or ('lon' not in ds.coords)): 
            raise Exception("""Must include "lat" and "lon" as coordinates. 
                  Please see GCM 3D input tutorials to learn how to reformat your input.""")
        else :
            lat = ds.coords['lat'].values
            len_lat = len(lat)
            lon = ds.coords['lon'].values
            len_lon = len(lon)
            nt = self.inputs['disco']['num_tangle']
            ng = self.inputs['disco']['num_gangle']
            phase = self.inputs['phase_angle']


        if regrid: 
            #cannot regrid from a course grid to a high one
            assert nt <= len(lat), f'Cannot regrid from a course grid. num_tangle={nt} and input grid has len(lat)={len_lat}'
            assert ng <= len(lon), f'Cannot regrid from a course grid. num_gangle={nt} and input grid has len(lon)={len_lon}'
            #call regridder to get to gauss angle chevychev angle grid
            if verbose: print(f'verbose=True;regrid=True; Regridding 3D output to ngangle={ng}, ntangle={nt}, with phase={phase}.')
            ds = regrid_xarray(ds, num_gangle=ng, num_tangle=nt, phase_angle=phase)
        else: 
            #check lat and lons match up
            assert np.array_equal(self.inputs['disco']['latitude']*180/np.pi,
                lat), f"""Latitudes from the GCM do not match the PICASO grid even 
                          though the number of grid points are the same. 
                          Most likely this could be that the input phase of {phase}, is 
                          different from what the regridder used prior to this function. 
                          A simple fix is to provide this function with the native 
                          GCM xarray, turn regrid=True and it will ensure the grids are 
                          the same."""
            assert np.array_equal(self.inputs['disco']['longitude']*180/np.pi,
                lon), f"""Longitude from the GCM do not match the PICASO grid even 
                          though the number of grid points are the same. 
                          Most likely this could be that the input phase of {phase}, is 
                          different from what the regridder used prior to this function. 
                          A simple fix is to provide this function with the native  
                          GCM xarray, turn regrid=True and it will ensure the grids are 
                          the same."""

        if plot: 
            if ((ng>1) & (nt>1)):
                ds['opd'].isel(pressure=iz_plot,wno=iw_plot).plot(x='lon', y ='lat')
            elif ((ng==1) & (nt>1)):
                ds['opd'].isel(pressure=iz_plot,wno=iw_plot).plot(y ='lat')
            elif ((ng>1) & (nt==1)):
                ds['opd'].isel(pressure=iz_plot,wno=iw_plot).plot(x ='lon')

        self.inputs['clouds']['profile'] = ds 
        self.inputs['clouds']['wavenumber'] = ds.coords['wno'].values
    
    def approx(self,single_phase='TTHG_ray',multi_phase='N=2',delta_eddington=True,
        raman='pollack',tthg_frac=[1,-1,2], tthg_back=-0.5, tthg_forward=1,
        p_reference=1, rt_method='toon', stream=2, toon_coefficients="quadrature",
        single_form='explicit', calculate_fluxes='off', 
        w_single_form='TTHG', w_multi_form='TTHG', psingle_form='TTHG', 
        w_single_rayleigh = 'on', w_multi_rayleigh='on', psingle_rayleigh='on', 
        get_lvl_flux = False):
        """
        This function REsets all the default approximations in the code from what is in config file.
        This means that it will rewrite what is specified via config file defaults.
        It transforms the string specificatons
        into a number so that they can be used in numba nopython routines. 

        To see the `str` cases such as `TTHG_ray` users see all the options by using the function `justdoit.single_phase_options`
        or `justdoit.multi_phase_options`, etc. 

        single_phase : str 
            Single scattering phase function approximation 
        multi_phase : str 
            Multiple scattering phase function approximation 
        delta_eddington : bool 
            Turns delta-eddington on and off
        raman : str 
            Uses various versions of raman scattering
            default is to use the pollack approximation 
        tthg_frac : list 
            Functional of forward to back scattering with the form of polynomial :
            tthg_frac[0] + tthg_frac[1]*g_b^tthg_frac[2]
            See eqn. 6 in picaso paper 
        tthg_back : float 
            Back scattering asymmetry factor gf = g_bar*tthg_back
        tthg_forward : float 
            Forward scattering asymmetry factor gb = g_bar * tthg_forward 
        p_reference : float 
            Reference pressure (bars) This is an arbitrary pressure that 
            corresponds do the user's input of radius. Usually something "at depth"
            around 1-10 bars. 
        method : str
            Toon ('toon') or spherical harmonics ('SH'). 
        stream : int 
            Two stream or four stream (options are 2 or 4). For 4 stream need to set method='SH'
        toon_coefficients: str
            Decide whether to use Quadrature ("quadrature") or Eddington ("eddington") schemes
            to define Toon coefficients in two-stream approximation (see Table 1 in Toon et al 1989)
        single_form : str 
            form of the phase function can either be written as an 'explicit' henyey greinstein 
            or it can be written as a 'legendre' expansion. Default is 'explicit'
        w_single_form : str 
            Single scattering phase function approximation for SH
        w_multi_form : str 
            Multiple scattering phase function approximation for SH
        psingle_form : str 
            Scattering phase function approximation for psingle in SH
        w_single_rayleigh : str 
            Toggle rayleigh scattering on/off for single scattering in SH
        w_multi_rayleigh : str 
            Toggle rayleigh scattering on/off for multi scattering in SH
        psingle_rayleigh : str 
            Toggle rayleigh scattering on/off for psingle in SH
        get_lvl_flux : bool 
            This parameter returns the level by level and layer by layer 
            fluxes in the full output
            Default is False
        """
        self.inputs['approx']['get_lvl_flux'] = get_lvl_flux

        self.inputs['approx']['rt_method'] = rt_method

        #common to any RT code
        if rt_method == 'toon':
                self.inputs['approx']['rt_params']['common']['stream'] = 2 # having method="Toon" and stream=4 messes up delta-eddington stuff
        else:
                self.inputs['approx']['rt_params']['common']['stream'] = stream

        self.inputs['approx']['rt_params']['common']['delta_eddington'] = delta_eddington
        self.inputs['approx']['rt_params']['common']['raman'] =  raman_options().index(raman)
        if isinstance(tthg_frac, (list, np.ndarray)):
            if len(tthg_frac) == 3:
                self.inputs['approx']['rt_params']['common']['TTHG_params']['fraction'] = tthg_frac
            else:
                raise Exception('tthg_frac should be of length=3 so that : tthg_frac[0] + tthg_frac[1]*g_b^tthg_frac[2]')
        else: 
            raise Exception('tthg_frac should be a list or ndarray of length=3')

        self.inputs['approx']['rt_params']['common']['TTHG_params']['constant_back'] = tthg_back
        self.inputs['approx']['rt_params']['common']['TTHG_params']['constant_forward']=tthg_forward

        #unique to toon 
        #eddington or quradrature
        self.inputs['approx']['rt_params']['toon']['toon_coefficients'] = toon_phase_coefficients(printout=False).index(toon_coefficients)
        self.inputs['approx']['rt_params']['toon']['multi_phase'] = multi_phase_options(printout=False).index(multi_phase)
        self.inputs['approx']['rt_params']['toon']['single_phase'] = single_phase_options(printout=False).index(single_phase)
        
        #unique to SH
        self.inputs['approx']['rt_params']['SH']['single_form'] = SH_psingle_form_options(printout=False).index(single_form)
        self.inputs['approx']['rt_params']['SH']['w_single_form'] = SH_scattering_options(printout=False).index(w_single_form)
        self.inputs['approx']['rt_params']['SH']['w_multi_form'] = SH_scattering_options(printout=False).index(w_multi_form)
        self.inputs['approx']['rt_params']['SH']['psingle_form'] = SH_scattering_options(printout=False).index(psingle_form)
        self.inputs['approx']['rt_params']['SH']['w_single_rayleigh'] = SH_rayleigh_options(printout=False).index(w_single_rayleigh)
        self.inputs['approx']['rt_params']['SH']['w_multi_rayleigh'] = SH_rayleigh_options(printout=False).index(w_multi_rayleigh)
        self.inputs['approx']['rt_params']['SH']['psingle_rayleigh'] = SH_rayleigh_options(printout=False).index(psingle_rayleigh)
        self.inputs['approx']['rt_params']['SH']['calculate_fluxes'] = SH_calculate_fluxes_options(printout=False).index(calculate_fluxes)

        self.inputs['approx']['p_reference']= p_reference
        

    def phase_curve(self, opacityclass,  full_output=False, 
        plot_opacity= False,n_cpu =1,verbose=True ): 
        """
        Run phase curve 
        Parameters
        -----------
        opacityclass : class
            Opacity class from `justdoit.opannection`
        full_output : bool 
            (Optional) Default = False. Returns atmosphere class, which enables several 
            plotting capabilities. 
        n_cpu : int 
            (Optional) Default = 1 (no parallelization). Number of cpu to parallelize calculation.
        """
        phases = self.inputs['phase_angle']
        calculation = self.inputs['disco']['calculation']
        all_geom = self.inputs['disco']
        #print("all_geom", all_geom)
        all_profiles = self.inputs['atmosphere']['profile']
        all_cld_profiles = self.inputs['clouds']['profile']

        def run_phases(iphase):
            self.inputs['phase_angle'] = iphase[1]
            self.inputs['atmosphere']['profile'] = all_profiles.isel(phase=iphase[0])

            if verbose: print("Currently computing Phase", iphase)

            self.inputs['disco'] = all_geom[iphase[1]]
            if not isinstance(all_cld_profiles, type(None)):
                self.inputs['clouds']['profile'] = all_cld_profiles.isel(phase=iphase[0])
            out = self.spectrum(opacityclass,calculation=calculation,dimension='3d',full_output=full_output)
            return out
        
        results = Parallel(n_jobs=n_cpu)(delayed(run_phases)(iphase) for iphase in enumerate(phases))
        
        #return dict such that each key is a different phase 
        return {iphase:results[i] for i,iphase in enumerate(phases)}

    def spectrum(self, opacityclass, calculation='reflected', dimension = '1d',  full_output=False, 
        plot_opacity= False, as_dict=True):
        """Run Spectrum
        Parameters
        -----------
        opacityclass : class
            Opacity class from `justdoit.opannection`
        calculation : str
            Either 'reflected' or 'thermal' for reflected light or thermal emission. 
            If running a brown dwarf, this will automatically default to thermal    
        dimension : str 
            (Optional) Dimensions of the calculation. Default = '1d'. But '3d' is also accepted. 
            In order to run '3d' calculations, user must build 3d input (see tutorials)
        full_output : bool 
            (Optional) Default = False. Returns atmosphere class, which enables several 
            plotting capabilities. 
        plot_opacity : bool 
            (Optional) Default = False, Creates pop up of the weighted opacity
        as_dict : bool 
            (Optional) Default = True. If true, returns a condensed dictionary to the user. 
            If false, returns the atmosphere class, which can be used for debugging. 
            The class is clunky to navigate so if you are consiering navigating through this, ping one of the 
            developers. 
        """
        #CHECKS 

        #if there is not star, the only picaso option to run is thermal emission
        try: 
            if self.inputs['star']['radius'] == 'nostar':
                calculation = 'thermal' 
        except KeyError: 
            pass

        #make sure phase angle has been run for reflected light
        try: 
            #phase angles dont need to be specified for thermal emission or transmission
            phase = self.inputs['phase_angle']
        except KeyError: 
            if 'reflected' not in calculation:
                self.phase_angle(0)
                phase = self.inputs['phase_angle']
            else: 
                raise Exception("Phase angle not specified. It is needed for reflected light. Please run the jdi.inputs().phase_angle() routine.")
        
        #make sure no one is running a nonzero phase with thermal emission in 1d
        if ((phase != 0) & ('thermal' in calculation) & (dimension=='1d')):
            raise Exception("Non-zero phase is not an option for this type of calculation. This includes a thermal calculation in 1 dimensions.  Unlike reflected light, thermal flux emanates from the planet in all directions regardless of phase. Thermal phase curves are computed by rotating 3D temperature maps, which can be done in PICASO using the 3d functionality.")
        
        #I don't make people add this as an input so adding a default here if it hasnt
        #been run 
        try:
            a = self.inputs['surface_reflect']
        except KeyError:
            if self.inputs.get('hard_surface',0)==1: 
                raise Exception('The user is requesting a hard_surface boundary condition but the surface reflectivity has not been set by the function surface_reflect')
            else: 
                self.inputs['surface_reflect'] = 0 
                self.inputs['hard_surface'] = 0 

            
        return picaso(self, opacityclass,dimension=dimension,calculation=calculation,
            full_output=full_output, plot_opacity=plot_opacity, as_dict=as_dict)

    def effective_temp(self, teff=None):
        """Same as T_eff with different notation


        Parameters
        ----------
        teff : float 
            (Optional) Effective temperature of Planet
        """
        return self.T_eff(teff)

    def T_eff(self, Teff=None):
        """
        Get Teff for climate run 

        Parameters
        ----------
        T_eff : float 
            (Optional) Effective temperature of Planet
        
        """
        if Teff is not None:
            self.inputs['planet']['T_eff'] = Teff
        else :
            self.inputs['planet']['T_eff'] = 0
    
    def interpret_run(self):
        print('SUMMARY')
        print('-------')
        print('Clouds:', self.inputs['climate'].get('cloudy',False))
        for i,j in self.inputs['approx']['chem_params'].items(): print(i,j)
        print('Moist Adiabat:', self.inputs['climate']['moistgrad'])

        if self.inputs['approx']['chem_params'].get('quench',True):
            kzz = self.inputs['atmosphere']['kzz'].get('constant_kzz',False)
            if isinstance(kzz,bool) :
                kzz = 'Self Consistent Treatment' 
            print('Kzz for chem:',kzz )
    
        return 
    
    def inputs_climate(self, temp_guess= None, pressure= None, rfaci = 1,nofczns = 1 ,
        nstr = None,  rfacv = None, moistgrad = False
        #deprecated and moved to atmosphere
        #photochem=False, photochem_init_args=None, sonora_abunds_photochem = False, df_sonora_photochem = None,
        #photochem_TOA_pressure = 1e-7*1e6, 
        #, 
        #deprecated and moved to virga and/or clouds 
        #fhole = None, do_holes = False, fthin_cld = None, 
        #cloudy = False, species = None, fsed = None, mieff_dir = None,
        # beta = 1, virga_param = 'const',
        #DEPRECATED and moved to atmosphere function
        #deq_rainout= False, quench_ph3 = True, no_ph3 = False, 
        #kinetic_CO2 = True, cold_trap = False,
        #mh = None, CtoO = None
        ):
        """
        Get Inputs for Climate run

        Parameters
        ----------
        temp_guess : array 
            Guess T(P) profile to begin with
        pressure : array
            Pressure Grid for climate code (this wont change on the fly)
        rfaci : float
            Default=1, Fractional contribution of thermal light in net flux
            Usually this is kept at one and then the redistribution is controlled 
            via rfacv
        nofczns : integer
            Number of guessed Convective Zones. 1 or 2
        nstr : array
            NSTR vector describes state of the atmosphere:
            0   is top layer [0]
            1   is top layer of top convective region
            2   is bottom layer of top convective region
            3   is top layer of lower radiative region
            4   is top layer of lower convective region
            5   is bottom layer of lower convective region [nlayer-1]
        rfacv : float
            Fractional contribution of reflected light in net flux.
            =0 for no stellar irradition, 
            =0.5 for full day-night heat redistribution
            =1 for dayside
        moistgrad: bool
            Moist adiabatic gradient option
        """
        if self.inputs['planet']['T_eff'] == 0.0:
            raise Exception('Need to specify Teff with jdi.input for climate run')
        if self.inputs['planet']['gravity'] == 0.0:
            raise Exception('Need to specify gravity with jdi.input for climate run')

        self.inputs['climate']['guess_temp'] = temp_guess
        self.inputs['climate']['pressure'] = pressure
        self.inputs['climate']['nstr'] = nstr
        self.inputs['climate']['nofczns'] = nofczns
        self.inputs['climate']['rfacv'] = rfacv
        self.inputs['climate']['rfaci'] = rfaci
        self.inputs['climate']['moistgrad'] = moistgrad

        self.add_pt(temp_guess, pressure)

    def photochem_init(self):
        """called in chemistry handler 
        """
        photochem_TOA_pressure = self.inputs['atmosphere']['photochem_init_args']['TOA_pressure']
        mass= self.inputs['planet']['mass']
        if np.isnan(mass): raise Exception('Photochem run is being requested but mass and radius were not supplied through gravity function')
        radius= self.inputs['planet']['radius']
        if np.isnan(radius): raise Exception('Photochem run is being requested but mass and radius were not supplied through gravity function')
        
        self.inputs['atmosphere']['photochem_init_args']["planet_mass"] =mass
        self.inputs['atmosphere']['photochem_init_args']["planet_radius"] = radius
        #this was just for picaso
        self.inputs['atmosphere']['photochem_init_args'].pop('TOA_pressure')
        photochem_init_args = self.inputs['atmosphere']['photochem_init_args']
        # Import and initialize the photochemical code.
        from .photochem import EvoAtmosphereGasGiantPicaso
        pc = EvoAtmosphereGasGiantPicaso(**photochem_init_args)
        pc.gdat.TOA_pressure_avg = photochem_TOA_pressure
        self.inputs['climate']['pc'] = pc
    
    def energy_injection(self, inject_energy = False, total_energy_injection = 0, press_max_energy = 1,
                        injection_scalehight= 1, inject_beam = False, beam_profile = 0):
        """
        Get Inputs for Energy Injection

        Parameters
        ----------
        inject_energy : bool
            If True, will inject energy into the atmosphere
        total_energy_injection : float
            Total energy injection in ergs/cm^2/s (energy injection for chapman function)
        press_max_energy : float
            Pressure for maximum energy injection in bars (for chapman function)
        injection_scaleheight : float
            Scale height ratio for energy injection (for chapman function)
        inject_beam : bool
            If True, will inject energy beam, this is meant for numerical profile input. Otherwise, will need inputs for chapman function
        beam_profile : array
            Beam profile (numerical) for energy injection (inject_beam must = True for this to be used)
        
        """

        self.inputs['climate']['inject_energy'] = inject_energy
        self.inputs['climate']['total_energy_injection'] = total_energy_injection
        self.inputs['climate']['press_max_energy'] = press_max_energy
        self.inputs['climate']['injection_scaleheight'] = injection_scalehight
        self.inputs['climate']['inject_beam'] = inject_beam
        self.inputs['climate']['beam_profile'] = beam_profile
    
    def climate(self, opacityclass, save_all_profiles = False, with_spec=False,
        save_all_kzz = False, diseq_chem = False, self_consistent_kzz =True
        ,verbose=True):#,
        #chemeq_first=True
       #deprecate: on_fly=False,gases_fly=None, as_dict=True, kz = None, 
        """
        Top Function to run the Climate Model

        Parameters
        -----------
        opacityclass : class
            Opacity class from `justdoit.opannection`
        save_all_profiles : bool or str
            If you want to save and return all iterations in the T(P) profile, True/False.
            If str, specifies a path to which all iterations are written as an HDF5 file.
        with_spec : bool 
            Runs picaso spectrum at the end to get the full converged outputs, Default=False
        save_all_kzz : bool
            If you want to save and return all iterations in the kzz profile,True/False
        diseq_chem : bool
            If you want to run `on-the-fly' mixing (takes longer),True/False
        self_consistent_kzz : bool
            If you want to run MLT in convective zones and Moses in the radiative zones
        verbose : bool  
            If True, triggers prints throughout code 
        """
        #save to user 
        all_out = {}
        
        #get necessary parameters from opacity ck-tables 
        wno = opacityclass.wno
        delta_wno = opacityclass.delta_wno
        nwno = opacityclass.nwno
        min_temp = min(opacityclass.temps)
        max_temp = max(opacityclass.temps)

        #we will extend the black body grid 30% beyond the min and max temp of the 
        #opacity grid just to be safe with the spline
        Teff = self.inputs['planet']['T_eff']
        extension = 0.3 
        #add threshold for tmin for convergence *JM
        if Teff > 300:
            tmin = min_temp*(1-extension)
        else:
            tmin = 10

        if Teff > 1600:
            tmax = 10000
        else:
            tmax = max_temp*(1+extension)

        Opagrid = namedtuple('Opagrid',['nwno','delta_wno','wno','ngauss','gauss_wts','tmin','tmax'])
        Opagrid = Opagrid(nwno, delta_wno, wno, opacityclass.ngauss,opacityclass.gauss_wts,tmin,tmax)

        nofczns = self.inputs['climate']['nofczns']
        nstr= self.inputs['climate']['nstr']

        rfaci= self.inputs['climate']['rfaci']
        
        #turn off stellar radiation if user has run "setup_nostar() function"
        if 'nostar' in self.inputs['star']['database']:
            rfacv=0.0 
            F0PI = np.zeros(nwno) + 1.0
            opacityclass.relative_flux=F0PI
        else:
            rfacv = self.inputs['climate']['rfacv']
            F0PI = opacityclass.relative_flux 

        #turn off reflected light permanently for all these runs if rfacv=0 
        if rfacv==0:compute_reflected=False
        else:compute_reflected=True
        compute_thermal = True #always true 

        all_profiles= []
        all_opd = []
        if save_all_profiles:
            save_profile = 1
        else :
            save_profile = 0

        #initial guess 
        pressure = self.inputs['climate']['pressure']
        TEMP1 = self.inputs['climate']['guess_temp']
        
        all_profiles=np.append(all_profiles,TEMP1)
        all_opd = np.append(all_opd,np.zeros(len(TEMP1)-1)) # just so the opd tracking matches the profile
        

        #adiabat info 
        t_table = self.inputs['climate']['t_table']
        p_table = self.inputs['climate']['p_table']
        grad = self.inputs['climate']['grad']
        cp = self.inputs['climate']['cp']
        moist = self.inputs['climate']['moistgrad']
        AdiabatBundle = namedtuple('AdiabatBundle', ['t_table', 'p_table', 'grad','cp'])
        AdiabatBundle = AdiabatBundle(t_table,p_table,grad,cp)

        # energy injection info
        try:
            inject_energy = self.inputs['climate']['inject_energy']
            inject_beam = self.inputs['climate']['inject_beam']
        except KeyError:
            inject_energy = False
            inject_beam = False

        if inject_energy == True:
        ## rest of these comments can be cleaned up later
        # for beam profile energy injection (numerical profiles) 
            # if inject_beam == True:
            #     beam_profile = self.inputs['climate']['beam_profile']
            #     if len(beam_profile) != len(pressure):
            #         raise Exception('Beam profile must on the same pressure grid as the climate profile')
                # wave_in = 0
                # pm = 1
                # hratio = 1
            # for chapman function energy injection.
            # else:

            #total input energy in erg/cm^2/s
            wave_in = self.inputs['climate']['total_energy_injection']
            #pressure of maximum energy injection
            pm = self.inputs['climate']['press_max_energy']
            #scale height ratio of energy injection
            hratio = self.inputs['climate']['injection_scaleheight']
            beam_profile = self.inputs['climate']['beam_profile']
            if inject_beam == True:
                if len(beam_profile) != len(pressure):
                    raise Exception('Beam profile must on the same pressure grid as the climate profile')
        else:
            wave_in = 0
            pm = 1
            hratio = 1
            beam_profile = 0
        
        InjectionBundle = namedtuple('InjectionBundle', ['inject_energy','inject_beam','wave_in', 'pm', 'hratio', 'beam_profile'])
        InjectionBundle = InjectionBundle(inject_energy, inject_beam, wave_in, pm, hratio, beam_profile)


        grav = 0.01*self.inputs['planet']['gravity'] # cgs to si
        #logmh = self.inputs['atmosphere'].get('mh',None)
        #logmh = float(logmh) if logmh is not None else 0
        #mh = 10**logmh
        sigma_sb = 0.56687e-4 # stefan-boltzmann constant
        
        col_den = 1e6*(pressure[1:] -pressure[:-1] ) / (grav/0.01) # cgs g/cm^2
        nlevel = len(pressure)
        tidal = tidal_flux(Teff, nlevel, pressure, col_den, InjectionBundle)
        if inject_energy: 
            if verbose: 
                print("Tidal Injection is Turned on. This is your new energy profile. Pressure, tidal (erg/cm3)/s:")
                for i in range(nlevel):
                    print(pressure[i],tidal[i])
        # old tidal flux calculation without energy injection function
        # tidal = np.zeros_like(pressure) - sigma_sb *(Teff**4)

        # cloud inputs
        cloudy = self.inputs['climate'].get('cloudy',False)

        #kzz treatment ? lets store a constant kz profile if it exists 
        #DO I NEED A KZZ? 
        need_kzz = cloudy or diseq_chem 
        if need_kzz: 
            #lets initiative a separate place to store this 
            self.inputs['atmosphere']['kzz']={}

        if not self_consistent_kzz:
            kzz = self.inputs['atmosphere']['profile'].get('kz',False)
            if isinstance(kzz, bool):
                raise Exception("""self_consistent_kzz=False but no kzz profile was supplised. Please add to self.inputs['atmosphere']['profile'] """)
            else: 
                self.inputs['atmosphere']['kzz']['constant_kzz'] = kzz.values
        else: 
                self.inputs['atmosphere']['kzz']['sc_kzz'] = 0 #placeholder


        #virga inputs 
        virga_kwargs = self.inputs['climate'].get('virga_kwargs',{})
        
        #now these are in virga kwargs!!! 
        #do_holes = self.inputs['climate']['do_holes']
        #fhole = self.inputs['climate']['fhole']
        #fthin_cld = self.inputs['climate']['fthin_cld']

        # check the dimensions of the mieff grid  
        if cloudy:
            mieff_dir = virga_kwargs.get('directory',None)
            if mieff_dir is None:
                raise Exception('Need to specify directory for cloudy runs via Virga function')
            # get_clouds should reinterpolate so it is okay that this isnt on the same grid but need to get the size 
            # check if the mieff file is on 661 grid
            miefftest = os.path.join(mieff_dir, [f for f in os.listdir(mieff_dir) if f.endswith('.mieff')][0])
            with open(miefftest, 'r') as file:
                nwno_clouds = int(float(file.readline().split()[0]))
        else: 
            nwno_clouds = nwno
            #if diseq_chem and not chemeq_first and gridsize != 661:
            #    raise Exception('Mieff grid is not on 661 grid.')
            #raise warning temporarily until I can think of the best way to handle this
            #if diseq_chem and chemeq_first and gridsize == 661:
            #    raise Exception('Currently cannot do chemical equilibrium first for disequilibrium runs with clouds')

        #scattering properties 
        opd_cld_climate = np.zeros(shape=(self.nlevel-1,nwno_clouds,4))
        g0_cld_climate = np.zeros(shape=(self.nlevel-1,nwno_clouds,4))
        w0_cld_climate = np.zeros(shape=(self.nlevel-1,nwno_clouds,4))
        #BUNDLING
        CloudParametersT = namedtuple('CloudParameters',['cloudy', 'OPD','G0','W0']+list(virga_kwargs.keys()))
        #this adds the cloud params that are always needed plus the virga kwargs, if they are used 
        CloudParameters=CloudParametersT(*([cloudy, opd_cld_climate,g0_cld_climate,w0_cld_climate,
                                        ]+list(virga_kwargs.values())))


        if verbose: self.interpret_run()

        if not diseq_chem:#chemeq_first: 
            final_conv_flag, pressure, temp, dtdp, nstr_new, flux_net_ir_final, flux_net_v_final, flux_plus_final,   \
                chem_out,cld_out,  all_profiles,  all_opd,all_kzz=run_chemeq_climate_workflow(self,
                    nofczns,nstr, #tracks convective zones 
                    TEMP1,pressure, #Atmosphere
                    AdiabatBundle, #t_table, p_table, grad, cp, 
                    opacityclass, grav, 
                    rfaci, rfacv,  tidal, #energy balance 
                    Opagrid, #delta_wno, tmin, tmax, 
                    CloudParameters,#cloudy,cld_species,mh,fsed,beta,param_flag,mieff_dir ,opd_cld_climate,g0_cld_climate,w0_cld_climate, #scattering/cloud properties 
                    save_profile,all_profiles, all_opd,
                    verbose=verbose, moist = moist,
                    save_kzz=save_all_kzz, self_consistent_kzz=self_consistent_kzz)


        if diseq_chem: 
            final_conv_flag, pressure, temp, dtdp, nstr_new, flux_net_ir_final, flux_net_v_final, flux_plus_final,   \
                chem_out,cld_out,  all_profiles,  all_opd, all_kzz = run_diseq_climate_workflow(self, nofczns, nstr, TEMP1, pressure,
                        AdiabatBundle,opacityclass,
                        grav,
                        rfaci,rfacv,tidal,
                        Opagrid,
                        CloudParameters,
                        save_profile,all_profiles,all_opd,
                        verbose=verbose, moist = moist, 
                        save_kzz=save_all_kzz, self_consistent_kzz=self_consistent_kzz)
        #all output to user
        all_out['pressure'] = pressure
        all_out['temperature'] = temp
        all_out['ptchem_df'] = chem_out
        all_out['dtdp'] = dtdp
        all_out['cvz_locs'] = nstr_new
        all_out['flux_ir_attop']=flux_plus_final
        flux_net_final = rfacv * flux_net_v_final + rfaci* flux_net_ir_final + tidal
        all_out['fnet/fnetir']=flux_net_final/flux_net_ir_final
        all_out['converged']=final_conv_flag
        all_out['flux_balance']=dict(flux_net_ir=flux_net_ir_final, 
                                    flux_net_v = flux_net_v_final,
                                    tidal=tidal,
                                    rfacv=rfacv,rfaci=rfaci,
                                    flux_net = flux_net_final)


        #put cld output in all_out
        if cloudy:
            df_cld = vj.picaso_format(cld_out['opd_per_layer'],cld_out['single_scattering'],cld_out['asymmetry'], 
                                      pressure = cld_out['pressure'], wavenumber=1e4/cld_out['wave'])
            all_out['cld_df'] = df_cld
            all_out['virga_output'] = cld_out
            #all_out['cld_output_final'] = df_cld_final

        if save_all_profiles: 
            all_out['all_profiles'] = all_profiles 
            all_out['all_opd'] = all_opd
            all_out['all_kzz'] = all_kzz

        if with_spec:
            #these inputs here are just to make sure that we know what we ran as we are directly inputting a dataframe
            self.atmosphere(df=chem_out,quench = self.inputs['approx']['chem_params']['quench'], 
                                        no_ph3 = self.inputs['approx']['chem_params']['no_ph3'],
                                        cold_trap = self.inputs['approx']['chem_params']['cold_trap'], 
                                        vol_rainout= self.inputs['approx']['chem_params']['vol_rainout'])
            if cloudy == 1:
                cld_kwargs =dict( do_holes=virga_kwargs.get('do_holes',False), 
                                  fhole = virga_kwargs.get('fhole',0),
                                  fthin_cld = virga_kwargs.get('fthin_cld',0))
                self.clouds(df=df_cld,**cld_kwargs)
            df_spec = self.spectrum(opacityclass,full_output=True,calculation='thermal')    
            all_out['spectrum_output'] = df_spec 

        #suggest retiring this and always returning dict 
        return all_out


def get_targets():
    """Function to grab available targets using exoplanet archive data. 

    Returns
    -------
    Dataframe from Exoplanet Archive
    """
    planets_df =  pd.read_csv('https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+PSCompPars&format=csv')
    # convert to float when possible
    for i in planets_df.columns: 
        planets_df[i] = planets_df[i].astype(float,errors='ignore')

    return planets_df
def load_planet(df, opacity, phase_angle = 0, stellar_db='ck04models', verbose=False,  **planet_kwargs):
    """
    Wrapper to simplify PICASO run. This really turns picaso into a black box. This was created 
    specifically Sagan School tutorial. It grabs planet parameters from the user supplied df, then 
    adds a parametric PT profile (Guillot et al. 2010). The user still has to add chemistry. 

    Parameters
    -----------
    df : pd.DataFrame
        This is single row from `all_planets()`
    opacity : np.array 
        Opacity loaded from opannection
    phase_angle : float 
        Observing phase angle (radians)
    stellar_db : str 
        Stellar database to pull from. Default is ck04models but you can also 
        use phoenix if you have those downloaded.
    verbose : bool , options
        Print out warnings 
    planet_kwargs : dict 
        List of parameters to supply NexSci database is values don't exist 
    """
    if len(df.index)>1: raise Exception("Dataframe consists of more than 1 row. Make sure to select single planet")
    if len(df.index)==0: raise Exception("No planets found in database. Check name.")

    for i in df.index:

        planet = df.loc[i,:].dropna().to_dict()

        temp = planet.get('st_teff', planet_kwargs.get('st_teff',np.nan))
        if np.isnan(temp) : raise Exception('Stellar temperature is not added to \
            dataframe input or to planet_kwargs through the column/key named st_teff. Please add it to one of them')

        logg = planet.get('st_logg', planet_kwargs.get('st_logg',np.nan))
        if np.isnan(logg) : raise Exception('Stellar logg is not added to \
            dataframe input or to planet_kwargs through the column/key named st_logg. Please add it to one of them')

        logmh = planet.get('st_metfe', planet_kwargs.get('st_metfe',np.nan))
        if np.isnan(logmh) : raise Exception('Stellar Fe/H is not added to \
            dataframe input or to planet_kwargs through the column/key named st_metfe. Please add it to one of them')


        if logmh > 0.5: 
            if verbose: print ('Stellar M/H exceeded max value of 0.5. Value has been reset to the maximum')
            logmh = 0.5
        elif logmh < -4.0 :
            if verbose: print ('Stellar M/H exceeded min value of -4.0 . Value has been reset to the mininum')
            logmh = -4.0 

        if logg > 4.5: 
            if verbose: print ('Stellar logg exceeded max value of 4.5. Value has been reset to the maximum')
            logg = 4.5   


        #the parameters
        #planet/star system params--typically not free parameters in retrieval
        # Planet radius in Jupiter Radii--this will be forced to be 10 bar radius--arbitrary (scaling to this is free par)

        Rp = planet.get('pl_radj', planet_kwargs.get('pl_radj',np.nan))
        if np.isnan(Rp) : raise Exception('Planet Radii is not added to \
            dataframe input or to planet_kwargs through the column/key named pl_radj. J for JUPITER! \
            Please add it to one of them')


        #Stellar Radius in Solar Radii
        Rstar = planet.get('st_rad', planet_kwargs.get('st_rad',np.nan))
        if np.isnan(Rstar) : raise Exception('Stellar Radii is not added to \
            dataframe input or to planet_kwargs through the column/key named st_rad. Solar radii! \
            Please add it to one of them')

        #Mass in Jupiter Masses
        Mp = planet.get('pl_bmassj', planet_kwargs.get('pl_bmassj',np.nan))
        if np.isnan(Mp) : raise Exception('Planet Mass is not added to \
            dataframe input or to planet_kwargs through the column/key named pl_bmassj. J for JUPITER! \
            Please add it to one of them')  

        #TP profile params (3--Guillot 2010, Parmentier & Guillot 2013--see Line et al. 2013a for implementation)
        Tirr=planet.get('pl_eqt', planet_kwargs.get('pl_eqt',np.nan))

        if np.isnan(Tirr): 
            p =  planet.get('pl_orbper', planet_kwargs.get('pl_orbper',np.nan))
            p = p * (1*u.day).to(u.yr).value #convert to year 
            a =  (p**(2/3)*u.au).to(u.R_sun).value
            temp = planet.get('st_teff', planet_kwargs.get('st_teff',np.nan))
            Tirr = temp * np.sqrt(Rstar/(2*a))

        if np.isnan(Tirr): raise Exception('Planet Eq Temp is not added to \
            dataframe input or to planet_kwargs through the column/key named pl_eqt. Kelvin \
            Please add it to one of them') 

        p=planet.get('pl_orbper', planet_kwargs.get('pl_orbper',np.nan))

        if np.isnan(Tirr): raise Exception('Orbital Period is not added to \
            dataframe input or to planet_kwargs through the column/key named pl_orbper. Days Units') 
        else: 
            p = p * (1*u.day).to(u.yr).value #convert to year 
            a =  p**(2/3) #semi major axis in AU


        #setup picaso
        start_case = inputs()
        start_case.approx(raman="none")
        start_case.phase_angle(phase_angle) #radians 

        #define gravity
        start_case.gravity(mass=Mp, mass_unit=u.Unit('M_jup'),
                            radius=Rp, radius_unit=u.Unit('R_jup')) #any astropy units available
        


        #define star
        start_case.star(opacity, temp,logmh,logg,radius=Rstar, radius_unit=u.Unit('R_sun'),
                            semi_major=a, semi_major_unit=u.Unit('au'),
                            database = stellar_db ) #opacity db, pysynphot database, temp, metallicity, logg

        ##running this with all default inputs (users can override whatever after this initial run)
        start_case.guillot_pt(Tirr) 

    return start_case
def jupiter_pt():
    """Function to get Jupiter's PT profile"""
    return os.path.join(__refdata__, 'base_cases','jupiter.pt')
def jupiter_cld():
    """Function to get rough Jupiter Cloud model with fsed=3"""
    return os.path.join(__refdata__, 'base_cases','jupiterf3.cld')
def HJ_pt():
    """Function to get Jupiter's PT profile"""
    return os.path.join(__refdata__, 'base_cases','HJ.pt')
def HJ_pt_3d(as_xarray=False,add_kz=False, input_file = os.path.join(__refdata__, 'base_cases','HJ_3d.pt')):
    """Function to get Jupiter's PT profile
    
    Parameters
    ----------
    as_xarray : bool 
        Returns as xarray, instead of dictionary
    add_kz : bool 
        Returns kzz along with PT info
    input_file : str 
        point to input file in the same format as mitgcm example 
        file in base_cases/HJ_3d.pt
    """
    #input_file = os.path.join(__refdata__, 'base_cases','HJ_3d.pt')
    threed_grid = pd.read_csv(input_file,sep=r'\s+',names=['p','t','k'])
    all_lon= threed_grid.loc[np.isnan(threed_grid['k'])]['p'].values
    all_lat=  threed_grid.loc[np.isnan(threed_grid['k'])]['t'].values
    latlong_ind = np.concatenate((np.array(threed_grid.loc[np.isnan(threed_grid['k'])].index),[threed_grid.shape[0]] ))
    threed_grid = threed_grid.dropna() 

    lon = np.unique(all_lon)
    lat = np.unique(all_lat)

    nlon = len(lon)
    nlat = len(lat)
    total_pts = nlon*nlat
    nz = latlong_ind[1] - 1 

    p = np.zeros((nlon,nlat,nz))
    t = np.zeros((nlon,nlat,nz))
    kzz = np.zeros((nlon,nlat,nz))

    for i in range(len(latlong_ind)-1):

        ilon = list(lon).index(all_lon[i])
        ilat = list(lat).index(all_lat[i])

        p[ilon, ilat, :] = threed_grid.loc[latlong_ind[i]:latlong_ind[i+1]]['p'].values
        t[ilon, ilat, :] = threed_grid.loc[latlong_ind[i]:latlong_ind[i+1]]['t'].values
        kzz[ilon, ilat, :] = threed_grid.loc[latlong_ind[i]:latlong_ind[i+1]]['k'].values
    
    gcm_out = {'pressure':p, 'temperature':t, 'kzz':kzz, 'latitude':lat, 'longitude':lon}
    if as_xarray:
        # create data
        data = gcm_out['temperature']

        # create coords
        lon = gcm_out['longitude']
        lat = gcm_out['latitude']
        pres = gcm_out['pressure'][0,0,:]

        # put data into a dataset
        if add_kz:
            data_vars = dict(
                temperature=(["lon", "lat","pressure"], data,{'units': 'Kelvin'}),#, required
                kz = (["lon", "lat","pressure"], kzz,{'units': 'm^2/s'})
            )
        else: 
            data_vars = dict(temperature=(["lon", "lat","pressure"], data,{'units': 'Kelvin'}))

        ds = xr.Dataset(
            data_vars=data_vars,
            coords=dict(
                lon=(["lon"], lon,{'units': 'degrees'}),#required
                lat=(["lat"], lat,{'units': 'degrees'}),#required
                pressure=(["pressure"], pres,{'units': 'bar'})#required*
            ),
            attrs=dict(description="coords with vectors"),
        )
        return  ds
    else: 
        return gcm_out
def HJ_cld():
    """Function to get rough Jupiter Cloud model with fsed=3"""
    return os.path.join(__refdata__, 'base_cases','HJ.cld')
def brown_dwarf_pt():
    """Function to get rough Brown Dwarf climate model with Teff=1270 K M/H=1xSolar, C/O=1xSolar, fsed=1"""
    return os.path.join(__refdata__, 'base_cases','t1270g200f1_m0.0_co1.0.cmp')    
def brown_dwarf_cld():
    """Function to get rough Brown Dwarf cloud model with Teff=1270 K M/H=1xSolar, C/O=1xSolar, fsed=1"""
    return os.path.join(__refdata__, 'base_cases','t1270g200f1_m0.0_co1.0.cld')    
def w17_data():
    """Function to get WASP-17 Grant et al data from here 
        https://zenodo.org/records/8360121/files/ExoTiC-MIRI.zip?download=1
    """
    return os.path.join(__refdata__, 'base_cases','Grant_etal_transmission_spectrum_vfinal_bin0.25_utc20230606_125313.nc')


def single_phase_options(printout=True):
    """Retrieve all the options for direct radation"""
    if printout: print("Can also set functional form of forward/back scattering in approx['TTHG_params']")
    return ['cahoy','OTHG','TTHG','TTHG_ray']
def multi_phase_options(printout=True):
    """Retrieve all the options for multiple scattering radiation"""
    if printout: print("Can also set delta_eddington=True/False in approx['delta_eddington']")
    return ['N=2','N=1','isotropic']
def SH_scattering_options(printout=True):
    """Retrieve all the options for scattering radiation in SH"""
    return  ["TTHG","OTHG","isotropic"]
def SH_rayleigh_options(printout=True):
    """Retrieve options for rayleigh scattering"""
    return ['off','on']
def SH_psingle_form_options(printout=True):
    """Retrieve options for direct scattering form approximation"""
    return  ["explicit","legendre"]
def SH_calculate_fluxes_options(printout=True):
    """Retrieve options for calculating layerwise fluxes"""
    return  ["off","on"]
def raman_options():
    """Retrieve options for raman scattering approximtions"""
    return ["oklopcic","pollack","none"]

def evolution_track(mass=1, age='all'):
    """
    Plot or grab an effective temperature for a certain age and mass planet. 
    Parameters
    ----------
    mass : int or str, optional
        (Optional) Mass of planet, in Jupiter Masses. Only valid options = 1, 2, 4, 6, 8, 10,'all'
        If another value is entered, it will find the nearest option. 
    age : float or str, optional
        (Optional) Age of planet, in years or 'all' to return full model 
    Return 
    ------
    if age=None: data = {'cold':all_data_cold_start, 'hot':all_data_hot_start}
    if age=float: data = {'cold':data_at_age, 'hot':data_at_age}
    
    if plot=False: returns data 
    else: returns data, plot
    
    """
    cols_return = ['age_years','Teff','grav_cgs','logL','R_cm'] #be careful when changing these as they are used to build all_cols
    valid_options = np.array([1,2,4,6,8,10]) # jupiter masses

    if mass == 'all':
        all_cols = np.concatenate([[cols_return[0]]]+[[f'{cols_return[1]}{iv}Mj',f'{cols_return[2]}{iv}Mj'] for iv in valid_options])
        for imass in valid_options:
            mass = f'00{imass}0'            
            if len(mass)==5:mass=mass[1:]
            cold = pd.read_csv(os.path.join(__refdata__, 'evolution','cold_start',f'model_seq.{mass}'),
                skiprows=12,sep=r'\s+',
                    header=None,names=['age_years','logL','R_cm','Ts','Teff',
                                       'log rc','log Pc','log Tc','grav_cgs','Uth','Ugrav','log Lnuc'])
            hot = pd.read_csv(os.path.join(__refdata__, 'evolution','hot_start',f'model_seq.{mass}'),
                skiprows=12,sep=r'\s+',
                    header=None,names=['age_years','logL','R_cm','Ts','Teff',
                                       'log rc','log Pc','log Tc','grav_cgs','Uth','Ugrav','log Lnuc'])
            if imass==1 :
                all_cold = pd.DataFrame(columns=all_cols,index=range(cold.shape[0]))
                all_cold['age_years'] = cold['age_years'].values
                all_hot = pd.DataFrame(columns=all_cols,index=range(hot.shape[0]))
                all_hot['age_years'] = hot['age_years'].values
            #add teff for this mass
            all_cold.loc[:,f'{cols_return[1]}{imass}Mj'] = cold.loc[:,f'{cols_return[1]}'].values
            #add gravity for this mass
            all_cold.loc[:,f'{cols_return[2]}{imass}Mj'] = cold.loc[:,f'{cols_return[2]}'].values
            #add luminosity
            all_cold.loc[:,f'{cols_return[3]}{imass}Mj'] = cold.loc[:,f'{cols_return[3]}'].values
            #add radius
            all_cold.loc[:,f'{cols_return[4]}{imass}Mj'] = cold.loc[:,f'{cols_return[4]}'].values
            #add teff for this mass
            all_hot.loc[:,f'{cols_return[1]}{imass}Mj'] = hot.loc[:,f'{cols_return[1]}'].values
            #add gravity for this mass
            all_hot.loc[:,f'{cols_return[2]}{imass}Mj'] = hot.loc[:,f'{cols_return[2]}'].values
            #add luminosity for this mass
            all_hot.loc[:,f'{cols_return[3]}{imass}Mj'] = hot.loc[:,f'{cols_return[3]}'].values
            #add luminosity for this mass
            all_hot.loc[:,f'{cols_return[4]}{imass}Mj'] = hot.loc[:,f'{cols_return[4]}'].values

        #grab the desired age, if the user asks for it
        if not isinstance(age, str):
            #returning to just hot and cold names so that they can be returned below 
            all_hot = (all_hot.iloc[(all_hot['age_years']-age).abs().argsort()[0:1]]).to_dict('records')[0]
            all_cold = (all_cold.iloc[(all_cold['age_years']-age).abs().argsort()[0:1]]).to_dict('records')[0]

        to_return = {'hot': all_hot, 
                'cold': all_cold}
    else:   
        
        idx = np.argmin(abs(valid_options - mass))
        mass = int(valid_options[idx])
        mass = f'00{mass}0'
        if len(mass)==5:mass=mass[1:]
        cold = pd.read_csv(os.path.join(__refdata__, 'evolution','cold_start',f'model_seq.{mass}'),skiprows=12,sep=r'\s+',
                    header=None,names=['age_years','logL','R_cm','Ts','Teff',
                                       'log rc','log Pc','log Tc','grav_cgs','Uth','Ugrav','log Lnuc'])
        hot = pd.read_csv(os.path.join(__refdata__, 'evolution','hot_start',f'model_seq.{mass}'),skiprows=12,sep=r'\s+',
                    header=None,names=['age_years','logL','R_cm','Ts','Teff',
                                       'log rc','log Pc','log Tc','grav_cgs','Uth','Ugrav','log Lnuc'])
        #return only what we want
        hot = hot.loc[:,cols_return]
        cold = cold.loc[:,cols_return]

        #grab the desired age, if the user asks for it
        if not isinstance(age, str):
            hot = (hot.iloc[(hot['age_years']-age).abs().argsort()[0:1]]).to_dict('records')[0]
            cold = (cold.iloc[(cold['age_years']-age).abs().argsort()[0:1]]).to_dict('records')[0]

        to_return = {'hot': hot, 
                    'cold': cold}


    return to_return

def all_planets():
    """
    Load all planets from https://exoplanetarchive.ipac.caltech.edu
    """
    # use this default URL to start out with 
    planets_df =  pd.read_csv('https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+PSCompPars&format=csv')
    # convert to float when possible
    for i in planets_df.columns: 
        planets_df[i] = planets_df[i].astype(float,errors='ignore')

    return planets_df

def young_planets(): 
    """
    Load planets from ZJ's paper
    """    
    planets_df = pd.read_csv(os.path.join(__refdata__, 'evolution','benchmarks_age_lbol.csv'),skiprows=12)
    return planets_df

def rt_methodology_options(printout=True):
    """Retrieve all the options for methodology"""
    if printout: print("Can calculate spectrum using Toon 1989 methodology or sperhical harmonics")
    return ['toon','SH']
def stream_options(printout=True):
    """Retrieve all the options for stream"""
    if printout: print("Can use 2-stream or 4-stream sperhical harmonics")
    return [2,4]
def toon_phase_coefficients(printout=True):
    """Retrieve options for coefficients used in Toon calculation
    """
    return ["quadrature","eddington"]

def convert_flux_units(xgrid,flux, to_f_unit, xgrid_unit='cm^(-1)',f_unit='erg*cm^(-3)*s^(-1)'): 
    """
    Simple function to convert flux units using sts source spectrum technique from picaso defaults
    Always returns flux in ordered 
    
    Parameters
    ----------
    xgrid : ndarray
        Wavelength or wavenumber array 
    flux : ndarray
        Flux array 
    to_xgrid_unit : str
        astropy approved string unit 
    to_f_unit : str 
        astropy approved string unit 
    xgrid_unit : str, default
        current units, default is picaso original 'cm^(-1)'
    f_unit : str, default 
        current units, default is picaso original 'erg*cm^(-3)*s^(-1)'
    
    """
    ST_SS = SourceSpectrum(Empirical1D, points=xgrid*u.Unit(xgrid_unit), 
                           lookup_table=flux*u.Unit(f_unit))
    y = ST_SS(ST_SS.waveset,flux_unit=u.Unit(to_f_unit))
    
    #if original units were inverse cm and it was an ordered increasing then flip axis 
    if ((xgrid_unit == 'cm^(-1)') & (xgrid[1]>xgrid[0])):
        y = y[::-1]
        
    return y