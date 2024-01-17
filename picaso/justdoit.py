from .atmsetup import ATMSETUP
from .fluxes import get_reflected_1d, get_reflected_3d , get_thermal_1d, get_thermal_3d, get_reflected_SH, get_transit_1d, get_thermal_SH

from .fluxes import tidal_flux, get_kzz#,set_bb_deprecate 
from .climate import  calculate_atm_deq, did_grad_cp, convec, calculate_atm, t_start, growdown, growup, get_fluxes

from .wavelength import get_cld_input_grid
from .optics import RetrieveOpacities,compute_opacity,RetrieveCKs
from .disco import get_angles_1d, get_angles_3d, compute_disco, compress_disco, compress_thermal
from .justplotit import numba_cumsum, find_nearest_2d, mean_regrid
from .deq_chem import quench_level,initiate_cld_matrices
from .build_3d_input import regrid_xarray
from .photochem import run_photochem


from virga import justdoit as vj
from scipy.interpolate import UnivariateSpline
from scipy import special
from numpy import exp, sqrt,log
from numba import jit,njit
from scipy.io import FortranFile


import os
import pickle as pk
import numpy as np
import pandas as pd
import copy
import json
import warnings
with warnings.catch_warnings():#

    warnings.filterwarnings("ignore")
    import pysynphot as psyn
import astropy.units as u
import astropy.constants as c
from astropy.utils.misc import JsonCustomEncoder
import math
import xarray as xr
from joblib import Parallel, delayed, cpu_count

# #testing error tracker
# from loguru import logger 
__refdata__ = os.environ.get('picaso_refdata')
__version__ = 3.1


if not os.path.exists(__refdata__): 
    raise Exception("You have not downloaded the PICASO reference data. You can find it on github here: https://github.com/natashabatalha/picaso/tree/master/reference . If you think you have already downloaded it then you likely just need to set your environment variable. See instructions here: https://natashabatalha.github.io/picaso/installation.html#download-and-link-reference-documentation . You can use `os.environ['PYSYN_CDBS']=<yourpath>` directly in python if you run the line of code before you import PICASO.")
else: 
    ref_v = json.load(open(os.path.join(__refdata__,'config.json'))).get('version',2.3)
    
    if __version__ > ref_v: 
        warnings.warn(f"Your code version is {__version__} but your reference data version is {ref_v}. For some functionality you may experience Keyword errors. Please download the newest ref version or update your code: https://github.com/natashabatalha/picaso/tree/master/reference")


if not os.path.exists(os.environ.get('PYSYN_CDBS')): 
    raise Exception("You have not downloaded the Stellar reference data. Follow the installation instructions here: https://natashabatalha.github.io/picaso/installation.html#download-and-link-pysynphot-stellar-data. If you think you have already downloaded it then you likely just need to set your environment variable. You can use `os.environ['PYSYN_CDBS']=<yourpath>` directly in python if you run the line of code before you import PICASO.")



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
    tridiagonal = 0 
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


    # save returns to output file
    output_dir = inputs['output_dir']

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
    F0PI = np.zeros(nwno) + 1.
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
    query_method = inputs['opacities'].get('query',0)
    exclude_mol = inputs['atmosphere']['exclude_mol']

    #only use nearest neighbor if not using CK method and not using specied by user
    if ((query_method == 0) & (isinstance(getattr(opacityclass,'ck_filename',1),int))): 
        get_opacities = opacityclass.get_opacities_nearest
    elif ((query_method == 1) | (isinstance(getattr(opacityclass,'ck_filename',1),str))):
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

                else:
                    #getting intensities, not fluxes (which is why second return is null)
                    xint = get_reflected_1d(nlevel, wno,nwno,ng,nt,
                                    DTAU[:,:,ig], TAU[:,:,ig], W0[:,:,ig], COSB[:,:,ig],
                                    GCOS2[:,:,ig],ftau_cld[:,:,ig],ftau_ray[:,:,ig],
                                    DTAU_OG[:,:,ig], TAU_OG[:,:,ig], W0_OG[:,:,ig], COSB_OG[:,:,ig],
                                    atm.surf_reflect, ubar0,ubar1,cos_theta, F0PI,
                                    single_phase,multi_phase,
                                    frac_a,frac_b,frac_c,constant_back,constant_forward, toon_coefficients,
                                    b_top=b_top)
                                    #get_toa_intensity=1, get_lvl_flux=0)

                xint_at_top += xint*gauss_wts[ig]

            #if full output is requested add in xint at top for 3d plots
            if full_output: 
                atm.xint_at_top = xint_at_top
                #atm.flux= flux_out
                #atm.int_layer = intensity

        
        if 'thermal' in calculation:
            #use toon method (and tridiagonal matrix solver) to get net cumulative fluxes 
            flux_at_top = 0 

            if get_lvl_flux: 
                atm.get_lvl_flux=True
                atm.lvl_output = dict(flux_minus=0, flux_plus=0, flux_minus_mdpt=0, flux_plus_mdpt=0)
            else: 
                atm.get_lvl_flux=False

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
                                                        wno*0, calc_type=0)


                elif rt_method == 'SH':
                    (flux, flux_out) = get_thermal_SH(nlevel, wno, nwno, ng, nt, atm.level['temperature'],
                                                DTAU[:,:,ig], TAU[:,:,ig], W0[:,:,ig], COSB[:,:,ig], 
                                                DTAU_OG[:,:,ig], TAU_OG[:,:,ig], W0_OG[:,:,ig], 
                                                W0_no_raman[:,:,ig], COSB_OG[:,:,ig], 
                                                atm.level['pressure'], ubar1, 
                                                atm.surf_reflect, stream, atm.hard_surface)

                if get_lvl_flux: 
                    atm.lvl_output['flux_minus']+=lvl_fluxes[0]*gauss_wts[ig]
                    atm.lvl_output['flux_plus']+=lvl_fluxes[1]*gauss_wts[ig]
                    atm.lvl_output['flux_minus_mdpt']+=lvl_fluxes[2]*gauss_wts[ig]
                    atm.lvl_output['flux_plus_mdpt']+=lvl_fluxes[3]*gauss_wts[ig]

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
        for g in range(ng):
            for t in range(nt): 

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
                #use toon method (and tridiagonal matrix solver) to get net cumulative fluxes 
                xint  = get_reflected_3d(nlevel, wno,nwno,ng,nt,
                                                DTAU_3d[:,:,:,:,ig], TAU_3d[:,:,:,:,ig], W0_3d[:,:,:,:,ig], COSB_3d[:,:,:,:,ig],GCOS2_3d[:,:,:,:,ig],
                                                FTAU_CLD_3d[:,:,:,:,ig],FTAU_RAY_3d[:,:,:,:,ig],
                                                DTAU_OG_3d[:,:,:,:,ig], TAU_OG_3d[:,:,:,:,ig], W0_OG_3d[:,:,:,:,ig], COSB_OG_3d[:,:,:,:,ig],
                                                atm.surf_reflect, ubar0,ubar1,cos_theta, F0PI,
                                                single_phase,multi_phase,
                                                frac_a,frac_b,frac_c,constant_back,constant_forward, tridiagonal)
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
                                            PLEVEL_3d,ubar1, atm.surf_reflect, atm.hard_surface, tridiagonal)
                flux_at_top += flux*gauss_wts[ig]
            #if full output is requested add in flux at top for 3d plots
            if full_output: 
                atm.flux_at_top = flux_at_top


    #COMPRESS FULL TANGLE-GANGLE FLUX OUTPUT ONTO 1D FLUX GRID

    #set up initial returns
    returns = {}
    returns['wavenumber'] = wno
    #returns['flux'] = flux
    if 'transmission' in calculation: 
        returns['transit_depth'] = rprs2


    #for reflected light use compress_disco routine
    #this takes the intensity as a functin of tangle/gangle and creates a 1d spectrum
    if  ('reflected' in calculation):
        albedo = compress_disco(nwno, cos_theta, xint_at_top, gweight, tweight,F0PI)
        returns['albedo'] = albedo 
        #returns['xint_at_top'] = xint_at_top 
        #returns['intensity'] = intensity 
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

        if get_lvl_flux: 
            for i in atm.lvl_output.keys():
                atm.lvl_output[i] = compress_thermal(nwno,atm.lvl_output[i], gweight, tweight)   

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

    if output_dir != None:
        filename = output_dir
        pk.dump({'pressure': atm.level['pressure'], 'temperature': atm.level['temperature'], 
            'nlevel':nlevel, 'wno':wno, 'nwno':nwno, 'ng':ng, 'nt':nt, 
            'dtau':DTAU, 'tau':TAU, 'w0':W0, 'cosb':COSB, 'gcos2':GCOS2,'ftcld':ftau_cld,'ftray': ftau_ray,
            'dtau_og':DTAU_OG, 'tau_og':TAU_OG, 'w0_og':W0_OG, 'cosb_og':COSB_OG, 
            'surf_reflect':atm.surf_reflect, 'ubar0':ubar0, 'ubar1':ubar1, 'costheta':cos_theta, 'F0PI':F0PI, 
            'single_phase':single_phase, 'multi_phase':multi_phase, 
            'frac_a':frac_a, 'frac_b':frac_b, 'frac_c':frac_c, 'constant_back':constant_back, 
            'constant_forward':constant_forward, 'dim':dimension, 'stream':stream,
            #'xint_at_top': xint_at_top, 'albedo': albedo, 'flux': flux_out, 'xint': intensity,
            'b_top': b_top, 'gweight': gweight, 'tweight': tweight, 'gangle': gangle, 'tangle': tangle}, 
            open(filename,'wb'), protocol=2)
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
        
    """
        
    full_output = df['full_output'] if 'full_output' in df.keys() else  df['spectrum_output']['full_output']
    df_atmo = picaso_class.inputs['atmosphere']['profile']
    molecules_included = full_output['weights']
    
    #start with simple layer T
    data_vars=dict(temperature = (["pressure"], 
                                  df_atmo['temperature'],
                                  {'units': 'Kelvin'}))
    
    #spectral data 
    if 'thermal' in df.keys(): 
        data_vars['flux_emission'] = (["wavelength"], df['thermal'],{'units': 'erg/cm**2/s/cm'}) 
    if 'transit_depth' in df.keys(): 
        data_vars['transit_depth'] = (["wavelength"], df['transit_depth'],{'units': 'R_jup**2/R_jup**2'}) 
    if 'temp_brightness' in df.keys(): 
        data_vars['temp_brightness'] = (["wavelength"], df['temp_brightness'],{'units': 'Kelvin'})
    if 'fpfs_thermal' in df.keys(): 
        if isinstance(df['fpfs_thermal'], np.ndarray): 
            data_vars['fpfs_emission'] = (["wavelength"], df['fpfs_thermal'],{'units': 'erg/cm**2/s/cm/(erg/cm**2/s/cm)'})
    if 'albedo' in df.keys(): 
        data_vars['albedo'] = (["wavelength"], df['albedo'],{'units': 'none'})
    if 'fpfs_reflected' in df.keys(): 
        if isinstance(df['fpfs_reflected'], np.ndarray): 
            data_vars['fpfs_reflected'] = (["wavelength"], df['fpfs_reflected'],{'units': 'erg/cm**2/s/cm/(erg/cm**2/s/cm)'})
    
    #atmospheric data data 
    for ikey in molecules_included:
        data_vars[ikey] = (["pressure"], df_atmo[ikey].values,{'units': 'v/v'})
        
    if 'kz' in picaso_class.inputs['atmosphere']['profile'].keys(): 
        data_vars['kzz'] = (["pressure"], picaso_class.inputs['atmosphere']['profile']['kz'].values,{'units': 'cm**2/s'})
      
    #clouds if they exist 
    if 'clouds' in picaso_class.inputs: 
        if not isinstance(picaso_class.inputs['clouds']['profile'],type(None)):
            for ikey,lbl in zip( ['opd', 'w0', 'g0'], ['opd','ssa','asy']):
                array = np.reshape(picaso_class.inputs['clouds']['profile'][ikey].values, 
                       (picaso_class.nlevel-1, 
                        len(picaso_class.inputs['clouds']['wavenumber'])))

                data_vars[lbl]=(['pressure_cld','wavenumber_cld'],array,{'units': 'unitless'})
    
    attrs = {}
    #basic info
    for ikey in ['author','code','doi','contact']:
        if add_output.get(ikey,'optional') != 'optional':
            attrs[ikey] = add_output[ikey]
            
    #planet params 
    planet_params = add_output.get('planet_params',{})
    attrs['planet_params'] = {}
    
    #find gravity in picaso
    gravity = picaso_class.inputs['planet'].get('gravity',np.nan)
    if np.isfinite(gravity): 
        gravity = gravity * check_units(picaso_class.inputs['planet']['gravity_unit'])
    #otherwise find gravity from user input
    else: 
        gravity = planet_params.get('logg', np.nan) 
    
    mp = picaso_class.inputs['planet'].get('mass',np.nan)
    if np.isfinite(mp):
        mp = mp * check_units(picaso_class.inputs['planet']['mass_unit'])
    else: 
        mp = planet_params.get('mp',np.nan) 
        
    rp = picaso_class.inputs['planet'].get('radius',np.nan)
    if np.isfinite(mp):
        rp = rp * check_units(picaso_class.inputs['planet']['radius_unit'])
    else: 
        rp = planet_params.get('rp',np.nan) 
        
    #add required RP/MP or gravity
    if (not np.isnan(mp) & (not np.isnan(rp))):
        attrs['planet_params']['mp'] = mp
        attrs['planet_params']['rp'] = rp
        assert isinstance(attrs['planet_params']['mp'],u.quantity.Quantity ), "User supplied mp in planet_params must be an astropy unit: e.g. 1*u.Unit('M_jup')"
        assert isinstance(attrs['planet_params']['rp'],u.quantity.Quantity ), "User supplied rp in planet_params must be an astropy unit: e.g. 1*u.Unit('R_jup')"
    elif (not np.isnan(gravity)): 
        attrs['planet_params']['gravity'] = gravity
        assert isinstance(attrs['planet_params']['gravity'],u.quantity.Quantity ), "User supplied gravity in planet_params must be an astropy unit: e.g. 1*u.Unit('m/s**2')"
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
            

               
        attrs['stellar_params'] = json.dumps(attrs['stellar_params'],cls=JsonCustomEncoder)
        
        
    #add anything else requested by the user
    for ikey in add_output.keys(): 
        if ikey not in attrs.keys(): 
            if add_output[ikey]!="optional":attrs[ikey] = add_output[ikey]
    
    
    coords=dict(
            pressure=(["pressure"], picaso_class.inputs['atmosphere']['profile']['pressure'].values,{'units': 'bar'}),#required*
            wavelength=(["wavelength"], 1e4/df['wavenumber'],{'units': 'micron'})
        )
    if 'clouds' in 'opd' in data_vars.keys(): 
        coords['wavenumber_cld'] = (["wavenumber_cld"], picaso_class.inputs['clouds']['wavenumber'],{'units': 'cm**(-1)'})
        coords['pressure_cld'] = (["pressure_cld"], full_output['layer']['pressure'] ,{'units': full_output['layer']['pressure_unit']})
        
    # put data into a dataset where each
    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=attrs
    )
    
    if isinstance(savefile, str): ds.to_netcdf(savefile)
    return ds
def input_xarray(xr_usr, opacity,p_reference=10, calculation='planet'):
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
    p_reference : float 
        Default is to take from xarray reference pressure in bars 
    calculation : str 
        'planet' or 'browndwarf'

    Example
    -------
    case = jdi.input_xarray(xr_user)
    case.spectrum(opacity,calculation='transit_depth')
    """
    case = inputs(calculation = calculation)
    case.phase_angle(0) #radians

    #define gravity
    planet_params = eval(xr_usr.attrs['planet_params'])
    if 'brown' not in calculation:
        stellar_params = eval(xr_usr.attrs['stellar_params'])
        orbit_params = eval(xr_usr.attrs['orbit_params'])
        steff = _finditem(stellar_params,'steff')
        feh = _finditem(stellar_params,'feh')
        logg = _finditem(stellar_params,'logg')
        database = 'phoenix' if type(_finditem(stellar_params,'database')) == type(None) else _finditem(stellar_params,'database')
        ms = _finditem(stellar_params,'ms')
        rs = _finditem(stellar_params,'rs')
        semi_major = _finditem(planet_params,'sma')
        case.star(opacity, steff,feh,logg, radius=rs['value'], 
                  radius_unit=u.Unit(rs['unit']), database=database)

    mp = _finditem(planet_params,'mp')
    rp = _finditem(planet_params,'rp')
    logg = _finditem(planet_params,'logg')

    if ((not isinstance(mp, type(None))) & (not isinstance(rp, type(None)))):
        case.gravity(mass = mp['value'], mass_unit=u.Unit(mp['unit']),
                    radius=rp['value'], radius_unit=u.Unit(rp['unit']))
    elif (not isinstance(logg, type(None))): 
        case.gravity(gravity = logg['value'], gravity_unit=u.Unit(logg['unit']))
    else: 
        print('Mass and Radius or gravity not provided in xarray, user needs to run gravity function')

    p_reference_xarray = _finditem(planet_params,'p_reference')
    if (not isinstance(p_reference_xarray, type(None))): 
        p_bar = p_reference_xarray['value']*u.Unit(p_reference_xarray['unit'])
        p_bar = p_bar.to('bar').value
        case.approx(p_reference=p_bar)
    elif (not isinstance(p_reference, type(None))): 
        #is it common to want to change the reference pressure
        case.approx(p_reference=p_reference)
    else: 
        raise Exception("p_reference couldnt be found in the xarray, nor was it supplied to this function inputs. Please rerun function with p_reference=10 (or another number in bars).")

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
    rt_method = inputs['approx']['rt_method']

    #set approx numbers options (to be used in numba compiled functions)
    stream = inputs['approx']['rt_params']['common']['stream']
    
    #only used in toon
    multi_phase = inputs['approx']['rt_params']['toon']['multi_phase']
    single_phase = inputs['approx']['rt_params']['toon']['single_phase']
    #define delta eddington approximinations 
    delta_eddington = inputs['approx']['rt_params']['common']['delta_eddington']    
    tridiagonal = 0 
    raman_approx = 2

    #parameters needed for the two term hg phase function. 
    #Defaults are set in config.json
    f = inputs['approx']['rt_params']['common']['TTHG_params']['fraction']
    frac_a = f[0]
    frac_b = f[1]
    frac_c = f[2]
    constant_back = inputs['approx']['rt_params']['common']['TTHG_params']['constant_back']
    constant_forward = inputs['approx']['rt_params']['common']['TTHG_params']['constant_forward']



    #pressure assumption
    p_reference =  inputs['approx']['p_reference']

    ############# DEFINE ALL GEOMETRY USED IN CALCULATION #############
    #see class `inputs` attribute `phase_angle`
    #get geometry
    geom = inputs['disco']

    ng, nt = geom['num_gangle'], geom['num_tangle']
    gangle,gweight,tangle,tweight = geom['gangle'], geom['gweight'],geom['tangle'], geom['tweight']
    lat, lon = geom['latitude'], geom['longitude']
    cos_theta = geom['cos_theta']
    ubar0, ubar1 = geom['ubar0'], geom['ubar1']

    #set star parameters
    radius_star = inputs['star']['radius']
    F0PI = np.zeros(nwno) + 1.
    #semi major axis
    sa = inputs['star']['semi_major']

    #begin atm setup
    atm = ATMSETUP(inputs)

    #Add inputs to class 
    atm.wavenumber = wno
    atm.planet.gravity = inputs['planet']['gravity']
    atm.planet.radius = inputs['planet']['radius']
    atm.planet.mass = inputs['planet']['mass']

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
    query_method = inputs['opacities'].get('query',0)

    if query_method == 0: 
        get_opacities = opacityclass.get_opacities_nearest
    elif query_method == 1:
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

def opannection(wave_range = None, filename_db = None, raman_db = None, 
                resample=1, ck_db=None, deq= False, on_fly=False,
                gases_fly =None,ck=False,
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
        Default is none which pulls opacity file that comes with distribution 
    raman_db : str 
        Filename of raman opacity cross section 
        Default is none which pulls opacity file that comes with distribution 
    resample : int 
        Default=1 (no resampling) PROCEED WITH CAUTION!!!!!This will resample your opacites. 
        This effectively takes opacity[::BINS] depending on what the 
        sampling requested is. Consult your local theorist before 
        using this. 
    ck_db : str 
        ASCII filename of ck file
    verbose : bool 
        Error message to warn users about resampling. Can be turned off by supplying 
        verbose=False
    """
    inputs = json.load(open(os.path.join(__refdata__,'config.json')))

    if isinstance(ck_db, type(None)): 
        #only allow raman if no correlated ck is used 
        if isinstance(raman_db,type(None)): raman_db = os.path.join(__refdata__, 'opacities', inputs['opacities']['files']['raman'])
        
        if isinstance(filename_db,type(None)): 
            filename_db = os.path.join(__refdata__, 'opacities', inputs['opacities']['files']['opacity'])
            if not os.path.isfile(filename_db):
                raise Exception(f'The default opacity file does not exist: {filename_db}. In order to have a default database please download one of the opacity files from Zenodo and place into this folder with the name opacities.db: https://zenodo.org/record/6928501#.Y2w4C-zMI8Y if you dont want a single default file then you just need to point to the opacity db using the keyword filename_db.')
        elif not isinstance(filename_db,type(None) ): 
            if not os.path.isfile(filename_db):
                raise Exception('The opacity file you have entered does not exist: '  + filename_db)

        if resample != 1:
            if verbose:print("YOU ARE REQUESTING RESAMPLING!! This could degrade the precision of your spectral calculations so should be used with caution. If you are unsure check out this tutorial: https://natashabatalha.github.io/picaso/notebooks/10_ResamplingOpacities.html")

        opacityclass=RetrieveOpacities(
                    filename_db, 
                    raman_db,
                    wave_range = wave_range, resample = resample)
    else: 

        if isinstance(filename_db,type(None)): 
            filename_db = os.path.join(__refdata__, 'opacities', inputs['opacities']['files']['ktable_continuum'])
        if not os.path.exists(ck_db):
            if ck_db[-1] == '/':ck_db = ck_db[0:-1]
            if os.path.isfile(ck_db+'.tar.gz'): 
                raise Exception('The CK filename that you have selected appears still be .tar.gz. Please unpack and rerun')
            else: 
                raise Exception('The CK filename that you have selected does not exist. Please make sure you have downloaded and unpacked the right CK file.')
        opacityclass=RetrieveCKs(
                    ck_db, 
                    filename_db, 
                    wave_range = wave_range, 
                    deq=deq,on_fly=on_fly,gases_fly = gases_fly)

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


    def inputs_climate(self, temp_guess= None, pressure= None, nstr = None, nofczns = None , rfacv = None, rfaci = None, cloudy = False, mh = None, CtoO = None, species = None, fsed = None, mieff_dir = None):
        """
        Get Inputs for Climate run

        Parameters
        ----------
        temp_guess : array 
            Guess T(P) profile to begin with
        pressure : array
            Pressure Grid for climate code (this wont change on the fly)
        nstr : array
            NSTR vector describes state of the atmosphere:
            0   is top layer [0]
            1   is top layer of top convective region
            2   is bottom layer of top convective region
            3   is top layer of lower radiative region
            4   is top layer of lower convective region
            5   is bottom layer of lower convective region [nlayer-1]
        rfacv : float
            Fractional contribution of reflected light in net flux
        rfaci : float
            Fractional contribution of thermal light in net flux

        
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
        if cloudy:
            self.inputs['climate']['cloudy'] = 1
            self.inputs['climate']['cld_species'] = species
            self.inputs['climate']['fsed'] = fsed
            self.inputs['climate']['mieff_dir'] = mieff_dir
        else :
            self.inputs['climate']['cloudy'] = 0
            self.inputs['climate']['cld_species'] = 0
            self.inputs['climate']['fsed'] = 0
            self.inputs['climate']['mieff_dir'] = mieff_dir
        self.inputs['climate']['mh'] = mh
        self.inputs['climate']['CtoO'] = CtoO

    def old_run_climate_model(self, opacityclass):
        """
        Top Function to run the Climate Model

        Parameters
        -----------
        
        """
        #get necessary parameters from opacity ck-tables 
        wno = opacityclass.wno
        delta_wno = opacityclass.delta_wno
        nwno = opacityclass.nwno
        min_temp = min(opacityclass.temps)
        max_temp = max(opacityclass.temps)

        
        
        # first calculate the BB grid
        ntmps = self.inputs['climate']['ntemp_bb_grid']
        dt = self.inputs['climate']['dt_bb_grid']
        #we will extend the black body grid 30% beyond the min and max temp of the 
        #opacity grid just to be safe with the spline
        extension = 0.3 
        tmin = min_temp*(1-extension)
        tmax = max_temp*(1+extension)

        bb , y2 , tp = 0,0,0
        #set_bb(wno,delta_wno,nwno,ntmps,dt,tmin,tmax)

        nofczns = self.inputs['climate']['nofczns']
        nstr= self.inputs['climate']['nstr']

        rfaci= self.inputs['climate']['rfaci']
        
        #turn off stellar radiation if user has run "setup_nostar() function"
        if 'nostar' in self.inputs['star']['database']:
            rfacv=0.0 
            F0PI = np.zeros(nwno) #+ 1.0
        #otherwise assume that there is stellar irradiation 
        else:
            rfacv = self.inputs['climate']['rfacv']
            r_star = self.inputs['star']['radius'] 
            r_star_unit = self.inputs['star']['radius_unit'] 
            semi_major = self.inputs['star']['semi_major']
            semi_major_unit = self.inputs['star']['semi_major_unit'] 
            

            fine_flux_star  = self.inputs['star']['flux']  # erg/s/cm^2
            F0PI = fine_flux_star * ((r_star/semi_major)**2)


        TEMP1 = self.inputs['climate']['guess_temp']
        pressure = self.inputs['climate']['pressure']
        t_table = self.inputs['climate']['t_table']
        p_table = self.inputs['climate']['p_table']
        grad = self.inputs['climate']['grad']
        cp = self.inputs['climate']['cp']


        Teff = self.inputs['planet']['T_eff']
        grav = 0.01*self.inputs['planet']['gravity'] # cgs to si
        mh = float(self.inputs['climate']['mh']) if self.inputs['climate']['mh'] != None else 0.0
        sigma_sb = 0.56687e-4 # stefan-boltzmann constant
        
        col_den = 1e6*(pressure[1:] -pressure[:-1] ) / (grav/0.01) # cgs g/cm^2
        wave_in, nlevel, pm, hratio = 0.9, len(pressure), 0.001, 0.1
        #tidal = tidal_flux(Teff, wave_in,nlevel, pressure, pm, hratio, col_den)
        tidal = np.zeros_like(pressure) - sigma_sb *(Teff**4)
        
        cloudy = self.inputs['climate']['cloudy']
        cld_species = self.inputs['climate']['cld_species']
        fsed = self.inputs['climate']['fsed']
        mieff_dir = self.inputs['climate']['mieff_dir']
        # first conv call
        it_max= 10
        itmx= 7
        conv = 10.0
        convt=5.0
        x_max_mult=7.0
        
        #print('NEB FIRST PROFILE RUN')
        final = False
        pressure, temperature, dtdp, profile_flag = profile(mieff_dir, it_max, itmx, conv, convt, nofczns,nstr,x_max_mult,
            TEMP1,pressure, F0PI, t_table, p_table, grad, cp, opacityclass, grav, 
            rfaci, rfacv, nlevel, tidal, tmin, tmax, delta_wno, bb , y2 , tp, final , cloudy, cld_species,mh,fsed )

        # second convergence call
        it_max= 7
        itmx= 5
        conv = 5.0
        convt=4.0
        x_max_mult=7.0


        #print('NEB SECOND PROFILE RUN')
        final = False
        pressure, temperature, dtdp, profile_flag = profile(mieff_dir, it_max, itmx, conv, convt, nofczns,nstr,x_max_mult,
                    temperature,pressure, F0PI, t_table, p_table, grad, cp, opacityclass, grav, 
                    rfaci, rfacv, nlevel, tidal, tmin, tmax, delta_wno, bb , y2 , tp, final, cloudy, cld_species, mh,fsed )   
        
        pressure, temp, dtdp, nstr_new, flux_plus_final =find_strat(mieff_dir, pressure, temperature, dtdp ,F0PI, nofczns,nstr,x_max_mult,
                             t_table, p_table, grad, cp, opacityclass, grav, 
                             rfaci, rfacv, nlevel, tidal, tmin, tmax, delta_wno, bb , y2 , tp , cloudy, cld_species, mh,fsed)

        
        return pressure , temp, dtdp, nstr_new, flux_plus_final
   

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

    def star(self, opannection,temp=None, metal=None, logg=None ,radius = None, radius_unit=None,
        semi_major=None, semi_major_unit = None, #deq = False, 
        database='ck04models',filename=None, w_unit=None, f_unit=None):
        """
        Get the stellar spectrum using pysynphot and interpolate onto a much finer grid than the 
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
            (Optional) Used for stellar file wave units. Needed for filename input.
            Pick: 'um', 'nm', 'cm', 'hz', or 'Angs'
        f_unit : str 
            (Optional) Used for stellar file flux units. Needed for filename input.
            Pick: 'FLAM' or 'Jy' or 'erg/cm2/s/Hz'
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
            semi_major = np.nan
            semi_major_unit = "Semi Major axis not supplied"        

        #upload from file  
        if (not isinstance(filename,type(None))):
            star = np.genfromtxt(filename, dtype=(float, float), names='w, f')
            flux = star['f']
            wave = star['w']
            #sort if not in ascending order 
            sort = np.array([wave,flux]).T
            sort= sort[sort[:,0].argsort()]
            wave = sort[:,0]
            flux = sort[:,1] 
            if w_unit == 'um':
                WAVEUNITS = 'um' 
            elif w_unit == 'nm':
                WAVEUNITS = 'nm'
            elif w_unit == 'cm' :
                WAVEUNITS = 'cm'
            elif w_unit == 'Angs' :
                WAVEUNITS = 'angstrom'
            elif w_unit == 'Hz' :
                WAVEUNITS = 'Hz'
            else: 
                raise Exception('Stellar units are not correct. Pick um, nm, cm, hz, or Angs')        

            #http://www.gemini.edu/sciops/instruments/integration-time-calculators/itc-help/source-definition
            if f_unit == 'Jy':
                FLUXUNITS = 'jy' 
            elif f_unit == 'FLAM' :
                FLUXUNITS = 'FLAM'
            elif f_unit == 'erg/cm2/s/Hz':
                flux = flux*1e23
                FLUXUNITS = 'jy' 
            else: 
                raise Exception('Stellar units are not correct. Pick FLAM or Jy or erg/cm2/s/Hz')

            sp = psyn.ArraySpectrum(wave, flux, waveunits=WAVEUNITS, fluxunits=FLUXUNITS)        #Convert evrything to nanometer for converstion based on gemini.edu  
            sp.convert("um")
            sp.convert('flam') #ergs/cm2/s/ang
            wno_star = 1e4/sp.wave[::-1] #convert to wave number and flip
            flux_star = sp.flux[::-1]*1e8 #flip and convert to ergs/cm3/s here to get correct order         
            

        elif ((not isinstance(temp, type(None))) & (not isinstance(metal, type(None))) & (not isinstance(logg, type(None)))):
            sp = psyn.Icat(database, temp, metal, logg)
            sp.convert("um")
            sp.convert('flam') 
            wno_star = 1e4/sp.wave[::-1] # cm-1 #convert to wave number and flip
            flux_star = sp.flux[::-1]*1e8    #flip here and convert to ergs/cm3/s to get correct order
        else: 
            raise Exception("Must enter 1) filename,w_unit & f_unit OR 2)temp, metal & logg ")

        #now convert to erg/cm2/s/wavenumber
        #flux_star = flux_star/wno_star**2

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
        elif 'climate' in self.inputs['calculation']: 
            #stellar flux of star 
            #print(len(wno_planet),len(flux_star[0:-1]),len(flux_star[1:]))
            # np.diff(1/wno_star) is wavelength window in cm.
            # when multiplied below with flux in ergs/cm3/s from above
            # stellar flux becomes ergs/cm^2/s which is the unit in RT in EGP
            # the fine_flux_star becomes same as "solarf" in EGP
            # remember distance and radius still needs to be adjusted for your case to get the incident flux on your planet
            nrg_flux = 0.5*np.flip(np.diff(1/np.flip(wno_star)))*(flux_star[0:-1]+flux_star[1:])
            fine_wno_star = wno_planet
            #_x,fine_flux_star = mean_regrid(wno_star[:-1], nrg_flux,newx=wno_planet)  
            # getting some Nans at very long wavelengths
            # they are not needed anyways so just moving them to 0
            # look why is this happening
            fine_flux_star = np.zeros(len(wno_planet))
            
            for j in range(len(wno_planet)-1):
                fl = 0
                
                for k in range(1,len(wno_star)):
        
                    if  (wno_star[k] > wno_planet[j]) and (wno_star[k] < wno_planet[j+1]):
                        fl+= 0.5*(flux_star[k-1] +flux_star[k])*abs((1.0/wno_star[k])-(1.0/wno_star[k-1]))
                fine_flux_star[j] = fl

            #where_are_NaNs = np.isnan(fine_flux_star)
            
            #fine_flux_star[where_are_NaNs] = 0   
            
            opannection.unshifted_stellar_spec = fine_flux_star  
            bin_flux_star = fine_flux_star          
        else :
            flux_star_interp = np.interp(wno_planet, wno_star, flux_star)
            _x,bin_flux_star = mean_regrid(wno_star, flux_star,newx=wno_planet)
            #where the star wasn't high enough resolution  
            idx_nobins = np.where(np.isnan(bin_flux_star))[0]
            #replace no bins with interpolated values 
            bin_flux_star[idx_nobins] = flux_star_interp[idx_nobins]
            opannection.unshifted_stellar_spec =bin_flux_star

        self.inputs['star']['database'] = database
        self.inputs['star']['temp'] = temp
        self.inputs['star']['logg'] = logg
        self.inputs['star']['metal'] = metal
        self.inputs['star']['radius'] = r 
        self.inputs['star']['radius_unit'] = radius_unit 
        self.inputs['star']['flux'] = bin_flux_star
        self.inputs['star']['flux_unit'] = 'ergs cm^{-2} s^{-1} cm^{-1}'
        self.inputs['star']['wno'] = wno_planet
        self.inputs['star']['semi_major'] = semi_major 
        self.inputs['star']['semi_major_unit'] = semi_major_unit    
        self.inputs['star']['filename'] = filename
        self.inputs['star']['w_unit'] = w_unit
        self.inputs['star']['f_unit'] = f_unit     

        """
        return not needed anymore
        if deq == True :
            FOPI = fine_flux_star * ((r/semi_major)**2)
            return FOPI
        """

    def atmosphere(self, df=None, filename=None, exclude_mol=None, verbose=True, **pd_kwargs):
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
        verbose : bool 
            (Optional) prints out warnings. Default set to True
        pd_kwargs : kwargs 
            Key word arguments for pd.read_csv to read in supplied atmosphere file 
        """
        if not isinstance(exclude_mol, type(None)):
            if  isinstance(exclude_mol, str):
                exclude_mol = [exclude_mol]

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

        if not isinstance(exclude_mol, type(None)):
            #df = df.drop(exclude_mol, axis=1)
            self.inputs['atmosphere']['exclude_mol'] = {i:1 for i in df.keys()}
            for i in exclude_mol: 
                self.inputs['atmosphere']['exclude_mol'][i]=0
        else: 
            self.inputs['atmosphere']['exclude_mol'] = 1

        self.inputs['atmosphere']['profile'] = df.sort_values('pressure').reset_index(drop=True)

        #lastly check to see if the atmosphere is non-H2 dominant. 
        #if it is, let's turn off Raman scattering for the user. 
        if df.shape[1]>2:
            if (("H2" not in df.keys()) and (self.inputs['approx']['rt_params']['common']['raman'] != 2)):
                if verbose: print("Turning off Raman for Non-H2 atmosphere")
                self.inputs['approx']['rt_params']['common']['raman'] = 2
            elif (("H2" in df.keys()) and (self.inputs['approx']['rt_params']['common']['raman'] != 2)): 
                if df['H2'].min() < 0.7: 
                    if verbose: print("Turning off Raman for Non-H2 atmosphere")
                    self.inputs['approx']['rt_params']['common']['raman'] = 2

    def premix_atmosphere(self, opa, df=None, filename=None, **pd_kwargs):
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

    def premix_atmosphere_diseq(self, opa, quench_levels,t_mix=None, df=None, filename=None, **pd_kwargs):
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
        
        # first quench PH3 from eq abundances of H2O and H2

        # quenching PH3 now, this will only have effect if everything is mixed on the fly
        # formalism from # https://iopscience.iop.org/article/10.1086/428493/pdf
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
        


        
        qvmrs=np.zeros(shape=(5))
        qvmrs2=np.zeros(shape=(3))

        if np.min(quench_levels) >-1 :
        
        ### quench point abundances of each
            qvmrs[0] = self.inputs['atmosphere']['profile']['CH4'][quench_levels[0]]
            qvmrs[1] = self.inputs['atmosphere']['profile']['H2O'][quench_levels[0]]
            qvmrs[2] = self.inputs['atmosphere']['profile']['CO'][quench_levels[0]]
        
            qvmrs2[0] = self.inputs['atmosphere']['profile']['CO2'][quench_levels[1]]

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

            # then quench co2
            self.inputs['atmosphere']['profile']['CO2'][0:quench_levels[1]+1] = self.inputs['atmosphere']['profile']['CO2'][0:quench_levels[1]+1]*0.0 + qvmrs2[0]

            # then quench nh3 and n2

            self.inputs['atmosphere']['profile']['NH3'][0:quench_levels[2]+1] = self.inputs['atmosphere']['profile']['NH3'][0:quench_levels[2]+1]*0.0 + qvmrs[3]
            self.inputs['atmosphere']['profile']['N2'][0:quench_levels[2]+1] = self.inputs['atmosphere']['profile']['N2'][0:quench_levels[2]+1]*0.0 + qvmrs2[1]

            # then quench hcn
            self.inputs['atmosphere']['profile']['HCN'][0:quench_levels[3]+1] = self.inputs['atmosphere']['profile']['HCN'][0:quench_levels[3]+1]*0.0 + qvmrs2[2]
            
                        
            # lastly quench H2 accordingly
            self.inputs['atmosphere']['profile']['H2'][0:quench_levels[0]+1] -= (dq_co + dq_ch4 + dq_h2o) 
            self.inputs['atmosphere']['profile']['H2'][0:quench_levels[1]+1] -= (dq_co2)
            self.inputs['atmosphere']['profile']['H2'][0:quench_levels[2]+1] -= (dq_nh3 + dq_n2)
            self.inputs['atmosphere']['profile']['H2'][0:quench_levels[3]+1] -= (dq_hcn)
            
        #self.inputs['atmosphere']['profile'][species] = pd.DataFrame(abunds)
        
        return qvmrs, qvmrs2
    

    
    def sonora(self, sonora_path, teff, chem='low'):
        """
        This queries Sonora temperature profile that can be downloaded from profiles.tar on 
        Zenodo: [profile.tar file](https://zenodo.org/record/1309035#.Xo5GbZNKjGJ)
        
        Alterntiavely you can grab the sonora bobcat models here: 
        https://zenodo.org/record/5063476/files/structures_m%2B0.0.tar.gz?download=1

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
            ptchem = pd.read_csv(os.path.join(sonora_path,build_filename),delim_whitespace=True,compression='gzip')
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

        if chem == 'high':
            self.channon_grid_high(filename=os.path.join(__refdata__, 'chemistry','grid75_feh+000_co_100_highP.txt'))
        elif chem == 'low':
            self.channon_grid_low(filename=os.path.join(__refdata__,'chemistry','visscher_abunds_m+0.0_co1.0' ))
        elif chem=='grid':
            #solar C/O and M/H 
            self.chemeq_visscher(c_o=1.0,log_mh=0.0)
        self.inputs['atmosphere']['sonora_filename'] = build_filename

    def chemeq_deprecate(self, CtoO, Met):
        """
        This interpolates from a precomputed grid of CEA runs (run by M.R. Line)
        Parameters
        ----------
        CtoO : float
            C to O ratio (solar = 0.55)
        Met : float 
            Metallicity relative to solar (solar = 1)
        """
         
        P, T = self.inputs['atmosphere']['profile']['pressure'].values,self.inputs['atmosphere']['profile']['temperature'].values
        
        T[T<400] = 400
        T[T>2800] = 2800

        logCtoO, logMet, Tarr, logParr, gases=self.chemeq_pic
        assert Met <= 10**np.max(logMet), 'Metallicity entered is higher than the max of the grid: M/H = '+ str(np.max(10**logMet))+'. Make sure units are not in log. Solar M/H = 1.'
        assert CtoO <= 10**np.max(logCtoO), 'C/O ratio entered is higher than the max of the grid: C/O = '+ str(np.max(10**logCtoO))+'. Make sure units are not in log. Solar C/O = 0.55'
        assert Met >= 10**np.min(logMet), 'Metallicity entered is lower than the min of the grid: M/H = '+ str(np.min(10**logMet))+'. Make sure units are not in log. Solar M/H = 1.'
        assert CtoO >= 10**np.min(logCtoO), 'C/O ratio entered is lower than the min of the grid: C/O = '+ str(np.min(10**logCtoO))+'. Make sure units are not in log. Solar C/O = 0.55'

        loggas=np.log10(gases)
        Ngas = loggas.shape[3]
        gas=np.zeros((Ngas,len(P)))
        for j in range(Ngas):
            gas_to_interp=loggas[:,:,:,j,:]
            IF=RegularGridInterpolator((logCtoO, logMet, np.log10(Tarr),logParr),gas_to_interp,bounds_error=False)
            for i in range(len(P)):
                gas[j,i]=10**IF(np.array([np.log10(CtoO), np.log10(Met), np.log10(T[i]), np.log10(P[i])]))
        H2Oarr, CH4arr, COarr, CO2arr, NH3arr, N2arr, HCNarr, H2Sarr,PH3arr, C2H2arr, C2H6arr, Naarr, Karr, TiOarr, VOarr, FeHarr, Harr,H2arr, Hearr, mmw=gas

        df = pd.DataFrame({'H2O': H2Oarr, 'CH4': CH4arr, 'CO': COarr, 'CO2': CO2arr, 'NH3': NH3arr, 
                           'N2' : N2arr, 'HCN': HCNarr, 'H2S': H2Sarr, 'PH3': PH3arr, 'C2H2': C2H2arr, 
                           'C2H6' :C2H6arr, 'Na' : Naarr, 'K' : Karr, 'TiO': TiOarr, 'VO' : VOarr, 
                           'Fe': FeHarr,  'H': Harr, 'H2' : H2arr, 'He' : Hearr, 'temperature':T, 
                           'pressure': P})
        self.inputs['atmosphere']['profile'] = df
        return 

    def channon_grid_high(self,filename=None):
        if isinstance(filename, type(None)):filename=os.path.join(__refdata__,'chemistry','grid75_feh+000_co_100_highP.txt')
        df = self.inputs['atmosphere']['profile'].sort_values('pressure').reset_index(drop=True)

        #sort pressure
        self.inputs['atmosphere']['profile'] = df
        self.nlevel = df.shape[0]
        
        #player = df['pressure'].values
        #tlayer  = df['temperature'].values
        
        grid = pd.read_csv(filename,delim_whitespace=True)
        grid['pressure'] = 10**grid['pressure']

        self.chem_interp(grid)

    def chemeq_visscher(self, c_o, log_mh):#, interp_window = 11, interp_poly=2):
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

        filename = os.path.join(__refdata__,'chemistry','visscher_grid',
            f'2015_06_1060grid_feh_{str_fe}_co_{str_co}.txt').replace('_m0','m0')

        header = pd.read_csv(filename).keys()[0]
        cols = header.replace('T (K)','temperature').replace('P (bar)','pressure').split()
        a = pd.read_csv(filename,delim_whitespace=True,skiprows=1,header=None, names=cols)
        a['pressure']=10**a['pressure']


        self.chem_interp(a)

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

    def add_pt(self, T, P):
        """
        Adds temperature pressure profile to atmosphere
        Parameters
        ----------
        T : array
            Temperature Array
        P : array 
            Pressure Array 
        nlevel : int
            # of atmospheric levels
        
            
        Returns
        -------
        T : numpy.array 
            Temperature grid 
        P : numpy.array
            Pressure grid
                
        """
        self.inputs['atmosphere']['profile']  = pd.DataFrame({'temperature': T, 'pressure': P})
        self.nlevel=len(T) 
        # Return TP profile
        return self.inputs['atmosphere']['profile'] 

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
        
        if zero_point == 'night_transit':
            shift = shift + 180
        elif zero_point == 'secondary_eclipse':
            shift=shift 
        else: 
            raise Exception("Do not regocnize input zero point. Please specify: night_transit or secondary_eclipse")

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
            new_phase_grid['temperature'].isel(pressure=iz_plot).plot(x='lon', y ='lat', col='phase',col_wrap=4)
        
        self.inputs['atmosphere']['profile'] = new_phase_grid

    def clouds_4d(self, ds=None, plot=True, iz_plot=0,iw_plot=0,verbose=True): 
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

    def surface_reflect(self, albedo, wavenumber, old_wavenumber = None):
        """
        Set atmospheric surface reflectivity. This preps the code to run a terrestrial 
        planet. This will automatically change the run to "hardsurface", which alters 
        the lower boundary condition of the thermal_1d flux calculation.
        Parameters
        ----------
        albedo : float
            Set constant albedo for surface reflectivity 
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
    
    def clouds(self, filename = None, g0=None, w0=None, opd=None,p=None, dp=None,df =None,**pd_kwargs):
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
    
    def virga(self, condensates, directory,
        fsed=1, b=1, eps=1e-2, param='const', 
        mh=1, mmw=2.2, kz_min=1e5, sig=2, 
        full_output=False, Teff=None, alpha_pressure=None, supsat=0,
        gas_mmr=None, do_virtual=False, verbose=True): 
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
        """
        
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
        
        cloud_p.ptk(df =df, kz_min = kz_min, Teff = Teff, alpha_pressure = alpha_pressure)
        out = vj.compute(cloud_p, as_dict=full_output,
                          directory=directory, do_virtual=do_virtual)
        if not full_output:
            opd, w0, g0 = out
            df = vj.picaso_format(opd, w0, g0)
        else: 
            opd, w0, g0 = out['opd_per_layer'],out['single_scattering'],out['asymmetry']
            pres = out['pressure']
            wno = 1e4/out['wave']
            df = vj.picaso_format(opd, w0, g0, pressure = pres, wavenumber=wno)
        #only pass through clouds 1d if clouds are one dimension 
        self.clouds(df=df)
        if full_output : return out
        else: return opd, w0, g0
    
    def virga_3d(self, condensates, directory,
        fsed=1, mh=1, mmw=2.2,kz_min=1e5,sig=2, full_output=False,
        n_cpu=1,verbose=True,smooth_kz=False):
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
                wno=(["wno"], wno_grid,{'units': 'cm^(-1)'})#required for clouds
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
        single_form='explicit', calculate_fluxes='off', query='nearest_neighbor',
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
        query : str 
            method to grab opacities. either "nearest_neighbor" or "interp" which 
            interpolates based on 4 nearest neighbors. Default is nearest_neighbor
            which is significantly faster.
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


        self.inputs['opacities']['query'] = query_options().index(query)

        self.inputs['approx']['p_reference']= p_reference
        

    def phase_curve(self, opacityclass,  full_output=False, 
        plot_opacity= False,n_cpu =1 ): 
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
        all_profiles = self.inputs['atmosphere']['profile']
        all_cld_profiles = self.inputs['clouds']['profile']

        def run_phases(iphase):
            self.inputs['phase_angle'] = iphase[1]
            self.inputs['atmosphere']['profile'] = all_profiles.isel(phase=iphase[0])
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

    def inputs_climate(self, temp_guess= None, pressure= None, rfaci = 1,nofczns = 1 ,
        nstr = None,  rfacv = None, 
        cloudy = False, mh = None, CtoO = None, species = None, fsed = None, mieff_dir = None,
        photochem=False, photochem_file=None,photochem_stfile = None,photonetwork_file = None,photonetworkct_file=None,tstop=1e7,psurf=10):
        """
        Get Inputs for Climate run

        Parameters
        ----------
        temp_guess : array 
            Guess T(P) profile to begin with
        pressure : array
            Pressure Grid for climate code (this wont change on the fly)
        nstr : array
            NSTR vector describes state of the atmosphere:
            0   is top layer [0]
            1   is top layer of top convective region
            2   is bottom layer of top convective region
            3   is top layer of lower radiative region
            4   is top layer of lower convective region
            5   is bottom layer of lower convective region [nlayer-1]
        nofczns : integer
            Number of guessed Convective Zones. 1 or 2
        rfacv : float
            Fractional contribution of reflected light in net flux.
            =0 for no stellar irradition, 
            =0.5 for full day-night heat redistribution
            =1 for dayside
        rfaci : float
            Default=1, Fractional contribution of thermal light in net flux
            Usually this is kept at one and then the redistribution is controlled 
            via rfacv
        cloudy : bool
            Include Clouds or not (True or False)
        mh : string
            Metallicity string for 1060 grid, '+0.5','0.0','-0.5'.
        CtoO : string
            C/O ratio string for 1060 grid
        species : string
            Cloud species to be included if cloudy
        fsed : float
            Sedimentation Efficiency (f_sed) if cloudy
        mieff_dir: str
            path to directory with mieff files for virga
        photochem : bool 
            Turns off (False) and on (True) Photochem 
        """
        
        if cloudy: 
            print("Cloudy functionality still in beta form and not ready for public use.")
            # raise Exception('Cloudy functionality still in beta fosrm and not ready for public use.')
        
        elif photochem == False: 
            #dummy values only used for cloud model
            mh = 0 
            CtoO = 0 

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
        if cloudy:
            self.inputs['climate']['cloudy'] = 1
            self.inputs['climate']['cld_species'] = species
            self.inputs['climate']['fsed'] = fsed
            self.inputs['climate']['mieff_dir'] = mieff_dir
        else :
            self.inputs['climate']['cloudy'] = 0
            self.inputs['climate']['cld_species'] = 0
            self.inputs['climate']['fsed'] = 0
            self.inputs['climate']['mieff_dir'] = mieff_dir
        self.inputs['climate']['mh'] = mh
        self.inputs['climate']['CtoO'] = CtoO


        if photochem:
            if m_planet is None:
                raiseExceptions("Supply planet mass if you want to run photochem")
            else:
                self.inputs['climate']['m_planet'] = m_planet

            if r_planet is None:
                raiseExceptions("Supply planet radius if you want to run photochem")
            if photochem_file is None:
                raiseExceptions("Supply photochem_filename if you want to run photochem")
            else:
                self.inputs['climate']['photochem_file'] =photochem_file

            if photochem_stfile is None:
                raiseExceptions("Supply photochem star filename if you want to run photochem")
            else:
                self.inputs['climate']['photochem_stfile'] =photochem_stfile
            self.inputs['climate']['tstop'] =tstop
            self.inputs['climate']['psurf'] =psurf
            self.inputs['climate']['photochem'] =photochem
            self.inputs['climate']['photochem_network'] =photonetwork_file
            self.inputs['climate']['photochem_networkct'] =photonetworkct_file
            

        else:
            self.inputs['climate']['photochem'] =False

    def climate(self, opacityclass, save_all_profiles = False, as_dict=True,with_spec=False,
        save_all_kzz = False, diseq_chem = False, self_consistent_kzz =False, kz = None, 
        on_fly=False,gases_fly=None, chemeq_first=True):#,
       
        """
        Top Function to run the Climate Model

        Parameters
        -----------
        opacityclass : class
            Opacity class from `justdoit.opannection`
        save_all_profiles : bool
            If you want to save and return all iterations in the T(P) profile,True/False
        with_spec : bool 
            Runs picaso spectrum at the end to get the full converged outputs, Default=False
        save_all_kzz : bool
            If you want to save and return all iterations in the kzz profile,True/False
        diseq_chem : bool
            If you want to run `on-the-fly' mixing (takes longer),True/False
        self_consistent_kzz : bool
            If you want to run MLT in convective zones and Moses in the radiative zones
        kz : array
            Kzz input array if user wants constant or whatever input profile (cgs)
        
        """
        #save to user 
        all_out = {}
        
        #get necessary parameters from opacity ck-tables 
        wno = opacityclass.wno
        delta_wno = opacityclass.delta_wno
        nwno = opacityclass.nwno
        min_temp = min(opacityclass.temps)
        max_temp = max(opacityclass.temps)

        
        
        # first calculate the BB grid
        ntmps = self.inputs['climate']['ntemp_bb_grid']
        dt = self.inputs['climate']['dt_bb_grid']
        #we will extend the black body grid 30% beyond the min and max temp of the 
        #opacity grid just to be safe with the spline
        extension = 0.3 
        tmin = min_temp*(1-extension)
        tmax = max_temp*(1+extension)
        ntmps = int((tmax-tmin)/dt)
        
        bb , y2 , tp = 0,0,0
        #bb , y2 , tp = set_bb_deprecate(wno,delta_wno,nwno,ntmps,dt,tmin,tmax)

        nofczns = self.inputs['climate']['nofczns']
        nstr= self.inputs['climate']['nstr']

        rfaci= self.inputs['climate']['rfaci']
        
        #turn off stellar radiation if user has run "setup_nostar() function"
        if 'nostar' in self.inputs['star']['database']:
            rfacv=0.0 
            FOPI = np.zeros(nwno) + 1.0
        #otherwise assume that there is stellar irradiation 
        else:
            rfacv = self.inputs['climate']['rfacv']
            r_star = self.inputs['star']['radius'] 
            r_star_unit = self.inputs['star']['radius_unit'] 
            semi_major = self.inputs['star']['semi_major']
            semi_major_unit = self.inputs['star']['semi_major_unit'] 
            fine_flux_star  = self.inputs['star']['flux']  # erg/s/cm^2
            FOPI = fine_flux_star * ((r_star/semi_major)**2)

        all_profiles= []
        if save_all_profiles:
            save_profile = 1
        else :
            save_profile = 0

        TEMP1 = self.inputs['climate']['guess_temp']
        all_profiles=np.append(all_profiles,TEMP1)
        pressure = self.inputs['climate']['pressure']
        t_table = self.inputs['climate']['t_table']
        p_table = self.inputs['climate']['p_table']
        grad = self.inputs['climate']['grad']
        cp = self.inputs['climate']['cp']


        Teff = self.inputs['planet']['T_eff']
        grav = 0.01*self.inputs['planet']['gravity'] # cgs to si
        mh = float(self.inputs['climate']['mh']) if self.inputs['climate']['mh'] != None else 0.0
        sigma_sb = 0.56687e-4 # stefan-boltzmann constant
        
        col_den = 1e6*(pressure[1:] -pressure[:-1] ) / (grav/0.01) # cgs g/cm^2
        wave_in, nlevel, pm, hratio = 0.9, len(pressure), 0.001, 0.1
        #tidal = tidal_flux(Teff, wave_in,nlevel, pressure, pm, hratio, col_den)
        tidal = np.zeros_like(pressure) - sigma_sb *(Teff**4)
        
        cloudy = self.inputs['climate']['cloudy']
        cld_species = self.inputs['climate']['cld_species']
        fsed = self.inputs['climate']['fsed']
        mieff_dir = self.inputs['climate']['mieff_dir']
        
        opd_cld_climate = np.zeros(shape=(nlevel-1,nwno,4))
        g0_cld_climate = np.zeros(shape=(nlevel-1,nwno,4))
        w0_cld_climate = np.zeros(shape=(nlevel-1,nwno,4))


        # first conv call
        
        it_max= 10   ### inner loop calls
        itmx= 7  ### outer loop calls (opacity re-calculation)
        conv = 10.0
        convt=5.0
        x_max_mult=7.0
        

        final = False
        flag_hack = False

        
        if chemeq_first: pressure, temperature, dtdp, profile_flag, all_profiles,opd_cld_climate,g0_cld_climate,w0_cld_climate,flux_net_ir_layer, flux_plus_ir_attop = profile(mieff_dir,it_max, itmx, conv, convt, nofczns,nstr,x_max_mult,
            TEMP1,pressure, FOPI, t_table, p_table, grad, cp, opacityclass, grav, 
            rfaci, rfacv, nlevel, tidal, tmin, tmax, delta_wno, bb , y2 , tp, final , cloudy, cld_species,mh,fsed,flag_hack, save_profile,all_profiles,opd_cld_climate,g0_cld_climate,w0_cld_climate,first_call_ever=True)

        # second convergence call
        it_max= 7
        itmx= 5
        conv = 5.0
        convt=4.0
        x_max_mult=7.0

        
        final = False
        if chemeq_first: pressure, temperature, dtdp, profile_flag, all_profiles,opd_cld_climate,g0_cld_climate,w0_cld_climate,flux_net_ir_layer, flux_plus_ir_attop = profile(mieff_dir, it_max, itmx, conv, convt, nofczns,nstr,x_max_mult,
                    temperature,pressure, FOPI, t_table, p_table, grad, cp, opacityclass, grav, 
                    rfaci, rfacv, nlevel, tidal, tmin, tmax, delta_wno, bb , y2 , tp, final, cloudy, cld_species, mh,fsed,flag_hack,save_profile,all_profiles,opd_cld_climate,g0_cld_climate,w0_cld_climate,flux_net_ir_layer, flux_plus_ir_attop )   

        if chemeq_first: pressure, temp, dtdp, nstr_new, flux_plus_final, df, all_profiles, opd_now,w0_now,g0_now =find_strat(mieff_dir, pressure, temperature, dtdp ,FOPI, nofczns,nstr,x_max_mult,
                             t_table, p_table, grad, cp, opacityclass, grav, 
                             rfaci, rfacv, nlevel, tidal, tmin, tmax, delta_wno, bb , y2 , tp , cloudy, cld_species, mh,fsed, flag_hack, save_profile,all_profiles,opd_cld_climate,g0_cld_climate,w0_cld_climate,flux_net_ir_layer, flux_plus_ir_attop)

        
        if diseq_chem:
            #Starting with user's guess since there was no request to converge a chemeq profile first 
            if not chemeq_first: 
                temp = TEMP1

            wv196 = 1e4/wno

            # first change the nstr vector because need to check if they grow or not
            # delete upper convective zone if one develops
            
            del_zone =0 # move 4 levels deeper
            if (nstr[1] > 0) & (nstr[4] > 0) & (nstr[3] > 0) :
                nstr[1] = nstr[4]+del_zone
                nstr[2] = 89
                nstr[3],nstr[4],nstr[5] = 0,0,0
                
                print("2 conv Zones, so making small adjustments")
            elif (nstr[1] > 0) & (nstr[3] == 0):
                if nstr[4] == 0:
                    nstr[1]+= del_zone #5#15
                else:
                    nstr[1] += del_zone #5#15  
                    nstr[3], nstr[4] ,nstr[5] = 0,0,0#6#16
                print("1 conv Zone, so making small adjustment")
            if nstr[1] >= nlevel -2 : # making sure we haven't pushed zones too deep
                nstr[1] = nlevel -4
            if nstr[4] >= nlevel -2:
                nstr[4] = nlevel -3
            
            print("New NSTR status is ", nstr)

            

            bundle = inputs(calculation='brown')

            bundle.phase_angle(0,num_gangle=10, num_tangle=1)
            bundle.gravity(gravity=grav , gravity_unit=u.Unit('m/s**2'))
            bundle.add_pt( temp, pressure)
            bundle.premix_atmosphere(opacityclass, df = bundle.inputs['atmosphere']['profile'].loc[:,['pressure','temperature']])
            DTAU, TAU, W0, COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, \
                W0_no_raman , surf_reflect, ubar0,ubar1,cos_theta, single_phase,multi_phase, \
                frac_a,frac_b,frac_c,constant_back,constant_forward, tridiagonal , \
                wno,nwno,ng,nt, nlevel, ngauss, gauss_wts, mmw,gweight,tweight =  calculate_atm(bundle, opacityclass)
            
            all_kzz= []
            if save_all_kzz == True :
                save_kzz = 1
            else :
                save_kzz = 0
            
            #here begins the self consistent Kzz calculation 
            # MLT plus some prescription in radiative zone
            if self_consistent_kzz or (not chemeq_first): 
                #flux_net_v_layer, flux_net_v, flux_plus_v, flux_minus_v , flux_net_ir_layer, flux_net_ir, flux_plus_ir, flux_minus_ir
                flux_net_v_layer_full, flux_net_v_full, flux_plus_v_full, flux_minus_v_full , flux_net_ir_layer_full, flux_net_ir_full, flux_plus_ir_full, flux_minus_ir_full = get_fluxes(pressure, temp, delta_wno, bb , y2, tp, tmin, tmax, DTAU, TAU, W0, 
                COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman , surf_reflect, 
                ubar0,ubar1,cos_theta, FOPI, single_phase,multi_phase,frac_a,frac_b,frac_c,constant_back,constant_forward, tridiagonal , 
                wno,nwno,ng,nt,gweight,tweight, nlevel, ngauss, gauss_wts,True, True)#True for reflected, True for thermal

                flux_net_ir_layer = flux_net_ir_layer_full[:]
                flux_plus_ir_attop = flux_plus_ir_full[0,:] 
                calc_type = 0
                
                # use mixing length theory to calculate Kzz profile
                if self_consistent_kzz: 
                    kz = get_kzz(pressure, temp,grav,mmw,tidal,flux_net_ir_layer, flux_plus_ir_attop,t_table, p_table, grad, cp, calc_type,nstr)
            
            
            
            
            # shift everything to the 661 grid now.
            #mh = '+0.0'  #don't change these as the opacities you are using are based on these 
            #CtoO = '1.0' # don't change these as the opacities you are using are based on these #
            filename_db=os.path.join(__refdata__, 'climate_INPUTS/ck_cx_cont_opacities_661.db')
            
            if on_fly:
                print("From now I will mix "+str(gases_fly)+" only on--the--fly")
                #mhdeq and ctodeq will be auto by opannection
                #NO Background, just CIA + whatever in gases_fly
                #ck_db=os.path.join(__refdata__, 'climate_INPUTS/sonora_2020_feh'+mhdeq+'_co_'+CtoOdeq+'.data.196')
                opacityclass = opannection(ck=True, 
                    ck_db=opacityclass.ck_filename,filename_db=filename_db,
                    deq = True,on_fly=True,gases_fly=gases_fly)
            else:
                #phillips comparison (discontinued) 
                #background + gases 
                #ck_db=os.path.join(__refdata__, 'climate_INPUTS/m+0.0_co1.0.data.196')
                opacityclass = opannection(ck=True, ck_db=opacityclass.ck_filename,
                    filename_db=filename_db,deq = True,on_fly=False)

        
            
            
            if cloudy == 1:    
                wv661 = 1e4/opacityclass.wno
                opd_cld_climate,g0_cld_climate,w0_cld_climate = initiate_cld_matrices(opd_cld_climate,g0_cld_climate,w0_cld_climate,wv196,wv661)
                print(np.shape(opd_cld_climate))
            
            #Rerun star so that F0PI can now be on the 
            #661 grid 
            if 'nostar' in self.inputs['star']['database']:
                FOPI = np.zeros(opacityclass.nwno) + 1.0
            else:
                T_star = self.inputs['star']['temp']
                r_star = self.inputs['star']['radius']
                r_star_unit = self.inputs['star']['radius_unit']
                logg = self.inputs['star']['logg']
                metal =  self.inputs['star']['metal']
                semi_major = self.inputs['star']['semi_major']
                sm_unit = self.inputs['star']['semi_major_unit']
                database = self.inputs['star']['database']
                filename = self.inputs['star']['filename']
                f_unit = self.inputs['star']['f_unit']
                w_unit = self.inputs['star']['w_unit']
                self.star(opacityclass, database=database,temp =T_star,metal =metal, logg =logg, 
                    radius = r_star, radius_unit=u.Unit(r_star_unit),semi_major= semi_major , 
                    semi_major_unit = u.Unit(sm_unit), 
                    filename = filename, 
                    f_unit=f_unit, 
                    w_unit=w_unit)
                fine_flux_star  = self.inputs['star']['flux']  # erg/s/cm^2
                FOPI = fine_flux_star * ((r_star/semi_major)**2)
            
            if self.inputs['climate']['photochem']==False:
                quench_levels, t_mix = quench_level(pressure, temp, kz ,mmw, grav, return_mix_timescale= True) # determine quench levels

                all_kzz = np.append(all_kzz, t_mix) # save kzz

                print("Quench Levels are CO, CO2, NH3, HCN, PH3 ", quench_levels) # print quench levels
                
                final = False
                #finall = False #### what is this thing?
                
                ## this code block is mostly safeguarding
                
                if quench_levels[2] > nlevel -2 :
                    quench_levels[2] = nlevel -2

                    if quench_levels[0] > nlevel -2 :
                        quench_levels[0] = nlevel -2
                    
                    if quench_levels[1] > nlevel -2 :
                        quench_levels[1] = nlevel -2
                    
                    if quench_levels[3] > nlevel -2 :
                        quench_levels[3] = nlevel -2 
                
                
                
                

                # determine the chemistry now

                qvmrs, qvmrs2= bundle.premix_atmosphere_diseq(opacityclass, df = bundle.inputs['atmosphere']['profile'].loc[:,['pressure','temperature']], quench_levels= quench_levels,t_mix=t_mix)
                #was for check SM
                #bundle.inputs['atmosphere']['profile'].to_csv('/data/users/samukher/Disequilibrium-picaso/first_iteration_testpls300min500',sep='\t')
                #raise SystemExit(0) 
                photo_inputs_dict = {}
                photo_inputs_dict['yesorno'] = False
            else :
                bundle.premix_atmosphere(opacityclass, df = bundle.inputs['atmosphere']['profile'].loc[:,['pressure','temperature']])

                pc= bundle.call_photochem(temp,pressure,float(self.inputs['climate']['mh']),float(self.inputs['climate']['CtoO']),self.inputs['climate']['psurf'],self.inputs['climate']['m_planet'],self.inputs['climate']['r_planet'],kz,tstop=self.inputs['climate']['tstop'],filename = self.inputs['climate']['photochem_file'],stfilename = self.inputs['climate']['photochem_stfile'],network = self.inputs['climate']['photochem_network'],network_ct = self.inputs['climate']['photochem_networkct'],first=True,pc=None)
                all_kzz = np.append(all_kzz, kz)
                quench_levels = np.array([0,0,0,0])
                photo_inputs_dict = {}
                photo_inputs_dict['yesorno'] = True
                photo_inputs_dict['mh'] = float(self.inputs['climate']['mh'])
                photo_inputs_dict['CtoO'] = float(self.inputs['climate']['CtoO'])
                photo_inputs_dict['psurf'] = self.inputs['climate']['psurf']
                photo_inputs_dict['m_planet'] = self.inputs['climate']['m_planet']
                photo_inputs_dict['r_planet'] = self.inputs['climate']['r_planet']
                photo_inputs_dict['tstop']=self.inputs['climate']['tstop']
                photo_inputs_dict['photochem_file']=self.inputs['climate']['photochem_file']
                photo_inputs_dict['photochem_stfile']=self.inputs['climate']['photochem_stfile']
                photo_inputs_dict['photochem_network']=self.inputs['climate']['photochem_network']
                photo_inputs_dict['photochem_networkct']=self.inputs['climate']['photochem_networkct']
                photo_inputs_dict['pc'] = pc
                photo_inputs_dict['kz'] = kz
                





            wno = opacityclass.wno
            delta_wno = opacityclass.delta_wno
            nwno = opacityclass.nwno
            min_temp = min(opacityclass.temps)
            max_temp = max(opacityclass.temps)

           
            print(nwno)             
            # first calculate the BB grid
            ntmps = self.inputs['climate']['ntemp_bb_grid']
            dt = self.inputs['climate']['dt_bb_grid']
            
            extension = 0.3 
            tmin = min_temp*(1-extension)
            tmax = max_temp*(1+extension)

            ntmps = int((tmax-tmin)/dt)
            

            #bb , y2 , tp = set_bb(wno,delta_wno,nwno,ntmps,dt,tmin,tmax)

        

            
            final = False

            
            # diseq calculations start here actually
            
            print("DOING DISEQ CALCULATIONS NOW")
            it_max= 7
            itmx= 5
            conv = 5.0
            convt=4.0
            x_max_mult=7.0

            #if nstr[2] < nstr[5]:
            #    nofczns = 2
            #    print("nofczns corrected") 

            
            pressure, temperature, dtdp, profile_flag, qvmrs, qvmrs2, all_profiles, all_kzz,opd_cld_climate,g0_cld_climate,w0_cld_climate,flux_net_ir_layer, flux_plus_ir_attop,photo_inputs_dict  = profile_deq(mieff_dir, it_max, itmx, conv, convt, nofczns,nstr,x_max_mult,
            temp,pressure, FOPI, t_table, p_table, grad, cp, opacityclass, grav, 
            rfaci, rfacv, nlevel, tidal, tmin, tmax, delta_wno, bb , y2 , tp, final , cloudy, cld_species,mh,fsed,flag_hack, quench_levels, kz, mmw,save_profile,all_profiles, self_consistent_kzz,save_kzz,all_kzz, opd_cld_climate,g0_cld_climate,w0_cld_climate,flux_net_ir_layer, flux_plus_ir_attop,photo_inputs_dict,on_fly=on_fly, gases_fly=gases_fly)
            
            print(photo_inputs_dict)
            pressure, temp, dtdp, nstr_new, flux_plus_final, qvmrs, qvmrs2, df, all_profiles, all_kzz,opd_now,g0_now,w0_now,photo_inputs_dict=find_strat_deq(mieff_dir, pressure, temperature, dtdp ,FOPI, nofczns,nstr,x_max_mult,
                            t_table, p_table, grad, cp, opacityclass, grav, 
                            rfaci, rfacv, nlevel, tidal, tmin, tmax, delta_wno, bb , y2 , tp , cloudy, cld_species, mh,fsed, flag_hack, quench_levels,kz ,mmw, save_profile,all_profiles, self_consistent_kzz,save_kzz,all_kzz, opd_cld_climate,g0_cld_climate,w0_cld_climate,flux_net_ir_layer, flux_plus_ir_attop,photo_inputs_dict,on_fly=on_fly, gases_fly=gases_fly)
            
                

            #diseq stuff
            all_out['diseq_out'] = {}
            if save_all_kzz: all_out['diseq_out']['all_kzz'] = all_kzz
            all_out['diseq_out']['quench_levels'] = quench_levels


            #return pressure , temp, dtdp, nstr_new, flux_plus_final, quench_levels, df, all_profiles, all_kzz, opd_now,w0_now,g0_now
        
        #all output to user
        all_out['pressure'] = pressure
        all_out['temperature'] = temp
        all_out['ptchem_df'] = df
        all_out['dtdp'] = dtdp
        all_out['cvz_locs'] = nstr_new
        all_out['flux']=flux_plus_final
        if save_all_profiles: all_out['all_profiles'] = all_profiles            
           
        if with_spec:
            opacityclass = opannection(ck=True, ck_db=opacityclass.ck_filename,deq=False)
            bundle = inputs(calculation='brown')
            bundle.phase_angle(0)
            bundle.gravity(gravity=grav , gravity_unit=u.Unit('m/s**2'))
            bundle.premix_atmosphere(opacityclass,df)
            df_spec = bundle.spectrum(opacityclass,full_output=True)    
            all_out['spectrum_output'] = df_spec 

        #put cld output in all_out
        if cloudy == 1:
            df_cld = vj.picaso_format(opd_now, w0_now, g0_now)
            all_out['cld_output'] = df_cld
        if as_dict: 
            return all_out
        else: 
            return pressure , temp, dtdp, nstr_new, flux_plus_final, df, all_profiles , opd_now,w0_now,g0_now
    
    def call_photochem(self,temp,pressure,logMH, cto,pressure_surf,mass,radius,kzz,tstop=1e7,filename = None,stfilename=None,network=None,network_ct=None,first=True,pc=None):
    
        pc,output_array,species,pressure = run_photochem(temp,pressure,logMH, cto,pressure_surf,mass,radius,kzz,tstop=tstop,filename = filename,stfilename=stfilename,network=network,network_ct= network_ct,first=first,pc=pc)
        #pc,output_array,species,pressure = run_photochem(temp,pressure,logMH, cto,pressure_surf,mass,radius,kzz,tstop=tstop,filename = filename,stfilename=stfilename,network=network,network_ct= network_ct,first=False,pc=pc)
        #pc,output_array,species,pressure = run_photochem(temp,pressure,logMH, cto,pressure_surf,mass,radius,kzz,tstop=tstop,filename = filename,stfilename=stfilename,network=network,network_ct= network_ct,first=False,pc=pc)
        #pc,output_array,species,pressure = run_photochem(temp,pressure,logMH, cto,pressure_surf,mass,radius,kzz,tstop=tstop,filename = filename,stfilename=stfilename,network=network,network_ct= network_ct,first=False,pc=pc)
        #pc,output_array,species,pressure = run_photochem(temp,pressure,logMH, cto,pressure_surf,mass,radius,kzz,tstop=tstop,filename = filename,stfilename=stfilename,network=network,network_ct= network_ct,first=False,pc=pc)
        #pc,output_array,species,pressure = run_photochem(temp,pressure,logMH, cto,pressure_surf,mass,radius,kzz,tstop=tstop,filename = filename,stfilename=stfilename,network=network,network_ct= network_ct,first=False,pc=pc)

        for i in range(len(species)):
            self.inputs['atmosphere']['profile'][species[i]] = output_array[i,:]
        return pc

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
    threed_grid = pd.read_csv(input_file,delim_whitespace=True,names=['p','t','k'])
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
def query_options():
    """Retrieve options for querying opacities """
    return ["nearest_neighbor","interp"]

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
                skiprows=12,delim_whitespace=True,
                    header=None,names=['age_years','logL','R_cm','Ts','Teff',
                                       'log rc','log Pc','log Tc','grav_cgs','Uth','Ugrav','log Lnuc'])
            hot = pd.read_csv(os.path.join(__refdata__, 'evolution','hot_start',f'model_seq.{mass}'),
                skiprows=12,delim_whitespace=True,
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
        cold = pd.read_csv(os.path.join(__refdata__, 'evolution','cold_start',f'model_seq.{mass}'),skiprows=12,delim_whitespace=True,
                    header=None,names=['age_years','logL','R_cm','Ts','Teff',
                                       'log rc','log Pc','log Tc','grav_cgs','Uth','Ugrav','log Lnuc'])
        hot = pd.read_csv(os.path.join(__refdata__, 'evolution','hot_start',f'model_seq.{mass}'),skiprows=12,delim_whitespace=True,
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

def profile(mieff_dir, it_max, itmx, conv, convt, nofczns,nstr,x_max_mult,
            temp,pressure,FOPI, t_table, p_table, grad, cp, opacityclass, grav, 
             rfaci, rfacv, nlevel, tidal, tmin, tmax, dwni, bb , y2 , tp, final, 
             cloudy, cld_species,mh,fsed,flag_hack, save_profile, 
             all_profiles,opd_cld_climate,g0_cld_climate,w0_cld_climate,
             flux_net_ir_layer=None, flux_plus_ir_attop=None,first_call_ever=False,
             verbose=True):
    """
    Function iterating on the TP profile by calling tstart and changing opacities as well
    Parameters
    ----------
    mieff_dir: str
        path to directory with mieff files for virga
    it_max : int
        Maximum iterations allowed in the inner no opa change loop
    itmx : int
        Maximum iterations allowed in the outer opa change loop
    conv : float
        
    convt: float
        Convergence criteria , if max avg change in temp is less than this then outer loop converges
        
    nofczns: int
        # of conv zones 
    nstr : array 
        dimension of 20
        NSTR vector describes state of the atmosphere:
        0   is top layer
        1   is top layer of top convective region
        2   is bottom layer of top convective region
        3   is top layer of lower radiative region
        4   is top layer of lower convective region
        5   is bottom layer of lower convective region
    xmaxmult : 
        
    temp : array 
        Guess temperatures to start with
    pressure : array
        Atmospheric pressure
    t_table : array
        Visible flux addition fraction
    nlevel : int
        # of levels
    temp : array
        Guess Temperature array, dimension is nlevel
    pressure : array
        Pressure array
    t_table : array
        Tabulated Temperature array for convection calculations
    p_table : array
        Tabulated pressure array for convection calculations
    grad : array
        Tabulated grad array for convection calculations
    cp : array
        Tabulated cp array for convection calculations
    opacityclass : class
        Opacity class created with jdi.oppanection
    grav : float
        Gravity of planet in SI
    rfaci : float 
        IR flux addition fraction 
    rfacv : float
        Visible flux addition fraction
    nlevel : int
        # of levels, not layers
    tidal : array
        Tidal Fluxes dimension = nlevel
    tmin : float
        Minimum allwed Temp in the profile
    tmax : float
        Maximum allowed Temp in the profile
    dwni : array
        Spectral interval corrections (dimension= nwvno)   
        
    Returns
    -------
    array 
        Temperature array and lapse ratio array if converged
        else Temperature array twice
    """
    conv_flag = 0
    # taudif is fixed to be 0 here since it is needed only for clouds mh
    taudif = 0.0
    taudif_tol = 0.1
    
    # first calculate the convective zones
    for nb in range(0,3*nofczns,3):
        
        n_strt_b= nstr[nb+1]
        n_ctop_b= n_strt_b+1
        n_bot_b= nstr[nb+2] +1

        for j1 in range(n_ctop_b,n_bot_b+1): 
            press = sqrt(pressure[j1-1]*pressure[j1])
            calc_type =  0 # only need grad_x in return
            grad_x, cp_x = did_grad_cp( temp[j1-1], press, t_table, p_table, grad, cp, calc_type)
            temp[j1]= exp(log(temp[j1-1]) + grad_x*(log(pressure[j1]) - log(pressure[j1-1])))
    
    temp_old= np.copy(temp)


    
    bundle = inputs(calculation='brown')
    bundle.phase_angle(0,num_gangle=10, num_tangle=1)
    bundle.gravity(gravity=grav , gravity_unit=u.Unit('m/s**2'))
    bundle.add_pt( temp, pressure)
    
    bundle.premix_atmosphere(opacityclass, df = bundle.inputs['atmosphere']['profile'].loc[:,['pressure','temperature']])
    if save_profile == 1:
            all_profiles = np.append(all_profiles,temp_old)
    
    if first_call_ever == False:
        if cloudy == 1 :
            DTAU, TAU, W0, COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, \
            W0_no_raman , surf_reflect, ubar0,ubar1,cos_theta, single_phase,multi_phase, \
            frac_a,frac_b,frac_c,constant_back,constant_forward, tridiagonal , \
            wno,nwno,ng,nt, nlevel, ngauss, gauss_wts, mmw,gweight,tweight =  calculate_atm(bundle, opacityclass )


            we0,we1,we2,we3 = 0.25,0.25,0.25,0.25
            opd_prev_cld_step = (we0*opd_cld_climate[:,:,0]+we1*opd_cld_climate[:,:,1]+we2*opd_cld_climate[:,:,2]+we3*opd_cld_climate[:,:,3]) # last average
            
            metallicity = 10**(mh) #atmospheric metallicity relative to Solar
            mean_molecular_weight = np.mean(mmw) # atmospheric mean molecular weight
            # directory ='/Users/sagnickmukherjee/Documents/GitHub/virga/refr_new'
            # directory = '/home/jjm6243/dev_virga/'
            directory = mieff_dir
            
            kzz  = get_kzz(pressure, temp,grav,mmw,tidal,flux_net_ir_layer, flux_plus_ir_attop,t_table, p_table, grad, cp, calc_type,nstr)
            bundle.inputs['atmosphere']['profile']['kz'] = kzz
        

            cld_out = bundle.virga(cld_species,directory, fsed=fsed,mh=metallicity,
                        mmw = mean_molecular_weight,full_output=False) #,climate=True)
            
            opd_now, w0_now, g0_now = cld_out
            
            opd_cld_climate[:,:,3], g0_cld_climate[:,:,3], w0_cld_climate[:,:,3] = opd_cld_climate[:,:,2], g0_cld_climate[:,:,2], w0_cld_climate[:,:,2]
            opd_cld_climate[:,:,2], g0_cld_climate[:,:,2], w0_cld_climate[:,:,2] = opd_cld_climate[:,:,1], g0_cld_climate[:,:,1], w0_cld_climate[:,:,1]
            opd_cld_climate[:,:,1], g0_cld_climate[:,:,1], w0_cld_climate[:,:,1] = opd_cld_climate[:,:,0], g0_cld_climate[:,:,0], w0_cld_climate[:,:,0]
                        
            opd_cld_climate[:,:,0], g0_cld_climate[:,:,0], w0_cld_climate[:,:,0] = opd_now, g0_now, w0_now
            
            #if np.sum(opd_cld_climate[:,:,1]) == 0 :
            #    w0,w1,w2,w3 = 1,0,0,0
            #elif (np.sum(opd_cld_climate[:,:,1]) != 0) and (np.sum(opd_cld_climate[:,:,2]) == 0):
            #    w0,w1,w2,w3 = 0.5,0.5,0,0
            #elif (np.sum(opd_cld_climate[:,:,2]) != 0) and (np.sum(opd_cld_climate[:,:,3]) == 0):
            #    w0,w1,w2,w3 = 0.33,0.33,0.33,0
            #else:
            #    w0,w1,w2,w3 = 0.25,0.25,0.25,0.25
            we0,we1,we2,we3 = 0.25,0.25,0.25,0.25
            
            #sum_opd_clmt = (opd_cld_climate[:,:,0]+opd_cld_climate[:,:,1]+opd_cld_climate[:,:,2]+opd_cld_climate[:,:,3])
            sum_opd_clmt = (we0*opd_cld_climate[:,:,0]+we1*opd_cld_climate[:,:,1]+we2*opd_cld_climate[:,:,2]+we3*opd_cld_climate[:,:,3])
            opd_clmt = (we0*opd_cld_climate[:,:,0]+we1*opd_cld_climate[:,:,1]+we2*opd_cld_climate[:,:,2]+we3*opd_cld_climate[:,:,3])
            g0_clmt = (we0*opd_cld_climate[:,:,0]*g0_cld_climate[:,:,0]+we1*opd_cld_climate[:,:,1]*g0_cld_climate[:,:,1]+we2*opd_cld_climate[:,:,2]*g0_cld_climate[:,:,2]+we3*opd_cld_climate[:,:,3]*g0_cld_climate[:,:,3])/(sum_opd_clmt)
            w0_clmt = (we0*opd_cld_climate[:,:,0]*w0_cld_climate[:,:,0]+we1*opd_cld_climate[:,:,1]*w0_cld_climate[:,:,1]+we2*opd_cld_climate[:,:,2]*w0_cld_climate[:,:,2]+we3*opd_cld_climate[:,:,3]*w0_cld_climate[:,:,3])/(sum_opd_clmt)
            g0_clmt = np.nan_to_num(g0_clmt,nan=0.0)
            w0_clmt = np.nan_to_num(w0_clmt,nan=0.0)
            opd_clmt[np.where(opd_clmt <= 1e-5)] = 0.0
            
            
            df_cld = vj.picaso_format(opd_clmt, w0_clmt, g0_clmt)
            bundle.clouds(df=df_cld)


    DTAU, TAU, W0, COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, \
        W0_no_raman , surf_reflect, ubar0,ubar1,cos_theta, single_phase,multi_phase, \
        frac_a,frac_b,frac_c,constant_back,constant_forward, tridiagonal , \
        wno,nwno,ng,nt, nlevel, ngauss, gauss_wts, mmw,gweight,tweight =  calculate_atm(bundle, opacityclass )
    
    ## begin bigger loop which gets opacities
    for iii in range(itmx):
        
        temp, dtdp, flag_converge, flux_net_ir_layer, flux_plus_ir_attop, all_profiles = t_start(
                    nofczns,nstr,it_max,conv,x_max_mult, 
                    rfaci, rfacv, nlevel, temp, pressure, p_table, t_table, 
                    grad, cp, tidal,tmin,tmax,dwni, bb , y2, tp, DTAU, TAU, W0, COSB,ftau_cld, ftau_ray,GCOS2, 
                    DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman , surf_reflect, 
                    ubar0,ubar1,cos_theta, FOPI, single_phase,multi_phase,frac_a,frac_b,frac_c,constant_back,constant_forward, tridiagonal , 
                    wno,nwno,ng,nt,gweight,tweight, 
                    ngauss, gauss_wts, save_profile, all_profiles)
        
        #NEB stage delete after confirmation from SM
        #if (temp <= min(opacityclass.cia_temps)).any():
        #    wh = np.where(temp <= min(opacityclass.cia_temps))
        #    if len(wh[0]) <= 30 :
        #        if verbose: print(len(wh[0])," points went below the opacity grid. Correcting those.")
        #        temp = correct_profile(temp,pressure,wh,min(opacityclass.cia_temps))
        #    else :
        #        raise Exception('Many points in your profile went off the grid to lower temperatures. Try re-starting from a different guess profile. Parametrized profiles can work better sometime as guess profiles.')
        
        
        
        bundle = inputs(calculation='brown')
        bundle.phase_angle(0)
        bundle.gravity(gravity=grav , gravity_unit=u.Unit('m/s**2'))
        bundle.add_pt( temp, pressure)
        
        bundle.premix_atmosphere(opacityclass, df = bundle.inputs['atmosphere']['profile'].loc[:,['pressure','temperature']])
        #if save_profile == 1:
        #    all_profiles = np.append(all_profiles,bundle.inputs['atmosphere']['profile']['NH3'].values)
        if cloudy == 1 :
            we0,we1,we2,we3 = 0.25,0.25,0.25,0.25
            opd_prev_cld_step = (we0*opd_cld_climate[:,:,0]+we1*opd_cld_climate[:,:,1]+we2*opd_cld_climate[:,:,2]+we3*opd_cld_climate[:,:,3]) # last average
            
            metallicity = 10**(mh) #atmospheric metallicity relative to Solar
            mean_molecular_weight = np.mean(mmw) # atmospheric mean molecular weight
            # directory ='/Users/sagnickmukherjee/Documents/GitHub/virga/refr_new'
            # directory = '/home/jjm6243/dev_virga/'
            directory = mieff_dir

            kzz  = get_kzz(pressure, temp,grav,mmw,tidal,flux_net_ir_layer, flux_plus_ir_attop,t_table, p_table, grad, cp, calc_type,nstr)
            bundle.inputs['atmosphere']['profile']['kz'] = kzz
    
            cld_out = bundle.virga(cld_species,directory, fsed=fsed,mh=metallicity,
                        mmw = mean_molecular_weight,full_output=False)#,climate=True)

            opd_now, w0_now, g0_now = cld_out
            
            opd_cld_climate[:,:,3], g0_cld_climate[:,:,3], w0_cld_climate[:,:,3] = opd_cld_climate[:,:,2], g0_cld_climate[:,:,2], w0_cld_climate[:,:,2]
            opd_cld_climate[:,:,2], g0_cld_climate[:,:,2], w0_cld_climate[:,:,2] = opd_cld_climate[:,:,1], g0_cld_climate[:,:,1], w0_cld_climate[:,:,1]
            opd_cld_climate[:,:,1], g0_cld_climate[:,:,1], w0_cld_climate[:,:,1] = opd_cld_climate[:,:,0], g0_cld_climate[:,:,0], w0_cld_climate[:,:,0]
                        
            opd_cld_climate[:,:,0], g0_cld_climate[:,:,0], w0_cld_climate[:,:,0] = opd_now, g0_now, w0_now
            
            #if np.sum(opd_cld_climate[:,:,1]) == 0 :
            #    w0,w1,w2,w3 = 1,0,0,0
            #elif (np.sum(opd_cld_climate[:,:,1]) != 0) and (np.sum(opd_cld_climate[:,:,2]) == 0):
            #    w0,w1,w2,w3 = 0.5,0.5,0,0
            #elif (np.sum(opd_cld_climate[:,:,2]) != 0) and (np.sum(opd_cld_climate[:,:,3]) == 0):
            #    w0,w1,w2,w3 = 0.33,0.33,0.33,0
            #else:
            #    w0,w1,w2,w3 = 0.25,0.25,0.25,0.25
            we0,we1,we2,we3 = 0.25,0.25,0.25,0.25
            
            #sum_opd_clmt = (opd_cld_climate[:,:,0]+opd_cld_climate[:,:,1]+opd_cld_climate[:,:,2]+opd_cld_climate[:,:,3])
            sum_opd_clmt = (we0*opd_cld_climate[:,:,0]+we1*opd_cld_climate[:,:,1]+we2*opd_cld_climate[:,:,2]+we3*opd_cld_climate[:,:,3])
            opd_clmt = (we0*opd_cld_climate[:,:,0]+we1*opd_cld_climate[:,:,1]+we2*opd_cld_climate[:,:,2]+we3*opd_cld_climate[:,:,3])
            g0_clmt = (we0*opd_cld_climate[:,:,0]*g0_cld_climate[:,:,0]+we1*opd_cld_climate[:,:,1]*g0_cld_climate[:,:,1]+we2*opd_cld_climate[:,:,2]*g0_cld_climate[:,:,2]+we3*opd_cld_climate[:,:,3]*g0_cld_climate[:,:,3])/(sum_opd_clmt)
            w0_clmt = (we0*opd_cld_climate[:,:,0]*w0_cld_climate[:,:,0]+we1*opd_cld_climate[:,:,1]*w0_cld_climate[:,:,1]+we2*opd_cld_climate[:,:,2]*w0_cld_climate[:,:,2]+we3*opd_cld_climate[:,:,3]*w0_cld_climate[:,:,3])/(sum_opd_clmt)
            g0_clmt = np.nan_to_num(g0_clmt,nan=0.0)
            w0_clmt = np.nan_to_num(w0_clmt,nan=0.0)
            opd_clmt[np.where(opd_clmt <= 1e-5)] = 0.0
            
            
            df_cld = vj.picaso_format(opd_clmt, w0_clmt, g0_clmt)
            bundle.clouds(df=df_cld)
            
            diff = (opd_clmt-opd_prev_cld_step)
            taudif = np.max(np.abs(diff))
            taudif_tol = 0.4*np.max(0.5*(opd_clmt+opd_prev_cld_step))
            
            print("Max TAUCLD diff is", taudif, " Tau tolerance is ", taudif_tol)



        

        DTAU, TAU, W0, COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, \
        W0_no_raman , surf_reflect, ubar0,ubar1,cos_theta, single_phase,multi_phase, \
        frac_a,frac_b,frac_c,constant_back,constant_forward, tridiagonal , \
        wno,nwno,ng,nt, nlevel, ngauss, gauss_wts, mmw,gweight,tweight  =  calculate_atm(bundle, opacityclass)

        ert = 0.0 # avg temp change
        scalt= 1.5

        dtx= abs(temp-temp_old)
        ert = np.sum(dtx) 
        
        temp_old= np.copy(temp)
        
        ert = ert/(float(nlevel)*scalt)
        
        if ((iii > 0) & (ert < convt) & (taudif < taudif_tol)) :
            if verbose: print("Profile converged before itmx")
            conv_flag = 1

            return pressure, temp , dtdp, conv_flag, all_profiles, opd_cld_climate,g0_cld_climate,w0_cld_climate, flux_net_ir_layer, flux_plus_ir_attop
        
        if verbose: print("Big iteration is ",min(temp), iii)
    
    
    if conv_flag == 0:
        if verbose: print("Not converged")
    else :
        if verbose: print("Profile converged after itmx hit")
    return pressure, temp, dtdp, conv_flag, all_profiles, opd_cld_climate,g0_cld_climate,w0_cld_climate,flux_net_ir_layer, flux_plus_ir_attop

def find_strat(mieff_dir, pressure, temp, dtdp , FOPI, nofczns,nstr,x_max_mult,
             t_table, p_table, grad, cp, opacityclass, grav, 
             rfaci, rfacv, nlevel, tidal, tmin, tmax, dwni, bb , y2 , tp, cloudy, cld_species,mh,fsed,flag_hack, save_profile, all_profiles, opd_cld_climate,g0_cld_climate,w0_cld_climate,flux_net_ir_layer, flux_plus_ir_attop):
    """
    Function iterating on the TP profile by calling tstart and changing opacities as well
    Parameters
    ----------
    mieff_dir: str
        path to directory with mieff files for virga
    it_max : int
        Maximum iterations allowed in the inner no opa change loop
    itmx : int
        Maximum iterations allowed in the outer opa change loop
    conv : float
        
    convt: float
        Convergence criteria , if max avg change in temp is less than this then outer loop converges
        
    nofczns: int
        # of conv zones 
    nstr : array 
        dimension of 20
        NSTR vector describes state of the atmosphere:
        0   is top layer
        1   is top layer of top convective region
        2   is bottom layer of top convective region
        3   is top layer of lower radiative region
        4   is top layer of lower convective region
        5   is bottom layer of lower convective region
    xmaxmult : 
        
    temp : array 
        Guess temperatures to start with
    pressure : array
        Atmospheric pressure
    t_table : array
        Visible flux addition fraction
    nlevel : int
        # of levels
    temp : array
        Guess Temperature array, dimension is nlevel
    pressure : array
        Pressure array
    t_table : array
        Tabulated Temperature array for convection calculations
    p_table : array
        Tabulated pressure array for convection calculations
    grad : array
        Tabulated grad array for convection calculations
    cp : array
        Tabulated cp array for convection calculations
    opacityclass : class
        Opacity class created with jdi.oppanection
    grav : float
        Gravity of planet in SI
    rfaci : float 
        IR flux addition fraction 
    rfacv : float
        Visible flux addition fraction
    nlevel : int
        # of levels, not layers
    tidal : array
        Tidal Fluxes dimension = nlevel
    tmin : float
        Minimum allwed Temp in the profile
    tmax : float
        Maximum allowed Temp in the profile
    dwni : array
        Spectral interval corrections (dimension= nwvno)
       
        
    Returns
    -------
    array 
        Temperature array and lapse ratio array if converged
        else Temperature array twice
    """
    # new conditions for this routine

    itmx_strat = 5 #itmx  # outer loop counter
    it_max_strat = 8 # its # inner loop counter # original code is 8
    conv_strat = 5.0 # conv
    convt_strat = 3.0 # convt 
    ip2 = -10 #?
    subad = 0.98 # degree to which layer can be subadiabatic and
                    # we still make it adiabatic
    ifirst = 10-1  # start looking after this many layers from top for a conv zone
                   # -1 is for python referencing
    iend = 0 #?
    final = False

    grad_x, cp_x =convec(temp,pressure, t_table, p_table, grad, cp)
    # grad_x = 
    while dtdp[nstr[1]-1] >= subad*grad_x[nstr[1]-1] :
        ratio = dtdp[nstr[1]-1]/grad_x[nstr[1]-1]

        if ratio > 1.8 :
            print("Move up two levels")
            ngrow = 2
            nstr = growup( 1, nstr , ngrow)
        else :
            ngrow = 1
            nstr = growup( 1, nstr , ngrow)
        
        if nstr[1] < 5 :
            raise ValueError( "Convection zone grew to Top of atmosphere, Need to Stop")
        
        pressure, temp, dtdp, profile_flag, all_profiles, opd_cld_climate,g0_cld_climate,w0_cld_climate,flux_net_ir_layer, flux_plus_ir_attop = profile(mieff_dir, it_max_strat, itmx_strat, conv_strat, convt_strat, nofczns,nstr,x_max_mult,
            temp,pressure, FOPI, t_table, p_table, grad, cp, opacityclass, grav, 
             rfaci, rfacv, nlevel, tidal, tmin, tmax, dwni, bb , y2 , tp, final, cloudy, cld_species, mh,fsed,flag_hack, save_profile, all_profiles, opd_cld_climate,g0_cld_climate,w0_cld_climate,flux_net_ir_layer, flux_plus_ir_attop)

    # now for the 2nd convection zone
    dt_max = 0.0 #DTMAX
    i_max = 0 #IMAX
    # -1 in ifirst to include ifirst index
    flag_super = 0
    for i in range(nstr[1]-1, ifirst-1, -1):
        add = dtdp[i] - grad_x[i]
        if add > dt_max and add/grad_x[i] >= 0.02 : # non-neglegible super-adiabaticity
            dt_max = add
            i_max =i
            break
    
    flag_final_convergence =0
    if i_max == 0 or dt_max/grad_x[i_max] < 0.02: # no superadiabaticity, we are done
        flag_final_convergence = 1

    if flag_final_convergence  == 0:
        print(" convection zone status")
        print(nstr[0],nstr[1],nstr[2],nstr[3],nstr[4],nstr[5])
        print(nofczns)

        nofczns = 2
        nstr[4]= nstr[1]
        nstr[5]= nstr[2]
        nstr[1]= i_max
        nstr[2] = i_max
        nstr[3] = i_max #+ 1
        print(nstr)
        if nstr[3] >= nstr[4] :
            #print(nstr[0],nstr[1],nstr[2],nstr[3],nstr[4],nstr[5])
            #print(nofczns)
            raise ValueError("Overlap happened !")
        pressure, temp, dtdp, profile_flag, all_profiles, opd_cld_climate,g0_cld_climate,w0_cld_climate,flux_net_ir_layer, flux_plus_ir_attop = profile(mieff_dir, it_max_strat, itmx_strat, conv_strat, convt_strat, nofczns,nstr,x_max_mult,
            temp,pressure, FOPI, t_table, p_table, grad, cp, opacityclass, grav, 
             rfaci, rfacv, nlevel, tidal, tmin, tmax, dwni, bb , y2 , tp, final, cloudy, cld_species,mh, fsed,flag_hack,save_profile, all_profiles,opd_cld_climate,g0_cld_climate,w0_cld_climate,flux_net_ir_layer, flux_plus_ir_attop)

        i_change = 1
        while i_change == 1 :
            print("Grow Phase : Upper Zone")
            i_change = 0

            d1 = dtdp[nstr[1]-1]
            d2 = dtdp[nstr[3]]
            c1 = grad_x[nstr[1]-1]
            c2 = grad_x[nstr[3]]

            while ((d1 > subad*c1) or (d2 > subad*c2)):

                if (((d1-c1)>= (d2-c2)) or (nofczns == 1)) :
                    ngrow = 1
                    nstr = growup( 1, nstr , ngrow)

                    if nstr[1] < 3 :
                        raise ValueError( "Convection zone grew to Top of atmosphere, Need to Stop")
                else :
                    ngrow = 1
                    nstr = growdown( 1, nstr , ngrow)

                    if nstr[2] == nstr[4]: # one conv zone
                        nofczns =1
                        nstr[2] = nstr[5]
                        nstr[3] = 0
                        i_change = 1
                print(nstr)
                pressure, temp, dtdp, profile_flag, all_profiles,opd_cld_climate,g0_cld_climate,w0_cld_climate,flux_net_ir_layer, flux_plus_ir_attop = profile(mieff_dir, it_max_strat, itmx_strat, conv_strat, convt_strat, nofczns,nstr,x_max_mult,
            temp,pressure, FOPI, t_table, p_table, grad, cp, opacityclass, grav, 
             rfaci, rfacv, nlevel, tidal, tmin, tmax, dwni, bb , y2 , tp, final, cloudy, cld_species, mh,fsed,flag_hack,save_profile, all_profiles,opd_cld_climate,g0_cld_climate,w0_cld_climate,flux_net_ir_layer, flux_plus_ir_attop)

                d1 = dtdp[nstr[1]-1]
                d2 = dtdp[nstr[3]]
                c1 = grad_x[nstr[1]-1]
                c2 = grad_x[nstr[3]]
            #Now grow the lower zone.
            while ((dtdp[nstr[4]-1] >= subad*grad_x[nstr[4]-1]) and nofczns > 1):
                
                ngrow = 1
                nstr = growup( 2, nstr , ngrow)
                #Now check to see if two zones have merged and stop further searching if so.
                if nstr[2] == nstr[4] :
                    nofczns = 1
                    nstr[2] = nstr[5]
                    nstr[3] = 0
                    i_change =1
                print(nstr)
                pressure, temp, dtdp, profile_flag, all_profiles,opd_cld_climate,g0_cld_climate,w0_cld_climate,flux_net_ir_layer, flux_plus_ir_attop = profile(mieff_dir, it_max_strat, itmx_strat, conv_strat, convt_strat, nofczns,nstr,x_max_mult,
                    temp,pressure, FOPI, t_table, p_table, grad, cp, opacityclass, grav, 
                    rfaci, rfacv, nlevel, tidal, tmin, tmax, dwni, bb , y2 , tp, final, cloudy, cld_species, mh,fsed,flag_hack,save_profile, all_profiles,opd_cld_climate,g0_cld_climate,w0_cld_climate,flux_net_ir_layer, flux_plus_ir_attop)
            

            flag_final_convergence = 1
        
    itmx_strat =6
    it_max_strat = 10
    convt_strat = 2.0
    convt_strat = 2.0
    x_max_mult = x_max_mult/2.0
    ip2 = -10

    final = True
    print("final",nstr)
    pressure, temp, dtdp, profile_flag, all_profiles,opd_cld_climate,g0_cld_climate,w0_cld_climate,flux_net_ir_layer, flux_plus_ir_attop = profile(mieff_dir, it_max_strat, itmx_strat, conv_strat, convt_strat, nofczns,nstr,x_max_mult,
                temp,pressure, FOPI, t_table, p_table, grad, cp,opacityclass, grav, 
                rfaci, rfacv, nlevel, tidal, tmin, tmax, dwni, bb , y2 , tp, final, cloudy, cld_species,mh,fsed,flag_hack,save_profile, all_profiles,opd_cld_climate,g0_cld_climate,w0_cld_climate,flux_net_ir_layer, flux_plus_ir_attop)

    #    else :
    #        raise ValueError("Some problem here with goto 125")
        
    if profile_flag == 0:
        print("ENDING WITHOUT CONVERGING")
    elif profile_flag == 1:
        print("YAY ! ENDING WITH CONVERGENCE")
        
    bundle = inputs(calculation='brown')
    bundle.phase_angle(0)
    bundle.gravity(gravity=grav , gravity_unit=u.Unit('m/s**2'))
    bundle.add_pt( temp, pressure)
    
    bundle.premix_atmosphere(opacityclass, df = bundle.inputs['atmosphere']['profile'].loc[:,['pressure','temperature']])

    if cloudy == 1:
        DTAU, TAU, W0, COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman , surf_reflect, ubar0,ubar1,cos_theta, single_phase,multi_phase,frac_a,frac_b,frac_c,constant_back,constant_forward, tridiagonal , wno,nwno,ng,nt, nlevel, ngauss, gauss_wts, mmw,gweight,tweight  =  calculate_atm(bundle, opacityclass)

        metallicity = 10**(mh) #atmospheric metallicity relative to Solar
        mean_molecular_weight = np.mean(mmw) # atmospheric mean molecular weight
        # directory ='/Users/sagnickmukherjee/Documents/GitHub/virga/refr_new'
        # directory = '/home/jjm6243/dev_virga/'
        directory = mieff_dir

        calc_type =0
        kzz  = get_kzz(pressure, temp,grav,mmw,tidal,flux_net_ir_layer, flux_plus_ir_attop,t_table, p_table, grad, cp, calc_type,nstr)
        bundle.inputs['atmosphere']['profile']['kz'] = kzz


        cld_out = bundle.virga(cld_species,directory, fsed=fsed,mh=metallicity,
                        mmw = mean_molecular_weight,full_output=False) #,climate=True)
        
        opd_now, w0_now, g0_now = cld_out
        df_cld = vj.picaso_format(opd_now, w0_now, g0_now)
        bundle.clouds(df=df_cld)
    else:
        opd_now,w0_now,g0_now = 0,0,0

    DTAU, TAU, W0, COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman , surf_reflect, ubar0,ubar1,cos_theta, single_phase,multi_phase,frac_a,frac_b,frac_c,constant_back,constant_forward, tridiagonal , wno,nwno,ng,nt, nlevel, ngauss, gauss_wts, mmw,gweight,tweight  =  calculate_atm(bundle, opacityclass)
    
    flux_net_v_layer_full, flux_net_v_full, flux_plus_v_full, flux_minus_v_full , flux_net_ir_layer_full, flux_net_ir_full, flux_plus_ir_full, flux_minus_ir_full = get_fluxes(pressure, temp, dwni, bb , y2, tp, tmin, tmax, DTAU, TAU, W0, 
            COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman , surf_reflect, 
            ubar0,ubar1,cos_theta, FOPI, single_phase,multi_phase,frac_a,frac_b,frac_c,constant_back,constant_forward, tridiagonal , 
            wno,nwno,ng,nt,gweight,tweight, nlevel, ngauss, gauss_wts, False, True) #false for reflected, true for thermal

      
    
    return pressure, temp, dtdp, nstr , flux_plus_ir_full, bundle.inputs['atmosphere']['profile'], all_profiles,opd_now,w0_now,g0_now

def profile_deq(mieff_dir, it_max, itmx, conv, convt, nofczns,nstr,x_max_mult,
            temp,pressure,FOPI, t_table, p_table, grad, cp, opacityclass, grav, 
             rfaci, rfacv, nlevel, tidal, tmin, tmax, dwni, bb , y2 , tp, final, cloudy, cld_species,mh,fsed,flag_hack,quench_levels,kz,mmw, save_profile, all_profiles,self_consistent_kzz,save_kzz,all_kzz, opd_cld_climate,g0_cld_climate,w0_cld_climate,flux_net_ir_layer, flux_plus_ir_attop,photo_inputs_dict=None,on_fly=False,gases_fly=None ):
    """
    Function iterating on the TP profile by calling tstart and changing opacities as well
    Parameters
    ----------
    mieff_dir: str
        path to directory with mieff files for virga
    it_max : int
        Maximum iterations allowed in the inner no opa change loop
    itmx : int
        Maximum iterations allowed in the outer opa change loop
    conv : float
        
    convt: float
        Convergence criteria , if max avg change in temp is less than this then outer loop converges
        
    nofczns: int
        # of conv zones 
    nstr : array 
        dimension of 20
        NSTR vector describes state of the atmosphere:
        0   is top layer
        1   is top layer of top convective region
        2   is bottom layer of top convective region
        3   is top layer of lower radiative region
        4   is top layer of lower convective region
        5   is bottom layer of lower convective region
    xmaxmult : 
        
    temp : array 
        Guess temperatures to start with
    pressure : array
        Atmospheric pressure
    t_table : array
        Visible flux addition fraction
    nlevel : int
        # of levels
    temp : array
        Guess Temperature array, dimension is nlevel
    pressure : array
        Pressure array
    t_table : array
        Tabulated Temperature array for convection calculations
    p_table : array
        Tabulated pressure array for convection calculations
    grad : array
        Tabulated grad array for convection calculations
    cp : array
        Tabulated cp array for convection calculations
    opacityclass : class
        Opacity class created with jdi.oppanection
    grav : float
        Gravity of planet in SI
    rfaci : float 
        IR flux addition fraction 
    rfacv : float
        Visible flux addition fraction
    nlevel : int
        # of levels, not layers
    tidal : array
        Tidal Fluxes dimension = nlevel
    tmin : float
        Minimum allwed Temp in the profile
    tmax : float
        Maximum allowed Temp in the profile
    dwni : array
        Spectral interval corrections (dimension= nwvno)   
        
    Returns
    -------
    array 
        Temperature array and lapse ratio array if converged
        else Temperature array twice
    """

    # taudif is fixed to be 0 here since it is needed only for clouds
    taudif = 0.0
    taudif_tol = 0.1
    
    # first calculate the convective zones
    for nb in range(0,3*nofczns,3):
        
        n_strt_b= nstr[nb+1]
        n_ctop_b= n_strt_b+1
        n_bot_b= nstr[nb+2] +1

        for j1 in range(n_ctop_b,n_bot_b+1): 
            press = sqrt(pressure[j1-1]*pressure[j1])
            calc_type =  0 # only need grad_x in return
            grad_x, cp_x = did_grad_cp( temp[j1-1], press, t_table, p_table, grad, cp, calc_type)
            temp[j1]= exp(log(temp[j1-1]) + grad_x*(log(pressure[j1]) - log(pressure[j1-1])))
    '''            
    if (temp <= min(opacityclass.cia_temps)).any():
            wh = np.where(temp <= min(opacityclass.cia_temps))
            if len(wh[0]) <= 30 :
                print(len(wh[0])," points went off the opacity grid. Correcting those.")
                temp = correct_profile(temp,pressure,wh,min(opacityclass.cia_temps))
            else :
                raise Exception('Many points in your profile went off the grid. Try re-starting from a different guess profile. Parametrized profiles can work better sometime as guess profiles.')
    '''
    
    temp_old= np.copy(temp)


    
    bundle = inputs(calculation='brown')
    bundle.phase_angle(0)
    bundle.gravity(gravity=grav , gravity_unit=u.Unit('m/s**2'))
    bundle.add_pt( temp, pressure)
    #### to get the last Kzz in the calculation
    

    if photo_inputs_dict['yesorno'] == False:
        k_b = 1.38e-23 # boltzmann constant
        m_p = 1.66e-27 # proton mass
        
        if len(mmw) < len(temp):
            mmw = np.append(mmw,mmw[-1])
        con  = k_b/(mmw*m_p)

        scale_H = con * temp*1e2/(grav)

        kz = scale_H**2/all_kzz[-len(temp):] ## level mixing timescales
        quench_levels, t_mix = quench_level(pressure, temp, kz ,mmw, grav, return_mix_timescale=True)
        if save_kzz == 1:
            all_kzz = np.append(all_kzz,t_mix)
        qvmrs, qvmrs2 = bundle.premix_atmosphere_diseq(opacityclass, quench_levels=quench_levels, df = bundle.inputs['atmosphere']['profile'].loc[:,['pressure','temperature']],t_mix=t_mix)
    else :
        bundle.premix_atmosphere(opacityclass, df = bundle.inputs['atmosphere']['profile'].loc[:,['pressure','temperature']])
        pc= bundle.call_photochem(temp,pressure,photo_inputs_dict['mh'],photo_inputs_dict['CtoO'],photo_inputs_dict['psurf'],photo_inputs_dict['m_planet'],photo_inputs_dict['r_planet'],photo_inputs_dict['kz'],tstop=photo_inputs_dict['tstop'],filename = photo_inputs_dict['photochem_file'],stfilename =photo_inputs_dict['photochem_stfile'],network = photo_inputs_dict['photochem_network'],network_ct=photo_inputs_dict['photochem_networkct'],first=False,pc=photo_inputs_dict['pc'])
        photo_inputs_dict['pc'] = pc
        all_kzz = np.append(all_kzz, kz)
        quench_levels = np.array([0,0,0,0])
        photo_inputs_dict['pc'] = pc
        qvmrs, qvmrs2=0,0
    
    #if save_profile == 1:
    #        all_profiles = np.append(all_profiles,bundle.inputs['atmosphere']['profile']['NH3'].values)
    # no flux calculation yet so no cloud calculation needed
    #if cloudy == 1 :
    #    metallicity = mh #atmospheric metallicity relative to Solar
    #    mean_molecular_weight = 2.2 # atmospheric mean molecular weight
    #    directory ='/Users/sagnickmukherjee/Documents/software/optics'
    #    bundle.inputs['atmosphere']['profile']['kz'] = 1e5 + np.zeros_like(temp) # start with kzmin
    #    
    
    #    bundle.virga(cld_species,directory, fsed=fsed,mh=metallicity,
    #    mmw = mean_molecular_weight,full_output=False)
    

    if cloudy == 1 :
            

            we0,we1,we2,we3 = 0.25,0.25,0.25,0.25
            opd_prev_cld_step = (we0*opd_cld_climate[:,:,0]+we1*opd_cld_climate[:,:,1]+we2*opd_cld_climate[:,:,2]+we3*opd_cld_climate[:,:,3]) # last average
            
            metallicity = 10**(0) #atmospheric metallicity relative to Solar
            mean_molecular_weight = np.mean(mmw) # atmospheric mean molecular weight
            # directory ='/Users/sagnickmukherjee/Documents/GitHub/virga/refr_new661'
            # directory = '/home/jjm6243/dev_virga/'
            directory = mieff_dir

            kzz  = get_kzz(pressure, temp,grav,mmw,tidal,flux_net_ir_layer, flux_plus_ir_attop,t_table, p_table, grad, cp, calc_type,nstr)
            bundle.inputs['atmosphere']['profile']['kz'] = kzz
            photo_inputs_dict['kz'] = kzz

            cld_out = bundle.virga(cld_species,directory, fsed=fsed,mh=metallicity,
                        mmw = mean_molecular_weight,full_output=False)#,climate=True)
            
            opd_now, w0_now, g0_now = cld_out
            
            opd_cld_climate[:,:,3], g0_cld_climate[:,:,3], w0_cld_climate[:,:,3] = opd_cld_climate[:,:,2], g0_cld_climate[:,:,2], w0_cld_climate[:,:,2]
            opd_cld_climate[:,:,2], g0_cld_climate[:,:,2], w0_cld_climate[:,:,2] = opd_cld_climate[:,:,1], g0_cld_climate[:,:,1], w0_cld_climate[:,:,1]
            opd_cld_climate[:,:,1], g0_cld_climate[:,:,1], w0_cld_climate[:,:,1] = opd_cld_climate[:,:,0], g0_cld_climate[:,:,0], w0_cld_climate[:,:,0]
                        
            opd_cld_climate[:,:,0], g0_cld_climate[:,:,0], w0_cld_climate[:,:,0] = opd_now, g0_now, w0_now
            
            #if np.sum(opd_cld_climate[:,:,1]) == 0 :
            #    w0,w1,w2,w3 = 1,0,0,0
            #elif (np.sum(opd_cld_climate[:,:,1]) != 0) and (np.sum(opd_cld_climate[:,:,2]) == 0):
            #    w0,w1,w2,w3 = 0.5,0.5,0,0
            #elif (np.sum(opd_cld_climate[:,:,2]) != 0) and (np.sum(opd_cld_climate[:,:,3]) == 0):
            #    w0,w1,w2,w3 = 0.33,0.33,0.33,0
            #else:
            #    w0,w1,w2,w3 = 0.25,0.25,0.25,0.25
            we0,we1,we2,we3 = 0.25,0.25,0.25,0.25
            
            #sum_opd_clmt = (opd_cld_climate[:,:,0]+opd_cld_climate[:,:,1]+opd_cld_climate[:,:,2]+opd_cld_climate[:,:,3])
            sum_opd_clmt = (we0*opd_cld_climate[:,:,0]+we1*opd_cld_climate[:,:,1]+we2*opd_cld_climate[:,:,2]+we3*opd_cld_climate[:,:,3])
            opd_clmt = (we0*opd_cld_climate[:,:,0]+we1*opd_cld_climate[:,:,1]+we2*opd_cld_climate[:,:,2]+we3*opd_cld_climate[:,:,3])
            g0_clmt = (we0*opd_cld_climate[:,:,0]*g0_cld_climate[:,:,0]+we1*opd_cld_climate[:,:,1]*g0_cld_climate[:,:,1]+we2*opd_cld_climate[:,:,2]*g0_cld_climate[:,:,2]+we3*opd_cld_climate[:,:,3]*g0_cld_climate[:,:,3])/(sum_opd_clmt)
            w0_clmt = (we0*opd_cld_climate[:,:,0]*w0_cld_climate[:,:,0]+we1*opd_cld_climate[:,:,1]*w0_cld_climate[:,:,1]+we2*opd_cld_climate[:,:,2]*w0_cld_climate[:,:,2]+we3*opd_cld_climate[:,:,3]*w0_cld_climate[:,:,3])/(sum_opd_clmt)
            g0_clmt = np.nan_to_num(g0_clmt,nan=0.0)
            w0_clmt = np.nan_to_num(w0_clmt,nan=0.0)
            opd_clmt[np.where(opd_clmt <= 1e-5)] = 0.0
            
            
            df_cld = vj.picaso_format(opd_clmt, w0_clmt, g0_clmt)
            bundle.clouds(df=df_cld)

    DTAU, TAU, W0, COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, \
        W0_no_raman , surf_reflect, ubar0,ubar1,cos_theta, single_phase,multi_phase, \
        frac_a,frac_b,frac_c,constant_back,constant_forward, tridiagonal , \
        wno,nwno,ng,nt, nlevel, ngauss, gauss_wts, mmw =  calculate_atm_deq(bundle, opacityclass,on_fly=on_fly,gases_fly=gases_fly)
    if self_consistent_kzz == True :
                
        flux_net_v_layer_full, flux_net_v_full, flux_plus_v_full, flux_minus_v_full , flux_net_ir_layer_full, flux_net_ir_full, flux_plus_ir_full, flux_minus_ir_full = get_fluxes(pressure, temp, dwni, bb , y2, tp, tmin, tmax, DTAU, TAU, W0, 
        COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman , surf_reflect, 
        ubar0,ubar1,cos_theta, FOPI, single_phase,multi_phase,frac_a,frac_b,frac_c,constant_back,constant_forward, tridiagonal , 
        wno,nwno,ng,nt, gweight,tweight,nlevel, ngauss, gauss_wts,True, True)#True for reflected, True for thermal

        flux_net_ir_layer = flux_net_ir_layer_full[:]
        flux_plus_ir_attop = flux_plus_ir_full[0,:] 
        calc_type = 0
    
        kz = get_kzz(pressure, temp,grav,mmw,tidal,flux_net_ir_layer, flux_plus_ir_attop,t_table, p_table, grad, cp, calc_type,nstr)
        photo_inputs_dict['kz'] = kz
    ## begin bigger loop which gets opacities
    for iii in range(itmx):
        
        temp, dtdp, flag_converge, flux_net_ir_layer, flux_plus_ir_attop, all_profiles = t_start(nofczns,nstr,it_max,conv,x_max_mult, 
            rfaci, rfacv, nlevel, temp, pressure, p_table, t_table, 
            grad, cp, tidal,tmin,tmax,dwni, bb , y2, tp, DTAU, TAU, W0, 
            COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, 
            W0_no_raman , surf_reflect, ubar0,ubar1,cos_theta, FOPI, 
            single_phase,multi_phase,frac_a,frac_b,frac_c,constant_back,constant_forward, 
            tridiagonal , wno,nwno,ng,nt,gweight,tweight, ngauss, gauss_wts, save_profile, all_profiles)
        '''
        if (temp <= min(opacityclass.cia_temps)).any():
            wh = np.where(temp <= min(opacityclass.cia_temps))
            if len(wh[0]) <= 30 :
                print(len(wh[0])," points went off the opacity grid. Correcting those.")
                temp = correct_profile(temp,pressure,wh,min(opacityclass.cia_temps))
            else :
                raise Exception('Many points in your profile went off the grid. Try re-starting from a different guess profile. Parametrized profiles can work better sometime as guess profiles.')
       '''
        
        
        bundle = inputs(calculation='brown')
        bundle.phase_angle(0)
        bundle.gravity(gravity=grav , gravity_unit=u.Unit('m/s**2'))
        bundle.add_pt( temp, pressure)
        if photo_inputs_dict['yesorno'] == False:
            quench_levels, t_mix = quench_level(pressure, temp, kz ,mmw, grav, return_mix_timescale=True)
            
            qvmrs, qvmrs2 = bundle.premix_atmosphere_diseq(opacityclass, quench_levels=quench_levels, df = bundle.inputs['atmosphere']['profile'].loc[:,['pressure','temperature']],t_mix=t_mix)
            print("Quench Levels are CO, CO2, NH3, HCN ", quench_levels)
        else :
            bundle.premix_atmosphere(opacityclass, df = bundle.inputs['atmosphere']['profile'].loc[:,['pressure','temperature']])
            pc= bundle.call_photochem(temp,pressure,photo_inputs_dict['mh'],photo_inputs_dict['CtoO'],photo_inputs_dict['psurf'],photo_inputs_dict['m_planet'],photo_inputs_dict['r_planet'],photo_inputs_dict['kz'],tstop=photo_inputs_dict['tstop'],filename = photo_inputs_dict['photochem_file'],stfilename =photo_inputs_dict['photochem_stfile'],network = photo_inputs_dict['photochem_network'],network_ct=photo_inputs_dict['photochem_networkct'],first=False,pc=photo_inputs_dict['pc'])
            photo_inputs_dict['pc'] = pc
            all_kzz = np.append(all_kzz, kz)
            quench_levels = np.array([0,0,0,0])
            photo_inputs_dict['pc'] = pc
            qvmrs, qvmrs2=0,0
            photo_inputs_dict['kz'] = kz
        
    
        #if save_profile == 1:
        #    all_profiles = np.append(all_profiles,bundle.inputs['atmosphere']['profile']['NH3'].values)
        
        if cloudy == 1 :
            we0,we1,we2,we3 = 0.25,0.25,0.25,0.25
            opd_prev_cld_step = (we0*opd_cld_climate[:,:,0]+we1*opd_cld_climate[:,:,1]+we2*opd_cld_climate[:,:,2]+we3*opd_cld_climate[:,:,3]) # last average
            
            metallicity = 10**(0) #atmospheric metallicity relative to Solar
            mean_molecular_weight = np.mean(mmw) # atmospheric mean molecular weight
            # directory ='/Users/sagnickmukherjee/Documents/GitHub/virga/refr_new661'
            # directory = '/home/jjm6243/dev_virga/'
            directory = mieff_dir

            kzz  = get_kzz(pressure, temp,grav,mmw,tidal,flux_net_ir_layer, flux_plus_ir_attop,t_table, p_table, grad, cp, calc_type,nstr)
            bundle.inputs['atmosphere']['profile']['kz'] = kzz
            photo_inputs_dict['kz'] =kzz
        
    
            cld_out = bundle.virga(cld_species,directory, fsed=fsed,mh=metallicity,
                        mmw = mean_molecular_weight,full_output=False)#,climate=True)
            
            opd_now, w0_now, g0_now = cld_out
            
            opd_cld_climate[:,:,3], g0_cld_climate[:,:,3], w0_cld_climate[:,:,3] = opd_cld_climate[:,:,2], g0_cld_climate[:,:,2], w0_cld_climate[:,:,2]
            opd_cld_climate[:,:,2], g0_cld_climate[:,:,2], w0_cld_climate[:,:,2] = opd_cld_climate[:,:,1], g0_cld_climate[:,:,1], w0_cld_climate[:,:,1]
            opd_cld_climate[:,:,1], g0_cld_climate[:,:,1], w0_cld_climate[:,:,1] = opd_cld_climate[:,:,0], g0_cld_climate[:,:,0], w0_cld_climate[:,:,0]
                        
            opd_cld_climate[:,:,0], g0_cld_climate[:,:,0], w0_cld_climate[:,:,0] = opd_now, g0_now, w0_now
            
            #if np.sum(opd_cld_climate[:,:,1]) == 0 :
            #    w0,w1,w2,w3 = 1,0,0,0
            #elif (np.sum(opd_cld_climate[:,:,1]) != 0) and (np.sum(opd_cld_climate[:,:,2]) == 0):
            #    w0,w1,w2,w3 = 0.5,0.5,0,0
            #elif (np.sum(opd_cld_climate[:,:,2]) != 0) and (np.sum(opd_cld_climate[:,:,3]) == 0):
            #    w0,w1,w2,w3 = 0.33,0.33,0.33,0
            #else:
            #    w0,w1,w2,w3 = 0.25,0.25,0.25,0.25
            we0,we1,we2,we3 = 0.25,0.25,0.25,0.25
            
            #sum_opd_clmt = (opd_cld_climate[:,:,0]+opd_cld_climate[:,:,1]+opd_cld_climate[:,:,2]+opd_cld_climate[:,:,3])
            sum_opd_clmt = (we0*opd_cld_climate[:,:,0]+we1*opd_cld_climate[:,:,1]+we2*opd_cld_climate[:,:,2]+we3*opd_cld_climate[:,:,3])
            opd_clmt = (we0*opd_cld_climate[:,:,0]+we1*opd_cld_climate[:,:,1]+we2*opd_cld_climate[:,:,2]+we3*opd_cld_climate[:,:,3])
            g0_clmt = (we0*opd_cld_climate[:,:,0]*g0_cld_climate[:,:,0]+we1*opd_cld_climate[:,:,1]*g0_cld_climate[:,:,1]+we2*opd_cld_climate[:,:,2]*g0_cld_climate[:,:,2]+we3*opd_cld_climate[:,:,3]*g0_cld_climate[:,:,3])/(sum_opd_clmt)
            w0_clmt = (we0*opd_cld_climate[:,:,0]*w0_cld_climate[:,:,0]+we1*opd_cld_climate[:,:,1]*w0_cld_climate[:,:,1]+we2*opd_cld_climate[:,:,2]*w0_cld_climate[:,:,2]+we3*opd_cld_climate[:,:,3]*w0_cld_climate[:,:,3])/(sum_opd_clmt)
            g0_clmt = np.nan_to_num(g0_clmt,nan=0.0)
            w0_clmt = np.nan_to_num(w0_clmt,nan=0.0)
            opd_clmt[np.where(opd_clmt <= 1e-5)] = 0.0
            
            
            df_cld = vj.picaso_format(opd_clmt, w0_clmt, g0_clmt)
            bundle.clouds(df=df_cld)
            
            diff = (opd_clmt-opd_prev_cld_step)
            taudif = np.max(np.abs(diff))
            taudif_tol = 0.4*np.max(0.5*(opd_clmt+opd_prev_cld_step))
            
            print("Max TAUCLD diff is", taudif, " Tau tolerance is ", taudif_tol)

        
        DTAU, TAU, W0, COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, \
        W0_no_raman , surf_reflect, ubar0,ubar1,cos_theta, single_phase,multi_phase, \
        frac_a,frac_b,frac_c,constant_back,constant_forward, tridiagonal , \
        wno,nwno,ng,nt, nlevel, ngauss, gauss_wts, mmw =  calculate_atm_deq(bundle, opacityclass,on_fly=on_fly, gases_fly=gases_fly)

        if self_consistent_kzz == True :
                
            flux_net_v_layer_full, flux_net_v_full, flux_plus_v_full, flux_minus_v_full , flux_net_ir_layer_full, flux_net_ir_full, flux_plus_ir_full, flux_minus_ir_full = get_fluxes(pressure, temp, dwni, bb , y2, tp, tmin, tmax, DTAU, TAU, W0, 
            COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman , surf_reflect, 
            ubar0,ubar1,cos_theta, FOPI, single_phase,multi_phase,frac_a,frac_b,frac_c,constant_back,constant_forward, tridiagonal , 
            wno,nwno,ng,nt, gweight,tweight, nlevel, ngauss, gauss_wts,True, True)#True for reflected, True for thermal

            flux_net_ir_layer = flux_net_ir_layer_full[:]
            flux_plus_ir_attop = flux_plus_ir_full[0,:] 
            calc_type = 0
        
            kz = get_kzz(pressure, temp,grav,mmw,tidal,flux_net_ir_layer, flux_plus_ir_attop,t_table, p_table, grad, cp, calc_type,nstr)
            photo_inputs_dict['kz'] = kz
        if save_kzz == 1: 
            if photo_inputs_dict['yesorno'] == False:
                all_kzz = np.append(all_kzz,t_mix)
            else:
                all_kzz = np.append(all_kzz,kz)


        ert = 0.0 # avg temp change
        scalt= 1.5

        dtx= abs(temp-temp_old)
        ert = np.sum(dtx)
        
        ## this is a terrible hack but it perhaps works
        ## do this hack only during findstrat maybe ?
        ## otherwise problematic
        #####################################
    #        if flag_hack == True:
    #            temp= 0.5*(temp+temp_old) 
    #            print("Hack Activated")
        #####################################   
        
        temp_old= np.copy(temp)
        
        ert = ert/(float(nlevel)*scalt)
        
        if ((iii > 0) & (ert < convt) & (taudif < taudif_tol)) :
            print("Profile converged")
            conv_flag = 1
            if final == True :
                itmx = 6
                it_max = it_max
            else :
                itmx = 3
                it_max= it_max
            '''
            for iprime in range(itmx):
                bundle = inputs(calculation='brown')
                bundle.phase_angle(0)
                bundle.gravity(gravity=grav , gravity_unit=u.Unit('m/s**2'))
                bundle.add_pt( temp, pressure)
    
                bundle.premix_atmosphere(opacityclass, df = bundle.inputs['atmosphere']['profile'].loc[:,['pressure','temperature']])
                if cloudy == 1 :
                    metallicity = mh #atmospheric metallicity relative to Solar
                    mean_molecular_weight = mmw # atmospheric mean molecular weight
                    directory ='/Users/sagnickmukherjee/Documents/software/optics'
                    kzz  = get_kzz(pressure, temp,grav,mmw,tidal,flux_net_ir_layer, flux_plus_ir_attop,t_table, p_table, grad, cp, calc_type)
                    bundle.inputs['atmosphere']['profile']['kz'] = kzz
                
            
                    bundle.virga(cld_species,directory, fsed=fsed,mh=metallicity,
                                mmw = mean_molecular_weight,full_output=False)
                DTAU, TAU, W0, COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, \
                W0_no_raman , surf_reflect, ubar0,ubar1,cos_theta, single_phase,multi_phase, \
                frac_a,frac_b,frac_c,constant_back,constant_forward, tridiagonal , \
                wno,nwno,ng,nt, nlevel, ngauss, gauss_wts , mmw =  calculate_atm(bundle, opacityclass)
                
                temp, dtdp, flag_converge, flux_net_ir_layer, flux_plus_ir_attop = t_start(nofczns,nstr,it_max,conv,x_max_mult, 
            rfaci, rfacv, nlevel, temp, pressure, p_table, t_table, 
            grad, cp, tidal,tmin,tmax,dwni, bb , y2, tp, DTAU, TAU, W0, COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman , surf_reflect, ubar0,ubar1,cos_theta, FOPI, single_phase,multi_phase,frac_a,frac_b,frac_c,constant_back,constant_forward, tridiagonal , wno,nwno,ng,nt, ngauss, gauss_wts)
                if (temp <= min(opacityclass.cia_temps)).any():
                    wh = np.where(temp <= min(opacityclass.cia_temps))
                    if len(wh[0]) <= 30 :
                        print(len(wh[0])," points went off the opacity grid. Correcting those.")
                        temp = correct_profile(temp,pressure,wh,min(opacityclass.cia_temps))
                    else :
                        raise Exception('Many points in your profile went off the grid. Try re-starting from a different guess profile. Parametrized profiles can work better sometime as guess profiles.')
            
                
                
                ert = 0.0 # avg temp change
                scalt= 1.0
                dtx= abs(temp-temp_old)
                ert = np.sum(dtx)
                temp_old= np.copy(temp)
                '''
            
            return pressure, temp , dtdp, conv_flag, qvmrs, qvmrs2, all_profiles, all_kzz, opd_cld_climate, g0_cld_climate, w0_cld_climate,flux_net_ir_layer, flux_plus_ir_attop,photo_inputs_dict
            
        
        print("Big iteration is ",min(temp), iii)
    conv_flag = 0
    ## this is supposed to be useless so testing this
    '''
    if final == True :
        itmx = 6
        it_max = it_max
    else :
        itmx = 3
        it_max= it_max
    
    for iprime in range(itmx):
        bundle = inputs(calculation='brown')
        bundle.phase_angle(0)
        bundle.gravity(gravity=grav , gravity_unit=u.Unit('m/s**2'))
        bundle.add_pt( temp, pressure)
        bundle.premix_atmosphere(opacityclass, df = bundle.inputs['atmosphere']['profile'].loc[:,['pressure','temperature']])
        
        if cloudy == 1 :
            metallicity = mh #atmospheric metallicity relative to Solar
            mean_molecular_weight = mmw # atmospheric mean molecular weight
            directory ='/Users/sagnickmukherjee/Documents/software/optics'
            kzz  = get_kzz(pressure, temp,grav,mmw,tidal,flux_net_ir_layer, flux_plus_ir_attop,t_table, p_table, grad, cp, calc_type)
            bundle.inputs['atmosphere']['profile']['kz'] = kzz
        
    
            bundle.virga(cld_species,directory, fsed=fsed,mh=metallicity,
                        mmw = mean_molecular_weight,full_output=False)
        DTAU, TAU, W0, COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman , surf_reflect, ubar0,ubar1,cos_theta, single_phase,multi_phase,frac_a,frac_b,frac_c,constant_back,constant_forward, tridiagonal , wno,nwno,ng,nt, nlevel, ngauss, gauss_wts, mmw =  calculate_atm(bundle, opacityclass)
        
        
        
        temp, dtdp, flag_converge, flux_net_ir_layer, flux_plus_ir_attop = t_start(nofczns,nstr,it_max,conv,x_max_mult, 
    rfaci, rfacv, nlevel, temp, pressure, p_table, t_table, 
    grad, cp, tidal,tmin,tmax,dwni, bb , y2, tp, DTAU, TAU, W0, COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman , surf_reflect, ubar0,ubar1,cos_theta, FOPI, single_phase,multi_phase,frac_a,frac_b,frac_c,constant_back,constant_forward, tridiagonal , wno,nwno,ng,nt, ngauss, gauss_wts)
        if (temp <= min(opacityclass.cia_temps)).any():
            wh = np.where(temp <= min(opacityclass.cia_temps))
            if len(wh[0]) <= 30 :
                print(len(wh[0])," points went off the opacity grid. Correcting those.")
                temp = correct_profile(temp,pressure,wh,min(opacityclass.cia_temps))
            else :
                raise Exception('Many points in your profile went off the grid. Try re-starting from a different guess profile. Parametrized profiles can work better sometime as guess profiles.')
        
        ert = 0.0 # avg temp change
        scalt= 1.0
        dtx= abs(temp-temp_old)
        ert = np.sum(dtx)
        temp_old= np.copy(temp)   
        ert = ert/(float(nlevel)*scalt)
        if ((iii > 0) & (ert < convt) & (taudif < 0.1)) :
            print("Profile converged")
            conv_flag = 1
    '''
    if conv_flag == 0:
        print("Not converged")
    else :
        print("Profile converged")
    
    return pressure, temp , dtdp, conv_flag, qvmrs, qvmrs2, all_profiles, all_kzz, opd_cld_climate, g0_cld_climate, w0_cld_climate,flux_net_ir_layer, flux_plus_ir_attop,photo_inputs_dict
    
def find_strat_deq(mieff_dir, pressure, temp, dtdp , FOPI, nofczns,nstr,x_max_mult,
             t_table, p_table, grad, cp, opacityclass, grav, 
             rfaci, rfacv, nlevel, tidal, tmin, tmax, dwni, bb , y2 , tp, cloudy, cld_species,mh,fsed,flag_hack, quench_levels, kz,mmw, save_profile, all_profiles,self_consistent_kzz ,save_kzz,all_kzz, opd_cld_climate,g0_cld_climate,w0_cld_climate,flux_net_ir_layer, flux_plus_ir_attop,photo_inputs_dict=None,on_fly=False, gases_fly=None ):
    """
    Function iterating on the TP profile by calling tstart and changing opacities as well
    Parameters
    ----------
    mieff_dir: str
        path to directory with mieff files for virga
    it_max : int
        Maximum iterations allowed in the inner no opa change loop
    itmx : int
        Maximum iterations allowed in the outer opa change loop
    conv : float
        
    convt: float
        Convergence criteria , if max avg change in temp is less than this then outer loop converges
        
    nofczns: int
        # of conv zones 
    nstr : array 
        dimension of 20
        NSTR vector describes state of the atmosphere:
        0   is top layer
        1   is top layer of top convective region
        2   is bottom layer of top convective region
        3   is top layer of lower radiative region
        4   is top layer of lower convective region
        5   is bottom layer of lower convective region
    xmaxmult : 
        
    temp : array 
        Guess temperatures to start with
    pressure : array
        Atmospheric pressure
    t_table : array
        Visible flux addition fraction
    nlevel : int
        # of levels
    temp : array
        Guess Temperature array, dimension is nlevel
    pressure : array
        Pressure array
    t_table : array
        Tabulated Temperature array for convection calculations
    p_table : array
        Tabulated pressure array for convection calculations
    grad : array
        Tabulated grad array for convection calculations
    cp : array
        Tabulated cp array for convection calculations
    opacityclass : class
        Opacity class created with jdi.oppanection
    grav : float
        Gravity of planet in SI
    rfaci : float 
        IR flux addition fraction 
    rfacv : float
        Visible flux addition fraction
    nlevel : int
        # of levels, not layers
    tidal : array
        Tidal Fluxes dimension = nlevel
    tmin : float
        Minimum allwed Temp in the profile
    tmax : float
        Maximum allowed Temp in the profile
    dwni : array
        Spectral interval corrections (dimension= nwvno)   
        
    Returns
    -------
    array 
        Temperature array and lapse ratio array if converged
        else Temperature array twice
    """
    # new conditions for this routine

    itmx_strat = 5 #itmx  # outer loop counter
    it_max_strat = 8 # its # inner loop counter # original code is 8
    conv_strat = 5.0 # conv
    convt_strat = 3.0 # convt 
    ip2 = -10 #?
    subad = 0.98 # degree to which layer can be subadiabatic and
                    # we still make it adiabatic
    ifirst = 10-1  # start looking after this many layers from top for a conv zone
                   # -1 is for python referencing
    iend = 0 #?
    final = False

    grad_x, cp_x =convec(temp,pressure, t_table, p_table, grad, cp)
    # grad_x = 
    while dtdp[nstr[1]-1] >= subad*grad_x[nstr[1]-1] :
        ratio = dtdp[nstr[1]-1]/grad_x[nstr[1]-1]

        if ratio > 1.8 :
            print("Move up two levels")
            ngrow = 2
            nstr = growup( 1, nstr , ngrow)
        else :
            ngrow = 1
            nstr = growup( 1, nstr , ngrow)
        
        if nstr[1] < 5 :
            raise ValueError( "Convection zone grew to Top of atmosphere, Need to Stop")
        pressure, temp, dtdp, profile_flag, qvmrs, qvmrs2, all_profiles, all_kzz,opd_cld_climate,g0_cld_climate,w0_cld_climate,flux_net_ir_layer, flux_plus_ir_attop,photo_inputs_dict = profile_deq(mieff_dir,it_max_strat, itmx_strat, conv_strat, convt_strat, nofczns,nstr,x_max_mult,\
            temp,pressure, FOPI, t_table, p_table, grad, cp, opacityclass, grav,rfaci, rfacv, nlevel, tidal, tmin, tmax, dwni, bb , y2 , tp, final, cloudy, cld_species, mh,fsed,flag_hack, quench_levels, kz, mmw, save_profile, all_profiles, self_consistent_kzz,save_kzz,all_kzz,opd_cld_climate,g0_cld_climate,\
            w0_cld_climate,flux_net_ir_layer, flux_plus_ir_attop,photo_inputs_dict,on_fly=on_fly, gases_fly=gases_fly )

    # now for the 2nd convection zone
    dt_max = 0.0 #DTMAX
    i_max = 0 #IMAX
    # -1 in ifirst to include ifirst index
    flag_super = 0
    for i in range(nstr[1]-1, ifirst-1, -1):
        add = dtdp[i] - grad_x[i]
        if add > dt_max and add/grad_x[i] >= 0.02 : # non-neglegible super-adiabaticity
            dt_max = add
            i_max =i
            break
    
    flag_final_convergence =0
    if i_max == 0 or dt_max/grad_x[i_max] < 0.02: # no superadiabaticity, we are done
        flag_final_convergence = 1

    if flag_final_convergence  == 0:
        print(" convection zone status")
        print(nstr[0],nstr[1],nstr[2],nstr[3],nstr[4],nstr[5])
        print(nofczns)

        nofczns = 2
        nstr[4]= nstr[1]
        nstr[5]= nstr[2]
        nstr[1]= i_max
        nstr[2] = i_max
        nstr[3] = i_max #+ 1
        print(nstr)
        if nstr[3] >= nstr[4] :
            #print(nstr[0],nstr[1],nstr[2],nstr[3],nstr[4],nstr[5])
            #print(nofczns)
            raise ValueError("Overlap happened !")
        pressure, temp, dtdp, profile_flag, qvmrs, qvmrs2, all_profiles, all_kzz,opd_cld_climate,g0_cld_climate,w0_cld_climate,flux_net_ir_layer, flux_plus_ir_attop,photo_inputs_dict = profile_deq(mieff_dir,it_max_strat, itmx_strat, conv_strat, convt_strat, nofczns,nstr,x_max_mult,\
            temp,pressure, FOPI, t_table, p_table, grad, cp, opacityclass, grav, \
             rfaci, rfacv, nlevel, tidal, tmin, tmax, dwni, bb , y2 , tp, final, cloudy, cld_species,mh, fsed,flag_hack, quench_levels, kz , mmw,save_profile, all_profiles, self_consistent_kzz, save_kzz,all_kzz,opd_cld_climate,g0_cld_climate,w0_cld_climate,flux_net_ir_layer, flux_plus_ir_attop,photo_inputs_dict,on_fly=on_fly, gases_fly=gases_fly )

        i_change = 1
        while i_change == 1 :
            print("Grow Phase : Upper Zone")
            i_change = 0

            d1 = dtdp[nstr[1]-1]
            d2 = dtdp[nstr[3]]
            c1 = grad_x[nstr[1]-1]
            c2 = grad_x[nstr[3]]

            while ((d1 > subad*c1) or (d2 > subad*c2)):

                if (((d1-c1)>= (d2-c2)) or (nofczns == 1)) :
                    ngrow = 1
                    nstr = growup( 1, nstr , ngrow)

                    if nstr[1] < 3 :
                        raise ValueError( "Convection zone grew to Top of atmosphere, Need to Stop")
                else :
                    ngrow = 1
                    nstr = growdown( 1, nstr , ngrow)

                    if nstr[2] == nstr[4]: # one conv zone
                        nofczns =1
                        nstr[2] = nstr[5]
                        nstr[3] = 0
                        i_change = 1
                print(nstr)
                pressure, temp, dtdp, profile_flag,qvmrs, qvmrs2, all_profiles, all_kzz,opd_cld_climate,g0_cld_climate,w0_cld_climate,flux_net_ir_layer, flux_plus_ir_attop,photo_inputs_dict = profile_deq(mieff_dir,it_max_strat, itmx_strat, conv_strat, convt_strat, nofczns,nstr,x_max_mult,\
            temp,pressure, FOPI, t_table, p_table, grad, cp, opacityclass, grav, \
             rfaci, rfacv, nlevel, tidal, tmin, tmax, dwni, bb , y2 , tp, final, cloudy, cld_species, mh,fsed,flag_hack, quench_levels, kz, mmw, save_profile, all_profiles,self_consistent_kzz,save_kzz,all_kzz,opd_cld_climate,g0_cld_climate,w0_cld_climate,flux_net_ir_layer, flux_plus_ir_attop,photo_inputs_dict,on_fly=on_fly, gases_fly=gases_fly)

                d1 = dtdp[nstr[1]-1]
                d2 = dtdp[nstr[3]]
                c1 = grad_x[nstr[1]-1]
                c2 = grad_x[nstr[3]]
            #Now grow the lower zone.
            while ((dtdp[nstr[4]-1] >= subad*grad_x[nstr[4]-1]) and nofczns > 1):
                
                ngrow = 1
                nstr = growup( 2, nstr , ngrow)
                #Now check to see if two zones have merged and stop further searching if so.
                if nstr[2] == nstr[4] :
                    nofczns = 1
                    nstr[2] = nstr[5]
                    nstr[3] = 0
                    i_change =1
                print(nstr)
                pressure, temp, dtdp, profile_flag, qvmrs, qvmrs2, all_profiles, all_kzz,opd_cld_climate,g0_cld_climate,w0_cld_climate,flux_net_ir_layer, flux_plus_ir_attop,photo_inputs_dict = profile_deq(mieff_dir,it_max_strat, itmx_strat, conv_strat, convt_strat, nofczns,nstr,x_max_mult, \
                                                        temp,pressure, FOPI, t_table, p_table, grad, cp, opacityclass, grav,rfaci, rfacv, nlevel, tidal, tmin, tmax, dwni, bb , y2 , tp, final, cloudy, cld_species, mh,fsed,flag_hack, quench_levels, kz, mmw,save_profile, all_profiles, self_consistent_kzz, save_kzz,all_kzz,opd_cld_climate,g0_cld_climate,w0_cld_climate,flux_net_ir_layer, flux_plus_ir_attop,photo_inputs_dict,on_fly=on_fly, gases_fly=gases_fly)
            

            flag_final_convergence = 1
        
    itmx_strat =6
    it_max_strat = 10
    convt_strat = 2.0
    convt_strat = 2.0
    x_max_mult = x_max_mult/2.0
    ip2 = -10

    final = True
    print("final",nstr)
    pressure, temp, dtdp, profile_flag,qvmrs, qvmrs2, all_profiles, all_kzz,opd_cld_climate,g0_cld_climate,w0_cld_climate,flux_net_ir_layer, flux_plus_ir_attop,photo_inputs_dict = profile_deq(mieff_dir,it_max_strat, itmx_strat, conv_strat, convt_strat, nofczns,nstr,x_max_mult,\
                temp,pressure, FOPI, t_table, p_table, grad, cp,opacityclass, grav, \
                rfaci, rfacv, nlevel, tidal, tmin, tmax, dwni, bb , y2 , tp, final, cloudy, cld_species,mh,fsed,flag_hack,quench_levels,kz, mmw,save_profile, all_profiles, self_consistent_kzz,save_kzz,all_kzz,opd_cld_climate,g0_cld_climate,w0_cld_climate,flux_net_ir_layer, flux_plus_ir_attop,photo_inputs_dict,on_fly=on_fly, gases_fly=gases_fly)

    #    else :
    #        raise ValueError("Some problem here with goto 125")
        
    if profile_flag == 0:
        print("ENDING WITHOUT CONVERGING")
    elif profile_flag == 1:
        print("YAY ! ENDING WITH CONVERGENCE")
        
    bundle = inputs(calculation='brown')
    bundle.phase_angle(0)
    bundle.gravity(gravity=grav , gravity_unit=u.Unit('m/s**2'))
    bundle.add_pt( temp, pressure)

    if photo_inputs_dict['yesorno'] == False:
        k_b = 1.38e-23 # boltzmann constant
        m_p = 1.66e-27 # proton mass

        if len(mmw) < len(temp):
            mmw = np.append(mmw,mmw[-1])
        con  = k_b/(mmw*m_p)

        scale_H = con * temp*1e2/(grav)

        kz = scale_H**2/all_kzz[-len(temp):] ## level mixing timescales

        quench_levels, t_mix = quench_level(pressure, temp, kz ,mmw, grav, return_mix_timescale=True)

        qvmrs, qvmrs2 = bundle.premix_atmosphere_diseq(opacityclass, quench_levels=quench_levels, df = bundle.inputs['atmosphere']['profile'].loc[:,['pressure','temperature']],t_mix=t_mix)
    else :
        bundle.premix_atmosphere(opacityclass, df = bundle.inputs['atmosphere']['profile'].loc[:,['pressure','temperature']])
        

        kz = all_kzz[-len(temp):] ## level mixing timescales
        pc= bundle.call_photochem(temp,pressure,photo_inputs_dict['mh'],photo_inputs_dict['CtoO'],photo_inputs_dict['psurf'],photo_inputs_dict['m_planet'],photo_inputs_dict['r_planet'],photo_inputs_dict['kz'],tstop=photo_inputs_dict['tstop'],filename = photo_inputs_dict['photochem_file'],stfilename =photo_inputs_dict['photochem_stfile'],network = photo_inputs_dict['photochem_network'],network_ct=photo_inputs_dict['photochem_networkct'],first=False,pc=photo_inputs_dict['pc'])
        photo_inputs_dict['pc'] = pc
        all_kzz = np.append(all_kzz, kz)
        quench_levels = np.array([0,0,0,0])
        photo_inputs_dict['pc'] = pc
        
        qvmrs,qvmrs2 = 0,0
    
    if cloudy == 1:
        DTAU, TAU, W0, COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman , surf_reflect, ubar0,ubar1,cos_theta, single_phase,multi_phase,frac_a,frac_b,frac_c,constant_back,constant_forward, tridiagonal , wno,nwno,ng,nt, nlevel, ngauss, gauss_wts, mmw =  calculate_atm_deq(bundle, opacityclass,on_fly=on_fly, gases_fly=gases_fly)
        metallicity = 10**(0.0) #atmospheric metallicity relative to Solar
        mean_molecular_weight = np.mean(mmw) # atmospheric mean molecular weight
        # directory ='/Users/sagnickmukherjee/Documents/GitHub/virga/refr_new661'
        # directory = '/home/jjm6243/dev_virga/'
        directory = mieff_dir

        calc_type =0
        kzz  = get_kzz(pressure, temp,grav,mmw,tidal,flux_net_ir_layer, flux_plus_ir_attop,t_table, p_table, grad, cp, calc_type,nstr)
        bundle.inputs['atmosphere']['profile']['kz'] = kzz


        cld_out = bundle.virga(cld_species,directory, fsed=fsed,mh=metallicity,
                        mmw = mean_molecular_weight,full_output=False)#,climate=True)
        
        opd_now, w0_now, g0_now = cld_out
        df_cld = vj.picaso_format(opd_now, w0_now, g0_now)
        bundle.clouds(df=df_cld)  
    else:
        opd_now,w0_now,g0_now = 0,0,0
    
    #bundle.premix_atmosphere_diseq(opacityclass, quench_levels=quench_levels, df = bundle.inputs['atmosphere']['profile'].loc[:,['pressure','temperature']])
    DTAU, TAU, W0, COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman , surf_reflect, ubar0,ubar1,cos_theta, single_phase,multi_phase,frac_a,frac_b,frac_c,constant_back,constant_forward, tridiagonal , wno,nwno,ng,nt, nlevel, ngauss, gauss_wts, mmw =  calculate_atm_deq(bundle, opacityclass,on_fly=on_fly, gases_fly=gases_fly)
    
    flux_net_v_layer_full, flux_net_v_full, flux_plus_v_full, flux_minus_v_full , flux_net_ir_layer_full, flux_net_ir_full, flux_plus_ir_full, flux_minus_ir_full = get_fluxes(pressure, temp, dwni, bb , y2, tp, tmin, tmax, DTAU, TAU, W0, 
            COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman , surf_reflect, 
            ubar0,ubar1,cos_theta, FOPI, single_phase,multi_phase,frac_a,frac_b,frac_c,constant_back,constant_forward, tridiagonal , 
            wno,nwno,ng,nt,gweight,tweight, nlevel, ngauss, gauss_wts, False, True) #false for reflected, true for thermal


    
    return pressure, temp, dtdp, nstr , flux_plus_ir_full, qvmrs, qvmrs2, bundle.inputs['atmosphere']['profile'], all_profiles, all_kzz,opd_now,w0_now,g0_now,photo_inputs_dict

#@jit(nopython=True, cache=True)
def OH_conc(temp,press,x_h2o,x_h2):
    K = 10**(3.672 - (14791/temp))
    kb= 1.3807e-16 #cgs
    
    x_oh = K * x_h2o * (x_h2**(-0.5)) * (press**(-0.5))
    press_cgs = press*1e6
    
    n = press_cgs/(kb*temp)
    
    return x_oh*n



'''
12/13/2023 - NEB deprecated thi code from SM as it was not proven to help correct 
profiles that went off the grid 

@jit(nopython=True, cache=True)
def correct_profile(temp,pressure,wh,min_temp):
    
    indices = wh[0]
    for i in range(len(indices)):
        if indices[i] == 0:
            temp[indices[i]] = min_temp+0.5
        elif (temp[indices[i]-1] > min_temp) and (temp[indices[i]+1]) > min_temp :
            temp_prev = temp[indices[i]-1]
            temp_next = temp[indices[i]+1]
            press_prev = pressure[indices[i]-1]
            press_next = pressure[indices[i]+1]
            dtdlnp = (temp_next-temp_prev)/np.log(press_next/press_prev)
            temp[indices[i]] = temp_prev +np.log(pressure[indices[i]]/press_prev)*dtdlnp
        else :
            temp[indices[i]] = min_temp+0.5
            temp[indices[i]-1] = min_temp+0.5
            temp[indices[i]+1] = min_temp + 0.5
    


    return temp
'''

