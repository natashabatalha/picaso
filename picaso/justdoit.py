from .atmsetup import ATMSETUP
from .fluxes import get_reflected_1d, get_reflected_3d , get_thermal_1d, get_thermal_3d, get_reflected_new, get_transit_1d
from .wavelength import get_cld_input_grid
from .opacity_factory import create_grid
from .optics import RetrieveOpacities,compute_opacity,RetrieveCKs
from .disco import get_angles_1d, get_angles_3d, compute_disco, compress_disco, compress_thermal
from .justplotit import numba_cumsum, find_nearest_2d, mean_regrid

from virga import justdoit as vj
from scipy.signal import savgol_filter
from scipy.interpolate import RegularGridInterpolator
import scipy as sp
from scipy import special

import requests
import os
import pickle as pk
import numpy as np
import pandas as pd
import copy
import json
import pysynphot as psyn
import astropy.units as u
import astropy.constants as c
import math

__refdata__ = os.environ.get('picaso_refdata')
if not os.path.exists(os.path.join(__refdata__)): 
    raise Exception('You have not downloaded the reference data. You can find it on github here: https://github.com/natashabatalha/picaso/tree/master/reference . If you think you have already downloaded it then you likely just need to set your environment variable. See instructions here: https://natashabatalha.github.io/picaso/installation.html#download-and-link-reference-documentation')

def picaso(bundle,opacityclass, dimension = '1d',calculation='reflected', full_output=False, 
    plot_opacity= False, as_dict=True):
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

    #set approx numbers options (to be used in numba compiled functions)
    single_phase = inputs['approx']['single_phase']
    multi_phase = inputs['approx']['multi_phase']
    raman_approx =inputs['approx']['raman']
    method = inputs['approx']['method']
    stream = inputs['approx']['stream']
    tridiagonal = 0 

    #parameters needed for the two term hg phase function. 
    #Defaults are set in config.json
    f = inputs['approx']['TTHG_params']['fraction']
    frac_a = f[0]
    frac_b = f[1]
    frac_c = f[2]
    constant_back = inputs['approx']['TTHG_params']['constant_back']
    constant_forward = inputs['approx']['TTHG_params']['constant_forward']

    #define delta eddington approximinations 
    delta_eddington = inputs['approx']['delta_eddington']

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

    #set star parameters
    radius_star = inputs['star']['radius']
    F0PI = np.zeros(nwno) + 1.
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
    atm.get_needed_continuum(opacityclass.rayleigh_molecules)

    #get cloud properties, if there are any and put it on current grid 
    atm.get_clouds(wno)

    #Make sure that all molecules are in opacityclass. If not, remove them and add warning
    no_opacities = [i for i in atm.molecules if i not in opacityclass.molecules]
    atm.add_warnings('No computed opacities for: '+','.join(no_opacities))
    atm.molecules = np.array([ x for x in atm.molecules if x not in no_opacities ])

    nlevel = atm.c.nlevel
    nlayer = atm.c.nlayer
    

    if dimension == '1d':
        #lastly grab needed opacities for the problem
        opacityclass.get_opacities(atm)
        #only need to get opacities for one pt profile

        #There are two sets of dtau,tau,w0,g in the event that the user chooses to use delta-eddington
        #We use HG function for single scattering which gets the forward scattering/back scattering peaks 
        #well. We only really want to use delta-edd for multi scattering legendre polynomials. 
        DTAU, TAU, W0, COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman= compute_opacity(
            atm, opacityclass, ngauss=ngauss, stream=stream, delta_eddington=delta_eddington,test_mode=test_mode,raman=raman_approx,
            full_output=full_output, plot_opacity=plot_opacity)


        if  'reflected' in calculation:
            #use toon method (and tridiagonal matrix solver) to get net cumulative fluxes 
            xint_at_top = 0 
            for ig in range(ngauss): # correlated - loop (which is different from gauss-tchevychev angle)
                nlevel = atm.c.nlevel
                if method == 'SH':
                    xint = get_reflected_new(nlevel, nwno, ng, nt, 
                                    DTAU[:,:,ig], TAU[:,:,ig], W0[:,:,ig], COSB[:,:,ig], 
                                    GCOS2[:,:,ig], ftau_cld[:,:,ig], ftau_ray[:,:,ig],
                                    DTAU_OG[:,:,ig], TAU_OG[:,:,ig], W0_OG[:,:,ig], COSB_OG[:,:,ig], 
                                    atm.surf_reflect, ubar0, ubar1, cos_theta, F0PI, 
                                    single_phase, multi_phase, 
                                    frac_a, frac_b, frac_c, constant_back, constant_forward, 
                                    dimension, stream, print_time)
                else:
                    xint = get_reflected_1d(nlevel, wno,nwno,ng,nt,
                                    DTAU[:,:,ig], TAU[:,:,ig], W0[:,:,ig], COSB[:,:,ig],
                                    GCOS2[:,:,ig],ftau_cld[:,:,ig],ftau_ray[:,:,ig],
                                    DTAU_OG[:,:,ig], TAU_OG[:,:,ig], W0_OG[:,:,ig], COSB_OG[:,:,ig],
                                    atm.surf_reflect, ubar0,ubar1,cos_theta, F0PI,
                                    single_phase,multi_phase,
                                    frac_a,frac_b,frac_c,constant_back,constant_forward, tridiagonal)

                xint_at_top += xint*gauss_wts[ig]
            #if full output is requested add in xint at top for 3d plots
            if full_output: 
                atm.xint_at_top = xint_at_top


        if 'thermal' in calculation:

            #use toon method (and tridiagonal matrix solver) to get net cumulative fluxes 
            flux_at_top = 0 
            for ig in range(ngauss): # correlated - loop (which is different from gauss-tchevychev angle)
                
                #remember all OG values (e.g. no delta eddington correction) go into thermal as well as 
                #the uncorrected raman single scattering 
                flux  = get_thermal_1d(nlevel, wno,nwno,ng,nt,atm.level['temperature'],
                                            DTAU_OG[:,:,ig], W0_no_raman[:,:,ig], COSB_OG[:,:,ig], 
                                            atm.level['pressure'],ubar1,
                                            atm.surf_reflect, atm.hard_surface, tridiagonal)
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

                opacityclass.get_opacities(atm_1d)

                dtau, tau, w0, cosb,ftau_cld, ftau_ray, gcos2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, WO_no_raman = compute_opacity(
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

            for ig in range(ngauss): # correlated - loop (which is different from gauss-tchevychev angle)
                #use toon method (and tridiagonal matrix solver) to get net cumulative fluxes 
                xint_at_top  = get_reflected_3d(nlevel, wno,nwno,ng,nt,
                                                DTAU_3d[:,:,:,:,ig], TAU_3d[:,:,:,:,ig], W0_3d[:,:,:,:,ig], COSB_3d[:,:,:,:,ig],GCOS2_3d[:,:,:,:,ig],
                                                FTAU_CLD_3d[:,:,:,:,ig],FTAU_RAY_3d[:,:,:,:,ig],
                                                DTAU_OG_3d[:,:,:,:,ig], TAU_OG_3d[:,:,:,:,ig], W0_OG_3d[:,:,:,:,ig], COSB_OG_3d[:,:,:,:,ig],
                                                atm.surf_reflect, ubar0,ubar1,cos_theta, F0PI,
                                                single_phase,multi_phase,
                                                frac_a,frac_b,frac_c,constant_back,constant_forward, tridiagonal)
                #if full output is requested add in xint at top for 3d plots
            if full_output: 
                atm.xint_at_top = xint_at_top

        elif 'thermal' in calculation:
            for ig in range(ngauss): # correlated - loop (which is different from gauss-tchevychev angle)
                #remember all OG values (e.g. no delta eddington correction) go into thermal as well as 
                #the uncorrected raman single scattering 
                flux_at_top  = get_thermal_3d(nlevel, wno,nwno,ng,nt,TLEVEL_3d,
                                            DTAU_OG_3d[:,:,:,:,ig], W0_no_raman_3d[:,:,:,:,ig], COSB_OG_3d[:,:,:,:,ig], 
                                            PLEVEL_3d,ubar1, tridiagonal)

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
        thermal = compress_thermal(nwno,ubar1, flux_at_top, gweight, tweight)
        returns['thermal'] = thermal
        returns['effective_temperature'] = (np.trapz(x=1/wno[::-1], y=thermal[::-1])/5.67e-5)**0.25

        if full_output: 
            atm.thermal_flux_planet = thermal
            

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

    return returns


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

    #set approx numbers options (to be used in numba compiled functions)
    single_phase = inputs['approx']['single_phase']
    multi_phase = inputs['approx']['multi_phase']
    method = inputs['approx']['method']
    stream = inputs['approx']['stream']
    tridiagonal = 0 
    raman_approx = 2

    #parameters needed for the two term hg phase function. 
    #Defaults are set in config.json
    f = inputs['approx']['TTHG_params']['fraction']
    frac_a = f[0]
    frac_b = f[1]
    frac_c = f[2]
    constant_back = inputs['approx']['TTHG_params']['constant_back']
    constant_forward = inputs['approx']['TTHG_params']['constant_forward']

    #define delta eddington approximinations 
    delta_eddington = inputs['approx']['delta_eddington']

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
    atm.get_needed_continuum(opacityclass.rayleigh_molecules)

    #get cloud properties, if there are any and put it on current grid 
    atm.get_clouds(wno)

    #Make sure that all molecules are in opacityclass. If not, remove them and add warning
    no_opacities = [i for i in atm.molecules if i not in opacityclass.molecules]
    atm.add_warnings('No computed opacities for: '+','.join(no_opacities))
    atm.molecules = np.array([ x for x in atm.molecules if x not in no_opacities ])

    nlevel = atm.c.nlevel
    nlayer = atm.c.nlayer
    
    #lastly grab needed opacities for the problem
    opacityclass.get_opacities(atm)
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
        #at_pressures = np.zeros(shape[1])
        #ind_gas = find_nearest_2d(cumsum_taus[i] , at_tau)
        
        #for iw in range(shape[1]):
        #    at_pressures[iw] = pressure[ind_gas[iw]]
        at_pressures=[]
        for iw in range(shape[1]): 
            at_pressures += [np.interp([at_tau],cumsum_taus[i][:,iw],
                                pressure )[0]]

        at_pressure_array[i] = at_pressures

    return taus_by_species, cumsum_taus, at_pressure_array

def opannection(ck=False, wave_range = None, filename_db = None, raman_db = None, 
                resample=1, ck_db=None):
    """
    Sets up database connection to opacities. 

    Parameters
    ----------
    ck : bool 
        Use premixed ck tables (beta)
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
    """
    inputs = json.load(open(os.path.join(__refdata__,'config.json')))

    if not ck: 
        #only allow raman if no correlated ck is used 
        if isinstance(raman_db,type(None)): raman_db = os.path.join(__refdata__, 'opacities', inputs['opacities']['files']['raman'])
        
        if isinstance(filename_db,type(None)): 
            filename_db = os.path.join(__refdata__, 'opacities', inputs['opacities']['files']['opacity'])
            if not os.path.isfile(filename_db):
                raise Exception('The opacity file does not exist: '  + filename_db+' The default is to a file opacities.db in reference/opacity/. If you have an older version of PICASO your file might be called opacity.db. Consider just adding the correct path to filename_db=')
        elif not isinstance(filename_db,type(None) ): 
            if not os.path.isfile(filename_db):
                raise Exception('The opacity file you have entered does not exist: '  + filename_db)

        if resample != 1:
            print("YOU ARE REQUESTING RESAMPLING!!")

        opacityclass=RetrieveOpacities(
                    filename_db, 
                    raman_db,
                    wave_range = wave_range, resample = resample)
    else: 
        if isinstance(filename_db,type(None)): 
            filename_db = os.path.join(__refdata__, 'opacities', inputs['opacities']['files']['ktable_continuum'])
        opacityclass=RetrieveCKs(
                    ck_db, 
                    filename_db, 
                    wave_range = wave_range)

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
            

    def phase_angle(self, phase=0,num_gangle=10, num_tangle=1,symmetry=False):
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
        """
        if (phase > 2*np.pi) or (phase<0): raise Exception('Oops! you input a phase angle greater than 2*pi or less than 0. Please make sure your inputs are in radian units: 0<phase<2pi')
        if ((num_tangle==1) or (num_gangle==1)): 
            #this is here so that we can compare to older models 
            #this model only works for full phase calculations

            if phase!=0: raise Exception('This method is faster because it makes use of symmetry \
                and only computs one fourth of the sphere. ')
            if num_gangle==1:  raise Exception('Please resubmit your run with num_tangle=1. \
                Chebyshev angles are used for 3d implementation only.')

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
        self.inputs['approx']['raman'] = 2 #turning off raman scattering
        self.phase_angle(0) #auto turn on zero phase
        self.inputs['calculation'] ='climate'


    def setup_nostar(self):
        """
        Turns off planet specific things, so program can run as usual
        """
        self.inputs['approx']['raman'] = 2 #turning off raman scattering
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
        semi_major=None, semi_major_unit = None,
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
            wno_star = 1e4/sp.wave[::-1] #convert to wave number and flip
            flux_star = sp.flux[::-1]*1e8    #flip here and convert to ergs/cm3/s to get correct order
        else: 
            raise Exception("Must enter 1) filename,w_unit & f_unit OR 2)temp, metal & logg ")

        wno_planet = opannection.wno
        #this adds stellar shifts 'self.raman_stellar_shifts' to the opacity class
        #the cross sections are computed later 
        if self.inputs['approx']['raman'] == 0: 
            max_shift = np.max(wno_planet)+6000 #this 6000 is just the max raman shift we could have 
            min_shift = np.min(wno_planet) -2000 #it is just to make sure we cut off the right wave ranges
            #do a fail safe to make sure that star is on a fine enough grid for planet case 
            fine_wno_star = np.linspace(min_shift, max_shift, len(wno_planet)*5)
            fine_flux_star = np.interp(fine_wno_star,wno_star, flux_star)
            opannection.compute_stellar_shits(fine_wno_star, fine_flux_star)
        elif 'climate' in self.inputs['calculation']: 
            #stellar flux of star 
            nrg_flux = 0.5*np.diff(1/wno_planet)*(flux_star[0:-1]+flux_star[1:])
            nrg_flux 
            fine_wno_star = wno_planet
            _x,fine_flux_star = mean_regrid(wno_star, flux_star,newx=wno_planet)  
            opannection.unshifted_stellar_spec = fine_flux_star            
        else :
            max_shift = np.max(wno_planet)+1  
            min_shift = np.min(wno_planet) -1 
            #gaurd against nans bcause stellar spectrum is too low res
            fine_wno_star = np.linspace(min_shift, max_shift, len(wno_planet)*5)
            fine_flux_star = np.interp(fine_wno_star, wno_star, flux_star)
            _x,fine_flux_star = mean_regrid(fine_wno_star, fine_flux_star,newx=wno_planet)  
            opannection.unshifted_stellar_spec =fine_flux_star

        self.inputs['star']['database'] = database
        self.inputs['star']['temp'] = temp
        self.inputs['star']['logg'] = logg
        self.inputs['star']['metal'] = metal
        self.inputs['star']['radius'] = r 
        self.inputs['star']['radius_unit'] = radius_unit 
        self.inputs['star']['flux'] = fine_flux_star 
        self.inputs['star']['wno'] = fine_wno_star 
        self.inputs['star']['semi_major'] = semi_major 
        self.inputs['star']['semi_major_unit'] = semi_major_unit 

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
            (Optional) List of molecules to ignore from file
        verbose : bool 
            (Optional) prints out warnings. Default set to True
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

        if not isinstance(exclude_mol, type(None)):
            df = df.drop(exclude_mol, axis=1)
            self.inputs['atmosphere']['exclude_mol'] = exclude_mol

        self.inputs['atmosphere']['profile'] = df.sort_values('pressure').reset_index(drop=True)

        #lastly check to see if the atmosphere is non-H2 dominant. 
        #if it is, let's turn off Raman scattering for the user. 
        if (("H2" not in df.keys()) and (self.inputs['approx']['raman'] != 2)):
            if verbose: print("Turning off Raman for Non-H2 atmosphere")
            self.inputs['approx']['raman'] = 2
        elif (("H2" in df.keys()) and (self.inputs['approx']['raman'] != 2)): 
            if df['H2'].min() < 0.7: 
                if verbose: print("Turning off Raman for Non-H2 atmosphere")
                self.inputs['approx']['raman'] = 2

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
        self.inputs['approx']['raman'] = 2

        plevel = self.inputs['atmosphere']['profile']['pressure'].values
        tlevel =self.inputs['atmosphere']['profile']['temperature'].values
        
        pt_pairs = []
        i=0
        for ip,it,p,t in zip(np.concatenate([list(range(opa.max_pc))*opa.max_tc]), 
                        np.concatenate([[it]*opa.max_pc for it in range(opa.max_tc)]),
                        opa.pressures, 
                        np.concatenate([[it]*opa.max_pc for it in opa.temps])):
            
            if p!=0 : pt_pairs += [[i,ip,it,p/1e3,t]];i+=1

        ind_pt_log = np.array([min(pt_pairs, 
                    key=lambda c: math.hypot(np.log(c[-2])- np.log(coordinate[0]), 
                                             c[-1]-coordinate[1]))[0:3] 
                        for coordinate in  zip(plevel,tlevel)])           
        ind_chem = ind_pt_log[:,0]

        self.inputs['atmosphere']['profile'][opa.full_abunds.keys()] = opa.full_abunds.iloc[ind_chem,:].reset_index(drop=True)

    def sonora(self, sonora_path, teff, chem='low'):
        """
        This queries Sonora temperature profile that can be downloaded from profiles.tar on 
        Zenodo: [profile.tar file](https://zenodo.org/record/1309035#.Xo5GbZNKjGJ)

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
        if ((len(flist)<300) & ('t400g3160nc_m0.0.cmp.gz' not in flist)):
            raise Exception('Oops! Looks like the sonora path you specified does not contain the ~390 .gz model files from Zenodo. Please untar the profile.tar file here https://zenodo.org/record/1309035#.Xo5GbZNKjGJ and point to this file path as your input.')

        flist = [i.split('/')[-1] for i in flist if 'gz' in i]
        ts = [i.split('g')[0][1:] for i in flist if 'gz' in i]
        gs = [i.split('g')[1].split('nc')[0] for i in flist]

        pairs = [[ind, float(i),float(j)] for ind, i, j in zip(range(len(ts)), ts, gs)]
        coordinate = [teff, g]

        get_ind = min(pairs, key=lambda c: math.hypot(c[1]- coordinate[0], c[2]-coordinate[1]))[0]

        build_filename = 't'+ts[get_ind]+'g'+gs[get_ind]+'nc_m0.0.cmp.gz'
        ptchem = pd.read_csv(os.path.join(sonora_path,build_filename),delim_whitespace=True,compression='gzip')
        ptchem = ptchem.rename(columns={'P(BARS)':'pressure',
                                        'TEMP':'temperature',
                                        'HE':'He'})

        self.inputs['atmosphere']['profile'] = ptchem.loc[:,['pressure','temperature']]
        if chem == 'high':
            self.channon_grid_high(filename=os.path.join(__refdata__, 'chemistry','grid75_feh+000_co_100_highP.txt'))
        elif chem == 'low':
            self.channon_grid_low(filename=os.path.join(__refdata__,'chemistry','visscher_abunds_m+0.0_co1.0' ))
        self.inputs['atmosphere']['sonora_filename'] = build_filename
        self.nlevel = ptchem.shape[0]

    def chemeq(self, CtoO, Met):
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
        #df = self.inputs['atmosphere']['profile']
        df = self.inputs['atmosphere']['profile'].sort_values('pressure').reset_index(drop=True)
        self.nlevel = df.shape[0]
        
        player = df['pressure'].values
        tlayer  = df['temperature'].values
        
        channon = pd.read_csv(filename,delim_whitespace=True)
        #get molecules list from channon db
        mols = list(channon.keys())
        mols.pop(mols.index('pressure'))
        mols.pop(mols.index('temperature'))
        
        #add molecules to df
        df = df.reindex(columns = ['pressure','temperature']+mols) 

        #add pt_ids so we can grab the right points
        pt_pairs = channon.loc[:,['pressure','temperature']]
        pt_pairs['pt_id'] = np.arange(pt_pairs.shape[0])
        channon['pt_id'] = pt_pairs['pt_id']

        pt_pairs = pt_pairs.values#p,t,id
        
        #find index corresponding to each pair
        ind_pt=[min(pt_pairs, 
                    key=lambda c: math.hypot(c[0]- coordinate[0], c[1]-coordinate[1]))[2] 
                        for coordinate in  zip(np.log10(player),tlayer)]
        #build dataframe with chemistry
        for i in range(df.shape[0]):
            pair_for_layer = ind_pt[i]
            df.loc[i,mols] = channon.loc[channon['pt_id'] == pair_for_layer,mols].values
        df['ptid'] = ind_pt
        
        self.inputs['atmosphere']['profile'] = df
    def chemeq_visscher(self, c_o, log_mh, interp_window = 11, interp_poly=2):
        """
        Find nearest neighbor from visscher grid

        JUNE 2015
        MODELS BASED ON 1060-POINT MARLEY GRID

        GRAPHITE ACTIVITY ADDED IN TEXT FILES (AFTER OCS)
        "ABUNDANCE" INDICATES CONDENSATION CONDITION (O OR 1)

        CURRENT GRID
        FE/H: 0.0, 0.5, 1.0, 1.5, 1.7, 2.0
        C/O: 0.5X, 1.0X, 1.5X, 2.0X, 2.5X

        C/O RATIO IS RELATIVE TO SOLAR C/O RATIO OF
        CARBON = 7.19E6 ATOMS
        OXYGEN = 1.57E7 ATOMS

        NOTE THAT THE C/O RATIO IS ADJUSTED BY SIMPLY MULTIPLYING BY C/O FACTOR
        THIS MAY YIELD EFFECTIVE METALLICITIES SLIGHTLY HIGHER THAN THE DEFINED METALLICITY
        
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
        cos = np.array([0.5,1.0,1.5,2.0,2.5])
        #allowable fehs
        fehs = np.array([0.0,0.5,1.0,1.5,1.7,2.0])

        if log_mh > max(fehs): 
            raise Exception('Choose a log metallicity less than 2.0')
        if c_o > max(cos): 
            raise Exception('Choose a C/O less than 2.5xSolar')

        grid_co = cos[np.argmin(np.abs(cos-c_o))]
        grid_feh = fehs[np.argmin(np.abs(fehs-log_mh))]
        str_co = str(grid_feh).replace('.','')
        str_fe = str(grid_co).replace('.','')

        filename = os.path.join(__refdata__,'chemistry','visscher_grid',
            f'2015_06_1060grid_feh_{str_co}_co_{str_fe}.txt')

        header = pd.read_csv(filename).keys()[0]
        cols = header.replace('T (K)','temperature').replace('P (bar)','pressure').split()
        a = pd.read_csv(filename,delim_whitespace=True,skiprows=1,header=None, names=cols)
        a['pressure']=10**a['pressure']

        mols = list(a.loc[:, ~a.columns.isin(['Unnamed: 0','pressure','temperature'])].keys())
        #get pt from what users have already input
        player_tlayer = self.inputs['atmosphere']['profile'].loc[:,['pressure','temperature']].sort_values('pressure').reset_index(drop=True)
        self.nlevel = player_tlayer.shape[0]

        #loop through all pressure and temperature layers
        for i in player_tlayer.index:
            p,t = player_tlayer.loc[i,['pressure','temperature']]
            #at each layer compute the distance to each point in the visscher table
            #create a weight that is 1/d**2
            a['weight'] = (1/((np.log10(a['pressure']) - np.log10(p))**2 + 
                                       (a['temperature'] - t)**2))
            #sort by the weight and only pick the TWO nearest neighbors. 
            #Note, there was a sensitivity study to determine how many nearest neighbors to choose 
            #It was determined that no more than 2 is best. 
            w = a.sort_values(by='weight',ascending=False).iloc[0:2,:]
            #now loop through all the molecules
            for im in mols:
                #compute the weighted mean of the mixing ratio
                weighted_mean = (np.sum(w['weight']*(w.loc[:,im]))/np.sum(w['weight']))
                player_tlayer.loc[i,im] = weighted_mean
        #now pass through everything and use savgol filter to average 
        for im in mols:
            y = np.log10(player_tlayer.loc[:,im])
            s = savgol_filter(y , interp_window, interp_poly)
            player_tlayer.loc[:,im] = 10**s #

        self.inputs['atmosphere']['profile'] = player_tlayer


    def channon_grid_low(self, filename = None, interp_window = 11, interp_poly=2):
        """
        Interpolate from visscher grid
        """
        if isinstance(filename, type(None)):filename= os.path.join(__refdata__,'chemistry','visscher_abunds_m+0.0_co1.0')
        a = pd.read_csv(filename)
        mols = list(a.loc[:, ~a.columns.isin(['Unnamed: 0','pressure','temperature'])].keys())
        #get pt from what users have already input
        player_tlayer = self.inputs['atmosphere']['profile'].loc[:,['pressure','temperature']].sort_values('pressure').reset_index(drop=True)
        self.nlevel = player_tlayer.shape[0]

        #loop through all pressure and temperature layers
        for i in player_tlayer.index:
            p,t = player_tlayer.loc[i,['pressure','temperature']]
            #at each layer compute the distance to each point in the visscher table
            #create a weight that is 1/d**2
            a['weight'] = (1/((np.log10(a['pressure']) - np.log10(p))**2 + 
                                       (a['temperature'] - t)**2))
            #sort by the weight and only pick the TWO nearest neighbors. 
            #Note, there was a sensitivity study to determine how many nearest neighbors to choose 
            #It was determined that no more than 2 is best. 
            w = a.sort_values(by='weight',ascending=False).iloc[0:2,:]
            #now loop through all the molecules
            for im in mols:
                #compute the weighted mean of the mixing ratio
                weighted_mean = (np.sum(w['weight']*(w.loc[:,im]))/np.sum(w['weight']))
                player_tlayer.loc[i,im] = weighted_mean
        #now pass through everything and use savgol filter to average 
        for im in mols:
            y = np.log10(player_tlayer.loc[:,im])
            s = savgol_filter(y , interp_window, interp_poly)
            player_tlayer.loc[:,im] = 10**s #

        self.inputs['atmosphere']['profile'] = player_tlayer

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
        f1 = 2.0/3.0 + 2.0/(3.0*gamma1)*(1.+(gamma1*tau/2.0-1.0)*sp.exp(-gamma1*tau))+2.0*gamma1/3.0*(1.0-tau**2.0/2.0)*special.expn(2.0,gamma1*tau)
        f2 = 2.0/3.0 + 2.0/(3.0*gamma2)*(1.+(gamma2*tau/2.0-1.0)*sp.exp(-gamma2*tau))+2.0*gamma2/3.0*(1.0-tau**2.0/2.0)*special.expn(2.0,gamma2*tau)
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


    def atmosphere_3d(self, dictionary=None, filename=None):
        """
        Builds a dataframe and makes sure that minimum necessary parameters have been suplied. 

        Parameters
        ----------
        df : dict
            (Optional) Dataframe with volume mixing ratios and pressure, temperature profile. 
            Must contain pressure (bars) and at least one molecule
        filename : str 
            (Optional) Pickle filename that is a dictionary structured: 
            dictionary[int(lat) in degrees][int(lon) in degrees] = pd.DataFrame({'pressure':[], 'temperature': [],
            'H2O':[]....}). As well as df['phase_angle'] = 0 (in radians).
            Therefore, the latitude and longitude keys should be INTEGERS taken from the lat/longs calculated in 
            inputs.phase_angle! Any other meta data should be  represented by a string.
            Must contain pressure and at least one molecule

        Examples
        --------

        >>import picaso.justdoit as jdi
        >>new = jdi.inputs()
        >>new.phase_angle(0) #loads the geometry you need to get your 3d model on
        >>lats= new.inputs['disco']['latitude']*180/np.pi).astype(int)
        >>>lons = (new.inputs['disco']['longitude']*180/np.pi).astype(int)

        Build an empty dictionary to be filled later 

        >>dict_3d = {la: {lo : [] for lo in lons} for la in lats} #build empty one 
        >>dict_3d['phase_angle'] = 0 #this is a double check for consistency 
        
        Now you need to bin your GCM output on this grid and fill the dictionary with the necessary dataframes. For example:
        
        >>dict_3d[lats[0]][lons[0]] = pd.DataFrame({'pressure':[], 'temperature': [],'H2O':[]})

        Once this is filled you can add it to this function 

        >>new.atmosphere_3d(dictionary=dict_3d)
        """

        if (isinstance(dictionary, dict) and isinstance(filename, type(None))):
            df3d = dictionary

        elif (isinstance(dictionary, type(None)) and isinstance(filename, str)):
            df3d = pk.load(open(filename,'rb'))
        else:
            raise Exception("Please input either a dict using dictionary or a str using filename")

        #get lats and lons out of dictionary and puts them in reverse 
        #order so that they match what comes out of disco calcualtion
        lats = np.sort(list(i for i in df3d.keys() if isinstance(i,int))) #latitude is positive to negative 
        lons = np.sort(list(i for i in df3d[lats[0]].keys() if isinstance(i,int))) #longitude is negative to positive 
        #check they are about the same as the ones computed in phase angle 
        df3d_nt = len(lats)
        df3d_ng =  len(df3d[lats[0]].keys())

        assert self.inputs['disco']['num_gangle'] == int(df3d_nt), 'Number of gauss angles input does not match creation of 3d input file. Check function `inputs.phase_angle()`. num_gangle=10 is set by default and you may have to change it.'
        assert self.inputs['disco']['num_tangle']  == int(df3d_ng), 'Number of Tchebyshev angles does not match creation of 3d input file.  Check function `inputs.phase_angle()`.  num_tangle=10 is set by default and you may have to change it.'
        
        self.nlevel=df3d[lats[0]][lons[0]].shape[0]

        #now check that the lats and longs are about the same
        for ilat ,nlat in  zip(np.sort(self.inputs['disco']['latitude']*180/np.pi), lats): 
            np.testing.assert_almost_equal(int(ilat) ,nlat, decimal=0,err_msg='Latitudes from dictionary(units degrees) are not the same as latitudes computed in inputs.phase_angle', verbose=True)

        for ilon ,nlon in  zip(np.sort(self.inputs['disco']['longitude']*180/np.pi), lons): 
            np.testing.assert_almost_equal(int(ilon) ,nlon , decimal=0,err_msg='Longitudes from dictionary(units degrees) are not the same as longitudes computed in inputs.phase_angle', verbose=True)
        
        self.inputs['atmosphere']['profile'] = df3d

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
                assert df.shape[0] == (self.nlevel-1)*196, "There are {0} rows in the df, which does not equal {1} layers x 196 eddysed wave pts".format(df.shape[0], self.nlevel-1) 
                
                self.inputs['clouds']['wavenumber'] = get_cld_input_grid('wave_EGP.dat')

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
        fsed=1, mh=1, mmw=2.2,kz_min=1e5,sig=2, full_output=False): 
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
        """
        
        cloud_p = vj.Atmosphere(condensates,fsed=fsed,mh=mh,
                 mmw = mmw, sig =sig) 

        if 'kz' not in self.inputs['atmosphere']['profile'].keys():
            raise Exception ("Must supply kz to atmosphere/chemistry DataFrame, \
                if running `virga` through `picaso`. This should go in the \
                same place that you specified you pressure-temperature profile. \
                Alternatively, you can manually add it by doing \
                `case.inputs['atmosphere']['profile']['kz'] = KZ`")
        df = self.inputs['atmosphere']['profile'].loc[:,['pressure','temperature','kz']]
        
        cloud_p.gravity(gravity=self.inputs['planet']['gravity'],
                 gravity_unit=u.Unit(self.inputs['planet']['gravity_unit']))#
        
        cloud_p.ptk(df =df, kz_min = kz_min)
        out = vj.compute(cloud_p, as_dict=full_output,
                          directory=directory)

        if not full_output:
            opd, w0, g0 = out
            df = vj.picaso_format(opd, w0, g0)
        else: 
            opd, w0, g0 = out['opd_per_layer'],out['single_scattering'],out['asymmetry']
            df = vj.picaso_format(opd, w0, g0)

        self.clouds(df=df)

        if full_output : return out

    def clouds_3d(self, filename = None, dictionary =None):
        """
        Cloud specification for the model. Clouds are parameterized by a single scattering albedo (w0), 
        an assymetry parameter (g0), and a total extinction per layer (opd).

        g0,w0, and opd are both wavelength and pressure dependent. Our cloud models come 
        from eddysed. Their output looks like this (where level 1=TOA, and wavenumber1=Smallest number)

        level wavenumber opd w0 g0
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
        dictionary : dict
            (Optional) Same as what would be included in the file, but in dict form

        Examples
        --------

        >>import picaso.justdoit as jdi
        >>new = jdi.inputs()
        >>new.phase_angle(0) #loads the geometry you need to get your 3d model on
        >>lats, lons = int(new.inputs['disco']['latitude']*180/np.pi), int(new.inputs['disco']['longitude']*180/np.pi)

        Build an empty dictionary to be filled later 

        >>dict_3d = {la: {lo : for lo in lons} for la in lats} #build empty one 
        >>dict_3d['phase_angle'] = 0 #add this for a consistensy check
        
        Now you need to bin your 3D clouds GCM output on this grid and fill the dictionary with the necessary dataframes. For example:
        
        >>dict_3d[lats[0]][lons[0]] = pd.DataFrame({'pressure':[], 'temperature': [],'H2O':[]})

        Once this is filled you can add it to this function 

        >>new.clouds_3d(dictionary=dict_3d)
        """

        #first complete options if user inputs dataframe or dict 
        if (isinstance(filename, str) & isinstance(dictionary, type(None))): 
            df = pk.load(open(filename,'rb'))
        elif (isinstance(filename, type(None)) & isinstance(dictionary, dict)):
            df = dictionary
        else: 
            raise Exception("Input must be a filename or a dictionary")

        cols = df[list(df.keys())[0]][list(df[list(df.keys())[0]].keys())[0]].keys()

        assert 'g0' in cols, "Please make sure g0 is a named column in cld file"
        assert 'w0' in cols, "Please make sure w0 is a named column in cld file"
        assert 'opd' in cols, "Please make sure opd is a named column in cld file"
        #again, check lat and lon in comparison to the computed ones 
        #get lats and lons out of dictionary 
        lats = np.sort(list(i for i in df.keys() if isinstance(i,int)))
        lons = np.sort(list(i for i in df[lats[0]].keys() if isinstance(i,int)))    
                
        for ilat ,nlat in  zip(np.sort(self.inputs['disco']['latitude']*180/np.pi), lats): 
            np.testing.assert_almost_equal(int(ilat) ,nlat, decimal=0,err_msg='Latitudes from dictionary(units degrees) are not the same as latitudes computed in inputs.phase_angle', verbose=True)
            
        for ilon ,nlon in  zip(np.sort(self.inputs['disco']['longitude']*180/np.pi), lons): 
            np.testing.assert_almost_equal(int(ilon) ,nlon , decimal=0,err_msg='Latitudes from dictionary(units degrees) are not the same as latitudes computed in inputs.phase_angle', verbose=True)

        #CHECK SIZES
        #if it's a user specified pressure and wavenumber
        if (('pressure' in cols) & ('wavenumber' in cols)):
            for i in df.keys():
                #loop through different lat and longs and make sure that each one of them is ordered correct 
                for j in df[i].keys():
                    df[i][j] = df[i][j].sort_values(['pressure', 'wavenumber']).reset_index(drop=True)
                    self.inputs['clouds']['wavenumber'] = df[i][j]['wavenumber'].unique()
                    nwave = len(self.inputs['clouds']['wavenumber'])
                    nlayer = len(df[i][j]['pressure'].unique())
                    assert df[i][j].shape[0] == (self.nlevel-1)*nwave, "There are {0} rows in the df, which does not equal {1} layers previously specified x {2} wave pts".format(df[i][j].shape[0], self.nlevel-1, nwave) 
                
        #if its eddysed, make sure there are 196 wave points 
        #this does not reorder so it assumes that 
        else: 
            shape = df[list(df.keys())[0]][list(df[list(df.keys())[0]].keys())[0]].shape[0]
            assert shape == (self.nlevel-1)*196, "There are {0} rows in the df, which does not equal {1} layers x 196 eddysed wave pts".format(shape, self.nlevel-1) 
            
            #get wavelength grid from ackerman code
            self.inputs['clouds']['wavenumber'] = get_cld_input_grid('wave_EGP.dat')

            #add it to input
        self.inputs['clouds']['profile'] = df

    def approx(self,single_phase='TTHG_ray',multi_phase='N=2',delta_eddington=True,
        raman='pollack',tthg_frac=[1,-1,2], tthg_back=-0.5, tthg_forward=1,
        p_reference=1,method='Toon', stream=2):
        """
        This function sets all the default approximations in the code. It transforms the string specificatons
        into a number so that they can be used in numba nopython routines. 

        For `str` cases such as `TTHG_ray` users see all the options by using the function `single_phase_options`
        or `multi_phase_options`, etc. 

        single_phase : str 
            Single scattering phase function approximation 
        multi_phase : str 
            Multiple scattering phase function approximation 
        delta_eddington : bool 
            Turns delta-eddington on and off
        raman : str 
            Uses various versions of raman scattering 
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
            Toon ('Toon') or spherical harmonics ('SH'). 
        stream : int 
            Two stream or four stream (options are 2 or 4). For 4 stream need to set method='SH'
        """

        self.inputs['approx']['single_phase'] = single_phase_options(printout=False).index(single_phase)
        self.inputs['approx']['multi_phase'] = multi_phase_options(printout=False).index(multi_phase)
        self.inputs['approx']['delta_eddington'] = delta_eddington
        self.inputs['approx']['raman'] =  raman_options().index(raman)
        self.inputs['approx']['method'] = method
        self.inputs['approx']['stream'] = stream
 
        if isinstance(tthg_frac, (list, np.ndarray)):
            if len(tthg_frac) == 3:
                self.inputs['approx']['TTHG_params']['fraction'] = tthg_frac
            else:
                raise Exception('tthg_frac should be of length=3 so that : tthg_frac[0] + tthg_frac[1]*g_b^tthg_frac[2]')
        else: 
            raise Exception('tthg_frac should be a list or ndarray of length=3')

        self.inputs['approx']['TTHG_params']['constant_back'] = tthg_back
        self.inputs['approx']['TTHG_params']['constant_forward']=tthg_forward

        self.inputs['approx']['p_reference']= p_reference

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
        try: 
            #if there is not star, the only picaso option to run is thermal emission
            if self.inputs['star']['radius'] == 'nostar':
                calculation = 'thermal' 
        except KeyError: 
            pass

        try: 
            #phase angles dont need to be specified for thermal emission or transmission
            a = self.inputs['phase_angle']
        except KeyError: 
            if calculation != 'reflected':
                self.phase_angle(0)
            else: 
                raise Exception("Phase angle not specified. It is needed for "+calculation)
        
        try:
            a = self.inputs['surface_reflect']
        except KeyError:
            #I don't make people add this as an input so adding a default here if it hasnt
            #been run 
            self.inputs['surface_reflect'] = 0 
            self.inputs['hard_surface'] = 0 

            
        return picaso(self, opacityclass,dimension=dimension,calculation=calculation,
            full_output=full_output, plot_opacity=plot_opacity, as_dict=as_dict)

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

def load_planet(df, opacity, phase_angle = 0, stellar_db='phoenix', verbose=False,  **planet_kwargs):
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

        stellar_db = 'phoenix'

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
def HJ_pt_3d():
    """Function to get Jupiter's PT profile"""
    return os.path.join(__refdata__, 'base_cases','HJ_3d.pt')
def HJ_cld():
    """Function to get rough Jupiter Cloud model with fsed=3"""
    return os.path.join(__refdata__, 'base_cases','HJ.cld')
def single_phase_options(printout=True):
    """Retrieve all the options for direct radation"""
    if printout: print("Can also set functional form of forward/back scattering in approx['TTHG_params']")
    return ['cahoy','OTHG','TTHG','TTHG_ray']
def multi_phase_options(printout=True):
    """Retrieve all the options for multiple scattering radiation"""
    if printout: print("Can also set delta_eddington=True/False in approx['delta_eddington']")
    return ['N=2','N=1']
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

def methodology_options(printout=True):
    """Retrieve all the options for methodology"""
    if printout: print("Can calculate spectrum using Toon 1989 methodology or sperhical harmonics")
    return ['Toon','SH']
def stream_options(printout=True):
    """Retrieve all the options for stream"""
    if printout: print("Can use 2-stream or 4-stream sperhical harmonics")
    return [2,4]

