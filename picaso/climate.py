import numpy as np 
import warnings
from numba import jit, vectorize,float32,float64
from numba.experimental import jitclass
from numpy import exp, zeros, where, sqrt, cumsum , pi, outer, sinh, cosh, min, dot, array,log,log10
import astropy.units as u
import virga.justdoit as vj

from .fluxes import get_reflected_1d,get_thermal_1d
#from .fluxes import get_thermal_1d_newclima, get_thermal_1d_gfluxi,get_reflected_1d_gfluxv #deprecated
from .calculate_atm import calculate_atm, Atmosphere_Tuple
from .clouds import get_kzz, update_clouds
from .disco import compress_thermal
from .deq_chem import get_quench_levels
from .grad import moist_grad, did_grad_cp
from .numerics import locate, mat_sol

import os

from collections import namedtuple


convergence_criteriaT = namedtuple('Conv',['it_max','itmx','conv','convt','x_max_mult'])

def update_quench_levels(bundle, Atmosphere, kz, grav,verbose=False): 
    """
    Compute and update the quench levels for disequilibrium chemistry

    Parameters
    ----------
    Atmosphere Tuple: 
        Contains temperature, pressure, dtdp, mmw, and scale height
    kz : array 
        array of Kz cm^2/s
    grav : float
        gravity cgs
    
    """
    #PH3 requires h2o and h2 abundances so if all those three 
    #are present than go forth and compute the PH3 quenching 
    if np.all(np.isin(['H2','H2O','PH3'], bundle.inputs['atmosphere']['profile'].keys())):
        H2O = bundle.inputs['atmosphere']['profile']['H2O'].values
        H2 = bundle.inputs['atmosphere']['profile']['H2'].values
        PH3=True 
    else: 
        H2O=None;H2=None;PH3=False

    #compute quench levels for everything
    quench_levels, timescale_mixing = get_quench_levels(Atmosphere, kz, grav, PH3=PH3, H2O=H2O, H2=H2)
    if verbose: print('Computed quenched levels at',quench_levels)
    return quench_levels

def update_kzz(grav, tidal, AdiabatBundle, nstr, Atmosphere, 
               #these are only needed if you dont have fluxes and need to compute them
               OpacityWEd=None, OpacityNoEd=None,ScatteringPhase=None,Disco=None,Opagrid=None, F0PI=None,OpacityWEd_clear=None,OpacityNoEd_clear=None,
               #if we have fluxes we do not need to recompute the kzz
               flux_net_ir_layer=None,flux_plus_ir_attop=None,
               #kwargs for get_kzz function
               moist=False, 
               do_holes=False, fhole=None,verbose=True):
    
    """
    Update the kzz profile using the mixing length theory.
    Parameters
    ----------
    grav : float
        gravity in cgs
    tidal : ndarray 
        effectively sigmaTeff^4, gets added to the convergence critiera (e.g. F_IR*rfacI + F_SOL*rfacV + tidal)
    AdiabatBundle : tuple
        contains 't_table', 'p_table', 'grad','cp'
    nstr : array
        array of convective zones
    Atmosphere : tuple
        Contains temperature, pressure, dtdp, mmw, and scale height
    OpacityWEd : namedtuple
        All opacity (e.g. dtau, tau, w0, g0) info with delta eddington corrected values 
    OpacityNoEd : namedtuple
        All opacity (e.g. dtau, tau, w0, g0) info without delta eddington corrected values 
    ScatteringPhase : namedtuple
        All scattering phase function inputs like ftau_cld and ftau_ray and fraction of forward to back scattering
    Disco : namedtuple
        All geometry inputs such as gauss/chebychev angles, incoming outgoing angles, etc 
    Opagrid : namedtuple
        Any opacity grid info such as wavelength grids, temperature pressure grids, tmax and tmin
    F0PI : ndarray
        Stellar spectrum if it exists otherwise this is just 1s array
    OpacityWEd_clear : namedtuple
        The clear opacities / scattering properties (opposed to the cloudy ones which are stored in the main opacity tuple)
    OpacityNoEd_clear: namedtuple
        The clear opacities / scattering properties w/o delta eddington correction (opposed to the cloudy ones which are stored in the main opacity tuple)
    do_holes : bool
        Default=False; if True, computes the fluxes with holes
    fhole : float
        Default=None ; fraction of the disk assumed to be clear 
    moist : bool 
        Defalt= False; computes moist adiabat

    """
    if verbose: print("I am updating kzz. This could be either because clouds were requested (which always use self consistent kzz) or because self consistent kzz profiles were requested for chemistry.")
    #Do I have fluxes or no? 
    if (np.any(flux_plus_ir_attop==None) and np.any(flux_net_ir_layer==None)): 
        if verbose: print('I dont have fluxes, let me compute them')
        if do_holes == True:
            flux_net_v_layer_full, flux_net_v_full, flux_plus_v_full, flux_minus_v_full , flux_net_ir_layer_full, flux_net_ir_full, flux_plus_ir_full, flux_minus_ir_full = get_fluxes(Atmosphere, OpacityWEd, OpacityNoEd,ScatteringPhase,
                        Disco,Opagrid, F0PI, reflected=False, thermal=True, 
                        do_holes=True, fhole=fhole, hole_OpacityWEd=OpacityWEd_clear,hole_OpacityNoEd=OpacityNoEd_clear)
        else:                
            flux_net_v_layer_full, flux_net_v_full, flux_plus_v_full, flux_minus_v_full , flux_net_ir_layer_full, flux_net_ir_full, flux_plus_ir_full, flux_minus_ir_full = get_fluxes(Atmosphere, OpacityWEd, OpacityNoEd,ScatteringPhase,
                        Disco,Opagrid, F0PI, reflected=False, thermal=True,
                        do_holes=False)

        flux_net_ir_layer = flux_net_ir_layer_full[:]
        flux_plus_ir_attop = flux_plus_ir_full[0,:]   
        
    
    # use mixing length theory to calculate Kzz profile nebhere
    
    kz = get_kzz(grav,tidal,flux_net_ir_layer, flux_plus_ir_attop,AdiabatBundle,nstr, Atmosphere, moist = moist)  
    
    return kz 

def run_diseq_climate_workflow(bundle, nofczns, nstr, temp, pressure,
            AdiabatBundle,opacityclass,
            grav,
            rfaci,rfacv,tidal,
            Opagrid,
            CloudParameters,
            save_profile,all_profiles,all_opd,
            verbose=True, moist = None,
            save_kzz=False, self_consistent_kzz=True):
    """
    Run the disequilibrium climate workflow. This function is called by the main function
    and runs the disequilibrium climate workflow. It updates the profile, kzz, and chemistry
    and returns the final profile, kzz, and chemistry.

    Parameters
    ----------
    nofczns : int
        number of convective zones
    nstr : array
        array of the layer of convective zone locations
    temp : array
        temperature profile
    pressure : array
        pressure profile
    AdiabatBundle : tuple
        tuple containing the adiabat table, pressure table, gradient, and cp
    opacityclass : object
    grav : float 
        gravity in cgs
    rfaci : float 
        IR flux addition fraction 
    rfacv : float
        Visible flux addition fraction
    tidal : ndarray 
        effectively sigmaTeff^4, gets added to the convergence critiera (e.g. F_IR*rfacI + F_SOL*rfacV + tidal)
    Opagrid : tuple
        tuple containing the opacities and other information
    CloudParameters : tuple
        tuple containing the cloud parameters, including the cloudy flag, fsed, mh, b, param, directory, and condensates
    save_profile : bool
    all_profiles : array
        array of all thermal structures
    all_opd : array
        array of all opacities
    moist : bool
        if True, use moist adiabat
    save_kzz : bool
        if True, save the kzz profile for every iteration
    self_consistent_kzz : bool
        if True, use the self-consistent kzz profile (not constant kzz)

    """

    """ Can deprecate all of this since we moved to profile 
    first_call_ever=False
    cloudy = CloudParameters.cloudy
    
    if cloudy: 
        virga_kwargs = {key:getattr(CloudParameters,key) for key in ['fsed','mh','b','param','directory','condensates']}
        hole_kwargs = {key:getattr(CloudParameters,key) for key in ['do_holes','fthin_cld','fhole']}
        do_holes = hole_kwargs['do_holes'];fhole=hole_kwargs['fhole']
        cld_species = CloudParameters.condensates
    else: 
        cld_species=[] ; do_holes = False; fhole=None

    F0PI = opacityclass.relative_flux

    #save profiles if requested 
    if save_kzz: all_kzz= []

    nlevel = len(temp)
    # first change the nstr vector because need to check if they grow or not
    # delete upper convective zone if one develops 

    #NEBQ: do we need this now that we arent running chemeq=first anymore? 
    del_zone =0 # move 4 levels deeper
    if (nstr[1] > 0) & (nstr[4] > 0) & (nstr[3] > 0) :
        nstr[1] = nstr[4]+del_zone
        nstr[2] = nlevel-2#89
        nstr[3],nstr[4],nstr[5] = 0,0,0
        
        if verbose: print("2 conv Zones, so making small adjustments")
    elif (nstr[1] > 0) & (nstr[3] == 0):
        if nstr[4] == 0:
            nstr[1]+= del_zone #5#15
        else:
            nstr[1] += del_zone #5#15  
            nstr[3], nstr[4] ,nstr[5] = 0,0,0#6#16
        if verbose: print("1 conv Zone, so making small adjustment")
    if nstr[1] >= nlevel -2 : # making sure we haven't pushed zones too deep
        nstr[1] = nlevel -4
    if nstr[4] >= nlevel -2:
        nstr[4] = nlevel -3
    
    if verbose: print("New NSTR status is ", nstr)

    ### 1) UPDATE PT, CHEM, OPACITIES 
    bundle.add_pt( temp, pressure)
    #no quenching in this first run because we need an atmosphere first to get the 
    #fluxes that gives us a kzz that gives us a quench level
    bundle.premix_atmosphere(opa = opacityclass,quench_levels=None, cld_species=cld_species)
    OpacityWEd, OpacityNoEd, ScatteringPhase, Disco, Atmosphere , holes=  calculate_atm(bundle, opacityclass)
    
    # Clearsky profile, define others with _clear to avoid overwriting cloudy profile
    OpacityWEd_clear=holes[0]; OpacityNoEd_clear=holes[1]

    ### 2) UPDATE KZZ
    if self_consistent_kzz:
        kz = update_kzz(grav, tidal, AdiabatBundle, nstr, Atmosphere, 
               #these are only needed if you dont have fluxes and need to compute them
               OpacityWEd=OpacityWEd, OpacityNoEd=OpacityNoEd,ScatteringPhase=ScatteringPhase,Disco=Disco,Opagrid=Opagrid, F0PI=F0PI,
               OpacityWEd_clear=OpacityWEd_clear,OpacityNoEd_clear=OpacityNoEd_clear,
               #kwargs for get_kzz function
               moist=moist, do_holes=do_holes, fhole=fhole)
        if save_kzz: all_kzz = np.append(all_kzz,kz)
    #Otherwise get the fixed profile in bundle
    else : 
        kz = bundle.inputs['atmosphere']['profile']['kz'].values

    ### 3) GET QUENCH LEVELS FOR DISEQ 
    if bundle.inputs['approx']['chem_method']!='photochem':
        quench_levels=update_quench_levels(bundle, Atmosphere, kz, grav)
    else: 
        quench_levels = None
        raise Exception('photochem not yet working')

    ### 4) UDPATE CHEM w/ new Quench Levels or Photochem
    bundle.premix_atmosphere(opa=opacityclass,quench_levels=quench_levels,
            cld_species=cld_species)
    """
    ### 5) RUN PROFILE to converge new profile 
    # define the initial convergence criteria for profile 
    convergence_criteria = convergence_criteriaT(it_max=10, itmx=7, conv=5.0, convt=4.0, x_max_mult=7.0) 

    final=False
    profile_flag, pressure, temperature, dtdp,CloudParameters,cld_out,flux_net_ir_layer,flux_net_v_layer,flux_plus_ir_attop,all_profiles,all_opd,all_kzz =profile(
            bundle, nofczns, nstr, temp, pressure, 
            AdiabatBundle,opacityclass,
            grav,
            rfaci,rfacv,tidal,
            Opagrid,
            CloudParameters,
            save_profile,all_profiles,all_opd,
            convergence_criteria, final,
            flux_net_ir_layer=None, flux_plus_ir_attop=None,first_call_ever=False,
            verbose=verbose, moist = moist,
            save_kzz=save_kzz,self_consistent_kzz=self_consistent_kzz,diseq=True)
    
    final_conv_flag, pressure, temp, dtdp, nstr_new,flux_net_ir_final,flux_net_v_final, flux_plus_final, chem_out, cld_out,all_profiles, all_opd ,all_kzz=find_strat(bundle,
            nofczns,nstr,
            temperature,pressure,dtdp, #Atmosphere
            AdiabatBundle,
            opacityclass, grav, 
            rfaci, rfacv, tidal ,
            Opagrid,
            CloudParameters,
            save_profile, all_profiles, all_opd,
            flux_net_ir_layer, flux_plus_ir_attop,
            verbose=verbose,  moist = moist,
            save_kzz=save_kzz,self_consistent_kzz=self_consistent_kzz,diseq=True, all_kzz=all_kzz)
    
    #if CloudParameters.cloudy == 1:
    #    opd_now,w0_now,g0_now = cld_out['opd_per_layer'],cld_out['single_scattering'],cld_out['asymmetry']
    #else:
    #    opd_now,w0_now,g0_now = 0,0,0
    
    return final_conv_flag, pressure, temp, dtdp, nstr_new, flux_net_ir_final,flux_net_v_final, flux_plus_final, chem_out, cld_out, all_profiles, all_opd,all_kzz     


def run_chemeq_climate_workflow(bundle, nofczns, nstr, temp, pressure, 
            AdiabatBundle,opacityclass,
            grav,
            rfaci,rfacv,tidal,
            Opagrid,
            CloudParameters,
            save_profile,all_profiles,all_opd,
            verbose=True, moist = None, 
            save_kzz = True, self_consistent_kzz=True): 
    """
    Run the equilibrium climate workflow. It updates the profile, kzz, and chemistry
    and returns the final profile, kzz, and chemistry.

    Parameters
    ----------
    nofczns : int
        number of convective zones
    nstr : array
        array of the layer of convective zone locations
    temp : array
        temperature profile
    pressure : array
        pressure profile
    AdiabatBundle : tuple
        tuple containing the adiabat table, pressure table, gradient, and cp
    opacityclass : object
    grav : float 
        gravity in cgs
    rfaci : float 
        IR flux addition fraction 
    rfacv : float
        Visible flux addition fraction
    tidal : ndarray 
        effectively sigmaTeff^4, gets added to the convergence critiera (e.g. F_IR*rfacI + F_SOL*rfacV + tidal)
    Opagrid : tuple
        tuple containing the opacities and other information
    CloudParameters : tuple
        tuple containing the cloud parameters, including the cloudy flag, fsed, mh, b, param, directory, and condensates
    save_profile : bool
    all_profiles : array
        array of all thermal structures
    all_opd : array
        array of all opacities
    moist : bool
        if True, use moist adiabat
    save_kzz : bool
        if True, save the kzz profile for every iteration
    self_consistent_kzz : bool
        if True, use the self-consistent kzz profile (not constant kzz)
    """
        
    #STEP 1) first profile call with lose convergence criteria 
    final = False
    
    convergence_criteria = convergence_criteriaT(it_max=10, itmx=7, conv=10.0, convt=5.0, x_max_mult=7.0)       
    
    profile_flag,pressure, temperature, dtdp,  CloudParameters,cld_out,flux_net_ir_layer, flux_net_v_layer, flux_plus_ir_attop,all_profiles, all_opd,all_kzz = profile(bundle,
            nofczns,nstr, #tracks convective zones 
            temp,pressure, #Atmosphere
            AdiabatBundle, #t_table, p_table, grad, cp, 
            opacityclass, grav, 
            rfaci, rfacv,  tidal, #energy balance 
            Opagrid, #delta_wno, tmin, tmax, 
            CloudParameters,#cloudy,cld_species,mh,fsed,beta,param_flag,mieff_dir ,opd_cld_climate,g0_cld_climate,w0_cld_climate, #scattering/cloud properties 
            save_profile,all_profiles, all_opd,
            convergence_criteria, final , 
            first_call_ever=True, verbose=verbose, moist = moist,
            save_kzz=save_kzz,all_kzz=[],self_consistent_kzz=self_consistent_kzz)

    #STEP 2) second profile call with stricter convergence criteria 
    it_max= 7
    itmx= 5
    conv = 5.0
    convt=4.0
    x_max_mult=7.0
    convergence_criteria = convergence_criteriaT(it_max, itmx, conv, convt, x_max_mult)

    final = False
    
    profile_flag,pressure, temperature, dtdp,CloudParameters,cld_out,flux_net_ir_layer, flux_net_v_layer, flux_plus_ir_attop,  all_profiles, all_opd,all_kzz = profile(bundle,
            nofczns,nstr, #tracks convective zones 
            temperature, pressure, 
            AdiabatBundle, #t_table, p_table, grad, cp, 
            opacityclass, grav, 
            rfaci, rfacv,  tidal, #energy balance 
            Opagrid, #delta_wno, tmin, tmax, 
            CloudParameters,#cloudy,cld_species,mh,fsed,beta,param_flag,mieff_dir ,opd_cld_climate,g0_cld_climate,w0_cld_climate, #scattering/cloud properties 
            save_profile,all_profiles, all_opd,               
            convergence_criteria,final ,      
            flux_net_ir_layer=flux_net_ir_layer, flux_plus_ir_attop=flux_plus_ir_attop, 
            verbose=verbose,moist = moist,
            save_kzz=True,all_kzz=all_kzz,self_consistent_kzz=self_consistent_kzz)   
    
    #STEP 3) find strat that will now run profile several times, each time updating the opacities and chemistry 
    #and also refine the convective zone guess while it does this. 
    final_conv_flag, pressure, temp, dtdp, nstr_new, flux_net_ir_final,flux_net_v_final, flux_plus_final, chem_out, cld_out, all_profiles, all_opd,all_kzz =find_strat(bundle,
            nofczns,nstr,
            temperature,pressure,dtdp, #Atmosphere
            AdiabatBundle,
            opacityclass, grav, 
            rfaci, rfacv, tidal ,
            Opagrid,
            CloudParameters,
            save_profile, all_profiles, all_opd,
            flux_net_ir_layer, flux_plus_ir_attop,
            verbose=verbose, moist = moist,self_consistent_kzz=self_consistent_kzz)
    
    return final_conv_flag, pressure, temp, dtdp, nstr_new, flux_net_ir_final,flux_net_v_final, flux_plus_final, chem_out, cld_out, all_profiles, all_opd,all_kzz  

    
@jit(nopython=True, cache=True)
def convec(temp,pressure,AdiabatBundle, Atmosphere, moist = False):
    """
    Calculates Grad arrays from profiles
    
    Parameters 
    ----------
    temp : array 
        level temperature array
    pressure : array
        level pressure array
    AdiabatBundle : tuple
        tuple containing the adiabat table, pressure table, gradient, and cp
    Atmosphere : tuple
        Contains temperature, pressure, dtdp, mmw, and scale height
    moist : bool
        if moist adiabat is to be used
    Return
    ------
    grad_x, cp_x
    """
    # layer profile arrays
    tbar= np.zeros(shape=(len(temp)-1))
    pbar= np.zeros(shape=(len(temp)-1))
    
    grad_x, cp_x = np.zeros(shape=(len(temp)-1)), np.zeros(shape=(len(temp)-1))

    if moist == True:
        for j in range(len(temp)-1):
            tbar[j] = 0.5*(temp[j]+temp[j+1])
            pbar[j] = sqrt(pressure[j]*pressure[j+1])
            grad_x[j], cp_x[j] =  moist_grad( tbar[j], pbar[j], AdiabatBundle , Atmosphere, j)

    else:
        for j in range(len(temp)-1):
            tbar[j] = 0.5*(temp[j]+temp[j+1])
            pbar[j] = sqrt(pressure[j]*pressure[j+1])
            grad_x[j], cp_x[j] =  did_grad_cp( tbar[j], pbar[j], AdiabatBundle)

    return grad_x, cp_x

@jit(nopython=True, cache=True)
def t_start(nofczns,nstr,convergence_criteria,# 
            rfaci, rfacv, tidal,
            Atmosphere, OpacityWEd, OpacityNoEd,ScatteringPhase, Disco,Opagrid, AdiabatBundle,
            F0PI,
            save_profile, all_profiles, 
            fhole=None, hole_OpacityWEd=None,hole_OpacityNoEd=None, 
            verbose=1, do_holes=None, 
            moist = False, egp_stepmax = False):
    """
    Module to iterate on the level TP profile to make the Net Flux as close to 0.
    Opacities/chemistry are not updated while iterating in this module.
    Parameters
    ----------
    nofczns : int
        # of convective zones 
    nstr : array
        dimension of 20
        NSTR vector describes state of the atmosphere:
        0   is top layer
        1   is top layer of top convective region
        2   is bottom layer of top convective region
        3   is top layer of lower radiative region
        4   is top layer of lower convective region
        5   is bottom layer of lower convective region
    convergence_criteria : namedtuple
        Defines convergence criteria for max number of loops and other numerical recipes values 
        TODO: rename these quantities so that its more readable 
    rfaci : float 
        IR flux addition fraction 
    rfacv : float
        Visible flux addition fraction
    tidal : ndarray 
        effectively sigmaTeff^4, gets added to the convergence critiera (e.g. F_IR*rfacI + F_SOL*rfacV + tidal)
    Atmosphere : namedtuple
        contains info such as PT profile, chemistry, condensable information nlevel : int
        # of levels
    OpacityWEd : namedtuple
        All opacity (e.g. dtau, tau, w0, g0) info with delta eddington corrected values 
    OpacityNoEd : namedtuple
        All opacity (e.g. dtau, tau, w0, g0) info without delta eddington corrected values 
    ScatteringPhase : namedtuple
        All scattering phase function inputs like ftau_cld and ftau_ray and fraction of forward to back scattering
    Disco : namedtuple
        All geometry inputs such as gauss/chebychev angles, incoming outgoing angles, etc 
    Opagrid : namedtuple
        Any opacity grid info such as wavelength grids, temperature pressure grids, tmax and tmin
    AdiabatBundle : namedtuple
        Any info for the adiabat calculations such as the precomputed tables from Didie 
    F0PI : ndarray
        Stellar spectrum if it exists otherwise this is just 1s array
    save_profile : bool 
        bool to specify if all intermediate profiles will be saved 
    all_profiles : ndarray 
        All saved profiles if it is requested 
    fhole
        Default=None ; fraction of the disk assumed to be clear 
    hole_OpacityWEd : namedtuple
        The clear opacities / scattering properties (opposed to the cloudy ones which are stored in the main opacity tuple)
    hole_OpacityNoEd : namedtuple
        The clear opacities / scattering properties w/o delta eddington correction (opposed to the cloudy ones which are stored in the main opacity tuple)
    verbose
        Default=1; prints things out (also slows down code)
    do_holes
        Default=None: will compute fractional cloud coverage 
    moist 
        Defalt= False; computes moist adiabat 
    egp_stepmax 
        Default = False; uses the "bug" turned feature where stepmax increases without bounds. Often leads to faster convergence but if you want small T steps this is not 
        recommended 
    verbose : int
        If verbose=0, nothing will print out
        If verbose=1, everything will print out during the run, 
    
    Returns
    -------
    array 
        Temperature array and lapse ratio array if converged
        else Temperature array twice
    """
    #     Routine to iteratively solve for T(P) profile.  Uses a Newton-
    #     Raphson iteration to zero out the net flux in the radiative
    #     zone above NSTRAT.  Repeats until average temperature change
    #     is less than CONV or until ITMAX is reached.

    # -- SM -- needs a lot of documentation

    #unpack 
    temp = Atmosphere.t_level; pressure = Atmosphere.p_level
    nlevel = len(temp)
    tmin =Opagrid.tmin 
    tmax = Opagrid.tmax 
    it_max,conv,x_max_mult = convergence_criteria.it_max,convergence_criteria.conv,convergence_criteria.x_max_mult


    cldsave_count = 0 # used to track how many cloud profiles to save outside of loop for animation *JM

    eps=1e-4

    n_top_r=nstr[0]-1

    # here are other  convergence and tolerance criterias

    step_max = 0.01e0 # scaled maximum step size in line searches
    alf = 1.e-4    # ? #1e-3 in EGP code I have (JM), was 1e-4 in original PICASO code
    alam2 = 0.0   # ? 
    tolmin=1.e-5   # ?
    tolf = 5e-3    # tolerance in fractional Flux we are aiming for
    tolx = 5e-3    # tolerance in fractional T change we are aiming for

    #both reflected and thermal
    #neb this double true in the first call to reflected light needs to be changed
    if rfacv==0:compute_reflected=False
    else:compute_reflected=True
    compute_thermal=True

    

    #if do_holes == True:
    #    flux_net_v_layer_full, flux_net_v_full, flux_plus_v_full, flux_minus_v_full , flux_net_ir_layer_full, flux_net_ir_full, flux_plus_ir_full, flux_minus_ir_full = get_fluxes(pressure, temp, dwni, bb , y2, tp, tmin, tmax, DTAU, TAU, W0, 
    #            COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman , surf_reflect, 
    #            ubar0,ubar1,cos_theta, F0PI, single_phase,multi_phase,frac_a,frac_b,frac_c,constant_back,constant_forward,
    #            wno,nwno,ng,nt, gweight, tweight, nlevel, ngauss, gauss_wts, compute_reflected, True, 
    #            fhole, DTAU_clear , TAU_clear , W0_clear , COSB_clear , DTAU_OG_clear , TAU_OG_clear, W0_OG_clear, 
    #            COSB_OG_clear , W0_no_raman_clear, do_holes=True) #True for reflected, True for thermal
    #else:
    #    flux_net_v_layer_full, flux_net_v_full, flux_plus_v_full, flux_minus_v_full , flux_net_ir_layer_full, flux_net_ir_full, flux_plus_ir_full, flux_minus_ir_full = get_fluxes(pressure, temp, dwni, bb , y2, tp, tmin, tmax, DTAU, TAU, W0, 
    #            COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman , surf_reflect, 
    #            ubar0,ubar1,cos_theta, F0PI, single_phase,multi_phase,frac_a,frac_b,frac_c,constant_back,constant_forward, 
    #            wno,nwno,ng,nt, gweight,tweight, nlevel, ngauss, gauss_wts,compute_reflected, True)#True for reflected, True for thermal
    if do_holes == True:
        flux_net_v_layer_full, flux_net_v_full, flux_plus_v_full, flux_minus_v_full , flux_net_ir_layer_full, flux_net_ir_full, flux_plus_ir_full, flux_minus_ir_full = get_fluxes(Atmosphere, OpacityWEd, OpacityNoEd,ScatteringPhase,
                    Disco,Opagrid, F0PI, compute_reflected, compute_thermal, 
                    do_holes=do_holes,fhole=fhole, hole_OpacityWEd=hole_OpacityWEd,hole_OpacityNoEd=hole_OpacityNoEd)

    else:                
        flux_net_v_layer_full, flux_net_v_full, flux_plus_v_full, flux_minus_v_full , flux_net_ir_layer_full, flux_net_ir_full, flux_plus_ir_full, flux_minus_ir_full = get_fluxes(Atmosphere, OpacityWEd, OpacityNoEd,ScatteringPhase,
                    Disco,Opagrid, F0PI, compute_reflected, compute_thermal)
 
    # extract visible fluxes
    flux_net_v_layer = flux_net_v_layer_full[0,0,:]  #fmnetv
    flux_net_v = flux_net_v_full[0,0,:]#fnetv
    flux_plus_v =  flux_plus_v_full[0,0,:,:]
    flux_minus_v = flux_minus_v_full[0,0,:,:]

    # extract ir fluxes
    flux_net_ir_layer = flux_net_ir_layer_full[:] #fmneti
    flux_net_ir = flux_net_ir_full[:]     #fneti
    flux_plus_ir = flux_plus_ir_full[:,:]  
    flux_minus_ir = flux_minus_ir_full[:,:]
    
   
    
    # arrays for total net fluxes = optical+ir + tidal
    flux_net=np.zeros(shape=(nlevel))
    flux_net_midpt=np.zeros(shape=(nlevel))
    dflux=np.zeros(shape=(nlevel))
    f_vec=np.zeros(shape=(nlevel)) #fvec
    p=np.zeros(shape=(nlevel)) #p
    g=np.zeros(shape=(nlevel))
    
    # jacobian of zeros
    A= np.zeros(shape=(nlevel,nlevel)) 
    

    
    for its in range(it_max):
        
        # the total net flux = optical + ir + tidal component
        
        flux_net = rfaci* flux_net_ir + rfacv* flux_net_v +tidal #fnet
        flux_net_midpt = rfaci* flux_net_ir_layer + rfacv* flux_net_v_layer +tidal #fmnet
        
        #print('flux_net_midpt',flux_net_midpt)

        #raise Exception ('stop in tstart')

        beta= temp.copy() # beta vector
        
       
        # store old fluxes and temp before iteration
        # do not store the ir+vis flux because we are going to perturb only thermal structure

        
        temp_old= temp.copy() 
        flux_net_old = flux_net_ir.copy() #fnetip
        flux_net_midpt_old= flux_net_ir_layer.copy()  #fmip

        nao = n_top_r
        n_total = 0#0 #ntotl

        sum = 0.0
        sum_1 = 0.0 # sum1
        test = 0.0
        
        flag_nao = 0
        if nao < 0 :
            nao_temporary = nao
            nao = 0 # to avoid negative of negative index leading to wrong indexing, fort vs. py
            flag_nao = 1 # so that it can be reversed back to previous value after loop

        for nca in range(0, 3*nofczns, 3): 
            
            # first fill in the dflux  vector
            # contains flux ant top and midpt fluxes for nstrat -1 layers in stratosphere

            n_top_a= nstr[nca] #ntopa -- top of atmosphere or top of other rad zones

            n_strt_a = nstr[nca+1] #nstrta -- top of top conv zone
            
            
            # n_top_a to n_strt_a is a radiative zone
            # n_bot_a is the top of the next rad zone than this
            
            n_bot_a= nstr[nca+2] +1 #nbota -- top of lower rad zone

            if n_top_a == n_top_r+1 : # if the rad zone is also top of atmos
                dflux[0] = flux_net[n_top_r+1]
                f_vec[0] = dflux[0]

                sum += f_vec[0]**2
                sum_1 += temp[0]**2

                if abs(f_vec[0]) > test :
                    test = abs(dflux[0])
                n_total += 1
            
            # +1 to include last element
            for j in range(n_top_a+1, n_strt_a+1):
                
                dflux[j-nao] = flux_net_midpt[j-1] 

                f_vec[j-nao] = dflux[j-nao] 

                sum += f_vec[j-nao]**2 

                sum_1 += temp[j-nao]**2 

                if abs(f_vec[j-nao]) > test : 
                    test = abs(dflux[j-nao]) 
                n_total += 1
            
            if flag_nao == 1 :
                nao= nao_temporary
                flag_nao = 0

            
            
            nao += n_bot_a - n_strt_a

        
        f = 0.5*sum # used in linesearch, defined in NR function fmin

        # test if we are already at a root
        if (test/abs(tidal[0])) < 0.01*tolf :
            if verbose: print(" We are already at a root, tolf , test = ",0.01*tolf,", ",test/abs(tidal[0]))
            flag_converge = 2
            dtdp=np.zeros(shape=(nlevel-1))
            for j in range(nlevel -1):
                dtdp[j] = (log( temp[j]) - log( temp[j+1]))/(log(pressure[j]) - log(pressure[j+1]))
            
            return   temp,  dtdp, all_profiles , flux_net_ir,flux_net_v, flux_plus_ir[0,:] 
            #return   temp,  dtdp, flag_converge, flux_net_ir, flux_plus_ir[0,:], all_profiles, cldsave_count
            
        
        # NEB NOTE about step max 
        # In the original fortran code this was originally 
        if egp_stepmax == True:
            step_max_tolerance = 0.005
            step_max = step_max_tolerance*max(sqrt(sum_1),n_total*1.0) #where step_max_tolerance=0.03
        # however when this was fixed, the code was progressing very slowly 
        # therefore, we are keeping this in the code for now 
        # the result of this is that there are sometimes large temperature 
        # steps that might be problematic for edge cases that get too hot or too cold 
        else:
            # added this to help with smoother convergence for cloudy cases and also helps speed up convergence
            # by a bit when running the default test cases
            iteration_factor = max(0.01, (it_max - its) / it_max)
            step_max *= max(sqrt(sum_1),n_total*1.0)*iteration_factor #step_max_tolerance*
        #if verbose: print('maximum scaled step size',step_max, n_total, sum_1, its)
        no =n_top_r
        
        i_count= 1 #icount
        
        flag_no = 0
        if no < 0 :
            no_temporary = no
            no = 0 # to avoid negative of negative index leading to wrong indexing, fort vs. py
            flag_no = 1 # so that it can be reversed back to previous value after loop
       
        for nz in range(0, 3*nofczns, 3):

            n_top = nstr[nz] +1 #ntop
            n_strt = nstr[nz+1] #nstrt
            n_conv_top = n_strt + 1 #ncnvtop
            n_conv_bot= nstr[nz+2] +1 #ncnvbot

            if nz == 0 :
                n_top -= 1
            
            
            
        # begin jacobian calculation here
        # +1 to include last element
            for jm in range(n_top, n_strt+1):

                # chose perturbation for each level

                i_count += 1

                #eps is just a tolerance value currently fixed at 1e-4
                del_t = max(eps * temp_old[jm], 3.0) # perturbation

                beta[jm] += del_t # perturb

                
                # now reconstruct Temp profile

                for nb in range(0, 3*nofczns, 3):

                    n_top_b = nstr[nb] + 1 # ntopb
                    
                    if nb == 0:
                        n_top_b -= 1 #ntopb
                    
                    n_strt_b = nstr[nb+1] # nstrtb
                    
                    n_conv_top_b = n_strt_b + 1 #nctopb

                    n_bot_b = nstr[nb+2] +1 #nbotb

                    
                    # +1 to include last element   
                    for j1 in range(n_top_b,n_strt_b+1):
                        temp[j1] = beta[j1]
                    
                    # +1 to include last element
                    for j1 in range(n_conv_top_b, n_bot_b+1): 
                        
                        press = sqrt(pressure[j1-1]*pressure[j1])

                        #update temp before throwing to moist_grad function
                        Atmosphere=replace_temp(Atmosphere,temp)
                        if moist == True:
                            grad_x, cp_x = moist_grad( beta[j1-1], press, AdiabatBundle, Atmosphere, j1-1)
                        else: 
                            grad_x, cp_x = did_grad_cp( beta[j1-1], press, AdiabatBundle)
                        
                        temp[j1]= exp(log(temp[j1-1]) + grad_x*(log(pressure[j1]) - log(pressure[j1-1])))
                
                

                # temperature has been perturbed
                # now recalculate the IR fluxes, so call picaso with only thermal

                #flux_net_v_layer_full, flux_net_v_full, flux_plus_v_full, flux_minus_v_full , flux_net_ir_layer_full, flux_net_ir_full, flux_plus_ir_full, flux_minus_ir_full = get_fluxes(pressure, temp, dwni, bb , y2, tp, tmin, tmax, DTAU, TAU, W0, 
                        #COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman , surf_reflect, 
                        #ubar0,ubar1,cos_theta, F0PI, single_phase,multi_phase,frac_a,frac_b,frac_c,constant_back,constant_forward, 
                        #wno,nwno,ng,nt,gweight,tweight, nlevel, ngauss, gauss_wts, False, True) #false for reflected, True for thermal
                
                #if do_holes == True:
                #    flux_net_v_layer_full, flux_net_v_full, flux_plus_v_full, flux_minus_v_full , flux_net_ir_layer_full, flux_net_ir_full, flux_plus_ir_full, flux_minus_ir_full = get_fluxes(pressure, temp, dwni, bb , y2, tp, tmin, tmax, DTAU, TAU, W0, 
                #        COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman , surf_reflect, 
                #        ubar0,ubar1,cos_theta, F0PI, single_phase,multi_phase,frac_a,frac_b,frac_c,constant_back,constant_forward,
                #        wno,nwno,ng,nt,gweight,tweight, nlevel, ngauss, gauss_wts, False, True, fhole, DTAU_clear , TAU_clear , W0_clear , COSB_clear , 
                #        DTAU_OG_clear , TAU_OG_clear , W0_OG_clear, COSB_OG_clear , W0_no_raman_clear, do_holes=True) #false for reflected, True for thermal
                
                Atmosphere=replace_temp(Atmosphere,temp)
                if do_holes == True:
                    flux_net_v_layer_full, flux_net_v_full, flux_plus_v_full, flux_minus_v_full , flux_net_ir_layer_full, flux_net_ir_full, flux_plus_ir_full, flux_minus_ir_full = get_fluxes(Atmosphere, OpacityWEd, OpacityNoEd,ScatteringPhase,
                                Disco,Opagrid, F0PI, compute_reflected, compute_thermal, 
                                do_holes=do_holes,fhole=fhole, hole_OpacityWEd=hole_OpacityWEd,hole_OpacityNoEd=hole_OpacityNoEd)

                else:                
                    flux_net_v_layer_full, flux_net_v_full, flux_plus_v_full, flux_minus_v_full , flux_net_ir_layer_full, flux_net_ir_full, flux_plus_ir_full, flux_minus_ir_full = get_fluxes(Atmosphere, OpacityWEd, OpacityNoEd,ScatteringPhase,
                                Disco,Opagrid, F0PI, compute_reflected, compute_thermal)
             

                # extract ir fluxes

                flux_net_ir_layer = flux_net_ir_layer_full[:] #fmneti
                flux_net_ir = flux_net_ir_full[:]     #fneti
                flux_plus_ir = flux_plus_ir_full[:,:]  
                flux_minus_ir = flux_minus_ir_full[:,:]

     
                
                # now calculate jacobian terms in the same way as dflux
                nco = n_top_r
                
                # -ve nco and no will mess indexing
                # so we want to set them to 0 temporarily if that occurs
                flag_nco = 0 
                if nco < 0 :
                    nco_temporary = nco
                    nco = 0 # to avoid negative of negative index leading to wrong indexing, fort vs. py
                    flag_nco = 1 # so that it can be reversed back to previous value after loop

                for nc in range(0,3*nofczns, 3):
                    
                    n_top_c = nstr[nc] +1 # ntopc

                    if nc ==0:
                        n_top_c -= 1
                    n_strt_c = nstr[nc+1]
                    n_bot_c = nstr[nc+2] +1
                    
                    
                        
                    
                   
                    if n_top_c == n_top_r+1 :
                        
                        A[n_top_c-nco,jm-no] = (flux_net_ir[n_top_c]-flux_net_old[n_top_c])/del_t
                        
                    else:
                        
                        A[n_top_c-nco,jm-no] = (flux_net_ir_layer[n_top_c-1]-flux_net_midpt_old[n_top_c-1])/del_t
                        
                    
                    
                    # omitted -1 to include last element 
                    
                    for im in range(n_top_c,n_strt_c):
                        #print(im+1-nco,jm-no, "3rd",jm,no)
                        A[im+1-nco,jm-no] = (flux_net_ir_layer[im]-flux_net_midpt_old[im])/del_t
                        
                    # changing them back to what they were
                    
                    if flag_nco == 1 :
                        nco= nco_temporary
                        flag_nco = 0
                    
                    
                       

                    nco+= n_bot_c-n_strt_c
                

                # undo beta vector perturbation
                beta[jm] = beta[jm] - del_t
            
            if flag_no == 1 :
                no= no_temporary
                flag_no = 0
            
            no += n_conv_bot-n_strt
        
        # a long print statement here in original. dont know if needed

        
        for i in range(n_total):
            sum=0.0
            for j in range(n_total):
                sum += A[j,i]*f_vec[j]
            
            g[i] = sum

            p[i] = -f_vec[i]
        
        f_old = f #fold
        
        #print('f_vec[0],f_vec[-1],min,max:f_vec', 
        #    f_vec[0],f_vec[-1],min(f_vec),max(f_vec))
        #print(f_vec)
        #raise Exception ("stop")

        A, p = mat_sol(A, nlevel, n_total, p)
        
        #print(p)
        
        

        check = False

        sum = 0.0
        # Now we are in the "line search" routine
        # we ignore the first two points since they are flaky
        # start from 2 (3rd array pos in fortran), so changing loop initial
        
        for i in range(2,n_total):
            sum += p[i]**2
        sum = sqrt(sum)
        
        
        # scale if attempted step is too big
        if sum > step_max:
            for i in range(n_total):
                p[i] *= step_max/sum

                dflux[i] = -p[i]
        
        slope = 0.0

        for i in range(n_total):
            slope += g[i]*p[i]
        # SM -- next two lines is problematic ? 
        #if slope >= 0.0 :
        #    raise ValueError("roundoff problem in linen search")
        
        ## checked till here -- SM
        test = 0.0
        
        for i in range(n_total):
            tmp = abs(p[i])/temp_old[i]
            if tmp > test :
                test= tmp 

        alamin = tolx/test
        alam = 1.0
        
        f2= f #################### to avoid call before assignment and run using numba
        #     Convergence test:  Find magnitude of correction by comparing
        #        temperature steps to a appropriate scale SCALT.  If average
        #        correction ERR is large, use only a fraction of the step.
        #        When ERR is less than CONV, routine has converged.
        
        flag_converge = 0
        # instead of the goto statement here
        #ct_num = 0
        while flag_converge == 0 :
            #ct_num+=1
            err = 0.0
            dmx = 0.0
            scalt = 1.0
            slow =8.0/scalt

            for j in range(n_total):
                dzx= abs(p[j])

                if dzx > dmx :
                    dmx = dzx
                    jmx = j+ n_top_r
                err += dzx
            
            err= err/(float(n_total)*scalt)

            if jmx > nstr[1] :
                jmx+= nstr[2]-nstr[1]
            
            ndo = n_top_r
            flag_ndo = 0 
            if ndo < 0 :
                ndo_temporary = ndo
                ndo = 0 # to avoid negative of negative index leading to wrong indexing, fort vs. py
                flag_ndo = 1 # so that it can be reversed back to previous value after loop
            
            for nd in range(0,3*nofczns, 3):
                n_top_d = nstr[nd] +1
                if nd == 0:
                    n_top_d -= 1

                n_strt_d = nstr[nd+1]

                n_bot_d= nstr[nd+2] +1
                
                
                   
                #+1 for fort to py
                
                for j in range(n_top_d,n_strt_d+1):
                    temp[j]= beta[j]+ alam*p[j-ndo]
                    #print(p[j-ndo],beta[j])
                #+1 for fort to py
                for j1 in range(n_strt_d+1, n_bot_d+1):

                    press = sqrt(pressure[j1-1]*pressure[j1])

                    Atmosphere=replace_temp(Atmosphere,temp)
                    if moist == True:
                        grad_x, cp_x = moist_grad( temp[j1-1], press, AdiabatBundle, Atmosphere, j1-1)
                    else:
                        grad_x, cp_x = did_grad_cp( temp[j1-1], press, AdiabatBundle)
                            
                    temp[j1]= exp(log(temp[j1-1]) + grad_x*(log(pressure[j1]) - log(pressure[j1-1])))
                
                if flag_ndo == 1 :
                        ndo= ndo_temporary
                        flag_ndo = 0

                ndo += n_bot_d - n_strt_d
            
            # artificial damper

            for j1 in range(n_top_r+1, nlevel):
                if temp[j1] < tmin:
                    temp[j1] = tmin+ 0.1
                elif temp[j1] > tmax:
                    temp[j1] = tmax- 0.1
            
            # re calculate thermal flux
            #flux_net_v_layer_full, flux_net_v_full, flux_plus_v_full, flux_minus_v_full , flux_net_ir_layer_full, flux_net_ir_full, flux_plus_ir_full, flux_minus_ir_full = get_fluxes(pressure, temp, dwni, bb , y2, tp, tmin, tmax, DTAU, TAU, W0, 
            #COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman , surf_reflect, 
            #ubar0,ubar1,cos_theta, F0PI, single_phase,multi_phase,frac_a,frac_b,frac_c,constant_back,constant_forward,
            #wno,nwno,ng,nt,gweight,tweight, nlevel, ngauss, gauss_wts, False, True) #false reflected, True thermal

            #if do_holes == True:
            #    flux_net_v_layer_full, flux_net_v_full, flux_plus_v_full, flux_minus_v_full , flux_net_ir_layer_full, flux_net_ir_full, flux_plus_ir_full, flux_minus_ir_full = get_fluxes(pressure, temp, dwni, bb , y2, tp, tmin, tmax, DTAU, TAU, W0, 
            #    COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman , surf_reflect, 
            #    ubar0,ubar1,cos_theta, F0PI, single_phase,multi_phase,frac_a,frac_b,frac_c,constant_back,constant_forward,
            #    wno,nwno,ng,nt, gweight, tweight, nlevel, ngauss, gauss_wts, False, True, fhole, DTAU_clear , TAU_clear , W0_clear , COSB_clear , 
            #    DTAU_OG_clear , TAU_OG_clear, W0_OG_clear, COSB_OG_clear , W0_no_raman_clear, do_holes=True) #false reflected, True thermal
            Atmosphere=replace_temp(Atmosphere,temp)
            if do_holes == True:
                flux_net_v_layer_full, flux_net_v_full, flux_plus_v_full, flux_minus_v_full , flux_net_ir_layer_full, flux_net_ir_full, flux_plus_ir_full, flux_minus_ir_full = get_fluxes(Atmosphere, OpacityWEd, OpacityNoEd,ScatteringPhase,
                            Disco,Opagrid, F0PI, reflected = False, thermal=compute_thermal, 
                            do_holes=do_holes,fhole=fhole, hole_OpacityWEd=hole_OpacityWEd,hole_OpacityNoEd=hole_OpacityNoEd)

            else:                
                flux_net_v_layer_full, flux_net_v_full, flux_plus_v_full, flux_minus_v_full , flux_net_ir_layer_full, flux_net_ir_full, flux_plus_ir_full, flux_minus_ir_full = get_fluxes(Atmosphere, OpacityWEd, OpacityNoEd,ScatteringPhase,
                            Disco,Opagrid, F0PI,reflected = False, thermal=compute_thermal)
         
           

            # extract ir fluxes

            flux_net_ir_layer = flux_net_ir_layer_full[:] #fmneti
            flux_net_ir = flux_net_ir_full[:]     #fneti
            flux_plus_ir = flux_plus_ir_full[:,:]  
            flux_minus_ir = flux_minus_ir_full[:,:]
    
            # re calculate net fluxes
            flux_net = rfaci* flux_net_ir + rfacv* flux_net_v +tidal #fnet
            flux_net_midpt = rfaci* flux_net_ir_layer + rfacv* flux_net_v_layer +tidal #fmnet
            
            sum = 0.0
            nao = n_top_r
            flag_nao = 0
            if nao < 0 :
                nao_temporary = nao
                nao = 0 # to avoid negative of negative index leading to wrong indexing, fort vs. py
                flag_nao = 1 # so that it can be reversed back to previous value after loop

            for nca in range(0,3*nofczns,3):
                n_top_a = nstr[nca] + 1
                if nca ==0 :
                    n_top_a -= 1
                
                n_strt_a=nstr[nca+1]
                n_bot_a = nstr[nca+2] + 1
                
                
                   

                if n_top_a == n_top_r +1 :
                    
                    f_vec[0]= flux_net[n_top_r +1]
                    sum += f_vec[0]**2
                else:
                    f_vec[n_top_a-nao] = flux_net_midpt[n_top_a -1]
                    sum += f_vec[n_top_a - nao]**2
                
                for j in range(n_top_a+1,n_strt_a+1):
                    #print(j-1)
                    f_vec[j-nao] = flux_net_midpt[j -1]
                    sum += f_vec[j-nao]**2
                
                if flag_nao == 1 :
                        nao= nao_temporary
                        flag_nao = 0
                
                nao+= n_bot_a - n_strt_a
                        
            f= 0.5*sum
            # if verbose: print('cond1:alam.lt.alamin',alam, alamin)
            # if verbose: print('cond2:f.le.CCC',f,f_old + alf*alam*slope)
            # if verbose: print('f,fold,alf,alam,slope',f,f_old,alf,alam,slope)
            #First check: Is T too small to continue? 
            if alam < alamin :
                check = True
                #if verbose: print(' CONVERGED ON SMALL T STEP alam, alamin', alam, alamin)
                flag_converge, check = check_convergence(f_vec, n_total, tolf, check, f, dflux, tolmin, temp, temp_old, g , tolx)
 
            #Second check: Has the net flux decreased enough that we are happy in the line search
            #If so you can proceed
            elif f <= f_old + alf*alam*slope :
                #if verbose: print ('Exit with decreased f')
                flag_converge, check = check_convergence(f_vec, n_total, tolf, check, f, dflux, tolmin, temp, temp_old, g , tolx)

            #Else: Let's back track     
            else:
                #if verbose: print(' Now backtracking, f, fold, alf, alam, slope', f, f_old, alf, alam, slope)
                if alam == 1.0:
                    
                    tmplam= -slope/ (2*(f-f_old-slope))
                else:
                    
                    rhs_1 = f- f_old - alam*slope
                    rhs_2 = f2 - f_old - alam2*slope
                    anr= ((rhs_1/alam**2)-(rhs_2/alam2**2))/(alam-alam2)
                    b= (-alam2*rhs_1/alam**2+alam*rhs_2/alam2**2)/(alam-alam2)
                    

                    if anr == 0 :
                        tmplam= -slope/(2.0*b)
                        
                        
                    else:
                        disc= b*b - 3.0*anr*slope
                        
                        if disc < 0.0 :
                            tmplam= 0.5*alam
                           
                        elif b <= 0.0:
                            tmplam=(-b + sqrt(disc))/(3.0*anr)
                            

                        else:
                            tmplam= -slope/(b+sqrt(disc))
                            
                    if tmplam > 0.5*alam:
                        
                        tmplam= 0.5*alam
            if ((flag_converge != 2) & (flag_converge != 1)):
                alam2=alam
                f2=f
                
                alam = max(tmplam,0.1*alam)

            if np.isnan(np.sum(temp)) == True:
                
                flag_converge = 1 # to avoid getting stuck here unnecesarily.
                temp = temp_old.copy() +0.5
                if verbose: print("Got stuck with temp NaN -- so escaping the while loop in tstart")
        

        if verbose: print("Iteration number ", its,", min , max temp ", min(temp),max(temp), ", flux balance ", flux_net[0]/abs(tidal[0])) #f/abs(tidal[0])**2) this other output here is slightly less straightforward with the square terms for exoplanets so making this just fnet/tidal for now

        if save_profile == 1:
            all_profiles = np.append(all_profiles,temp_old)
            cldsave_count += 1
        if flag_converge == 2 : # converged
            # calculate  lapse rate
            dtdp=np.zeros(shape=(nlevel-1))
            for j in range(nlevel -1):
                dtdp[j] = (log( temp[j]) - log( temp[j+1]))/(log(pressure[j]) - log(pressure[j+1]))
            
            if verbose: print("In t_start: Converged Solution in iterations ",its)
            
           
           
            return   temp,  dtdp, all_profiles , flux_net_ir,flux_net_v, flux_plus_ir[0,:] 
        
    if verbose: print("Iterations exceeded it_max ! sorry ")
    dtdp=np.zeros(shape=(nlevel-1))
    for j in range(nlevel -1):
        dtdp[j] = (log( temp[j]) - log( temp[j+1]))/(log(pressure[j]) - log(pressure[j+1]))

    return temp, dtdp , all_profiles , flux_net_ir_layer,flux_net_v, flux_plus_ir[0,:]

@jit(nopython=True, cache=True)
def check_convergence(f_vec, n_total, tolf, check, f, dflux, tolmin, temp, temp_old, g , tolx):
    """
    
    Module for checking convergence. Used in t_start module.

    Parameters
    ----------
    f_vec : array
        flux vector
    n_total : int
        number of total levels
    tolf : float
        tolerance factor
    check : bool
        check for convergence
    f : float
        flux
    dflux : array
        flux difference
    tolmin : float
        minimum tolerance
    temp : array
        temperature
    temp_old : array
        old temperature
    g : array
        gradient
    tolx : float
        tolerance for temperature


    """
    test = 0.0
    for i in range(n_total):
        if abs(f_vec[i]) > test:
            test=abs(f_vec[i])
    
    if test < tolf :
        check = False
        
        flag_converge = 2
        return flag_converge , check

    if check == True :
        test = 0.0
        den1 = max(f,0.5*(n_total))
        
        for i in range(n_total):
            tmp= abs(g[i])*dflux[i]/den1
            if tmp > test:
                test= tmp
        
        if test < tolmin :
            check= True
        else :
            check= False
        

        flag_converge = 2
        return flag_converge, check
    
    test = 0.0
    
    for i in range(n_total):
        tmp = (abs(temp[i]-temp_old[i]))/temp_old[i]
        if tmp > test:
            test=tmp
    if test < tolx :
        
        
        flag_converge = 2

        return flag_converge, check
    

    flag_converge = 1
    return flag_converge , check

@jit(nopython=True, cache=True)
def growup(nlv, nstr, ngrow) :
    """
    
    Module for growing conv zone. Used in find_strat module.

    Parameters
    ----------
    nlv : int
        number of levels
    nstr : array
        current levels of the conv zone
    ngrow : int
        number of levels to grow
    
    """
    n = 2+3*(nlv-1) -1 # -1 for the py referencing
    nstr[n]= nstr[n]-1*ngrow

    return nstr

@jit(nopython=True, cache=True)
def growdown(nlv,nstr, ngrow) :
    """
    
    Module for growing down conv zone. Used in find_strat module.

    Parameters
    ----------
    nlv : int
        number of levels
    nstr : array
        current levels of the conv zone
    ngrow : int
        number of levels to grow down
    
    """

    n = 3+3*(nlv-1) -1 # -1 for the py referencing
    nstr[n] = nstr[n] + 1*ngrow
    nstr[n+1] = nstr[n+1] + 1*ngrow

    return nstr


OpacityWEd_Tuple_defaultT = namedtuple("OpacityWEd_Tuple", ["DTAU", "TAU", "W0", "COSB",'ftau_cld','ftau_ray','GCOS2', 'W0_no_raman','f_deltaM'])
OpacityWEd_Tuple_default = OpacityWEd_Tuple_defaultT(np.zeros((8,8,8)), 
                    np.zeros((8,8,8)), np.zeros((8,8,8)), np.zeros((8,8,8)),np.zeros((8,8,8)), np.zeros((8,8,8)),np.zeros((8,8,8)),  
                    np.zeros((8,8,8)) , np.zeros((8,8,8)))

OpacityNoEd_Tuple_defaultT = namedtuple("OpacityNoEd_Tuple", ["DTAU", "TAU", "W0", "COSB"])
OpacityNoEd_Tuple_default = OpacityNoEd_Tuple_defaultT(np.zeros((8,8,8)), 
                    np.zeros((8,8,8)), np.zeros((8,8,8)), np.zeros((8,8,8)))
@jit(nopython=True, cache=False)
def get_fluxes(Atmosphere, OpacityWEd, OpacityNoEd,ScatteringPhase,
                Disco,Opagrid, F0PI, reflected, thermal, 
                do_holes=False, fhole=0.0, hole_OpacityWEd=OpacityWEd_Tuple_default,hole_OpacityNoEd=OpacityNoEd_Tuple_default):
    """
    Program to run RT for climate calculations. Runs the thermal and reflected module.
    And combines the results with wavenumber widths.

    Parameters 
    ----------
    Atmosphere : tuple
        Contains temperature, pressure, dtdp, mmw, and scale height
    OpacityWEd : namedtuple
        All opacity (e.g. dtau, tau, w0, g0) info with delta eddington corrected values 
    OpacityNoEd : namedtuple
        All opacity (e.g. dtau, tau, w0, g0) info without delta eddington corrected values 
    ScatteringPhase : namedtuple
        All scattering phase function inputs like ftau_cld and ftau_ray and fraction of forward to back scattering
    Disco : namedtuple
        All geometry inputs such as gauss/chebychev angles, incoming outgoing angles, etc 
    Opagrid : namedtuple
        Any opacity grid info such as wavelength grids, temperature pressure grids, tmax and tmin
    F0PI : ndarray
        Stellar spectrum if it exists otherwise this is just 1s array
    reflected : bool 
        Run reflected light
    thermal : bool 
        Run thermal emission
    do_holes: bool
        run patchy/fractional cloudy and clear model
    fhole: float
        Fraction of cloudy area
    hole_OpacityWEd : namedtuple
        The clear opacities / scattering properties (opposed to the cloudy ones which are stored in the main opacity tuple)
    hole_OpacityNoEd : namedtuple
        The clear opacities / scattering properties w/o delta eddington correction (opposed to the cloudy ones which are stored in the main opacity tuple)
    
        
    Return
    ------
    array
        Visible and IR -- net (layer and level), upward (level) and downward (level)  fluxes
    """  
    if do_holes: 
        if fhole==0: 
            print('Warning. A cloud hole is requested but the hole is set to zero which is doing a fully cloudy run but unecessarily running opacities twice.')
    #import dill as pickle
    #with open('tuples.pkl', 'wb') as file:
    #    pickle.dump([Atmosphere, OpacityWEd, OpacityNoEd,ScatteringPhase,
    #            Disco,Opagrid, F0PI, reflected, thermal, 
    #            do_holes, fhole, hole_OpacityWEd,hole_OpacityNoEd], file)
    #unpack atmosphere items 
    pressure, temperature ,nlevel= Atmosphere.p_level, Atmosphere.t_level,Atmosphere.nlevel
    #unpack opacity items w/ delta eddington correctioin
    DTAU, TAU, W0, COSB,ftau_cld, ftau_ray,GCOS2,W0_no_raman = OpacityWEd.DTAU, OpacityWEd.TAU, OpacityWEd.W0, OpacityWEd.COSB,OpacityWEd.ftau_cld, OpacityWEd.ftau_ray,OpacityWEd.GCOS2,OpacityWEd.W0_no_raman
    #unpack opacity items w/o delta eddington correctioin
    DTAU_OG,TAU_OG, W0_OG, COSB_OG = OpacityNoEd.DTAU,OpacityNoEd.TAU, OpacityNoEd.W0, OpacityNoEd.COSB
    #unpack scattering phase items 
    surf_reflect, single_phase,multi_phase,frac_a,frac_b,frac_c,constant_back,constant_forward = ScatteringPhase.surf_reflect, ScatteringPhase.single_phase,ScatteringPhase.multi_phase,ScatteringPhase.frac_a,ScatteringPhase.frac_b,ScatteringPhase.frac_c,ScatteringPhase.constant_back,ScatteringPhase.constant_forward
    #unpack disco items 
    ng,nt,gweight,tweight,ubar0,ubar1,cos_theta = Disco.ng,Disco.nt,Disco.gweight,Disco.tweight,Disco.ubar0,Disco.ubar1,Disco.cos_theta
    #unpack Opagrid info 
    nwno,dwni,wno,ngauss, gauss_wts = Opagrid.nwno,Opagrid.delta_wno,Opagrid.wno,Opagrid.ngauss, Opagrid.gauss_wts
   
    #if we are doing holes, then unpack those taus as well 
    if do_holes: 
        DTAU_clear, TAU_clear, W0_clear, COSB_clear,W0_no_raman_clear=hole_OpacityWEd.DTAU, hole_OpacityWEd.TAU, hole_OpacityWEd.W0, hole_OpacityWEd.COSB,hole_OpacityWEd.W0_no_raman
        ftau_cld_clear, ftau_ray_clear,GCOS2_clear = hole_OpacityWEd.ftau_cld, hole_OpacityWEd.ftau_ray,hole_OpacityWEd.GCOS2
        DTAU_OG_clear,TAU_OG_clear, W0_OG_clear, COSB_OG_clear=hole_OpacityNoEd.DTAU, hole_OpacityNoEd.TAU, hole_OpacityNoEd.W0, hole_OpacityNoEd.COSB
        

    # for visible
    flux_net_v = np.zeros(shape=(ng,nt,nlevel)) #net level visible fluxes
    flux_net_v_layer=np.zeros(shape=(ng,nt,nlevel)) #net layer visible fluxes

    flux_plus_v= np.zeros(shape=(ng,nt,nlevel,nwno)) # level plus visible fluxes
    flux_minus_v= np.zeros(shape=(ng,nt,nlevel,nwno)) # level minus visible fluxes
    
    #"""<<<<<<< NEWCLIMA
    # for thermal
    flux_plus_midpt = np.zeros(shape=(ng,nt,nlevel,nwno))
    flux_minus_midpt = np.zeros(shape=(ng,nt,nlevel,nwno))

    flux_plus = np.zeros(shape=(ng,nt,nlevel,nwno))
    flux_minus = np.zeros(shape=(ng,nt,nlevel,nwno))
    #"""

    """<<<<<<< OG
    # for thermal
    flux_plus_midpt = np.zeros(shape=(nlevel,nwno))
    flux_minus_midpt = np.zeros(shape=(nlevel,nwno))

    flux_plus = np.zeros(shape=(nlevel,nwno))
    flux_minus = np.zeros(shape=(nlevel,nwno))
    """

    # outputs needed for climate
    flux_net_ir = np.zeros(shape=(nlevel)) #net level visible fluxes
    flux_net_ir_layer=np.zeros(shape=(nlevel)) #net layer visible fluxes

    flux_plus_ir= np.zeros(shape=(nlevel,nwno)) # level plus visible fluxes
    flux_minus_ir= np.zeros(shape=(nlevel,nwno)) # level minus visible fluxes

    
    #ugauss_angles= np.array([0.0985350858,0.3045357266,0.5620251898,0.8019865821,0.9601901429])    
    #ugauss_weights = np.array([0.0157479145,0.0739088701,0.1463869871,0.1671746381,0.0967815902])
    #ugauss_angles = np.array([0.66666])
    #ugauss_weights = np.array([0.5])

    if reflected:
        #use toon method (and tridiagonal matrix solver) to get net cumulative fluxes 
        b_top = 0.0
        for ig in range(ngauss): # correlated - loop (which is different from gauss-tchevychev angle)
            #"""
            #<<<<<<< NEWCLIMA
            #here only the fluxes are returned since we dont care about the outgoing intensity at the 
            #top, which is only used for albedo/ref light spectra
            ng_clima,nt_clima=1,1
            ubar0_clima = ubar0*0+0.5
            ubar1_clima = ubar1*0+0.5
            _, out_ref_fluxes = get_reflected_1d(nlevel, wno,nwno,ng_clima,nt_clima,
                                    DTAU[:,:,ig], TAU[:,:,ig], W0[:,:,ig], COSB[:,:,ig],
                                    GCOS2[:,:,ig],ftau_cld[:,:,ig],ftau_ray[:,:,ig],
                                    DTAU_OG[:,:,ig], TAU_OG[:,:,ig], W0_OG[:,:,ig], COSB_OG[:,:,ig],
                                    surf_reflect, ubar0_clima,ubar1_clima,cos_theta, F0PI,
                                    single_phase,multi_phase,
                                    frac_a,frac_b,frac_c,constant_back,constant_forward, 
                                    get_toa_intensity=0, get_lvl_flux=1)

            flux_minus_all_v, flux_plus_all_v, flux_minus_midpt_all_v, flux_plus_midpt_all_v = out_ref_fluxes
            
            #import pickle as pk
            #pk.dump([flux_minus_all_v, flux_plus_all_v, flux_minus_midpt_all_v, flux_plus_midpt_all_v], open('newclima.pk','wb'))

            # call radiation for clearsky case
            if do_holes == True:
                _, out_ref_fluxes_clear = get_reflected_1d(nlevel, wno,nwno,ng_clima,nt_clima,
                        DTAU_clear[:,:,ig], TAU_clear[:,:,ig], W0_clear[:,:,ig], COSB_clear[:,:,ig],
                        GCOS2_clear[:,:,ig],ftau_cld_clear[:,:,ig],ftau_ray_clear[:,:,ig],
                        DTAU_OG_clear[:,:,ig], TAU_OG_clear[:,:,ig], W0_OG_clear[:,:,ig], COSB_OG_clear[:,:,ig],
                        surf_reflect, ubar0_clima,ubar1_clima,cos_theta, F0PI,
                        single_phase,multi_phase,
                        frac_a,frac_b,frac_c,constant_back,constant_forward, 
                        get_toa_intensity=0, get_lvl_flux=1)
            
                flux_minus_all_v_clear, flux_plus_all_v_clear, flux_minus_midpt_all_v_clear, flux_plus_midpt_all_v_clear = out_ref_fluxes_clear
                
                #weighted average of cloudy and clearsky
                flux_plus_midpt_all_v = (1.0 - fhole)* flux_plus_midpt_all_v + fhole * flux_plus_midpt_all_v_clear
                flux_minus_midpt_all_v = (1.0 - fhole)* flux_minus_midpt_all_v + fhole * flux_minus_midpt_all_v_clear
                flux_plus_all_v = (1.0 - fhole)* flux_plus_all_v + fhole * flux_plus_all_v_clear
                flux_minus_all_v = (1.0 - fhole)* flux_minus_all_v + fhole * flux_minus_all_v_clear

            flux_net_v_layer += (np.sum(flux_plus_midpt_all_v,axis=3)-np.sum(flux_minus_midpt_all_v,axis=3))*gauss_wts[ig]
            flux_net_v += (np.sum(flux_plus_all_v,axis=3)-np.sum(flux_minus_all_v,axis=3))*gauss_wts[ig]

            #======="""
            #nlevel = atm.c.nlevel

            """
            <<<<<<< GFLUXV
            ng_clima,nt_clima=1,1
            ubar0_clima = ubar0*0+0.5
            ubar1_clima = ubar1*0+0.5

            RSFV = 0.01 # from tgmdat.f of EGP
            
            b_surface = 0.0 +RSFV*ubar0[0]*F0PI*np.exp(-TAU[-1,:,ig]/ubar0[0])
            
            delta_approx = 0 # assuming delta approx is already applied on opds 
                        
            flux_minus_all_v, flux_plus_all_v, flux_minus_midpt_all_v, flux_plus_midpt_all_v = get_reflected_1d_gfluxv(nlevel, wno,nwno, ng_clima,nt_clima, DTAU[:,:,ig], TAU[:,:,ig], W0[:,:,ig], COSB[:,:,ig],
                                                                                       surf_reflect,b_top,b_surface,ubar0_clima, F0PI,tridiagonal, delta_approx)
            
            import pickle as pk
            pk.dump([flux_minus_all_v, flux_plus_all_v, flux_minus_midpt_all_v, flux_plus_midpt_all_v], open('gfluxv.pk','wb'))

            flux_net_v_layer += (np.sum(flux_plus_midpt_all_v,axis=3)-np.sum(flux_minus_midpt_all_v,axis=3))*gauss_wts[ig]
            flux_net_v += (np.sum(flux_plus_all_v,axis=3)-np.sum(flux_minus_all_v,axis=3))*gauss_wts[ig]
            """

            flux_plus_v += flux_plus_all_v*gauss_wts[ig]
            flux_minus_v += flux_minus_all_v*gauss_wts[ig]

        #if full output is requested add in xint at top for 3d plots


    if thermal:

        #use toon method (and tridiagonal matrix solver) to get net cumulative fluxes 
        
        for ig in range(ngauss): # correlated - loop (which is different from gauss-tchevychev angle)
            
            #remember all OG values (e.g. no delta eddington correction) go into thermal as well as 
            #the uncorrected raman single scattering 
            
            #"""<<<<<<< NEWCLIMA
            hard_surface = 0 
            _,out_therm_fluxes = get_thermal_1d(nlevel, wno,nwno,ng,nt,temperature,
                                            DTAU_OG[:,:,ig], W0_no_raman[:,:,ig], COSB_OG[:,:,ig], 
                                            pressure,ubar1,
                                            surf_reflect, hard_surface, dwni, calc_type=1)

            flux_minus_all_i, flux_plus_all_i, flux_minus_midpt_all_i, flux_plus_midpt_all_i = out_therm_fluxes

            if do_holes == True:
            #clearsky case
                _,out_therm_fluxes_clear = get_thermal_1d(nlevel, wno,nwno,ng,nt,temperature,
                                            DTAU_OG_clear[:,:,ig], W0_no_raman_clear[:,:,ig], COSB_OG_clear[:,:,ig], 
                                            pressure,ubar1,
                                            surf_reflect, hard_surface, dwni, calc_type=1)
                
                flux_minus_all_i_clear, flux_plus_all_i_clear, flux_minus_midpt_all_i_clear, flux_plus_midpt_all_i_clear= out_therm_fluxes_clear
                
                #weighted average of cloudy and clearsky
                flux_plus_midpt_all_i = (1.0 - fhole)* flux_plus_midpt_all_i + fhole * flux_plus_midpt_all_i_clear
                flux_minus_midpt_all_i = (1.0 - fhole)* flux_minus_midpt_all_i + fhole * flux_minus_midpt_all_i_clear
                flux_plus_all_i = (1.0 - fhole)* flux_plus_all_i + fhole * flux_plus_all_i_clear
                flux_minus_all_i = (1.0 - fhole)* flux_minus_all_i + fhole * flux_minus_all_i_clear

            flux_plus += flux_plus_all_i*gauss_wts[ig]
            flux_minus += flux_minus_all_i*gauss_wts[ig]
            flux_plus_midpt += flux_plus_midpt_all_i*gauss_wts[ig]#*weights
            flux_minus_midpt += flux_minus_midpt_all_i*gauss_wts[ig]#*weights
            #"""
            
            """<<<<<<< OG CODE
            calc_type=1 # this line might change depending on Natasha's new function
            
            #for iubar,weights in zip(ugauss_angles,ugauss_weights):
            flux_minus_all_i, flux_plus_all_i, flux_minus_midpt_all_i, flux_plus_midpt_all_i=get_thermal_1d_gfluxi(nlevel,wno,nwno,ng,nt,temperature,DTAU_OG[:,:,ig], W0_no_raman[:,:,ig], COSB_OG[:,:,ig], pressure,ubar1,surf_reflect, ugauss_angles,ugauss_weights, tridiagonal,calc_type, dwni)#,bb , y2, tp, tmin, tmax)
            
            flux_plus += flux_plus_all_i*gauss_wts[ig]#*weights
            flux_minus += flux_minus_all_i*gauss_wts[ig]#*weights

            flux_plus_midpt += flux_plus_midpt_all_i*gauss_wts[ig]#*weights
            flux_minus_midpt += flux_minus_midpt_all_i*gauss_wts[ig]#*weights
            """

        #"""<<<<<<< NEWCLIMA
        #compresses in gauss-chebyshev angle space 
        #the integration over the "disk" of the planet opposed to the 
        #other gauss angles which are for the correlatedk tables
        #gweight = np.array([0.01574791, 0.07390887, 0.14638699, 0.16717464, 0.09678159])
        #tweight = np.array([1])#[6.28318531])
        flux_plus = compress_thermal(nwno, flux_plus, gweight, tweight)
        flux_minus= compress_thermal(nwno, flux_minus, gweight, tweight)
        flux_plus_midpt= compress_thermal(nwno, flux_plus_midpt, gweight, tweight)
        flux_minus_midpt= compress_thermal(nwno, flux_minus_midpt, gweight, tweight)
        #"""

        for wvi in range(nwno):
            flux_net_ir_layer += (flux_plus_midpt[:,wvi]-flux_minus_midpt[:,wvi]) * dwni[wvi]
            flux_net_ir += (flux_plus[:,wvi]-flux_minus[:,wvi]) * dwni[wvi]

            flux_plus_ir[:,wvi] += flux_plus[:,wvi] * dwni[wvi]
            flux_minus_ir[:,wvi] += flux_minus[:,wvi] * dwni[wvi]
        """
        print('debug fluxes in get_fluxes', temperature)
        for wvi in range(nwno):
            for il in range(len(flux_plus_midpt[:,0])):
                print(wvi, dwni[wvi],flux_plus_midpt[il,wvi],flux_minus_midpt[il,wvi] )
        """

        #if full output is requested add in flux at top for 3d plots
    
    return flux_net_v_layer, flux_net_v, flux_plus_v, flux_minus_v , flux_net_ir_layer, flux_net_ir, flux_plus_ir, flux_minus_ir


@jit(nopython=True)
def replace_temp(tuple_old, NEW_TEMP): 
    return Atmosphere_Tuple(tuple_old.dtdp,tuple_old.mmw_layer,
        tuple_old.nlevel,NEW_TEMP,tuple_old.p_level,tuple_old.condensables,
        tuple_old.condensable_abundances,tuple_old.condensable_weights,tuple_old.scale_height)

def find_strat(bundle, nofczns,nstr,
        temp,pressure,dtdp, 
        AdiabatBundle,
        opacityclass, grav, 
        rfaci, rfacv, tidal ,
        Opagrid,
        CloudParameters,
        save_profile, all_profiles, all_opd,
        flux_net_ir_layer, flux_plus_ir_attop,
        verbose=1, moist = None,
        save_kzz=False,self_consistent_kzz=True,diseq=False, all_kzz=[]):
    
    """
    Parameters
    ----------
    bundle : object
    
    nofczns : int
        number of convection zones
    nstr : list
        list of indices for the convective layers
    temp : array
        array of temperature profile
    pressure : array
        array of pressure profile
    dtdp : array
        array of lapse rate/adiabat profile
    AdiabatBundle : namedtuple
        includes:
        - t_table : array 
            array of Temperature values with 53 entries
        - p_table : array 
            array of Pressure value with 26 entries
        - grad : array 
            array of gradients of dimension 53*26
        - cp : array 
            array of cp of dimension 53*26
    opacityclass : object
        object that contains the opacity information
    grav : float
        gravity cgs
    rfaci : float 
        IR flux addition fraction 
    rfacv : float
        Visible flux addition fraction
    tidal : ndarray 
        effectively sigmaTeff^4, gets added to the convergence critiera (e.g. F_IR*rfacI + F_SOL*rfacV + tidal)
    Opagrid : namedtuple
        Any opacity grid info such as wavelength grids, temperature pressure grids, tmax and tmin
    CloudParameters : namedtuple
        tuple containing the cloud parameters, including the cloudy flag, fsed, mh, b, param, directory, and condensates
    save_profile : bool
        flag to save the profile
    all_profiles : list
        list of all thermal structures
    all_opd : array
        array of all opacities
    flux_net_ir_layer : array
        array of net fluxes in the IR
    flux_plus_ir_attop : array
        array of IR fluxes at the top of the atmosphere
    moist : bool
        Defalt= False; computes moist adiabat
    save_kzz : bool
        Default = False, if True save the kzz profile for all iterations
    self_consistent_kzz : bool
        Default = True, calculates kzz for each profile, does not use constant kzz
    diseq : bool
        Default = False, flags whether to do disequilibrium chemistry calculations or not
    
    """
    #unpack  #neb does not lok used 
    #F0PI = opacityclass.relative_flux

    #loudy = CloudParameters.cloudy #neb does not lok used 
    #if cloudy: #neb does not lok used 
    #    cld_species = virga_condensates
    #else: 
    #    cld_species= []

    # new conditions for this routine
    convergence_criteriaT = namedtuple('Conv',['it_max','itmx','conv','convt','x_max_mult'])

    #itmx_strat = 5 #itmx  # outer loop counter
    #it_max_strat = 8 # its # inner loop counter # original code is 8
    #conv_strat = 5.0 # conv
    #convt_strat = 3.0 # convt 
    x_max_mult = 7.0
    
    convergence_criteria = convergence_criteriaT(it_max=8, itmx=5, conv=5.0, convt=3.0, x_max_mult=x_max_mult)

    ip2 = -10 #?
    subad = 0.98 # degree to which layer can be subadiabatic and
                    # we still make it adiabatic
    ifirst = 10-1  # start looking after this many layers from top for a conv zone
                   # -1 is for python referencing
    iend = 0 #?
    final = False

    #call bundle for moist adiabat option (moved out of if statement for numba issue)
    bundle.add_pt( temp, pressure)
    bundle.premix_atmosphere(opacityclass,verbose=verbose)
    Atmosphere = calculate_atm(bundle,opacityclass,only_atmosphere=True)
    dtdp = Atmosphere.dtdp

    grad_x, cp_x =convec(temp,pressure, AdiabatBundle, Atmosphere, moist = moist)

    while dtdp[nstr[1]-1] >= subad*grad_x[nstr[1]-1] :
        ratio = dtdp[nstr[1]-1]/grad_x[nstr[1]-1]

        if ratio > 1.8 :
            if verbose: print("Move up two levels")
            ngrow = 2
            nstr = growup( 1, nstr , ngrow)
        else :
            ngrow = 1
            nstr = growup( 1, nstr , ngrow)
        
        if nstr[1] < 5 :
            raise ValueError( "Convection zone grew to Top of atmosphere, Need to Stop")
        
        profile_flag, pressure, temp, dtdp, CloudParameters, cld_out, flux_net_ir_layer, flux_net_v_layer,flux_plus_ir_attop, all_profiles,all_opd,all_kzz = profile(bundle,
            nofczns, nstr, temp, pressure, 
            AdiabatBundle,opacityclass,
            grav,
            rfaci,rfacv,tidal,
            Opagrid,
            CloudParameters,
            save_profile,all_profiles,all_opd,
            convergence_criteria, final,
            flux_net_ir_layer=flux_net_ir_layer, flux_plus_ir_attop=flux_plus_ir_attop,
            verbose=verbose,moist = moist,
            save_kzz=save_kzz,self_consistent_kzz=self_consistent_kzz,diseq=diseq, all_kzz=all_kzz)


    # if nofczns == 2: JM* #should be a flag here since this block in EGP is skipped if only 1 convective zone but convergence is better when enabled
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
        if verbose: print(" convection zone status")
        if verbose: print(nstr[0],nstr[1],nstr[2],nstr[3],nstr[4],nstr[5])
        if verbose: print(nofczns)

        nofczns = 2 #keeping this hardcoded here to easily transition to future 3 zone calculation
        nstr[4]= nstr[1]
        nstr[5]= nstr[2]
        nstr[1]= i_max
        nstr[2] = i_max
        nstr[3] = i_max #+ 1 #JM: Should be i_max + 1 according to EGP, but runs into ValueError when used
        if verbose: print(nstr)
        if nstr[3] >= nstr[4] :
            #print(nstr[0],nstr[1],nstr[2],nstr[3],nstr[4],nstr[5])
            #print(nofczns)
            raise ValueError("Overlap happened !")
        profile_flag,pressure, temp, dtdp, CloudParameters,cld_out,flux_net_ir_layer,flux_net_v_layer, flux_plus_ir_attop, all_profiles,  all_opd,all_kzz = profile(bundle,
            nofczns, nstr, temp, pressure, 
            AdiabatBundle,opacityclass,
            grav,
            rfaci,rfacv,tidal,
            Opagrid,
            CloudParameters,
            save_profile,all_profiles,all_opd,
            convergence_criteria, final,
            flux_net_ir_layer=flux_net_ir_layer, flux_plus_ir_attop=flux_plus_ir_attop,
            verbose=verbose,moist = moist,
            save_kzz=save_kzz,self_consistent_kzz=self_consistent_kzz,diseq=diseq, all_kzz=all_kzz)
        

        i_change = 1
        while i_change == 1 :
            if verbose: print("Grow Phase : Upper Zone")
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
                if verbose: print(nstr)

                profile_flag, pressure, temp, dtdp,CloudParameters,cld_out,flux_net_ir_layer,flux_net_v_layer, flux_plus_ir_attop,  all_profiles,all_opd ,all_kzz= profile(bundle,
                                nofczns, nstr, temp, pressure, 
                                AdiabatBundle,opacityclass,
                                grav,
                                rfaci,rfacv,tidal,
                                Opagrid,
                                CloudParameters,
                                save_profile,all_profiles,all_opd,
                                convergence_criteria, final,
                                flux_net_ir_layer=flux_net_ir_layer, flux_plus_ir_attop=flux_plus_ir_attop,
                                verbose=verbose, moist = moist,
            save_kzz=save_kzz,self_consistent_kzz=self_consistent_kzz,diseq=diseq, all_kzz=all_kzz)

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
                if verbose: print(nstr)
                  
                profile_flag,pressure, temp, dtdp, CloudParameters,cld_out,flux_net_ir_layer, flux_net_v_layer, flux_plus_ir_attop,all_profiles, all_opd,all_kzz = profile(bundle,
                                nofczns, nstr, temp, pressure, 
                                AdiabatBundle,opacityclass,
                                grav,
                                rfaci,rfacv,tidal,
                                Opagrid,
                                CloudParameters,
                                save_profile,all_profiles,all_opd,
                                convergence_criteria, final,
                                flux_net_ir_layer=flux_net_ir_layer, flux_plus_ir_attop=flux_plus_ir_attop,
                                verbose=verbose, moist = moist,
            save_kzz=save_kzz,self_consistent_kzz=self_consistent_kzz,diseq=diseq, all_kzz=all_kzz)

            flag_final_convergence = 1
        
    itmx_strat =6
    it_max_strat = 10
    conv_strat = 2.0 
    convt_strat = 2.0
    x_max_mult = x_max_mult/2.0
    ip2 = -10
    convergence_criteria=convergence_criteriaT(it_max_strat,itmx_strat,conv_strat,convt_strat,x_max_mult)
    final = True
    if verbose: print("final",nstr)

    profile_flag,pressure, temp, dtdp, CloudParameters,cld_out,flux_net_ir_layer,flux_net_v_layer, flux_plus_ir_attop,  all_profiles,all_opd,all_kzz = profile(bundle,
                nofczns, nstr, temp, pressure, 
                AdiabatBundle,opacityclass,
                grav,
                rfaci,rfacv,tidal,
                Opagrid,
                CloudParameters,
                save_profile,all_profiles,all_opd,
                convergence_criteria, final,
                flux_net_ir_layer=flux_net_ir_layer, flux_plus_ir_attop=flux_plus_ir_attop,
                verbose=verbose,moist = moist,
            save_kzz=save_kzz,self_consistent_kzz=self_consistent_kzz,diseq=diseq, all_kzz=all_kzz)
                #(mieff_dir, it_max_strat, itmx_strat, conv_strat, convt_strat, nofczns,nstr,x_max_mult,
                #temp,pressure, F0PI, t_table, p_table, grad, cp,opacityclass, grav, 
                #rfaci, rfacv, nlevel, tidal, tmin, tmax, dwni, bb , y2 , tp, final, 
                #cloudy, cld_species,mh,fsed,flag_hack,save_profile, all_profiles, all_opd,
                #opd_cld_climate,g0_cld_climate,w0_cld_climate,beta, param_flag,flux_net_ir_layer, 
                #flux_plus_ir_attop, verbose=verbose,
            #fhole=fhole, fthin_cld=fthin_cld, do_holes = do_holes, moist = moist, cold_trap = cold_trap)

    #    else :
    #        raise ValueError("Some problem here with goto 125")
        
    if profile_flag == 0:
        if verbose: print("ENDING WITHOUT CONVERGING")
    elif profile_flag == 1:
        if verbose: print("YAY ! ENDING WITH CONVERGENCE")

    chem = bundle.inputs['atmosphere']['profile']
    #right now this bundle does not have the up to date chemistry
    # TO DO : add chemistry and also condense last three "all" variables into one tuple
    return profile_flag, pressure, temp, dtdp, nstr ,flux_net_ir_layer, flux_net_v_layer, flux_plus_ir_attop, chem, cld_out,all_profiles,all_opd,all_kzz

def profile(bundle, nofczns, nstr, temp, pressure, 
            AdiabatBundle,opacityclass,
            grav,
            rfaci,rfacv,tidal,
            Opagrid,
            CloudParameters,
            save_profile,all_profiles,all_opd,
            convergence_criteria, final,
            flux_net_ir_layer=None, flux_plus_ir_attop=None,first_call_ever=False,
            verbose=True, moist = None,
            save_kzz=False,self_consistent_kzz=True,diseq=False,all_kzz=[]):
    """
    Parameters
    ----------
    bundle : object
        The bundle object containing the inputs and other parameters.
    nofczns : int
        Number of convective zones.
    nstr : array
        Array of the layer of convective zone locations.
    temp : array
        Temperature profile.
    pressure : array
        Pressure profile.
    AdiabatBundle : tuple
        Tuple containing the adiabat table, pressure table, gradient, and cp.
    opacityclass : object
        Object containing opacity data.
    grav : float
        Gravity in cgs.
    rfaci : float
        IR flux addition fraction.
    rfacv : float
        Visible flux addition fraction.
    tidal : ndarray
        Effectively sigmaTeff^4, gets added to the convergence criteria.
    Opagrid : tuple
        Tuple containing the opacities and other information.
    CloudParameters : tuple
        Tuple containing the cloud parameters, including the cloudy flag, fsed, mh, b, param, directory, and condensates.
    save_profile : bool
        If True, saves the profile for every iteration.
    all_profiles : array
        Array of all thermal structures.
    all_opd : array
        Array of all opacities.
    convergence_criteria : namedtuple
        Defines convergence criteria for max number of loops and other numerical recipes values.
    final : bool
        If True, indicates the final iteration.
    flux_net_ir_layer : array, optional
        Array of net fluxes in the IR. Default is None.
    flux_plus_ir_attop : array, optional
        Array of IR fluxes at the top of the atmosphere. Default is None.
    first_call_ever : bool, optional
        If True, indicates the first call to the function. Default is False.
    verbose : bool, optional
        If True, prints additional information. Default is True.
    moist : bool, optional
        If True, uses moist adiabat. Default is None.
    save_kzz : bool, optional
        If True, saves the kzz profile for every iteration. Default is False.
    self_consistent_kzz : bool, optional
        If True, uses the self-consistent kzz profile (not constant kzz). Default is True.
    diseq : bool, optional
        If True, runs disequilibrium chemistry workflow. Default is False.
    all_kzz : array, optional
        Array of all kzz profiles. Default is an empty array.
    """
    #under what circumstances to we compute quench levels 
    full_kinetis ='photochem' in str(bundle.inputs['approx']['chem_method'])
    do_quench_appox = diseq and (not full_kinetis)

    #unpack 
    F0PI = opacityclass.relative_flux 

    convt = convergence_criteria.convt
    itmx = convergence_criteria.itmx
    cloudy =  CloudParameters.cloudy

   #under what circumstances to do we compute a self consistent kzz calc 
    sc_kzz_and_clouds = cloudy #THIS IS ALWAYS BE TRUE.. 
    sc_kzz_and_diseq = self_consistent_kzz and diseq  
    do_kzz_calc = sc_kzz_and_clouds or sc_kzz_and_diseq
    constant_kzz =  ((not self_consistent_kzz) and diseq) #((not self_consistent_kzz) and cloudy) or
    if constant_kzz: 
        kz_chem = bundle.inputs['atmosphere']['kzz'].get('constant_kzz')
        
    virga_kwargs = {key.replace('virga_',''):getattr(CloudParameters,key) for key in CloudParameters._fields if 'virga' in key}
    hole_kwargs = {key.replace('patchy_',''):getattr(CloudParameters,key) for key in CloudParameters._fields if 'patchy' in key}
    do_holes = hole_kwargs.get('do_holes', None); fhole=hole_kwargs.get('fhole', None)
    #cld_species = CloudParameters.virga_condensates #neb commenting out not used 
        

    min_temp = np.min(temp)
    # Don't use large step_max option for cold models, much better converged with smaller stepping unless it's cloudy
    if min_temp <= 250:# and cloudy != 1:
        egp_stepmax = True
    else: 
        egp_stepmax = False

    conv_flag = 0
    # taudif is fixed to be 0 here since it is needed only for clouds mh
    taudif = 0.0
    taudif_tol = 0.1

    ### 0) Recompte INPUT PT with new Adiabat 
    if moist == True:
        #moist adiabat with simple chemeq 
        bundle.add_pt( temp, pressure)
        bundle.premix_atmosphere(opa=opacityclass,verbose=verbose)
        Atmosphere = calculate_atm(bundle,opacityclass,only_atmosphere=True)

        # first calculate the convective zones
        for nb in range(0,3*nofczns,3):
        
            n_strt_b= nstr[nb+1]
            n_ctop_b= n_strt_b+1
            n_bot_b= nstr[nb+2] +1

            for j1 in range(n_ctop_b,n_bot_b+1): 
                press = sqrt(pressure[j1-1]*pressure[j1])
                grad_x, cp_x = moist_grad( temp[j1-1], press, AdiabatBundle, Atmosphere, j1-1)
                temp[j1]= exp(log(temp[j1-1]) + grad_x*(log(pressure[j1]) - log(pressure[j1-1])))

    else: #non moist adiabat case
        # first calculate the convective zones
        for nb in range(0,3*nofczns,3):
            
            n_strt_b= nstr[nb+1]
            n_ctop_b= n_strt_b+1
            n_bot_b= nstr[nb+2] +1

            for j1 in range(n_ctop_b,n_bot_b+1): 
                press = sqrt(pressure[j1-1]*pressure[j1])
                grad_x, cp_x = did_grad_cp( temp[j1-1], press, AdiabatBundle)
                temp[j1]= exp(log(temp[j1-1]) + grad_x*(log(pressure[j1]) - log(pressure[j1-1])))
        
    temp_old= np.copy(temp)
    if save_profile == 1: all_profiles = np.append(all_profiles,temp_old)

    ### 1) ALWAYS UPDATE PT, CHEM, OPACITIES
    bundle.add_pt( temp, pressure)
    bundle.premix_atmosphere(opa = opacityclass,quench_levels=None,verbose=verbose)
    
    # get opacities for the first time with simple chem 
    # this first call will be refreshed before tstart if things like chemistry are changed from the quench approx
    OpacityWEd, OpacityNoEd, ScatteringPhase, Disco, Atmosphere , holes =  calculate_atm(bundle, opacityclass)
    #was there hole information returned? 
    #i am conpressing this for readability as these things are usually None, unless the user sets it 
    OpacityWEd_clear=holes[0]; OpacityNoEd_clear=holes[0]

    ### 2) IF: UPDATE KZZ 
    kz_cloud = 0 # dummy value so that the variable exists in cloudless cases
    if do_kzz_calc:
        kz = update_kzz(grav, tidal, AdiabatBundle, nstr, Atmosphere, 
               #these are only needed if you dont have fluxes and need to compute them
               OpacityWEd=OpacityWEd, OpacityNoEd=OpacityNoEd,ScatteringPhase=ScatteringPhase,Disco=Disco,Opagrid=Opagrid, F0PI=F0PI,
               OpacityWEd_clear=OpacityWEd_clear,OpacityNoEd_clear=OpacityNoEd_clear,
               #kwargs for get_kzz function
               moist=moist, do_holes=do_holes, fhole=fhole)
        if CloudParameters.cloudy != "cloudless":
            # this is a slightly bad hack just so i can move on; 
            bundle.inputs['atmosphere']['kzz']['sc_kzz']=kz #bookeeping current kz 

        if save_kzz: all_kzz = np.append(all_kzz,kz)
        #are clouds turned on such that we need the sc kzz for virga? we'll say always yes
        kz_cloud = kz 
        #is self consistent kz needed for diseq chem too? 
        if sc_kzz_and_diseq: 
            kz_chem = kz

    ### 3) IF: COMPLEX CHEM
    ##  3-a) option 1: GET QUENCH LEVELS FOR DISEQ and UPDATE CHEM
    if do_quench_appox:
        quench_levels=update_quench_levels(bundle, Atmosphere, kz_chem, grav,verbose=verbose)
        bundle.premix_atmosphere(opa=opacityclass,quench_levels=quench_levels,verbose=verbose)
    ##  3-b) option 2: GET PHOTOCHEM
    if full_kinetis: 
        quench_levels=update_quench_levels(bundle, Atmosphere, kz_chem, grav,verbose=verbose)
        bundle.premix_atmosphere_photochem(quench_levels=quench_levels,verbose=verbose)
    
    ### 4) COMPUTE CLOUDS - always pass to update_clouds, which will not update the cloud profile unless cloudy == "selfconsistent"
    cld_out,df_cld, taudif, taudif_tol, CloudParameters=update_clouds(bundle, opacityclass, CloudParameters,Atmosphere,
                                                                        kz_cloud,virga_kwargs,hole_kwargs,verbose=verbose)
    
    ### 5) IF NEEDED: COMPUTE OPACITIES 
    refresh_needed = full_kinetis or do_quench_appox or cloudy != "cloudless"
    if refresh_needed:
        OpacityWEd, OpacityNoEd, ScatteringPhase, Disco, Atmosphere,hole=calculate_atm(bundle, opacityclass )
        #these are most of the time returned as None, if no clouds and no patchy clouds are requested
        OpacityWEd_clear=hole[0]; OpacityNoEd_clear=hole[1]
    
    #import dill 
    #with open('from_prof_tstart_new.pk','wb') as file: 
    #    dill.dump([bundle, nofczns,nstr,convergence_criteria, rfaci, rfacv, tidal,
    #            Atmosphere, OpacityWEd, OpacityNoEd,ScatteringPhase, Disco,Opagrid, AdiabatBundle,
    #            F0PI,
    #            save_profile, all_profiles, 
    #            verbose, moist , egp_stepmax ],file)

    ## begin bigger loop which gets opacities
    for iii in range(itmx):

        if do_holes == True:
            temp, dtdp, all_profiles,  flux_net_ir_layer,flux_net_v_layer, flux_plus_ir_attop = t_start(
                nofczns,nstr,convergence_criteria, rfaci, rfacv, tidal,
                Atmosphere, OpacityWEd, OpacityNoEd,ScatteringPhase, Disco,Opagrid, AdiabatBundle,
                F0PI,
                save_profile, all_profiles, 
                verbose=verbose, moist = moist, egp_stepmax = egp_stepmax, 
                do_holes=do_holes, fhole=fhole, hole_OpacityWEd=OpacityWEd_clear,hole_OpacityNoEd=OpacityNoEd_clear)
        else:
            temp, dtdp, all_profiles,  flux_net_ir_layer,flux_net_v_layer, flux_plus_ir_attop = t_start(
                    nofczns,nstr,convergence_criteria, rfaci, rfacv, tidal,
                    Atmosphere, OpacityWEd, OpacityNoEd,ScatteringPhase, Disco,Opagrid, AdiabatBundle,
                    F0PI,
                    save_profile, all_profiles, 
                    verbose=verbose, moist = moist, egp_stepmax = egp_stepmax)
        
        ### 1) ALWAYS UPDATE PT, CHEM, OPACITIES
        bundle.add_pt( temp, pressure)
        #simple chem no quenching 
        bundle.premix_atmosphere(opa = opacityclass,quench_levels=None,verbose=verbose) 
        
        ### 2) IF: UPDATE KZZ 
        if do_kzz_calc:
            #NEB: commenting out because we have fluxes from the above output. lets rely on those for now. 
            #get opacities for KZZ calculation
            #OpacityWEd, OpacityNoEd, ScatteringPhase, Disco, Atmosphere =  calculate_atm(bundle, opacityclass)
            #if do_holes == True:
            #    OpacityWEd_clear, OpacityNoEd_clear, _, _, _ =  calculate_atm(bundle, opacityclass, fthin_cld, do_holes=True)    
            #else: 
            #    OpacityWEd_clear=None; OpacityNoEd_clear=None
            kz = update_kzz(grav, tidal, AdiabatBundle, nstr, Atmosphere, 
                #these are only needed if you dont have fluxes and need to compute them
                #OpacityWEd=OpacityWEd, OpacityNoEd=OpacityNoEd,ScatteringPhase=ScatteringPhase,Disco=Disco,Opagrid=Opagrid, F0PI=F0PI,
                #OpacityWEd_clear=OpacityWEd_clear,OpacityNoEd_clear=OpacityNoEd_clear,
                flux_net_ir_layer=flux_net_ir_layer,flux_plus_ir_attop=flux_plus_ir_attop,
                #kwargs for get_kzz function
                moist=moist, do_holes=do_holes,fhole=fhole)
            if save_kzz: all_kzz = np.append(all_kzz,kz)
            #are clouds turned on such that we need the sc kzz for virga? 
            if sc_kzz_and_clouds: 
                kz_cloud = kz 
            #is self consistent kz needed for diseq chem too? 
            if sc_kzz_and_diseq: 
                kz_chem = kz 

        ### 3) IF: COMPLEX CHEM
        ##  3-a) option 1: GET QUENCH LEVELS FOR DISEQ and UPDATE CHEM
        if do_quench_appox:   
            quench_levels=update_quench_levels(bundle, Atmosphere, kz_chem, grav,verbose=verbose)
            bundle.premix_atmosphere(opa=opacityclass,quench_levels=quench_levels,verbose=verbose)
        
        ##  3-b) option 2: GET PHOTOCHEM
        if full_kinetis: 
            quench_levels=update_quench_levels(bundle, Atmosphere, kz_chem, grav,verbose=verbose)
            bundle.premix_atmosphere_photochem(quench_levels=quench_levels,verbose=verbose)
            
        ### 4) IF: COMPUTE CLOUDS 
        cld_out,df_cld, taudif, taudif_tol, CloudParameters = update_clouds(bundle, opacityclass, CloudParameters,Atmosphere,
                                                                        kz_cloud,virga_kwargs,hole_kwargs,verbose=verbose)
        
        if save_profile and cloudy == "selfconsistent":
            all_opd = np.append(all_opd,df_cld['opd'].values[55::196]) #save opd at 4 micron
        
        ### 5) IF NEEDED: COMPUTE OPACITIES 
        refresh_needed = full_kinetis or do_quench_appox or cloudy == "selfconsistent"
        if refresh_needed:
            OpacityWEd, OpacityNoEd, ScatteringPhase, Disco, Atmosphere,hole=calculate_atm(bundle, opacityclass )
            OpacityWEd_clear=hole[0]; OpacityNoEd_clear=hole[0]
        
        # 6) PREP RETURNS! 
        # TO DO : add chemistry and also condense last three "all" variables into one tuple
        RETURNS = [conv_flag, pressure, temp , dtdp, 
                        CloudParameters, cld_out,
                        flux_net_ir_layer, flux_net_v_layer, flux_plus_ir_attop, 
                        all_profiles, all_opd, all_kzz]
        
        ert = 0.0 # avg temp change
        scalt= 1.5

        dtx= abs(temp-temp_old)
        ert = np.sum(dtx) 
        
        temp_old= np.copy(temp)
        
        ert = ert/(float(len(temp))*scalt)
        
        if ((iii > 0) & (ert < convt) & (taudif < taudif_tol)) :
            if verbose: print("Profile converged before itmx")
            conv_flag = 1
            #update convergence flag!! 
            RETURNS[0] = conv_flag
            if final == True :
                #itmx = 6
                convergence_criteria=convergence_criteria._replace(itmx=6)
            else :
                #itmx = 3       
                convergence_criteria=convergence_criteria._replace(itmx=3)     
            
            return RETURNS
        
        if verbose: print("Big iteration is ",min(temp), iii)
    
    if conv_flag == 0:
        if verbose: print("Not converged")
    else :
        if verbose: print("Profile converged after itmx hit")
    
    return RETURNS

