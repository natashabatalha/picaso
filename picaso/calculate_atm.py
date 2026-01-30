import numpy as np
from .atmsetup import ATMSETUP
from .optics import compute_opacity

from collections import namedtuple

Atmosphere_Tuple = namedtuple('Atmosphere_Tuple',['dtdp','mmw_layer','nlevel','t_level','p_level','condensables','condensable_abundances','condensable_weights','scale_height'])
OpacityWEd_Tuple = namedtuple("OpacityWEd_Tuple", ["DTAU", "TAU", "W0", "COSB",'ftau_cld','ftau_ray','GCOS2', 'W0_no_raman','f_deltaM'])
ScatteringPhase_Tuple = namedtuple('ScatteringPhase_Tuple',['surf_reflect','single_phase','multi_phase','frac_a','frac_b','frac_c','constant_back','constant_forward'])
Disco_Tuple = namedtuple('Disco_Tuple',['ng','nt', 'gweight','tweight', 'ubar0','ubar1','cos_theta'])
OpacityNoEd_Tuple = namedtuple("OpacityNoEd_Tuple", ["DTAU", "TAU", "W0", "COSB"])


def calculate_atm(bundle, opacityclass, only_atmosphere=False):
    """
    Function to calculate the atmosphere and opacities for the given inputs.
    Parameters
    ----------
    bundle : object
        The bundle object containing the inputs and other parameters.
    opacityclass : object
        The opacity class containing the opacity data.
    only_atmosphere : bool, optional
        If True, no opacities are calculated, just updates the Atmosphere tuple. The default is False.

    """

    inputs = bundle.inputs

    wno = opacityclass.wno
    #nwno = opacityclass.nwno
    ngauss = opacityclass.ngauss
    #gauss_wts = opacityclass.gauss_wts #for opacity

    #check to see if we are running in test mode
    test_mode = inputs['test_mode']

    ############# DEFINE ALL APPROXIMATIONS USED IN CALCULATION #############
    #see class `inputs` attribute `approx`

    #set approx numbers options (to be used in numba compiled functions)
    single_phase = inputs['approx']['rt_params']['toon']['single_phase']
    multi_phase = inputs['approx']['rt_params']['toon']['multi_phase']
    raman_approx =inputs['approx']['rt_params']['common']['raman']
    #method = inputs['approx']['rt_method']
    stream = inputs['approx']['rt_params']['common']['stream']

    #parameters needed for the two term hg phase function. 
    #Defaults are set in config.json
    f = inputs['approx']['rt_params']['common']['TTHG_params']['fraction']
    frac_a = f[0]
    frac_b = f[1]
    frac_c = f[2]
    constant_back = inputs['approx']['rt_params']['common']['TTHG_params']['constant_back']
    constant_forward = inputs['approx']['rt_params']['common']['TTHG_params']['constant_forward']

    #define delta eddington approximinations 
    delta_eddington = inputs['approx']['rt_params']['common']['delta_eddington']

    #pressure assumption
    p_reference =  inputs['approx']['p_reference']

    ############# DEFINE ALL GEOMETRY USED IN CALCULATION #############
    #see class `inputs` attribute `phase_angle`
    

    #phase angle 
    #phase_angle = inputs['phase_angle']
    #get geometry
    geom = inputs['disco']

    #""" NEWCLIMA
    ng, nt = geom['num_gangle'], geom['num_tangle']#1,1 #
    gangle,gweight,tangle,tweight = geom['gangle'], geom['gweight'],geom['tangle'], geom['tweight']
    #lat, lon = geom['latitude'], geom['longitude']
    cos_theta = geom['cos_theta']
    ubar0, ubar1 = geom['ubar0'], geom['ubar1']
    #"""

    #set star parameters
    radius_star = inputs['star']['radius']

    #semi major axis
    #sa = inputs['star']['semi_major']

    #define cloud inputs 
    #for patchy clouds
    do_holes = inputs['clouds'].get('do_holes',False)
    if do_holes == True:
        fthin_cld = inputs['clouds']['fthin_cld']

    #begin atm setup
    atm = ATMSETUP(inputs)

    #Add inputs to class 
    ##############################
    atm.surf_reflect = 0#inputs['surface_reflect']#inputs.get('surface_reflect',0)
    ##############################
    atm.wavenumber = wno
    atm.planet.gravity = inputs['planet']['gravity']
    atm.planet.radius = inputs['planet']['radius']
    atm.planet.mass = inputs['planet']['mass']

    #if dimension == '1d':
    atm.get_profile()
    #elif dimension == '3d':
    #    atm.get_profile_3d()

    #now can get these 
    atm.get_mmw()
    atm.get_density()
    atm.get_altitude(p_reference = p_reference)#will calculate altitude if r and m are given (opposed to just g)
    atm.get_column_density()
    atm.get_dtdp()

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

    nlevel = atm.c.nlevel
    nlayer = atm.c.nlayer

    
    allowed_condensibles = ['H2O', 'CH4', 'NH3', 'Fe']
    our_condesables = [i for i in allowed_condensibles if i in  bundle.inputs['atmosphere']['profile'].keys()]
    condensable_abundances = bundle.inputs['atmosphere']['profile'].loc[:,our_condesables].T.values
    condensable_weights = [atm.weights[i].values[0] for i in our_condesables]

    Atmosphere= Atmosphere_Tuple(atm.layer['dtdp'], atm.layer['mmw'],nlevel,atm.level['temperature'],atm.level['pressure_bar'],
                                    our_condesables,condensable_abundances,condensable_weights,atm.level['scale_height'])
    
    if only_atmosphere: 
        return Atmosphere

    opacityclass.get_opacities(atm)
    
        #check if patchy clouds are requested
    if do_holes == True:
        DTAU, TAU, W0, COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman, f_deltaM = compute_opacity(
            atm, opacityclass, ngauss=ngauss, stream=stream, delta_eddington=delta_eddington,test_mode=test_mode,raman=raman_approx,
            full_output=False, plot_opacity=False, fthin_cld = fthin_cld, do_holes = True)
        #Now let's organize all the data we need for the climate calculations
        #these named tuples operate like classes but they are supported by numba no python 
        OpacityWEd_hole = OpacityWEd_Tuple(DTAU, TAU, W0, COSB,ftau_cld, ftau_ray,GCOS2,  W0_no_raman, f_deltaM)

        OpacityNoEd_hole = OpacityNoEd_Tuple(DTAU_OG, TAU_OG, W0_OG, COSB_OG)
    else: 
        OpacityWEd_hole=None;OpacityNoEd_hole=None 
    
    return_opa_holes = (OpacityWEd_hole,OpacityNoEd_hole)
    #this could refined and deleted by adjust fthin in clouds input, not compute opacity. 
    DTAU, TAU, W0, COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman, f_deltaM= compute_opacity(
            atm, opacityclass, ngauss=ngauss, stream=stream, delta_eddington=delta_eddington,test_mode=test_mode,raman=raman_approx,
            full_output=False, plot_opacity=False)

    #Now let's organize all the data we need for the climate calculations
    #these named tuples operate like classes but they are supported by numba no python 
    OpacityWEd = OpacityWEd_Tuple(DTAU, TAU, W0, COSB,ftau_cld, ftau_ray,GCOS2,  W0_no_raman, f_deltaM)

    OpacityNoEd = OpacityNoEd_Tuple(DTAU_OG, TAU_OG, W0_OG, COSB_OG)

    ScatteringPhase= ScatteringPhase_Tuple(atm.surf_reflect,single_phase,multi_phase,frac_a,frac_b,frac_c,constant_back,constant_forward)

    Disco = Disco_Tuple(ng,nt, gweight,tweight, ubar0,ubar1,cos_theta)



    return OpacityWEd, OpacityNoEd,ScatteringPhase,Disco,Atmosphere, return_opa_holes
    #return DTAU, TAU, W0, COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman , atm.surf_reflect, ubar0,ubar1,cos_theta, single_phase,multi_phase,frac_a,frac_b,frac_c,constant_back,constant_forward, wno,nwno,ng,nt, nlevel, ngauss, gauss_wts, mmw,gweight,tweight
