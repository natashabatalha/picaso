import numpy as np
from numba import jit

from virga import justdoit as vj

from .grad import moist_grad, did_grad_cp, HVapClass, CPClass
from .climate import calculate_atm

# still not developed fully. virga has a function already maybe just use that
#def get_kzz(pressure, temp,grav,mmw,tidal,flux_net_ir_layer, flux_plus_ir_attop,AdiabatBundle,nstr, Atmosphere, moist = False):
@jit(nopython=True, cache=True)
def get_kzz(grav,tidal,flux_net_ir_layer, flux_plus_ir_attop,Adiabat,nstr, Atmosphere, moist = False):

    """
    Parameters
    ----------
    grav : float
        gravity in cgs
    tidal : ndarray 
        effectively sigmaTeff^4, gets added to the convergence critiera (e.g. F_IR*rfacI + F_SOL*rfacV + tidal)
    flux_net_ir_layer : array
        array of net fluxes in the IR
    flux_plus_ir_attop : array
        array of IR fluxes at the top of the atmosphere
    Adiabat : tuple
        tuple containing the adiabat table, pressure table, gradient, and cp
    nstr : array
        array of the layer of convective zone locations
    Atmosphere : tuple
        Contains temperature, pressure, dtdp, mmw, and scale height
    moist : bool
        if True, use moist adiabat
    """

    pressure = Atmosphere.p_level #in bars 
    temp = Atmosphere.t_level # in kelvin
    mmw = Atmosphere.mmw_layer
    dtdp = Atmosphere.dtdp

    grav_cgs = grav*1e2
    p_cgs = pressure *1e6
    p_bar = pressure 
    
    nlevel = len(temp)
    
    r_atmos = 8.3143e7/mmw

    nz= nlevel -1
    p_layer = np.sqrt(p_cgs[1:]*p_cgs[0:-1])#np.zeros_like(p_cgs)
    
    t_layer = 0.5*(temp[1:]+temp[0:-1])
    p_layer_bar = np.sqrt(p_bar[1:]*p_bar[0:-1])
    
    # flux_plux_ir is already summed up with dwni in climate routine
    # so just add to get f_sum

    f_sum = np.sum(flux_plus_ir_attop)

    sigmab =  0.56687e-4 #cgs

    teff_now = (f_sum/sigmab)**0.25
    target_teff = (abs(tidal[0])/sigmab)**0.25
    flx_min = sigmab*((target_teff*0.05)**4)
    
    #print("Teff now ", teff_now, "Target Teff ", target_teff)

    #     we explictly assume that the bottom layer is 100%
    #     convective energy transport.  This helps with
    #     the correction logic below and should always be true
    #     in a well formed model.

    chf = np.zeros_like(tidal)

    chf[nz-1] = f_sum
    
    for iz in range(nz-1-1,-1,-1):
        chf[iz] = f_sum - flux_net_ir_layer[iz]
        ratio_min = (1./3.)*p_layer[iz]/p_layer[iz+1]
        
#     set the minimum allowed heat flux in a layer by assuming some overshoot
#     the 1/3 is arbitrary, allowing convective flux to fall faster than
#     pressure scale height
        
        if chf[iz] < ratio_min*chf[iz+1]:
            chf[iz]= ratio_min*chf[iz+1]
        
#     Now we adjust so that the convective flux is equal to the3
#     target convective flux to see if this helps with the
#     convergence.
    f_target = abs(tidal[0])
    f_actual = chf[nz-1]
    
    ratio = f_target/f_actual
    
    for iz in range(nz-1,-1,-1):
        
        chf[iz] = max(chf[iz]*ratio,flx_min) 

    lapse_ratio = np.zeros_like(t_layer)
    for j in range(len(pressure)-1):
        if moist == True:
            grad_x,cp_x = moist_grad(t_layer[j], p_layer_bar[j], Adiabat, Atmosphere, j)
        else:
            grad_x,cp_x = did_grad_cp(t_layer[j], p_layer_bar[j], Adiabat)
        lapse_ratio[j] = min(np.array([1.0, dtdp[j]/grad_x]))

    
    rho_atmos = p_layer/ (r_atmos * t_layer)
    
    c_p = (7./2.)*r_atmos
    scale_h = r_atmos * t_layer / (grav_cgs)
    
    mixl = np.zeros_like(lapse_ratio)
    for jj in range(len(pressure)-1):
        mixl[jj] = max(0.1,lapse_ratio[jj])*scale_h[jj]
    
    scalef_kz = 1./3.
    
    kz = scalef_kz * scale_h * (mixl/scale_h)**(4./3.) *( ( r_atmos*chf[:-1] ) / ( rho_atmos*c_p ) )**(1./3.)
    
    
    kz = np.append(kz,kz[-1])
    
    
    #### julien moses 2021
    
    # logp = np.log10(pressure)
    # wh = np.where(np.absolute(logp-(-3)) == np.min(np.absolute(logp-(-3))))
    
    # kzrad1 = (5e8/np.sqrt(pressure[nstr[0]:nstr[1]]))*(scale_h[wh]/(620*1e5))*((target_teff/1450)**4)
    # kzrad2 = (5e8/np.sqrt(pressure[nstr[3]:nstr[4]]))*(scale_h[wh]/(620*1e5))*((target_teff/1450)**4)
    # #
    # if nstr[3] != 0:
    #     kz[nstr[0]:nstr[1]] = kzrad1#/100 #*10#kz[nstr[0]:nstr[1]]/1.0
    #     kz[nstr[3]:nstr[4]] = kzrad2#/100 #*10 #kz[nstr[3]:nstr[4]]/1.0
    # else:
    #     kz[nstr[0]:nstr[1]] = kzrad1#/100
    pm_range = 8 # aribitrary range to average over to get the mean kz *JM working on changing to be based on scale height
    dz = scale_h[1:] * (np.log((p_layer[:-1]/1e6) / (p_layer[1:]/1e6)))
    z = np.zeros(nlevel - 1)
    z[0] = dz[0]

    for i in range(1, nlevel-2):#nlevel - 1):#index change neb
        z[i] = z[i - 1] + dz[i] 

    kz_upper = [] # average the upper radiative zone kz
    for i in range(nstr[0], nstr[1]):
        above_range = np.abs(i - (np.abs(z - (z[i] + 2*scale_h[i]))).argmin()) # find the index of the layer 1 scale height above the current layer
        below_range = np.abs(i - (np.abs(z - (z[i] - 2*scale_h[i]))).argmin())
        # start_index = np.maximum(nstr[0], i - pm_range)
        # end_index = np.minimum(nstr[1], i + pm_range)
        start_index = np.maximum(nstr[0], i - above_range)
        end_index = np.minimum(nstr[1], i + below_range)
        kz_upper.append(np.mean(kz[start_index:end_index]))
    kz_upper = np.array(kz_upper)

    if nstr[3] != 0:
        kz[nstr[0]:nstr[1]] = kz_upper

        kz_lower = [] # average the lower radiative zone kz
        for i in range(nstr[3], nstr[4]):
            above_range = np.abs(i - (np.abs(z - (z[i] + 2*scale_h[i]))).argmin()) # find the index of the layer 1 scale height above the current layer
            below_range = np.abs(i - (np.abs(z - (z[i] - 2*scale_h[i]))).argmin())
            start_index = np.maximum(nstr[3], i - above_range)
            end_index = np.minimum(nstr[4], i + below_range)
            # start_index = np.maximum(nstr[3], i - pm_range)
            # end_index = np.minimum(nstr[4], i + pm_range)
            kz_lower.append(np.mean(kz[start_index:end_index]))
        kz_lower = np.array(kz_lower)
        kz[nstr[3]:nstr[4]] = kz_lower
    else:
        kz[nstr[0]:nstr[1]] = kz_upper
    
    return kz

def update_clouds(bundle, opacityclass, CloudParameters, Atmosphere, kzz,virga_kwargs,hole_kwargs,verbose=False):
    """
    Updates cloud parameters and returns the cloud output.

    Parameters
    ----------
    bundle : object
        The bundle object containing the inputs and other parameters.
    CloudParameters : namedtuple
        Tuple containing the cloud parameters, including the cloudy flag, fsed, mh, b, param, directory, and condensates.
    Atmosphere : namedtuple
        Contains temperature, pressure, dtdp, mmw, and scale height.
    kzz : array
        Array of Kz cm^2/s.
    virga_kwargs : dict
        Dictionary of keyword arguments for the virga function.
    verbose : bool, optional
        If True, prints additional information. Default is False.
    save_profile : bool, optional
        If True, saves the profile. Default is True.
    all_opd : list, optional
        List to store all optical depth profiles. Default is an empty list.

    Returns
    -------
    cld_out : dict
        Dictionary containing cloud output parameters.
    df_cld : DataFrame
        DataFrame containing cloud properties in PICASO format.
    taudif : float
        Maximum difference in optical depth between iterations.
    taudif_tol : float
        Tolerance for the maximum optical depth difference.
    CloudParameters : namedtuple
        Updated CloudParameters namedtuple.
    ```
    """
    cloudy = CloudParameters.cloudy
    if cloudy == "selfconsistent":
        opd_cld_climate, g0_cld_climate, w0_cld_climate = CloudParameters.OPD, CloudParameters.G0, CloudParameters.W0
        we0, we1, we2, we3 = 0.25, 0.25, 0.25, 0.25

        opd_prev_cld_step = (we0 * opd_cld_climate[:, :, 0] + we1 * opd_cld_climate[:, :, 1] + we2 * opd_cld_climate[:, :, 2] + we3 * opd_cld_climate[:, :, 3])

        virga_kwargs['mmw'] = np.mean(Atmosphere.mmw_layer)

        bundle.inputs['atmosphere']['profile']['kz'] = kzz

        #if not average_only: 
        cld_out = bundle.virga(**virga_kwargs)

        opd_now, w0_now, g0_now = cld_out['opd_per_layer'], cld_out['single_scattering'], cld_out['asymmetry']

        opd_cld_climate[:, :, 3], g0_cld_climate[:, :, 3], w0_cld_climate[:, :, 3] = opd_cld_climate[:, :, 2], g0_cld_climate[:, :, 2], w0_cld_climate[:, :, 2]
        opd_cld_climate[:, :, 2], g0_cld_climate[:, :, 2], w0_cld_climate[:, :, 2] = opd_cld_climate[:, :, 1], g0_cld_climate[:, :, 1], w0_cld_climate[:, :, 1]
        opd_cld_climate[:, :, 1], g0_cld_climate[:, :, 1], w0_cld_climate[:, :, 1] = opd_cld_climate[:, :, 0], g0_cld_climate[:, :, 0], w0_cld_climate[:, :, 0]

        opd_cld_climate[:, :, 0], g0_cld_climate[:, :, 0], w0_cld_climate[:, :, 0] = opd_now, g0_now, w0_now

        sum_opd_clmt = (we0 * opd_cld_climate[:, :, 0] + we1 * opd_cld_climate[:, :, 1] + we2 * opd_cld_climate[:, :, 2] + we3 * opd_cld_climate[:, :, 3])
        opd_clmt = (we0 * opd_cld_climate[:, :, 0] + we1 * opd_cld_climate[:, :, 1] + we2 * opd_cld_climate[:, :, 2] + we3 * opd_cld_climate[:, :, 3])
        g0_clmt = (we0 * opd_cld_climate[:, :, 0] * g0_cld_climate[:, :, 0] + we1 * opd_cld_climate[:, :, 1] * g0_cld_climate[:, :, 1] + we2 * opd_cld_climate[:, :, 2] * g0_cld_climate[:, :, 2] + we3 * opd_cld_climate[:, :, 3] * g0_cld_climate[:, :, 3]) / (sum_opd_clmt)
        w0_clmt = (we0 * opd_cld_climate[:, :, 0] * w0_cld_climate[:, :, 0] + we1 * opd_cld_climate[:, :, 1] * w0_cld_climate[:, :, 1] + we2 * opd_cld_climate[:, :, 2] * w0_cld_climate[:, :, 2] + we3 * opd_cld_climate[:, :, 3] * w0_cld_climate[:, :, 3]) / (sum_opd_clmt)
        g0_clmt = np.nan_to_num(g0_clmt, nan=0.0)
        w0_clmt = np.nan_to_num(w0_clmt, nan=0.0)
        opd_clmt[np.where(opd_clmt <= 1e-5)] = 0.0

        df_cld = vj.picaso_format(opd_clmt, w0_clmt, g0_clmt, pressure=cld_out['pressure'], wavenumber=1e4 / cld_out['wave'])
        #bundle.clouds(df=df_cld)

        diff = (opd_clmt - opd_prev_cld_step)
        taudif = np.max(np.abs(diff))
        taudif_tol = 0.4 * np.max(0.5 * (opd_clmt + opd_prev_cld_step))
        if verbose:
            print("Doing clouds: Max TAUCLD diff is", taudif, " Tau tolerance is ", taudif_tol)
        CloudParameters = CloudParameters._replace(OPD=opd_cld_climate,G0=g0_cld_climate,W0=w0_cld_climate)
        bundle.clouds(df=df_cld,**hole_kwargs)
    elif cloudy == "fixed":
        level_pressure, wno = bundle.inputs['atmosphere']['profile']['pressure'], opacityclass.wno
        opd_clmt = opd_cld_climate[:,:,0]
        w0_clmt = w0_cld_climate[:,:,0]
        g0_clmt = g0_cld_climate[:,:,0]
        layer_pressure = np.sqrt(level_pressure.values[:-1] * level_pressure.values[1:])
        df_cld = vj.picaso_format(opd_clmt, w0_clmt, g0_clmt, pressure=layer_pressure, wavenumber=wno)
        bundle.clouds(df=df_cld)
        cld_out = "fixed"
    elif cloudy == "cloudless":
        cld_out, df_cld, taudif, taudif_tol = 0, 0, 1, 0.1
    else:
        raise NotImplementedError(f"The only supported cloud modes are 'cloudless', 'fixed', 'selfconsistent'; got {cloudy}")
        
    return cld_out, df_cld, taudif, taudif_tol, CloudParameters
