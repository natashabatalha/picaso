import numpy as np
from virga import justdoit as vj

from .climate import calculate_atm
from .fluxes import get_kzz

def run_clouds_for_climate(cld_species, cloudy, fsed, beta, param_flag, bundle, opacityclass, directory, opd_cld_climate, g0_cld_climate, w0_cld_climate, mh, pressure, temp, grav, mmw, tidal, flux_net_ir_layer, flux_plus_ir_attop,t_table, p_table, grad, cp, calc_type,nstr, output_abunds, moist, verbose=False):
    """
    Handles the cloud modeling choice appropriately based on the requested cloud-climate coupling: "cloudless", "fixed", "selfconsistent".
    """
    cld_out = None
    df_cld = None
    diff = 0
    taudif = 0
    taudif_tol = 1
    if cloudy == "selfconsistent":
        """DTAU, TAU, W0, COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, \
        W0_no_raman , surf_reflect, ubar0,ubar1,cos_theta, single_phase,multi_phase, \
        frac_a,frac_b,frac_c,constant_back,constant_forward,  \
        wno,nwno,ng,nt, nlevel, ngauss, gauss_wts, mmw,gweight,tweight =  calculate_atm(bundle, opacityclass )"""

        we0,we1,we2,we3 = 0.25,0.25,0.25,0.25
        opd_prev_cld_step = (we0*opd_cld_climate[:,:,0]+we1*opd_cld_climate[:,:,1]+we2*opd_cld_climate[:,:,2]+we3*opd_cld_climate[:,:,3]) # last average
        
        metallicity = 10**(mh) #atmospheric metallicity relative to Solar
        mean_molecular_weight = np.mean(mmw) # atmospheric mean molecular weight
        
        #get the abundances
        output_abunds = bundle.inputs['atmosphere']['profile'].T.values
            
        kzz = get_kzz(pressure, temp,grav,mmw,tidal,flux_net_ir_layer, flux_plus_ir_attop,t_table, p_table, grad, cp, calc_type,nstr, output_abunds, moist = moist)
        bundle.inputs['atmosphere']['profile']['kz'] = kzz
    

        cld_out = bundle.virga(cld_species,directory, fsed=fsed,mh=metallicity,
                    mmw = mean_molecular_weight, b = beta, param = param_flag, verbose=verbose) #,climate=True)
        
        opd_now, w0_now, g0_now = cld_out['opd_per_layer'],cld_out['single_scattering'],cld_out['asymmetry']
        
        opd_cld_climate[:,:,3], g0_cld_climate[:,:,3], w0_cld_climate[:,:,3] = opd_cld_climate[:,:,2], g0_cld_climate[:,:,2], w0_cld_climate[:,:,2]
        opd_cld_climate[:,:,2], g0_cld_climate[:,:,2], w0_cld_climate[:,:,2] = opd_cld_climate[:,:,1], g0_cld_climate[:,:,1], w0_cld_climate[:,:,1]
        opd_cld_climate[:,:,1], g0_cld_climate[:,:,1], w0_cld_climate[:,:,1] = opd_cld_climate[:,:,0], g0_cld_climate[:,:,0], w0_cld_climate[:,:,0]
                    
        opd_cld_climate[:,:,0], g0_cld_climate[:,:,0], w0_cld_climate[:,:,0] = opd_now, g0_now, w0_now
        
        sum_opd_clmt = (we0*opd_cld_climate[:,:,0]+we1*opd_cld_climate[:,:,1]+we2*opd_cld_climate[:,:,2]+we3*opd_cld_climate[:,:,3])
        opd_clmt = (we0*opd_cld_climate[:,:,0]+we1*opd_cld_climate[:,:,1]+we2*opd_cld_climate[:,:,2]+we3*opd_cld_climate[:,:,3])
        g0_clmt = (we0*opd_cld_climate[:,:,0]*g0_cld_climate[:,:,0]+we1*opd_cld_climate[:,:,1]*g0_cld_climate[:,:,1]+we2*opd_cld_climate[:,:,2]*g0_cld_climate[:,:,2]+we3*opd_cld_climate[:,:,3]*g0_cld_climate[:,:,3])/(sum_opd_clmt)
        w0_clmt = (we0*opd_cld_climate[:,:,0]*w0_cld_climate[:,:,0]+we1*opd_cld_climate[:,:,1]*w0_cld_climate[:,:,1]+we2*opd_cld_climate[:,:,2]*w0_cld_climate[:,:,2]+we3*opd_cld_climate[:,:,3]*w0_cld_climate[:,:,3])/(sum_opd_clmt)
        g0_clmt = np.nan_to_num(g0_clmt,nan=0.0)
        w0_clmt = np.nan_to_num(w0_clmt,nan=0.0)
        opd_clmt[np.where(opd_clmt <= 1e-5)] = 0.0                
        df_cld = vj.picaso_format(opd_clmt, w0_clmt, g0_clmt, pressure = cld_out['pressure'], wavenumber= 1e4/cld_out['wave'])
        bundle.clouds(df=df_cld)
        diff = (opd_clmt-opd_prev_cld_step)
        taudif = np.max(np.abs(diff))
        taudif_tol = 0.4*np.max(0.5*(opd_clmt+opd_prev_cld_step))
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
        cld_out = 0 
        
    return cld_out, df_cld, diff, taudif, taudif_tol