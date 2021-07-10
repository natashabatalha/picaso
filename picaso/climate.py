import numpy as np 
from numba import jit, vectorize
from numpy import exp, zeros, where, sqrt, cumsum , pi, outer, sinh, cosh, min, dot, array,log,log10


# compute initial net fluxes 
tidal = 

flux_net_i, flux_mid_net_i = get_thermal_1d()
flux_net_v, flux_mid_net_v = get_reflected_1d()

flux_net = flux_net_i*rfaci + flux_net_v*rfacv + tidal 
flux_net_mid = flux_mid_net_i*rfaci + flux_mid_net_v*rfacv + tidal 

#store value of the flux calculated before the perturbation
old_flux_net = flux_net # FNETIP
old_flux_net_mid = flux_net_mid # FMIP
old_temp = temp #beta

#start cz at the very bottom and have it grow upward 

itop_cz = nlayer # NSTRTA 
ibot_cz = nlayer+1 # NBOTA
cz_or_rad = np.zeros(nlevel)#zero=radiative, 1=convective
cz_or_rad[-1] = 1 #flip botton layer to convective

#need to get initial maximum T step size 
max_temp_step = np.sqrt(np.sum([temp[i]**2 for i in range(nlevel) if cz_or_rad[i] == 0]))

@jit(nopython=True, cache=True)
def did_grad( t, p, t_table, p_table, grad):
    """
    Parameters
    ----------
    t : float
        Temperature  value
    p : float 
        Pressure value
    t_table : array 
        array of Pressure values with 53 entries
    p_table : array 
        array of Pressure value with 26 entries
    grad : array 
        array of gradients of dimension 53*26
    
    Returns
    -------
    float 
        grad_x 
    
    """
    # Python version of DIDGRAD function in convec.f in EGP
    # This has been benchmarked with the fortran version
    
       
    temp_log= log10(t)
    pres_log= log10(p)
    
    pos_t = locate(t_table, temp_log)
    pos_p = locate(p_table, pres_log)

    ipflag=0
    if pos_p ==0: ## lowest pressure point
        factkp= 0.0
        ipflag=1
    elif pos_p ==25 : ## highest pressure point
        factkp= 1.0
        pos_p=24  ## use highest point
        ipflag=1

    itflag=0
    if pos_t ==0: ## lowest pressure point
        factkt= 0.0
        itflag=1
    elif pos_t == 52 : ## highest temp point
        factkt= 1.0
        pos_t=51 ## use highest point
        itflag=1
    
    if (pos_p > 0) and (pos_p < 26) and (ipflag == 0):
        factkp= (-p_table[pos_p]+pres_log)/(p_table[pos_p+1]-p_table[pos_p])
    
    if (pos_t > 0) and (pos_t < 53) and (itflag == 0):
        factkt= (-t_table[pos_t]+temp_log)/(t_table[pos_t+1]-t_table[pos_t])

    
    gp1 = grad[pos_t,pos_p]
    gp2 = grad[pos_t+1,pos_p]
    gp3 = grad[pos_t+1,pos_p+1]
    gp4 = grad[pos_t,pos_p+1]
    

    grad_x = (1.0-factkt)*(1.0-factkp)*gp1 + factkt*(1.0-factkp)*gp2 + factkt*factkp*gp3 + (1.0-factkt)*factkp*gp4

    return grad_x


@jit(nopython=True, cache=True)
def locate(array,value):
    """
    Parameters
    ----------
    array : array
        Array to be searched.
    value : float 
        Value to be searched for.
    
    
    Returns
    -------
    int 
        location of nearest point by bisection method 
    
    """
    
    
    n = len(array)
    
    
    jl = 0
    ju = n
    while (ju-jl > 1):
        jm=int(0.5*(ju+jl)) 
        if (value >= array[jm]):
            jl=jm
        else:
            ju=jm
    
    if (value <= array[0]): # if value lower than first point
        jl=0
    elif (value >= array[-1]): # if value higher than first point
        jl= n-1
    
    return jl















