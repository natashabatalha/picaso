import numpy as np 

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
