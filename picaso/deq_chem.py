import numpy as np
from numba import jit

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.legend as lg
import scipy
import scipy.optimize as sop
from scipy.interpolate import interp1d
import time, timeit, os, sys
import ast




#@jit(nopython=True, cache=True)
def quench_level(pressure, temp, kz,mmw, grav, return_mix_timescale = False):
    """
    Quench Level Calculation from T. Karilidi (Quench_Level routine)
    Parameters
    ----------
    pressure : array
        Level Pressures [bar] 
    temp : array 
        Level Temperature [bar]
    kz : array 
        array of Kz cm^2/s
    grav : float
        gravity cgs
    
    Returns
    -------
    array 
        quench levels of gases
    
    """

    k_b = 1.38e-23 # boltzmann constant
    m_p = 1.66e-27 # proton mass
    nlevel = len(temp)
    # print('quench mmw before nlevel:',mmw)
    if len(mmw) < nlevel:
        # print('quench mmw appended')
        mmw = np.append(mmw,mmw[-1])
    quench_levels = np.zeros(shape=(4))
    quench_levels  = quench_levels.astype(int)

    con  = k_b/(mmw*m_p)
    # print('quench mmw:',mmw) #JM deq nan issue
    scale_H = con * temp*1e2/(grav) #cgs
    # print('quench temp input:',temp)
    # print('quench con:',con)
    # print('quench grav:',grav)
    # print('Scale Height:',scale_H)
    t_mix = scale_H**2/kz ## level mixing timescales
    # this is the CO- CH4 - H2O quench level 
    t_chem_co = (3.0e-6/pressure)*np.exp(42000/temp) ## level chemical timescale (Zahnle and Marley 2014)
    if np.max(t_mix) < np.min(t_chem_co):
        raise Exception("CO/H2O/CH4 mixing across Pressure Ranges, Start with deeper Pressure Grid")
    for j in range(nlevel-1,0,-1):

        if ((t_mix[j-1]/1e15) <=  (t_chem_co[j-1]/1e15)) and ((t_mix[j]/1e15) >=  (t_chem_co[j]/1e15)):
            quench_levels[0] = j
            break
    
    


    # now calculate CO2 quench level
    

    t_chem_co2 = (1e-10/(pressure**0.5))*np.exp(38000./temp) #(Zahnle and Marley 2014)  
    if np.max(t_mix) < np.min(t_chem_co2):
        raise Exception("CO2 mixing across Pressure Ranges, Start with deeper Pressure Grid")

    for j in range(nlevel-1,0,-1):

        if ((t_mix[j-1]/1e15) <=  (t_chem_co2[j-1]/1e15)) and ((t_mix[j]/1e15) >=  (t_chem_co2[j]/1e15)):
            quench_levels[1] = j
            break
    
    # now calculate the NH3/N2 quench level

    t_chem_nh3 = (1e-7/pressure)*np.exp(52000/temp) #(Zahnle and Marley 2014)  
    if np.max(t_mix) < np.min(t_chem_nh3):
        raise Exception("NH3 mixing across Pressure Ranges, Start with deeper Pressure Grid")

    for j in range(nlevel-1,0,-1):

        if ((t_mix[j-1]/1e15) <=  (t_chem_nh3[j-1]/1e15)) and ((t_mix[j]/1e15) >=  (t_chem_nh3[j]/1e15)):
            quench_levels[2] = j
            break

    # now calculate the HCN quench level

    t_chem_hcn = (1.5e-4/(pressure*(3.**0.7)))*np.exp(36000./temp) #(Zahnle and Marley 2014)    

    if np.max(t_mix) < np.min(t_chem_hcn):
        raise Exception("HCN mixing across Pressure Ranges, Start with deeper Pressure Grid")


    for j in range(nlevel-1,0,-1):

        if ((t_mix[j-1]/1e15) <=  (t_chem_hcn[j-1]/1e15)) and ((t_mix[j]/1e15) >=  (t_chem_hcn[j]/1e15)):
            quench_levels[3] = j
            break
    
    
    
    

    if return_mix_timescale == False :
        return quench_levels
    elif return_mix_timescale == True :
        return quench_levels, t_mix


    
@jit(nopython=True, cache=True)
def mix_all_gases(kappa1,kappa2,kappa3,kappa4,kappa5,mix1,mix2,mix3,mix4,mix5,gauss_pts, gauss_wts,indices):
    """
    Function to perform "on-the-fly" mixing of 5 opacity sources from Amundsen et al. (2017)
    Parameters
    ----------
    kappa1 : array
        K-coefficients of gas mixture 1
    kappa2 : array 
        K-coefficients of gas mixture 2
    kappa3 : array 
        K-coefficients of gas mixture 3
    kappa4 : array
        K-coefficients of gas mixture 4
    kappa5 : array
        K-coefficients of gas mixture 5
    mix1 : array
        mixing ratios of gas mixture 1
    mix2 : array 
        mixing ratios of gas mixture 2
    mix3 : array 
        mixing ratios of gas mixture 3
    mix4 : array
        mixing ratios of gas mixture 4
    mix5 : array
        mixing ratios of gas mixture 5
    gauss_pts : array
        Gauss points of the K-coefficients
    gauss_wts : array
        Gauss weights of the K-coefficients
    indices : array
        Nearest neighbor indices of the T(P) profile in the grid currently
    
    Returns
    -------
    array 
        Mixed K-coefficients
    
    """
    
    Nk=len(gauss_wts) # number of gauss points
    Nlayer = len(indices[0]) # number of atmosphere layers
    
    
    kappa_mixed = np.zeros(shape=(Nlayer,kappa1.shape[2],Nk,4))  # array to be returned
    # shape of kappa_mixed is Nlayer*nwno*ngauss*number of nearest neighbors (4)
    for ilayer in range(Nlayer):
        ct =0
        for p_ind in [indices[0][ilayer],indices[1][ilayer]]:
            for t_ind in [indices[2][ilayer],indices[3][ilayer]]: 
                for iw in range(kappa1.shape[2]): # mixing needs to be done at each wno bin separately.

                    kmix_bin = do_mixing_mono(kappa1[p_ind,t_ind,iw,:],kappa2[p_ind,t_ind,iw,:],kappa3[p_ind,t_ind,iw,:],kappa4[p_ind,t_ind,iw,:],kappa5[p_ind,t_ind,iw,:],
                                    mix1[ilayer],mix2[ilayer],mix3[ilayer],mix4[ilayer],mix5[ilayer],gauss_pts,gauss_wts)

                    kappa_mixed[ilayer,iw,:,ct] = kmix_bin

                ct+=1   
    # k coefficients were raised to exponentials in do_mixing_mono routine so taking a log to take them back
    return np.log(kappa_mixed) # this array will be interpolated now


@jit(nopython=True, cache=True)
def mix_all_gases_gasesfly(kappa1,kappa2,kappa3,kappa4,kappa5,kappa6,kappa7,kappa8,kappa9,kappa10,kappa11,kappa12,kappa13,kappa14,kappa15,kappa16,kappa17,mix1,mix2,mix3,mix4,mix5,mix6,mix7,mix8,mix9,mix10,mix11,mix12,mix13,mix14,mix15,mix16,mix17,gauss_pts, gauss_wts,indices):
    """
    Function to perform "on-the-fly" mixing of 5 opacity sources from Amundsen et al. (2017)
    Parameters
    ----------
    kappa1 : array
        K-coefficients of gas mixture 1
    kappa2 : array 
        K-coefficients of gas mixture 2
    kappa3 : array 
        K-coefficients of gas mixture 3
    kappa4 : array
        K-coefficients of gas mixture 4
    kappa5 : array
        K-coefficients of gas mixture 5
    mix1 : array
        mixing ratios of gas mixture 1
    mix2 : array 
        mixing ratios of gas mixture 2
    mix3 : array 
        mixing ratios of gas mixture 3
    mix4 : array
        mixing ratios of gas mixture 4
    mix5 : array
        mixing ratios of gas mixture 5
    gauss_pts : array
        Gauss points of the K-coefficients
    gauss_wts : array
        Gauss weights of the K-coefficients
    indices : array
        Nearest neighbor indices of the T(P) profile in the grid currently
    
    Returns
    -------
    array 
        Mixed K-coefficients
    
    """
    
    Nk=len(gauss_wts) # number of gauss points
    Nlayer = len(indices[0]) # number of atmosphere layers
    
    
    kappa_mixed = np.zeros(shape=(Nlayer,kappa1.shape[2],Nk,4))  # array to be returned
    # shape of kappa_mixed is Nlayer*nwno*ngauss*number of nearest neighbors (4)
    for ilayer in range(Nlayer):
        ct =0
        for p_ind in [indices[0][ilayer],indices[1][ilayer]]:
            for t_ind in [indices[2][ilayer],indices[3][ilayer]]: 
                for iw in range(kappa1.shape[2]): # mixing needs to be done at each wno bin separately.

                    kmix_bin = do_mixing_mono_gasesfly(kappa1[p_ind,t_ind,iw,:],kappa2[p_ind,t_ind,iw,:],kappa3[p_ind,t_ind,iw,:],kappa4[p_ind,t_ind,iw,:],kappa5[p_ind,t_ind,iw,:],
                                                        kappa6[p_ind,t_ind,iw,:],kappa7[p_ind,t_ind,iw,:],kappa8[p_ind,t_ind,iw,:],kappa9[p_ind,t_ind,iw,:],kappa10[p_ind,t_ind,iw,:],kappa11[p_ind,t_ind,iw,:],kappa12[p_ind,t_ind,iw,:],kappa13[p_ind,t_ind,iw,:],kappa14[p_ind,t_ind,iw,:],kappa15[p_ind,t_ind,iw,:],kappa16[p_ind,t_ind,iw,:],kappa17[p_ind,t_ind,iw,:],
                                    mix1[ilayer],mix2[ilayer],mix3[ilayer],mix4[ilayer],mix5[ilayer],mix6[ilayer],mix7[ilayer],mix8[ilayer],mix9[ilayer],mix10[ilayer],mix11[ilayer],mix12[ilayer],mix13[ilayer],mix14[ilayer],mix15[ilayer],mix16[ilayer],mix17[ilayer],gauss_pts,gauss_wts)

                    kappa_mixed[ilayer,iw,:,ct] = kmix_bin

                ct+=1   
    # k coefficients were raised to exponentials in do_mixing_mono routine so taking a log to take them back
    return np.log(kappa_mixed) # this array will be interpolated now


@jit(nopython=True, cache=True)
def do_mixing_mono_gasesfly(kappa1_mono,kappa2_mono,kappa3_mono,kappa4_mono,kappa5_mono,
                          kappa6_mono,kappa7_mono,kappa8_mono,kappa9_mono,kappa10_mono,kappa11_mono,kappa12_mono,kappa13_mono,kappa14_mono,kappa15_mono,kappa16_mono,kappa17_mono,
                          mix1,mix2,mix3,mix4,mix5,mix6,mix7,mix8,mix9,mix10,mix11,mix12,mix13,mix14,mix15,mix16,mix17,gauss_pts,gauss_wts):
    """
    Function which mixes all the gases together at a single wavenumber bin
    Parameters
    ----------
    kappa1_mono : array
        K-coefficients of gas mixture 1
    kappa2_mono : array 
        K-coefficients of gas mixture 2
    kappa3_mono : array 
        K-coefficients of gas mixture 3
    kappa4_mono : array
        K-coefficients of gas mixture 4
    kappa5_mono : array
        K-coefficients of gas mixture 5
    mix1 : array
        mixing ratios of gas mixture 1
    mix2 : array 
        mixing ratios of gas mixture 2
    mix3 : array 
        mixing ratios of gas mixture 3
    mix4 : array
        mixing ratios of gas mixture 4
    mix5 : array
        mixing ratios of gas mixture 5
    gauss_pts : array
        Gauss points of the K-coefficients
    gauss_wts : array
        Gauss weights of the K-coefficients
    
    Returns
    -------
    array 
        Mixed K-coefficients at a single wavelength
    
    """
    
    kmix_bin,mix_t =mix_2_gases(np.exp(kappa1_mono),np.exp(kappa2_mono), mix1,mix2,gauss_pts,gauss_wts) # mix 2 gases
    
    kmix_bin,mix_t = mix_2_gases(kmix_bin,np.exp(kappa3_mono), mix_t,mix3,gauss_pts,gauss_wts) # mix 3rd with mixture from previous
    
    kmix_bin,mix_t = mix_2_gases(kmix_bin,np.exp(kappa4_mono), mix_t,mix4,gauss_pts,gauss_wts) # mix 4th with mixture from previous
    
    kmix_bin,mix_t = mix_2_gases(kmix_bin,np.exp(kappa5_mono), mix_t,mix5,gauss_pts,gauss_wts) # and so on
       
    kmix_bin,mix_t = mix_2_gases(kmix_bin,np.exp(kappa6_mono), mix_t,mix6,gauss_pts,gauss_wts) 
    
    kmix_bin,mix_t = mix_2_gases(kmix_bin,np.exp(kappa7_mono), mix_t,mix7,gauss_pts,gauss_wts) 
    
    kmix_bin,mix_t = mix_2_gases(kmix_bin,np.exp(kappa8_mono), mix_t,mix8,gauss_pts,gauss_wts) 
    
    kmix_bin,mix_t = mix_2_gases(kmix_bin,np.exp(kappa9_mono), mix_t,mix9,gauss_pts,gauss_wts) 
    
    kmix_bin,mix_t = mix_2_gases(kmix_bin,np.exp(kappa10_mono), mix_t,mix10,gauss_pts,gauss_wts) 
   
    kmix_bin,mix_t = mix_2_gases(kmix_bin,np.exp(kappa11_mono), mix_t,mix11,gauss_pts,gauss_wts)
    
    kmix_bin,mix_t = mix_2_gases(kmix_bin,np.exp(kappa12_mono), mix_t,mix12,gauss_pts,gauss_wts) 

    kmix_bin,mix_t = mix_2_gases(kmix_bin,np.exp(kappa13_mono), mix_t,mix13,gauss_pts,gauss_wts) 

    kmix_bin,mix_t = mix_2_gases(kmix_bin,np.exp(kappa14_mono), mix_t,mix14,gauss_pts,gauss_wts) 

    kmix_bin,mix_t = mix_2_gases(kmix_bin,np.exp(kappa15_mono), mix_t,mix15,gauss_pts,gauss_wts) 

    kmix_bin,mix_t = mix_2_gases(kmix_bin,np.exp(kappa16_mono), mix_t,mix16,gauss_pts,gauss_wts) 
    
    kmix_bin,mix_t = mix_2_gases(kmix_bin,np.exp(kappa17_mono), mix_t,mix17,gauss_pts,gauss_wts) 
    
    '''
    kmix_bin,mix_t = mix_2_gases(kmix_bin,np.exp(kappa6_mono), mix_t,mix6,gauss_pts,gauss_wts)
    
    kmix_bin,mix_t = mix_2_gases(kmix_bin,np.exp(kappa7_mono), mix_t,mix6,gauss_pts,gauss_wts)
    
    kmix_bin,mix_t = mix_2_gases(kmix_bin,np.exp(kappa8_mono), mix_t,mix6,gauss_pts,gauss_wts)
    '''
    
    
    return kmix_bin

@jit(nopython=True, cache=True)
def do_mixing_mono(kappa1_mono,kappa2_mono,kappa3_mono,kappa4_mono,kappa5_mono,
                          mix1,mix2,mix3,mix4,mix5,gauss_pts,gauss_wts):
    """
    Function which mixes all the gases together at a single wavenumber bin
    Parameters
    ----------
    kappa1_mono : array
        K-coefficients of gas mixture 1
    kappa2_mono : array 
        K-coefficients of gas mixture 2
    kappa3_mono : array 
        K-coefficients of gas mixture 3
    kappa4_mono : array
        K-coefficients of gas mixture 4
    kappa5_mono : array
        K-coefficients of gas mixture 5
    mix1 : array
        mixing ratios of gas mixture 1
    mix2 : array 
        mixing ratios of gas mixture 2
    mix3 : array 
        mixing ratios of gas mixture 3
    mix4 : array
        mixing ratios of gas mixture 4
    mix5 : array
        mixing ratios of gas mixture 5
    gauss_pts : array
        Gauss points of the K-coefficients
    gauss_wts : array
        Gauss weights of the K-coefficients
    
    Returns
    -------
    array 
        Mixed K-coefficients at a single wavelength
    
    """
    
    kmix_bin,mix_t =mix_2_gases(np.exp(kappa1_mono),np.exp(kappa2_mono), mix1,mix2,gauss_pts,gauss_wts) # mix 2 gases
    
    kmix_bin,mix_t = mix_2_gases(kmix_bin,np.exp(kappa3_mono), mix_t,mix3,gauss_pts,gauss_wts) # mix 3rd with mixture from previous
    
    kmix_bin,mix_t = mix_2_gases(kmix_bin,np.exp(kappa4_mono), mix_t,mix4,gauss_pts,gauss_wts) # mix 4th with mixture from previous
    
    kmix_bin,mix_t = mix_2_gases(kmix_bin,np.exp(kappa5_mono), mix_t,mix5,gauss_pts,gauss_wts) # and so on
    '''
    kmix_bin,mix_t = mix_2_gases(kmix_bin,np.exp(kappa6_mono), mix_t,mix6,gauss_pts,gauss_wts)
    
    kmix_bin,mix_t = mix_2_gases(kmix_bin,np.exp(kappa7_mono), mix_t,mix6,gauss_pts,gauss_wts)
    
    kmix_bin,mix_t = mix_2_gases(kmix_bin,np.exp(kappa8_mono), mix_t,mix6,gauss_pts,gauss_wts)
    '''
    
    
    return kmix_bin

@jit(nopython=True, cache=True)
def mix_2_gases(k1,k2,mix1,mix2,gauss_pts,gauss_wts):
    """
    Function actually performing the mixing of K-coefficients between two gases.

    This module is adapted from the CHIMERA code (https://github.com/ExoCTK/chimera)
    written by Michael Line (Michael.Line@asu.edu).

    Parameters
    ----------
    k1 : array
        K-coefficients of gas mixture 1
    k2 : array 
        K-coefficients of gas mixture 2
    mix1 : array
        mixing ratios of gas mixture 1
    mix2 : array 
        mixing ratios of gas mixture 2
    gauss_pts : array
        Gauss points of the K-coefficients
    gauss_wts : array
        Gauss weights of the K-coefficients
    
    Returns
    -------
    array 
        Mixed K-coefficients of 2 gases at a single wavelength 
    
    """
    mix_t=mix1+mix2  #"new" VMR is sum of individual VMR's
    Nk = len(gauss_wts)           
    kmix=np.zeros(Nk**2)   #Nk^2 mixed k-coeff array
    wtsmix=np.zeros(Nk**2) #Nk^2 mixed weights array
    #mixing two gases weighting by their relative VMR
    for i in range(Nk):
        for j in range(Nk):
            kmix[i*Nk+j]=(mix1*k1[i]+mix2*k2[j])/mix_t #equation 9 Amundsen 2017 (equation 20 Mollier 2015)
            wtsmix[i*Nk+j]=gauss_wts[i]*gauss_wts[j]    #equation 10 Amundsen 2017

    #resort-rebin procedure--see Amundsen et al. 2016 or section B.2.1 in Molliere et al. 2015
    sort_indicies=np.argsort(kmix)  #sort new "mixed" k-coeff's from low to high--these are indicies
    kmix_sort=kmix[sort_indicies]  #sort k-coeffs from low to high
    wtsmix_sort=wtsmix[sort_indicies]  #sort mixed weights using same indicie mapping from sorted mixed k-coeffs
    #combining w/weights--see description on Molliere et al. 2015--not sure why this works..similar to Amundson 2016 weighted avg?
    int=np.cumsum(wtsmix_sort)
    x=int/np.max(int)#*2.-1  # Here had to remove *2-1 to match results with eq calculations. Same as Fortran.
    #logkmix=np.log10(kmix_sort)
    logkmix = kmix_sort
    #kmix_bin=10**np.interp(gauss_pts,x,logkmix)  #interpolating via cumulative sum of sorted weights...
    kmix_bin=np.zeros(Nk)
    for i in range(Nk):
        loc=np.where(x > gauss_pts[i])[0][0]
        
        if loc >= 0:
            if logkmix[loc-1] == logkmix[loc]:
                kmix_bin[i] = logkmix[loc]
            else :
                interp = np.log(logkmix[loc-1]) + np.log(logkmix[loc]/logkmix[loc-1])*((gauss_pts[i]-x[loc-1])/(x[loc]-x[loc-1]))
                kmix_bin[i]=np.exp(interp)
    
    return kmix_bin, mix_t

def initiate_cld_matrices(opd_cld_climate,g0_cld_climate,w0_cld_climate,wv196,wv661):
    opd_cld_climate_new =  np.zeros(shape=(len(opd_cld_climate[:,0,0]),len(wv661),4))
    g0_cld_climate_new,w0_cld_climate_new = np.zeros_like(opd_cld_climate_new),np.zeros_like(opd_cld_climate_new)
    for j in range(4):

        for ilayer in range(len(opd_cld_climate[:,0,j])):

            fopd = interp1d(wv196,opd_cld_climate[ilayer,:,j] , kind='cubic',fill_value="extrapolate")
            fg0 = interp1d(wv196,g0_cld_climate[ilayer,:,j] , kind='cubic',fill_value="extrapolate")
            fw0 = interp1d(wv196,w0_cld_climate[ilayer,:,j] , kind='cubic',fill_value="extrapolate")

            opd_cld_climate_new[ilayer,:,j] = fopd(wv661)
            g0_cld_climate_new[ilayer,:,j] = fg0(wv661)
            w0_cld_climate_new[ilayer,:,j] = fw0(wv661)





    return opd_cld_climate_new,g0_cld_climate_new,w0_cld_climate_new
'''
def run_vulcan(pressure,temp,kz,grav):
    


    np.savetxt("vulcan/atm/tpkzz.txt",np.transpose([pressure*1e6,temp,kz]))
    
    vulcan_cfg.atm_file  = 'vulcan/atm/tpkzz.txt'
    vulcan_cfg.use_solar = True # use solar abuncances
    vulcan_cfg.ini_mix = 'EQ' # start from equilibrium
    vulcan_cfg.use_photo = False # no photochem
    vulcan_cfg.nz = len(pressure) # # of levels
    vulcan_cfg.P_b = np.max(pressure*1e6) #dyne/cm^2
    vulcan_cfg.P_t = np.min(pressure*1e6) #dyne/cm^2
    vulcan_cfg.gs  = grav*100 #cgs
    vulcan_cfg.use_live_plot = True

    abspath = os.path.abspath(sys.argv[0])
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    np.set_printoptions(threshold=np.inf)
    species = chem_funs.spec_list

    with open(vulcan_cfg.com_file, 'r') as f:
        columns = f.readline() # reading in the first line
        num_ele = len(columns.split())-2 # number of elements (-2 for removing "species" and "mass") 
    
    type_list = ['int' for i in range(num_ele)]
    type_list.insert(0,'U20'); type_list.append('float')
    compo = np.genfromtxt(vulcan_cfg.com_file,names=True,dtype=type_list)
    # dtype=None in python 2.X but Sx -> Ux in python3
    compo_row = list(compo['species'])
    ### read in the basic chemistry data

    ### creat the instances for storing the variables and parameters
    data_var = store.Variables()
    data_atm = store.AtmData()
    data_para = store.Parameters()

    # record starting CPU time
    data_para.start_time = time.time()

    make_atm = build_atm.Atm()

    # for plotting and printing
    output = op.Output()

    # saving the config file
    output.save_cfg(dname)

    # construct pico
    data_atm = make_atm.f_pico(data_atm)
    # construct Tco and Kzz 
    data_atm =  make_atm.load_TPK(data_atm)
    # construct Dzz (molecular diffusion)

    # Only setting up ms (the species molecular weight) if vulcan_cfg.use_moldiff == False
    make_atm.mol_diff(data_atm)

    # calculating the saturation pressure
    if vulcan_cfg.use_condense == True: make_atm.sp_sat(data_atm)

    # for reading rates
    rate = op.ReadRate()

    # read-in network and calculating forward rates
    data_var = rate.read_rate(data_var, data_atm)

    # for low-T rates e.g. Jupiter       
    if vulcan_cfg.use_lowT_limit_rates == True: data_var = rate.lim_lowT_rates(data_var, data_atm)
        
    # reversing rates
    data_var = rate.rev_rate(data_var, data_atm)
    # removing rates
    data_var = rate.remove_rate(data_var)

    ini_abun = build_atm.InitialAbun()
    # initialing y and ymix (the number density and the mixing ratio of every species)
    data_var = ini_abun.ini_y(data_var, data_atm)

    # storing the initial total number of atmos
    data_var = ini_abun.ele_sum(data_var)

    # calculating mean molecular weight, dz, and dzi and plotting TP
    data_atm = make_atm.f_mu_dz(data_var, data_atm, output)

    # specify the BC
    make_atm.BC_flux(data_atm)


    # ============== Execute VULCAN  ==============
    # time-steping in the while loop until conv() returns True or count > count_max 

    # setting the numerical solver to the desinated one in vulcan_cfg
    solver_str = vulcan_cfg.ode_solver
    solver = getattr(op, solver_str)()

    # Setting up for photo chemistry
    if vulcan_cfg.use_photo == True:
        rate.make_bins_read_cross(data_var, data_atm)
        #rate.read_cross(data_var)
        make_atm.read_sflux(data_var, data_atm)
        
        # computing the optical depth (tau), flux, and the photolisys rates (J) for the first time 
        solver.compute_tau(data_var, data_atm)
        solver.compute_flux(data_var, data_atm)
        solver.compute_J(data_var, data_atm)
        # they will be updated in op.Integration by the assigned frequence
        
        # removing rates
        data_var = rate.remove_rate(data_var)

    integ = op.Integration(solver, output)
    # Assgining the specific solver corresponding to different B.C.s
    solver.naming_solver(data_para)
    
    # Running the integration loop
    integ(data_var, data_atm, data_para, make_atm)
    
    output.save_out(data_var, data_atm, data_para, dname)
'''


    







    
    
