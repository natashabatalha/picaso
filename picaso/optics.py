from .rayleigh import Rayleigh

import pandas as pd
import numpy as np
import json
import os
from numba import jit
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import inferno
import io 
import sqlite3
import math

#@jit(nopython=True)
def compute_opacity(atmosphere, opacityclass, ngauss=1, stream=2, delta_eddington=True,
    test_mode=False,raman=0, plot_opacity=False,full_output=False, return_mode=False):
    """
    Returns total optical depth per slab layer including molecular opacity, continuum opacity. 
    It should automatically select the molecules needed
    
    Parameters
    ----------
    atmosphere : class ATMSETUP
        This inherets the class from atmsetup.py 
    opacityclass : class opacity
        This inherets the class from optics.py. It is done this way so that the opacity db doesnt have 
        to be reloaded in a retrieval 
    ngauss : int 
        Number of gauss angles if using correlated-k. If using monochromatic opacites, 
        ngauss should one. 
    stream : int 
        Number of streams (only affects detal eddington approximation)
    delta_eddington : bool 
        (Optional) Default=True, With Delta-eddington on, it incorporates the forward peak 
        contribution by adjusting optical properties such that the fraction of scattered energy
        in the forward direction is removed from the scattering parameters 
    raman : int 
        (Optional) Default =0 which corresponds to oklopcic+2018 raman scattering cross sections. 
        Other options include 1 for original pollack approximation for a 6000K blackbody. 
        And 2 for nothing.  
    test_mode : bool 
        (Optional) Default = False to run as normal. This will overwrite the opacities and fix the 
        delta tau at 0.5. 
    full_output : bool 
        (Optional) Default = False. If true, This will add taugas, taucld, tauray to the atmosphere class. 
        This is done so that the users can debug, plot, etc. 
    plot_opacity : bool 
        (Optional) Default = False. If true, Will create a pop up plot of the weighted of each absorber 
        at the middle layer
    return_mode : bool 
        (Optional) Default = False, If true, will only return matrices for all the weighted opacity 
        contributions

    Returns
    -------
    DTAU : ndarray 
        This is a matrix with # layer by # wavelength. It is the opacity contained within a layer 
        including the continuum, scattering, cloud (if specified), and molecular opacity
        **If requested, this is corrected for with Delta-Eddington.**
    TAU : ndarray
        This is a matrix with # level by # wavelength. It is the cumsum of opacity contained 
        including the continuum, scattering, cloud (if specified), and molecular opacity
        **If requested, this is corrected for with Delta-Eddington.**
    WBAR : ndarray
        This is the single scattering albedo that includes rayleigh, raman and user input scattering sources. 
        It has dimensions: # layer by # wavelength
        **If requested, this is corrected for with Delta-Eddington.**
    COSB : ndarray
        This is the asymettry factor which accounts for rayleigh and user specified values 
        It has dimensions: # layer by # wavelength
        **If requested, this is corrected for with Delta-Eddington.**
    ftau_cld : ndarray 
        This is the fraction of cloud opacity relative to the total TAUCLD/(TAUCLD + TAURAY)
    ftau_ray : ndarray 
        This is the fraction of rayleigh opacity relative to the total TAURAY/(TAUCLD + TAURAY)
    GCOS2 : ndarray
        This is used for Cahoy+2010 methodology for accounting for rayleigh scattering. It 
        replaces the use of the actual rayleigh phase function by just multiplying ftau_ray by 2
    DTAU : ndarray 
        This is a matrix with # layer by # wavelength. It is the opacity contained within a layer 
        including the continuum, scattering, cloud (if specified), and molecular opacity
        **If requested, this is corrected for with Delta-Eddington.**
    TAU : ndarray
        This is a matrix with # level by # wavelength. It is the cumsum of opacity contained 
        including the continuum, scattering, cloud (if specified), and molecular opacity
        **Original, never corrected for with Delta-Eddington.**
    WBAR : ndarray
        This is the single scattering albedo that includes rayleigh, raman and user input scattering sources. 
        It has dimensions: # layer by # wavelength
        **Original, never corrected for with Delta-Eddington.**
    COSB : ndarray
        This is the asymettry factor which accounts for rayleigh and user specified values 
        It has dimensions: # layer by # wavelength
        **Original, never corrected for with Delta-Eddington.**

    Notes
    -----
    This was baselined against jupiter with the old fortran code. It matches 100% for all cases 
    except for hotter cases where Na & K are present. This differences is not a product of the code 
    but a product of the different opacities (1060 grid versus old 736 grid)

    Todo 
    -----
    Add a better approximation than delta-scale (e.g. M.Marley suggests a paper by Cuzzi that has 
    a better methodology)
    """
    atm = atmosphere
    nlayer = atm.c.nlayer
    nwno = opacityclass.nwno

    if return_mode:
        taus_by_species = {}

    if plot_opacity: 
        plot_layer=int(nlayer/1.5)#np.size(tlayer)-1
        opt_figure = figure(x_axis_label = 'Wavelength', y_axis_label='TAUGAS in optics.py', 
        title = 'Opacity at T='+str(atm.layer['temperature'][plot_layer])+' P='+str(atm.layer['pressure'][plot_layer]/atm.c.pconv)
        ,y_axis_type='log',height=700, width=600)

    #====================== INITIALIZE TAUGAS#======================
    TAUGAS = np.zeros((nlayer,nwno,ngauss)) #nlayer x nwave x ngauss
    TAURAY = np.zeros((nlayer,nwno,ngauss)) #nlayer x nwave x ngauss
    TAUCLD = np.zeros((nlayer,nwno,ngauss)) #nlayer x nwave x ngauss
    asym_factor_cld = np.zeros((nlayer,nwno,ngauss)) #nlayer x nwave x ngauss
    single_scattering_cld = np.zeros((nlayer,nwno,ngauss)) #nlayer x nwave x ngauss
    raman_factor = np.zeros((nlayer,nwno,ngauss)) #nlayer x nwave x ngauss 

    c=1
    #set color scheme.. adding 3 for raman, rayleigh, and total
    if plot_opacity: colors = inferno(3+len(atm.continuum_molecules) + len(atm.molecules))

    #====================== ADD CONTIMUUM OPACITY====================== 
    #Set up coefficients needed to convert amagat to a normal human unit
    #these COEF's are only used for the continuum opacity. 
    tlevel = atm.level['temperature']
    #these units are purely because of the wonky continuum units
    #plevel in this routine is only used for those 
    plevel = atm.level['pressure']/atm.c.pconv #THIS IS DANGEROUS, used for continuum COEF's below
    tlayer = atm.layer['temperature']
    player = atm.layer['pressure']/atm.c.pconv #THIS IS DANGEROUS, used for continuum
    gravity = atm.planet.gravity / 100.0  #THIS IS DANGEROUS, used for continuum

    ACOEF = (tlayer/(tlevel[:-1]*tlevel[1:]))*(
            tlevel[1:]*plevel[1:] - tlevel[:-1]*plevel[:-1])/(plevel[1:]-plevel[:-1]) #UNITLESS

    BCOEF = (tlayer/(tlevel[:-1]*tlevel[1:]))*(
            tlevel[:-1] - tlevel[1:])/(plevel[1:]-plevel[:-1]) #INVERSE PRESSURE

    COEF1 = atm.c.rgas*273.15**2*.5E5* (
        ACOEF* (plevel[1:]**2 - plevel[:-1]**2) + BCOEF*(
            2./3.)*(plevel[1:]**3 - plevel[:-1]**3) ) / (
        1.01325**2 *gravity*tlayer*atm.layer['mmw']) 


    #go through every molecule in the continuum first 
    colden = atm.layer['colden'][:,np.newaxis]
    mmw = atm.layer['mmw'][:,np.newaxis]
    player = atm.layer['pressure'][:,np.newaxis]
    tlayer = atm.layer['temperature'][:,np.newaxis]
    for m in atm.continuum_molecules:

        #H- Bound-Free
        if (m[0] == "H-") and (m[1] == "bf"):
            ADDTAU = (opacityclass.continuum_opa['H-bf']*( #[(nlayer x nwno) *(
                            atm.layer['mixingratios'][m[0]].values[:,np.newaxis] *  #nlayer
                            colden/                     #nlayer
                            (mmw*atm.c.amu))   )     #nlayer)]

            TAUGAS[:,:,0] += ADDTAU
            if plot_opacity: opt_figure.line(1e4/opacityclass.wno, ADDTAU[plot_layer,:], alpha=0.7,legend_label=m[0]+m[1], line_width=3, color=colors[c],
            muted_color=colors[c], muted_alpha=0.2)
            if return_mode: taus_by_species[m[0]+m[1]] = ADDTAU
        
        #H- Free-Free
        elif (m[0] == "H-") and (m[1] == "ff"):
            ADDTAU = (opacityclass.continuum_opa['H-ff']*(    #[(nlayer x nwno) *(
                            player*                                       #nlayer
                            atm.layer['mixingratios']['H'].values[:,np.newaxis] *
                            atm.layer['electrons'][:,np.newaxis] *#nlayer
                            colden/                                         #nlayer
                            (tlayer*mmw*atm.c.amu*atm.c.k_b))  )          #nlayer)].T

            TAUGAS[:,:,0] += ADDTAU

            if plot_opacity: opt_figure.line(1e4/opacityclass.wno, ADDTAU[plot_layer,:], alpha=0.7,legend_label=m[0]+m[1], line_width=3, color=colors[c],
            muted_color=colors[c], muted_alpha=0.2)

            if return_mode: taus_by_species[m[0]+m[1]] = ADDTAU

        #H2- 
        elif (m[0] == "H2-") and (m[1] == ""): 
            #calculate opacity
            #this is a hefty matrix multiplication to make sure that we are 
            #multiplying each column of the opacities by the same 1D vector (as opposed to traditional 
            #matrix multiplication). This is the reason for the transposes.
            ADDTAU = (opacityclass.continuum_opa['H2-']*(    #[(nlayer x nwno) *(
                            player*                                      #nlayer
                            atm.layer['mixingratios']['H2'].values[:,np.newaxis] *
                            atm.layer['electrons'][:,np.newaxis] *  #nlayer
                            colden/                                        #nlayer
                            (mmw*atm.c.amu))   )                        #nlayer)]


            TAUGAS[:,:,0] += ADDTAU
            if plot_opacity: opt_figure.line(1e4/opacityclass.wno, ADDTAU[plot_layer,:], alpha=0.7,legend_label=m[0]+m[1], line_width=3, color=colors[c],
            muted_color=colors[c], muted_alpha=0.2)
            if return_mode: taus_by_species[m[0]+m[1]] = ADDTAU
        #everything else.. e.g. H2-H2, H2-CH4. Automatically determined by which molecules were requested
        else:

            #calculate opacity
            ADDTAU = (opacityclass.continuum_opa[m[0]+m[1]] * ( #[(nlayer x nwno) *(
                                COEF1[:,np.newaxis]*                                          #nlayer
                                atm.layer['mixingratios'][m[0]].values[:,np.newaxis]  *                #nlayer
                                atm.layer['mixingratios'][m[1]].values[:,np.newaxis]  )  )          #nlayer)]

            TAUGAS[:,:,0] += ADDTAU
            if plot_opacity: opt_figure.line(1e4/opacityclass.wno, ADDTAU[plot_layer,:], alpha=0.7,legend_label=m[0]+m[1], line_width=3, color=colors[c],
            muted_color=colors[c], muted_alpha=0.2)
            if return_mode: taus_by_species[m[0]+m[1]] = ADDTAU
        c+=1
    
    # ADD CONTINUUM TO OTHER GAUSS POINTS 
    # Continuum doesn't need to go through normal correlated K path way since there 
    # are no lines to worry about. 
    for igauss in range(1,ngauss):TAUGAS[:,:,igauss] = TAUGAS[:,:,0]

    #====================== ADD MOLECULAR OPACITY======================
    #if monochromatic opacities, grap each molecular opacity individually
    
    if ngauss == 1:  
        for m in atm.molecules:

            ADDTAU = (opacityclass.molecular_opa[m] * ( #[(nlayer x nwno) *(
                        colden*
                        atm.layer['mixingratios'][m].values[:,np.newaxis]  / 
                        mmw) )
            TAUGAS[:,:,0] += ADDTAU
            #testing[m] = ADDTAU
            if plot_opacity: opt_figure.line(1e4/opacityclass.wno, ADDTAU[plot_layer,:,], alpha=0.7,legend_label=m, line_width=3, color=colors[c],
                muted_color=colors[c], muted_alpha=0.2)
            if return_mode: taus_by_species[m] = ADDTAU
            c+=1

    elif ngauss > 1: 
        for igauss in range(ngauss):
            ADDTAU = (opacityclass.molecular_opa[:,:,igauss] * ( # nlayer x nwave x ngauss
                        colden/
                        mmw) )
            TAUGAS[:,:,igauss] += ADDTAU
    
    #====================== ADD RAYLEIGH OPACITY======================  
    for m in atm.rayleigh_molecules:
        ray_matrix = np.array([opacityclass.rayleigh_opa[m]]*nlayer)
        ADDTAU = (ray_matrix * ( #[(nwno x nlayer) *(
                    colden*
                    atm.layer['mixingratios'][m].values[:,np.newaxis] / #removing this bc of opa unit change *atm.weights[m].values[0]/ 
                    mmw))
        TAURAY[:,:,0] += ADDTAU 


    # ADD RAYLEIGH TO OTHER GAUSS POINTS 
    # Continuum doesn't need to go through normal correlated K path way since there 
    # are no lines to worry about. 
    for igauss in range(1,ngauss): TAURAY[:,:,igauss] = TAURAY[:,:,0]


    if plot_opacity: opt_figure.line(1e4/opacityclass.wno, TAURAY[plot_layer,:,0], alpha=0.7,legend_label='Rayleigh', line_width=3, color=colors[c],
            muted_color=colors[c], muted_alpha=0.2) 
    if return_mode: taus_by_species['rayleigh'] = ADDTAU

    #====================== ADD RAMAN OPACITY======================
    #OKLOPCIC OPACITY
    if raman == 0 :
        raman_db = opacityclass.raman_db
        raman_factor[:,:,0] = compute_raman(nwno, nlayer,opacityclass.wno, 
            opacityclass.raman_stellar_shifts, atm.layer['temperature'], raman_db['c'].values,
                raman_db['ji'].values, raman_db['deltanu'].values)
        if plot_opacity: opt_figure.line(1e4/opacityclass.wno, raman_factor[plot_layer,:,0]*TAURAY[plot_layer,:,0], alpha=0.7,legend_label='Shifted Raman', line_width=3, color=colors[c],
                muted_color=colors[c], muted_alpha=0.2)
        raman_factor[:,:,0] = np.minimum(raman_factor[:,:,0], raman_factor[:,:,0]*0+0.99999)
    #POLLACK OPACITY
    elif raman ==1: 
        raman_factor[:,:,0] = raman_pollack(nlayer,1e4/opacityclass.wno)
        raman_factor[:,:,0] = np.minimum(raman_factor[:,:,0], raman_factor[:,:,0]*0+0.99999)  
        if plot_opacity: opt_figure.line(1e4/opacityclass.wno, raman_factor[plot_layer,:,0]*TAURAY[plot_layer,:,0], alpha=0.7,legend_label='Shifted Raman', line_width=3, color=colors[c],
                muted_color=colors[c], muted_alpha=0.2)
    #NOTHING
    else: 
        raman_factor = 0.99999 + np.zeros((nlayer, nwno,ngauss))

    #fill rest of gauss points
    for igauss in range(1,ngauss):raman_factor[:,:,igauss] = raman_factor[:,:,0]

    #====================== ADD CLOUD OPACITY====================== 
    for igauss in range(ngauss):
        TAUCLD[:,:,igauss] = atm.layer['cloud']['opd'] #TAUCLD is the total extinction from cloud = (abs + scattering)
        asym_factor_cld[:,:,igauss] = atm.layer['cloud']['g0']
        single_scattering_cld[:,:,igauss] = atm.layer['cloud']['w0'] 

    if return_mode: 
        taus_by_species['cloud'] = TAUCLD[:,:,0]*single_scattering_cld[:,:,0]
        return taus_by_species
        
    #====================== If user requests full output, add Tau's to atmosphere class=====
    if full_output:
        atmosphere.taugas = TAUGAS
        atmosphere.tauray = TAURAY
        atmosphere.taucld = TAUCLD

    #====================== ADD EVERYTHING TOGETHER PER LAYER====================== 
    #formerly DTAU
    DTAU = TAUGAS + TAURAY + TAUCLD 
    
    # This is the fractional of the total scattering that will be due to the cloud
    #VERY important note. You must weight the taucld by the single scattering 
    #this is because we only care about the fractional opacity from the cloud that is 
    #scattering. 
    ftau_cld = (single_scattering_cld * TAUCLD)/(single_scattering_cld * TAUCLD + TAURAY)

    COSB = ftau_cld*asym_factor_cld

    #formerly GCOSB2 
    ftau_ray = TAURAY/(TAURAY + single_scattering_cld * TAUCLD)
    GCOS2 = 0.5*ftau_ray #Hansen & Travis 1974 for Rayleigh scattering 

    #Raman correction is usually used for reflected light calculations 
    #although users to have option turn it off in reflected light as well 
    W0 = (TAURAY*raman_factor + TAUCLD*single_scattering_cld) / (TAUGAS + TAURAY + TAUCLD) #TOTAL single scattering 

    #if a user wants both reflected and thermal, this computes SSA without raman correction, but with 
    #scattering from clouds still
    W0_no_raman = (TAURAY*0.99999 + TAUCLD*single_scattering_cld) / (TAUGAS + TAURAY + TAUCLD) #TOTAL single scattering 

    #sum up taus starting at the top, going to depth
    TAU = np.zeros((nlayer+1, nwno,ngauss))
    for igauss in range(ngauss): TAU[1:,:,igauss]=numba_cumsum(DTAU[:,:,igauss])

    if plot_opacity:
        opt_figure.line(1e4/opacityclass.wno, DTAU[plot_layer,:,0], legend_label='TOTAL', line_width=4, color=colors[0],
            muted_color=colors[c], muted_alpha=0.2)
        opt_figure.legend.click_policy="mute"
        show(opt_figure)

    if test_mode != None:  
            #this is to check against Dlugach & Yanovitskij 
            #https://www.sciencedirect.com/science/article/pii/0019103574901675?via%3Dihub
            if test_mode=='rayleigh':
                DTAU = TAURAY 
                #GCOS2 = 0.5
                GCOS2 = np.zeros(DTAU.shape) + 0.5
                #ftau_ray = 1.0
                ftau_ray = np.zeros(DTAU.shape) + 1.0
                #ftau_cld = 1e-6
                ftau_cld = np.zeros(DTAU.shape) #+ 1e-6
            else: 
                DTAU = atm.layer['cloud']['opd']#TAURAY*0+0.05
                GCOS2 = np.zeros(DTAU.shape)#0.0
                ftau_ray = np.zeros(DTAU.shape)#+1e-6
                ftau_cld = np.zeros(DTAU.shape)+1            
            COSB = atm.layer['cloud']['g0']
            W0 = atm.layer['cloud']['w0']
            W0_no_raman = atm.layer['cloud']['w0']
            TAU = np.zeros((nlayer+1, nwno,ngauss))
            TAU[1:,:]=numba_cumsum(DTAU)
            for igauss in range(ngauss): TAU[1:,:,igauss]=numba_cumsum(DTAU[:,:,igauss])
    #====================== D-Eddington Approximation======================
    if delta_eddington:
        #First thing to do is to use the delta function to icorporate the forward 
        #peak contribution of scattering by adjusting optical properties such that 
        #the fraction of scattered energy in the forward direction is removed from 
        #the scattering parameters 

        #Joseph, J.H., W. J. Wiscombe, and J. A. Weinman, 
        #The Delta-Eddington approximation for radiative flux transfer, J. Atmos. Sci. 33, 2452-2459, 1976.

        #also see these lecture notes are pretty good
        #http://irina.eas.gatech.edu/EAS8803_SPRING2012/Lec20.pdf
        w0_dedd=W0*(1.-COSB**stream)/(1.0-W0*COSB**stream)
        cosb_dedd=COSB/(1.+COSB)
        dtau_dedd=DTAU*(1.-W0*COSB**stream) 

        #sum up taus starting at the top, going to depth
        tau_dedd = np.zeros((nlayer+1, nwno, ngauss))
        for igauss in range(ngauss): tau_dedd[1:,:,igauss]=numba_cumsum(dtau_dedd[:,:,igauss])
    
        #returning the terms used in 
        return (dtau_dedd, tau_dedd, w0_dedd, cosb_dedd ,ftau_cld, ftau_ray, GCOS2, 
                DTAU, TAU, W0, COSB,    #these are returned twice because we need the uncorrected 
                W0_no_raman)            #values for single scattering terms where we use the TTHG phase function
                                        # w0_no_raman is used in thermal calculations only

    else: 
        return (DTAU, TAU, W0, COSB, ftau_cld, ftau_ray, GCOS2, 
                DTAU, TAU, W0, COSB,  #these are returned twice for consistency with the delta-eddington option
                W0_no_raman)          #W0_no_raman is used for thermal calculations only 


@jit(nopython=True, cache=True)
def compute_raman(nwno, nlayer, wno, stellar_shifts, tlayer, cross_sections, j_initial, deltanu):
    """
    The Ramam scattering will alter the rayleigh scattering. The returned value is 
    modified single scattering albedo. 

    Cross sectiosn from: 
    http://iopscience.iop.org/0004-637X/832/1/30/suppdata/apjaa3ec7t2_mrt.txt

    This method is described in Pollack+1986. Albeit not the best method. Sromovsky+2005 
    pointed out the inconsistencies in this method. You can see from his comparisons 
    that the Pollack approximations don't accurately capture the depths of the line centers. 
    Since then, OKLOPCIC+2016 recomputed cross sections for J<=9. We are using those cross 
    sections here with a different star. Huge improvement over what we had before. 
    Number of J levels is hard coded to 10 ! 

    Will be added to the rayleigh scattering as : TAURAY*RAMAN

    Parameters
    ----------
    nwno : int 
        Number of wave points
    nlayer : int 
        Number of layers 
    wno : array
        Array of output grid of wavenumbers
    stellar_shifts : ndarray 
        Array of floats that has dimensions (n wave pts x n J levels: 0-9)
    tlayer : array 
        Array of floats that has dimensions nlayer 
    cross_sections : ndarray 
        The row of "C's" from Antonija's table. 
    j_initial : ndarray 
        The row of initial rotational energy states from Antonija's table
    deltanu : ndarray
        The row of delta nu's from Antonia's table

    """
    raman_sigma_w_shift = np.zeros(( nlayer,nwno))
    raman_sigma_wo_shift = np.zeros(( nlayer,nwno))
    rayleigh_sigma = np.zeros(( nlayer,nwno))

    #first calculate the j fraction at every layer 

    number_of_Js = 10 #(including 0)
    j_at_temp = np.zeros((number_of_Js,nlayer))
    for i in range(number_of_Js):
        j_at_temp[i,:] = j_fraction(i,tlayer)
    
    for i in range(cross_sections.shape[0]):
        #define initial state
        ji = j_initial[i]
        #for that initial state.. What is the expected wavenumber shift 
        shifted_wno = wno + deltanu[i]
        #compute the cross section 
        Q = cross_sections[i] / wno**3.0 / shifted_wno #see A4 in Antonija's 2018 paper
        #if deltanu is zero that is technically rayleigh scattering
        if deltanu[i] == 0 :
            rayleigh_sigma += np.outer(j_at_temp[ji,:] , Q )
        else:
            #if not, then compute the shifted and unshifted raman scattering
            raman_sigma_w_shift += np.outer(j_at_temp[ji,:] , Q*stellar_shifts[:,i])
            raman_sigma_wo_shift += np.outer(j_at_temp[ji,:] , Q)

    #finally return the contribution that will be added to total rayleigh
    return (rayleigh_sigma + raman_sigma_w_shift)/ (rayleigh_sigma + raman_sigma_wo_shift)

@jit(nopython=True, cache=True)
def bin_star(wno_new,wno_old, Fp):
    """
    Takes average of group of points using uniform tophat 
    
    Parameters
    ----------
    wno_new : numpy.array
        inverse cm grid to bin to (cm-1) 
    wno_old : numpy.array
        inverse cm grid (old grid)
    Fp : numpy.array
        transmission spectra, which is on wno grid
    """
    szmod=wno_new.shape[0]

    delta=np.zeros(szmod)
    Fint=np.zeros(szmod)
    delta[0:-1]=wno_new[1:]-wno_new[:-1]  
    delta[szmod-1]=delta[szmod-2] 
    for i in range(1,szmod):
        loc=np.where((wno_old >= wno_new[i]-0.5*delta[i-1]) & (wno_old < wno_new[i]+0.5*delta[i]))
        Fint[i]=np.mean(Fp[loc])
    loc=np.where((wno_old > wno_new[0]-0.5*delta[0]) & (wno_old < wno_new[0]+0.5*delta[0]))
    Fint[0]=np.mean(Fp[loc])
    return Fint


@jit(nopython=True, cache=True)
def partition_function(j, T):
    """
    Given j and T computes partition function for ro-vibrational levels of H2. 
    This is the exponential and the statistical weight g_J in 
    Eqn 3 in https://arxiv.org/pdf/1605.07185.pdf
    It is also used to compute the partition sum Z.

    Parameters
    ----------
    j : int 
        Energy level 
    T : array float 
        Temperature at each atmospheric layer in the atmosphere

    Returns
    -------
    Returns partition function 
    """
    k = 1.38064852e-16  #(c.k_B.cgs)
    b = 60.853# units of /u.cm #rotational constant for H2
    c =  29979245800 #.c.cgs
    h = 6.62607004e-27 #c.h.cgs
    b_energy = (b*(h)*(c)*j*(j+1)/k)
    if (j % 2 == 0 ):
        return (2.0*j+1.0)*np.exp(-0.5*b_energy*j*(j+1)/T) #1/2 is the symmetry # for homonuclear molecule
    else:
        return 3.0*(2.0*j+1.0)*np.exp(-0.5*b_energy*j*(j+1)/T)

@jit(nopython=True, cache=True)
def partition_sum(T):
    """
    This is the total partition sum. I am truncating it at 20 since it seems to approach 1 around then. 
    This is also pretty fast to compute so 20 is fine for now. 

    Parameters
    ----------
    T : array 
        Array of temperatures for each layer 

    Returns
    -------
    Z, the partition sum 
    """
    Z=np.zeros(T.size)
    for j in range(0,20):
        Z += partition_function(j,T)
    return Z

@jit(nopython=True, cache=True)
def j_fraction(j,T):
    """
    This computes the fraction of molecules at each J level. 

    Parameters
    ----------
    j : int 
        The initial rotational levels ranging from J=0 to 9 for hydrogen only

    Returns
    -------
    f_J in eqn. 3 https://arxiv.org/pdf/1605.07185.pdf
    """
    return partition_function(j,T)/partition_sum(T)

#@jit(nopython=True, cache=True)
def raman_pollack(nlayer,wave):
    """
    Mystery raman scattering. Couldn't figure out where it came from.. so discontinuing. 
    Currently function doesnt' totally work. In half fortran-half python. Legacy from 
    fortran albedo code. 

    This method is described in Pollack+1986. Albeit not the best method. Sromovsky+2005 
    pointed out the inconsistencies in this method. You can see from his comparisons 
    that the Pollack approximations don't accurately capture the depths of the line centers. 
    Since then, OKLOPCIC+2016 did a much
    better model of raman scattring (with ghost lines). Might be worth it to consider a more 
    sophisticated version of Raman scattering. 

    Will be added to the rayleigh scattering as : TAURAY*RAMAN
    
    #OLD FORTRAN CODE
    #constants 
    h = 6.6252e-27
    c = 2.9978e10
    bohrd = 5.2917e-9
    hmass = 1.6734e-24
    rmu = .5 * hmass

    #set wavelength shift of the ramam scatterer
    shift_v0 = 4161.0 

    facip = h * c / ( 1.e-4 * 27.2 * 1.602e-12 ) 
    facray = 1.e16 * bohrd ** 3 * 128. * np.pi ** 5 * bohrd ** 3 / 9. 
    facv = 2.08 / 2.38 * facray / bohrd ** 2 / ( 8. * np.pi * np.pi * rmu * c * shift_v0 ) * h

    #cross section of the unshifted rayleigh and the vibrationally shifted rayleigh
    gli = np.zeros(5)
    wli = np.zeros(5) 
    gri = np.zeros(5) 
    wri = np.zeros(5) 
    gli[:] = [1.296, .247, .297,  .157,  .003]
    wli[:] = [.507, .628, .733, 1.175, 2.526]
    gri[:] = [.913, .239, .440,  .344,  .064]
    wri[:] = [.537, .639, .789, 1.304, 3.263]

    alp = np.zeros(7)
    arp = np.zeros(7)
    alp[:] = [6.84, 6.96, 7.33, 8.02, 9.18, 11.1, 14.5 ]
    arp[:] = [3.66, 3.71, 3.88, 4.19, 4.70, 5.52, 6.88 ]

    omega = facip / wavelength

    #first compute extinction cross section for unshifted component 
    #e.g. rayleigh
    alpha_l=0
    alpha_r=0

    for i in range(5):
        alpha_l += gli[i] / ( wli[i] ** 2 - omega ** 2 ) 
        alpha_r += gri[i] / ( wri[i] ** 2 - omega ** 2 )

    alpha2 = (( 2. * alpha_r + alpha_l ) / 3. ) ** 2
    gamma2 = ( alpha_l - alpha_r ) ** 2
    qray = facray * ( 3. * alpha2 + 2./3. * gamma2 ) / wavelength ** 4

    #next, compute the extinction cross section for vibrationally 
    #shifted component 
    ip = np.zeros(2)
    ip[:] = [int(omega/0.05), 5.0]
    ip = np.min(ip) + 1 
    f = omega / 0.05 - float(ip-1)
    alpha_pl = ( 1. - f ) * alp[ip] + f * alp[ip+1]
    alpha_pr = ( 1. - f ) * arp[ip] + f * arp[ip+1]
    alpha_p2 = (( 2. * alpha_pr + alpha_pl ) / 3. ) ** 2
    gamma_p2 = ( alpha_pl - alpha_pr ) ** 2
    qv = facv / SHIFT( WAVEL, -SHIFTV0 ) ** 4 * ( 3. * alpha_p2 + 2./3. * gamma_p2 )    
    """
    dat = pd.read_csv(os.path.join(os.environ.get('picaso_refdata'), 'opacities','raman_fortran.txt'),
                        delim_whitespace=True, header=None, names = ['w','f'])
    #fill in matrix to match real raman format
    interp_raman = np.interp(wave, dat['w'].values, dat['f'].values, )
    raman_factor = np.zeros((nlayer, len(wave)))
    for i in range(nlayer): 
        raman_factor[i,:] = interp_raman#return flipped values for raman
    return  raman_factor 

class RetrieveCKs():
    """
    This will be the class to retrieve correlated-k tables from the database. 
    Right now this is in beta mode and is retrieving the direct heritage 
    196 grid files. 

    Parameters
    ----------
    ck_dir : str 
        Directory of the pre-mixed correlated k table data. Check that you are pointing to a 
        directory with the files "ascii_data" and "full_abunds". 
    cont_dir : str 
        This should be in the references directory on Github. This has all the continuum opacity 
        that you need (e.g. H2H2). It also has some extra cross sections that you may want to add 
        in like Karkoschka methane and the optical ozone band. 
    wave_range : list 
        NOT FUNCTIONAL YET. 
        Wavelength range to compuate in the format [min micron, max micron]
    """
    def __init__(self, ck_dir, cont_dir, wave_range=None):
        self.ck_filename = ck_dir
        self.db_filename = cont_dir
        self.get_legacy_data(wave_range) #wave_range not used yet
        self.get_available_continuum()
        self.get_available_rayleigh()
        return

    def get_legacy_data(self,wave_range):
        data = pd.read_csv(os.path.join(self.ck_filename,'ascii_data'), 
                  delim_whitespace=True,header=None, 
                  names=list(range(9)),dtype=str)

        self.full_abunds =  pd.read_csv(os.path.join(self.ck_filename,'full_abunds'),
            delim_whitespace=True)

        num_species = int(data.iloc[0,0])
        max_ele = 35 
        self.max_tc = 60 
        self.max_pc = 18
        max_windows = 200 

        self.molecules = [str(data.iloc[i,j]) for i in [0,1,2] 
           for j in range(9)][1:num_species+1]

        first = [float(i) for i in data.iloc[2,2:4]]
        last = [float(data.iloc[int(max_ele*self.max_pc*self.max_tc/3)+2,0])]

        end_abunds = 2+int(max_ele*self.max_pc*self.max_tc/3)
        abunds = list(np.array(
            data.iloc[3:end_abunds,0:3].astype(float)
            ).ravel())
        abunds = first + abunds + last

        #abundances for elements as a funtion of pressure, temp, and element
        #self.abunds = np.reshape(abunds,(self.max_pc,self.max_tc,max_ele),order='F')

        self.nwno = int(data.iloc[end_abunds,1])

        end_window = int(max_windows/3)
        self.wno = (data.iloc[end_abunds:end_abunds+end_window,0:3].astype(float)).values.ravel(
        )[2:]
        self.delta_wno = (data.iloc[end_abunds+end_window+1:1+end_abunds+2*end_window,0:3].astype(float)).values.ravel(
        )[1:-1]
        end_windows =2+end_abunds+2*end_window

        nc_t=int(data.iloc[end_windows,0])
        nc_p = np.array(data.iloc[end_windows:1+end_windows+int(self.max_tc/6),0:6].astype(int
                    )).ravel()[1:-5]
        end_npt = 1+end_windows+int(self.max_tc/6) + 9 #9 dummy rows

        first = list(data.iloc[end_npt,2:4].astype(float))

        self.pressures = first+list(np.array(data.iloc[end_npt+1:end_npt + int(self.max_pc*self.max_tc/3) + 1,0:3]
                 .astype(float))
                 .ravel()[0:-2])
        #pressures = np.array(pressures)[np.where(np.array(pressures)>0)]
        end_ps = end_npt + int(self.max_pc*self.max_tc/3)

        self.temps = list(np.array(data.iloc[end_ps:1+int(end_ps+nc_t/3),0:3]
                .astype(float))
                .ravel()[1:-2])
        end_temps = int(end_ps+nc_t/3)

        ngauss1, ngauss2, gfrac =data.iloc[end_temps,1:4].astype(float)
        self.ngauss = int(data.iloc[end_temps+1,0])

        assert self.ngauss == 8, 'Legacy code uses 8 gauss points not {0}. Check read in statements'.format(self.ngauss)

        gpts_wts = np.reshape(np.array(data.iloc[end_temps+1:2+end_temps+int(2*self.ngauss/3),0:3]
                 .astype(float)).ravel()[1:-1], (self.ngauss,2))

        self.gauss_pts = [i[0] for i in gpts_wts]
        self.gauss_wts = [i[1] for i in gpts_wts]

        kappa = np.array(data.iloc[3+end_temps+int(2*self.ngauss/3):-2,0:3].astype(float)).ravel()
        kappa = np.reshape(kappa, (max_windows,self.ngauss*2,self.max_pc,self.max_tc),order='F')
        #want the axes to be [npressure, ntemperature, nwave, ngauss ]
        kappa = kappa.swapaxes(1,3)
        kappa = kappa.swapaxes(0,2)
        self.kappa = kappa[:, :, 0:self.nwno, 0:self.ngauss] 

        
    
    def get_available_rayleigh(self):
        data = Rayleigh(self.wno)
        self.rayleigh_molecules = data.rayleigh_molecules
        self.rayleigh_opa={}
        for i in data.rayleigh_molecules: 
            self.rayleigh_opa[i] = data.compute_sigma(i)

    def get_available_continuum(self):
        """Get the pressures and temperatures that are available for the continuum and molecules
        """
        #open connection 
        cur, conn = self.open_local()

        #get temps
        cur.execute('SELECT temperature FROM continuum')
        self.cia_temps = np.unique(cur.fetchall())
        
    def get_pre_mix_ck(self,atmosphere):
        """
        Takes in atmosphere profile and returns an array which is 
        nlayer by ngauss by nwno
        """
        nlayer =atmosphere.c.nlayer
        tlayer =atmosphere.layer['temperature']
        player = atmosphere.layer['pressure']/atmosphere.c.pconv

        pt_pairs = []
        i=0
        for ip,it,p,t in zip(np.concatenate([list(range(self.max_pc))*self.max_tc]), 
                        np.concatenate([[it]*self.max_pc for it in range(self.max_tc)]),
                        self.pressures, 
                        np.concatenate([[it]*self.max_pc for it in self.temps])):
            
            if p!=0 : pt_pairs += [[i,ip,it,p/1e3,t]];i+=1

        ind_pt_log = np.array([min(pt_pairs, 
                    key=lambda c: math.hypot(np.log(c[-2])- np.log(coordinate[0]), 
                                             c[-1]-coordinate[1]))[0:3] 
                        for coordinate in  zip(player,tlayer)])

        ind_p = ind_pt_log[:,1]
        ind_t = ind_pt_log[:,2]

        self.molecular_opa = np.exp(self.kappa[ind_p, ind_t, :, :])*6.02214086e+23  #avogadro constant!

    def get_continuum(self, atmosphere):
        #open connection 
        cur, conn = self.open_local()

        nlayer =atmosphere.c.nlayer
        tlayer =atmosphere.layer['temperature']
        player = atmosphere.layer['pressure']/atmosphere.c.pconv

        cia_molecules = atmosphere.continuum_molecules


        self.continuum_opa = {key[0]+key[1]:np.zeros((nlayer,self.nwno)) for key in cia_molecules}

        #continuum
        #find nearest temp for cia grid
        tcia = [np.unique(self.cia_temps)[find_nearest(np.unique(self.cia_temps),i)] for i in tlayer]

        #if user only runs a single molecule or temperature
        if len(tcia) ==1: 
            query_temp = """AND temperature= '{}' """.format(str(tcia[0]))
        else:
            query_temp = 'AND temperature in '+str(tuple(tcia) )
        cia_mol = list(self.continuum_opa.keys())
        if len(cia_mol) ==1: 
            query_mol = """WHERE molecule= '{}' """.format(str(cia_mol[0]))
        else:
            query_mol = 'WHERE molecule in '+str(tuple(cia_mol) )       

        cur.execute("""SELECT molecule,temperature,opacity 
                    FROM continuum 
                    {} 
                    {}""".format(query_mol, query_temp))

        data = cur.fetchall()
        data = dict((x+'_'+str(y), dat) for x, y,dat in data)
        for i in self.continuum_opa.keys():
            for j,ind in zip(tcia,range(nlayer)):
                self.continuum_opa[i][ind,:] = data[i+'_'+str(j)]

        conn.close()

    def open_local(self):
        """Code needed to open up local database, interpret arrays from bytes and return cursor"""
        conn = sqlite3.connect(self.db_filename, detect_types=sqlite3.PARSE_DECLTYPES)
        #tell sqlite what to do with an array
        sqlite3.register_adapter(np.ndarray, self.adapt_array)
        sqlite3.register_converter("array", self.convert_array)
        cur = conn.cursor()
        return cur,conn

    def get_opacities(self, atmosphere):
        self.get_continuum(atmosphere)
        self.get_pre_mix_ck(atmosphere)

    def adapt_array(arr):
        """needed to interpret bytes to array"""
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())

    def convert_array(clas, text):
        """needed to interpret bytes to array"""
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)

class RetrieveOpacities():
    """
    This will be the class that will retrieve the opacities from the sqlite3 database. Right 
    now we are just employing nearest neighbors to grab the respective opacities. 
    
    Parameters
    ----------
    db_filename : str 
        Opacity file name 
    raman_data : str 
        file name of the raman cross section data 
    wave_range : list of float
        Wavelength range (in microns) to run. 
        Default None grabs all the full wavelength range available 
    location : str 
        Default = local. This is the only option currently available since 
        we dont have AWS or other services enabled. 
    resample : int 
        Default =1 (no resampling)

    Attributes
    ----------
    raman_db : pandas.DataFrame
        Database of Raman cross sections 
    wno : array 
        Wavenumber grid from opacity database (CGS cm-1)
    wave : array 
        Wavelength grid from opacity database (micron)
    nwno : int 
        Number of wavenumber points in the grid 
    cia_temps : array 
        CIA temps available from database 
    molecules : array 
        Molecules availalbe from database 
    pt_pairs : array 
        pressure-temperature pairs available from grid (our grid is usually 
        computed on a fixed 1060 PT grid)
    molecular_opa : dict 
        Molecular Opacity dictionary that contains all the queried information necessary 
        to do calculation 
    continuum_opa : dict 
        Continuum opacity dictionary that contains all the queried information 
        necessary for the calculation 
    rayleigh_opa : dict 
        Rayleigh opacity dictionary for all the sources we have. This is computed as 
        cm^2/species when the users runs `opannection`. We technically could just 
        store this in the DB as well. However, because its so fast, it doesn't take 
        a lot of time, and because its non-temperature dependent, we only have to do 
        it once. 

    Methods 
    -------
    db_connect 
        Inherents functionality of `open_local`. Eventually it will 
        inherent functionality of "open_non_locals" (e.g. cloud or other sources)
    open_local 
        Opens sqlite3 connection and enables array interpretation from bytes 
    get_available_data 
        Gets opacity database attributes such as available T-Ps , availalbe molecules. etc. 
    get_available_rayleigh
        Gets rayleigh cross sections when opannection is opened.
    get_opacities 
        Gets opacities after user specifies atmospheric profile (e.g. full PT and Composition)
    """
    def __init__(self, db_filename, raman_data, wave_range=None,location = 'local',resample=1):

        #monochromatic opacity option forces number of gauss points to 1
        self.ngauss = 1
        self.gauss_wts = [1]

        if location == 'local':
            #self.conn = sqlite3.connect(db_filename, detect_types=sqlite3.PARSE_DECLTYPES)
            self.db_filename = db_filename
            self.db_connect = self.open_local

        self.get_available_data(wave_range, resample)
        
        #raman cross sections 
        self.raman_db = pd.read_csv(raman_data,
                     delim_whitespace=True, skiprows=16,header=None, names=['ji','jf','vf','c','deltanu'])
        
        #compute available Rayleigh scatterers 
        self.get_available_rayleigh()

    def open_local(self):
        """Code needed to open up local database, interpret arrays from bytes and return cursor"""
        conn = sqlite3.connect(self.db_filename, detect_types=sqlite3.PARSE_DECLTYPES)
        #tell sqlite what to do with an array
        sqlite3.register_adapter(np.ndarray, self.adapt_array)
        sqlite3.register_converter("array", self.convert_array)
        cur = conn.cursor()
        return cur,conn
    def adapt_array(arr):
        """needed to interpret bytes to array"""
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())

    def convert_array(clas, text):
        """needed to interpret bytes to array"""
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)
    def get_available_data(self, wave_range, resample):
        """Get the pressures and temperatures that are available for the continuum and molecules"""
        self.resample = resample
        #open connection 
        cur, conn = self.db_connect()

        #get temps
        cur.execute('SELECT temperature FROM continuum')
        self.cia_temps = np.unique(cur.fetchall())

        #get available molecules
        cur.execute('SELECT molecule FROM molecular')
        self.molecules = np.unique(cur.fetchall())

        #get PT indexes for getting opacities 
        cur.execute('SELECT ptid, pressure, temperature FROM molecular')
        data= cur.fetchall()
        self.pt_pairs = sorted(list(set(data)),key=lambda x: (x[0]) )

        #Get the wave grid info
        cur.execute('SELECT wavenumber_grid FROM header')
        self.wno =  cur.fetchone()[0][::self.resample]
        self.wave = 1e4/self.wno 
        if wave_range == None: 
            self.loc = [True]*len(self.wno )
        else:
            self.loc = np.where(((self.wave>min(wave_range)) & (self.wave<max(wave_range))))
        self.wave = self.wave[self.loc]
        self.wno = self.wno[self.loc]
        self.nwno = np.size(self.wno)

        conn.close()

    def get_available_rayleigh(self):
        data = Rayleigh(self.wno)
        self.rayleigh_molecules = data.rayleigh_molecules
        self.rayleigh_opa={}
        for i in data.rayleigh_molecules: 
            self.rayleigh_opa[i] = data.compute_sigma(i)

    def get_opacities(self,atmosphere, dimension='1d'):
        """
        Get's opacities using the atmosphere class
        """
        #open connection 
        cur, conn = self.db_connect()
        
        nlayer =atmosphere.c.nlayer
        tlayer =atmosphere.layer['temperature']
        player = atmosphere.layer['pressure']/atmosphere.c.pconv

        molecules = atmosphere.molecules
        cia_molecules = atmosphere.continuum_molecules

        #struture opacity dictionary
        self.molecular_opa = {key:np.zeros((nlayer, self.nwno)) for key in molecules}
        self.continuum_opa = {key[0]+key[1]:np.zeros((nlayer, self.nwno)) for key in cia_molecules}

        #this will make getting opacities faster 
        #this is getting the ptid corresponding to the pairs

        ind_pt=[min(self.pt_pairs, 
            key=lambda c: math.hypot(np.log(c[1])- np.log(coordinate[0]), c[2]-coordinate[1]))[0] 
                for coordinate in  zip(player,tlayer)]
        #query molecular opacities from sqlite3
        if len(molecules) ==1: 
            query_mol = """WHERE molecule= '{}' """.format(str(molecules[0]))
        else:
            query_mol = 'WHERE molecule in '+str(tuple(molecules) )

        atmosphere.layer['pt_opa_index'] = ind_pt
        cur.execute("""SELECT molecule,ptid,opacity 
                    FROM molecular 
                    {} 
                    AND ptid in {}""".format(query_mol, str(tuple(np.unique(ind_pt)))))
        #fetch everything and stick into a dictionary where we can find the right
        #pt and molecules
        data= cur.fetchall()
        data = dict((x+'_'+str(y), dat[::self.resample][self.loc]) for x,y,dat in data)        

        #structure it into a dictionary e.g. {'H2O':ndarray(nwave x nlayer), 'CH4':ndarray(nwave x nlayer)}.. 
        for i in self.molecular_opa.keys():
            for j,ind in zip(ind_pt,range(nlayer)): # multiply by avogadro constant 
                self.molecular_opa[i][ind, :] = data[i+'_'+str(j)]*6.02214086e+23 #add to opacity bundle

        #continuum
        #find nearest temp for cia grid
        tcia = [np.unique(self.cia_temps)[find_nearest(np.unique(self.cia_temps),i)] for i in tlayer]

        #if user only runs a single molecule or temperature
        if len(tcia) ==1: 
            query_temp = """AND temperature= '{}' """.format(str(tcia[0]))
        else:
            query_temp = 'AND temperature in '+str(tuple(tcia) )
        cia_mol = list(self.continuum_opa.keys())
        if len(cia_mol) ==1: 
            query_mol = """WHERE molecule= '{}' """.format(str(cia_mol[0]))
        else:
            query_mol = 'WHERE molecule in '+str(tuple(cia_mol) )       

        cur.execute("""SELECT molecule,temperature,opacity 
                    FROM continuum 
                    {} 
                    {}""".format(query_mol, query_temp))

        data = cur.fetchall()
        data = dict((x+'_'+str(y), dat) for x, y,dat in data)
        for i in self.continuum_opa.keys():
            for j,ind in zip(tcia,range(nlayer)):
                self.continuum_opa[i][ind,:] = data[i+'_'+str(j)][::self.resample][self.loc]

        conn.close() 
    def compute_stellar_shits(self, wno_star, flux_star):
        """
        Takes the hires stellar spectrum and computes Raman shifts 
        
        Parameters
        ----------
        wno_star : array 
            hires wavelength grid of the star in wave number
        flux_star : array 
            Flux of the star in whatever units. Doesn't matter for this since it's returning a ratio
        
        Returns
        -------
        matrix
            array of the shifted stellar spec divided by the unshifted stellar spec on the model wave
            number grid 
        """
        model_wno = self.wno
        deltanu = self.raman_db['deltanu'].values

        all_shifted_spec = np.zeros((len(model_wno), len(deltanu)))
        
        self.unshifted_stellar_spec = bin_star(model_wno, wno_star, flux_star)

        for i in range(len(deltanu)):
            dnu = deltanu[i]
            shifted_wno = model_wno + dnu 
            shifted_flux = bin_star(shifted_wno,wno_star, flux_star)
            if i ==0:
                unshifted = shifted_flux*0+shifted_flux
            all_shifted_spec[:,i] = shifted_flux/unshifted

        self.raman_stellar_shifts = all_shifted_spec

def adapt_array(arr):
    """needed to interpret bytes to array"""
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(clas, text):
    """needed to interpret bytes to array"""
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

@jit(nopython=True, cache=True)
def find_nearest(array,value):
    #small program to find the nearest neighbor in temperature  
    idx = (np.abs(array-value)).argmin()
    return idx
@jit(nopython=True, cache=True)
def numba_cumsum(mat):
    """Function to compute cumsum along axis=0 to bypass numba not allowing kwargs in 
    cumsum 
    """
    new_mat = np.zeros(mat.shape)
    for i in range(mat.shape[1]):
        new_mat[:,i] = np.cumsum(mat[:,i])
    return new_mat


@jit(nopython=True, cache=True)
def rayleigh_old(colden,gasmixing,wave,xmu,amu):
    """DISCONTINUED
    Rayleigh function taken from old albedo code. Keeping this modular, as we may want 
    to swap out different methods to calculate rayleigh opacity 

    Parameters
    ----------
    colden : array of float 
        Column Density in CGS units 
    mixingratios : array of float 
        Mixing ratio as function of altitude for [H2  He CH4 ] 
    wave : array of float 
        wavelength (microns) of grid. This should be in DECENDING order to comply with 
        everything else being in wave numberes 
    xmu : array of float
        mean molecular weight of atmosphere in amu 
    amu : float 
        amu constant in grams 
    """
    #define all rayleigh constants
    TAURAY = np.zeros((colden.size, wave.size))
    dpol= np.zeros(3)
    dpol[0], dpol[1], dpol[2]=1.022 , 1.0, 1.0
    gnu = np.zeros((2,3))
    gnu[0,0]=1.355e-4
    gnu[1,0]=1.235e-6
    gnu[0,1]=3.469e-5
    gnu[1,1]=8.139e-8
    gnu[0,2]=4.318e-4
    gnu[1,2]=3.408e-6
    XN0 = 2.687E19
    cfray = 32.0*np.pi**3*1.e21/(3.0*2.687e19)
    cold = colden / (xmu * amu) #nlayers
    #add rayleigh from each contributing gas using corresponding mixing 
    for i in np.arange(0,3,1):
        tec = cfray*(dpol[i]/wave**4)*(gnu[0,i]+gnu[1,i]/   #nwave
                     wave**2)**2 
        TAUR = (cold*gasmixing[:,i]).reshape((-1, 1)) * tec * 1e-5 / XN0
        
        TAURAY += TAUR

    return TAURAY
