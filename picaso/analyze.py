import numpy as np
import pandas as pd
import xarray as xr
import json
import matplotlib.pyplot as plt
import os
from astropy.convolution import convolve, Box1DKernel, Gaussian1DKernel
from scipy import optimize

from matplotlib.ticker import StrMethodFormatter
try:
    import virga.justdoit as vj
    import picaso.justdoit as jdi
except:
    print("Please install virga or Picaso to use all the functionalities of this analysis code")



def read_parameter_space_models(model,location,grid_dimensions=False,Verbose=True):
    
    
    dir_exists = os.path.isdir(location)
    
    
    
    if (model=="Phoenix") & (dir_exists == True):
        if Verbose == True:
            print("\033[1m Loading parameters for Phoenix Grid \033[0m")
    elif (model == "Picaso") & (dir_exists == True):
        if Verbose == True:
            print("\033[1m Loading parameters for Picaso Grid \033[0m ")
    elif (model == "Picaso_cld") & (dir_exists == True):
        if Verbose == True:
            print("\033[1m Loading parameters for Picaso Cloud Grid \033[0m")
    elif (model == "Picaso_deq") & (dir_exists == True):
        if Verbose == True:
            print("\033[1m Loading parameters for Picaso DEQ Grid \033[0m")
    elif (model == "Picaso_deq_cld") & (dir_exists == True):
        if Verbose == True:
            print("\033[1m Loading parameters for Picaso DEQ CLD Grid \033[0m")
    elif (model == "Atmo") & (dir_exists == True):
        if Verbose == True:
            print("\033[1m Loading parameters for Atmo Grid \033[0m")
    else:
        raise ValueError("Please check what grid you are loading or if the location of grid exists; options are 'Phoenix', 'Picaso' or 'Atmo Grid'")
    
    rp_arr,mp_arr,Tint_arr,heat_redis_arr,pref_arr,logkzz_arr = np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])        
    rs_arr,logg_arr,steff_arr,feh_arr,ms_arr = np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
    mh_arr,cto_arr,pquench_arr = np.array([]),np.array([]),np.array([])
    opd_arr,ssa_arr,asy_arr,p_cloud_arr,haze_eff_arr = np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
    rainout_arr = np.array([])
    fsed_arr = np.array([])
    list_of_files = os.listdir(location) # dir is your directory path
    number_files = len(list_of_files)
    filename_arr = np.array([])
    
    ct=0
    for filename in os.listdir(location):
        
        f = os.path.join(location, filename)
        
        # checking if it is a file
        if os.path.isfile(f):
            
            ds = xr.open_dataset(f)
            
            
            filename_arr = np.append(filename_arr,filename)
            if ct == 0:
                nwave = len(ds['wavelength'].values)
                npress = len(ds['pressure'].values)
                
                spectra_grid = np.zeros(shape=(number_files,nwave))
                temperature_grid = np.zeros(shape=(number_files,npress))
                pressure_grid = np.zeros(shape=(number_files,npress))
                wavelength = ds['wavelength'].values
            
            
            temperature_grid[ct,:] = ds['temperature'].values[:]
            pressure_grid[ct,:] = ds['pressure'].values[:]
            spectra_grid[ct,:] = ds['transit_depth'].values[:]
            
            # Read all the paramaters in the Xarray so that User can gain insight into the 
            # grid parameters
            rp_arr = np.append(rp_arr,json.loads(ds.attrs['planet_params'])['rp']['value'])
            mp_arr = np.append(mp_arr,json.loads(ds.attrs['planet_params'])['mp']['value'])
            Tint_arr = np.append(Tint_arr,json.loads(ds.attrs['planet_params'])['tint'])
            heat_redis_arr = np.append(heat_redis_arr,json.loads(ds.attrs['planet_params'])['heat_redis'])
            pref_arr = np.append(pref_arr,json.loads(ds.attrs['planet_params'])['p_reference']['value'])

            mh_arr = np.append(mh_arr,json.loads(ds.attrs['planet_params'])['mh'])
            cto_arr = np.append(cto_arr,np.round(json.loads(ds.attrs['planet_params'])['cto'],5))
                        
            
            
            
            rs_arr = np.append(rs_arr,json.loads(ds.attrs['stellar_params'])['rs']['value'])
            temperature_grid[ct,:] = ds['temperature'].values[:]
            pressure_grid[ct,:] = ds['pressure'].values[:]
                

            
            try:
                fsed_arr = np.append(fsed_arr, json.loads(ds.attrs['cld_params'])['fsed'])
            except:
                fsed_arr = np.append(fsed_arr, 'Clear')
                
            
            try:
                logkzz_arr = np.append(logkzz_arr,json.loads(ds.attrs['planet_params'])['logkzz']['value'])
            except:
                logkzz_arr = np.append(logkzz_arr,json.loads(ds.attrs['planet_params'])['logkzz'])
                  
            #print(ds.attrs['planet_params'].get['logkzz'])
            #logkzz_arr = np.append(logkzz_arr,json.loads(ds.attrs['planet_params'])['logkzz']['value'])
            ms_arr = np.append(ms_arr,json.loads(ds.attrs['stellar_params'])['ms']['value'])
            
            pquench_arr = np.append(pquench_arr,"Not Included, Kzz instead")
            logg_arr = np.append(logg_arr,json.loads(ds.attrs['stellar_params'])['logg'])
            steff_arr = np.append(steff_arr,json.loads(ds.attrs['stellar_params'])['steff'])
            feh_arr = np.append(feh_arr,json.loads(ds.attrs['stellar_params'])['feh'])
                
            p_cloud_arr = np.append(p_cloud_arr,"Not Included")
            haze_eff_arr = np.append(haze_eff_arr,"Not Included")
                
            rainout_arr = np.append(rainout_arr,'T')
            
            
                
            
            ct+=1
            
            
    if Verbose == True:        
        print("Total Number of Models in your grid is", ct)       
        
    rp_grid = np.unique(rp_arr)
    mp_grid = np.unique(mp_arr)
    Tint_grid = np.unique(Tint_arr)
    heat_redis_grid = np.unique(heat_redis_arr)
    pref_grid = np.unique(pref_arr)
    mh_grid = np.unique(mh_arr)
    cto_grid = np.unique(cto_arr)
    rs_grid = np.unique(rs_arr)
    logkzz_grid = np.unique(logkzz_arr)
    pquench_grid = np.unique(pquench_arr)
    logg_grid = np.unique(logg_arr)
    steff_grid = np.unique(steff_arr)
    feh_grid = np.unique(feh_arr)
    ms_grid = np.unique(ms_arr)
    p_cloud_grid = np.unique(p_cloud_arr)
    haze_eff_grid = np.unique(haze_eff_arr)
    
    rainout_grid = np.unique(rainout_arr)
    fsed_grid = np.unique(fsed_arr)
    
    
    if grid_dimensions == True:
        if Verbose==True:
            print("Planet T_int Grid:",Tint_grid)
            print("Planet heat_distribution Grid:",heat_redis_grid)
            print("Planet P_ref Grid:",pref_grid)
            print("Planet Metallicity Grid:",mh_grid)
            print("Planet C/O Grid:",cto_grid)
            print("Planet logKzz Grid:",logkzz_grid)
            print("Planet fsed Grid:",fsed_grid)
            print("Planet P_quench Grid:",pquench_grid)
            print("Planet rainout Grid:",rainout_grid)
            print("Planet P_cloud Grid:",p_cloud_grid)
            print("Planet haze_eff Grid:",haze_eff_grid)
            
    
    
    return rp_arr,mp_arr,Tint_arr,heat_redis_arr,pref_arr,mh_arr,cto_arr,rs_arr,logkzz_arr,pquench_arr,logg_arr,steff_arr,feh_arr,ms_arr,p_cloud_arr,haze_eff_arr,opd_arr,ssa_arr,asy_arr,rainout_arr,wavelength,spectra_grid,temperature_grid,pressure_grid,fsed_arr,filename_arr


def fit_grid(grid,location,wlgrid_center,wlgrid_width,rprs_data2,e_rprs2,numparams,rp_arr,mp_arr,Tint_arr,heat_redis_arr,pref_arr,mh_arr,cto_arr,rs_arr,logkzz_arr,pquench_arr,logg_arr,steff_arr,feh_arr,ms_arr,p_cloud_arr,haze_eff_arr,opd_arr,ssa_arr,asy_arr,rainout_arr,wavelength,spectra,temperature,pressure,fsed,filename_arr):
    
        
    chi_sq_arr = np.zeros(shape=(len(Tint_arr)))
    spectra_bf = np.zeros(shape=(len(Tint_arr),len(wlgrid_center)))
    offset_arr = np.zeros(len(Tint_arr))
    numparams = numparams



    def shift_spectrum(waves,shift):
            return flux_in_bin+shift
    
    
    for index in range(len(Tint_arr)):

        #spec = convolve(spectra[index,:],Gaussian1DKernel(10))
        #flux_in_bin = [np.average(spec[np.where(np.logical_and(wavelength>=wlgrid_center[i]-wlgrid_width[i], wavelength<=wlgrid_center[i]+wlgrid_width[i]))]) for i in range(len(wlgrid_center))]

        xw , flux_in_bin = jdi.mean_regrid(wavelength,spectra[index,:],newx= wlgrid_center)

        popt, pcov = optimize.curve_fit(shift_spectrum, wlgrid_center, rprs_data2,p0=[-0.001])

        chi_sq_arr[index]= chi_squared(rprs_data2,e_rprs2,flux_in_bin+popt[0],numparams)

        spectra_bf[index,:] = flux_in_bin+popt[0]
        offset_arr[index] = popt[0]
            
   
       
    
    inds = chi_sq_arr.argsort()

    chi_sq_sort = chi_sq_arr[inds]
    Tint_sort = Tint_arr[inds]
    mh_sort = mh_arr[inds]
    cto_sort = cto_arr[inds]
    heat_redis_sort = heat_redis_arr[inds]
    logkzz_sort = logkzz_arr[inds]
    pquench_sort = pquench_arr[inds]
    p_cloud_sort = p_cloud_arr[inds]
    haze_eff_sort = haze_eff_arr[inds]
    rainout_sort = rainout_arr[inds]
    spectra_sort = spectra[inds,:]
    spectra_bf_sort = spectra_bf[inds,:]
    temperature_bf_sort = temperature[inds,:]
    pressure_bf_sort = pressure[inds,:]
    fsed_bf_sort = fsed[inds]
    filename_bf_sort = filename_arr[inds]
    offset_sort = offset_arr[inds]
    
    return chi_sq_sort,Tint_sort,mh_sort,cto_sort,heat_redis_sort,logkzz_sort,pquench_sort,p_cloud_sort,haze_eff_sort,rainout_sort,wavelength,spectra_sort,spectra_bf_sort,temperature_bf_sort,pressure_bf_sort,fsed_bf_sort,filename_bf_sort,offset_sort



def plot_best_fit(wlgrid_center,wlgrid_width,rprs_data2,e_rprs2,grid1=None,spectra_bf_1= None,chi1=None, grid2=None, spectra_bf_2=None,chi2=None,grid3=None, spectra_bf_3=None,chi3=None,grid4=None, spectra_bf_4=None,chi4=None,grid5=None, spectra_bf_5=None,chi5=None,grid6=None, spectra_bf_6=None,chi6=None,reduction_name=None):

    x='''
    AA
    ..
    BB
    '''
    fig = plt.figure(figsize=(18,10))
    plt.style.use('seaborn-paper')
    plt.rcParams['figure.figsize'] = [7, 4]           # Figure dimensions
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['image.aspect'] = 1.2                       # Aspect ratio (the CCD is quite long!!!)
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['lines.markersize'] = 3
    plt.rcParams['lines.markeredgewidth'] = 0
    
    cmap = plt.cm.magma
    #cmap.set_bad('k',1.)
    
    plt.rcParams['image.cmap'] = 'magma'                   # Colormap.
    plt.rcParams['image.interpolation'] = None
    plt.rcParams['image.origin'] = 'lower'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'DejaVu Serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams['axes.prop_cycle'] = \
    plt.cycler(color=["tomato", "dodgerblue", "gold", 'forestgreen', 'mediumorchid', 'lightblue'])
    plt.rcParams['figure.dpi'] = 300
    

    ax = fig.subplot_mosaic(x,gridspec_kw={
            # set the height ratios between the rows
            "height_ratios": [1,0.00001,0.2],
            # set the width ratios between the columns
            "width_ratios": [1,1]})



    ax['A'].set_xlim(np.min(wlgrid_center)-0.2,np.max(wlgrid_center)+0.5)
    ax['B'].set_xlim(np.min(wlgrid_center)-0.2,np.max(wlgrid_center)+0.5)
    #ax['A'].set_ylim(np.min(rprs_data2)-0.01*np.min(rprs_data2),np.max(rprs_data2)+0.01*np.max(rprs_data2))



    #if (np.max(wlgrid_center)-np.min(wlgrid_center)) > 4:
    #    ax['A'].semilogx(15,0)
    #    ax['B'].semilogx(15,0)
    #    ax['A'].set_xticks([0.3,0.5,0.7,1.0,2.0,3.0,4.0,5.0])
    #    ax['A'].set_xticklabels(['0.3','0.5','0.7','1.0','2.0','3.0','4.0','5.0'])
    #    ax['B'].set_xticks([0.3,0.5,0.7,1.0,2.0,3.0,4.0,5.0])
    #    ax['B'].set_xticklabels(['0.3','0.5','0.7','1.0','2.0','3.0','4.0','5.0'])

    ax['A'].errorbar(wlgrid_center,rprs_data2,yerr=e_rprs2,fmt="ko",label=reduction_name+" Reduction",markersize=5)
    if (grid1 != None) :
        ax['A'].plot(wlgrid_center,spectra_bf_1[0,:],"tomato",linewidth=2,label=r"Best Fit "+grid1+", ${\chi}_{\\nu}$$^2$= "+ str(np.round(chi1,2))) #Eq Chem, $\chi^2$= 4.37, T$_{int}$ = 200 K, mh= +0.5, C/O = 1.5$\times$solar")
    if (grid2 != None) :
        ax['A'].plot(wlgrid_center,spectra_bf_2[0,:],"dodgerblue",linewidth=2,label=r"Best Fit "+grid2+", ${\chi}_{\\nu}$$^2$= "+ str(np.round(chi2,2))) 
    if (grid3 != None) :
        ax['A'].plot(wlgrid_center,spectra_bf_3[0,:],"forestgreen",linewidth=2,label=r"Best Fit "+grid3+", ${\chi}_{\\nu}$$^2$= "+ str(np.round(chi3,2))) 
    if (grid4 != None) :
        ax['A'].plot(wlgrid_center,spectra_bf_4[0,:],"green",linewidth=2,label=r"Best Fit "+grid4+", ${\chi}_{\\nu}$$^2$= "+ str(np.round(chi4,2))) 
    if (grid5 != None) :
        ax['A'].plot(wlgrid_center,spectra_bf_5[0,:],"orchid",linewidth=2,label=r"Best Fit "+grid5+", ${\chi}_{\\nu}$$^2$= "+ str(np.round(chi5,2))) 
    if (grid6 != None) :
        ax['A'].plot(wlgrid_center,spectra_bf_6[0,:],"slateblue",linewidth=2,label=r"Best Fit "+grid6+", ${\chi}_{\\nu}^2$= "+ str(np.round(chi6,2))) 
    
    
    ax['B'].set_xlabel(r"Wavelength [$\mu$m]",fontsize=20)
    ax['A'].set_ylabel(r"(R$_{\rm p}$/R$_{*}$)$^2$",fontsize=20)

    ax['A'].minorticks_on()
    ax['A'].tick_params(axis='y',which='major',length =20, width=3,direction='in',labelsize=20)
    ax['A'].tick_params(axis='y',which='minor',length =10, width=2,direction='in',labelsize=20)
    ax['A'].tick_params(axis='x',which='major',length =20, width=3,direction='in',labelsize=20)
    ax['A'].tick_params(axis='x',which='minor',length =10, width=2,direction='in',labelsize=20)

    if (grid1 != None) :
        ax['B'].plot(wlgrid_center,(rprs_data2-spectra_bf_1[0,:])/e_rprs2,"o",color="tomato",markersize=5)
        ax['B'].plot(wlgrid_center,0*(rprs_data2-spectra_bf_1[0,:])/e_rprs2,"k")
    if (grid2 != None) :
        ax['B'].plot(wlgrid_center,(rprs_data2-spectra_bf_2[0,:])/e_rprs2,"o",color="dodgerblue",markersize=5)
    if (grid3 != None) :
        ax['B'].plot(wlgrid_center,(rprs_data2-spectra_bf_3[0,:])/e_rprs2,"o",color="forestgreen",markersize=5)
    if (grid4 != None) :
        ax['B'].plot(wlgrid_center,(rprs_data2-spectra_bf_4[0,:])/e_rprs2,"o",color="green",markersize=5)
    if (grid5 != None) :
        ax['B'].plot(wlgrid_center,(rprs_data2-spectra_bf_5[0,:])/e_rprs2,"o",color="orchid",markersize=5)
    if (grid6 != None) :
        ax['B'].plot(wlgrid_center,(rprs_data2-spectra_bf_6[0,:])/e_rprs2,"o",color="slateblue",markersize=5)
    

    ax['B'].minorticks_on()
    ax['B'].tick_params(axis='y',which='major',length =20, width=3,direction='in',labelsize=20)
    ax['B'].tick_params(axis='y',which='minor',length =10, width=2,direction='in',labelsize=20)
    ax['B'].tick_params(axis='x',which='major',length =20, width=3,direction='in',labelsize=20)
    ax['B'].tick_params(axis='x',which='minor',length =10, width=2,direction='in',labelsize=20)

    
    ax['B'].set_ylabel("${\delta}/N$",fontsize=20)
    
        
    ax['A'].legend(fontsize=12)
    
    
    return fig,ax


def get_posteriors(parameter_sort,chi_sq):
    parameter_grid =np.unique(parameter_sort)
    
    prob_array = np.exp(-chi_sq/2.0)
    alpha = 1.0/np.sum(prob_array)
    prob_array = prob_array*alpha
    
    prob= np.array([])

    for m in parameter_grid:
        wh = np.where(parameter_sort == m)
        prob = np.append(prob,np.sum(prob_array[wh]))
        
    return parameter_grid,prob


def plot_posteriors(Tint_sort1,mh_sort1,cto_sort1,heat_redis_sort1,fsed_sort1,kzz_sort1,chi_sq1,grid_name1=None,fig=None,ax=None,color=None):
    
    if fig == None:
        if ax == None:
            plt.style.use('seaborn-paper')
            plt.rcParams['figure.figsize'] = [7, 4]           # Figure dimensions
            plt.rcParams['figure.dpi'] = 300
            plt.rcParams['image.aspect'] = 1.2                       # Aspect ratio (the CCD is quite long!!!)
            plt.rcParams['lines.linewidth'] = 1
            plt.rcParams['lines.markersize'] = 3
            plt.rcParams['lines.markeredgewidth'] = 0

            cmap = plt.cm.magma
            #cmap.set_bad('k',1.)

            plt.rcParams['image.cmap'] = 'magma'                   # Colormap.
            plt.rcParams['image.interpolation'] = None
            plt.rcParams['image.origin'] = 'lower'
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = 'DejaVu Serif'
            plt.rcParams['mathtext.fontset'] = 'dejavuserif'
            plt.rcParams['axes.prop_cycle'] = \
            plt.cycler(color=["tomato", "dodgerblue", "gold", 'forestgreen', 'mediumorchid', 'lightblue'])
            plt.rcParams['figure.dpi'] = 300
            fig,ax = plt.subplots(nrows=2,ncols=3,figsize=(30,20))
    
    
    tint_grid,prob_tint = get_posteriors(Tint_sort1,chi_sq1)
    m_grid,prob_m = get_posteriors(mh_sort1,chi_sq1)
    cto_grid,prob_cto = get_posteriors(cto_sort1,chi_sq1)
    redis_grid,prob_heat_redis = get_posteriors(heat_redis_sort1,chi_sq1)
    fsed_grid,prob_fsed= get_posteriors(fsed_sort1,chi_sq1)
    kzz_grid,prob_kzz= get_posteriors(kzz_sort1,chi_sq1)
    
    


    ax[0,0].set_ylim(1e-2,1)
    ax[0,0].set_xlim(0,500)
    ax[0,0].bar(tint_grid, prob_tint,width=[100,100,100], color="None",edgecolor=color,linewidth=5,label=grid_name1)
    

    ax[0,1].set_ylim(1e-2,1)
    ax[0,1].set_xlim(np.min(np.log10(m_grid))-1,np.max(np.log10(m_grid))+1)
    
    ax[0,1].bar(np.log10(m_grid), prob_m,width=0.22, color="None",edgecolor=color,linewidth=5,label=grid_name1)
    
    
    
    ax[0,2].set_ylim(1e-6,1)
    ax[0,2].bar(cto_grid/0.458, prob_cto,cto_grid[1]/0.458-cto_grid[0]/0.458, color="None",edgecolor=color,linewidth=5,label=grid_name1)
    ax[0,2].set_xlim(0,2.5)
    
        
        

    
    ax[1,0].set_ylim(1e-6,1)
    ax[1,0].bar(redis_grid, prob_heat_redis,0.1, color="None",edgecolor=color,linewidth=5,label=grid_name1)
    ax[1,0].set_xlim(np.min(redis_grid)-0.2,np.max(redis_grid)+0.2)
    
    if fsed_grid[0] == 'None':
        
        ax[1,1].set_ylim(1e-6,1)
        ax[1,1].bar(redis_grid*0, prob_heat_redis*0,0.1, color="None",edgecolor=color,linewidth=5)
        ax[1,1].set_xlim(0,12)
    else:
        ax[1,1].set_ylim(1e-6,1)
        ax[1,1].bar(fsed_grid, prob_fsed,width=[0.3,0.7,1,1,1], color="None",edgecolor=color,linewidth=5,label=grid_name1)
        ax[1,1].set_xlim(0,12)
       
    if kzz_grid[0] == 'None':
        
        ax[1,2].set_ylim(1e-6,1)
        ax[1,2].bar(redis_grid*0, prob_heat_redis*0,0.1, color="None",edgecolor=color,linewidth=5)
        ax[1,2].set_xlim(4,12)
    else:
        ax[1,2].set_ylim(1e-6,1)
        ax[1,2].bar(kzz_grid, prob_kzz,width=1, color="None",edgecolor=color,linewidth=5,label=grid_name1)
        ax[1,2].set_xlim(4,12)
    
    

    ax[0,0].tick_params(axis='both',which='major',length =40, width=3,direction='in',labelsize=30)
    ax[0,0].tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=30)

    ax[0,1].tick_params(axis='both',which='major',length =40, width=3,direction='in',labelsize=30)
    ax[0,1].tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=30)

    ax[0,2].tick_params(axis='both',which='major',length =40, width=3,direction='in',labelsize=30)
    ax[0,2].tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=30)
    
    ax[1,0].tick_params(axis='both',which='major',length =40, width=3,direction='in',labelsize=30)
    ax[1,0].tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=30)

    ax[1,1].tick_params(axis='both',which='major',length =40, width=3,direction='in',labelsize=30)
    ax[1,1].tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=30)

    ax[1,2].tick_params(axis='both',which='major',length =40, width=3,direction='in',labelsize=30)
    ax[1,2].tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=30)
    

    ax[0,0].set_xlabel(r"T$_{\rm int}$ [K]",fontsize=50)
    ax[0,1].set_xlabel("log[M/H]",fontsize=50)
    ax[0,2].set_xlabel(r"C/O [$\times$Solar]",fontsize=50)
    ax[1,0].set_xlabel("Heat Redis",fontsize=50)
    ax[1,1].set_xlabel(r"f$_{\rm sed}$",fontsize=50)
    ax[1,2].set_xlabel(r"K$_{\rm zz}$",fontsize=50)

    ax[0,0].set_ylabel("Probability",fontsize=50)
    ax[1,0].set_ylabel("Probability",fontsize=50)
    
    ax[0,0].legend(fontsize=20)
    ax[0,1].legend(fontsize=20)
    ax[0,2].legend(fontsize=20)
    ax[1,0].legend(fontsize=20)
    ax[1,1].legend(fontsize=20)
    ax[1,2].legend(fontsize=20)
    
    return fig,ax


def chi_squared(data,data_err,model,numparams):
    
    chi_squared = np.sum(((data-model)/(data_err))**2)/(len(data)-numparams)
    
    return chi_squared

def plot_atmosphere(location,bf_filename,gas_names=None,fig=None,ax=None,linestyle=None,color=None,label=None):

    f = os.path.join(location, bf_filename)

        
        
    if os.path.isfile(f):
            
        ds = xr.open_dataset(f)

        temp = ds['temperature'].values[:]
        pressure = ds['pressure'].values[:]
        gas_vmr = np.zeros(shape=(len(gas_names),len(pressure)))

        try: # see if clouds are here
            pressure_cld = ds['pressure_cld'].values[:]
            wno_cld = ds['wno_cld'].values[:]
            wno_cld = ds['wno_cld'].values[:]
            wno_cld = ds['wno_cld'].values[:]
            opd_cld = ds['opd'].values
            asy_cld = ds['asy'].values
            ssa_cld = ds['ssa'].values
        except:
            pressure_cld = 0
            wno_cld = 0
            wno_cld = 0
            wno_cld = 0
            opd_cld = 0
            asy_cld = 0
            ssa_cld = 0

        for igas,gases in zip(range(0,len(gas_names)),gas_names):

            try: # see if clouds are here
                gas_vmr[igas,:]= ds[gases].values[:]
                
            except:
                gas_vmr[igas,:]= 0.0
        

        if fig == None:
            if ax == None:
                plt.style.use('seaborn-paper')
                plt.rcParams['figure.figsize'] = [7, 4]           # Figure dimensions
                plt.rcParams['figure.dpi'] = 300
                plt.rcParams['image.aspect'] = 1.2                       # Aspect ratio (the CCD is quite long!!!)
                plt.rcParams['lines.linewidth'] = 1
                plt.rcParams['lines.markersize'] = 3
                plt.rcParams['lines.markeredgewidth'] = 0

                cmap = plt.cm.get_cmap('tab20b', len(gas_names))
                #cmap.set_bad('k',1.)

                plt.rcParams['image.cmap'] = 'magma'                   # Colormap.
                plt.rcParams['image.interpolation'] = None
                plt.rcParams['image.origin'] = 'lower'
                plt.rcParams['font.family'] = 'serif'
                plt.rcParams['font.serif'] = 'DejaVu Serif'
                plt.rcParams['mathtext.fontset'] = 'dejavuserif'
                plt.rcParams['axes.prop_cycle'] = \
                plt.cycler(color=["tomato", "dodgerblue", "gold", 'forestgreen', 'mediumorchid', 'lightblue'])
                plt.rcParams['figure.dpi'] = 300
                fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(30,10))
        
        cmap = plt.cm.get_cmap('tab20b', len(gas_names))


        ax[0].set_ylim(1000,1e-6)
        ax[0].set_xlim(500,3500)

        ax[0].semilogy(temp,pressure,linewidth=3,linestyle=linestyle,color=color,label=label)
        ax[0].minorticks_on()
        ax[0].tick_params(axis='both',which='major',length =40, width=3,direction='in',labelsize=30)
        ax[0].tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=30)
        ax[0].legend(fontsize=15)
        ax[0].set_xlabel(r"Temperature [K]",fontsize=30)
        ax[0].set_ylabel(r"Pressure [Bars]",fontsize=30)

        ax[1].set_ylim(1000,1e-6)
        ax[1].set_xlim(1e-8,1)
        
        
        #ax[1].set_ylabel(r"Pressure [Bars]",fontsize=50)


        for igas,gases in zip(range(0,len(gas_names)),gas_names):
            
            ax[1].loglog(gas_vmr[igas,:],pressure,linewidth=3,linestyle=linestyle,color=cmap(igas),label=gases+" "+label)
            
        
        ax[1].minorticks_on()
        ax[1].tick_params(axis='both',which='major',length =40, width=3,direction='in',labelsize=30)
        ax[1].tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=30)
        ax[1].set_xlabel(r"VMR",fontsize=30)
        
        
        ax[1].legend(fontsize=12)


        ax[2].set_ylim(1000,1e-6)
        ax[2].set_xlim(1e-5,500)

        
        #ax[2].set_ylabel(r"Pressure [Bars]",fontsize=50)
        if np.sum(opd_cld) != 0:
            ax[2].loglog(opd_cld[:,150],pressure_cld,linewidth=2,linestyle=linestyle,color=color,label=label)
        
        ax[2].minorticks_on()
        ax[2].tick_params(axis='both',which='major',length =40, width=3,direction='in',labelsize=30)
        ax[2].tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=30)
        ax[2].set_xlabel(r"Cloud OPD (1 ${\mu}$m)",fontsize=30)
        ax[2].legend(fontsize=15)

    else:
        print("Filename or directory is not correct.")
        fig,ax=0


    
    
    # function to show best fit atmospheric abundances
    
    return fig,ax



def plot_contribution(mass,radius,T_st,met_st,logg_st,radius_st,opa,location,bf_filename,offset_mp,wlgrid_center,rprs_data2,e_rprs2,gas_contribution=None,fig=None,ax=None):
    # function to show best fit spectrum
    # and contribution of each gas to the best fit spectrum

    gas_names = [ 'e-','H2','He',  'H', 'H+', 'H-',
       'H2-', 'H2+', 'H3+', 'H2O', 'CH4', 'CO', 'NH3', 'N2', 'PH3',
       'H2S', 'TiO', 'VO', 'Fe', 'FeH', 'CrH', 'Na', 'K', 'Rb', 'Cs', 'CO2',
       'HCN', 'C2H2', 'C2H4', 'C2H6', 'SiO', 'MgH', 'OCS', 'Li', 'LiOH', 'LiH',
       'LiCl'] 

    f = os.path.join(location, bf_filename)

        
        
    if os.path.isfile(f):
            
        ds = xr.open_dataset(f)

        temp = ds['temperature'].values[:]
        pressure = ds['pressure'].values[:]
        gas_vmr = np.zeros(shape=(len(gas_names),len(pressure)))
        spectra_mp = ds['transit_depth'].values[:]
        wvl_mp = ds['wavelength'].values[:]
        
        try: # see if clouds are here
            pressure_cld = ds['pressure_cld'].values[:]
            wno_cld = ds['wno_cld'].values[:]
            wno_cld = ds['wno_cld'].values[:]
            wno_cld = ds['wno_cld'].values[:]
            opd_cld = ds['opd'].values
            asy_cld = ds['asy'].values
            ssa_cld = ds['ssa'].values
        except:
            pressure_cld = 0
            wno_cld = 0
            wno_cld = 0
            wno_cld = 0
            opd_cld = 0
            asy_cld = 0
            ssa_cld = 0

        for igas,gases in zip(range(0,len(gas_names)),gas_names):

            try: # see if clouds are here
                gas_vmr[igas,:]= ds[gases].values[:]
                
            except:
                gas_vmr[igas,:]= 0.0
        

        if fig == None:
            if ax == None:
                plt.style.use('seaborn-paper')
                plt.rcParams['figure.figsize'] = [7, 4]           # Figure dimensions
                plt.rcParams['figure.dpi'] = 300
                plt.rcParams['image.aspect'] = 1.2                       # Aspect ratio (the CCD is quite long!!!)
                plt.rcParams['lines.linewidth'] = 1
                plt.rcParams['lines.markersize'] = 3
                plt.rcParams['lines.markeredgewidth'] = 0

                cmap = plt.cm.get_cmap('tab20b', len(gas_names))
                #cmap.set_bad('k',1.)

                plt.rcParams['image.cmap'] = 'magma'                   # Colormap.
                plt.rcParams['image.interpolation'] = None
                plt.rcParams['image.origin'] = 'lower'
                plt.rcParams['font.family'] = 'serif'
                plt.rcParams['font.serif'] = 'DejaVu Serif'
                plt.rcParams['mathtext.fontset'] = 'dejavuserif'
                plt.rcParams['axes.prop_cycle'] = \
                plt.cycler(color=["tomato", "dodgerblue", "gold", 'forestgreen', 'mediumorchid', 'lightblue'])
                plt.rcParams['figure.dpi'] = 300
                fig,ax = plt.subplots(nrows=1,figsize=(18,10))
    
    
    
    

    
    ax.set_xlim(np.min(wlgrid_center)-0.2,np.max(wlgrid_center)+0.5)
    ax.errorbar(wlgrid_center,rprs_data2,yerr=e_rprs2,fmt="ko",label="Data",markersize=5)

    
    one_by_gas = np.zeros(shape=(len(gas_contribution),len(wlgrid_center)))
    
    count=0
    
    data_atm = {'pressure': pressure,
        'temperature': temp}
  
    # Convert the dictionary into DataFrame
    df_atm = pd.DataFrame(data_atm)
    
    def shift_spectrum(waves,shift):
            return y+shift
    for igas,gases in zip(range(0,len(gas_names)),gas_names):
    
        df_atm[gases] = gas_vmr[igas,:]
       
    case1 = jdi.inputs()
    case1.approx(p_reference=10)
    case1.phase_angle(0)
    case1.gravity(mass=mass, mass_unit=jdi.u.Unit('M_jup'),
                  radius=radius, radius_unit=jdi.u.Unit('R_jup'))

    T_st,met_st,logg_st,radius_st = T_st,met_st,logg_st,radius_st

    case1.star(opa, T_st,met_st,logg_st,radius=radius_st, radius_unit = jdi.u.Unit('R_sun') )

    case1.atmosphere( df =df_atm)

    if np.sum(opd_cld) != 0:
        df_cld= vj.picaso_format(opd_cld, ssa_cld, asy_cld)
    
        case1.clouds(df=df_cld)
        #case1.inputs['clouds']['wavenumber'] = wno_cld 

    df_one_gas= case1.spectrum(opa, full_output=True,calculation=['transmission']) #note the new last key

    x,y = jdi.mean_regrid(1e4/df_one_gas['wavenumber'],df_one_gas['transit_depth'],newx=wlgrid_center)
    
    mmw_old = np.copy(df_one_gas['full_output']['layer']['mmw'])
    

    ax.plot(x,y+offset_mp,linewidth=5,label="Best Fit Complete")
    
    
    count=0
    for gases in gas_contribution:
        
        opa = jdi.opannection(wave_range=[0.49,5.7])
        
        def shift_spectrum(waves,shift):
            return y+shift
        
        case1 = jdi.inputs()
        case1.approx(p_reference=10)
        case1.phase_angle(0)
        case1.gravity(mass=mass, mass_unit=jdi.u.Unit('M_jup'),
                      radius=radius, radius_unit=jdi.u.Unit('R_jup'))
        
        T_st,met_st,logg_st,radius_st = T_st,met_st,logg_st,radius_st
        
        case1.star(opa, T_st,met_st,logg_st,radius=radius_st, radius_unit = jdi.u.Unit('R_sun') )
        
        #case1.atmosphere( df =df_atm)
        case1.atmosphere( df = df_atm,exclude_mol=[gases])
        df_one_gas= case1.spectrum(opa, full_output=True,calculation=['transmission']) #note the new last key
        
        mmw_new = df_one_gas['full_output']['layer']['mmw']
        
        df_temp = df_atm.copy()
        df_temp['temperature'] = df_atm['temperature']*(np.mean(mmw_new)/np.mean(mmw_old))
        
        case1.atmosphere( df = df_temp,exclude_mol=[gases])
        df_one_gas= case1.spectrum(opa, full_output=True,calculation=['transmission']) #note the new last key

        x,y = jdi.mean_regrid(1e4/df_one_gas['wavenumber'],df_one_gas['transit_depth'],newx=wlgrid_center)
        
        
        #popt, pcov = optimize.curve_fit(shift_spectrum, wlgrid_center, rprs_data2)
        
        ax.plot(x,y+offset_mp,label="No "+gases,linewidth=2)
        
        count+=1

    if np.sum(opd_cld) != 0:

        case1 = jdi.inputs()
        case1.approx(p_reference=10)
        case1.phase_angle(0)
        case1.gravity(mass=mass, mass_unit=jdi.u.Unit('M_jup'),
                    radius=radius, radius_unit=jdi.u.Unit('R_jup'))

        T_st,met_st,logg_st,radius_st = T_st,met_st,logg_st,radius_st

        case1.star(opa, T_st,met_st,logg_st,radius=radius_st, radius_unit = jdi.u.Unit('R_sun') )

        case1.atmosphere( df =df_atm)

        df_one_gas= case1.spectrum(opa, full_output=True,calculation=['transmission']) #note the new last key

        x,y = jdi.mean_regrid(1e4/df_one_gas['wavenumber'],df_one_gas['transit_depth'],newx=wlgrid_center)
        
        mmw_old = np.copy(df_one_gas['full_output']['layer']['mmw'])
        

        ax.plot(x,y+offset_mp,linewidth=4,label="No Clouds",linestyle="--",color="k")

        
    
    ax.legend(fontsize=20,ncol=3)
    
    
    
    
    
    
    
    #ax.legend(fontsize=20)
    ax.set_xlabel(r"Wavelength [$\mu$m]",fontsize=20)
    ax.set_ylabel(r"(R$_{\rm p}$/R$_{*}$)$^2$",fontsize=20)


    ax.minorticks_on()
    ax.tick_params(axis='y',which='major',length =20, width=3,direction='in',labelsize=20)
    ax.tick_params(axis='y',which='minor',length =10, width=2,direction='in',labelsize=20)
    ax.tick_params(axis='x',which='major',length =20, width=3,direction='in',labelsize=20)
    ax.tick_params(axis='x',which='minor',length =10, width=2,direction='in',labelsize=20)

    plt.show()

    return fig,ax

