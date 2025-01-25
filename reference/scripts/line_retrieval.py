#!/usr/bin/env python
# coding: utf-8

# Author: Natasha E Batalha 

# PICASO Retrieval Script Template for fitting a line to a spectrum

# CTRL-F "CHANGEME" for areas you should consider changing
# IF CHANGEME is commented out then it is just a suggestion 
# IF CHANGEME is not commented you must change before proceeding 

# Use this as a starting point for create a retrieval script 

# HOW to run with OpenMPI

# Running with mpiexec command line 

# mpiexec -n numprocs python -m mpi4py pyfile
# for example: (-n specifies number of jobs)
# mpiexec -n 5 python -m mpi4py run.py

import numpy as np
import os 
import picaso.justdoit as jdi
import picaso.analyze as lyz
import xarray as xr


# ## Step 1) Develop function to get data

def get_data(): 
    """
    Create a function to process your data in any way you see fit.
    Here we are using the ExoTiC-MIRI data 
    https://zenodo.org/records/8360121/files/ExoTiC-MIRI.zip?download=1
    But no need to download it.


    Checklist
    ---------
    - your function returns a spectrum that will be in the same units as your picaso model (e.g. rp/rs^2, erg/s/cm/cm or other) 
    - your function retuns a spectrum that is in ascending order of wavenumber 
    - your function returns a dictionary where the key specifies the instrument name (in the event there are multiple)

    Returns
    -------
    dict: 
        dictionary key: wavenumber (ascneding), flux or transit depth, and error.
        e.g. {'MIRI LRS':[wavenumber, transit depth, transit depth error], 'NIRSpec G395H':[wavenumber, transit depth, transit depth error]}
    """
    #CHANGEME add your own data here 
    dat = xr.load_dataset(jdi.w17_data())
    #build nice dataframe so we can easily 
    final = jdi.pd.DataFrame(dict(wlgrid_center=dat.coords['central_wavelength'].values,
                transit_depth=dat.data_vars['transit_depth'].values,
                transit_depth_error=dat.data_vars['transit_depth_error'].values))

    #create a wavenumber grid 
    final['wavenumber'] = 1e4/final['wlgrid_center']

    #always ensure we are ordered correctly
    final = final.sort_values(by='wavenumber').reset_index(drop=True)

    #return a nice dictionary with the info we need 
    # CHANGEME Consider adding your own naming scheme for your data
    returns = {'MIRI_LRS': [final['wavenumber'].values, 
             final['transit_depth'].values  ,final['transit_depth_error'].values]   }
    return returns


# ## Step 2) Define Free Parameters
class param_set:
    """
    This is for book keeping what parameters you have run in each retrieval.
    It helps if you keep variables uniform.
    
    Checklist
    ---------
    - Make sure that the order of variables here match how you are unpacking your cube in the model_set class and prior_set
    - Make sure that the variable names here match the function names in model_set and prior_set
    """
    # CHANGEME you will want to add your own model
    line=['m','b','log_err_inf'] 


# ## Step 3) Define Initial Guesses
class gesses_set: 
    """
    Optional! 

    Tips
    ----
    - Usually you might have some guess (however incorrect) of what the answer might be. You can use this in the testing phase!
    """
    # CHANGEME you will want to add your own sensible guess 
    line=[0,0.016633,-1,1] #here I am guessing a zero slope, and the measured transit depth reported from exo.mast, and a small error inflation term


# ## Step 4) Define Model Set

class model_set:
    """
    This is your full model set. It will include all the functions you want to test
    for a particular data set.

    Tips
    ----
    - if you keep the structure of all your returns identically you will thank yourself later. 
      For exmaple, below I always return x,y,dict of instrument offsets,dict of error inflation, if exists

    Checklist
    ---------
    - unpacking the cube should unpack the parameters you have set in your param_set class. I like to use 
    list indexing with strings so I dont have to rely on remembering a specific order
    """     
    # CHANGEME you will want to create your own model
    #note, you have downloaded the simplest template "line"
    #Other options include: "grid only", "grid+clouds", "free"
    def line(cube): 
        wno_grid = np.linspace(600,3000,int(1e4)) #in the future this will be defined by the picaso opacity db
        m = cube[param_set.line.index('m')] 
        b = cube[param_set.line.index('b')] 
        err_inf = {'MIRI_LRS':10**cube[param_set.line.index('log_err_inf')] }
        y = m*wno_grid + b 
        offsets = {} #I like to keep the returns of all my model sets the same 
        return wno_grid,y,offsets,err_inf


# ## Step 5) Define Prior Set

class prior_set:
    """
    Store all your priors. You should have the same exact function names in here as
    you do in model_set and param_set

    Checklist
    ---------
    - Make sure the order of the unpacked cube follows the unpacking in your model 
      set and in your parameter set. 
    """   
    # CHANGEME you will want to create your own set of priors 
    def line(cube):#,ndim, nparams):
        params = cube.copy()
        #slope min max
        min = -1e-5
        max = 1e-5
        i=0;params[i] = min + (max-min)*params[i];i+=1
        #intercept min max
        min = 0.015
        max = 0.02
        params[i] = min + (max-min)*params[i];i+=1
        #log err inflation min max 
        min = -10
        max = 3
        params[i] = min + (max-min)*params[i];i+=1
        return params                


# ## Step 6) Define Likelihood Function

def loglikelihood(cube):
    """
    Log_likelihood function that ultimately is given to the sampler
    Note if you keep to our same formats you will not have to change this code move 

    Tips
    ----
    - Remember how we put our data dict, error inflation, and offsets all in dictionary format? Now we can utilize that 
    functionality if we properly named them all with the right keys! 

    Checklist
    --------- 
    - ensure that error inflation and offsets are incorporated in the way that suits your problem 
    - note there are many different ways to incorporate error inflation! this is just one example 
    """
    #compute model spectra
    resultx,resulty,offset_all,err_inf_all = MODEL(cube) # we will define MODEL below 

    #initiate the four terms we willn eed for the likelihood
    ydat_all=[];ymod_all=[];sigma_all=[];extra_term_all=[];

    #loop through data (if multiple instruments, add offsets if present, add err inflation if present)
    for ikey in DATA_DICT.keys(): #we will also define DATA_DICT below
        xdata,ydata,edata = DATA_DICT[ikey]
        xbin_model , y_model = jdi.mean_regrid(resultx, resulty, newx=xdata)#remember we put everything already sorted on wavenumber

        #add offsets if they exist to the data
        offset = offset_all.get(ikey,0) #if offset for that instrument doesnt exist, return 0
        ydata = ydata+offset 

        #add error inflation if they exist
        err_inf = err_inf_all.get(ikey,0) #if err inf term for that instrument doesnt exist, return 0
        sigma = edata**2 + (err_inf)**2 #there are multiple ways to do this, here just adding in an extra noise term
        if err_inf !=0: 
            #see formalism here for example https://emcee.readthedocs.io/en/stable/tutorials/line/#maximum-likelihood-estimation
            extra_term = np.log(2*np.pi*sigma)
        else: 
            extra_term=sigma*0

        ydat_all.append(ydata);ymod_all.append(y_model);sigma_all.append(sigma);extra_term_all.append(extra_term); 

    ymod_all = np.concatenate(ymod_all)    
    ydat_all = np.concatenate(ydat_all)    
    sigma_all = np.concatenate(sigma_all)  
    extra_term_all = np.concatenate(extra_term_all)

    #compute likelihood
    loglike = -0.5*np.sum((ydat_all-ymod_all)**2/sigma_all + extra_term_all)
    return loglike


if __name__ == "__main__":
    import ultranest
    # CHANGEME based on what model you want to run 
    MODEL_TO_RUN = 'line'
    ultranest_output= sampler_output_pathCHANGEME # CHANGEME 

    DATA_DICT = get_data()
    PARAMS = getattr(param_set,MODEL_TO_RUN)
    MODEL = getattr(model_set,MODEL_TO_RUN)
    PRIOR = getattr(prior_set,MODEL_TO_RUN)
    GUESS = getattr(guesses_set,MODEL_TO_RUN)

    
    sampler = ultranest.ReactiveNestedSampler(PARAMS, loglikelihood, PRIOR,
        log_dir = ultranest_output,resume=True)
    result = sampler.run()

