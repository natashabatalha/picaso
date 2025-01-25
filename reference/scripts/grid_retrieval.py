##!/usr/bin/env python
# coding: utf-8

# Author: Natasha E Batalha 

# PICASO Retrieval Script Template for fitting a grid model to a spectrum

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
    returns = {'MIRI_LRS': [final['wavenumber'].values, #CHANGEME 
             final['transit_depth'].values  ,final['transit_depth_error'].values]   }
    return returns


# ## Step 2) Load Grid 

grid_location = grid_locationCHANGEME # CHANGEME should ultimately point to location of all .nc files
grid_name = grid_nameCHANGEME #CHANGEME for your own book-keeping
to_fit = to_fitCHANGEME # CHANGEME to the quantity you are fitting based on what is in the xarray (options are usually transit_depth, flux)

fitter = lyz.GridFitter(grid_name,grid_location, verbose=True, 
    to_fit=to_fit, save_chem=True)
fitter.prep_gridtrieval(grid_name)


# ## Step 3) Param Set
grid_parameters_unique = fitter.interp_params[grid_name]['grid_parameters_unique']
class param_set:
    """
    This is much easier now since it is the grid parameters plus an offset to account for unknown reference pressure
    """
    #CHANGEME: are there other parameters you need to fit for, such as error inflation?
    grid = list(grid_parameters_unique.keys())+['offset']


# ## Step 4) Guesses Set
class guesses_set: 
    """
    For our guesses, we can verify things are working by just taking the first instance of each grid point
    """
    #completely random guesses just to make sure it runs
    grid=[grid_parameters_unique[i][0]
             for i in grid_parameters_unique.keys()] + [0]


# ## Step 5) Model Set

class model_set:
    """
    There are several different ways to interpolate onto a grid. Here, we use picaso's fast custom interpolator that 
    uses a flex linear interpolator and can be used on any type of matrix grid. 
    """     
    def grid(cube): 
        """Simple grid interpolation + offset for spectra
        """
        final_goal = cube[0:len(grid_parameters_unique.keys())]
        spectra_interp = lyz.custom_interp(final_goal, fitter, grid_name)
        micron = fitter.wavelength[grid_name]
        wno = 1e4/fitter.wavelength[grid_name] 
        ##CHANGEME MIRI_LRS should match the instrument dictionary defined in get_data
        offset={'MIRI_LRS':cube[-1]} #remember before it helps always to have the same model returns
        error_inf={} # let's not add error inf 
        return wno, spectra_interp,offset,error_inf


# ## Step 6) Prior Set

class prior_set:
    """
    Store all your priors. You should have the same exact function names in here as
    you do in model_set and param_set

    Make sure the order of the unpacked cube follows the unpacking in your model 
    set and in your parameter set. 
    #pymultinest: http://johannesbuchner.github.io/pymultinest-tutorial/example1.html
    """   
    def grid(cube):
        params = cube.copy()
        for i,key in enumerate(grid_parameters_unique): 
            minn = np.min(grid_parameters_unique[key]) 
            maxx = np.max(grid_parameters_unique[key]) 
            params[i] = minn + (maxx-minn)*params[i]
        #the offset parameter here let's allow these spectra to move up and down by ~1000pm. This is a guess so let's check firts! 
        #CHANGEME : Define your own priors based on the data. 
        minn=-.008;maxx=-.015;
        i+=1;params[i] = minn + (maxx-minn)*params[i]
        return params


# ## Step 7) Loglikelihood
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

        #add offsets if they exist
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

    # CHANGEME based on what model you want to run 
    MODEL_TO_RUN = 'grid'
    ultranest_output= sampler_output_pathCHANGEME  # point to your own directory

    #we can easity grab all the important pieces now that they are neatly stored in a class structure 
    DATA_DICT = get_data()
    PARAMS = getattr(param_set,MODEL_TO_RUN)
    MODEL = getattr(model_set,MODEL_TO_RUN)
    PRIOR = getattr(prior_set,MODEL_TO_RUN)
    GUESS = getattr(guesses_set,MODEL_TO_RUN)


    sampler = ultranest.ReactiveNestedSampler(PARAMS, loglikelihood, PRIOR,
        log_dir = ultranest_output,resume=True)
    result = sampler.run()


