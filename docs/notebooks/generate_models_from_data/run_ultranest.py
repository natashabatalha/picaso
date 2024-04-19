import math, os
import multiprocessing as mp
import numpy as np
import pickle
import pandas as pd
import astropy.units as u
from bokeh.plotting import figure, show, output_file,save
import time
import picaso.justdoit as jdi
import picaso.justplotit as jpi
import uuid

import ultranest
import ultranest.stepsampler

import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

#command line 
#mpiexec -n numprocs python -m mpi4py pyfile
#for example: mpiexec -n 5 python -m mpi4py run_pymulti.py
#jupyter notebook: %run -i 'run_ultranest.py'


#Here is your main file with all priors, and model sets
from model_setup import *


#1) load data always {'name':(wno_data, y_data, err_data)}
nir_options = ['Gressier','Wakeford']
miri_options = ['bristol_dv','Cornell_Eureka!']
data_file = nir_options[1]+miri_options[0]
data_dict = get_data(nir_options[1],miri_options[0])

#2) specify tag name to help you keep track of the run 
tag = nir_options[1]+miri_options[0]

#3) where do you want to save your results 
out_dir = '/home/ddoud1/Sagan Workshop/tutorial_test1/emissions/'

#4) Specify models to test
model_type = 'cld_free'

#5) define loglikelihood
def loglike(cube):#, ndim, nparams):
    #compute model spectra
    resultx,resulty = MODEL(cube)

    #i check for nans or infinite values 
    if isinstance(resulty, float): 
        if np.isinf(resulty): 
            return -1e8 #very large negative number 
    
    #regrid to data wavenumber grid
    y_model_all = []
    x_data_all = []
    y_data_all = []
    e_data_all = []

    #loop through data and regird 
    for idata in data_dict.keys():
        x_chunk, y_chunk=jdi.mean_regrid(resultx, resulty, newx=data_dict[idata][0])
        y_model_all += [y_chunk]
        x_data_all += [x_chunk]
        y_data_all += [data_dict[idata][1]]
        e_data_all += [data_dict[idata][2]]
        
    y_model_all = np.concatenate(y_model_all)    
    x_data_all = np.concatenate(x_data_all)    
    y_data_all = np.concatenate(y_data_all)    
    e_data_all = np.concatenate(e_data_all)    
    
    #compute loglikelihood
    loglikelihood=-0.5*np.sum((y_data_all-y_model_all)**2/e_data_all**2)
    return loglikelihood





#6 ultranest kwargs 
multi_kwargs = {'resume':True,#'resume-similar',
                'warmstart_max_tau':-1,#0.7, #only used for resume-similar (small changes in likelihood. 0=very conservative, 1=very negligent) 
                'n_live_points':'50*nparam',
                'max_ncalls':None}#1000000}

#7 Run Ultranest
# this is one method of running ultranest, and in reality there are many more (see ultranest docs) 
print(model_type)
params = getattr(param_set, model_type).split(',')
Nparam=len(params)
print(params, Nparam)
MODEL = getattr(model_set, model_type)
PRIOR = getattr(prior_set, model_type)
if isinstance(multi_kwargs['n_live_points'],str):
    multi_kwargs['n_live_points']= (Nparam)*int(multi_kwargs['n_live_points'].split('*')[0])
jdi.json.dump({
        'tag':tag, 
        'data_file':data_file, 
        'retrieval_type': model_type,
        'nparams': Nparam, 
        'params':params, 
        'n_live_points': multi_kwargs['n_live_points'], 
        'max_ncalls':multi_kwargs['max_ncalls']
    } , open(f'{out_dir}/{tag}_{model_type}.json','w'))#_{unique_id}

sampler = ultranest.ReactiveNestedSampler(
                        params,
                        loglike,
                        PRIOR,
                        log_dir=out_dir, 
resume=multi_kwargs['resume'],
                        warmstart_max_tau=multi_kwargs['warmstart_max_tau'])
nsteps = 2 * len(params)
sampler.stepsampler = ultranest.stepsampler.SliceSampler(
                    nsteps=nsteps, generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
                )
results = sampler.run(min_num_live_points=multi_kwargs['n_live_points'], max_ncalls=multi_kwargs['max_ncalls'])