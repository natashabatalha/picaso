import numpy as np
import json 
from astropy.utils.misc import JsonCustomEncoder
from astropy.convolution import convolve, Gaussian1DKernel
import arviz as az
import pandas as pd
from scipy import interpolate
from scipy.stats import gaussian_kde

from .justdoit import mean_regrid,vj,u,get_cld_input_grid,special

from .analyze import chi_squared
from .justplotit import pals

import matplotlib.pyplot as plt
import xarray as xr
import os 
import pickle as pk
import re


class _QuantileLine:
    def __init__(self, data):
        self.data = data


class PredictionBand:
    """Small sampler-independent replacement for UltraNest's PredictionBand."""

    def __init__(self, x):
        self.x = np.asarray(x)
        self._values = []

    def add(self, value):
        self._values.append(np.asarray(value))

    def get_line(self, q):
        if not self._values:
            raise ValueError("Cannot calculate a prediction band without samples")
        return _QuantileLine(np.quantile(np.asarray(self._values), q, axis=0))


def _resample_equal(samples, weights, seed=0):
    """Systematically resample weighted samples without requiring Dynesty."""
    samples = np.asarray(samples)
    weights = np.asarray(weights, dtype=float)
    total = np.sum(weights)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Posterior weights must have a finite, positive sum")
    weights = weights / total
    rng = np.random.RandomState(seed)
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    indices = np.searchsorted(np.cumsum(weights), positions, side="right")
    return samples[np.minimum(indices, len(weights) - 1)]

__refdata__ = os.environ.get('picaso_refdata')

## BEGIN ALL RETR ANALYSIS TOOLS 
def log_parameters(fitpars):
    """
    Parameters whose prior is sampled in log10 space (log = true in the retrieval toml). 
    The samples of these parameters are stored as 10**x, see driver.hypercube.

    Parameters 
    ----------
    fitpars : dict 
        output of driver.prior_finder(config['retrieval'])

    Returns 
    -------
    list of str
    """
    return [key for key, prior in fitpars.items() if prior.get('log') in (True, 'True', 'true')]

def read_retrievals(dirr, params, log_params=None, code=None):
    """
    Function to parse dynesty, ultranest, and nautilus results.

    Parameters 
    ----------
    dirr : str
        directory of the output (contains dynesty.save, nautilus.hdf5, or ultranest files)
    params : list of str or dict
        list of parameter names, or the output of driver.prior_finder(config['retrieval']) 
        in which case the log sampled parameters are found automatically. Nautilus outputs 
        require the dict, since nautilus stores samples in the unit cube and the priors 
        are needed to transform them back to physical values 
    log_params : list of str 
        parameters that were sampled in log10 space. Stored in the output as 'log_params' 
        so that plot_pair can plot them in log space. Default = None, which uses params if 
        it is a dict, and otherwise assumes no log sampled parameters 
    code : str 
        sampler that produced the output (dynesty, ultranest, or nautilus). Default = None, 
        which detects it from the files in dirr (dynesty.save, then nautilus.hdf5, otherwise ultranest)

    Returns 
    -------
    dict
    """
    fitpars = params if isinstance(params, dict) else None
    if fitpars is not None:
        if log_params is None:
            log_params = log_parameters(fitpars)
        params = list(fitpars.keys())

    if code is None:
        if os.path.exists(os.path.join(dirr, 'dynesty.save')):
            code = 'dynesty'
        elif os.path.exists(os.path.join(dirr, 'nautilus.hdf5')):
            code = 'nautilus'
        else:
            code = 'ultranest'
    code = code.lower()

    if code == 'dynesty':
        info = read_dynesty(dirr, params)
    elif code == 'nautilus':
        if fitpars is None:
            raise ValueError('Reading nautilus results requires params to be the output of '
                             "driver.prior_finder(config['retrieval']) so that the unit cube "
                             'samples can be transformed with the priors')
        info = read_nautilus(dirr, fitpars)
    elif code == 'ultranest':
        info = read_ultranest(dirr, params)
    else:
        raise ValueError(f'Unknown sampler code {code}. Options are dynesty, ultranest, nautilus')
    info['log_params'] = [] if log_params is None else list(log_params)
    return info
    
def read_dynesty(dirr,params):
    """
    Function to parse ultranest results. Returns dictionary with 
    - samples_equal : equally weighted samples that can be directly used for corner plots, finding quantiles, etc 
    - max_logl : Bayesian Evidence - the lnZ of the max likelihood point 
    - max_logl_point : the values of associated with the max likelihood point 
    - med_intervals : median, errlo, errup of all the constrained parameters 
    - param_names : list string of the parameter names 
    - ultranest_out : all the raw output of ultranest 

    Parameters 
    ----------
    dirr : str
        directory of the ultranest output 
    params : int
        number of parameters 

    Returns 
    -------
    dict 
    """
    try:
        import dynesty
    except ImportError as exc:
        raise ImportError(
            "Reading Dynesty results requires the optional Dynesty package. "
            "Install it with `pip install 'picaso[dynesty]'`."
        ) from exc

    checkpoint_file = os.path.join(dirr, 'dynesty.save')
    sampler = dynesty.NestedSampler.restore(checkpoint_file)
    results = sampler.results
    samples = results.samples
    logwt = results.logwt
    logz = results.logz
    logl = results.logl
    
    # Calculate weights and resample equal
    weights = np.exp(logwt - logz[-1])
    # Ensure weights sum to 1 to avoid possible NaN/Inf issues
    sum_weights = np.sum(weights)
    if sum_weights > 0:
        weights = weights / sum_weights
        
    samples_rew = _resample_equal(samples, weights)
    
    max_logl_idx = np.argmax(logl)
    max_logl_point = samples[max_logl_idx]
    max_logl_value = logl[max_logl_idx]
    
    errlo = pd.DataFrame(samples_rew, columns=params).quantile(.158655)
    errup = pd.DataFrame(samples_rew, columns=params).quantile(.841345)
    median = pd.DataFrame(samples_rew, columns=params).quantile(0.50)
    summary = {}
    for i in params:
        summary[i+'_errlo'] = errlo[i]
        summary[i+'_errup'] = errup[i]
        summary[i+'_median'] = median[i]
    summary = pd.DataFrame(summary, index=[0])
    eval_at_med = [summary[i+'_'+'median'].values[0] for i in params]
    
    return {
        'samples_equal': samples_rew,
        'max_logl': max_logl_value,
        'max_logl_point': max_logl_point,
        'med_point': eval_at_med,
        'med_intervals': summary,
        'param_names': params,
        'dynesty_out': results
    }

def read_ultranest(dirr,params):
    """
    Function to parse ultranest results. Returns dictionary with 
    - samples_equal : equally weighted samples that can be directly used for corner plots, finding quantiles, etc 
    - max_logl : Bayesian Evidence - the lnZ of the max likelihood point 
    - max_logl_point : the values of associated with the max likelihood point 
    - med_intervals : median, errlo, errup of all the constrained parameters 
    - param_names : list string of the parameter names 
    - converged : False if the run has not finished, in which case all results are intermediate 
    - niter, ess : number of iterations and effective sample size of the (possibly intermediate) results 
    - ultranest_out : all the raw output of ultranest 

    Parameters 
    ----------
    dirr : str
        directory of the ultranest output 
    params : int
        number of parameters 

    Returns 
    -------
    dict 
    """
    try:
        import ultranest.integrator as uint
    except ImportError as exc:
        raise ImportError(
            "Reading UltraNest results requires the optional UltraNest package. "
            "Install it with `pip install 'picaso[ultranest]'`."
        ) from exc

    res = uint.read_file(dirr,len(params))
    res[1]['paramnames'] =  params
    results = res[1]
    paramnames = results['paramnames']
    data = np.array(results['weighted_samples']['points'])
    weights = np.array(results['weighted_samples']['weights'])
    cumsumweights = np.cumsum(weights)
    logl = np.array(res[0]['logl'])
    mask = cumsumweights > 1e-4
    samples = results['samples']
    maxlogl_point = results['maximum_likelihood']['point']
    maxlogl_value = results['maximum_likelihood']['logl']
    
    samples_rew = _resample_equal(data, weights)
    #ultranest only writes info/post_summary.csv once the run finishes, so without it these are intermediate results
    converged = os.path.exists(os.path.join(dirr, 'info','post_summary.csv'))
    if converged:
        summary = pd.read_csv(os.path.join(dirr, 'info','post_summary.csv'))
        eval_at_med = [summary[i+'_'+'median'].values[0] for i in params]   
    else: 
        print(f"Ultranest run not converged: showing intermediate results after {results['niter']} iterations (ESS={results['ess']:.0f})")
        errlo = pd.DataFrame(samples_rew,columns=params).quantile(.158655)
        errup = pd.DataFrame(samples_rew,columns=params).quantile(.841345)
        median = pd.DataFrame(samples_rew,columns=params).quantile(0.50)
        summary={}
        for i in params: 
            summary[i+'_errlo']=errlo[i]
            summary[i+'_errup']=errup[i]
            summary[i+'_median']=median[i]
        summary = pd.DataFrame(summary,index=[0])
        eval_at_med = [summary[i+'_'+'median'].values[0] for i in params]   
        
    return {'samples_equal':samples_rew,
            'max_logl':maxlogl_value,
            'max_logl_point': maxlogl_point,
            'med_point': eval_at_med, 
            'med_intervals': summary,
            'param_names': params,
            'converged': converged,
            'niter': results['niter'],
            'ess': results['ess'],
            'ultranest_out':res[1]}

def read_nautilus(dirr, fitpars, param_tools=None):
    """
    Function to parse nautilus results. Returns dictionary with 
    - samples_equal : equally weighted samples that can be directly used for corner plots, finding quantiles, etc 
    - max_logl : the max log likelihood value 
    - max_logl_point : the values of associated with the max likelihood point 
    - med_intervals : median, errlo, errup of all the constrained parameters 
    - param_names : list string of the parameter names 
    - log_z : Bayesian Evidence estimate lnZ 
    - ess : effective sample size of the posterior 
    - n_like : number of likelihood evaluations 
    - nautilus_out : the nautilus Sampler restored from the checkpoint 

    Nautilus checkpoints after every shell so these can be read while the run is in progress. 
    Unlike ultranest, the checkpoint does not record whether the run finished, so check 
    that ess reached the n_eff requested in run_kwargs (default 10000). 

    Parameters 
    ----------
    dirr : str
        directory of the nautilus output (contains nautilus.hdf5)
    fitpars : dict
        output of driver.prior_finder(config['retrieval']), used to transform the 
        unit cube samples stored by nautilus to physical values 
    param_tools : class 
        parameterize class, only needed for xarray_grid priors 

    Returns 
    -------
    dict 
    """
    try:
        from nautilus import Sampler
    except ImportError as exc:
        raise ImportError(
            "Reading Nautilus results requires the optional 'nautilus-sampler' "
            "package. Install it with `pip install 'picaso[nautilus]'`."
        ) from exc
    from functools import partial
    from .driver import hypercube

    checkpoint_file = os.path.join(dirr, 'nautilus.hdf5')
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f'Nautilus checkpoint does not exist: {checkpoint_file}')
    params = list(fitpars.keys())

    #the likelihood is never called, the restored sampler is only used to get the posterior
    def _no_likelihood(x):
        raise RuntimeError('read_nautilus does not evaluate the likelihood')

    sampler = Sampler(partial(hypercube, fitpars=fitpars, param_tools=param_tools),
                      _no_likelihood, n_dim=len(params),
                      filepath=checkpoint_file, resume=True)
    if len(sampler.points) == 0:
        raise ValueError(f'Nautilus checkpoint {checkpoint_file} does not contain any samples yet')

    samples, log_w, logl = sampler.posterior()
    weights = np.exp(log_w)
    weights = weights / np.sum(weights)
    samples_rew = _resample_equal(samples, weights)

    max_logl_idx = np.argmax(logl)
    max_logl_point = samples[max_logl_idx]
    max_logl_value = logl[max_logl_idx]

    errlo = pd.DataFrame(samples_rew, columns=params).quantile(.158655)
    errup = pd.DataFrame(samples_rew, columns=params).quantile(.841345)
    median = pd.DataFrame(samples_rew, columns=params).quantile(0.50)
    summary = {}
    for i in params:
        summary[i+'_errlo'] = errlo[i]
        summary[i+'_errup'] = errup[i]
        summary[i+'_median'] = median[i]
    summary = pd.DataFrame(summary, index=[0])
    eval_at_med = [summary[i+'_'+'median'].values[0] for i in params]

    return {
        'samples_equal': samples_rew,
        'max_logl': max_logl_value,
        'max_logl_point': max_logl_point,
        'med_point': eval_at_med,
        'med_intervals': summary,
        'param_names': params,
        'log_z': sampler.log_z,
        'ess': sampler.n_eff,
        'n_like': int(sampler.n_like),
        'nautilus_out': sampler
    }

def get_bands(config, retrieval_results,
              N=100,
              pressure_bands=['temperature','H2O','CO2'],
              eval_maxlogl=False):
    """
    Loads in configuration file and retrieval results from read_retrievals and 
    computes model evaluations for N samples. returns all the relevant band 
    information necessary to plot pressure and spectra interval bands. 

    Parameters
    ----------
    config : dict
        Loaded configuration toml file
    retrieval_results : dict 
        The result of read_retrieval function 
    N : int 
        Number of samples
    pressure_bands : list of str
        List of string values that determine what molecules will be plot with temperature
    eval_maxlogl : bool 
        Default=False, but if true in addition to getting the median profile it will also 
        get the max loglikelihood info and return it as well 
    
    Returns
    -------
    dict 
        - all_samples_out : raw returns from check_model_samples 
        - wavenumber : xaxis in cm-1 
        - wavelength : xaxis in um 
        - pressure : pressure axis in bars
        - bands_spectra : dict with spectra bands info for median and 1sig, 2sig, and 3sig 
        - bands_ptchem : dict with prsesure functioned bands info and 1-3 sigma 
        - max_logl_spectra : max likelihood spectrum 
        - max_logl_ptchem : max likelihood profile 
        - max_logl_chisq : chi sq per data point of max likelihood profile 
    """
    samples_equal = retrieval_results['samples_equal']
    from .driver import check_model_samples
    draws=np.random.randint(0, 
                samples_equal.shape[0], N)
    subset_samples = samples_equal[draws,:]

    returns = {}

    out = check_model_samples(
        config, N=N, 
        samples=subset_samples,
        full_likelihood=True)
    
    returns['all_samples_out'] = out 
    returns['observation_type'] = config.get('observation_type')
    
    xaxis = out['xdata']
    returns['wavenumber'] = xaxis
    returns['wavelength'] = 1e4/xaxis
    band_spec = PredictionBand(xaxis)
    paxis = out['profiles'][0]['pressure']
    returns['pressure'] = paxis
    band_profile = {ival :PredictionBand(paxis) for ival in pressure_bands}


    returns['bands_spectra']={}
    returns['bands_ptchem']={ival:{} for ival in pressure_bands}
    for i in range(N):
        band_spec.add(out['ymodel'][i])
        for ival in pressure_bands: 
            band_profile[ival].add(out['profiles'][i][ival].values)

    for q ,key in zip([k/100/2 for k in [68.27, 95.45, 99.73]], ['1sig','2sig','3sig']): 
        returns['bands_spectra'][key+'_lo'] = band_spec.get_line(0.5 - q).data
        returns['bands_spectra'][key+'_hi'] = band_spec.get_line(0.5 + q).data
        for ival in pressure_bands: 
            returns['bands_ptchem'][ival][key+'_lo'] = band_profile[ival].get_line(0.5 - q).data
            returns['bands_ptchem'][ival][key+'_hi'] = band_profile[ival].get_line(0.5 + q).data

    returns['bands_spectra']['median'] = band_spec.get_line(0.5).data
    for ival in pressure_bands: 
        returns['bands_ptchem'][ival]['median'] = band_profile[ival].get_line(0.5).data
    
    if eval_maxlogl: 
        if 'max_logl_point' in retrieval_results and retrieval_results['max_logl_point'] is not None:
            out_max = check_model_samples(
                config, N=1,
                samples=np.atleast_2d(retrieval_results['max_logl_point']),
                full_likelihood=True)
            returns['max_logl_spectra'] = out_max['ymodel'][0]
            returns['max_logl_ptchem'] = out_max['profiles'][0]
            returns['max_logl_chisq'] = out_max['chi_sq_per_pt'][0]
            returns['all_maxlogl_out'] = out_max
    return returns 


def plot_pressure_bands(returns,colors=pals.Muted5):
    fig,ax=plt.subplots(1,2) 
    #temperature 
    for ii,ival in enumerate(returns['bands_ptchem'].keys()): 
        if ival == 'temperature':
            iax = 0 
        else: 
            iax = 1

        for i in range(1,3):
            lo=returns['bands_ptchem'][ival][f'{i}sig_lo']
            hi= returns['bands_ptchem'][ival][f'{i}sig_hi']
                
            ax[iax].fill_betweenx(returns['pressure'], lo,
                                        hi,
                                        color=colors[ii],alpha=0.4)
        ax[iax].plot(returns['bands_ptchem'][ival]['median'],
                     returns['pressure'],
                label=rf'Median {ival}',color=colors[ii],linestyle='-.')
    
    for i in ax: i.set_yscale('log')
    ax[1].set_xscale('log')
    for i in ax: i.set_ylim([1e2,1e-6])
    for i in ax: i.legend()
    ax[0].set_ylabel('Pressure (bars)')
    ax[0].set_xlabel('Temperature (Kelvin)')
    ax[1].set_xlabel('Mixing Ratio (v/v)')
    return fig,ax 

SPECTRUM_YLABELS = {
    'transit_depth': r'Transit Depth $(R_p/R_s)^2$',
    'fpfs_reflected': 'Planet Flux / Stellar Flux',
    'fpfs_thermal': 'Planet Flux / Stellar Flux',
    'thermal': r'Flux [erg/cm$^2$/s/cm]',
    'albedo': 'Apparent Albedo',
}

def plot_spectra_bands(returns,colors=pals.Muted5,observation_type=None):
    """
    observation_type : str 
        Sets the y axis units (e.g. transit_depth). Default uses returns['observation_type'] from get_bands 
    """
    fig,ax=plt.subplots()
    #data can be stitched from several instruments out of order, sort so bands are contiguous
    order = np.argsort(returns['wavelength'])
    xgrid = np.asarray(returns['wavelength'])[order]
    for i in range(1,3):
        lo=np.asarray(returns['bands_spectra'][f'{i}sig_lo'])[order]
        hi= np.asarray(returns['bands_spectra'][f'{i}sig_hi'])[order]
        ax.fill_between(xgrid, lo,
                                  hi,
                                   color='red',alpha=0.2)
    med=np.asarray(returns['bands_spectra']['median'])[order]
    ax.plot(xgrid,med,color='black', label='Median')
    ax.legend()
    ax.set_xlabel(r'Wavelength [$\mu$m]')
    observation_type = observation_type or returns.get('observation_type')
    if observation_type is not None:
        ax.set_ylabel(SPECTRUM_YLABELS.get(observation_type, observation_type))
    return fig,ax

def retrieval_results(evaluations, info, filename,round=3,return_samples=True,
                     spectrum_tag='transit_depth',spectrum_unit='cm**2/cm**2',
                    author="",contact="",model_description="",code="PICASO", **extra_attrs):
    """
    Returns all data output and creates xarray, pickle of samples, and sample plots 
    """
    coords = {}
    coords["wavelength"]=(["wavelength"], evaluations['wavelength'],{'units': 'um'})
    if 'pressure' in evaluations.keys() :
        coords["pressure"]=(["pressure"], evaluations['pressure'],{'units': 'bar'})
        
    #maxlogL temperature pressure & chemistry
    data_vars={}
    all_mols = []
    if 'pressure' in evaluations.keys() and 'max_logl_ptchem' in evaluations.keys():
        for ikey in evaluations['max_logl_ptchem'].keys():
            if 'temp' in ikey: 
                unit='Kelvin'
            else:
                unit='v/v'
                all_mols+=[ikey]
            if 'pressure' not in ikey: 
                data_vars['max_logl_'+ikey]=(["pressure"], evaluations['max_logl_ptchem'][ikey],{'units': unit})

    #maxlogL spectrum 
    if 'max_logl_spectra' in evaluations.keys():
        data_vars['max_logl_'+spectrum_tag]=(["wavelength"], evaluations['max_logl_spectra'],{'units': spectrum_unit})

    #median ptchem
    if 'pressure' in evaluations.keys() and 'bands_ptchem' in evaluations.keys():
        for ikey in evaluations['bands_ptchem'].keys():
            if 'temp' in ikey: 
                unit='Kelvin'
            else:
                unit='v/v'
        
            data_vars['median_'+ikey]=(["pressure"], evaluations['bands_ptchem'][ikey]['median'],{'units': unit})
            #siglo and hi 
            for i in range(1,3):
                data_vars[f'{i}sig_lo_'+ikey]=(["pressure"], evaluations['bands_ptchem'][ikey][f'{i}sig_lo'],{'units': unit})
                data_vars[f'{i}sig_hi_'+ikey]=(["pressure"], evaluations['bands_ptchem'][ikey][f'{i}sig_hi'],{'units': unit})

    #median spectra
    if 'bands_spectra' in evaluations.keys():
        data_vars['median_'+spectrum_tag]=(["wavelength"], evaluations['bands_spectra']['median'],{'units': spectrum_unit})
        for i in range(1,3):
            data_vars[f'{i}sig_lo_'+spectrum_tag]=(["wavelength"], evaluations['bands_spectra'][f'{i}sig_lo'],{'units': spectrum_unit})
            data_vars[f'{i}sig_hi_'+spectrum_tag]=(["wavelength"], evaluations['bands_spectra'][f'{i}sig_hi'],{'units': spectrum_unit})
    
    #names
    param_names =  info['param_names']
    #maxlogL values 
    maxlogl = info['max_logl_point']

    #med errorbars 
    summary = info['med_intervals']
    strvals={}
    
    if isinstance(round,int):
        round=[round]*len(param_names)
        
    for i,r in zip(param_names,round):
        median = info['med_intervals'][i+'_median'].values[0]
        lo = info['med_intervals'][i+'_errlo'].values[0]
        hi = info['med_intervals'][i+'_errup'].values[0]
        errlo = median-lo
        errhi = hi-median 
        median=int(median*(10**r))/(10**r)
        errlo =int(errlo*(10**r))/(10**r)
        errhi=int(errhi*(10**r))/(10**r)
        strvals[i] = f'${median}'+'^{+' + f'{errhi}' + '}_{-' + f'{errlo}' + '}$'
    
    max_logl_chisq = evaluations['max_logl_chisq']

    attributes = dict(author=author,#required
                      contact=contact,#required
                      model=model_description,
                      max_logl_chisq = max_logl_chisq,
                      code=code, #required, in this case I used numpy to make my fake model.
                      max_logl_params=json.dumps({ip:maxlogl[i] for i,ip in enumerate(param_names)},cls=JsonCustomEncoder),
                      intervals_params=json.dumps({ip:strvals[ip] for i,ip in enumerate(param_names)},cls=JsonCustomEncoder),
                      molecules = all_mols,
                     )
    attributes.update(extra_attrs)

    build_xarray = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=attributes,
    )
    #output main datafile 
    build_xarray.to_netcdf(filename+'_median_and_max_logl.nc')

    #default figures 
    f,a=plot_spectra_bands(evaluations, pals.Blues[3])
    f.savefig(filename+'_spectra.png')

    if 'pressure' in evaluations.keys() :
        f,a=plot_pressure_bands(evaluations,pals.Greys[3])
        f.savefig(filename+'_ptchem.png')

    #return samples 
    if return_samples:
        pk.dump([info['param_names'],info['samples_equal']], open(filename+'_equally_weighted_samples.pk','wb'))

    #finally make corner plot 
    f, a = plot_pair(info['samples_equal'], info['param_names'], log_params=info.get('log_params'))
    f.savefig(filename+'_plotpair.png')
    return build_xarray


def stylized_ticks(min_val, max_val, num_ticks):
    """
    Generates stylized tick marks for a plot axis.

    Args:
        min_val: Minimum value of the axis.
        max_val: Maximum value of the axis.
        num_ticks: Desired number of tick marks (approximately).

    Returns:
        A tuple containing:
            - ticks: A list of tick values.
            - format_string: A format string for displaying the ticks.
    """

    if min_val >= max_val or num_ticks <= 1:
        return [min_val, max_val], "{}"  # Handle invalid input
        

    range_val = max_val - min_val
    approx_step = range_val / (num_ticks - 1)

    # Round to "nice" numbers (powers of 1, 2, 5, 10)
    power = np.floor(np.log10(approx_step))
    scaled_step = approx_step / (10**power)

    if scaled_step < 2:
        nice_step = 1
    elif scaled_step < 5:
        nice_step = 2
    else:
        nice_step = 5

    step = nice_step * (10**power)
    
    #adjust min and max to be on the grid
    min_val = np.floor(min_val/step)*step
    max_val = np.ceil(max_val/step)*step

    ticks = np.arange(min_val, max_val + step/10, step) #add a tiny amount to avoid floating point errors

    # Determine formatting
    decimal_places = int(max(0, -power))
    format_string = "{:." + str(decimal_places) + "f}"

    return ticks, [format_string.format(t) for t in ticks]

def plot_pair(samples, params, pretty_labels=None,ranges=None,figsize=(11, 11), contour_cmap="GnBu",intervals=None,
              log_params=None):
    """
    Plot stylized corner plots 

    Parameters
    ----------
    samples : ndarray 
        matrix of samples 
    params : list of str
        List of string parameters 
    pretty_labels : list of str 
        List of string parameters that match params stylized for plots 
        Note : the order of pretty_labels has to match the corner plot, not params input ! This is very annoying 
        but it is because of arviz. Suggest running with None and then checking the order of the corner plot 
        Default = None
    ranges : list of list 
        list of min and max values for each parameters 
        Note : the order of pretty_labels has to match the corner plot, not params input ! This is very annoying 
        but it is because of arviz. Suggest running with None and then checking the order of the corner plot 
        Default = None 
    figsize : tuple 
        width, height of plot 
    countour_cmap : str 
        String of matplotlib colormaps 
    intervals : list of str 
        list of stylized intervals for the top of the plots. you can get them from the xarray intervals. 
    log_params : list of str 
        parameters that were sampled in log10 space (e.g. info['log_params'] from read_retrievals). 
        These are plotted as log10 of the samples, so ranges and intervals for them should be in log10 too. 
        Default = None 
        
    """
    az.style.use('arviz-vibrant')

    log_params = [] if log_params is None else list(log_params)
    samples = np.array(samples, dtype=float)
    for ip in log_params:
        if ip in params:
            samples[:, params.index(ip)] = np.log10(samples[:, params.index(ip)])

    if isinstance(pretty_labels,type(None)):
        pretty_labels=[f'log10({ip})' if ip in log_params else ip for ip in params]
    elif isinstance(pretty_labels,dict):
        pretty_labels=[pretty_labels[i] for i in params]
    elif isinstance(pretty_labels, list):
        pass
    else: 
        raise Exception('Pretty labels must be None, dict, or list')


    if isinstance(intervals, type(None)):
        intervals_dict = {}
        for ip in params:
            idx = params.index(ip)
            vals = samples[:, idx]
            q_lo, q_mid, q_hi = np.percentile(vals, [15.8655, 50.0, 84.1345])
            errlo = q_mid - q_lo
            errhi = q_hi - q_mid
            err = min(errlo, errhi) if min(errlo, errhi) > 0 else max(errlo, errhi)
            if err <= 0:
                dec = 2
            else:
                log_err = np.log10(err)
                if log_err >= 0:
                    dec = max(0, 2 - int(np.floor(log_err)))
                else:
                    dec = int(np.ceil(-log_err)) + 1
                dec = min(max(dec, 1), 6)
            fmt = f"{{:.{dec}f}}"
            val_str = fmt.format(q_mid)
            errlo_str = fmt.format(errlo)
            errhi_str = fmt.format(errhi)
            intervals_dict[ip] = f"${val_str}_{{-{errlo_str}}}^{{+{errhi_str}}}$"
        intervals = [intervals_dict[ip] for ip in params]
    elif isinstance(intervals,dict):
        intervals=[intervals[i] for i in params]
    elif isinstance(intervals, list):
        pass
    else: 
        raise Exception('Intervals must be None, dict, or list')


    if isinstance(ranges,type(None)):
        ranges_dict = {}
        for ip in params:
            idx = params.index(ip)
            vals = samples[:, idx]
            ranges_dict[ip] = [np.min(vals), np.max(vals)]
        ranges = [ranges_dict[ip] for ip in params]
    elif isinstance(ranges,dict):
        ranges=[ranges[i] for i in params]
    elif isinstance(ranges, list):
        pass
    else: 
        raise Exception('Ranges must be None, dict, or list')



    #built directly on matplotlib because az.plot_pair changed its whole API in arviz>=1.0
    nparams = len(params)
    medians = np.median(samples, axis=0)
    hdi_probs = [0.989, 0.865, 0.393]  # 3, 2 and 1 sigma contours
    #kde cost scales with the number of samples, so thin the samples used for the contours
    kde_samples = samples[np.random.RandomState(0).permutation(len(samples))[:2000]]

    fig, ax = plt.subplots(nparams, nparams, figsize=figsize, squeeze=False, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0, hspace=0)
    for i in range(nparams):
        for j in range(nparams):
            a = ax[i,j]
            if j > i:
                a.set_visible(False)
                continue
            if i == j:
                a.hist(samples[:,i], bins=20, color='grey', alpha=0.7)
                a.axvline(medians[i], color='k')
                a.set_yticks([])
            else:
                x, y = samples[:,j], samples[:,i]
                a.scatter(x, y, color='grey', alpha=0.5, s=2, rasterized=True)
                try:
                    kde = gaussian_kde(np.vstack([kde_samples[:,j], kde_samples[:,i]]))
                    xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 60), np.linspace(y.min(), y.max(), 60))
                    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                    #density thresholds that enclose each hdi probability
                    dsort = np.sort(density.ravel())[::-1]
                    cumsum = np.cumsum(dsort) / np.sum(dsort)
                    levels = [dsort[np.searchsorted(cumsum, p)] for p in hdi_probs] + [dsort[0]]
                    a.contourf(xx, yy, density, levels=levels, cmap=contour_cmap, alpha=0.8)
                    a.contour(xx, yy, density, levels=levels[:-1], colors='k', alpha=0.5, linewidths=0.8)
                except (np.linalg.LinAlgError, ValueError):
                    pass #parameter has no spread yet (e.g. very early in a run), so just show the scatter
                a.plot(medians[j], medians[i], 'o', color='k', ms=4)
                if j > 0:
                    a.sharey(ax[i,0])
            if i < nparams-1:
                a.sharex(ax[-1,j])
                a.tick_params(labelbottom=False)
            if j > 0:
                a.tick_params(labelleft=False)
            a.set_xlabel(pretty_labels[j] if i == nparams-1 else '')
            a.set_ylabel(pretty_labels[i] if (j == 0 and i > 0) else '')

    if not isinstance(pretty_labels,type(None)):
        assert len(pretty_labels)==len(params), "length of pretty labels must match length of parameters (second input)"
        for i,ip in enumerate(pretty_labels): 
            ax[i,0].set_ylabel(ip, rotation=45, ha='right', va='center')
            ax[-1,i].set_xlabel(ip, rotation=45, ha='right')
    
    if ((not isinstance(pretty_labels,type(None))) & (not isinstance(intervals,type(None)))):  
        assert len(intervals)==len(params), "length of intervals must match length of parameters (second input)"
        for i,ip in enumerate(pretty_labels): 
            ax[i,i].set_title(ip+'\n'+intervals[i], fontsize='small')
    elif ((isinstance(pretty_labels,type(None))) & (not isinstance(intervals,type(None)))):  
        assert len(intervals)==len(params), "length of intervals must match length of parameters (second input)"
        for i,ip in enumerate(params): 
            ax[i,i].set_title(ip+'\n'+intervals[i], fontsize='small')        
            
            
    
    if not isinstance(ranges,type(None)):
        assert len(ranges)==len(params), "length of ranges must match length of parameters (second input)"
        for i in range(len(ranges)): 
            ticks, form_ticks = stylized_ticks(ranges[i][0],ranges[i][1],2)
            ax[-1,i].set_xticks(ticks)
            ax[-1,i].set_xticklabels(form_ticks, rotation=45) 
            ax[-1,i].set_xlim(ranges[i][0],ranges[i][1])
        
            if i!=0:
                ax[i,0].set_yticks(ticks)
                ax[i,0].set_yticklabels(form_ticks) 
                ax[i,0].set_ylim(ranges[i][0],ranges[i][1])

        
    fig = ax.ravel()[0].figure
    return fig, ax


## Parameterizations 