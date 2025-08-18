import numpy as np
import json 
from astropy.utils.misc import JsonCustomEncoder
from astropy.convolution import convolve, Gaussian1DKernel
import arviz as az
import dynesty
import pandas as pd
from scipy import interpolate

from .justdoit import mean_regrid,vj,u,get_cld_input_grid

from .analyze import chi_squared
from .justplotit import pals

import matplotlib.pyplot as plt
import xarray as xr
from ultranest.plot import PredictionBand
import ultranest.integrator as uint
import os 
import pickle as pk
import re

__refdata__ = os.environ.get('picaso_refdata')

allowed_keys={ 'line':
                    {},
               'grid':
                    {'grid_location':str, 'grid_name':str,'to_fit':str}, 
               'grid_virga':
                    {'grid_location':str, 'grid_name':str,'to_fit':str,'opacity_filename_db':str, 'molecules':list, 'virga_mieff_dir':str,'cloud_species':list},
               'grid_addchem':
                    {'grid_location':str, 'grid_name':str,'to_fit':str,'opacity_filename_db':str, 'molecules':list,'overwrite_molecule':None},
               'grid_flexcloud':
                    {'grid_location':str, 'grid_name':str,'to_fit':str,'opacity_filename_db':str, 'molecules':list, 'virga_mieff_dir':str,'cloud_species':list}
            
            }

def create_template(rtype, script_filename, sampler_output_path, 
    grid_kwargs={}):
    """
    Creates a Python script template with xarray and a function.

    Parameters
    ----------
    type: str
        will create different kinds of templates for you 
        options: line, grid, free, grid+cloud
    
    script_filename : str 
        Path to save the script to. 

    grid_kwargs : dict 
        Dictionary of key words to pass to the grid retrieval script
        Options include 
        - grid_location : path to where the grid is located 
        - grid_name : name of the grid 

    Returns:
        The generated script as a string, or None if saved to a file.
    """
    allowed = allowed_keys.keys()
    assert rtype in allowed, f"Only options for rtype are currently: {allowed.keys()}"
    
    if 'grid_' in rtype: 
        input_file = os.path.join(__refdata__, 'scripts',f'gridplus_retrieval.py')
    else: 
        input_file = os.path.join(__refdata__, 'scripts',f'{rtype}_retrieval.py')

    with open(input_file, 'r') as f:
        content = f.read()

    # Modify the content by replacing 
    content = content.replace('sampler_output_pathCHANGEME',f"""'{sampler_output_path}'""")#.replace("var1=3", f"var1={new_value}")

    #replace all grid kwargs for the grid line
    if 'grid' in rtype: 
        for ikey in grid_kwargs.keys(): 
            if ikey not in allowed_keys[rtype].keys(): 
                raise Exception(f"{ikey} is not yet supported key. Try these:",allowed_keys[rtype].keys())
            string = grid_kwargs.get(ikey,'CHANGEME')#if there is no key then just replace the value with CHANGEME for the user
            #if the replaceabel value is a string we will 
            #want to add it to the script with quotes 
            #otherwise just stick it in as expected 
            if allowed_keys[rtype][ikey] == str: 
                replace = f"""'{string}'"""
            else:  
                replace = f"""{string}"""

            content = content.replace(ikey+'CHANGEME',replace)
    
        #now remove the unecessary code blocks from the template 
        #for the case where we have the grid_addon 
        if 'grid_' in rtype: 
            content = content.replace('rtypeCHANGEME',f"""'{rtype}'""")
            allgrids = [i for i in allowed_keys.keys() if 'grid_' in i]
            allgrids.pop(allgrids.index(rtype))
            for gridtype in allgrids: 
                content = remove_code_blocks(content,gridtype)

        #now remove any left over tags that start with #[[ or #]]
        content = remove_lines_with_markers(content)

    with open(script_filename, 'w') as f:
        f.write(content)

def remove_lines_with_markers(text):
    """Removes all lines from a string that contain either '#[[' or '#]]'.

    Args:
        text: The input string.

    Returns:
        The string with the specified lines removed.
    """
    lines = text.splitlines()
    new_lines = [line for line in lines if not re.search(r"#\[\[|#\]\]", line)]  # More efficient
    return "\n".join(new_lines)
def remove_code_blocks(text, tag):
    """
    Removes blocks of text from a string that are delimited by '#[[tag' and '#]]tag' lines.

    Args:
        text: The input string.

    Returns:
        The string with the code blocks removed.  Returns the original string if no matching blocks are found.
    """

    # Use a regular expression to find and remove the blocks.  The re.DOTALL flag allows the . to match newlines.
    # The ? makes the match non-greedy, so it matches the shortest possible block.
    pattern = rf"#\[\[{tag}.*?#\]\]{tag}"  # More robust pattern
    new_text = re.sub(pattern, "", text, flags=re.DOTALL)

    return new_text

## BEGIN ALL RETR ANALYSIS TOOLS 


def get_info(dirr,params):
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
    
    samples_rew = dynesty.utils.resample_equal(
        data, weights, rstate=np.random.RandomState(0))
    try:
        summary = pd.read_csv(os.path.join(dirr, 'info','post_summary.csv'))
        eval_at_med = [summary[i+'_'+'median'].values[0] for i in params]   
    except: 
        print('not converged')
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
            'ultranest_out':res[1]}

def get_evaluations(samples_equal, max_logl, model, n_draws, regrid=False,pressure_bands=['temperature','H2O','CO2']):
    """
    Return model at the max logZ and also get the banded 1,2 and 3 sigma values for both spectra and chemistry. 

    Parameters
    ----------
    samples_equal : ndarray
        Equally resampled weights 
    max_logl : float 
        Bayesian evidence 
    model : function 
        Model in model_set 
    n_draws : int 
        number of draws to compute banded spectra/chemistry 
    regrid : bool or ndarray or float 
        Default = False, which does not regrid the opacities. can also input wavenumber grid, or resolution 
    pressure_bands : list
        List of strings to specify which chemicals to return 
        Default = ['temperature','H2O','CO2']
        If running a nonphysical model, make sure to set this to an empty list = []
        This can ONLY be used if your `model_set` model has the key word argument 
        `return_ptchem` which is either True/False. If True it should return 
        the entire planet class in either dictionary or class format. e.g. 
        {'opa db1':picaso.inputs class} or just picaso.inputs class

    Returns
    -------
    dict 
        - max_logl_ptchem : associated chemistry and temperature values of the max logl model 
        - bands_spectra : 1, 2, and 3 sigma spectra and median
        - bands_ptchem : 1, 2, and 3 sigma chemistry and temperature and median
        - max_logl_error_inflation : associated error inflation of the max logl model if there is error inflation 
        - max_logl_offsets : associated offsets of the max logl model if there are offsets included 
        - pressure : pressure grid 
        - wavelength : regridded wavelength grid  
        
    """
    returns = {}
    if len(pressure_bands)>0:
        picaso_class = model(max_logl,return_ptchem=True)
        #users sometimes return a dictionary of classes 
        if isinstance(picaso_class, dict): 
            key = list(picaso_class.keys())[0]
            returns['max_logl_ptchem'] = picaso_class[key].inputs['atmosphere']['profile']
        else: 
            returns['max_logl_ptchem'] = picaso_class.inputs['atmosphere']['profile']
        df = returns['max_logl_ptchem'] 

    returns['bands_spectra']={}
    if len(pressure_bands)>0: 
        returns['bands_ptchem']={i:{} for i in pressure_bands}
        pband_classes={i:{} for i in pressure_bands}
    
    draws=np.random.randint(0, samples_equal.shape[0], n_draws)
    
    first = True
    for idraw in draws:
        x,y,of,er = model(samples_equal[idraw,:])
        
        if len(pressure_bands)>0:
            picaso_class = model(samples_equal[idraw,:], return_ptchem=True)
            #users sometimes return a dictionary of classes 
            if isinstance(picaso_class, dict): 
                key = list(picaso_class.keys())[0]
                chem = picaso_class[key].inputs['atmosphere']['profile']
            else: 
                chem = picaso_class.inputs['atmosphere']['profile']        
        
        if isinstance(regrid,np.ndarray): 
            #assumed to be wavenumber grid 
            _,y = mean_regrid(x,y,newx=regrid);binning=True
            um_xgrid=1e4/regrid
        elif isinstance(regrid,float): 
            wno_xgrid,y = mean_regrid(x,y,R=regrid);binning=True
            um_xgrid=1e4/wno_xgrid
        else: 
            um_xgrid=1e4/x
            
        if first:
            band = PredictionBand(um_xgrid)
            if len(pressure_bands)>0:
                for i in pband_classes.keys(): pband_classes[i]=PredictionBand(df['pressure'].values)
            first=False
            
        band.add(y)
        if len(pressure_bands)>0:
            for i in pband_classes.keys(): pband_classes[i].add(chem[i].values)
            

    for q ,key in zip([k/100/2 for k in [68.27, 95.45, 99.73]], ['1sig','2sig','3sig']): 
        returns['bands_spectra'][key+'_lo'] = band.get_line(0.5 - q).data
        returns['bands_spectra'][key+'_hi'] = band.get_line(0.5 + q).data
        if len(pressure_bands)>0:
            for i in pband_classes.keys(): returns['bands_ptchem'][i][key+'_lo'] = pband_classes[i].get_line(0.5 - q).data
        if len(pressure_bands)>0:
            for i in pband_classes.keys(): returns['bands_ptchem'][i][key+'_hi'] = pband_classes[i].get_line(0.5 + q).data

    returns['bands_spectra']['median'] = band.get_line(0.5).data
    if len(pressure_bands)>0:
        for i in pband_classes.keys(): returns['bands_ptchem'][i]['median'] = pband_classes[i].get_line(0.5).data

    maxx,maxlogl,offsets,err = model(max_logl)
    if binning: _,maxlogl = mean_regrid(maxx,maxlogl,newx=1e4/um_xgrid)
    returns['max_logl_spectra'] =maxlogl
    
    returns['max_logl_error_inflation'] = err
    returns['max_logl_offsets'] = offsets


    if len(pressure_bands)>0:returns['pressure'] = df['pressure']
    returns['wavelength'] = um_xgrid
    return returns

def get_chisq_max(at_evaluations, data_dict):
    """
    Compute the chi squared at the max logl spectra incl offsets and error inflation if it is included 

    Parameters
    ----------
    at_evaluations : dict 
        Dictionary returned from get_evaluations 
    data_dict : dict 
        Dectionary returned from get_data in setup.py 

    Results
    -------
    dict 
        - wavenumber
        - model regridded at data 
        - datay 
        - datae 
        - chisq per data point
    """
    
    offsets = at_evaluations['max_logl_offsets']
    resultx, resulty = 1e4/at_evaluations['wavelength'], at_evaluations['max_logl_spectra']
    
    y_model_all=[]
    x_data_all=[]
    y_data_all=[]
    e_data_all=[]
    for idata in data_dict.keys():
        if idata in offsets.keys(): 
            offset=offsets[idata]
        else: 
            offset = 0
        x_chunk, y_chunk=mean_regrid(resultx, resulty, newx=data_dict[idata][0])
        y_model_all += [y_chunk]
        x_data_all += [x_chunk]
        y_data_all += [data_dict[idata][1]+offset]
        e_data_all += [data_dict[idata][2]]

    df = pd.DataFrame(dict(x=np.concatenate(x_data_all), 
                     ymod = np.concatenate(y_model_all), 
                     ydat = np.concatenate(y_data_all), 
                     edat = np.concatenate(e_data_all)))
    
    df = df.sort_values(by='x')
    
    sorted_x = df['x'].values#np.array([pair[0] for pair in combined])
    sorted_y = df['ymod'].values#np.array([pair[1] for pair in combined])
    datasorted_y = df['ydat'].values#np.array([pair[2] for pair in combined])
    datasorted_e = df['edat'].values#np.array([pair[3] for pair in combined])

    chisq = chi_squared(np.array(datasorted_y),np.array(datasorted_e),
                np.array(sorted_y),0)
    
    return {'wavenumber':sorted_x, 'model':sorted_y,
            'datay':datasorted_y,'datae':datasorted_e, 'chisq_per_datapt':chisq}


def plot_spectra_bands(evaluations_dat, colors, ax=None,subplots_kwargs={},R=None):
    """
    Plots banded spectra and returns fig, ax
    """
    if ax is None: 
        fig,ax=plt.subplots(**subplots_kwargs)
    else: 
        fig=None
    og_xgrid = evaluations_dat['wavelength']

    for i in range(1,3):
        if isinstance(R,(float,int)):
            wno,lo = mean_regrid(1e4/og_xgrid, evaluations_dat['bands_spectra'][f'{i}sig_lo'],R=R)
            wno,hi = mean_regrid(1e4/og_xgrid, evaluations_dat['bands_spectra'][f'{i}sig_hi'],R=R)
            um_xgrid=1e4/wno
        else: 
            lo=evaluations_dat['bands_spectra'][f'{i}sig_lo']
            hi= evaluations_dat['bands_spectra'][f'{i}sig_hi']
            um_xgrid=og_xgrid
        
        ax.fill_between(um_xgrid, lo,
                                  hi,
                                   color=colors[i-1],alpha=0.2)
    if isinstance(R,(float,int)):
        wno,med = mean_regrid(1e4/og_xgrid, evaluations_dat['bands_spectra']['median'],R=R)
        wno,maxx = mean_regrid(1e4/og_xgrid, evaluations_dat['max_logl_spectra'],R=R)
        um_xgrid=1e4/wno
    else: 
        med=evaluations_dat['bands_spectra']['median']
        maxx=evaluations_dat['max_logl_spectra']
        um_xgrid=og_xgrid
    ax.plot(um_xgrid,med,color=colors[0], label='Median')
    ax.plot(um_xgrid,maxx,color='black', label='Max logL')
    ax.legend()
    ax.set_xlabel('Wavelength')
    return fig,ax
    
def plot_pressure_bands(evaluations_dat,colors,ax=None): 
    """
    Plots pressure bands and returns fix, axs
    """
    if ax is None: 
        fig,ax=plt.subplots(1,2) 
    else: 
        fig=None

    pressure = evaluations_dat['pressure']

    #temperature 
    for i in range(1,3):
        lo=evaluations_dat['bands_ptchem']['temperature'][f'{i}sig_lo']
        hi= evaluations_dat['bands_ptchem']['temperature'][f'{i}sig_hi']
            
        ax[0].fill_betweenx(evaluations_dat['pressure'], lo,
                                  hi,
                                   color=colors[i-1],alpha=0.4)
    ax[0].plot(evaluations_dat['max_logl_ptchem']['temperature'],evaluations_dat['pressure'],label='Max',color='black',linestyle='--')
    ax[0].plot(evaluations_dat['bands_ptchem']['temperature']['median'],evaluations_dat['pressure'],label='Median',color='black',linestyle='-.')
    
        
    #chemistry 
    imol=0
    for ikey in evaluations_dat['bands_ptchem'].keys(): 
        if 'temperature' not in ikey :
            for i in range(1,3):
                lo=evaluations_dat['bands_ptchem'][ikey][f'{i}sig_lo']
                hi= evaluations_dat['bands_ptchem'][ikey][f'{i}sig_hi']  
                ax[1].fill_betweenx(evaluations_dat['pressure'], lo,
                                      hi,
                                       color=colors[i-1],alpha=0.4)
                
            max = evaluations_dat['max_logl_ptchem'][ikey]
            ax[1].plot(max,pressure,color=pals.Light9[imol], label=ikey,linestyle='--')
            med = evaluations_dat['bands_ptchem'][ikey]['median']
            ax[1].plot(med,pressure,linestyle='-.',color=pals.Light9[imol])
            imol+=1
    for i in ax: i.set_yscale('log')
    ax[1].set_xscale('log')
    for i in ax: i.set_ylim([1e2,1e-6])
    for i in ax: i.legend()
    ax[0].set_ylabel('Pressure (bars)')
    ax[0].set_xlabel('Temperature (Kelvin)')
    ax[1].set_xlabel('Mixing Ratio (v/v)')
    return fig,ax
    

def data_output(evaluations, info, chisqout, filename,round=3,return_samples=True,
                     spectrum_tag='transit_depth',spectrum_unit='cm**2/cm**2',
                    author="",contact="",model_description="",code="PICASO"):
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
    if 'pressure' in evaluations.keys() :
        for ikey in evaluations['max_logl_ptchem'].keys():
            if 'temp' in ikey: 
                unit='Kelvin'
            else:
                unit='v/v'
                all_mols+=[ikey]
            if 'pressure' not in ikey: 
                data_vars['max_logl_'+ikey]=(["pressure"], evaluations['max_logl_ptchem'][ikey],{'units': unit})

    #maxlogL spectrum 
    data_vars['max_logl_'+spectrum_tag]=(["wavelength"], evaluations['max_logl_spectra'],{'units': spectrum_unit})

    #median ptchem
    if 'pressure' in evaluations.keys() :
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
    
    build_xarray = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=dict(author=author,#required
                   contact=contact,#required
                   model=model_description,
                   max_logl_chisq = chisqout['chisq_per_datapt'],
                   code=code, #required, in this case I used numpy to make my fake model.
                   max_logl_params=json.dumps({ip:maxlogl[i] for i,ip in enumerate(param_names)},cls=JsonCustomEncoder),
                   intervals_params=json.dumps({ip:strvals[ip] for i,ip in enumerate(param_names)},cls=JsonCustomEncoder),
                   molecules = all_mols,
                  ),
    )
    #output main datafile 
    build_xarray.to_netcdf(filename+'_median_and_max_logl.nc')

    #default figures 
    f,a=plot_spectra_bands(evaluations, pals.Blues[3],R=None)
    f.savefig(filename+'_spectra.png')

    if 'pressure' in evaluations.keys() :
        f,a=plot_pressure_bands(evaluations,pals.Greys[3])
        f.savefig(filename+'_ptchem.png')

    #return samples 
    pk.dump([info['param_names'],info['samples_equal']], open(filename+'_equally_weighted_samples.pk','wb'))

    #finally make corner plot 
    f, a = plot_pair(info['samples_equal'], info['param_names'])
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

def plot_pair(samples, params, pretty_labels=None,ranges=None,figsize=(11, 11), contour_cmap="GnBu",intervals=None):
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
        
    """
    az.style.use('default')

    if isinstance(pretty_labels,type(None)):
        pretty_labels=sorted(params)
    elif isinstance(pretty_labels,dict):
        pretty_labels=[pretty_labels[i] for i in sorted(params)]
    else: 
        raise Exception('Pretty labels must be None or dict')

    if isinstance(intervals,type(None)):
        intervals=None
    elif isinstance(intervals,dict):
        intervals=[intervals[i] for i in sorted(params)]
    else: 
        raise Exception('Intervals must be None or dict')

    if isinstance(ranges,type(None)):
        ranges=None
    elif isinstance(ranges,dict):
        ranges=[ranges[i] for i in sorted(params)]
    else: 
        raise Exception('Intervals must be None or dict')


    if len(params)*len(params)>40:
        az.rcParams["plot.max_subplots"] = len(params)*len(params)
    
    ax = az.plot_pair(
        {ip:samples[:,i] for i,ip in enumerate(params)},
        kind=["scatter", "kde"],
        kde_kwargs={"fill_last": False,
                    'hdi_probs':[0.393, 0.865, 0.989],  # 1, 2 and 3 sigma contours
                    'contourf_kwargs':{"cmap": contour_cmap},
                    'contour_kwargs':{"alpha": 0.5},
                   },
        scatter_kwargs={'color':'grey','alpha':0.5},
        marginal_kwargs={'color':'grey','kind':"hist"},#,'hist_kwargs':{'bins':15}
        marginals=True,
        point_estimate="median",
        figsize=figsize,
    )
    
    if not isinstance(pretty_labels,type(None)):
        assert len(pretty_labels)==len(params), "length of pretty labels must match length of parameters (second input)"
        for i,ip in enumerate(pretty_labels): 
            ax[i,0].set_ylabel(ip)
            ax[-1,i].set_xlabel(ip)
    
    if ((not isinstance(pretty_labels,type(None))) & (not isinstance(intervals,type(None)))):  
        assert len(intervals)==len(params), "length of intervals must match length of parameters (second input)"
        for i,ip in enumerate(pretty_labels): 
            ax[i,i].set_title(ip+'='+intervals[i])
    elif ((isinstance(pretty_labels,type(None))) & (not isinstance(intervals,type(None)))):  
        assert len(intervals)==len(params), "length of intervals must match length of parameters (second input)"
        for i,ip in enumerate(params): 
            ax[i,i].set_title(ip+'='+intervals[i])        
            
            
    
    if not isinstance(ranges,type(None)):
        assert len(ranges)==len(params), "length of ranges must match length of parameters (second input)"
        for i in range(len(ranges)): 
            ticks, form_ticks = stylized_ticks(ranges[i][0],ranges[i][1],2)
            ax[-1,i].set_xticks(ticks)
            ax[-1,i].set_xticklabels(form_ticks) 
            ax[-1,i].set_xlim(ranges[i][0],ranges[i][1])
        
            if i!=0:
                ax[i,0].set_yticks(ticks)
                ax[i,0].set_yticklabels(form_ticks) 
                ax[i,0].set_ylim(ranges[i][0],ranges[i][1])

        
    fig = ax.ravel()[0].figure
    return fig, ax


## Parameterizations 



class Parameterize():
    """
    """
    def __init__(self, load_cld_optical = None, mieff_dir = None):
        """
        picaso_inputs_class : class picaso.justdoit.inputs 
            PICASO inputs class 
        load_cld_optical : str
            Load optical constants for a certain cloud species 
            see virga.available to see available species you can load 
        mieff_dir : str 
            If a condensate species is supplied to load_cld_optical, then you must also supplie a mieff directory
            
        """

        if isinstance(load_cld_optical, (str,list)):
            if isinstance(load_cld_optical,str): load_cld_optical=[load_cld_optical]
            if isinstance(mieff_dir, str):
                if os.path.exists(mieff_dir):
                    self.qext, self.qscat, self.cos_qscat, self.nwave, self.radius, self.wave_in = {},{},{},{},{},{}
                    for isp in load_cld_optical:
                        self.qext[isp], self.qscat[isp], self.cos_qscat[isp], self.nwave[isp], self.radius[isp], self.wave_in[isp]=vj.get_mie(
                                    isp,directory=mieff_dir)
                else: 
                    raise Exception("path supplied through mieff_dir does not exist")
            else: 
                raise Exception("mieff_dir was not supplied as a str but needs to be if a condensate species was supplied through load_cld_optical")
            
        return 
        
    def add_class(self,picaso_inputs_class):
        """Add a picaso class that loads in the pressure grid (at the very least)

        Example
        -------
        start = jdi.inputs()
        start.add_pt(P=np.logspace(-6,3,91))
        param = Parameterize(load_cld_optical=['SiO2','Al2O3'],mieff_dir='/data/virga')
        param.add_class(start)
        """
        self.picaso = picaso_inputs_class
        self.pressure_level = picaso_inputs_class.inputs['atmosphere']['profile']['pressure'].values
        self.pressure_layer = np.sqrt(self.pressure_level [0:-1]*self.pressure_level [1:])
        self.nlevel = len(self.pressure_level )
        self.nlayer = self.nlevel -1 

    def get_particle_dist(self,species,distribution,
                  lognorm_kwargs = {'sigma':np.nan, 'lograd[cm]':np.nan}, 
                  hansen_kwargs={'b':np.nan,'lograd[cm]':np.nan}):
        logradius = np.log10(self.radius[species])
        
        if 'lognorm' in distribution:
            sigma=lognorm_kwargs['sigma']
            lograd=lognorm_kwargs['lograd[cm]']
            
            if np.isnan(sigma):
                raise Exception('lognorm_kwargs have not been defined')
            
            dist = (1/(sigma * np.sqrt(2 * np.pi)) *
                       np.exp( - (logradius - lograd)**2 / (2 * sigma**2)))
        elif 'hansen' in distribution: 
            a = 10**hansen_kwargs['lograd[cm]']
            b = hansen_kwargs['b']
            dist = (10**self.radius[species])**((1-3*b)/b)*np.exp(-self.radius[species]/(a*b))
        else: 
            raise Exception("Only lognormal and hansen distributions available")        
        
        return dist 

    def flex_fsed_cloud(self, species, base_pressure, ndz, fsed, distribution, 
                  lognorm_kwargs = {'sigma':np.nan, 'lograd[cm]':np.nan}, 
                  hansen_kwargs={'b':np.nan,'lograd[cm]':np.nan}): 
        """
        Given a base_pressure and fsed to set the exponential drop of the cloud integrate a particle 
        radius distribution via gaussian or hansen distributions to get optical properties in picaso 
        format. 

        Parameters
        ----------
        species : str 
            Name of species. Should already have been preloaded via Parameterize options in load_cld_optical
        base_pressure : float 
            base of the cloud deck in bars 
        ndz : float 
            number density of the cloud deck cgs 
        fsed : float 
            sedimentation efficiency 
        distribution : str 
            either lognormal or hansen 
        lognorm_kwargs : dict 
            diectionary with the format: {'sigma':np.nan, 'lograd[cm]':np.nan}
            lograd[cm] median particle radius in cm 
            sigma width of the distribtuion must be >1 
        hansen_kwargs : dict 
            dictionary with the format: {'b':np.nan,'lograd[cm]':np.nan}
            lograd[cm] and b from Hansen+1971: https://web.gps.caltech.edu/~vijay/Papers/Polarisation/hansen-71b.pdf
            lograd[cm] = a = effective particle radius 
            b = varience of the particle radius 

        Returns 
        -------
        pandas.DataFrame 
            PICASO formatted cld input dataframe 
        """
        scale_h = 10 #just arbitrary as this gets fit for via fsed and ndz 
        z = np.linspace(100,0,self.nlayer)
        
        dist = self.get_particle_dist(species,distribution,lognorm_kwargs,hansen_kwargs)
            
        opd,w0,g0,wavenumber_grid=vj.calc_optics_user_r_dist(self.wave_in[species], ndz ,self.radius[species], u.cm,
                                                              dist, self.qext[species], self.qscat[species], self.cos_qscat[species])
        
        opd_h = self.pressure_layer*0+10
        opd_h[base_pressure<self.pressure_layer]=0
        opd_h[base_pressure>=self.pressure_layer]=opd_h[base_pressure>=self.pressure_layer]*np.exp(
                              -fsed*z[base_pressure>=self.pressure_layer]/scale_h)
        opd_h = opd_h/np.max(opd_h)
        
        df_cld = picaso_format(opd, w0, g0, wavenumber_grid, self.pressure_layer, 
                                          p_bottom=base_pressure,p_decay=opd_h)

        return df_cld 
    flex_cloud =  flex_fsed_cloud  
    def brewster_mie_cloud(self, species, distribution, decay_type,
                  lognorm_kwargs = {'sigma':np.nan, 'lograd[cm]':np.nan}, 
                  hansen_kwargs={'b':np.nan,'lograd[cm]':np.nan},
                  slab_kwargs={'ptop':np.nan,'dp':np.nan, 'reference_tau':np.nan},
                  deck_kwargs={'ptop':np.nan,'dp':np.nan}): 
        """
        Given a base_pressure and fsed to set the exponential drop of the cloud integrate a particle 
        radius distribution via gaussian or hansen distributions to get optical properties in picaso 
        format. 

        Parameters
        ----------
        species : str 
            Name of species. Should already have been preloaded via Parameterize options in load_cld_optical
        base_pressure : float 
            base of the cloud deck in bars 
        ndz : float 
            number density of the cloud deck cgs 
        fsed : float 
            sedimentation efficiency 
        distribution : str 
            either lognormal or hansen 
        lognorm_kwargs : dict 
            diectionary with the format: {'sigma':np.nan, 'lograd[cm]':np.nan}
            lograd[cm] median particle radius in cm 
            sigma width of the distribtuion must be >1 
        hansen_kwargs : dict 
            dictionary with the format: {'b':np.nan,'lograd[cm]':np.nan}
            lograd[cm] and b from Hansen+1971: https://web.gps.caltech.edu/~vijay/Papers/Polarisation/hansen-71b.pdf
            lograd[cm] = a = effective particle radius 
            b = varience of the particle radius 

        Returns 
        -------
        pandas.DataFrame 
            PICASO formatted cld input dataframe 
        """
        
        dist = self.get_particle_dist(species,distribution,lognorm_kwargs,hansen_kwargs)
            
        opd,w0,g0,wavenumber_grid=vj.calc_optics_user_r_dist(self.wave_in[species], 1 ,self.radius[species], u.cm,
                                                              dist, self.qext[species], self.qscat[species], self.cos_qscat[species])
        
        if decay_type == 'slab':
            opd_profile = self.slab_decay(**slab_kwargs)
        elif decay_type == 'deck':
            opd_profile = self.deck_decay(**deck_kwargs)
        
        df = picaso_format(opd, w0, g0, wavenumber_grid, self.pressure_layer, opd_profile=opd_profile)

        return df 
    
    def brewster_grey_cloud(self, decay_type, alpha, ssa, reference_wave=1,
                  slab_kwargs={'ptop':np.nan,'dp':np.nan, 'reference_tau':np.nan},
                  deck_kwargs={'ptop':np.nan,'dp':np.nan}): 
        """
        Creates grey cloud with either slab or deck decay and an alpha wavelength scaling 

        Parameters
        ----------
        decay_type: str
            One of 'deck' or 'slab'
        ssa: float
            Single Scattering Albedo: can have values from 0 to 1
        alpha: float
            set to 0 for grey cloud
 
        Returns 
        -------
        pandas.DataFrame 
            PICASO formatted cld input dataframe 
        """
        
        wavenumber_grid =get_cld_input_grid()
        wavelength= 1e4/wavenumber_grid

        if decay_type == 'slab':
            opd_profile = self.slab_decay(**slab_kwargs)
        elif decay_type == 'deck':
            opd_profile = self.deck_decay(**deck_kwargs)

        wave_dependent_opd =  np.concatenate([opd_profile[i]*(wavelength/reference_wave)**(-alpha) for i in range(self.nlayer)])
        wvnos =  np.concatenate([wavenumber_grid for i in range(self.nlayer)])
        pressures =  np.concatenate([[self.pressure_layer[i]]*len(wavelength) for i in range(self.nlayer)])
        w0=wave_dependent_opd*0+ssa
        g0=wave_dependent_opd*0
        df=pd.DataFrame({
                'opd':wave_dependent_opd,
                'g0':g0,
                'w0':w0,
                'wavenumber':wvnos,
                'pressure':pressures
            })

        return df 

    def deck_decay(self,ptop, dp=0.005): 
        """
        Emualates brewster opacity decay for the deck model 
        
        Parameters 
        ----------
        ptop : float 
            ptop is log pressure (bar) at which tau of cloud ~1 
        dp : float 
            dtau / dP = const * exp((P-P0) / pressure_scale)
        """
        pressure_layer=self.pressure_layer
        nlayer = len(self.pressure_layer)
        opd_by_layer = np.zeros(nlayer)

        pressure_top = 10**ptop

        pressure_scale = ((pressure_top * 10.**dp) - pressure_top)  / 10.**dp
        const = 1. / (1 - np.exp(-pressure_top / pressure_scale))

        for i in range (0,nlayer):
            p_grid_top, p_grid_bot = atlev(i,pressure_layer)
            # now get dtau for each layer, where tau = 1 at pressure_top
            term1 = (p_grid_bot - pressure_top) / pressure_scale
            term2 = (p_grid_top - pressure_top) / pressure_scale
            if (term1 > 10 or term2 > 10):
                #sets large optical depths to 100 
                opd_by_layer[i] = 100.00
            else:
                opd_by_layer[i] = const * (np.exp(term1) - np.exp(term2))
        
        return opd_by_layer

    def slab_decay(self, ptop, dp=0.005, reference_tau=1): 
        """
        Modeled after brewster slabs see Eqn 13 and 14 Whiteford et al. 

        Parameters 
        ----------
        ptop : float 
            pressure top in log bars 
        dp : float 
            pressure thickness in dex bars, default - 0.005
        reference_tau : float 
            reference tau for 1 micron 

        Returns 
        -------
        optical depth per layer as a function of layer  
        """
        pressure = self.pressure_layer #levels 
        nlayer = len(pressure)

        opd_by_layer = np.zeros(nlayer)

        pressure_top = 10**ptop #p1 brewster e.g., 1e-3
        pressure_bottom = pressure_top * 10.**dp #p2 brewster e.g. 1e-3*10^2 = 1e-1

        #find index of layer for pressure top and pressure bottom 
        index_top = np.argmin(abs(np.log(pressure) - np.log(pressure_top)))
        index_bottom = np.argmin(abs(np.log(pressure) - np.log(pressure_bottom)))
        if index_top == index_bottom: 
            raise Exception('dp entered was not large enough to create a cloud given the pressure grid spacing')

        #compute tau scaling 
        tau_scaling = reference_tau / (pressure_bottom**2 - pressure_top**2)

        _ , p_grid_bot = atlev(index_top,pressure)
        opd_by_layer[index_top] = tau_scaling * (p_grid_bot**2 - pressure_top**2)  

        p_grid_top , _ = atlev(index_bottom,pressure)
        opd_by_layer[index_bottom] = tau_scaling * (pressure_bottom**2 - p_grid_top**2)        

        for i in range (index_top+1,index_bottom):
            p_grid_top,p_grid_bot = atlev(i,pressure)
            opd_by_layer[i] = tau_scaling * (p_grid_bot**2 - p_grid_top**2)

        return opd_by_layer

    def free_constant_abundance(self,  species):
        ''''
        Abundance profile

        Parameters
        ----------
        species: dict
            Dictionary containing the species and their abundances. Should 
            also contain background gases and their ratios. 
            Example: species=dict(H2O=dict(value=1e-4, unit='v/v'), background=dict(gases=['H2', 'He'], ratios=[0.85, 0.15]))

        Return
        ------
        Data frame with chemical abundances per level
        '''

        pressure=self.pressure_level
        nlevels=len(pressure)

        # Initialize dictionary to put in the datafram
        mixing_ratios=dict(pressure=pressure)

        # Keep track of the total abundances to make sure they add to 1
        total_abundance=0
        
        for i in species.keys():
            if i!='background':
                mixing_ratios[i]=np.ones(nlevels)*species[i]['value']
                total_abundance+=species[i]['value']

        # Add background gases
        n_background=len(species['background']['gases'])
        for i in range(n_background):
            mol=species['background']['gases'][i]
            abun=species['background']['ratios'][i]
            mixing_ratios[mol]=np.ones(nlevels)*(1-total_abundance)*abun
        
        self.picaso.inputs['atmosphere']['profile'] = pd.DataFrame(mixing_ratios)
        
        return pd.DataFrame(mixing_ratios)

    def madhu_seager_09_noinversion(self, alpha_1, alpha_2, P1, P3, T3, beta=0.5):
        """"
        Implements the temperature structure parameterization from Madhusudhan & Seager (2009)

        Parameters
        -----------

        Returns
        -------
        Temperature per layer
        """

        pressure = self.pressure_level
        nlevel = len(pressure)

        temp_by_level = np.zeros(nlevel)

        # Set T1 from T3
        T1 = T3 - (np.log(P3/P1) / alpha_2)**(1/beta)
        # Set T0 from T1
        T0 = T1 - (np.log(P1/P0) / alpha_1)**(1/beta)

        P0 = pressure[0]

        # Set pressure ranges
        layer_1=(pressure<P1)
        layer_2=(pressure>=P1)*(pressure<P3)
        layer_3=(pressure>=P3)

        # Define temperature at each pressure range
        temp_by_level[layer_1] = T0 + ((1/alpha_1)*np.log(pressure[layer_1]/P0))**(1/beta)
        temp_by_level[layer_2] = T1 + ((1/alpha_2)*np.log(pressure[layer_2]/P1))**(1/beta)
        temp_by_level[layer_3] = T3

        temp_by_level = convolve(temp_by_level,Gaussian1DKernel(5),boundary='extend')

        return pd.DataFrame(dict(pressure=pressure, temperature=temp_by_level))
    
    def madhu_seager_09_inversion(self, alpha_1, alpha_2, P1, P2, P3, T3, beta=0.5):
        """"
        Implements the temperature structure parameterization from Madhusudhan & Seager (2009)
          allowing for inversions

        Parameters
        -----------

        Returns
        -------
        Temperature per layer
        """

        pressure = self.pressure_level
        nlevel = len(pressure)

        temp_by_level = np.zeros(nlevel)

        P0 = pressure[0]

        # Set pressure ranges
        layer_1=(pressure<P1)
        layer_2=(pressure<P3)*(pressure>=P1)
        layer_3=(pressure>=P3)

        # Define temperatures at boundaries to ensure continuity
        T2 = T3 - (np.log(P3/P2) / alpha_2)**(1/beta)
        T1 = T2 + (np.log(P1/P2) / alpha_2)**(1/beta)
        T0 = T1 - (np.log(P1/P0) / alpha_1)**(1/beta)

        # Define temperature at each pressure range
        temp_by_level[layer_1] = T0 + (np.log(pressure[layer_1]/P0)/alpha_1)**(1/beta)
        temp_by_level[layer_2] = T2 + (np.log(pressure[layer_2]/P2)/alpha_2)**(1/beta)
        temp_by_level[layer_3] = T3

        temp_by_level = convolve(temp_by_level,Gaussian1DKernel(5),boundary='extend')

        return pd.DataFrame(dict(pressure=pressure, temperature=temp_by_level))
    
    def knot_profile(self,  P_knots, T_knots, interpolation='brewster'):
        """"
        Knot-based temperature profile. Implements different types of interpolation.

        Parameters
        -----------

        Returns
        -------
        Temperature per layer
        """

        pressure = self.pressure_level
        nlevel = len(pressure)

        temp_by_level = np.zeros(nlevel)

        # Interpolation requires pressures to be sorted from lowest to highest
        order = np.argsort(P_knots)
        P_knots=np.array(P_knots)[order]
        T_knots=np.array(T_knots)[order]
        
        # Perform the interpolation
        if interpolation=='brewster':
            interpolator = interpolate.splrep(np.log10(P_knots), T_knots, s=0)
            temp_by_level=np.abs(interpolate.splev(np.log10(pressure),interpolator,der=0))
        elif interpolation=='linear':
            interpolator = interpolate.interp1d(np.log10(P_knots), T_knots, kind='linear', bounds_error=False, fill_value='extrapolate')
            temp_by_level = interpolator(np.log10(pressure))
        elif interpolation=='quadratic_spline':
            assert len(P_knots)>=3, 'Quadratic splines require at least 3 knots'
            interpolator = interpolate.interp1d(np.log10(P_knots), T_knots, kind='quadratic', bounds_error=False, fill_value='extrapolate')
            temp_by_level = interpolator(np.log10(pressure))
        elif interpolation=='cubic_spline':
            assert len(P_knots)>=4, 'Cubic splines require at least 4 knots'
            interpolator = interpolate.interp1d(np.log10(P_knots), T_knots, kind='cubic', bounds_error=False, fill_value='extrapolate')
            temp_by_level = interpolator(np.log10(pressure))
        elif getattr(interpolate, interpolation, np.nan)!=np.nan:
            interpolator = getattr(interpolate, interpolation)
            interpolator(np.log10(P_knots), T_knots, *scipy_interpolate_kwargs)
        else:
            raise Exception(f'Unknown interpolation method \'{interpolation}\'')

        return pd.DataFrame(dict(pressure=pressure, temperature=temp_by_level))

def atlev(l0,pressure_layer):
    nlayers = pressure_layer.size
    if (l0 <= nlayers-2):
        pressure_top = np.exp(((1.5)*np.log(pressure_layer[l0])) - ((0.5)*np.log(pressure_layer[l0+1])))
        pressure_bottom = np.exp((0.5)*(np.log(pressure_layer[l0] * pressure_layer[l0+1])))
    else:
        pressure_top = np.exp((0.5 * np.log(pressure_layer[l0-1] * pressure_layer[l0])))
        pressure_bottom = pressure_layer[l0]**2 / pressure_top

    return pressure_top, pressure_bottom

def picaso_format(opd, w0, g0, wavenumber_grid, pressure_grid ,
                       p_bottom=None,p_top=None,p_decay=None,opd_profile=None):
    """
    Sets up a PICASO-readable dataframe that inserts a wavelength dependent aerosol layer at the user's 
    given pressure bounds, i.e., a wavelength-dependent slab of clouds or haze.
    
    Parameters
    ----------
    p_bottom : float 
        the cloud/haze base pressure
        the upper bound of pressure (i.e., lower altitude bound) to set the aerosol layer. (Bars)
    opd : ndarray
        wavelength-dependent optical depth of the aerosol
    w0 : ndarray
        wavelength-dependent single scattering albedo of the aerosol
    g0 : ndarray
        asymmetry parameter = Q_scat wtd avg of <cos theta>
    wavenumber_grid : ndarray
        wavenumber grid in (cm^-1) 
    pressure_grid : ndarray
        bars, user-defined pressure grid for the model atmosphere
    p_top : float
         bars, the cloud/haze-top pressure
         This cuts off the upper cloud region as a step function. 
         You must specify either p_top or p_decay. 
    p_decay : ndarray
        noramlized to 1, unitless
        array the same size as pressure_grid which specifies a 
        height dependent optical depth. The usual format of p_decay is 
        a fsed like exponential decay ~np.exp(-fsed*z/H)


    Returns
    -------
    Dataframe of aerosol layer with pressure (in levels - non-physical units!), wavenumber, opd, w0, and g0 to be read by PICASO
    """
    if isinstance(p_bottom, type(None)): 
        p_bottom = np.max(pressure_grid)+10#arbitrarily big to make sure float comparison includes clouds
        
    if (isinstance(p_top, type(None)) & isinstance(p_decay, type(None)) & isinstance(opd_profile, type(None))): 
        raise Exception("Must specify cloud top pressure via p_top, or the vertical pressure decay via p_decay, or an opd profile via opd_profile")
    
    if (isinstance(p_top, type(None))): 
        p_top = 1e-10#arbitarily small pressure to make sure float comparison doest break


    df = pd.DataFrame(index=[ i for i in range(pressure_grid.shape[0]*opd.shape[0])], columns=['pressure','wavenumber','opd','w0','g0'])
    i = 0 
    LVL = []
    WV,OPD,WW0,GG0 =[],[],[],[]
    
    # this loops the opd, w0, and g0 between p and dp bounds and put zeroes for them everywhere else
    for j in range(pressure_grid.shape[0]):
           for w in range(opd.shape[0]):
                #stick in pressure bounds for the aerosol layer:
                if p_top <= pressure_grid[j] <= p_bottom:
                    LVL+=[pressure_grid[j]]
                    WV+=[wavenumber_grid[w]]
                    if (isinstance(p_decay,type(None)) & isinstance(opd_profile,type(None))):
                        OPD+=[opd[w]]
                    elif not (isinstance(p_decay,type(None))): 
                        OPD+=[p_decay[j]/np.max(p_decay)*opd[w]]
                    elif not (isinstance(opd_profile,type(None))): 
                        OPD+=[opd_profile[j]*(opd[w]/np.max(opd))]
                    WW0+=[w0[w]]
                    GG0+=[g0[w]]
                else:
                    LVL+=[pressure_grid[j]]
                    WV+=[wavenumber_grid[w]]
                    OPD+=[opd[w]*0]
                    WW0+=[w0[w]*0]
                    GG0+=[g0[w]*0]       
                    
    df.iloc[:,0 ] = LVL
    df.iloc[:,1 ] = WV
    df.iloc[:,2 ] = OPD
    df.iloc[:,3 ] = WW0
    df.iloc[:,4 ] = GG0
    return df
