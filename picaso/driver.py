"""
The content in here is beta and should not be used for published works until the 
release of PICASO 5. Please work with the developers if using this before the PICASO 5 release: 
Authors: Natasha Batalha, Francisco Ardevol Martinez & Sierra Janson
"""
from .justdoit import *
from .justplotit import *
from .parameterizations import Parameterize,cloud_averaging

import warnings
import tomllib 
import toml
import shutil
from collections.abc import Mapping
from scipy import stats
import scipy.interpolate as sci
from scipy.special import erf
import dill
import dynesty
import dynesty.utils
from functools import partial
import sys
from schwimmbad import MPIPool,choose_pool
dynesty.utils.pickle_module = dill
from astropy.io import fits
import spectres
import contextlib

#all these come from justdoit
#import os
#import numpy as np
#import pandas as pd
#import xarray as xr
#from astropy import units as u
#import copy

# import ultranest
# from mpi4py import MPI
# import dynesty


chem_options = ['visscher', 'free', 'chemeq_on_the_fly','userfile', 'xarray_grid']
cloud_options = ['brewster_grey', 'brewster_mie', 'virga', 'flex_fsed', 'hard_grey', 'userfile']
pt_options = ['userfile','isothermal', 'knots', 'guillot', 'sonora_bobcat',  'madhu_seager_09_inversion','madhu_seager_09_noinversion', 'zj24', 'xarray_grid'] #, 'molliere_20', 'Kitzman_20', 

# Mappings for observation types to picaso calculation types
OBSERVATION_CALC_MAP = {
    'thermal': 'thermal',
    'fpfs_thermal': 'thermal',
    'albedo': 'reflected',
    'fpfs_reflected': 'reflected',
    'transit_depth': 'transmission',
    'reflected': 'reflected',
    'transmission': 'transmission'
}


def run(driver_file=None,driver_dict=None,return_class=False):
    if isinstance(driver_file,str):
        with open(driver_file, "rb") as f:
            config = tomllib.load(f)
    elif isinstance(driver_dict,dict):
        config = driver_dict
    else: 
        raise Exception('Could not interpret either driver file or dictionary input')
    
    #PRELOAD OPACITIES OR OPTICAL CONSTANTS
    preload_cloud_miefs = find_values_for_key(config ,'condensate')
    virga_mieff   = config['OpticalProperties'].get('virga_mieff',None)
    #if the above are both blank then this is just returning a set of functions
    model_dir = config['temperature'].get('xarray_grid',{}).get('model_dir',None)
    param_tools = Parameterize(load_cld_optical=preload_cloud_miefs,
                               mieff_dir=virga_mieff,
                               model_dir=model_dir)
    
    #setup opacity outside main run
    opacity = opannection(
        filename_db=config['OpticalProperties']['opacity_file'], #database(s)
        method=config['OpticalProperties']['opacity_method'], #resampled, preweighted, resortrebin
        **config['OpticalProperties']['opacity_kwargs'] #additonal inputs 
        )
     
    if config['calc_type'] =='spectrum':
        picaso_class = setup_spectrum_class(config,opacity,param_tools)
        output =  picaso_class.spectrum(opacity, full_output=True, 
                                        calculation = OBSERVATION_CALC_MAP.get(config['observation_type'], config['observation_type'])) 
    elif config['calc_type']=='retrieval':
        #Create output directory
        os.makedirs(config['InputOutput']['retrieval_output'], exist_ok=True)

        # copy input toml into output folder for reproducibility
        output_file_name = config['InputOutput']['retrieval_output']+"/inputs.toml"
        if driver_file is not None and not config.get('retrieval', {}).get('sampler', {}).get('resume', False):
            shutil.copy(driver_file, output_file_name)

        #run retrieval
        output = retrieve(config, param_tools)

    ### I made these # because they stopped the run fucntion from doing return out and wouldn't let me use my plot PT fucntion
    elif config['calc_type']=='climate':
        raise Exception('WIP not ready yet')
        out = climate(config)
    
    if return_class:
        return output ,picaso_class
    else: 
        return output

def is_valid_astropy_unit(unit_str):
    try:
        u.Unit(unit_str)  # will raise ValueError if invalid
        return True
    except (ValueError, TypeError):
        return False

def convert_flux_units(xgrid, flux, to_f_unit, flux_err=None, xgrid_unit='cm^(-1)', f_unit='erg*cm^(-3)*s^(-1)'): 
    """
    Converts both flux and its associated error to new units using synphot's 
    SourceSpectrum. Automatically handles axis flipping if the input grid 
    direction needs to match synphot's internally sorted waveset.
    
    Parameters
    ----------
    xgrid : ndarray
        Wavelength or wavenumber array 
    flux : ndarray
        Flux array 
    to_f_unit : str 
        Astropy approved string unit for the output flux
    flux_err : ndarray, optional
        Uncertainty/error array on the flux. If provided, it is scaled 
        identically to the flux.
    xgrid_unit : str, default 'cm^(-1)'
        Current coordinate units.
    f_unit : str, default 'erg*cm^(-3)*s^(-1)'
        Current flux units.
        
    Returns
    -------
    astropy.units.Quantity or tuple
        If flux_err is None: returns converted flux.
        If flux_err is provided: returns tuple of (converted_flux, converted_flux_err).
    """
    # 1. Setup the SourceSpectrum with actual flux
    ST_SS = SourceSpectrum(
        Empirical1D, 
        points=xgrid * u.Unit(xgrid_unit), 
        lookup_table=flux * u.Unit(f_unit)
    )
    
    # 2. Evaluate at the native waveset to get converted flux
    flux_converted = ST_SS(ST_SS.waveset, flux_unit=u.Unit(to_f_unit))
    
    # 3. Handle original axis sorting rules
    # If original units were inverse cm and ordered increasing, synphot flips 
    # the internal waveset to sort by increasing wavelength. We match that logic.
    if (xgrid_unit == 'cm^(-1)') and (xgrid[1] > xgrid[0]):
        flux_converted = flux_converted[::-1]

    # 4. Handle error conversion if provided
    if flux_err is not None:
        # Determine the exact wavelength-dependent scale factor applied by synphot
        # (flux_converted / flux) handles any non-trivial per-pixel unit scaling
        scale_factor = flux_converted.value / flux
        
        # Scale the error and attach the destination unit
        flux_err_converted = (flux_err * scale_factor) * u.Unit(to_f_unit)
        
        return flux_converted, flux_err_converted
        
    return flux_converted

def _err_inf_to_model_variance(err_inf, xgrid, obs_key, config):
    """
    Convert error-inflation variance from the input data flux units to PICASO units.

    The retrieval prior is specified in the same variance units as the input data
    file, while parse_data converts the data/error arrays to erg/(s cm2 cm).
    """
    observations_config = config.get('ObservationData', {})
    input_unit = observations_config.get('data_unit')
    if input_unit is None:
        return err_inf

    target_unit = u.Unit('erg/(s*cm**2*cm)')
    try:
        source_unit = u.Unit(input_unit)
    except Exception:
        return err_inf

    if source_unit == target_unit:
        return err_inf

    cache = config.setdefault('_err_inf_variance_scale_cache', {})
    if obs_key not in cache:
        try:
            scale = convert_flux_units(
                xgrid,
                np.ones_like(xgrid, dtype=float),
                to_f_unit=target_unit,
                f_unit=input_unit,
            ).value
            cache[obs_key] = np.asarray(scale, dtype=float) ** 2
        except Exception as exc:
            warnings.warn(
                f"Could not convert err_inf for {obs_key} from {input_unit} "
                f"to {target_unit}; using raw err_inf value. Error: {exc}",
                UserWarning,
            )
            cache[obs_key] = None

    variance_scale = cache[obs_key]
    if variance_scale is None:
        return err_inf
    return err_inf * variance_scale

def convert_to_wavenumber(value, input_unit):
    """
    Converts a given grid from a specified Astropy unit (or string representation)
    to wavenumber in inverse centimeters (cm^-1).
    
    Parameters:
    -----------
    value : float, numpy.ndarray, or astropy.units.Quantity
        The grid values to convert.
    input_unit : str or astropy.units.Unit
        The unit of the input values (e.g., 'micron', 'nm', 'Hz', 'THz', 'eV', u.micron).
        
    Returns:
    --------
    array
        The converted values in cm^-1 (with astropy unit attached).
        Use .value to extract just the raw number/array if needed.
    """
    # 1. Ensure input_unit is an actual Astropy Unit object
    if isinstance(input_unit, str):
        # Map common custom shorthand strings if needed, otherwise parse directly
        if input_unit.strip().lower() in ['inv_cm', 'cm-1']:
            unit_obj = u.cm**-1
        else:
            unit_obj = u.Unit(input_unit)
    else:
        unit_obj = input_unit

    # 2. Attach the unit to the value if it doesn't already have one
    if not isinstance(value, u.Quantity):
        quantity = value * unit_obj
    else:
        quantity = value.to(unit_obj) # Ensure it matches the specified input_unit

    # 3. Perform the conversion using spectral equivalencies
    # This automatically handles wavelength <-> frequency <-> energy conversions
    wavenumber_quantity = quantity.to(u.cm**-1, equivalencies=u.spectral())
    
    return wavenumber_quantity.value

def parse_data(filenames, coord, data,  error, coord_unit=None, data_unit=None):
    """
    Parses observational data files (ASCII or xarray) for PICASO.

    Parameters
    ----------
    filenames : list of str or str
        One path or list of paths to the data files 
    coord : str 
        name of coordinate (should map to column name or xarray coord)
    data : str 
        name of data to fit (should map to column name or xarray data_var)
    error : str 
        name of data error to use (should map to column name or xarray coord)
    coord_unit : str 
        Name of astropy unit for coordinate
    data_unit : str 
        Name of astropy unit for data

    Returns
    -------
    dict
        Dictionary where keys are filenames without extensions and values are
        lists of [wavenumber, to_fit, to_fit_error].
    """
    returns = {}
    if isinstance(filenames, str): filenames = [filenames]

    for filename in filenames:
        ext = os.path.splitext(filename)[1].lower()
        # 1) determine if the files are comma separated ascii files or xarray
        if ext in ['.csv', '.txt', '.ascii', '.dat']:
            # 2) if they are comma separated ascii files the user also needs to specify the units of the data.
            # the function should check that units were specified via kwarg input.
            if ((data_unit is None) or (coord_unit is None)):
                raise ValueError(f"Units must be specified for ASCII file: {filename}")
            
            df = pd.read_csv(filename)
            
            # 3) convert the ascii read data to an xarray with unit specification 
            # where all over columns are converted to data_vars with their column name
            ds = xr.Dataset.from_dataframe(df.set_index(coord))
            ds.coords[coord].attrs['units'] = coord_unit
            ds[data].attrs['units'] = data_unit
            ds[error].attrs['units'] = data_unit
        else:
            # Assume xarray
            ds = xr.load_dataset(filename)

        # 4) the user must specify the name of the data it wants to fit through a variable called "data" 
        # data variable should now match the name of a data_var 
        # there should also be a column specifying the error
        if data not in ds.data_vars:
            raise ValueError(f"Variable '{data}' not found in {filename}")
        if error not in ds.data_vars:
            raise ValueError(f"Error variable '{error}' not found in {filename}")

        # 5) Add wavenumber coordinate called wavenumber
        # Could be redundant but that is okay just a verification
        wavenumber = convert_to_wavenumber(ds.coords[coord].values, ds.coords[coord].attrs['units'])
        ds = ds.assign_coords(wavenumber=([coord], wavenumber))
        ds.coords['wavenumber'].attrs['units'] = 'cm^(-1)'

        # 6) if data unit is flux it needs to be converted to picaso units erg/cm3/s
        current_unit_str = ds[data].attrs.get('units', '')
        # Lets try and convert the data and add it to the data bundle
        try: 
            current_unit = u.Unit(current_unit_str)
            target_unit = u.Unit('erg/(s*cm**2*cm)')
            flux_converted, flux_err_converted = convert_flux_units(
                ds.coords['wavenumber'].values,
                ds.data_vars[data].values,
                to_f_unit = target_unit,
                flux_err = ds.data_vars[error].values,
                f_unit = current_unit_str
            )
            ds[data].values = flux_converted
            ds[error].values = flux_err_converted
            ds[data].attrs['units'] = 'erg/(s*cm**2*cm)'
            ds[error].attrs['units'] = 'erg/(s*cm**2*cm)'
        except: 
            print('Flux unit converstion failed so assuming unitless or not convertable data')
            pass 

        ds_sorted = ds.sortby('wavenumber')
        
        name = os.path.splitext(os.path.basename(filename))[0]
        returns[name] = [ds_sorted.coords['wavenumber'].values, 
                         ds_sorted[data].values, 
                         ds_sorted[error].values]
        
    return returns

def get_data(config): 
    """
    Processes observational data using parse_data and instrument properties 
    for resolution convolution. 

    Currently only supports jwst instrument names. can see options using instrument options 
    
    Returns
    -------
    data_dict, resolution_conv_dict 
        data dictionary with wavenumber, y, e and matching dictionary with 
        resolution as a fucntion of wavelength (micron) for instruments
    """
    observations_config = config.get('ObservationData').copy()
    
    instruments = observations_config.pop('instruments', [])
    convolve_observations = observations_config.pop('convolve', False)
    filenames = observations_config.get('filenames', [])
    if isinstance(filenames, str):
        filenames = [filenames]
    coord = observations_config.get('coord')
    
    data_dict = parse_data(**observations_config)

    # remove NaNs from data and return a user warning
    for key in data_dict.keys():
        wno, y, e = data_dict[key]
        nan_mask = np.isnan(y) | np.isnan(e)
        if np.any(nan_mask):
            nan_count = np.sum(nan_mask)
            total_count = len(y)
            warnings.warn(f"Removed {nan_count} NaNs out of {total_count} data points from {key}", UserWarning)
            data_dict[key] = [wno[~nan_mask], y[nan_mask == False], e[nan_mask == False]]

    resolution_dict = {}
    for filename in filenames:
        ext = os.path.splitext(filename)[1].lower()
        if ext in ['.csv', '.txt', '.ascii', '.dat']:
            continue
        ds = xr.load_dataset(filename)
        if 'R' not in ds:
            continue

        coord_values = ds.coords[coord].values
        coord_unit = ds.coords[coord].attrs.get('units', observations_config.get('coord_unit'))
        if coord_unit:
            wno = convert_to_wavenumber(coord_values, coord_unit)
        else:
            wno = 1e4 / coord_values

        sort_indices = np.argsort(wno)
        name = os.path.splitext(os.path.basename(filename))[0]
        resolution_dict[name] = (wno[sort_indices], ds['R'].values[sort_indices], convolve_observations)

    if len(instruments)>0:
        if len(instruments) != len(data_dict.keys()):
            raise Exception("need to input the same number of instrument keys for convolution as filenames. if you plan to use the same instrument for all files then enter multiple entries (e.g., ['jwst nirspec prism','jwst nirspec prism'])")
        else: 
            
            for i,name in enumerate(data_dict.keys()):
                if ((name not in resolution_dict) & ('None' not in str(instruments[i]))):
                    wno_inst, R_inst = get_instrument_R_fits(instruments[i])
                    resolution_dict[name] = (wno_inst, R_inst, convolve_observations)
        
    # Leaving this out for now until there is rationale 
    # to enforce mapping. For now parse_data will try and 
    # convert units and units for non flux type data is not converted

    # obs_type = config['observation_type']
    # if obs_type=='thermal':
    #     to_fit = 'flux'
    # elif obs_type=='reflected':
    #     to_fit = 'flux'
    # elif obs_type=='transmission':
    #     to_fit = 'transit_depth'
        
    return data_dict,resolution_dict

#retrieval funs
def get_data_old(config): 
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
    # datadict = {}
    obs_type = config['observation_type']
    observations = config['InputOutput']['observation_data']
    
    ## this could be another entry in the toml file to give extra flexibility
    calc_type = OBSERVATION_CALC_MAP.get(obs_type, obs_type)
    if calc_type=='thermal':
        to_fit = 'flux'
    elif calc_type=='reflected':
        to_fit = 'flux'
    elif calc_type=='transmission':
        to_fit = 'transit_depth'

    returns = {'_R': {}}
    for i,key in enumerate(observations):
        #load observation file
        dat = xr.load_dataset(observations[key])

        #check for valid astropy unit 
        # if is_valid_astropy_unit(dat.data_vars[to_fit].unit):
        #     unity = u.Unit(dat.data_vars[to_fit].unit)
        # else:
        #     raise Exception('Not a valid astropy unit for data_vars')

        # if is_valid_astropy_unit(dat.data_vars[to_fit].unit):
        #     unitx = u.Unit(dat.data_vars[to_fit].unit)
        # else:
        #     raise Exception("Not a valid unit for coords")

        final = pd.DataFrame(dict(x=dat.coords['wavelength'].values,
		                y=dat.data_vars[to_fit].values,
		                e=dat.data_vars[to_fit+'_error'].values))
        if 'R' in dat:
            final['R'] = dat['R'].values
        
        final['micron'] = (dat.coords['wavelength'].values)
        final['wavenumber'] = 1e4/final['micron']

	    #always ensure we are ordered correctly
        final = final.sort_values(by='wavenumber').reset_index(drop=True)

	    #return a nice dictionary with the info we need 
        returns[key] = [final['wavenumber'].values, 
			             final['y'].values, final['e'].values]
        if 'R' in final:
            returns['_R'][key] = final['R'].values
        
    return returns

def prior_finder(d):
    sections = {}

    def recurse(path, current):
        if not isinstance(current, Mapping):
            return
        if "prior" in current:
            sections[".".join(path)] = current
        for k, v in current.items():  # preserves order
            if isinstance(v, Mapping):
                recurse(path + (k,), v)

    recurse((), d)
    return sections

def hypercube(u, fitpars,param_tools=None):
    x=np.empty(len(u))
    for i,key in enumerate(fitpars.keys()):
        if fitpars[key]['prior'] == 'uniform':
            minn=fitpars[key]['uniform_kwargs']['min']
            maxx=fitpars[key]['uniform_kwargs']['max']
            x[i] = minn+(maxx-minn)*u[i]
        elif fitpars[key]['prior'] == 'gaussian':
            mean=fitpars[key]['gaussian_kwargs']['mean']
            std=fitpars[key]['gaussian_kwargs']['std']
            x[i]=stats.norm.ppf(u[i], loc=mean, scale=std)
        elif fitpars[key]['prior'] == 'xarray_grid':
            if param_tools is None:
                raise ValueError("param_tools must be provided to hypercube to use xarray_grid prior")
            param_name = key.split('.')[-1]
            grid_name = param_tools.grid_name
            grid_parameters_unique = param_tools.interp_params[grid_name]['grid_parameters_unique']
            if param_name not in grid_parameters_unique:
                raise ValueError(f"Parameter '{param_name}' not found in grid '{grid_name}' unique parameters.")
            values = grid_parameters_unique[param_name]
            minn = np.min(values)
            maxx = np.max(values)
            x[i] = minn + (maxx - minn) * u[i]
        else:
            raise Exception('Prior type not available')
        if fitpars[key]['log'] == True or fitpars[key]['log'] == 'True':
            x[i]=10**x[i]  
    return x


def _wvl_offset_to_micron(wvl_offset, obs_key, config):
    """Convert a per-observation wavelength offset to micron.

    By default retrieval.wvl_offset.<obs_key> is interpreted in micron. A TOML
    entry can include unit = "AA", "nm", "um", etc. to sample in another unit.
    """
    offset_cfg = config.get('retrieval', {}).get('wvl_offset', {}).get(obs_key, {})
    unit = offset_cfg.get('unit', 'um') if isinstance(offset_cfg, Mapping) else 'um'
    return wvl_offset * u.Unit(unit).to(u.um)


def apply_wavelength_offset_to_model(y_model, xdata, wvl_offset_micron):
    """Shift model features in wavelength and return values on the original data grid.

    Parameters
    ----------
    y_model : array_like
        Model flux already sampled on the observation grid.
    xdata : array_like
        Observation grid in wavenumber (cm^-1), matching DATA_DICT.
    wvl_offset_micron : float
        Additive model wavelength offset in micron. Positive values move model
        features to longer wavelengths. The returned array remains on xdata.
    """
    if wvl_offset_micron == 0 or wvl_offset_micron is None:
        return np.asarray(y_model, dtype=float).copy()

    y_model = np.asarray(y_model, dtype=float)
    wave_um = 1e4 / np.asarray(xdata, dtype=float)
    order = np.argsort(wave_um)
    shifted_wave = wave_um[order] + wvl_offset_micron
    shifted_model = y_model[order]

    shifted = np.interp(
        wave_um[order],
        shifted_wave,
        shifted_model,
        left=np.nan,
        right=np.nan,
    )
    out = np.empty_like(y_model, dtype=float)
    out[order] = shifted
    return out

def _velocity_config(config, key):
    value = config.get('object', {}).get(key, None)
    if value is None:
        value = config.get(key, None)
    if value is None:
        return None
    if isinstance(value, Mapping):
        if 'value' in value:
            if value['value']  == 0:
                #avoids code slow downs if config is set but is set to zero 
                return None
            unit = value.get('unit', 'km/s')
            kwargs = {k: v for k, v in value.items() if k not in ('value', 'unit')}
            kwargs['v_array'] = value['value'] * u.Unit(unit).to(u.km / u.s)
            return kwargs
        return dict(value)
    return {'v_array': value}

def process_model(resultx, resulty, data_dict=None, conv_dict=None, config=None, regrid_R=None):
    """
    Processes model output by applying distance scaling, Doppler shift (RV), 
    rotational broadening (vrot), and rebinning to data axes.

    Parameters
    ----------
    resultx : array
        PICASO output wavenumber.
    resulty : array
        PICASO model output.
    data_dict : dict, optional
        Observational data dictionary {key: [wavenumber, y, error]}.
    config : dict, optional
        Configuration dictionary to extract RV, vrot, distance scaling, and convolution.
    CONV_DICT : dict, optional
        Instrument resolution convolution dictionary.
    regrid_R : float, optional
        Fallback resolution for mean_regrid if data_dict is not provided.

    Returns
    -------
    dict
        Dictionary with keys from data_dict (or 'model' if data_dict is None) 
        containing lists of [x, y, mask] of processed/binned model.
    """
    config = config or {}
    conv_dict = conv_dict or {}
    
    distance_scaling = 1.0
    if config.get('observation_type') == 'thermal' and 'object' in config:
        R_dict = config['object'].get('radius')
        d_dict = config['object'].get('distance')
        if R_dict and d_dict:
            R = R_dict['value'] * u.Unit(R_dict['unit']).to(u.m)
            d = d_dict['value'] * u.Unit(d_dict['unit']).to(u.m)
            distance_scaling = (R / d) ** 2

    resulty = distance_scaling * resulty

    RV_config = _velocity_config(config, 'RV')
    if RV_config:
        resulty = RV(resultx, resulty, **RV_config)

    vrot_config = _velocity_config(config, 'vrot')
    if vrot_config:
        resulty = vrot(resultx, resulty, **vrot_config)

    convolve_config = config.get('convolve')

    returns = {}
    if data_dict is not None and len(data_dict) > 0:
        for obs_key in data_dict.keys():
            xdata, _, _ = data_dict[obs_key]
            
            jwst_instrument_conv_from_file = conv_dict.get(obs_key) if conv_dict else None
            
            if convolve_config:
                conv_cfg = convolve_config.get(obs_key, convolve_config) if isinstance(convolve_config, dict) else {'R': convolve_config}
                R_conv = conv_cfg.get('R', conv_cfg.get('resolution')) if isinstance(conv_cfg, dict) else conv_cfg
                if R_conv is None:
                    raise ValueError("convolve_config must define 'R' or 'resolution'")
                if np.isscalar(R_conv):
                    R_conv = np.full_like(xdata, R_conv, dtype=float)
                rebinned = conv_non_uniform_R(resulty, 1e4/resultx, np.asarray(R_conv), 1e4/xdata)
            elif jwst_instrument_conv_from_file:
                # Note: this was noted as not fully tested in MODEL
                if len(jwst_instrument_conv_from_file) == 3:
                    wno_inst, R_conv, convolve_observation = jwst_instrument_conv_from_file
                else:
                    wno_inst, R_conv = jwst_instrument_conv_from_file
                    convolve_observation = True
                if len(wno_inst) == len(xdata) and np.allclose(wno_inst, xdata, rtol=1e-8, atol=0.0):
                    if convolve_observation:
                        rebinned = convolve_top_hat_rebin_R(1e4/resultx, resulty, 1e4/xdata, np.asarray(R_conv))
                    else:
                        rebinned = top_hat_rebin_R(1e4/resultx, resulty, 1e4/xdata, np.asarray(R_conv))
                else:
                    # raise Exception('This is not fully tested... Need to enable this before proceeding')
                    if convolve_observation:
                        rebinned_to_inst = conv_non_uniform_R(resulty, 1e4/resultx, np.asarray(R_conv), 1e4/wno_inst)
                    else:
                        rebinned_to_inst = top_hat_rebin_R(1e4/resultx, resulty, 1e4/wno_inst, np.asarray(R_conv))
                    rebinned = spectres.spectres(xdata,wno_inst, rebinned_to_inst)
            else:
                rebinned = spectres.spectres(xdata,resultx, resulty)
            
            mask = np.isnan(rebinned)
            if np.any(mask):
                warnings.warn(f"Created {np.sum(mask)} NaNs out of {len(rebinned)} model points for {obs_key}", UserWarning)

            returns[obs_key] = [xdata, rebinned, mask]
    else:
        #only used if data_dict is not supplied 
        #otherwise it always returns the model on the data grid. 
        if regrid_R is not None:
            x_regrid, y_regrid = mean_regrid(resultx, resulty, R=regrid_R)
            mask = np.isnan(y_regrid)
            if np.any(mask):
                warnings.warn(f"Created {np.sum(mask)} NaNs out of {len(y_regrid)} model points for model", UserWarning)
            returns['model'] = [x_regrid, y_regrid, mask]
        else:
            mask = np.isnan(resulty)
            if np.any(mask):
                warnings.warn(f"Created {np.sum(mask)} NaNs out of {len(resulty)} model points for model", UserWarning)
            returns['model'] = [resultx, resulty, mask]

    return returns

def MODEL(cube, fitpars, config, OPA, param_tools, DATA_DICT, retrieval=True,CONV_DICT={}):
    """
    Generate model spectra for parameter sets.

    Parameters
    ----------
    cube : array-like
        Parameter values. Shape can be:
          - (N_params,)  -> single parameter set
          - (N_samples, N_params) -> multiple sets
    fitpars : dict
        Dictionary of fit parameters.
    config : dict
        Configuration dictionary.
    OPA : object
        Opacity connection.
    param_tools : object
        Parameterization helper.
    DATA_DICT : dict
        Observational data dictionary.
    CONV_DICT : dict 
        Instrument Resolution convolution dictionary from parse data 

    Returns
    -------
    y_model : dict
        Dictionary with the same keys as observation data. 
        Each value is an array with shape:
          - (N_samples, len(xdata))
    y_mask : dict
        Dictionary with same keys as y_model. Boolean masks (True where model is NaN).
    """
    cube = np.atleast_2d(cube)  # ensure shape (N_samples, N_params)
    n_samples = cube.shape[0]

    # initialize storage based on data dict keys (basename of filenames)
    y_model = {key: [] for key in DATA_DICT.keys()}
    y_mask = {key: [] for key in DATA_DICT.keys()}

    if not retrieval:
        profiles={}                                   

    for j,row in enumerate(cube):
        # update parameters
        # change to use function with checker that logs everything being changed 
        # this will raise except if the changed parameters does not match len fitpars
        # this only ignores scaling, offset, err_inf, and wvl_offset which are handled in loglikelihood function
        config=update_config_w_cube(config, fitpars,row)

        # compute spectrum
        calculation = OBSERVATION_CALC_MAP.get(config['observation_type'],None)
        if calculation == None: 
            raise Exception (rf'Not a recognized observation_type. Options are: thermal fpfs_thermal albedo fpfs_reflected transit_depth reflected transmission. Input was {calculation}')
        
        picaso_class = setup_spectrum_class(config, opacity=OPA, param_tools=param_tools)
        out = picaso_class.spectrum(OPA, full_output=True, 
                                    calculation=calculation)

        resultx = out['wavenumber']
        result_key = config['observation_type']
        resulty = out[result_key]
        
        #add RV, vsini, and resolution convolution, and binning
        processed_outputs = process_model(resultx, resulty, 
                                          data_dict=DATA_DICT, 
                                          config=config, 
                                          conv_dict=CONV_DICT)

        # rebin to observed wavelengths
        for obs_key in DATA_DICT.keys():
            y_model[obs_key].append(processed_outputs[obs_key][1])
            y_mask[obs_key].append(processed_outputs[obs_key][2])

        if not retrieval:
            profiles[j]=picaso_class.inputs['atmosphere']['profile']

    # stack results into arrays
    for obs_key in y_model:
        y_model[obs_key] = np.vstack(y_model[obs_key])
        y_mask[obs_key] = np.vstack(y_mask[obs_key])
        # if single sample, flatten to 1D
        # if n_samples == 1:
        #     y_model[obs_key] = y_model[obs_key][0]

    if retrieval:
        return y_model, y_mask
    else:
        return y_model, y_mask, profiles

def log_likelihood(cube, fitpars, config, OPA, DATA_DICT, param_tools,CONV_DICT={},retrieval=True):
    """
    Vectorized log-likelihood.

    Parameters
    ----------
    cube : array-like
        Parameter values.
        Shape can be:
          - (N_params,)  -> single parameter set
          - (N_samples, N_params) -> multiple parameter sets
    fitpars, config, OPA, DATA_DICT, param_tools : as before

    Returns
    -------
    logl : float or ndarray
        Log-likelihood(s).
        Shape (N_samples,) if multiple sets, or scalar if single.
    """
    cube = np.atleast_2d(cube)  # (N_samples, N_params)
    n_samples = cube.shape[0]

    # Compute model spectra for all samples
    mod_out = MODEL(cube, fitpars, config, OPA, param_tools, DATA_DICT,CONV_DICT=CONV_DICT,retrieval=retrieval)
    if retrieval:
        y_model_dict, y_mask_dict = mod_out
    else: 
        y_model_dict, y_mask_dict,profiles = mod_out

    offset_dict = config.get('retrieval',{}).get('offset',{})
    err_inf_dict = config.get('retrieval',{}).get('err_inf',{})
    scaling_dict = config.get('retrieval',{}).get('scaling',{})
    wvl_offset_dict = config.get('retrieval',{}).get('wvl_offset',{})

    logls = []

    if not retrieval: 
        #prep storage of models and data for each sample 
        ydat_all_2d = []
        xdat_all_2d = []
        ymod_all_2d = []
        sigma_all_2d = []
        chi_sqs = []

    for j in range(n_samples):
        ydat_all = []
        xdat_all=[]
        ymod_all = []
        sigma_all = []
        extra_term_all = []
        mask_all = []

        added_extras = 0 
        for key in DATA_DICT.keys():
            xdata, ydata, edata = DATA_DICT[key]
            y_model = y_model_dict[key][j].copy()   # pick j-th sample
            y_mask = y_mask_dict[key][j]
            ydata_ = ydata
            calc_type = OBSERVATION_CALC_MAP.get(config['observation_type'])

            # add offsets
            if key in offset_dict:
                icube = list(fitpars.keys()).index(f'offset.{key}')
                y_model += cube[j, icube];added_extras+=1

            # add scalings
            if key in scaling_dict:
                icube = list(fitpars.keys()).index(f'scaling.{key}')
                y_model *= cube[j, icube];added_extras+=1

            # add wavelength offsets. This shifts the model wavelength coordinate,
            # then interpolates back to the unchanged observation grid.
            if key in wvl_offset_dict:
                icube = list(fitpars.keys()).index(f'wvl_offset.{key}')
                wvl_offset = _wvl_offset_to_micron(cube[j, icube], key, config)
                y_model = apply_wavelength_offset_to_model(y_model, xdata, wvl_offset)
                y_mask = y_mask | np.isnan(y_model)
                added_extras += 1

            # add error inflation if exists
            if key in err_inf_dict:
                icube = list(fitpars.keys()).index(f'err_inf.{key}')
                err_inf = _err_inf_to_model_variance(cube[j, icube], xdata, key, config);added_extras+=1
            else:
                err_inf = 0

            sigma = edata**2 + err_inf

            extra_term = np.log(2 * np.pi * sigma)

            ydat_all.append(ydata_)
            ymod_all.append(y_model)
            sigma_all.append(sigma)
            extra_term_all.append(extra_term)
            mask_all.append(y_mask)
            if not retrieval: 
                xdat_all.append(xdata)

        # check added vals after going through all data instnaces 
        should_add_extras = len(offset_dict)+len(scaling_dict)+len(err_inf_dict)+len(wvl_offset_dict)
        assert should_add_extras == added_extras, 'The number of added offsets, scalings, wavelength offsets or error inflations does not match occurances in config file'


        ydat_all = np.concatenate(ydat_all)
        ymod_all = np.concatenate(ymod_all)
        sigma_all = np.concatenate(sigma_all)
        extra_term_all = np.concatenate(extra_term_all)
        mask_all = np.concatenate(mask_all)
        if not retrieval: xdat_all= np.concatenate(xdat_all)

        # mask all nans before computing the logl
        ydat_all = ydat_all[~mask_all]
        ymod_all = ymod_all[~mask_all]
        sigma_all = sigma_all[~mask_all]
        extra_term_all = extra_term_all[~mask_all]
        if not retrieval: xdat_all= xdat_all[~mask_all]

        logl = -0.5 * np.sum((ydat_all - ymod_all) ** 2 / sigma_all + extra_term_all)
        logls.append(logl)

        if not retrieval: 
            simpl_chi_sq = np.sum((ydat_all-ymod_all)**2/sigma_all)/len(sigma_all)
            #prep storage of models and data for each sample 
            ydat_all_2d += [ydat_all]
            ymod_all_2d += [ymod_all]
            sigma_all_2d += [np.sqrt(sigma_all)]
            chi_sqs += [simpl_chi_sq]

    logls = np.array(logls)

    # return scalar if only one input sample
    if retrieval:
        return logls[0] if n_samples == 1 else logls
    else: 
        return {'logl':logls, 
                'xdata':xdat_all,
                'ydata':ydat_all_2d, 
                'ymodel':ymod_all_2d, 
                'edata':sigma_all_2d,
                'chi_sq_per_pt':chi_sqs,
                'model_dict':y_model_dict,
                'mask_dict':y_mask_dict,
                'profiles':profiles}


def vrot(wvl, spectrum_array, v_array=0.0, eps=0.6, nr=10, ntheta=100, dif=0.0):
    """
    Apply rotational broadening to single or multiple spectra.

    Parameters
    ----------
    wvl : array_like, shape (n_wvl,)
        Wavelength grid.
    spectrum_array : array_like, shape (n_wvl,) or (n_spectra, n_wvl)
        Input spectra to be rotationally broadened.
    v_array : float or array_like, shape (n_spectra,), optional
        Projected rotational velocities (km/s). Default is 0.0.
    eps : float, optional
        Limb darkening coefficient (default: 0.6).
    nr : int, optional
        Number of radial bins (default: 10).
    ntheta : int, optional
        Azimuthal bins in outer annulus (default: 100).
    dif : float, optional
        Differential rotation coefficient (default: 0.0).

    Returns
    -------
    broadened_array : array_like, shape (n_wvl,) or (n_spectra, n_wvl)
        Rotationally broadened spectra.
    """
    is_1d = (np.ndim(spectrum_array) == 1)
    spectrum_array = np.atleast_2d(spectrum_array)
    v_array = np.atleast_1d(v_array)

    n_spectra = spectrum_array.shape[0]
    if len(v_array) == 1 and n_spectra > 1:
        v_array = np.full(n_spectra, v_array[0])

    broadened_array = np.zeros_like(spectrum_array)

    for i, (v, spectrum) in enumerate(zip(v_array, spectrum_array)):
        if v == 0:
            broadened_array[i] = spectrum
            continue
        ns = np.zeros_like(spectrum)
        tarea = 0.0
        dr = 1.0 / nr

        for j in range(nr):
            r = dr / 2.0 + j * dr
            nphi = int(ntheta * r)
            if nphi == 0: nphi = 1
            area = ((r + dr/2.0)**2 - (r - dr/2.0)**2) / nphi * (1.0 - eps + eps * np.cos(np.arcsin(r)))

            for k in range(nphi):
                th = np.pi / nphi + k * 2.0 * np.pi / nphi

                if dif != 0:
                    vl = v * r * np.sin(th) * (1.0 - dif/2.0 - dif/2.0 * np.cos(2.0 * np.arccos(r * np.cos(th))))
                else:
                    vl = v * r * np.sin(th)

                shifted = wvl + wvl * vl / 2.9979e5  # Doppler shift
                ns += area * np.interp(shifted, wvl, spectrum)
                tarea += area

        broadened_array[i] = ns / tarea

    return broadened_array[0] if is_1d else broadened_array

def RV(wvl, flux_array, v_array=0.0, edgeHandling='firstlast', fillValue=None):
    """
    Doppler shift single or multiple spectra given an array of velocities.

    Parameters
    ----------
    wvl : array_like, shape (n_wvl,)
        Shared input wavelength grid.
    flux_array : array_like, shape (n_wvl,) or (n_spectra, n_wvl)
        Flux values for each spectrum.
    v_array : float or array_like, shape (n_spectra,), optional
        Doppler velocities in km/s. Default is 0.0.
    edgeHandling : str, optional
        'firstlast' (default) or 'fillValue'.
    fillValue : float, optional
        Value to use if edgeHandling='fillValue'.

    Returns
    -------
    nflux_array : array_like, shape (n_wvl,) or (n_spectra, n_wvl)
        Doppler-shifted fluxes, resampled onto the original wavelength grid.
    """
    cvel = 299_792.458  # speed of light in km/s
    is_1d = (np.ndim(flux_array) == 1)
    flux_array = np.atleast_2d(flux_array)
    v_array = np.atleast_1d(v_array)

    n_spectra, n_wvl = flux_array.shape
    if len(v_array) == 1 and n_spectra > 1:
        v_array = np.full(n_spectra, v_array[0])

    nflux_array = np.empty_like(flux_array)

    for i in range(n_spectra):
        v = v_array[i]
        flux = flux_array[i]

        if v == 0:
            nflux_array[i] = flux
            continue

        # Shifted wavelength
        wlprime = wvl * (1.0 + v / cvel)

        # Set interpolation fill value
        fv = np.nan if edgeHandling != "fillValue" else fillValue

        # Interpolate shifted flux back to original wavelength grid
        nflux = interp1d(wlprime, flux, bounds_error=False, fill_value=fv)(wvl)

        if edgeHandling == "firstlast":
            nin = ~np.isnan(nflux)
            if not nin.any():
                nflux = np.full_like(nflux, fv)
            else:
                if not nin[0]:
                    fvindex = np.argmax(nin)
                    nflux[:fvindex] = nflux[fvindex]
                if not nin[-1]:
                    lvindex = len(nin) - 1 - np.argmax(nin[::-1])
                    nflux[lvindex + 1:] = nflux[lvindex]

        nflux_array[i] = nflux

    return nflux_array[0] if is_1d else nflux_array

def convolver(newx, x, y):
    #
    return y

def top_hat_rebin_R(model_wl, model_flux, obs_wl, R):
    """
    Bin-integrate a model spectrum onto observed bins using wavelength-dependent R.

    Each observed point is treated as a top-hat bin centered on obs_wl[i] with
    width dlambda = obs_wl[i] / R[i]. The returned value is the mean flux over
    that bin. Output order matches obs_wl.
    """
    model_wl = np.asarray(model_wl, dtype=float)
    model_flux = np.asarray(model_flux, dtype=float)
    obs_wl = np.asarray(obs_wl, dtype=float)
    R = np.asarray(R, dtype=float)

    if R.ndim == 0:
        R = np.full_like(obs_wl, float(R), dtype=float)
    if R.shape != obs_wl.shape:
        raise ValueError("R must be scalar or have the same shape as obs_wl")

    order = np.argsort(model_wl)
    model_wl = model_wl[order]
    model_flux = model_flux[order]

    trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    rebinned_flux = np.empty_like(obs_wl, dtype=float)

    for i, wl_center in enumerate(obs_wl):
        width = wl_center / R[i]
        if not np.isfinite(width) or width <= 0:
            raise ValueError("R values must be finite and positive")

        lo = wl_center - 0.5 * width
        hi = wl_center + 0.5 * width
        inside = (model_wl > lo) & (model_wl < hi)
        wl_bin = np.concatenate(([lo], model_wl[inside], [hi]))
        flux_bin = np.interp(wl_bin, model_wl, model_flux)
        rebinned_flux[i] = trapz(flux_bin, wl_bin) / (hi - lo)

    return rebinned_flux

def _cell_widths(grid):
    if len(grid) < 2:
        raise ValueError("model_wl must contain at least two points")

    edges = np.empty(len(grid) + 1, dtype=float)
    edges[1:-1] = 0.5 * (grid[:-1] + grid[1:])
    edges[0] = grid[0] - 0.5 * (grid[1] - grid[0])
    edges[-1] = grid[-1] + 0.5 * (grid[-1] - grid[-2])
    return np.diff(edges)

def _bin_edges_from_centers(centers):
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1:
        raise ValueError("Bin centers must be a 1D array")
    if len(centers) < 2:
        raise ValueError("At least two observed wavelengths are required to define bin edges")

    edges = np.empty(len(centers) + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges

def convolve_top_hat_rebin_R(model_wl, model_flux, obs_wl, R, nsigma=5):
    """
    Convolve with a wavelength-dependent Gaussian LSF and bin-average onto data.

    For each observed point, R defines the Gaussian FWHM as dlambda=lambda/R.
    The top-hat bin edges are inferred from the observed wavelength centers,
    so the binning step follows the actual data sampling instead of applying a
    second resolution-element-width smoothing.
    """
    model_wl = np.asarray(model_wl, dtype=float)
    model_flux = np.asarray(model_flux, dtype=float)
    obs_wl = np.asarray(obs_wl, dtype=float)
    R = np.asarray(R, dtype=float)

    if R.ndim == 0:
        R = np.full_like(obs_wl, float(R), dtype=float)
    if R.shape != obs_wl.shape:
        raise ValueError("R must be scalar or have the same shape as obs_wl")

    model_order = np.argsort(model_wl)
    model_wl = model_wl[model_order]
    model_flux = model_flux[model_order]
    model_dw = _cell_widths(model_wl)

    obs_order = np.argsort(obs_wl)
    obs_wl_sorted = obs_wl[obs_order]
    R_sorted = R[obs_order]
    obs_edges = _bin_edges_from_centers(obs_wl_sorted)

    rebinned_sorted = np.empty_like(obs_wl_sorted, dtype=float)
    sqrt2 = np.sqrt(2.0)

    for i, wl_center in enumerate(obs_wl_sorted):
        fwhm = wl_center / R_sorted[i]
        if not np.isfinite(fwhm) or fwhm <= 0:
            raise ValueError("R values must be finite and positive")

        sigma = fwhm / 2.355
        lo = obs_edges[i]
        hi = obs_edges[i + 1]
        support = (model_wl >= lo - nsigma * sigma) & (model_wl <= hi + nsigma * sigma)

        if not np.any(support):
            rebinned_sorted[i] = np.interp(wl_center, model_wl, model_flux)
            continue

        wl_local = model_wl[support]
        response = 0.5 * (
            erf((hi - wl_local) / (sqrt2 * sigma))
            - erf((lo - wl_local) / (sqrt2 * sigma))
        )
        weights = response * model_dw[support]
        norm = np.sum(weights)

        if not np.isfinite(norm) or norm <= 0:
            rebinned_sorted[i] = np.interp(wl_center, model_wl, model_flux)
        else:
            rebinned_sorted[i] = np.sum(model_flux[support] * weights) / norm

    rebinned_flux = np.empty_like(obs_wl, dtype=float)
    rebinned_flux[obs_order] = rebinned_sorted
    return rebinned_flux

def conv_non_uniform_R(model_flux, model_wl, R, obs_wl):
    """
    From brewster
    Convolve a model spectrum with a wavelength-dependent resolving power 
    onto the observed wavelength grid ???

    Parameters:
    - model_flux: 1D array of model flux values.
    - model_wl: 1D array of model wl values.
    - obs_wl: 1D array of observed wl values.
    - R: 1D array of resolving power values (for the obs_wl grid.)

    Returns:
    - convolved_flux: 1D array of convolved flux values on the obs_wl grid.
    """
    # create the array for the convolved flux
    convolved_flux = np.zeros_like(obs_wl)

    for i, wl_center in enumerate(obs_wl): 
        
        # compute FWHM and sigma for each wl
        # print('wl_center', wl_center)
        # print('R[i]', R[i])
        
        fwhm = wl_center / R[i]
        # print('fwhm', fwhm)
        sigma = fwhm / 2.355


        # compute the Gaussian kernel for the current wl
       
        gaussian_kernel = np.exp(-((model_wl-wl_center) ** 2) / (2 * sigma **2))
        #print('gaussian_kernel before normalisation', gaussian_kernel)

        # normalisation
        gaussian_kernel /= np.sum(gaussian_kernel)
        # print('gaussian_kernel after normalisation', gaussian_kernel)



        # apply the kernel to the flux
        convolved_flux[i] = np.sum(model_flux * gaussian_kernel)
    
    return convolved_flux

def _resume_check_config(config):
    check = copy.deepcopy(config)
    check.get('retrieval', {}).get('sampler', {}).pop('resume', None)
    return check

SAMPLER_REGISTRY = {}

def register_sampler(name):
    def decorator(func):
        SAMPLER_REGISTRY[name.lower()] = func
        return func
    return decorator

@register_sampler('dynesty')
def run_dynesty(config, loglike_fn, hypercube_fn, ndims, pool, prior_config):
    checkpoint_file = config['InputOutput']['retrieval_output']+'/dynesty.save'
    sampler_args = prior_config['sampler']['sampler_kwargs'].copy()
    
    #fallback for SerialPool or single-process pools which don't use pool.size
    pool_size = getattr(pool, 'size', 1) if pool is not None else 1
    sampler_args.setdefault('queue_size', pool_size)
    
    run_args = prior_config['sampler']['run_kwargs']
    prior_config['sampler']['resume'] = prior_config['sampler'].get('resume', False)
    
    if prior_config['sampler']['resume']:
        print('Resuming retrieval...')
        sampler = dynesty.NestedSampler.restore(checkpoint_file, pool=pool)
        resume = True
    else:
        sampler = dynesty.NestedSampler(loglike_fn, hypercube_fn, ndims, pool=pool, **sampler_args) 
        resume = False

    sampler.run_nested(checkpoint_file=checkpoint_file,
                       resume=resume,
                       **run_args)
    return sampler

@register_sampler('ultranest')
def run_ultranest(config, loglike_fn, hypercube_fn, ndims, pool, prior_config):
    """
    TODO: ultranest only runs through MPI and so this function does not use pool 
    instead ultranest should be run via mpiexec -np num_procs python script.py 
    which supports either local and cluster machines and will force mpi=True
    """
    import ultranest
    fitpars = prior_finder(prior_config)
    param_names = list(fitpars.keys())
    
    log_dir = config['InputOutput']['retrieval_output']
    
    sampler_args = prior_config['sampler'].get('sampler_kwargs', {}).copy()
    run_args = prior_config['sampler'].get('run_kwargs', {}).copy()
    
    resume_val = prior_config['sampler'].get('resume', False)
    resume = 'resume' if resume_val else 'overwrite'
    
    sampler = ultranest.ReactiveNestedSampler(
        param_names,
        loglike_fn,
        transform=hypercube_fn,
        log_dir=log_dir,
        resume=resume,
        **sampler_args
    )
    
    sampler.run(**run_args)
    return sampler

def check_resume_exists(code, retrieval_output):
    if code.lower() == 'dynesty':
        checkpoint_file = os.path.join(retrieval_output, 'dynesty.save')
        return os.path.exists(checkpoint_file)
    elif code.lower() == 'ultranest':
        checkpoint_file = os.path.join(retrieval_output, 'results', 'points.hdf5')
        return os.path.exists(checkpoint_file)
    else:
        return False


def retrieve(config=None, param_tools=None, driver_file=None):
    """
    This spawns a retrieval run based on config file. 
    Required is config or driver_file. 

    param_tools can be created from the config or driver_file or can be directly supplied for 
    custom loading. 

    Parameters 
    ----------
    config : dict 
        configuration of run 
    param_tools : class
        parameterize class 
    driver_file : str 
        valide driver file (e.g., driver.toml)
    """
    #if a driver file is specified read it in
    if driver_file: 
        with open(driver_file, "rb") as f:
            config = tomllib.load(f)  
    if not config: 
        raise Exception('config and driver_file are None. Please specify one or the other')
    
    if not param_tools: 
        preload_cloud_miefs = find_values_for_key(config ,'condensate')
        virga_mieff   = config['OpticalProperties'].get('virga_mieff',None)
        model_dir = config['temperature'].get('xarray_grid',{}).get('model_dir',None)
        param_tools = Parameterize(load_cld_optical=preload_cloud_miefs,
                                    mieff_dir=virga_mieff,
                                    model_dir=model_dir)        

    OPA = opannection(
        filename_db=config['OpticalProperties']['opacity_file'], #database(s)
        method=config['OpticalProperties']['opacity_method'], #resampled, preweighted, resortrebin
        **config['OpticalProperties']['opacity_kwargs'] #additonal inputs 
        )
    
    prior_config=config['retrieval']

    #if it isn't specified just assume its being run on a local machine
    mpi = config['retrieval'].get('mpi',False)
    
    #if it isn't specified just grab half the users cpu 
    #note processes is not used for mpi = True. 
    #mpi uses -np instead e.g., mpirun -np 200 python your_script.py
    #but processes would be effectively equivalent to nodes*cpu
    processes = config['retrieval'].get('processes',int(os.cpu_count()/2))


    code = prior_config.get('sampler', {}).get('code', 'dynesty')
    retrieval_output = config['InputOutput']['retrieval_output']
    input_file = os.path.join(retrieval_output, 'inputs.toml')
    
    if prior_config['sampler'].get('resume', False):
        if not check_resume_exists(code, retrieval_output):
            raise FileNotFoundError(f"Cannot resume retrieval; checkpoint/resume file does not exist for {code} in: {retrieval_output}")
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Cannot verify resume config; original inputs file does not exist: {input_file}")
        with open(input_file, "rb") as f:
            saved_config = tomllib.load(f)
        if _resume_check_config(config) != _resume_check_config(saved_config):
            raise ValueError(f"Current config does not match original retrieval config in {input_file}")
    
    fitpars=prior_finder(prior_config)
    ndims=len(fitpars)
    
    # def convolver(newx,x,y):
    #     return newy
      
    DATA_DICT,CONV_DICT = get_data(config)
    hypercube_fn = partial(hypercube, fitpars=fitpars)
    loglike_fn = partial(log_likelihood, fitpars=fitpars, config=config,
                  OPA=OPA, param_tools=param_tools, DATA_DICT=DATA_DICT,CONV_DICT=CONV_DICT)
    

    

    @contextlib.contextmanager
    def maybe_pool(mpi, processes):
        if not mpi and processes == 1:
            yield None
        else:
            with choose_pool(mpi=mpi, processes=processes) as pool:
                yield pool

    # Dynamic pool allocation using maybe_pool
    with maybe_pool(mpi, processes) as pool:
        
        # If MPI is selected, worker nodes must wait here
        if mpi and pool is not None and hasattr(pool, 'is_master') and not pool.is_master():
            pool.wait()
            sys.exit(0)
            
        print('Running retrieval...')

        sampler_func = SAMPLER_REGISTRY.get(code.lower())
        if sampler_func is None:
            raise ValueError(f"Sampler '{code}' is not supported. Supported samplers are: {list(SAMPLER_REGISTRY.keys())}")
        
        sampler = sampler_func(config, loglike_fn, hypercube_fn, ndims, pool, prior_config)
        return sampler

def check_model_samples(config, N=100, samples=None,full_likelihood=False):
    """
    Tests the prior distribution by generating models based on the provided configuration.

    Args:
        config (dict): Configuration dictionary containing the necessary parameters for 
            optical properties, retrieval, and other settings.
            - 'OpticalProperties': A dictionary with keys:
                - 'opacity_file' (str): Path to the opacity database(s).
                - 'opacity_method' (str): Method for handling opacity ('resampled', 'preweighted', etc.).
                - 'opacity_kwargs' (dict): Additional arguments for opacity handling.
                - 'virga_mieff' (str, optional): Directory for virga Mie efficiency files.
            - 'retrieval': A dictionary containing retrieval parameters.
        N (int, optional): Number of samples to generate if no sampler is provided. Defaults to 100.
        sampler (object, optional): A sampler object with a `results.samples_equal()` method 
            to provide sample points. If None, random samples are generated. Defaults to None.

    Returns:
        numpy.ndarray: An array of generated models based on the prior distribution.

    Notes:
        - The function initializes optical properties and parameterization tools using the 
          provided configuration.
        - If a sampler is provided, it uses the sampler's results to generate thetas; otherwise, 
          it generates random samples.
        - The function constructs models for each set of parameters (thetas) and returns them 
          as a numpy array.
    """
    OPA = opannection(
        filename_db=config['OpticalProperties']['opacity_file'], #database(s)
        method=config['OpticalProperties']['opacity_method'], #resampled, preweighted, resortrebin
        **config['OpticalProperties']['opacity_kwargs'] #additonal inputs 
        )
    preload_cloud_miefs = find_values_for_key(config ,'condensate')
    virga_mieff   = config['OpticalProperties'].get('virga_mieff',None)
    model_dir = config['temperature'].get('xarray_grid',{}).get('model_dir',None)
    param_tools = Parameterize(load_cld_optical=preload_cloud_miefs,
                               mieff_dir=virga_mieff,
                               model_dir=model_dir)
    
    fitpars=prior_finder(config['retrieval'])
    ndims=len(fitpars)
    if samples is not None:
        #this is specific to dynesty
        thetas = samples
    else:
        cube = np.random.random([N, ndims])
        thetas = [hypercube(cube[i], fitpars, param_tools=param_tools) for i in range(N)]
    
    thetas = np.array(thetas)

    DATA_DICT,CONV_DICT = get_data(config)
    if full_likelihood: 
        output_dict = log_likelihood(thetas[:N], fitpars, config, OPA, DATA_DICT, param_tools,CONV_DICT=CONV_DICT,retrieval=False)
        return output_dict 
    else: 
        models, masks, profiles = MODEL(thetas[:N], fitpars, config, OPA, param_tools, DATA_DICT, retrieval=False,CONV_DICT=CONV_DICT)

        return models, thetas, profiles

def setup_spectrum_class(config, opacity, param_tools, stage=None):

    if isinstance(opacity,type(None)):
        opacity_kwargs = config['OpticalProperties'].get('opacity_kwargs',{})
        opacity = opannection(
        filename_db=config['OpticalProperties']['opacity_file'], #database(s)
        method=config['OpticalProperties']['opacity_method'], #resampled, preweighted, resortrebin
        **opacity_kwargs #additonal inputs 
        ) #opanecction connects to the opacity database
    
    irradiated = config['irradiated']
    if not irradiated: 
        A = inputs(calculation='browndwarf',climate=False) #if it isn't irradiated we are calculating a browndwarf
    else: 
        A = inputs(calculation='planet',climate=False) #if irradiated we are calculating a planet 
    
    #WIP TODO A.approx()

    phase = config['geometry'].get('phase', {}).get('value',None)
    phase_unit = config['geometry'].get('phase', {}).get('unit',None)
    rad = (phase * u.Unit(phase_unit)).to(u.rad).value
    phase_kwargs = config['geometry'].get('phase_kwargs', {})
    A.phase_angle(rad,**phase_kwargs) #input the radian angle of the event/geometry of browndwarf/planet

    A.gravity(gravity     = config['object'].get('gravity', {}).get('value',None), 
            gravity_unit= u.Unit(config['object'].get('gravity', {}).get('unit',None)), 
            radius      = config['object'].get('radius', {}).get('value',None), 
            radius_unit = u.Unit(config['object'].get('radius', {}).get('unit',None)),
            mass        = config['object'].get('mass', {}).get('value',None), 
            mass_unit   = u.Unit(config['object'].get('mass', {}).get('unit',None)) 
            )

    if stage == 'object':
        return A
    #gravity parameters for a planet/browndwarf

    
    if irradiated: #calculating spectrum for a planet by defining star properties
        typestar = config['star'].get('type')
        
        #check if userfile is requested
        if typestar=='userfile':
            filename = config['star'].get('userfile',{}).get('filename',None)
            if os.path.exists(str(filename)): #file with wavelength and flux 
                w_unit=config['star']['userfile'].get('w_unit')
                f_unit=config['star']['userfile'].get('f_unit')
            else: 
                raise Exception('Stellar path provided does not exist ')
        else: #properties of star 
            w_unit=None
            f_unit=None
            filename=None
            temp= config['star'].get('grid',{}).get('teff',None) #temperature of star
            metal= config['star'].get('grid',{}).get('feh',None) #metallicity of star
            logg= config['star'].get('grid',{}).get('logg',None) #log gravity of star
            database= config['star'].get('grid',{}).get('database',None) #specify database

        A.star(opacity,
               temp=temp, 
               metal=metal, 
               logg=logg ,
               database=database,
               radius = config['star'].get('radius', {}).get('value',None), 
               radius_unit= u.Unit(config['star'].get('radius', {}).get('unit',None)),
               semi_major=config['star'].get('semi_major', {}).get('value',None), 
               semi_major_unit = u.Unit(config['star'].get('semi_major', {}).get('unit',None)), 
               filename=filename, 
               w_unit=w_unit, 
               f_unit=f_unit
               ) 
        if stage == 'star':
            return A
    #WIP TODO: A.surface_reflect()

    
    # tempreature 
    pt_config = config['temperature']
    df_pt = PT_handler(pt_config, A, param_tools) #datafile for pressure temperature profile
    A.atmosphere(df=df_pt) #will include chemistry if it was added to userfile
    param_tools.add_class(A)
    if stage == 'temperature':
        return A
    # chemistry
    chem_config = config.get('chemistry',{})
    chem_type = chem_config.get('method','')
    if chem_type == 'userfile':
        kwargs = config['chemistry'][chem_type].get('pd_kwargs',{})
        df_mixingratio = pd.read_csv(config['chemistry'][chem_type]['filename'],**kwargs) 
        #default remove prssure and temperature 
        df_cleaned = df_mixingratio.drop(columns=['temperature', 'pressure'])
        df_mixingratio = pd.merge(A.inputs['atmosphere']['profile'].loc[:,['temperature', 'pressure']], 
                                  df_cleaned, left_index=True, right_index=True, how='inner')
    elif chem_type!='': 
        chemistry_function = getattr(param_tools, f'chem_{chem_type}')
        df_mixingratio  = chemistry_function(**chem_config[chem_type].get('grid_kwargs',chem_config[chem_type]))#note, this includes P and T already
    
    #set final with chem
    A.atmosphere(df = df_mixingratio)
    if stage == 'chemistry':
        return A
    # clouds 
    cloud_config = config.get('clouds',None)
    if isinstance(cloud_config , dict):
        do_clouds=True
        cloud_names = [i.split('_type')[0] for i in cloud_config.keys() if 'type' in i]
    else: 
        do_clouds=False
        cloud_names = []
    
    all_dfs = []
    for icld in cloud_names: 
        cld_type = cloud_config[f'{icld}_type']
        if cld_type == 'userfile':
            kwargs = cloud_config[icld][cld_type].get('pd_kwargs',{})
            df_cld = pd.read_csv(cloud_config[icld][cld_type]['filename'],**kwargs) 
        else:
            cloud_function = getattr(param_tools, f'cloud_{cld_type}')
            df_cld = cloud_function(**cloud_config[icld][cld_type])
        
        all_dfs += [df_cld]

    if do_clouds:    
        df_cld = cloud_averaging(all_dfs) 
        A.clouds(df=df_cld)

    return A


def PT_handler(pt_config, picaso_class, param_tools): #WIP
    type = pt_config['profile']

    #check if supplied file for pt profile
    if type == 'userfile': 
        filename = pt_config['userfile']['filename']
        kwargs = pt_config['userfile'].get('pd_kwargs', {})
        pt_df = pd.read_csv(filename, **kwargs)

    elif type == 'sonora_bobcat':
        #sonora bobcat grid pt profile from picaso-data
        params = pt_config.get('sonora_bobcat', {})
        #call picaso's sonora function with parameters
        picaso_class.sonora(**params)
        #the resulting pt profile is stored inside a.inputs['atmosphere']['profile']
        pt_df = picaso_class.inputs['atmosphere']['profile']
    
    elif type == 'xarray_grid':
        if not hasattr(param_tools, 'grid_name'):
            raise Exception('Grid has not been pre-loaded')
        else: 
            grid_name = param_tools.grid_name
        pressure = param_tools.pressure[grid_name][0]
        picaso_class.add_pt(P=pressure)
        param_tools.add_class(picaso_class)
        temperature_function = getattr(param_tools, 'pt_xarray_grid')
        pt_df = temperature_function(**pt_config['xarray_grid']['grid_kwargs'])

    else: #build pt profile using param tools built in to param_tools?
        picaso_class.add_pt(P_config = pt_config['pressure'])
        #update param tools with new pressure array
        param_tools.add_class(picaso_class)

        #grab the correct temp function for parameterization
        temperature_function = getattr(param_tools, f'pt_{type}')
        #compute tmeperature with correct parameters
        pt_df = temperature_function(**pt_config[type])
    
    return pt_df

def update_config_w_cube(config_og, fitpars, cube):
    """
    Create function that ensures all parameters are reset 
    """
    log_change = [False]*len(fitpars)
    for i, key in enumerate(fitpars.keys()):
        if not (key.startswith("offset") or key.startswith("scaling") or key.startswith("err_inf") or key.startswith("wvl_offset")):
            log_change[i]=set_dict_value(config_og, key, cube[i])
        else:  
            log_change[i]=True#no need to change config
    
    if np.sum(log_change) != len(fitpars):
        raise Exception('Cound not change config parameters ffrom cube for' , np.array(list(fitpars.keys()))[~np.array(log_change)])
    return config_og

def set_dict_value(data, path_string, new_value):
    """
    Sets the value of a key in a nested dictionary or a 
    column in a DataFrame using a dot-separated path string.
    """
    keys = path_string.split('.')
    current_level = data
    
    for i, key in enumerate(keys):
        # Determine if we are at the target (the last key in the path)
        is_last_key = (i == len(keys) - 1)
        
        if is_last_key:
            # Case 1: Final target is a dictionary key
            if isinstance(current_level, (dict,pd.DataFrame)):
                current_level[key] = new_value
                return True
            
            # Case 2: Final target is a list for intsances where we are editting a list like knots 
            elif isinstance(current_level, list):
                # This replaces the entire column with the new_value
                current_level[int(key)] = new_value
                return True
            
            else:
                print(f"Error: Target container for '{key}' is neither a dict nor a DataFrame.")
                return False
        
        else:
            # Traversal Logic (moving deeper into the structure)
            if isinstance(current_level, dict) and key in current_level:
                current_level = current_level[key]
            
            # Allow traversal through a DataFrame if the path continues
            # Note: This would only work if the DF cell contains another dict/DF
            elif isinstance(current_level, pd.DataFrame) and key in current_level.columns:
                current_level = current_level[key]
                
            else:
                print(f"Error: Path component '{key}' not found or invalid traversal.")
                return False

    return False

def find_values_for_key(data, target_key):
    """
    Recursively crawls a dictionary and its nested dictionaries to find all
    values associated with a specified key, returning them in a list.

    Args:
        data (dict): The dictionary to search.
        target_key (str): The key to search for.

    Returns:
        list: A list of all values found for the target key.
    """
    results = []

    if isinstance(data, dict):
        # Iterate through the dictionary's key-value pairs
        for key, value in data.items():
            # If the current key matches the target key, add the value to the results
            if key == target_key:
                if isinstance(value,str):value=[value]
                results.append(value)
                results = [str(i) for i in np.unique(results)]
            # If the value is another dictionary, recursively call the function
            # and extend the current results list with the results from the nested dictionary
            elif isinstance(value, dict):
                results.extend(find_values_for_key(value, target_key))
            # If the value is a list, iterate through the list items
            # and recursively call the function if an item is a dictionary
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        results.extend(find_values_for_key(item, target_key))
    
    return results

def get_instrument_options():
    pandeia = os.environ.get('pandeia_refdata',None)
    if pandeia == None: 
        return {}
    
    jwst_options = ['niriss','nircam','miri','nirspec']
    dispersion_files = {i:glob.glob(os.path.join(pandeia,'jwst',i,'dispersion','*fits')) for i in jwst_options}
    pretty_options_map={}
    for inst in jwst_options: 
        for f in dispersion_files[inst]: 
            pretty = ' '.join(os.path.splitext(os.path.basename(f))[0].split('_')[0:-2])
            pretty_options_map[pretty] = f

    return pretty_options_map

def get_instrument_R_fits(key): 
    """
    Reads the JWST Pandeia data fits file data 

    Parameters 
    ----------
    filepath : str 
        filepath to pandeia data jwst fits file 
    
    Returns 
    -------
    array, array 
        wavenumber array (ascending order), resolution
    """
    options = get_instrument_options()
    filepath = options.get(key,None)
    if filepath == None: 
        raise Exception('I could not find a jwst filepath with the key you specified:',key)
    with fits.open(filepath) as hdu:
        rdata = hdu[1].data
    w = np.array([i[0] for i in rdata])
    r = np.array([i[2] for i in rdata])
    wno = 1e4/w

    sort_indices = np.argsort(wno)
    wno_sorted = wno[sort_indices]
    w_sorted = w[sort_indices]
    r_sorted = r[sort_indices] 

    return wno_sorted, r_sorted



def viz(picaso_output):
    figs = []

    #spectra 
    spectra_figs = plot_spectra(picaso_output)
    if spectra_figs is not None:
        figs.extend(spectra_figs)

    #pt + mr in same row on dashboard 
    pt_fig = plot_pt(picaso_output)
    mr_fig = plot_mr(picaso_output)
    if pt_fig is not None and mr_fig is not None:
        figs.append(row(pt_fig, mr_fig))
    elif pt_fig is not None:
        figs.append(pt_fig)
    elif mr_fig is not None:
        figs.append(mr_fig)

    #cloud plot 
    cloud_fig = plot_cloud(picaso_output)
    if cloud_fig is not None:
        figs.append(cloud_fig)

    title_div = Div(text="<h1 style='text-align:center;'>Dashboard</h1>")
    output_file("dashboard.html")
    show(column(title_div, *figs, sizing_mode='scale_width'))

    return figs

def plot_spectra(picaso_output):
    figs = []

    if isinstance(picaso_output.get('transit_depth', jpi.np.nan), jpi.np.ndarray):
        figs.append(jpi.spectrum(picaso_output['wavenumber'],
                                 picaso_output['transit_depth'],
                                 title='Transit Depth Spectrum'))

    if isinstance(picaso_output.get('albedo', jpi.np.nan), jpi.np.ndarray):
        figs.append(jpi.spectrum(picaso_output['wavenumber'],
                                 picaso_output['albedo'],
                                 title='Albedo Spectrum'))

    if isinstance(picaso_output.get('thermal', jpi.np.nan), jpi.np.ndarray):
        figs.append(jpi.spectrum(picaso_output['wavenumber'],
                                 picaso_output['thermal'],
                                 title='Thermal Emission Spectrum'))

    if isinstance(picaso_output.get('fpfs_reflected', jpi.np.nan), jpi.np.ndarray):
        figs.append(jpi.spectrum(picaso_output['wavenumber'],
                                 picaso_output['fpfs_reflected'],
                                 title='Reflected Light Spectrum'))

    if isinstance(picaso_output.get('fpfs_thermal', jpi.np.nan), jpi.np.ndarray):
        figs.append(jpi.spectrum(picaso_output['wavenumber'],
                                 picaso_output['fpfs_thermal'],
                                 title='Relative Thermal Emission Spectrum'))

    if isinstance(picaso_output.get('fpfs_total', jpi.np.nan), jpi.np.ndarray):
        figs.append(jpi.spectrum(picaso_output['wavenumber'],
                                 picaso_output['fpfs_total'],
                                 title='Relative Full Spectrum'))


    return figs if figs else None



def plot_pt(picaso_output):
    full_output = picaso_output['full_output']
    fig = jpi.pt(full_output)
    return fig

def plot_cloud(picaso_output):
    full_output = picaso_output['full_output']
    fig = jpi.cloud(full_output)
    return fig

def plot_mr(picaso_output):
    full_output = picaso_output['full_output']
    fig = jpi.mixing_ratio(full_output, plot_type='bokeh', limit= 10) #limit controls the amount of outputs for the plot
    return fig


#two functions to directly pull the master config and prune it of options that are not needed for the driver notebook. This is to avoid confusion with the large number of options in the master config that are not relevant to the driver notebook.
def prune_dict_by_key(obj, sequence):
    # 1. Identify keys at the current level that contain the sequence
    keys_to_delete = [k for k in obj.keys() if sequence in str(k)]
    
    # 2. Delete those keys
    for k in keys_to_delete:
        del obj[k]
    
    # 3. Recurse into remaining dictionary values
    for v in obj.values():
        if isinstance(v, dict):
            prune_dict_by_key(v, sequence)
            
    return obj

def load_template_config():
    master_driver = os.path.join(os.getenv('picaso_refdata'), 'input_tomls', 'driver.toml')
    with open(master_driver, "rb") as f:
        config = tomllib.load(f)
    config = prune_dict_by_key(config,'_options')
    config['OpticalProperties']['opacity_file']=config['OpticalProperties']['opacity_file'].replace('_default_',os.getenv('picaso_refdata'))
    config['OpticalProperties']['virga_mieff']=config['OpticalProperties']['virga_mieff'].replace('_default_',os.getenv('picaso_refdata'))
    config['temperature']['userfile']['filename'] =config['temperature']['userfile']['filename'].replace('_default_',os.getenv('picaso_refdata'))
    config['temperature']['sonora_bobcat']['sonora_path'] =config['temperature']['sonora_bobcat']['sonora_path'].replace('_default_',os.getenv('picaso_refdata'))
    config['chemistry']['userfile']['filename'] =config['chemistry']['userfile']['filename'].replace('_default_',os.getenv('picaso_refdata'))
    config['clouds']['cloud1']['userfile']['filename'] =config['clouds']['cloud1']['userfile']['filename'].replace('_default_',os.getenv('picaso_refdata'))

    return config

def load_config(driver_file): 
    with open(driver_file, "rb") as f:
        config = tomllib.load(f)
    return config