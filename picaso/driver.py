from .justdoit import *
from .justplotit import *
from .parameterizations import Parameterize,cloud_averaging
from .retrieval import Parameterize_DEPRECATE
#import picaso.justdoit as jdi
#import picaso.justplotit as jpi
import tomllib 
import toml
import dynesty
from collections.abc import Mapping
from scipy import stats
import dill
import dynesty.utils
from functools import partial
dynesty.utils.pickle_module = dill


chem_options = ['visscher', 'free', 'userfile']
cloud_options = ['brewster_grey', 'brewster_mie', 'virga', 'flex_fsed', 'hard_grey', 'userfile']
pt_options = ['userfile','isothermal', 'knots', 'guillot', 'sonora_bobcat',  'madhu_seager_09_inversion','madhu_seager_09_noinversion' ] # 'zj_24', 'molliere_20', 'Kitzman_20', 

def run(driver_file=None,driver_dict=None):
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
    param_tools = Parameterize(load_cld_optical=preload_cloud_miefs,
                                    mieff_dir=virga_mieff)
    
    #setup opacity outside main run
    opacity = opannection(
        filename_db=config['OpticalProperties']['opacity_files'], #database(s)
        method=config['OpticalProperties']['opacity_method'], #resampled, preweighted, resortrebin
        **config['OpticalProperties']['opacity_kwargs'] #additonal inputs 
        )
     
    if config['calc_type'] =='spectrum':
        picaso_class = setup_spectrum_class(config,opacity,param_tools)
        output =  picaso_class.spectrum(opacity, full_output=True, calculation = config['observation_type']) 
    
    elif config['calc_type']=='retrieval':
        output = retrieve(config, param_tools)

    ### I made these # because they stopped the run fucntion from doing return out and wouldn't let me use my plot PT fucntion
    elif config['calc_type']=='climate':
        raise Exception('WIP not ready yet')
        out = climate(config)

    return output 

def is_valid_astropy_unit(unit_str):
    try:
        u.Unit(unit_str)  # will raise ValueError if invalid
        return True
    except (ValueError, TypeError):
        return False

#retrieval funs
def get_data(config): 
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
    if obs_type=='thermal':
        to_fit = 'flux'
    elif obs_type=='reflected':
        to_fit = 'flux'
    elif obs_type=='transmission':
        to_fit = 'transit_depth'

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
        
        final['micron'] = (dat.coords['wavelength'].values)
        final['wavenumber'] = 1e4/final['micron']

	    #always ensure we are ordered correctly
        final = final.sort_values(by='wavenumber').reset_index(drop=True)

	    #return a nice dictionary with the info we need 
        returns = {key: [final['wavenumber'].values, 
	             final['y'].values, final['e'].values]   }
        
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

def hypercube(u, fitpars):
    x=np.empty(len(u))

    for i,key in enumerate(fitpars.keys()):
        if fitpars[key]['prior'] == 'uniform':
            minn=fitpars[key]['min']
            maxx=fitpars[key]['max']
            x[i] = minn+(maxx-minn)*u[i]
        elif fitpars[key]['prior'] == 'gaussian':
            mean=fitpars[key]['mean']
            std=fitpars[key]['std']
            x[i]=stats.norm.ppf(u[i], loc=mean, scale=std)
        else:
            raise Exception('Prior type not available')
        if fitpars[key]['log']:
            x[i]=10**x[i]  
    return x

def MODEL(cube, fitpars, config, OPA, param_tools, DATA_DICT, retrieval=True):
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

    Returns
    -------
    dict
        Dictionary with the same keys as observation data. 
        Each value is an array with shape:
          - (len(xdata),) if input was 1D
          - (N_samples, len(xdata)) if input was 2D
    """
    cube = np.atleast_2d(cube)  # ensure shape (N_samples, N_params)
    n_samples = cube.shape[0]

    # initialize storage
    y_model = {key: [] for key in config['InputOutput']['observation_data']}

    if not retrieval:
        profiles={}                                   

    for j,row in enumerate(cube):
        # update parameters
        for i, key in enumerate(fitpars.keys()):
            if not (key.startswith("offset") or key.startswith("scaling") or key.startswith("err_inf")):
                set_dict_value(config, key + ".value", row[i])

        # compute spectrum
        picaso_class = setup_spectrum_class(config, opacity=OPA, param_tools=param_tools)
        out = picaso_class.spectrum(OPA, full_output=True, calculation=config['observation_type'])

        R_dict = config['object']['radius']
        R = R_dict['value'] * u.Unit(R_dict['unit']).to(u.m)
        d_dict = config['object']['distance']
        d = d_dict['value'] * u.Unit(d_dict['unit']).to(u.m)

        resultx = out['wavenumber']
        resulty = 1e-8 * (R / d) ** 2 * out['thermal']

        # rebin to observed wavelengths
        for obs_key in config['InputOutput']['observation_data']:
            xdata, _, _ = DATA_DICT[obs_key]
            _, rebinned = mean_regrid(resultx, resulty, newx=xdata)
            y_model[obs_key].append(rebinned)

        if not retrieval:
            profiles[j]=picaso_class.inputs['atmosphere']['profile']

    # stack results into arrays
    for obs_key in y_model:
        y_model[obs_key] = np.vstack(y_model[obs_key])
        # if single sample, flatten to 1D
        if n_samples == 1:
            y_model[obs_key] = y_model[obs_key][0]

    if retrieval:
        return y_model
    else:
        return y_model, profiles

def log_likelihood(cube, fitpars, config, OPA, DATA_DICT, param_tools): 
    y_model_dict = MODEL(cube, fitpars, config, OPA, param_tools, DATA_DICT)

    ydat_all=[];ymod_all=[];sigma_all=[];extra_term_all=[]

    for i,key in enumerate(config['InputOutput']['observation_data']):
        xdata,ydata,edata = DATA_DICT[key]
        y_model=y_model_dict[key]

        #add offsets
        if key in config.get("retrieval", {}).get("offset", {}):
            icube = list(fitpars.keys()).index(f'offset.{key}')
            offset = cube[icube]
        else:
            offset = 0
        ydata += offset

        #add scalings
        if key in config.get("retrieval", {}).get("scaling", {}):
            icube = list(fitpars.keys()).index(f'scaling.{key}')
            scaling = cube[icube]
        else:
            scaling = 1
        ydata*=scaling

        #add error inflation if they exist
        if key in config.get("retrieval", {}).get("err_inf", {}):
            icube = list(fitpars.keys()).index(f'err_inf.{key}')
            err_inf = cube[icube]
        else:
            err_inf=0
        sigma = edata**2 + err_inf #there are multiple ways to do this, here just adding in an extra noise term

        extra_term = np.log(2*np.pi*sigma)

        ydat_all.append(ydata);ymod_all.append(y_model);sigma_all.append(sigma);extra_term_all.append(extra_term); 

    ymod_all = np.concatenate(ymod_all)    
    ydat_all = np.concatenate(ydat_all)    
    sigma_all = np.concatenate(sigma_all)  
    extra_term_all = np.concatenate(extra_term_all)

    return -0.5*np.sum((ydat_all-ymod_all)**2/sigma_all + extra_term_all)

def convolver(newx, x, y):
    #
    return y

def retrieve(config, param_tools):
    print('Running retrieval...')

    # copy input toml into output folder for reproducibility
    output_file_name = config['InputOutput']['retrieval_output']+"/inputs.toml"
    with open(output_file_name, "w") as toml_file:
        toml.dump(config, toml_file)

    OPA = opannection(
        filename_db=config['OpticalProperties']['opacity_files'], #database(s)
        method=config['OpticalProperties']['opacity_method'], #resampled, preweighted, resortrebin
        **config['OpticalProperties']['opacity_kwargs'] #additonal inputs 
        )
    
    os.makedirs(config['InputOutput']['retrieval_output'], exist_ok=True)

    prior_config=config['retrieval']
    
    fitpars=prior_finder(prior_config)
    ndims=len(fitpars)
    
    # def convolver(newx,x,y):
    #     return newy
      
    DATA_DICT = get_data(config)
    hypercube_fn = partial(hypercube, fitpars=fitpars)
    loglike_fn = partial(log_likelihood, fitpars=fitpars, config=config,
                  OPA=OPA, param_tools=param_tools, DATA_DICT=DATA_DICT)
    
    #pool (MPI for clusters)

    #doing dynesty but this should be generic
    sampler_args = prior_config['sampler']['sampler_kwargs']
    if prior_config['sampler']['resume']:
        print('Resuming retrieval...')
        sampler = dynesty.NestedSampler.restore(config['InputOutput']['retrieval_output']+'/dynesty.save')
    else:
        sampler = dynesty.NestedSampler(loglike_fn, hypercube_fn, ndims, nlive=prior_config['sampler']['nlive'], bootstrap=0) 
    sampler.run_nested(checkpoint_file=config['InputOutput']['retrieval_output']+'/dynesty.save', **sampler_args)
    dill.dump(sampler, open(config['InputOutput']['retrieval_output']+'/sampler.pkl', 'wb'))
    return sampler

def check_model_samples(config, N=100, sampler=None):
    """
    Tests the prior distribution by generating models based on the provided configuration.

    Args:
        config (dict): Configuration dictionary containing the necessary parameters for 
            optical properties, retrieval, and other settings.
            - 'OpticalProperties': A dictionary with keys:
                - 'opacity_files' (str): Path to the opacity database(s).
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
        filename_db=config['OpticalProperties']['opacity_files'], #database(s)
        method=config['OpticalProperties']['opacity_method'], #resampled, preweighted, resortrebin
        **config['OpticalProperties']['opacity_kwargs'] #additonal inputs 
        )
    preload_cloud_miefs = find_values_for_key(config ,'condensate')
    virga_mieff   = config['OpticalProperties'].get('virga_mieff',None)
    param_tools = Parameterize(load_cld_optical=preload_cloud_miefs,
                                        mieff_dir=virga_mieff)
    fitpars=prior_finder(config['retrieval'])
    ndims=len(fitpars)
    if sampler is not None:
        thetas = sampler.results.samples_equal()
    else:
        cube = np.random.random([N, ndims])
        thetas = [hypercube(cube[i], fitpars) for i in range(N)]
    
    thetas = np.array(thetas)

    DATA_DICT = get_data(config)

    models, profiles = MODEL(thetas[:N], fitpars, config, OPA, param_tools, DATA_DICT, retrieval=False)

    return models, profiles

def setup_spectrum_class(config, opacity , param_tools ):

    if isinstance(opacity,type(None)):
        opacity = opannection(
        filename_db=config['OpticalProperties']['opacity_files'], #database(s)
        method=config['OpticalProperties']['opacity_method'], #resampled, preweighted, resortrebin
        **config['OpticalProperties']['opacity_kwargs'] #additonal inputs 
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
    A.phase_angle(rad) #input the radian angle of the event/geometry of browndwarf/planet

    try:
        A.gravity(gravity     = config['object'].get('gravity', {}).get('value',None), 
                gravity_unit= u.Unit(config['object'].get('gravity', {}).get('unit',None)), 
                radius      = config['object'].get('radius', {}).get('value',None), 
                radius_unit = u.Unit(config['object'].get('radius', {}).get('unit',None)), 
                )
    except:
        A.gravity(radius      = config['object'].get('radius', {}).get('value',None), 
                radius_unit = u.Unit(config['object'].get('radius', {}).get('unit',None)), 
                mass        = config['object'].get('mass', {}).get('value',None), 
                mass_unit   = u.Unit(config['object'].get('mass', {}).get('unit',None))
                )
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
    
    #WIP TODO: A.surface_reflect()

    
    # tempreature 
    pt_config = config['temperature']
    df_pt = PT_handler(pt_config, A, param_tools) #datafile for pressure temperature profile
    A.atmosphere(df=df_pt) #will include chemistry if it was added to userfile
    param_tools.add_class(A)

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
        df_mixingratio  = chemistry_function(**chem_config[chem_type])#note, this includes P and T already
    
    #set final with chem
    A.atmosphere(df = df_mixingratio)

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

    else: #build pt profile using param tools built in to param_tools?
        picaso_class.add_pt(P_config = pt_config['pressure'])
        #update param tools with new pressure array
        param_tools.add_class(picaso_class)

        #grab the correct temp function for parameterization
        temperature_function = getattr(param_tools, f'pt_{type}')
        #compute tmeperature with correct parameters
        pt_df = temperature_function(**pt_config[type])
    
    return pt_df
    

def viz(picaso_output): 
    spectrum_plot_list = []

    if isinstance(picaso_output.get('transit_depth', np.nan), np.ndarray):
        spectrum_plot_list += [spectrum(picaso_output['wavenumber'], picaso_output['transit_depth'], title='Transit Depth Spectrum')]

    if isinstance(picaso_output.get('albedo', np.nan), np.ndarray):
        spectrum_plot_list += [spectrum(picaso_output['wavenumber'], picaso_output['albedo'], title='Albedo Spectrum')]

    if isinstance(picaso_output.get('thermal', np.nan), np.ndarray):
        spectrum_plot_list += [spectrum(picaso_output['wavenumber'], picaso_output['thermal'], title='Thermal Emission Spectrum')]

    if isinstance(picaso_output.get('fpfs_reflected', np.nan), np.ndarray):
        spectrum_plot_list += [spectrum(picaso_output['wavenumber'], picaso_output['fpfs_reflected'], title='Reflected Light Spectrum')]

    if isinstance(picaso_output.get('fpfs_thermal', np.nan), np.ndarray):
        spectrum_plot_list += [spectrum(picaso_output['wavenumber'], picaso_output['fpfs_thermal'], title='Relative Thermal Emission Spectrum')]

    if isinstance(picaso_output.get('fpfs_total', np.nan), np.ndarray):
        spectrum_plot_list += [spectrum(picaso_output['wavenumber'], picaso_output['fpfs_total'], title='Relative Full Spectrum')]

    output_file("spectrum_output.html")
    show(column(children=spectrum_plot_list, sizing_mode="scale_width"))
    
    return spectrum_plot_list


def set_dict_value(data, path_string, new_value):
    """
    Sets the value of a key in a nested dictionary using a dot-separated path string.

    For example, a path_string of "details.owner.id" will set the 'id' key
    inside the 'owner' dictionary, which is inside the 'details' dictionary.

    Args:
        data (dict): The dictionary to modify.
        path_string (str): The dot-separated key path to the target value.
        new_value: The new value to set.

    Returns:
        bool: True if the value was successfully set, False otherwise.
    """
    keys = path_string.split('.')
    current_level = data
    
    # Traverse the dictionary down to the final key
    for i, key in enumerate(keys):
        # Check if we are at the final key in the path
        if i == len(keys) - 1:
            # Set the value for the final key
            if isinstance(current_level, dict) and key in current_level:
                current_level[key] = new_value
                return True
            else:
                print(f"Error: The path to key '{path_string}' is invalid or does not exist.")
                return False
        else:
            # Check if the next key in the path exists and is a dictionary
            if isinstance(current_level, dict) and key in current_level and isinstance(current_level[key], dict):
                current_level = current_level[key]
            else:
                print(f"Error: The path is invalid. '{path_string}' is not a dictionary or does not exist.")
                return False

def plot_pt_profile(full_output, **kwargs):
    fig = pt(full_output, **kwargs)
    show(fig)
    return fig

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