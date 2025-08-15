from .justdoit import *
from .justplotit import *
from .parameterizations import Parameterize,cloud_averaging
#import picaso.justdoit as jdi
#import picaso.justplotit as jpi
import tomllib 

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

    ### I made these # because they stopped the run fucntion from doing return out and wouldn't let me use my plot PT fucntion
    elif config['calc_type']=='climate':
        raise Exception('WIP not ready yet')
        out = climate(config)

    return output 

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

#def prior_handler(): 
#    returnm 

"""def retrieve(config):
    OPA = opannection(
        filename_db=config['OpticalProperties']['opacity_files'], #database(s)
        method=config['OpticalProperties']['opacity_method'], #resampled, preweighted, resortrebin
        **config['OpticalProperties']['opacity_kwargs'] #additonal inputs 
        )
    def read_data(): 

    def prior(): 
        prior_handler
    
    def loglikelihood(): 
        chi2 = 

    def model(): 
        y = sepctrum(config, OPA=OPA)
        spectra_interp = convolver(**convolverinputs)
        offset = {}
        error_inf = {} 
        return wno, spectra_interp,offset,error_inf
    
    def convolver(newx,x,y)
        return newy
    
    ultranest.run(logliklihood, prior, hyperparams)

"""    

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

    A.gravity(gravity     = config['object'].get('gravity', {}).get('value',None), 
              gravity_unit= u.Unit(config['object'].get('gravity', {}).get('unit',None)), 
              radius      = config['object'].get('radius', {}).get('value',None), 
              radius_unit = u.Unit(config['object'].get('radius', {}).get('unit',None)), 
              mass        = config['object'].get('mass', {}).get('value',None), 
              mass_unit   = u.Unit(config['object'].get('mass', {}).get('unit',None)))
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
                print(f"Error: The path to key '{key}' is invalid or does not exist.")
                return False
        else:
            # Check if the next key in the path exists and is a dictionary
            if isinstance(current_level, dict) and key in current_level and isinstance(current_level[key], dict):
                current_level = current_level[key]
            else:
                print(f"Error: The path is invalid. '{key}' is not a dictionary or does not exist.")
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