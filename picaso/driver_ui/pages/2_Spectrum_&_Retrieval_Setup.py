# Run PICASO via a Streamlit UI
# 
# run UI locally with: 
# > conda isntall -c conda-forge streamlit
# > pip install streamlit-bokeh
# > streamlit run Run_PICASO.py

# =======================================
# CONSTANTS 
# =======================================
MOLECULES_LIMIT = 10

# =======================================
# ENVIRONMENT SETUP 
# =======================================
import streamlit as st
import os
from pathlib import Path

# HEADER
st.logo('https://natashabatalha.github.io/picaso/_images/logo.png', size="large", link="https://github.com/natashabatalha/picaso")
st.header('Run PICASO',divider='rainbow')
st.subheader('Administrative')

PICASO_REFDATA_ENV_VAR = os.environ.get('picaso_refdata', 'None')
PYSYN_CBDS_ENV_VAR = os.environ.get('PYSYN_CDBS', 'None')
os.environ['picaso_refdata'] = st.text_input("Enter in the datapath to your reference data", value=PICASO_REFDATA_ENV_VAR)
os.environ['PYSYN_CDBS'] = st.text_input("Enter in the datapath to your PYSYN_CBDS data", value=PYSYN_CBDS_ENV_VAR)
# =======================================
# IMPORTS
# =======================================
import pandas as pd
import toml
import tempfile
import tomllib 
import numpy as np 
import copy 

from bokeh.plotting import figure
import matplotlib.pyplot as plt
import bokeh.palettes as pals
from bokeh.models import Legend
from streamlit_bokeh import streamlit_bokeh

import picaso.driver as go
from picaso import justdoit as jdi 
from picaso import WIP_justplotit as jpi
from picaso.parameterizations import Parameterize

# =======================================
# HELPER FUNCTIONS 
# =======================================
def format_config_section_for_df(obj, ignore_keys=None):
    """
    Formats a driver.toml section to be rendered as an input 

    Parameters
    ----------
    obj : dict
        Object that is a section of the driver.toml configuration file
    
    Return
    -------
    Dictionary with formatted keys and values
    """
    pass_to_df = {}
    keys = obj.keys()
    if ignore_keys:
        keys = {key: val for key, val in obj.items() if key not in ignore_keys}
    for attr in keys:
        if not f'{attr}_options' in obj and not attr.endswith('_options') and not attr.endswith('_kwargs'):
            # if there are options, we will display it as a dropdown
            values = obj[attr]
            key = attr

            # handling special types (list & dict)
            if isinstance(obj[attr], list):
                # have to convert #s to string and back
                values = [str(item) for item in obj[attr]]
            if isinstance(obj[attr], dict):
                if 'value' in obj[attr] and 'unit' in obj[attr]:
                    # want to include only values so they're editable
                    values = obj[attr]['value']
                    key = f'{attr} ({obj[attr]['unit']})'
            pass_to_df[key] = values
    
    return pass_to_df

def write_results_to_config(grid, base):
    """
    Writes the results of a Streamlit input component to our configuration dictionary

    Parameters
    ----------
    grid : dict
        Streamlit object
    base : str
        A pointer to where the data should be written to config
    Return
    ------
    Cleaned configuration object
    """
    for item in grid:
        if ' (' in item:
            key, unit = item.split()
            if key.lower() in base:
                base[key.lower()]['value'] = grid[item][0]
            elif key in base:
                base[key]['value'] = grid[item][0]
            else: 
                raise Exception('Could not write value to config file')
        elif isinstance(base[item], list):
            try:
                base[item] = [float(ele) for ele in grid[item][0]]
            except:
                base[item] = [str(ele) for ele in grid[item][0]]
        elif not isinstance(base[item], dict):
            if isinstance(grid[item][0], (np.floating, np.integer)):
                base[item] = grid[item][0].item()
            else:
                base[item] = grid[item][0]
        else: 
            raise Exception('Could not write value to config file')

def update_config_recursively(base, user_defaults):
    """
    Recursively updates the base configuration dictionary with the user defaults.
    """
    for key, val in user_defaults.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            update_config_recursively(base[key], val)
        else:
            base[key] = copy.deepcopy(val)

def clean_dictionary(data, suffix="_options"):
    """
    Recursively removes a certain keyword from any part of a dictionary (used to clean the driver.toml configuration of _options keywords before getting passed to a PICASO function). 
    Also converts numpy types to native python types for TOML serialization.
    
    Parameters
    ----------
    data : dict
        The dictionary with all parameters inputted by the user so far
    suffix : str
        The keyword/pattern/string that will be deleted from the configuration so PICASO doesn't throw errors for unexpected keywords
    Return
    ------
    The cleaned and parsed dictionary
    """
    if isinstance(data, dict):
        return {
            k: clean_dictionary(v, suffix)
            for k,v in data.items()
            if not k.endswith(suffix)
        }
    if isinstance(data, list):
        return [clean_dictionary(v, suffix) for v in data]
    
    if isinstance(data, (np.floating, np.integer, np.str_, np.bool_)):
        return data.item()
    if isinstance(data, np.ndarray):
        return data.tolist()

    return data

def run_spectrum_class(local_param_tools,stage=None):
    """
    Runs driver.py's spectrum class as far as the level specified in stage with current configuration

    Parameters
    ----------
    stage : string
        Options are planet, star, temperature, chemistry; if left blank, the whole class will run (including clouds)
    
    Return
    -------
    picaso.justdoit.inputs
        Configured class
    """
    return go.setup_spectrum_class(clean_dictionary(config), opacity, local_param_tools, stage)

def update_toml_with_a_value_for_a_free_parameter_deprecate(dictionary, keys, value):
    """
    Write a sampled value to a new copy of the main configuration file, used for retrievals
        
    Parameters
    ----------
    dictionary : dict
        The hard copy of the configuration that will get overriden with the sampled value for the specified free parameter
    keys : str
        The path of the free parameter in dictionary
    value : float
        The sampled value, to get written to the dictionary
    """
    stopIndex = -1
    if keys[-1].isdigit():
        stopIndex = -2
    for key in keys[:stopIndex]:
        dictionary = dictionary[key]

    if keys[-1].isdigit(): # to handle lists
        dictionary[keys[-2]][int(keys[-1])] = value
    else:
        dictionary[keys[-1]] = value

def uploaded_config_is_valid(uploaded_config):
    """
    Template validator function for user-uploaded config TOML
        
    Parameters
    ----------
    uploaded_config : dict
        uploaded user config TOML
    Return
    -------
    bool
        True if valid
    """
    if uploaded_config is None:
        return False

    required_structure = {
        'observation_type': None,
        'observation_type_options': None,
        'irradiated': None,
        'calc_type': None,
        'OpticalProperties': ['opacity_file', 'opacity_method', 'virga_mieff'],
        'object': None,
        'temperature': ['pressure', 'profile'],
        'chemistry': ['method']
    }

    is_valid = True

    for key, subkeys in required_structure.items():
        if key not in uploaded_config:
            st.error(f"Missing required top-level key: '{key}'")
            is_valid = False
        elif subkeys:
            if not isinstance(uploaded_config[key], dict):
                st.error(f"Section '{key}' must be a table (dictionary)")
                is_valid = False
                continue
            for subkey in subkeys:
                if subkey not in uploaded_config[key]:
                    st.error(f"Missing required key '{subkey}' in '{key}' section")
                    is_valid = False
    
    # Check if ObservationData exists if it's used
    if 'ObservationData' in uploaded_config:
        if not isinstance(uploaded_config['ObservationData'], dict):
             st.error("Section 'ObservationData' must be a table (dictionary)")
             is_valid = False
    elif 'retrieval' in uploaded_config:
        # Based on driver.toml, it should probably be there
        st.error("Missing 'ObservationData' section when 'retrieval' seection is included. Please ensure it is provided in your configuration or remove retrieval options.")
        is_valid = False

    return is_valid

# ===============================
# STREAMLIT HELPER FUNCTIONS 
# ===============================
def editable_section(section, key, ignore_keys=None):
    df = pd.DataFrame([format_config_section_for_df(section, ignore_keys)])
    edited = st.data_editor(df, key=key)
    write_results_to_config(edited, section)

# ===============================
# GLOBALS
# ===============================

wavelength_range = (0,15)
spectral_resolution = 150
config = None
param_tools = None 
opacity = None
# ---------------------------------------------- #
# -- BEGINNING OF APP -------------------------- #
# ---------------------------------------------- #

# ============================================
# ADMINISTRATIVE CONFIGURATION
# ============================================
def setup_config():
    picaso_ref = os.environ.get('picaso_refdata')
    if picaso_ref and picaso_ref != 'None' and os.path.isdir(picaso_ref):
        DRIVER_CONFIG = os.path.join(picaso_ref, 'input_tomls', 'driver.toml')
        if os.path.exists(DRIVER_CONFIG):
            with open(DRIVER_CONFIG, "rb") as f:
                return tomllib.load(f)
    return None

def reinitialize_paramtools(param_tools, model_dir): 
    """
    This is used for uploading model grids if param tools needs to be reinitalized in the interface

    This is slightly repetetive as the optical cloud properties gets loaded twice. 
    """
    cached_grid = st.session_state.get('param_tools_grid')
    if cached_grid is not None:
        if getattr(cached_grid, 'model_dir', None) == model_dir and getattr(cached_grid, 'mieff_dir', None) == param_tools.mieff_dir:
            return cached_grid

    load_cld_optical = param_tools.load_cld_optical
    mieff_dir=param_tools.mieff_dir
    param_tools = Parameterize(load_cld_optical=load_cld_optical,
                               mieff_dir=mieff_dir,
                               model_dir=model_dir)
    st.session_state['param_tools_grid'] = param_tools
    return param_tools

def render_admin():
    # DATAPATH ENTERING
    check_default_opa = config.get('OpticalProperties').get('opacity_file','')
    if '_default_' in check_default_opa: 
        opa = jdi.opannection()
        default_opa = opa.db_filename
    config['OpticalProperties']['opacity_file'] = st.text_input("Enter in the datapath to your opacities.db", value = default_opa)
    config['OpticalProperties']['opacity_method'] = st.selectbox("Opacity method", ("resampled")) #, "preweighted", "resortrebin"))
    config['OpticalProperties']['virga_mieff'] = st.text_input("Enter in the datapath to your virga files", value = config.get('OpticalProperties').get('virga_mieff').replace('_default_',PICASO_REFDATA_ENV_VAR))

    # OPACITY AND PARAM_TOOLS CONFIG
    opacity = jdi.opannection(
        filename_db=config['OpticalProperties']['opacity_file'], #database
        method=config['OpticalProperties']['opacity_method'], #resampled, preweighted, resortrebin
        **config['OpticalProperties']['opacity_kwargs'] #additonal inputs 
    )

    # for later, once the jdi.vj.available() is updated, below pops can be removed
    mieff_dir = config['OpticalProperties'].get('virga_mieff', None)
    cached_param_tools = st.session_state.get('param_tools')
    if cached_param_tools is not None and getattr(cached_param_tools, 'mieff_dir', None) == mieff_dir:
        param_tools = cached_param_tools
    else:
        a = jdi.vj.available()
        a.pop(a.index('CaAl12O19'))
        a.pop(a.index('CaTiO3'))
        a.pop(a.index('SiO2'))
        param_tools = Parameterize(load_cld_optical=a, mieff_dir=mieff_dir)
        st.session_state['param_tools'] = param_tools

    # CALCULATION TYPE AND OBSERVATION TYPE SETTING
    st.subheader('Select calculation to perform')
    
    upload_option = st.radio("Would you like to upload a toml file to set default values?", ("No", "Yes"), index=0)
    if upload_option == "Yes":
        uploaded_file = st.file_uploader("Choose a TOML file with default values", type="toml")
        if uploaded_file is not None:
            uploaded_config = tomllib.load(uploaded_file)
            update_config_recursively(config, uploaded_config)
            st.success("Successfully loaded user-provided default values.")

    calc_type = config.get('calc_type','spectrum')

    #config['calc_type'] = st.selectbox("Calculation type", ['spectrum','climate'], index=None)
    if calc_type == "spectrum":
        pass
    elif calc_type == 'climate':
        st.warning('Uploaded driver.toml has calc_type set to climate and should be used on the climate page for a climate run. Proceeding to generate spectrum for specified setup but might experience issues if full setup has not been provided. ')

    # TODO : This can eventually be a select multi option 
    config['observation_type'] = st.selectbox("Observation type", config['observation_type_options'], index=None)
    # E.g., "reflected+thermal" or "reflected+transmission"
    if config['observation_type']:
        st.divider()
        st.header(f'{config['observation_type'].capitalize()} Spectrum Config')
    return opacity, param_tools

def render_star():
    # SET IS IRRIDATED
    config['irradiated'] = True
    if config['observation_type'] == 'thermal':
        choice = st.selectbox("Do your want your object to be irradiated?", ('Yes', 'No'), index=None)
        config['irradiated'] = choice == 'Yes'
    if not config['irradiated']:
        if 'star' in config: 
            del config['star']
    # EDITABLE STAR VARIABLES SECTION
    if config['irradiated']:
        st.subheader("Star Variables")
        editable_section(config['star'], 'star', config['star']['type_options'])
        for attr in config['star'].keys():
            if attr.endswith('_options'):
                pure_attr = attr.split('_')[0]
                config['star'][pure_attr] = st.selectbox(f"{pure_attr.capitalize()} Options", config['star'][attr], index=None)
        if (config['star']['type']):
            editable_section(config['star'][config['star']['type']], config['star']['type'])

def render_object():
    st.subheader("Object Variables")
    editable_section(config['object'], 'object')

def render_phase_angle():
    if go.OBSERVATION_CALC_MAP.get(config['observation_type']) == 'reflected':
        config['geometry']['phase']['value'] = st.number_input('Enter phase angle in radians 0-2π', min_value=0, max_value=6, value=0)

# ============================================
# PRESSURE AND TEMPERATURE
# ============================================
def render_pressure_and_temperature(param_tools=None):
    # EDIABLE PRESSURE SECTION
    st.text('Configure pressure (can ignore if using a userfile or sonora bobcat for temperature)')
    editable_section(config['temperature']['pressure'], 'pt')

    # EDITABLE TEMPERATURE SECTION
    temperature_options = [option for option in config['temperature'].keys() if option != 'profile' and option != 'pressure']
    if len(temperature_options) == 0:
        st.warning('No temperature options found in driver.toml file.')
    config['temperature']['profile'] = st.selectbox(
        "Select a temperature profile", temperature_options, index=None 
    )
    temp_profile = config['temperature']['profile']
    if temp_profile:
        temp_profile_obj = config['temperature'][f'{config['temperature']['profile']}']
        formatted_obj = format_config_section_for_df(temp_profile_obj)
        temp_df = pd.DataFrame([formatted_obj])
        temp_grid = st.data_editor(temp_df)
        write_results_to_config(temp_grid, config['temperature'][temp_profile])

        for attr in config['temperature'][temp_profile].keys():
            if attr.endswith('_options'):
                pure_attr = attr.split('_')[0]
                config['temperature'][temp_profile][pure_attr] = st.selectbox(f"{temp_profile.capitalize()} {pure_attr.capitalize()} Options", config['temperature'][temp_profile][attr], index=None)
            elif attr.endswith('model_dir'):
                model_dir = str(config['temperature'][temp_profile]['model_dir'])
                if not os.path.isdir(model_dir): 
                    st.error('Enter valid path to xarray model grid to proceed with this option')
                    st.stop()
                param_tools = reinitialize_paramtools(param_tools,model_dir)
                grid_name = param_tools.grid_name
                grid_parameters_unique = param_tools.interp_params[grid_name]['grid_parameters_unique']
                grid_kwargs=config['temperature'][temp_profile].get('grid_kwargs',{})
                with st.container(border=True):
                    st.session_state['xarray_grid_ranges'] = {}
                    for param_name in grid_parameters_unique:
                        values = grid_parameters_unique[param_name]
                        minn = np.min(values)
                        maxx = np.max(values)
                        st.session_state['xarray_grid_ranges'][param_name] = [minn, maxx]
                        if 'xarray_grid_params' in st.session_state:
                            value = float(st.session_state['xarray_grid_params'].get(param_name,None))
                        else: 
                            value = float(grid_kwargs.get(param_name,None))
                            if value: 
                                if not ((value<=maxx) & (value>=minn)): 
                                    value=None
                        grid_kwargs[param_name] = st.slider(param_name, 
                                                            value = value,
                                                            min_value=minn, max_value=maxx)
                        st.session_state['xarray_grid_params'] = grid_kwargs
                config['temperature'][temp_profile]['grid_kwargs'] =    st.session_state['xarray_grid_params']         

    # GRAPH PRESSURE-TEMPERATURE
    if st.button('See Pressure-Temperature plot'):
        data_class = run_spectrum_class(param_tools, 'temperature')
        analyzer = jpi.Analyzer({'layer': data_class.inputs['atmosphere']['profile']}, backend='plotly')
        st.plotly_chart(analyzer.pt(), use_container_width=True)
    return param_tools
# ============================================
# INPUT CHEMISTRY INFORMATION
# ============================================
def get_possible_molecules(): 
    possible_cont=[]
    continuum_sources = ['H2','He','H','H2-','H-']
    for icont in list(opacity.avail_continuum): #e.g., H2H2, H2He H-ff etc 
        for ispe in continuum_sources:
            if ispe in icont: possible_cont+=[ispe]
    all_mols = list(set(list(opacity.molecules) + possible_cont))
    # Get current molecules from config by checking against all_mols
    return all_mols

def render_free_chem_options():
    """
    Renders the UI for free chemistry options. 

    At the moment quite hard coded for the current options. This will not 
    work freely if major input changes happen to the free chemistry specification. 
    """
    st.subheader("Free Chemistry Configuration")
    
    # 1) Select molecules
    all_mols = get_possible_molecules()
    current_mols = [k for k in config['chemistry']['free'].keys() if k in all_mols]

    selected_mols = st.multiselect("Select molecules for chemistry", all_mols, default=current_mols)
    
    

    profile_options_dict = config['chemistry']['free']['profile_options']
    
    # Remove molecules that are no longer selected
    for mol in list(config['chemistry']['free'].keys()):
        if mol not in selected_mols and mol not in ['background', 'method','species']:
            if 'options' not in mol: del config['chemistry']['free'][mol]

    # 2) Render subheaders and options for each molecule
    num_background = 0 
    background_config = dict(gases=[])
    for mol in selected_mols:
        st.subheader(mol)
        if mol not in config['chemistry']['free']:
            config['chemistry']['free'][mol] = {'profile': 'constant', 'unit': 'v/v'}
        
        mol_config = config['chemistry']['free'][mol]
        
        # Ensure profile exists
        if 'profile' not in mol_config:
            mol_config['profile'] = 'constant'
            
        selected_profile = st.selectbox(
            f"Select profile type for {mol}", 
            list(profile_options_dict.keys()), 
            index=list(profile_options_dict.keys()).index(mol_config['profile']),
            key=f"profile_{mol}"
        )
        mol_config['profile'] = selected_profile
        
        if 'background' in selected_profile: 
            num_background += 1 
            background_config['gases']+=[mol]
            if num_background>2: st.error('Only can support up to two background gases')
        else: 
            # 3) Render inputs based on profile_options
            for param in profile_options_dict[selected_profile]:
                if param == 'interpolation_method':
                    mol_config[param] = st.text_input(
                        f"{param} for {mol}", 
                        value=mol_config.get(param, 'slinear'),
                        key=f"{param}_{mol}"
                    )
                elif param in ['P_knots', 'vmr_knots']:
                    # Handle list inputs as comma-separated strings
                    current_val = mol_config.get(param, [])
                    if isinstance(current_val, list):
                        current_val_str = ", ".join([str(v) for v in current_val])
                    else:
                        current_val_str = str(current_val)
                        
                    input_str = st.text_input(
                        f"{param} for {mol} (comma separated)", 
                        value=current_val_str,
                        key=f"{param}_{mol}"
                    )
                    try:
                        mol_config[param] = [float(x.strip()) for x in input_str.split(',') if x.strip()]
                    except ValueError:
                        st.error(f"Error parsing {param} for {mol}. Please enter numeric values separated by commas.")
                
                else:
                    # Numeric inputs
                    mol_config[param] = st.number_input(
                        f"{param} for {mol}", 
                        value=float(mol_config.get(param, 0.0)),
                        format="%.2e",
                        key=f"{param}_{mol}"
                    )
    
    # 5) Incorporate background gas logic
    if num_background>1:
        st.subheader("Background Gas Fraction")
        mol1 = background_config['gases'][0]
        mol2 = background_config['gases'][1]
        fraction = st.number_input(
            rf'Fraction between {mol1}:{mol2}', 
            value=float(background_config.get('fraction', 5.667)),
            key="bg_fraction"
        )
        background_config['fraction'] = fraction
        config['chemistry']['free']['background'] = background_config
    elif num_background == 1: 
        if 'fraction' in config['chemistry']['free']['background']: del config['chemistry']['free']['background']['fraction']
    elif num_background == 0:
        if 'background' in config['chemistry']['free']:
            del config['chemistry']['free']['background']


    if 'gases' in background_config: 
        for i in background_config['gases']: 
            selected_mols.pop(selected_mols.index(i))
            if i in config['chemistry']['free']: del config['chemistry']['free'][i]

    config['chemistry']['free']['species']=selected_mols

def render_xarray_chem(): 
    grid_params = st.session_state.get('xarray_grid_params',None)
    if not grid_params: 
        st.error('If using xarray grid for chemistry must select the parameters through temperature pressure profile for consistency')
        st.stop()
    molecules_grid = param_tools.species
    molecules_opa = get_possible_molecules()
    default = [i for i in molecules_opa if i in molecules_grid]
    selected_mols = st.multiselect("Select subset of molecules to include in spectra", molecules_grid, default=default)

    config['chemistry']['xarray_grid'] = config['temperature']['xarray_grid']
    config['chemistry']['xarray_grid']['grid_kwargs']['molecules'] = selected_mols


def render_chemistry():
    # SET CHEMISTRY METHOD
    chemistry_options = [option for option in config['chemistry'] if option != 'method']
    if len(chemistry_options) == 0: st.warning('No chemistry option found in driver.toml.')
    config['chemistry']['method'] = st.selectbox("How to model chemistry", chemistry_options, index=None)
    chem_method = config['chemistry']['method']

    if chem_method:
        # RENDER FREE CHEMISTRY 
        if 'free' in chem_method:
            render_free_chem_options()
        elif 'xarray' in chem_method: 
            render_xarray_chem()
        else:
            # EDITABLE SECTION FOR OTHER CHEMISTRY METHODS
            editable_section(config['chemistry'][f'{config['chemistry']['method']}'], 'chemistry')

    # GRAPH MIXING RATIOS
    if st.button('See Mixing Ratios'):
        try:
            data_class = run_spectrum_class(param_tools,'chemistry')
            chem_df = data_class.inputs['atmosphere']['profile']
            # form {mixingratios: {'H20': [...], ...}} to pass to jpi.mixing_ratio
            # chem_df.keys() would have [temperature, pressure, H20, CO2, <other example molecules> ]
            for key in chem_df.keys():
                if key != 'pressure' or key != 'temperature':
                    chem_df[key] = chem_df[key]
            full_output = dict({'layer':{'pressure': chem_df['pressure'], 'mixingratios': chem_df}})
            analyzer = jpi.Analyzer(full_output, backend='plotly')
            st.plotly_chart(analyzer.mixing_ratio(), use_container_width=True)
        except Exception as e:
            st.warning('Make sure you have configured chemistry and temperature.')
            st.write(e)

# ============================================
# CLOUDS
# TODO: add option to do multiple cloud types (cloud1, 2 3 etc ..)
# ============================================
def render_clouds():
    include_clouds = st.selectbox("Do you want clouds?", ('Yes', 'No'), index=None)
    if include_clouds == 'Yes':
        num_clouds = st.number_input("How many cloud types?", min_value=1, value=1)
        
        # Cleanup extra clouds from config if num_clouds decreased
        all_cloud_keys = [k for k in config['clouds'].keys() if k.startswith('cloud')]
        for k in all_cloud_keys:
            try:
                if k.startswith('cloud') and not k.endswith('_type'):
                    idx = int(k.replace('cloud', ''))
                    if idx > num_clouds:
                        del config['clouds'][k]
                        if f'{k}_type' in config['clouds']:
                            del config['clouds'][f'{k}_type']
            except ValueError:
                pass

        for i in range(1, int(num_clouds) + 1):
            cloud_id = f'cloud{i}'
            st.subheader(f"Cloud {i} Configuration")
            
            if cloud_id not in config['clouds']:
                config['clouds'][cloud_id] = copy.deepcopy(config['clouds']['cloud1'])
            
            cloud_obj = config['clouds'][cloud_id]

            # set cloud type
            current_type = config['clouds'].get(f'{cloud_id}_type')
            type_options = list(cloud_obj.keys())
            try:
                type_index = type_options.index(current_type) if current_type in type_options else 0
            except ValueError:
                type_index = 0
                
            cloud_type = st.selectbox(f"Cloud type for {cloud_id}", type_options, index=type_index, key=f"{cloud_id}_type_select")
            config['clouds'][f'{cloud_id}_type'] = cloud_type

            # create editable df for cloud so users can set parameters
            cloud_type_df = pd.DataFrame([format_config_section_for_df(cloud_obj[cloud_type])])
            cloud_type_editable_df = st.data_editor(cloud_type_df, key=f"{cloud_id}_{cloud_type}_editor")
            
            # render any options sections dynamically
            cloud_list_iterate = copy.deepcopy(config['clouds'][cloud_id][cloud_type])
            for attr in cloud_list_iterate:
                if attr.endswith('_options'):
                    pure_attr = '_'.join(attr.split('_')[:-1])
                    
                    current_opt = cloud_obj[cloud_type].get(pure_attr)
                    options = cloud_obj[cloud_type][attr]
                    try:
                        opt_index = options.index(current_opt) if current_opt in options else 0
                    except ValueError:
                        opt_index = 0

                    cloud_obj[cloud_type][pure_attr] = st.selectbox(
                        f"{cloud_id} {pure_attr.capitalize()} Options", 
                        options, 
                        index=opt_index,
                        key=f"{cloud_id}_{cloud_type}_{pure_attr}_select"
                    )
                    
                    # render any kwargs for the options dynamically
                    if cloud_obj[cloud_type][pure_attr]:
                        var_with_options = cloud_obj[cloud_type][pure_attr]
                        kwargs_key = var_with_options + '_kwargs'
                        if kwargs_key in cloud_obj[cloud_type]:
                            editable_section(cloud_obj[cloud_type][kwargs_key],key=f"{cloud_id}_{cloud_type}_{kwargs_key}_editor")
                            #options_editable_df = st.data_editor(
                            #    cloud_obj[cloud_type][kwargs_key],
                            #    key=f"{cloud_id}_{cloud_type}_{kwargs_key}_editor"
                            #)
                            #cloud_obj[cloud_type][kwargs_key] = options_editable_df

            write_results_to_config(cloud_type_editable_df, config['clouds'][cloud_id][cloud_type])
        
        ##########################
        # GRAPH CLOUDS
        ##########################
        if st.button('See Clouds'):
            try:
                data_class = run_spectrum_class(param_tools)
                if 'clouds' in data_class.inputs and 'profile' in data_class.inputs['clouds']:
                    df = data_class.inputs['clouds']['profile'].astype('float')
                    wavenumber = df['wavenumber'].unique()
                    nwno = len(wavenumber)
                    wavelength = 1e4/wavenumber
                    pressure = df['pressure'].unique()
                    nlayer = len(pressure)
                    
                    # reconstruct expected structured full_output for Analyzer.cloud()
                    # It needs: full_output['layer']['cloud'] with keys 'w0', 'opd', 'g0'
                    # and full_output['layer']['pressure'] and full_output['wavenumber']
                    w0_reshaped = np.reshape(df['w0'].values, (nlayer, nwno))
                    opd_reshaped = np.reshape(df['opd'].values, (nlayer, nwno))
                    g0_reshaped = np.reshape(df['g0'].values, (nlayer, nwno))
                    
                    full_output_cloud = {
                        'layer': {
                            'pressure': pressure,
                            'cloud': {
                                'w0': w0_reshaped,
                                'opd': opd_reshaped,
                                'g0': g0_reshaped
                            }
                        },
                        'wavenumber': wavenumber
                    }
                    analyzer = jpi.Analyzer(full_output_cloud, backend='plotly')
                    st.plotly_chart(analyzer.cloud(), use_container_width=True)
                else:
                    st.warning("No cloud profile generated. Please check cloud configuration.")
            except Exception as e:
                st.warning('Make sure you have configured chemistry and temperature.')
                st.write(e)
    else:
        if 'clouds' in config:
            del config['clouds']

def render_velocities_deprecate():
    st.subheader("Doppler and Rotational Velocities (Optional)")
    include_doppler= st.selectbox("Do you want to add a doppler shift?", ('Yes', 'No'), index=None)
    if include_doppler == 'Yes':
        editable_section(config['doppler_shift'], 'doppler_shift')
    else: 
        if 'doppler_shift' in config: 
            del config['doppler_shift']
    
    include_rotation= st.selectbox("Do you want to add rotational broadening?", ('Yes', 'No'), index=None)
    if include_rotation == 'Yes':
        editable_section(config['rotational_broadening'], 'rotational_broadening')
    else: 
        if 'rotational_broadening' in config: 
            del config['rotational_broadening']
        
def render_wavelength_range(opacity):
    return st.slider(
        "Select wavelength range (μm)",
        min_value=np.min(1e4/opacity.wno),
        max_value=np.max(1e4/opacity.wno),
        value=(np.min(1e4/opacity.wno), np.max(1e4/opacity.wno))
    )

def render_spectral_resolution():
    return st.number_input('Spectral Resolution', min_value=10, value=150)

def get_nan_ranges(wvl, mask):
    """
    Finds contiguous NaN wavelength ranges for processed model so that user can make sure 
    they are okay with the nans that exist in the processed data.
    """
    if not np.any(mask):
        return []
    
    nan_indices = np.where(mask)[0]
    ranges = []
    if len(nan_indices) > 0:
        start_idx = nan_indices[0]
        for i in range(1, len(nan_indices)):
            if nan_indices[i] != nan_indices[i-1] + 1:
                ranges.append((wvl[start_idx], wvl[nan_indices[i-1]]))
                start_idx = nan_indices[i]
        ranges.append((wvl[start_idx], wvl[nan_indices[-1]]))
    return ranges

# ---------------------------------#
# RUN A SPECTRUM ----------------- #
# ---------------------------------#
def run_spectrum():
    """
    If users clicks run spectrum this will run the spectrum and create a figure that then 
    can be passed 

    Returns 
    bokeh.Figure 
        If users clicks run spectrum 
    None 
        If user does not click run spectrum 
    """
    if config['calc_type'] =='spectrum' and st.button(f'Run {config['calc_type']}'):
        try:
            clean_dict = clean_dictionary(config)
            df, picaso_class = go.run(driver_dict=clean_dict, return_class=True)
            observation_key = config['observation_type']
            resultx, resulty = df['wavenumber'] , df[observation_key]
            
            processed = go.process_model(resultx, resulty, 
                                         config=clean_dict, 
                                         regrid_R=spectral_resolution)
            
            wavenumber, albedo_or_fluxes, mask = processed['model']

            if np.any(mask):
                nan_count = np.sum(mask)
                total_count = len(mask)
                nan_ranges = get_nan_ranges(1e4/wavenumber, mask)
                range_str = ", ".join([f"{r[0]:.2f}-{r[1]:.2f} um" if r[0] != r[1] else f"{r[0]:.2f} um" for r in nan_ranges])
                st.warning(f"Warning: Encoutered {nan_count} NaNs out of {total_count} model points. NaN wavelength ranges: {range_str}")

            spec_fig = jpi.spectrum(wavenumber, albedo_or_fluxes, plot_width=500, x_range=wavelength_range, backend='plotly')
            
            st.session_state['spectrum_results'] = {
                'df': df,
                'picaso_class': picaso_class,
                'wavenumber': wavenumber,
                'albedo_or_fluxes': albedo_or_fluxes,
                'spec_fig': spec_fig,
                'observation_key': observation_key
            }
        except Exception as e:
            st.warning('Make sure you have configured temperature, pressure, and chemistry before running a spectrum.')
            st.write(e)
        st.divider()

    if 'spectrum_results' in st.session_state:
        results = st.session_state['spectrum_results']
        # plot spectrum
        st.plotly_chart(results['spec_fig'], use_container_width=True)

        # DOWNLOAD BUTTONS
        col1, col2 = st.columns(2)
        with col1:
            # NetCDF Download
            if 'netcdf_data' not in results:
                try:
                    ds = jdi.output_xarray(results['df'], results['picaso_class'])
                    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                        ds.to_netcdf(tmp.name)
                        with open(tmp.name, "rb") as f:
                            results['netcdf_data'] = f.read()
                    os.remove(tmp.name)
                except Exception as e:
                    st.error(f"Error creating NetCDF: {e}")
            
            if 'netcdf_data' in results:
                st.download_button(
                    label="Download PICASO xarray (NetCDF)",
                    data=results['netcdf_data'],
                    file_name="picaso_spectrum.nc",
                    mime="application/x-netcdf"
                )

        with col2:
            # ASCII Download
            if 'ascii_data' not in results:
                df_ascii = pd.DataFrame({
                    'wavenumber': results['wavenumber'],
                    results['observation_key']: results['albedo_or_fluxes']
                })
                results['ascii_data'] = df_ascii.to_csv(index=False)

            st.download_button(
                label="Download ASCII (wavenumber, spectrum)",
                data=results['ascii_data'],
                file_name="picaso_spectrum.csv",
                mime="text/csv"
            )
        return results['spec_fig']

    return None

def render_free_parameter_selection():
    parameter_handler = {}
    st.subheader("Select which available free parameters you'd like to do a retrieval on:")
    # clean up config so only selected options are shown
    config['temperature'] = {
        config['temperature']['profile']: config['temperature'][config['temperature']['profile']],
        'pressure': config['temperature']['pressure'],
        'profile': config['temperature']['profile']
    }
    config['chemistry'] = {
        config['chemistry']['method']: config['chemistry'][config['chemistry']['method']],
        'method': config['chemistry']['method']
    }
    # clean up free chemistry parameters if they are selected
    if config['chemistry']['method'] == 'free':
        free_chem = config['chemistry']['free']
        profile_options = free_chem.get('profile_options', {})
        for mol in free_chem.get('species', []):
            if mol in free_chem:
                current_profile = free_chem[mol].get('profile')
                if current_profile in profile_options:
                    allowed = profile_options[current_profile]
                    if not isinstance(allowed, list): 
                        allowed = []
                    keep = ['profile', 'unit'] + allowed
                    free_chem[mol] = {k: v for k, v in free_chem[mol].items() if k in keep}

    if 'clouds' in config:
        new_clouds = {}
        for k in config['clouds'].keys():
            if k.endswith('_type'):
                cloud_id = k.replace('_type', '')
                cloud_type = config['clouds'][k]
                new_clouds[k] = cloud_type
                new_clouds[cloud_id] = {cloud_type: config['clouds'][cloud_id][cloud_type]}
        config['clouds'] = new_clouds
    if 'retrieval' in config:
        st.session_state['priors_from_config'] = copy.deepcopy(config['retrieval'])
        del config['retrieval']
    else:
        st.session_state['priors_from_config'] = None

    def list_available_free_parameters(data, current_path=""):
        for key, value in data.items():
            new_path = f"{current_path}.{key}" if current_path else key

            if isinstance(value, dict):
                list_available_free_parameters(value, new_path)
            elif isinstance(value, (float, np.floating)):
                parameter_handler[new_path] = [st.checkbox(f"{new_path} {value}"), value]
            elif isinstance(value, (int, np.integer)) and not isinstance(value, bool) and key != 'nlevel':
                parameter_handler[new_path] = [st.checkbox(f"{new_path} {value}"), value]
            elif isinstance(value, list) and all(isinstance(item, (int, float, np.integer, np.floating)) for item in value):
                for index, item in enumerate(value):
                    parameter_handler[new_path + f'.{index}'] = [st.checkbox(f"{new_path + f'.{index}'} {item}"), item]
    list_available_free_parameters(config)
    return parameter_handler


def find_prior_for_key(priors, key):
    """
    Looks up a parameter's prior configuration in the provided retrieval dictionary.
    Supports both flat string-path keys and nested dictionary configurations, and handles
    cloud name differences (e.g. cloud1 vs generic cloud).
    """
    if not priors:
        return None
    
    # 1. Exact match in flat priors (e.g., priors["object.radius"])
    if key in priors:
        return priors[key]
    
    # 2. Nested dictionary lookups (e.g., priors["object"]["radius"])
    parts = key.split('.')
    current = priors
    found = True
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            found = False
            break
    if found and isinstance(current, dict) and ('prior' in current or 'type' in current or 'uniform_kwargs' in current or 'gaussian_kwargs' in current):
        return current

    # 3. Fallback: try removing cloud identifiers (e.g. "clouds.cloud1.slab-grey.ptop" -> "clouds.slab-grey.ptop")
    import re
    simplified_key = re.sub(r'clouds\.cloud\d+\.', 'clouds.', key)
    if simplified_key != key:
        if simplified_key in priors:
            return priors[simplified_key]
        
        parts = simplified_key.split('.')
        current = priors
        found = True
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                found = False
                break
        if found and isinstance(current, dict) and ('prior' in current or 'type' in current or 'uniform_kwargs' in current or 'gaussian_kwargs' in current):
            return current
            
    return None


def render_ranges_for_selected_parameters(parameter_handler):
    # filter for what items have been selected
    prior_set_items = {}
    selected_items = {path_to_parameter: state_value_list[1] for path_to_parameter, state_value_list in parameter_handler.items() if state_value_list[0]}

    priors_cfg = st.session_state.get('priors_from_config')

    # Min, Max, Log, Prior Type Listing
    # Right Now not swapping out Gaussian Kwargs for Uniform Kwargs...
    for i, (key, value) in enumerate(selected_items.items()):
        st.subheader(key)

        p_dict = find_prior_for_key(priors_cfg, key) if priors_cfg else None

        p_type = 'uniform'
        log_val = False
        min_val = None
        max_val = None
        mean_val = None
        std_val = None

        #get ranges from grid kwargs as a first pass
        if 'xarray' in key: 
            grid_par_name = key.split('.')[-1]
            min_val = st.session_state['xarray_grid_ranges'][grid_par_name][0]
            max_val = st.session_state['xarray_grid_ranges'][grid_par_name][1]

        if p_dict and isinstance(p_dict, dict):
            p_type = p_dict.get('prior') or p_dict.get('type') or 'uniform'
            log_val = p_dict.get('log', False)
            
            # Extract uniform values if present
            uni_kwargs = p_dict.get('uniform_kwargs') or p_dict.get('uniform_options') or {}
            min_val = uni_kwargs.get('min') if 'min' in uni_kwargs else p_dict.get('min')
            max_val = uni_kwargs.get('max') if 'max' in uni_kwargs else p_dict.get('max')
            
            # Extract gaussian values if present
            gauss_kwargs = p_dict.get('gaussian_kwargs') or p_dict.get('gaussian_options') or {}
            mean_val = gauss_kwargs.get('mean') if 'mean' in gauss_kwargs else p_dict.get('mean')
            std_val = gauss_kwargs.get('std') if 'std' in gauss_kwargs else p_dict.get('std')

        if p_type not in ['uniform', 'gaussian']:
            p_type = 'uniform'


        prior_type_index = 0 if p_type == 'uniform' else 1

        prior_type = st.selectbox('prior', ['uniform', 'gaussian'], index=prior_type_index, key=f'prior{i}')
        prior_set_items[key] = dict(
            log=st.text_input('log', log_val, key=f'log{i}'),
            prior=prior_type
        )
        # if value == 0:
        #     value = 0.00001
        if prior_type == 'uniform':
            if min_val is None:
                min_val = value * 0.75
            if max_val is None:
                max_val = value * 1.25
            prior_set_items[key][f'{prior_type}_kwargs'] = dict(
                min=st.number_input('min', value=float(min_val), min_value=None, max_value=None, key=f'min{i}', format="%.6f"),
                max=st.number_input('max', value=float(max_val), min_value=None, max_value=None, key=f'max{i}', format="%.6f"),
            )
        else:
            if mean_val is None:
                mean_val = value
            if std_val is None:
                std_val = 1.0
            prior_set_items[key][f'{prior_type}_kwargs'] = dict(
                mean=st.number_input('mean', float(mean_val), key=f'mean{i}'),
                std=st.number_input('std', float(std_val), key=f'std{i}'),
            )
    return prior_set_items

def sampler(prior_set_items, nsamples):
    ALL_TOMLS = []
    save_all_class_pt = []
    for _ in range(nsamples):
        # get samples for values
        check_all_values = go.hypercube(np.random.rand(len(prior_set_items.keys())), dict(prior_set_items))
        
        # create a new copy of the config to write to
        GUESS_TOML = copy.deepcopy(config)
        # write sampled values to config
        #for index, free_parameter in enumerate(prior_set_items.keys()):
        #    if any(free_parameter.startswith(s) for s in ['err_inf', 'offset', 'scaling']):
        #        continue
        #    sampled_value = check_all_values[index]
        #    keys = free_parameter.split('.')
        #    update_toml_with_a_value_for_a_free_parameter(GUESS_TOML, keys, sampled_value)
        GUESS_TOML = go.update_config_w_cube(GUESS_TOML, prior_set_items,check_all_values)

        # save that config
        ALL_TOMLS.append(GUESS_TOML)
        # run config through spectrum class
        data_class = go.setup_spectrum_class(clean_dictionary(GUESS_TOML), opacity, param_tools)
        # extract results needed for graphs
        t = data_class.inputs['atmosphere']['profile']['temperature']
        p = data_class.inputs['atmosphere']['profile']['pressure']
        cloud_profile = data_class.inputs['clouds']['profile']
        mixingratios = data_class.inputs['atmosphere']['profile']
        # parse
        for key in mixingratios.keys():
            if key != 'pressure' or key != 'temperature':
                mixingratios[key] = mixingratios[key]
        molecules = [mol for mol in mixingratios.keys() if mol not in ['pressure', 'temperature', 'kz']][:MOLECULES_LIMIT]
        # save information
        save_all_class_pt.append({
            'temperature':t,
            'pressure':p,
            'mixingratios':mixingratios,
            'molecules': molecules,
            'cloudprofile': cloud_profile
        })
    return ALL_TOMLS, save_all_class_pt

def sample_plots(ALL_TOMLS, save_all_class_pt, nsamples, run_clouds=True, run_spectrum=True):
    import plotly.graph_objects as go_plotly
    import plotly.express as px
    from plotly.subplots import make_subplots

    ################################
    # PRESSURE TEMPERATURE GRAPH
    ################################
    pressure_temperature_fig = go_plotly.Figure()
    for i in range(nsamples):
        pressure = save_all_class_pt[i]['pressure']
        temperature = save_all_class_pt[i]['temperature']
        pressure_temperature_fig.add_trace(go_plotly.Scatter(
            x=temperature, y=pressure,
            mode='lines',
            line=dict(color='red', width=1.5),
            opacity=0.3,
            showlegend=False
        ))
    pressure_temperature_fig.update_xaxes(title_text="Temperature (K)")
    pressure_temperature_fig.update_yaxes(type="log", title_text="Pressure(Bars)", autorange="reversed")
    pressure_temperature_fig.update_layout(title=f"Pressure-Temperature Profiles ({nsamples} Samples)")

    ################################
    # MIXING RATIO GRAPH
    ################################
    mixing_ratio_fig = go_plotly.Figure()
    molecules = save_all_class_pt[0]['molecules']
    
    cols = px.colors.qualitative.Plotly
    shown_legends = set()
    for i in range(nsamples):
        pressure = save_all_class_pt[i]['pressure']
        mixingratios = save_all_class_pt[i]['mixingratios']
        for idx, mol in enumerate(molecules):
            color = cols[idx % len(cols)]
            show_leg = (mol not in shown_legends)
            mixing_ratio_fig.add_trace(go_plotly.Scatter(
                x=mixingratios[mol], y=pressure,
                mode='lines',
                name=mol,
                line=dict(color=color, width=2),
                opacity=0.3,
                showlegend=show_leg,
                legendgroup=mol
            ))
            if show_leg:
                shown_legends.add(mol)
    mixing_ratio_fig.update_xaxes(type="log", title_text="Mixing Ratio(v/v)", range=[-20, 1])
    mixing_ratio_fig.update_yaxes(type="log", title_text="Pressure(Bars)", autorange="reversed")
    mixing_ratio_fig.update_layout(title="Mixing Ratios")

    ################################
    # CLOUDS GRAPH
    ################################
    clouds_fig = None
    if run_clouds:
        clouds_fig = make_subplots(
            rows=1, cols=3, 
            subplot_titles=("Single scattering albedo vs Pressure", "Asymmetry vs Pressure", "Optical Depth vs Pressure")
        )
        for i in range(nsamples):
            cloud_df = save_all_class_pt[i]['cloudprofile']
            cloud_pressure = cloud_df['pressure'].unique()
            wavenumber = cloud_df['wavenumber'].unique()

            nwno = len(wavenumber)
            nlayer = len(cloud_pressure)

            w0 = np.reshape(cloud_df['w0'].values, (nlayer, nwno))
            opd = np.reshape(cloud_df['opd'].values, (nlayer, nwno)) + 1e-60
            g0 = np.reshape(cloud_df['g0'].values, (nlayer, nwno))

            ssa1d = np.mean(w0, axis=1)
            g01d = np.mean(g0, axis=1)
            opd1d = np.mean(opd, axis=1)

            clouds_fig.add_trace(go_plotly.Scatter(x=ssa1d, y=cloud_pressure, mode='lines', line=dict(color='blue', width=1.5), opacity=0.3, showlegend=False), row=1, col=1)
            clouds_fig.add_trace(go_plotly.Scatter(x=g01d, y=cloud_pressure, mode='lines', line=dict(color='green', width=1.5), opacity=0.3, showlegend=False), row=1, col=2)
            clouds_fig.add_trace(go_plotly.Scatter(x=opd1d, y=cloud_pressure, mode='lines', line=dict(color='red', width=1.5), opacity=0.3, showlegend=False), row=1, col=3)
            
        clouds_fig.update_yaxes(type="log", autorange="reversed")
        clouds_fig.update_xaxes(title_text="Single Scattering Albedo", row=1, col=1)
        clouds_fig.update_xaxes(title_text="Asymmetry", row=1, col=2)
        clouds_fig.update_xaxes(type="log", title_text="Optical Depth", row=1, col=3)
        clouds_fig.update_layout(height=500, width=1200)

    ################################
    # SPECTRUM GRAPH
    ################################
    spectrum_fig = None
    if run_spectrum:
        DATA_DICT, CONV_DICT = go.get_data(config)
        
        WNO_LIST = []
        ALB_LIST = []

        lolg_nan = 0
        for idx, prior_toml in enumerate(ALL_TOMLS):
            clean_dict = clean_dictionary(prior_toml)
            df = go.run(driver_dict=clean_dict)
            obs_key = prior_toml['observation_type']
            resultx, resulty = df['wavenumber'] , df[obs_key]
            
            processed = go.process_model(resultx, resulty, 
                                         data_dict=DATA_DICT, 
                                         config=clean_dict, 
                                         conv_dict=CONV_DICT)

            for key in processed:
                x,y,e = DATA_DICT[key]
                x_rebinned, y_rebinned, mask = processed[key]
                valid = ~mask
                mockliklihood = np.sum((y[valid]**2-y_rebinned[valid]**2)/e[valid]**2)
                if np.isnan(mockliklihood):
                    lolg_nan+=1

                WNO_LIST.append(x_rebinned)
                ALB_LIST.append(y_rebinned)

                if np.any(mask):
                    nan_count = np.sum(mask)
                    total_count = len(mask)
                    nan_ranges = get_nan_ranges(1e4/x_rebinned, mask)
                    range_str = ", ".join([f"{r[0]:.2f}-{r[1]:.2f} um" if r[0] != r[1] else f"{r[0]:.2f} um" for r in nan_ranges])

                if np.any(mask) and np.isnan(mockliklihood):
                    st.warning(f'Sample {idx+1}: Tried masking {nan_count} points out of {total_count} model points but NaN still persisting in mock likelihood for {key}. Masked points are here: {range_str}')
                elif np.any(mask) and not np.isnan(mockliklihood):
                    st.warning(f'Sample {idx+1}: Masked {nan_count} points out of {total_count} model points but with the mask I can successful compute a likelihood for {key}. Masked points are here: {range_str}')

        spectrum_fig = jpi.spectrum(WNO_LIST, ALB_LIST, palette=['rgba(255,0,0,0.3)'], plot_width=500, x_range=wavelength_range, backend='plotly')
        
        for i in DATA_DICT.keys(): 
            x,y,e = DATA_DICT[i]
            spectrum_fig = jpi.plot_errorbar(1e4/x,y,e,plot=spectrum_fig, backend='plotly')

    return pressure_temperature_fig, mixing_ratio_fig, clouds_fig, spectrum_fig

def render_additional_retrieval_parameters(parameter_dict, config):
    #adds other retrieval parameters: vsini, data offsets, error inflation
    set_instruments=[None]*len(config['ObservationData']['filenames'])
    loaded_options_mapping = go.get_instrument_options()
    if len(loaded_options_mapping.keys())>=1:
        st.subheader("Set Instrument Convolution")
        for ind, f in enumerate(config['ObservationData']['filenames']):
            name = os.path.splitext(os.path.basename(f))[0]
            with st.expander(f"Resolution file to convolve with {name}"):
                set_instruments[ind] = st.selectbox(
                    "Select an instrument", loaded_options_mapping.keys(), index=None, key=name)
        config['ObservationData']['instruments'] = set_instruments
    
    for f in config['ObservationData']['filenames']:
        st.subheader("Data Systematics")
        name = os.path.splitext(os.path.basename(f))[0]
        with st.expander(f"Systematics for {name}"):
            if st.checkbox(f"Add error inflation term", key=f"err_inf_{name}"):
                parameter_dict[f'err_inf.{name}'] = [True, 0.0]
            if st.checkbox(f"Add instrumental offset", key=f"offset_{name}"):
                parameter_dict[f'offset.{name}'] = [True, 0.0]
            if st.checkbox(f"Add scaling term", key=f"scaling_{name}"):
                parameter_dict[f'scaling.{name}'] = [True, 1.0]
    return parameter_dict

def render_retrievals(spectrum_figure=None):
    """
    Sets up the retrieval figure. 

    If spectrum_figure is passed it can be used for the data configuration plot 

    Parameters
    ----------
    spectrum_figure : bokeh.Figure 
        Optional, will plot the spectrum created along with the data
    """
    # Configure Data 
    st.subheader("Observational Data Configuration")
    default_paths = config.get('ObservationData', {}).get('filepaths', [])
    if isinstance(default_paths,list):
        if len(default_paths)>1: 
            default_paths="\n".join(default_paths)
        elif len(default_paths)==1: 
            default_paths=str(default_paths[0])
        else: 
            default_paths=''
    
    obs_data_input = st.text_area("Enter in the datapath(s) to your observation data (one per line). Xarray or CSV format allowed. For CSV, only 1 row with comma-separated header is allowed to be read as pandas.read_csv(filename). For xarray, coord and data_vars must be set appropriately.", 
                                   value=default_paths)
    
    # Process the text area input into a list
    if obs_data_input:
        config['ObservationData']['filenames'] = [line.strip() for line in obs_data_input.split('\n') if line.strip()]
    else:
        config['ObservationData']['filenames'] = []

    # Determine default data
    units_df = format_config_section_for_df(config['ObservationData'], ignore_keys=['filenames','instruments'])
    
    st.text('Specify column, coord, or data_var names and units. Units are only required if not specified through xarray.')
    st.text('Units should be in astropy format. If unitless (e.g., albedo, or transit depth) enter unitless ')
    edited_units = st.data_editor(pd.DataFrame([units_df]))
    #edited_units = edited_units.to_dict(orient='records')[0]
    for i in edited_units.keys(): 
        config['ObservationData'][i]=edited_units[i].values[0]

    if spectrum_figure is not None:
        col1, col2 = st.columns(2)
        with col1:
            plot_only_data = st.button("Data check & plot")
        with col2:
            plot_with_ref = st.button("Data check & plot w/ ref model")
    else:
        plot_only_data = st.button("Data check & plot")
        plot_with_ref = False

    if plot_with_ref or plot_only_data:
        #parses data and plots it to verify it is correct
        try:
            data_dict,conv_dict = go.get_data(config)
            st.success(f"Successfully parsed {len(data_dict)} files: {list(data_dict.keys())}")
            for key, val in data_dict.items():
                st.write(f"**{key}**: {len(val[0])} points, wavenumber range: {min(val[0]):.2f} - {max(val[0]):.2f}")
            
            if plot_with_ref and 'spectrum_results' in st.session_state:

                clean_dict = clean_dictionary(config)
                observation_key = config['observation_type']
                df = st.session_state['spectrum_results']['df']
                resultx, resulty = df['wavenumber'] , df[observation_key]
            
                processed = go.process_model(resultx, resulty, 
                                             data_dict=data_dict,conv_dict=conv_dict,
                                         config=clean_dict)
                x,y,l = [],[],[]
                for i in processed.keys(): 
                    x_rebinned, y_rebinned, mask = processed[i]
                    x+=[x_rebinned]
                    y+=[y_rebinned]
                    l+=[i]
                    if np.any(mask):
                        st.warning(f"Reference model binned to {i} data grid encountered {np.sum(mask)} NaNs out of {len(y_rebinned)} points.")

                basefig = jpi.spectrum(x, y, 
                                       plot_width=500, 
                                       x_range=wavelength_range,
                                       legend=l,
                                       backend='plotly')
            else:
                basefig = None
                
            fig = basefig
            for i in data_dict.keys():
                x,y,e=data_dict[i]
                fig = jpi.plot_errorbar(1e4/x,y,e,plot=fig, backend='plotly')
            
            st.session_state['data_plot'] = fig
        except Exception as e:
            st.error(f"Error parsing data: {e}")

    if 'data_plot' in st.session_state:
        st.plotly_chart(st.session_state['data_plot'], use_container_width=True)

    # ADD Additional retrieval parameters?
    parameter_handler = render_additional_retrieval_parameters({}, config)

    # LIST OUT ALL FREE PARAMETERS
    parameter_handler.update(render_free_parameter_selection())

    prior_set_items = {}
    retrieval_stage_state_manager = {} # for streamlit rendering organization

    # WHEN USER IS DONE SELECTING, RENDER ALL RANGES FOR SELECTED PARAMETERS
    retrieval_stage_state_manager['done_selecting_parameters'] =  st.selectbox("Done Selecting Free Parameters", ("Yes", "No"), index=None)
    if retrieval_stage_state_manager['done_selecting_parameters'] == 'Yes':
        prior_set_items = render_ranges_for_selected_parameters(parameter_handler)
    
    retrieval_object = {}

    st.divider()
    st.subheader('Sampler Options')

    retrieval_object['sampler']={}

    code_options = retrieval_object.get('sampler',{}).get('code_options',['dynesty', 'ultranest'])

    retrieval_object['sampler']['code'] =  st.selectbox("Choose bayesian code to use", code_options, index=None)

    retrieval_object['sampler']['sampler_kwargs']  = eval(
        st.text_input("Enter sampler_kwargs as parsable dictionary e.g., {'live_points' : 700}.", 
                      value=str(retrieval_object['sampler'].get('sampler_kwargs',{}))))
    
    retrieval_object['sampler']['run_kwargs'] = eval(
        st.text_input("Enter run_kwargs as parsable dictionary e.g., {'max_iter' : 10000}.", 
                      value=str(retrieval_object['sampler'].get('run_kwargs',{}))))

    config['InputOutput']['retrieval_output'] =st.text_input("Enter retrieval run output path", 
                          value='')
    
    
    st.divider()
    st.subheader("Set and test your prior bounds")
    st.text('Now that you have set prior ranges you can use the functionality below to test the prior. Below you can run X-number of samples through the prior ranges and visualize chemistry, p-t profiles, and spectra.')

    ALL_TOMLS = []
    save_all_class_pt = []
    nsamples = st.number_input('Number of samples?', 5)
    

    # extract data to be able to write to toml to recreate
    for parameter in prior_set_items.keys():
        base = prior_set_items[parameter]
        prior_type = base['prior']
        retrieval_variables = {
            'prior' : prior_type,
            'log' : base['log'],
            f'{prior_type}_kwargs' : base[f'{prior_type}_kwargs']
        }

        prev = retrieval_object
        for i, key in enumerate(parameter.split('.')):
            if i == len(parameter.split('.')) -1:
                prev[key] = retrieval_variables
            else:
                if key not in prev:
                    prev[key] = {}
            prev = prev[key]

    # WHEN USER IS GOOD WITH RANGES/PRIORS, SAMPLE VALUES AND CREATE PLOTS
    retrieval_stage_state_manager['done_configuring_priors'] =  st.selectbox("Done Configuring Priors", ("Yes", "No"), index=None)
    if retrieval_stage_state_manager['done_configuring_priors'] == 'Yes':
        ALL_TOMLS, save_all_class_pt = sampler(prior_set_items, nsamples)        
        pressure_temperature_fig, mixing_ratio_fig, clouds_fig, _ = sample_plots(ALL_TOMLS, save_all_class_pt, nsamples, run_spectrum=False, run_clouds=('clouds' in config))

        # PLOT PT, MR, CLOUDS
        st.plotly_chart(pressure_temperature_fig, use_container_width=True)
        st.plotly_chart(mixing_ratio_fig, use_container_width=True)
        if 'clouds' in config and clouds_fig is not None:
            st.plotly_chart(clouds_fig, use_container_width=True)

        st.divider()

        # PLOT SPECTRUM
        retrieval_stage_state_manager['see_prior_spectrums'] =  st.selectbox("Run Samples for Full Spectrums?", ("Yes", "No"), index=None)
        if retrieval_stage_state_manager['see_prior_spectrums'] == 'Yes':
            _, _, _, spectrum_fig = sample_plots(ALL_TOMLS, save_all_class_pt, nsamples, run_clouds=('clouds' in config))
            st.plotly_chart(spectrum_fig, use_container_width=True)
    
    
    
    
    return retrieval_object

def render_download_config(retrieval_object):
    cleaned_config = clean_dictionary(config)
    if 'retrieval' in cleaned_config:
        del cleaned_config['retrieval']
    # TODO: change writing reitreval stuff to use kwargs
    if retrieval_object != {}:
        cleaned_config['retrieval'] = retrieval_object

    st.code("""
# runs retrieval
import picaso.driver as d
d.retrieve(driver_file="configured_toml.toml")

# recreates the spectrum
d.run(driver_file="configured_toml.toml")""")
    st.download_button(
        label="Download current config",
        data=toml.dumps(cleaned_config),
        file_name="configured_toml.toml",
        mime="application/toml"
    )
# ===========================
# MAIN
# =========================== 
config = setup_config()
if config is None:
    st.error('Cannot find driver.toml file. Please ensure that os.environ[\'picaso_refdata\'] is set correctly in the Administrative section or on the Reference Data page.')
else:
    opacity, param_tools = render_admin()
    if config['observation_type']:
        render_star()
        render_object()
        render_phase_angle()

        # ATMOSPHERIC VARIABLES
        st.subheader("Atmospheric Variables")
        param_tools=render_pressure_and_temperature(param_tools=param_tools)
        render_chemistry()
        render_clouds()

        #VELOCITIES (doppler and/or rotational)
        #added to object properties instead 
        #render_velocities()


        # SPECTRUM
        wavelength_range = render_wavelength_range(opacity)
        spectral_resolution = render_spectral_resolution()
        spec_figure = run_spectrum()
            
        # RETRIEVALS
        retrieval_object = {}
        st.header("Retrievals")
        if st.selectbox("Do you want to do a retrieval?", ('Yes', 'No'), index=None) == 'Yes':
            retrieval_object = render_retrievals(spectrum_figure=spec_figure)

        render_download_config(retrieval_object)