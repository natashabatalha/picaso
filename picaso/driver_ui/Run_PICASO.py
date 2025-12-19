# Run PICASO via a Streamlit UI
# 
# run UI locally with: 
# > streamlit run driver_ui.py

# =======================================
# CONSTANTS 
# =======================================
MOLECULES_LIMIT = 50

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
# optional to set the below, but convenient so you don't have to set it in the UI every time
PICASO_REFDATA_ENV_VAR = "/Users/sjanson/Desktop/code/picaso/reference"
PYSYN_CBDS_ENV_VAR = '/Users/sjanson/Desktop/code/picaso/reference/grp/redcat/trds'
if st.selectbox('Do you need to specify paths for your environment variables?', ['Yes', 'No']) == 'Yes':
    os.environ['picaso_refdata'] = st.text_input("Enter in the datapath to your reference data", value=PICASO_REFDATA_ENV_VAR)
    os.environ['PYSYN_CDBS'] = st.text_input("Enter in the datapath to your PYSYN_CBDS data", value=PYSYN_CBDS_ENV_VAR)

"""TODO
PICASO_REFDATA_ENV_VAR = os.environ.get('picaso_refdata',"None")
PYSYN_CBDS_ENV_VAR = os.environ.get('picaso_refdata',"PYSYN_CDBS")
msg1 = f'We have autodetected these paths: {PICASO_REFDATA_ENV_VAR}, Do you need to change paths for your environment variables? ', ['Yes', 'No'])'
mgs2 = 'We dont have env vars set please set them nw'
if st.selectbox(f'We have autodetected these paths: {PICASO_REFDATA_ENV_VAR}, Do you need to change paths for your environment variables?', ['Yes', 'No']) == 'Yes':
    os.environ['picaso_refdata'] = st.text_input("Enter in the datapath to your reference data", value=PICASO_REFDATA_ENV_VAR)
    os.environ['PYSYN_CDBS'] = st.text_input("Enter in the datapath to your PYSYN_CBDS data", value=PYSYN_CBDS_ENV_VAR)
"""
# =======================================
# IMPORTS
# =======================================
import pandas as pd
import toml
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
from picaso import justplotit as jpi
from picaso.parameterizations import Parameterize

# =======================================
# HELPER FUNCTIONS 
# =======================================
def format_config_section_for_df(obj):
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
    for attr in obj.keys():
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
            if key.lower() in base and grid[item][0]:
                base[key.lower()]['value'] = grid[item][0]
        elif isinstance(base[item], list):
            try:
                base[item] = [float(ele) for ele in grid[item][0]]
            except:
                base[item] = [str(ele) for ele in grid[item][0]]
        elif not isinstance(base[item], dict):
            if isinstance(grid[item][0], np.int64):
                base[item] = int(grid[item][0])
            else:
                base[item] = grid[item][0]

def clean_dictionary(data, suffix="_options"):
    """
    Recursively removes a certain keyword from any part of a dictionary (used to clean the driver.toml configuration of _options keywords before getting passed to a PICASO function)
    
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
    return data

def run_spectrum_class(stage=None):
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
    return go.setup_spectrum_class(clean_dictionary(config), opacity, param_tools, stage)

def update_toml_with_a_value_for_a_free_parameter(dictionary, keys, value):
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
    dict : dict
        uploaded user config TOML
    Return
    -------
    bool
        True if valid
    """
    return uploaded_config is not None

# ===============================
# STREAMLIT HELPER FUNCTIONS 
# ===============================
def editable_section(section, key):
    df = pd.DataFrame([format_config_section_for_df(section)])
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
    if st.selectbox('Do you want to upload or provide the datapath to driver.toml?', ['Datapath', 'Upload']) == 'Upload':
        uploaded_file = st.file_uploader("Choose a TOML config file", type="toml")
        if uploaded_file is not None:
            uploaded_config = tomllib.load(uploaded_file)
            # TODO update validator to check that TOML is in correct format (has _options, has necessary sections, etc.)
            if uploaded_config_is_valid(uploaded_config):
                return uploaded_config
    else:
        # dynamically finds a driver.toml in the below datapath platform independently
        # DRIVER_CONFIG = "/Users/sjanson/Desktop/code/picaso/reference/input_tomls/driver.toml"
        DRIVER_CONFIG = str(Path(__file__).resolve().parents[2] / "reference" / "input_tomls" / "driver.toml")
        st.text_input('Enter path to driver.toml', value=DRIVER_CONFIG)
        if isinstance(DRIVER_CONFIG, str):
            with open(DRIVER_CONFIG, "rb") as f:
                return tomllib.load(f) 

def render_admin():
    # DATAPATH ENTERING
    config['OpticalProperties']['opacity_files'] = st.text_input("Enter in the datapath to your opacities.db", value = config.get('OpticalProperties').get('opacity_files'))
    config['OpticalProperties']['opacity_method'] = st.selectbox("Opacity method", ("resampled")) #, "preweighted", "resortrebin"))
    config['OpticalProperties']['virga_mieff'] = st.text_input("Enter in the datapath to your virga files", value = config.get('OpticalProperties').get('virga_mieff'))

    # OPACITY AND PARAM_TOOLS CONFIG
    opacity = jdi.opannection(
        filename_db=config['OpticalProperties']['opacity_files'], #database(s)
        method=config['OpticalProperties']['opacity_method'], #resampled, preweighted, resortrebin
        **config['OpticalProperties']['opacity_kwargs'] #additonal inputs 
    )
    # TODO turn below into input: go.find_values_for_key(config ,'condensate')
    # TODO once the jdi.vj.available() is updated, below pops can be removed
    a =jdi.vj.available()
    a.pop(a.index('CaAl12O19'))
    a.pop(a.index('CaTiO3'))
    a.pop(a.index('SiO2'))
    param_tools = Parameterize(load_cld_optical=a, mieff_dir=config['OpticalProperties'].get('virga_mieff', None))

    # CALCULATION TYPE AND OBSERVATION TYPE SETTING
    st.subheader('Select calculation to perform')
    config['calc_type'] = st.selectbox("Calculation type", ['spectrum','climate'], index=None)
    if config['calc_type'] == "spectrum":
        config['observation_type'] = st.selectbox("Observation type", config['observation_type_options'], index=None)
    elif config['calc_type'] != None:
        st.warning(f'The {config['calc_type']} option has not been implemented yet.')
    # TODO : This can be a select multi option 
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
    
    # EDITABLE STAR VARIABLES SECTION
    # TODO : User should input star.keys() values [type, star_radius, semi_major]. Then be prompted for star[option].
    # This follows the structure of temperature ? 
    if config['irradiated']:
        st.subheader("Star Variables")
        # TODO: pull these values from driver.toml
        star_df = pd.DataFrame({
            'radius': [1],
            'r unit': 'Rsun',
            'semi_major': '200',
            'unit': 'AU',
            'temperature': [5400],
            'metallicity': [0.01],
            'logg': [4.45],
            # type
        })
        star_grid = st.data_editor(star_df)

        # updating config file
        for key in star_grid:
            if key.lower() in config['object'] and star_grid[key][0]:
                config['star'][key.lower()]['value'] = star_grid[key][0]

def render_object():
    st.subheader("Object Variables")
    editable_section(config['object'], 'object')

def render_phase_angle():
    if 'reflected' in (config['observation_type']):
        config['geometry']['phase']['value'] = st.number_input('Enter phase angle in radians 0-2π', min_value=0, max_value=6, value=0)

# ============================================
# PRESSURE AND TEMPERATURE
# ============================================
def render_pressure_and_temperature():
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

        for attr in config['temperature'][temp_profile].keys():
            if attr.endswith('_options'):
                pure_attr = attr.split('_')[0]
                config['temperature'][temp_profile][pure_attr] = st.selectbox(f"{temp_profile.capitalize()} {pure_attr.capitalize()} Options", config['temperature'][temp_profile][attr], index=None)
        write_results_to_config(temp_grid, config['temperature'][temp_profile])

    # GRAPH PRESSURE-TEMPERATURE
    if st.button('See Pressure-Temperature graph'):
        data_class = run_spectrum_class('temperature')
        streamlit_bokeh(jpi.pt({'layer': data_class.inputs['atmosphere']['profile']}))

# ============================================
# INPUT CHEMISTRY INFORMATION
# ============================================
def render_chemistry():
    # SET CHEMISTRY METHOD
    chemistry_options = [option for option in config['chemistry'] if option != 'method']
    if len(chemistry_options) == 0: st.warning('No chemistry option found in driver.toml.')
    config['chemistry']['method'] = st.selectbox("How to model chemistry", chemistry_options, index=None)
    chem_method = config['chemistry']['method']

    if chem_method:
        # RENDER FREE CHEMISTRY 
        if 'free' in chem_method:
            # TODO : This could be cleaned up if driver.py / drover.toml was simplified somehow (long term)

            # FREE CHEM: FORMAT EDITOR BOX
            molecules = [mole for mole in config['chemistry'][chem_method] if mole != 'background']
            mole_unit = config['chemistry'][chem_method][molecules[0]]['unit']
            molecule_values = []
            for mole in molecules:
                if 'values' in config['chemistry'][chem_method][mole]:
                    molecule_values.append([str(ele) for ele in config['chemistry'][chem_method][mole]['values']])
                elif 'value' in config['chemistry'][chem_method][mole] and '[' in str(config['chemistry'][chem_method][mole]['value']):
                    molecule_values.append([str(ele) for ele in config['chemistry'][chem_method][mole]['value']])
                elif 'value' in config['chemistry'][chem_method][mole]:
                    molecule_values.append([str(config['chemistry'][chem_method][mole]['value'])])
            chem_free_df = pd.DataFrame({
                f'Molecule ({mole_unit})': molecules,
                'Values': molecule_values,
                'Pressures (bar)': [[str(ele) for ele in config['chemistry'][chem_method][mole].get('pressures', '')] for mole in molecules],
            })
            st.info('Molecule names are case sensitive (ex: TiO, H2O). You only need to specify a pressure if you provide multiple values for a molecule (to indicate what altitude the amount of the molecule changes). Only correctly filled out rows will be included in the graph.')
            chem_free_grid = st.data_editor(chem_free_df, num_rows="dynamic")
            
            # FREE CHEM: WRITE RESULTS TO A DATAFRAME
            for i,mole in enumerate(chem_free_grid[f'Molecule ({mole_unit})']):
                if mole != None and chem_free_grid['Values'][i] != None and (len(chem_free_grid['Values'][i]) == 1 or chem_free_grid['Pressures (bar)'][i] != None):
                    if mole not in config['chemistry'][chem_method]:
                        config['chemistry'][chem_method][mole] = {'values': [], 'unit': 'v/v', 'pressures': [], 'pressure_unit': 'bar'}
                    values = [float(value) for value in chem_free_grid['Values'][i]]
                    if len(values) == 1:
                        # don't need a pressure point if there's only one value
                        config['chemistry'][chem_method][mole]['value'] = values[0]
                    else:
                        # TODO: add warning to make sure there's the right # of pressures per values specified
                        config['chemistry'][chem_method][mole]['values'] = values
                        config['chemistry'][chem_method][mole]['pressures'] = [float(pressure) for pressure in chem_free_grid['Pressures (bar)'][i]]
            # FREE CHEM: CONFIGURE BACKGROUND GAS
            num_background_gases = st.selectbox("0, 1, or 2 Background Gases?", ('0', '1', '2'), index=None)
            if num_background_gases == '1':
                config['chemistry']['free']['background']['gases']= [st.text_input('Background gas')]
                del config['chemistry']['free']['background']['fraction']
            elif num_background_gases == '2':
                background_gas1 = st.text_input('Background gas 1')
                background_gas2 = st.text_input('Background gas 2')
                config['chemistry']['free']['background']['gases'] = [background_gas1, background_gas2]
                fraction = st.number_input('Fraction between them')
                config['chemistry']['free']['background']['fraction'] = fraction
            elif num_background_gases == '0':
                del config['chemistry']['free']['background']
        else:
            # EDITABLE SECTION FOR OTHER CHEMISTRY METHODS
            editable_section(config['chemistry'][f'{config['chemistry']['method']}'], 'chemistry')

    # GRAPH MIXING RATIOS
    if st.button('See Mixing Ratios'):
        try:
            data_class = run_spectrum_class('chemistry')
            chem_df = data_class.inputs['atmosphere']['profile']
            # form {mixingratios: {'H20': [...], ...}} to pass to jpi.mixing_ratio
            # chem_df.keys() would have [temperature, pressure, H20, CO2, <other example molecules> ]
            for key in chem_df.keys():
                if key != 'pressure' or key != 'temperature':
                    chem_df[key] = chem_df[key]
            full_output = dict({'layer':{'pressure': chem_df['pressure'], 'mixingratios': chem_df}})
            streamlit_bokeh(jpi.mixing_ratio(full_output))
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
        cloud_id = 'cloud1'
        cloud_obj = config['clouds'][cloud_id]

        # set cloud type
        cloud_type = st.selectbox("Cloud type", cloud_obj.keys())
        config['clouds'][f'{cloud_id}_type'] = cloud_type

        # create editable df for cloud so users can set parameters
        cloud_type_df = pd.DataFrame([format_config_section_for_df(cloud_obj[cloud_type])])
        cloud_type_editable_df = st.data_editor(cloud_type_df)
        # render any options sections dynamically
        cloud_list_iterate = copy.deepcopy(config['clouds'][cloud_id][cloud_type])
        for attr in cloud_list_iterate:
            if attr.endswith('_options'):
                pure_attr = '_'.join(attr.split('_')[:-1])
                cloud_obj[cloud_type][pure_attr] = st.selectbox(f"{cloud_id} {pure_attr.capitalize()} Options", cloud_obj[cloud_type][attr], index=None)
                
                # render any kwargs for the options dynamically
                if cloud_obj[cloud_type][pure_attr]:
                    var_with_options = cloud_obj[cloud_type][pure_attr]
                    kwargs_for_attribute_option_exist = cloud_obj[cloud_type][pure_attr]+'_kwargs' in cloud_obj[cloud_type]
                    if kwargs_for_attribute_option_exist:
                        options_editable_df = st.data_editor(cloud_obj[cloud_type][ cloud_obj[cloud_type][pure_attr]+'_kwargs' ])
                        cloud_obj[cloud_type][ cloud_obj[cloud_type][pure_attr]+'_kwargs' ] = options_editable_df

        write_results_to_config(cloud_type_editable_df, config['clouds']['cloud1'][cloud_type])
        ##########################
        # GRAPH CLOUDS
        ##########################
        if st.button('See Clouds'):
            try:
                data_class = run_spectrum_class()
                df = data_class.inputs['clouds']['profile'].astype('float')
                wavenumber = df['wavenumber'].unique()
                nwno = len(wavenumber)
                wavelength = 1e4/wavenumber
                pressure = df['pressure'].unique()
                nlayer = len(pressure)
                bokeh_plot = jpi.plot_cld_input(nwno, nlayer, df=df,pressure=pressure, wavelength=wavelength)
                st.write(bokeh_plot)
            except Exception as e:
                st.warning('Make sure you have configured chemistry and temperature.')
                st.write(e)
    else:
        if 'clouds' in config:
            del config['clouds']

def render_wavelength_range(opacity):
    return st.slider(
        "Select wavelength range (μm)",
        min_value=np.min(1e4/opacity.wno),
        max_value=np.max(1e4/opacity.wno),
        value=(np.min(1e4/opacity.wno), np.max(1e4/opacity.wno))
    )

def render_spectral_resolution():
    return st.number_input('Spectral Resolution', min_value=10, value=150)
# ---------------------------------#
# RUN A SPECTRUM ----------------- #
# ---------------------------------#
def run_spectrum():
    if config['calc_type'] =='spectrum' and st.button(f'Run {config['calc_type']}'):
        try:

            df = go.run(driver_dict=clean_dictionary(config))
            #TODO : thermal could either map to thermal or fpfs_thermal. reflected could either map to albedo or fpfs_reflected
            #spectral key options: fpfs_thermal, fpfs_reflected, transit_depth, albedo, temp_brightness, thermal
            observation_key_mapping = {
                'thermal': 'thermal',
                'reflected': 'albedo',
                'transmission': 'transit_depth'
            }
            observation_key = observation_key_mapping[config['observation_type']]
            wavenumber, albedo_or_fluxes = df['wavenumber'] , df[observation_key]
            wavenumber, albedo_or_fluxes = jdi.mean_regrid(wavenumber, albedo_or_fluxes, R=spectral_resolution)
            spec_fig = jpi.spectrum(wavenumber, albedo_or_fluxes, plot_width=500, x_range=wavelength_range)
            # graph spectrum
            streamlit_bokeh(spec_fig, theme="streamlit", key="spectrum")
        except Exception as e:
            st.warning('Make sure you have configured temperature, pressure, and chemistry before running a spectrum.')
            st.write(e)
        st.divider()

def render_free_parameter_selection():
    parameter_handler = {}
    st.subheader("Select which available free parameters you'd like to do a retrieval on:")
    config['temperature'] = {
        config['temperature']['profile']: config['temperature'][config['temperature']['profile']],
        'pressure': config['temperature']['pressure'],
        'profile': config['temperature']['profile']
    }
    config['chemistry'] = {
        config['chemistry']['method']: config['chemistry'][config['chemistry']['method']],
        'method': config['chemistry']['method']
    }
    if 'clouds' in config:
        config['clouds'] = {
            'cloud1':{config['clouds']['cloud1_type']: config['clouds']['cloud1'][config['clouds']['cloud1_type']]},
            'cloud1_type': config['clouds']['cloud1_type']
        }
    del config['retrieval']
    del config['sampler']
    def list_available_free_parameters(data, current_path=""):
        for key, value in data.items():
            new_path = f"{current_path}.{key}" if current_path else key

            if isinstance(value, dict):
                list_available_free_parameters(value, new_path)
            elif isinstance(value, float):
                parameter_handler[new_path] = [st.checkbox(f"{new_path} {value}"), value]
            elif isinstance(value, int) and not isinstance(value, bool) and key != 'nlevel':
                parameter_handler[new_path] = [st.checkbox(f"{new_path} {value}"), value]
            elif isinstance(value, np.int64):
                parameter_handler[new_path] = [st.checkbox(f"{new_path} {value}"), value]
            elif isinstance(value, list) and all(isinstance(item, (int, float)) for item in value):
                for index, item in enumerate(value):
                    parameter_handler[new_path + f'.{index}'] = [st.checkbox(f"{new_path + f'.{index}'} {item}"), item]
    list_available_free_parameters(config)
    return parameter_handler
def render_ranges_for_selected_parameters(parameter_handler):
    # filter for what items have been selected
    prior_set_items = {}
    selected_items = {path_to_parameter: state_value_list[1] for path_to_parameter, state_value_list in parameter_handler.items() if state_value_list[0]}

    # Min, Max, Log, Prior Type Listing
    # Right Now not swapping out Gaussian Kwargs for Uniform Kwargs...
    for i, (key, value) in enumerate(selected_items.items()):
        st.subheader(key)
        prior_type = st.selectbox('prior', ['uniform', 'gaussian'], key=f'prior{i}')
        prior_set_items[key] = dict(
            log=st.text_input('log', False, key=f'log{i}'),
            prior=prior_type
        )
        # if value == 0:
        #     value = 0.00001
        if prior_type == 'uniform':
            prior_set_items[key][f'{prior_type}_kwargs'] =dict(
                min=st.number_input('min', value=value*0.75, min_value=None, max_value=None, key=f'min{i}', format="%.6f"),
                max=st.number_input('max', value=value*1.25, min_value=None, max_value=None, key=f'max{i}', format="%.6f"),
            )
        else:
            prior_set_items[key][f'{prior_type}_kwargs'] =dict(
                mean=st.number_input('mean', value, key=f'mean{i}'),
                std=st.number_input('std', 1, key=f'std{i}'),
            )
    return prior_set_items

def sampler(prior_set_items, nsamples):
    st.write(prior_set_items)
    ALL_TOMLS = []
    save_all_class_pt = []
    for _ in range(nsamples):
        # get samples for values
        check_all_values = go.hypercube(np.random.rand(len(prior_set_items.keys())), dict(prior_set_items))
        # create a new copy of the config to write to
        GUESS_TOML = copy.deepcopy(config)
        # write sampled values to config
        for index, free_parameter in enumerate(prior_set_items.keys()):
            sampled_value = check_all_values[index]
            keys = free_parameter.split('.')
            update_toml_with_a_value_for_a_free_parameter(GUESS_TOML, keys, sampled_value)
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

def sample_plots(ALL_TOMLS, save_all_class_pt, nsamples,run_clouds=True, run_spectrum=True):
    ################################
    # MIXING RATIO GRAPH 
    ################################
    mixing_ratio_kwargs = {}
    mixing_ratio_kwargs['y_axis_label'] = mixing_ratio_kwargs.get('y_axis_label','Pressure(Bars)')
    mixing_ratio_kwargs['x_axis_label'] = mixing_ratio_kwargs.get('x_axis_label','Mixing Ratio(v/v)')
    mixing_ratio_kwargs['y_axis_type'] = mixing_ratio_kwargs.get('y_axis_type','log')
    mixing_ratio_kwargs['x_axis_type'] = mixing_ratio_kwargs.get('x_axis_type','log') 
    mixing_ratio_bokeh_fig = figure(**mixing_ratio_kwargs)
    molecules = save_all_class_pt[0]['molecules']
    cols = pals.magma(min([len(molecules),MOLECULES_LIMIT]))
    legend_it=[]

    moles = {mol:[] for mol in molecules}
    pressure_temperature_fig, axes = plt.subplots(figsize=(15, 5))
    clouds_fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    for i in range(nsamples):
        pressure = save_all_class_pt[i]['pressure']
        temperature = save_all_class_pt[i]['temperature']
        mixingratios = save_all_class_pt[i]['mixingratios']
        axes.semilogy(temperature,pressure, color='red', alpha=0.1)
        if run_clouds:
            cloud_df = save_all_class_pt[i]['cloudprofile']
            cloud_pressure = cloud_df['pressure']
            wavenumber = cloud_df['wavenumber'].unique()

            nwno = len(wavenumber)
            cloud_pressure = cloud_df['pressure'].unique()
            nlayer = len(cloud_df['pressure'].unique())

            w0 = np.reshape(cloud_df['w0'].values,(nlayer,nwno))
            opd = np.reshape(cloud_df['opd'].values,(nlayer,nwno)) + 1e-60
            g0 = np.reshape(cloud_df['g0'].values,(nlayer,nwno))

            ssa1d = np.mean(w0,axis=1) # ssa [nlayer, nwavelength]
            g01d = np.mean(g0,axis=1)
            opd1d = np.mean(opd,axis=1)

            ax1.semilogy(ssa1d, cloud_pressure)
            ax1.invert_yaxis()
            ax1.set_title("Single scattering albedo vs Pressure")
            ax2.semilogy(g01d, cloud_pressure)
            ax2.invert_yaxis()
            ax2.set_title("Asymmetry vs Pressure")
            ax3.loglog(opd1d, cloud_pressure)
            ax3.set_title("Optical Depth vs Pressure")
            ax3.invert_yaxis()
        for mol, c in zip(molecules, cols):
            f = mixing_ratio_bokeh_fig.line(mixingratios[mol],pressure, color=c, line_width=2,
                muted_color=c, muted_alpha=0.05, line_alpha=1)
            moles[mol].append(f)
    for mol in moles.keys():
        legend_it.append((mol, moles[mol]))
    legend = Legend(items=legend_it, location=(0, -20))
    legend.click_policy="mute"
    mixing_ratio_bokeh_fig.add_layout(legend, 'left')
    mixing_ratio_bokeh_fig.y_range.flipped = True
    # TODO: nice to make units accurate to what in driver.toml
    axes.set_xlabel("Temperature (K)") 
    axes.set_ylabel("Log Pressure(Bars)")
    axes.set_title(f"Pressure-Temperature Profiles ({nsamples} Samples)")
    axes.invert_yaxis()
    axes.set_yscale('log')

    spectrum_fig = None
    if run_spectrum:
        WNO_LIST = []
        ALB_LIST = []
        for prior_toml in ALL_TOMLS:
            df = go.run(driver_dict=clean_dictionary(prior_toml))
            if prior_toml['observation_type'] == 'transmission':
                wnos, transit_depth = jdi.mean_regrid(df['wavenumber'],
                                                df['transit_depth'], R=spectral_resolution)
                WNO_LIST.append(wnos)
                ALB_LIST.append(transit_depth)
            else:
                obs_key = 'thermal' if prior_toml['observation_type'] == 'thermal' else 'albedo'
                wno, alb = df['wavenumber'] , df[obs_key]
                wno, alb = jdi.mean_regrid(wno, alb, R=spectral_resolution)
                WNO_LIST.append(wno)
                ALB_LIST.append(alb)

        spectrum_fig = jpi.spectrum(WNO_LIST, ALB_LIST, palette=[(255,0,0,0.3)], plot_width=500,x_range=wavelength_range)
    return pressure_temperature_fig, mixing_ratio_bokeh_fig, clouds_fig, spectrum_fig

def render_retrievals():
    # LIST OUT ALL FREE PARAMETERS
    parameter_handler = render_free_parameter_selection()

    prior_set_items = {}
    retrieval_stage_state_manager = {} # for streamlit rendering organization

    # WHEN USER IS DONE SELECTING, RENDER ALL RANGES FOR SELECTED PARAMETERS
    retrieval_stage_state_manager['done_selecting_parameters'] =  st.selectbox("Done Selecting Methods", ("Yes", "No"), index=None)
    if retrieval_stage_state_manager['done_selecting_parameters'] == 'Yes':
        prior_set_items = render_ranges_for_selected_parameters(parameter_handler)
    st.divider()


    ALL_TOMLS = []
    save_all_class_pt = []
    nsamples = st.number_input('Number of samples?', 5)
    retrieval_object = {}

    # extract data to be able to write to toml to recreate
    for parameter in prior_set_items.keys():
        base = prior_set_items[parameter]
        prior_type = base['prior']
        retrieval_variables = {
            'prior' : prior_type,
            'log' : base['log'],
        }
        for kwarg in base[f'{prior_type}_kwargs'].keys():
            retrieval_variables[kwarg] = base[f'{prior_type}_kwargs'][kwarg]

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
        pressure_temperature_fig, mixing_ratio_bokeh_fig, clouds_fig, _ = sample_plots(ALL_TOMLS, save_all_class_pt, nsamples, run_spectrum=False, run_clouds=('clouds' in config))

        # PLOT PT, MR, CLOUDS
        st.pyplot(pressure_temperature_fig)
        streamlit_bokeh(mixing_ratio_bokeh_fig)
        if 'clouds' in config:
            st.pyplot(clouds_fig)

        st.divider()

        # PLOT SPECTRUM
        retrieval_stage_state_manager['see_prior_spectrums'] =  st.selectbox("See Spectrums for Priors?", ("Yes", "No"), index=None)
        if retrieval_stage_state_manager['see_prior_spectrums'] == 'Yes':
            _, _, _, spectrum_fig = sample_plots(ALL_TOMLS, save_all_class_pt, nsamples, run_clouds=('clouds' in config))
            streamlit_bokeh(spectrum_fig)
    config['InputOutput']['observation_data'] = st.text_input("Enter in the datapath(s) to your observation data for retrievals", value = config.get('InputOutput').get('observation_data', ['']))
    return retrieval_object

def render_download_config(retrieval_object):
    cleaned_config = clean_dictionary(config)
    if 'retrieval' in cleaned_config:
        del cleaned_config['retrieval']
    # TODO: change writing reitreval stuff
    if retrieval_object != {}:
        cleaned_config['retrieval'] = retrieval_object

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
if config is None: st.error('Cannot find driver.toml file')
opacity, param_tools = render_admin()
if config['observation_type']:
    render_star()
    render_object()
    render_phase_angle()

    # ATMOSPHERIC VARIABLES
    st.subheader("Atmospheric Variables")
    render_pressure_and_temperature()
    render_chemistry()
    render_clouds()


    # SPECTRUM
    wavelength_range = render_wavelength_range(opacity)
    spectral_resolution = render_spectral_resolution()
    run_spectrum()
        
    # RETRIEVALS
    retrieval_object = {}
    st.header("Retrievals")
    if st.selectbox("Do you want to do a retrieval?", ('Yes', 'No'), index=None) == 'Yes':
        retrieval_object = render_retrievals()

    render_download_config(retrieval_object)