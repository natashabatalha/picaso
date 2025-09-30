##################################################
# BEFORE YOU GO ON ------------------------------#
# you must set these paths with your own paths !!!
##################################################
# run UI locally with: streamlit run driver_ui.py

path_to_opacity_DB = '/Users/sjanson/Desktop/code/picaso/reference/opacities/opacities.db'
path_to_virga_mieff = '/Users/sjanson/Desktop/code/picaso/reference/virga/'

# if you have to manually specify paths for env vars, change below; else comment out the os.environ commands below
picaso_refdata_env_var = "/Users/sjanson/Desktop/code/picaso/reference"
pysyn_cdbs_env_var = '/Users/sjanson/Desktop/code/picaso/reference/grp/redcat/trds'

# --- TODO SECTION ----------------- #
# TODO change all paths to os dynamic finds instead of hardcoded paths
# add validation & options for users to find necessary data /specify datapaths
# TODO options should be what files users have in their directories and or rendered off of driver.toml
# add other unit options & validation
# TODO add docstrings to all functions
# TODO handle duplicates if dup pressure/chem from a userfile
# TODO remove index from editable df or minimize amap
# TODO --> assumes options for calc and observation
#---IMPORTS--------------------------------#
import pandas as pd
import os
import toml
import tomllib 
import json 

os.environ['picaso_refdata'] = picaso_refdata_env_var
os.environ['PYSYN_CDBS'] = pysyn_cdbs_env_var

import picaso.driver as go
from picaso import justdoit as jdi 
from picaso import justplotit as jpi
from picaso.parameterizations import Parameterize

import streamlit as st
from streamlit_bokeh import streamlit_bokeh

#---INITIALIZE VARIABLES--------------------------#
observation_type = None

config = None
driver_config = '/Users/sjanson/Desktop/code/picaso/reference/input_tomls/driver.toml'
if isinstance(driver_config, str):
    with open(driver_config, "rb") as f:
        config = tomllib.load(f)
else:
    st.error('Cannot find driver.toml file')

param_tools = None 
opacity = None 

# -- HELPER FUNCTIONS -------------------------- #
def run_spectrum_class(stage=None):
    """
    Runs driver.py's spectrum class as far as the level specified in stage with current configuration

    Parameters
    ----------
    stage : string
        Options are planet, star, temperature, chemistry; if left blank, the whole class will run (including clouds)
    
    Returns
    -------
    picaso.justdoit.inputs
        Configured class
    """
    return go.setup_spectrum_class(config, opacity, param_tools, stage)

def format_config_section_for_df(obj):
    pass_to_df = {}
    for attr in obj.keys():
        if not f'{attr}_options' in obj and not attr.endswith('_options'):
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
    for item in grid:
        if ' (' in item:
            key, unit = item.split()
            if key.lower() in base and grid[item][0]:
                base[key.lower()]['value'] = grid[item][0]
        elif isinstance(base[item], list):
            base[item] = [float(ele) for ele in grid[item][0]]
        elif not isinstance(base[item], dict):
            # ignore dictionaries that we didn't alter
            base[item] = grid[item][0]

# ---------------------------------------------- #
# -- BEGINNING OF APP -------------------------- #
# ---------------------------------------------- #
# st.set_page_config(layout="wide")

# HEADERS -----------------------#
st.logo('https://natashabatalha.github.io/picaso/_images/logo.png', size="large", link="https://github.com/natashabatalha/picaso")
st.header('Try out PICASO functionalities',divider='rainbow')

st.info('This page assumes you have your data and environment variables set up. If this is not the case, please navigate to the Setup Data page.')
st.subheader('Administrative')

# ADMINISTRATIVE OPTIONS --------------#
# TODO: handle this section (database options?)
database_method = None # = st.selectbox("Database method",("phoenix", "ck04models"),)
config['OpticalProperties']['opacity_method'] = st.selectbox("Opacity method", ("resampled")) #, "preweighted", "resortrebin"))
# if opacity_method == 'resampled': st.warning('Warning, you are selecting resampling! This could degrade the precision of your spectral calculations so should be used with caution. If you are unsure check out this tutorial: https://natashabatalha.github.io/picaso/notebooks/10_ResamplingOpacities.html',
#     icon='⚠️')
opacity = jdi.opannection(
    filename_db=config['OpticalProperties']['opacity_files'], #database(s)
    method=config['OpticalProperties']['opacity_method'], #resampled, preweighted, resortrebin
    **config['OpticalProperties']['opacity_kwargs'] #additonal inputs 
)
param_tools = Parameterize(load_cld_optical=go.find_values_for_key(config ,'condensate'), mieff_dir=config['OpticalProperties'].get('virga_mieff', None))



st.subheader('Select calculation to perform')
config['calc_type'] = st.selectbox("Calculation type", config['calc_type_options'], index=None)
if config['calc_type'] == "spectrum":
    observation_type = st.selectbox("Observation type", config['observation_type_options'], index=None)
elif config['calc_type'] != None:
    st.warning(f'The {config['calc_type']} option has not been implemented yet.')

# configure heading
if observation_type:
    st.divider()
if observation_type == 'reflected':
    st.header('Reflected Exoplanet Spectrum Config')
elif observation_type == 'thermal':
    st.header('Brown Dwarf Thermal Emission Spectrum Config')
    choice = st.selectbox("Do your want your object to be irradiated?", ('Yes', 'No'), index=None)
    config['irradiated'] = choice == 'Yes'

# --- CONFIGURE STAR ------------------- #
# TODO: pull dynamically from config
if observation_type == 'reflected' or observation_type == 'transmission' or config['irradiated']:
    st.subheader("Star Variables")
    star_df = pd.DataFrame({
        'Radius': [1],
        'R Unit': 'Rsun',
        'Semi_Major': '200',
        'Unit': 'AU',
        'Temperature': [5400],
        'Metallicity': [0.01],
        'Logg': [4.45],
        # type
    })
    star_grid = st.data_editor(star_df)

    # updating config file
    for key in star_grid:
        if key.lower() in config['object'] and star_grid[key][0]:
            config['star'][key.lower()]['value'] = star_grid[key][0]

if observation_type:
    # CONFIGURE PLANET/BD ---------------------- #
    st.subheader("Object Variables")

    formatted_obj = format_config_section_for_df(config['object'])
    object_df = pd.DataFrame([formatted_obj])
    object_grid = st.data_editor(object_df)
    # TODO: users should be able to leave gravity or M/R blank
    write_results_to_config(object_grid, config['object'])


    # TODO switch to degrees/ give flexibility between radians and degrees
    config['geometry']['phase']['value'] = st.number_input('Enter phase angle in radians 0-2π', min_value=0, max_value=6, value=0)

    # ATMOSPHERIC VARIABLES ---------------------- #
    st.subheader("Atmospheric Variables")

    # PRESSURE
    st.text('Configure pressure (can ignore if using a userfile for temperature)')
    formatted_obj = format_config_section_for_df(config['temperature']['pressure'])
    pressure_df = pd.DataFrame([formatted_obj])
    pressure_grid = st.data_editor(pressure_df)

    write_results_to_config(pressure_grid, config['temperature']['pressure'])

    # TEMPERATURE
    # TODO: hide until pressure is configured...unless userfile...
    temperature_options = ['isothermal'] if not 'options' in config['temperature'] else [option for option in config['temperature']['options'] if option in config['temperature']]
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

    # see PT graph
    if st.button('See Pressure-Temperature graph'):
        st.write(config['temperature'][temp_profile])
        data_class = run_spectrum_class('temperature')
        streamlit_bokeh(jpi.pt({'layer': data_class.inputs['atmosphere']['profile']}))

    #------modelling chemistry ----------------------#
    chemistry_options = ['free'] if 'options' not in config['chemistry'] else [option for option in config['chemistry']['options'] if option in config['chemistry']]
    config['chemistry']['method'] = st.selectbox(
        "How to model chemistry", chemistry_options, index=None
    )

    # TODO num_rows="dynamic" for free & those can be empty w/o problem
    chem_method = config['chemistry']['method']
    if chem_method:
        if 'free' in chem_method:
            molecules = [mole for mole in config['chemistry'][chem_method] if mole != 'background']
            mole_unit = config['chemistry'][chem_method][molecules[0]]['unit']
            
            # pressure_unit = config['']
            # for now must specify free.. or need some sort of pattern since the moleculues need special rendering
            # assuming all moleculues have the same unit
            df = pd.DataFrame({
                f'Molecule ({mole_unit})': molecules,
                'Values': [config['chemistry'][chem_method][mole]['value'] for mole in molecules],
                'Pressures': [['0.1'], ['1e-2'], ['0.1']],
            })
        else:
            formatted_obj = format_config_section_for_df(config['chemistry'][f'{config['chemistry']['method']}'])
            chem_df = pd.DataFrame([formatted_obj])
            chem_grid = st.data_editor(chem_df)


    # if config['chemistry']['method'] == 'free':
    #     gridcol, buttoncol = st.columns([6,1])
    #     with gridcol:
    #         df = pd.DataFrame({
    #             'Molecule': ['H2O', 'CH4', 'NH3'],
    #             'Values': [['1e-3'],['4e-4', '1e-3'],['1e-5']],
    #             'Unit': ['v/v','v/v','v/v'],
    #             'Pressures': [['0.1'], ['1e-2'], ['0.1']],
    #             'Pressure Unit': ['bar', 'bar', 'bar']
    #         })

    #         st.info('Molecules are case sensitive (ex: TiO, H2O)')
    #         if 'chem_free_df' not in st.session_state:
    #             st.session_state.chem_free_df = pd.DataFrame(df)
    #         grid = st.data_editor(st.session_state.chem_free_df)

    #         # updating config file
    #         for i,mole in enumerate(grid['Molecule']):
    #             if mole not in config['chemistry']['free']:
    #                 config['chemistry']['free'][mole] = {'values': [], 'unit': 'v/v', 'pressures': [], 'pressure_unit': 'bar'}
    #             values = [float(value) for value in grid['Values'][i]]
    #             if len(values) == 1:
    #                 # don't need a pressure point if there's only one value
    #                 config['chemistry']['free'][mole]['value'] = values[0]
    #             else:
    #                 config['chemistry']['free'][mole]['values'] = values
    #                 config['chemistry']['free'][mole]['pressures'] = [float(pressure) for pressure in grid['Pressures'][i]]

    #     with buttoncol:
    #         # code for figuring out how to add new rows heavily inspired by @ferdy here: https://discuss.streamlit.io/t/how-to-add-delete-a-new-row-in-data-editor-in-streamlit/70608
    #         if st.button("Add Row"):
    #             new_row_df = pd.DataFrame([{'Molecule': '', 'Values': [], 'Unit': 'v/v', 'Pressures': [], 'Pressure Unit': 'bar'}])
    #             st.session_state.chem_free_df = pd.concat([st.session_state.chem_free_df, new_row_df], ignore_index=True)
    #             st.rerun()

    if st.button('See Mixing Ratios'):
        # TODO: some sort of warning that temp/object should be configured before this
        data_class = run_spectrum_class('chemistry')
        chem_df = data_class.inputs['atmosphere']['profile']

        # form {mixingratios: {'H20': [...], ...}} to pass to jpi.mixing_ratio
        # chem_df.keys() would have [temperature, pressure, H20, CO2, <other example molecules> ]
        for key in chem_df.keys():
            if key != 'pressure' or key != 'temperature':
                chem_df[key] = chem_df[key]
        full_output = dict({'layer':{'pressure': chem_df['pressure'], 'mixingratios': chem_df}})
        streamlit_bokeh(jpi.mixing_ratio(full_output))


# ---------------------------------#
# RUN A SPECTRUM ----------------- #
# ---------------------------------#

# TODO fix transmission and clean up this section...
if config['calc_type'] =='spectrum' and st.button(f'Run {config['calc_type']}'):
    config['irradiated'] = config['irradiated'] or observation_type == 'reflected' or observation_type == 'transmission'
    df = go.run(driver_dict=config)
    obs_key = 'thermal' if observation_type == 'thermal' else 'albedo'
    wno, alb, fpfs, full = df['wavenumber'] , df[obs_key] , df[f'fpfs_{observation_type}'], df['full_output']
    wno, alb = jdi.mean_regrid(wno, alb , R=150)
    spec_fig = jpi.spectrum(wno, alb, plot_width=500,x_range=[0.3,1])
    streamlit_bokeh(jpi.mixing_ratio(full,plot_width=500))
    streamlit_bokeh(spec_fig, theme="streamlit", key="spectrum")
    # streamlit_bokeh(jpi.pt(full), theme="streamlit", key="pt")
    # streamlit_bokeh(jpi.cloud(full), theme="streamlit", key="cloud")
    # TODO: could add w/o molecule functionality & photon attenuation depth, heatmap 


# TODO: figure out better place to put this
# TODO: test downloaded toml file more
st.download_button(
    label="Download current config",
    data=toml.dumps(config),
    file_name="configured_toml.toml",
    mime="application/toml"
)