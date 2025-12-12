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
# TODO --> chemistry userfile doesn't work
# TODO --> don't hardcode star information
# TODO --> keep graphs in place
# TODO --> render retrieval keywords provided in driver.toml
# condenstaes & pressure grid
# np.flip(np.reshape[dat01['w0]]) in plot_cld_inputs()
# clouds jdi.available() from virga autopopulate
# check for decay type & distribution when rendering kwargs
# ^ use x_options instead
# add option for retrieval after spectrum
# visualization of priors
# for rendering the retrievals do floats for lists 
# have multiple min and maxes for each item of the list


#---IMPORTS--------------------------------#
import pandas as pd
import os
import toml
import tomllib 
import numpy as np 
from scipy import stats
import copy 

os.environ['picaso_refdata'] = picaso_refdata_env_var
os.environ['PYSYN_CDBS'] = pysyn_cdbs_env_var

import picaso.driver as go
from picaso import justdoit as jdi 
from picaso import justplotit as jpi
from picaso.parameterizations import Parameterize
from bokeh.plotting import figure, show
import matplotlib.pyplot as plt
import bokeh.palettes as pals
from bokeh.models import Legend

import streamlit as st
from streamlit_bokeh import streamlit_bokeh
# st.set_page_config(layout="wide")

#---INITIALIZE VARIABLES--------------------------#

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


def clean_dictionary(data, key_to_remove):
    if isinstance(data, dict):
        keys_to_check = list(data.keys())
        for key in keys_to_check:
                if key.endswith(key_to_remove):
                    del data[key]
                else:
                    data[key] = clean_dictionary(data[key], key_to_remove)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            data[i] = clean_dictionary(item, key_to_remove)
    return data

def write_to_guess_toml(dictionary, keys, value):
    stopIndex = -1
    if keys[-1].isdigit():
        stopIndex = -2
    for key in keys[:stopIndex]:
        dictionary = dictionary[key]

    if keys[-1].isdigit(): # to handle lists
        dictionary[keys[-2]][int(keys[-1])] = value
    else:
        dictionary[keys[-1]] = value

# which of Natasha's files did I pull this from just import from there
def hypercube(u, fitpars):
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
        else:
            raise Exception('Prior type not available')
        if fitpars[key]['log'] == True or fitpars[key]['log'] == 'True':
            x[i]=10**x[i]  
    return x

# ---------------------------------------------- #
# -- BEGINNING OF APP -------------------------- #
# ---------------------------------------------- #

# HEADERS -----------------------#
st.logo('https://natashabatalha.github.io/picaso/_images/logo.png', size="large", link="https://github.com/natashabatalha/picaso")
st.header('Run PICASO',divider='rainbow')

# st.info('This page assumes you have your data and environment variables set up. If this is not the case, please navigate to the Setup Data page.')
st.subheader('Administrative')
################################
#
# CONFIGURE ADMINISTRATIVE STUFF
#
################################
# ## TODO: handle this section (database options?)
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
config['calc_type'] = st.selectbox("Calculation type", ['spectrum','climate'], index=None)
if config['calc_type'] == "spectrum":
    config['observation_type'] = st.selectbox("Observation type", config['observation_type_options'], index=None)
elif config['calc_type'] != None:
    st.warning(f'The {config['calc_type']} option has not been implemented yet.')

# configure heading
if config['observation_type']:
    st.divider()
    st.header(f'{config['observation_type'].capitalize()} Spectrum Config')

if config['observation_type'] == 'thermal':
    choice = st.selectbox("Do your want your object to be irradiated?", ('Yes', 'No'), index=None)
    config['irradiated'] = choice == 'Yes'

###############################
#
# CONFIGURE STAR
#
################################
# TODO: pull dynamically from config
if config['observation_type'] == 'reflected' or config['observation_type'] == 'transmission' or config['irradiated']:
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

if config['observation_type']:
    ###############################
    #
    # CONFIGURE OBJECT
    #
    ################################
    st.subheader("Object Variables")

    formatted_obj = format_config_section_for_df(config['object'])
    object_df = pd.DataFrame([formatted_obj])
    object_grid = st.data_editor(object_df)
    # TODO: users should be able to leave gravity or M/R blank
    write_results_to_config(object_grid, config['object'])

    # TODO ---> REMOVE?
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
    # TODO: hide until pressure is configured...unless userfile or sonora_bobcat
    # TODO: remove most options
    # move pressure to after

    ###############################
    #
    # INPUT TEMPERATURE INFORMATION
    #
    ################################
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

    ########################
    # GRAPH PT
    ########################
    if st.button('See Pressure-Temperature graph'):
        data_class = run_spectrum_class('temperature')
        streamlit_bokeh(jpi.pt({'layer': data_class.inputs['atmosphere']['profile']}))

    #############################
    #
    # INPUT CHEMISTRY INFORMATION
    #
    #############################
    chemistry_options = [option for option in config['chemistry'] if option != 'method']
    if len(chemistry_options) == 0:
        st.warning('No chemistry option found in driver.toml.')
    config['chemistry']['method'] = st.selectbox(
        "How to model chemistry", chemistry_options, index=None
    )
    
    chem_method = config['chemistry']['method']
    if chem_method:
        if 'free' in chem_method:
            molecules = [mole for mole in config['chemistry'][chem_method] if mole != 'background']
            mole_unit = config['chemistry'][chem_method][molecules[0]]['unit']
            # for now must specify free.. or need some sort of pattern since the moleculues need special rendering
            # assuming all moleculues have the same unit

            molecule_values = []
            for mole in molecules:
                if 'values' in config['chemistry'][chem_method][mole]:
                    molecule_values.append([str(ele) for ele in config['chemistry'][chem_method][mole]['values']])
                elif 'value' in config['chemistry'][chem_method][mole]:
                    molecule_values.append([str(config['chemistry'][chem_method][mole]['value'])])
            chem_free_df = pd.DataFrame({
                f'Molecule ({mole_unit})': molecules,
                'Values': molecule_values,
                'Pressures (bar)': [[str(ele) for ele in config['chemistry'][chem_method][mole].get('pressures', '')] for mole in molecules],
            })
            st.info('Molecule names are case sensitive (ex: TiO, H2O). You only need to specify a pressure if you provide multiple values for a molecule (to indicate what altitude the amount of the molecule changes). Only correctly filled out rows will be included in the graph.')
            chem_free_grid = st.data_editor(chem_free_df, num_rows="dynamic")

            # writing to grid
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

        else:
            formatted_obj = format_config_section_for_df(config['chemistry'][f'{config['chemistry']['method']}'])
            chem_df = pd.DataFrame([formatted_obj])
            chem_grid = st.data_editor(chem_df)
            # TODO --> do we not write to grid here...

    ##########################
    # GRAPH MIXING RATIOS
    ##########################

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

##########################
#
# INPUT CLOUD INFORMATION
#
##########################
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
    import copy
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
    # config['clouds']['cloud1'][cloud_type] = dict(p=1,dp=1,w0=1,g0=1,opd=1)
    # st.write(config['clouds']['cloud1'][cloud_type])
    ##########################
    # GRAPH CLOUDS
    ##########################
    if st.button('See Clouds'):
        config = clean_dictionary(config, '_options')
        data_class = run_spectrum_class()
        df = data_class.inputs['clouds']['profile'].astype('float')
        wavenumber = df['wavenumber'].unique()
        nwno = len(wavenumber)
        wavelength = 1e4/wavenumber
        pressure = df['pressure'].unique()
        nlayer = len(pressure)
        bokeh_plot = jpi.plot_cld_input(nwno, nlayer, df=df,pressure=pressure, wavelength=wavelength)
        st.write(bokeh_plot)

# ---------------------------------#
# RUN A SPECTRUM ----------------- #
# ---------------------------------#
x_range = [0,15]
spectral_resolution = 150
if config['calc_type'] =='spectrum' and st.button(f'Run {config['calc_type']}'):
    config['irradiated'] = config['irradiated'] or config['observation_type'] == 'reflected' or config['observation_type'] == 'transmission'
    cleaned = clean_dictionary(config, '_options')
    # st.write(config['clouds']['cloud1'])
    df = go.run(driver_dict=cleaned)

    # check numpy 3.11 elijah commit to fix below
    # use mapping dictionary to make this cleaner
    # investigate other graph visuals
    if config['observation_type'] == 'transmission':
        # TODO --> why is transit_depth none...
        wnos, transit_depth = jdi.mean_regrid(df['wavenumber'],
                                        df['transit_depth'], R=spectral_resolution)
        fig = jpi.spectrum(wnos, transit_depth,
                            plot_width=800, y_axis_label='Relative (Rp/Rs)^2',
                            x_range=x_range)
        streamlit_bokeh(fig, theme="streamlit", key="transmission_spectrum")
    else:
        obs_key = 'thermal' if config['observation_type'] == 'thermal' else 'albedo'
        wno, alb, fpfs, full = df['wavenumber'] , df[obs_key] , df[f'fpfs_{config['observation_type']}'], df['full_output']
        wno, alb = jdi.mean_regrid(wno, alb, R=spectral_resolution)
        spec_fig = jpi.spectrum(wno, alb, plot_width=500,x_range=x_range)
        streamlit_bokeh(spec_fig, theme="streamlit", key="spectrum")

# ---------------------------------#
# RETRIEVALS     ----------------- #
# ---------------------------------#
st.divider()
st.header("Retrievals")
do_retrieval = st.selectbox("Do you want to do a retrieval?", ('Yes', 'No'), index=None)
parameter_handler={}
if do_retrieval:
    st.subheader("Select which available free parameters you'd like to do a retrieval on:")
    def list_available_free_parameters(data, current_path=""):
        for key, value in data.items():
            new_path = f"{current_path}.{key}" if current_path else key

            if isinstance(value, dict):
                list_available_free_parameters(value, new_path)
            elif isinstance(value, float):
                parameter_handler[new_path] = [st.checkbox(f"{new_path} {value}"), value]
            elif isinstance(value, int) and not isinstance(value, bool):
                parameter_handler[new_path] = [st.checkbox(f"{new_path} {value}"), value]
            elif isinstance(value, np.int64):
                parameter_handler[new_path] = [st.checkbox(f"{new_path} {value}"), value]

    config['temperature'] = {
        config['temperature']['profile']: config['temperature'][config['temperature']['profile']],
        'pressure': config['temperature']['pressure'],
        'profile': config['temperature']['profile']
    }
    config['chemistry'] = {
        config['chemistry']['method']: config['chemistry'][config['chemistry']['method']],
        'method': config['chemistry']['method']
    }
    config['clouds'] = {
        'cloud1':{config['clouds']['cloud1_type']: config['clouds']['cloud1'][config['clouds']['cloud1_type']]},
        'cloud1_type': config['clouds']['cloud1_type']
    }
    del config['retrieval']
    del config['sampler']
    list_available_free_parameters(config)

selected_items = {}

prior_set_items = {}

done_selecting_parameters = None
new_config = {}
new_config['done_selecting_parameters'] =  st.selectbox("Done Selecting Methods", ("Yes", "No"), index=None)

if new_config['done_selecting_parameters'] == 'Yes':

    # filter for what items have been selected
    # this can be turned into a one line comprehension
    for key,value in parameter_handler.items():
        # if checkbox is true
        if value[0] == True:
            # save the value in selected items
            selected_items[key] = value[1] 

    # Min, Max, Log, Prior Type Listing
    # Right Now not swapping out Gaussian Kwargs for Uniform Kwargs...
    for i, (key, value) in enumerate(selected_items.items()):
        st.subheader(key)
        prior_set_items[key] = dict(
            log=st.text_input('log', False, key=f'log{i}'),
            prior=st.selectbox('prior', ['uniform', 'gaussian'], key=f'prior{i}'),
            uniform_kwargs=dict(
                min=st.number_input('min', value*0.75, key=f'min{i}'),
                max=st.number_input('max', value*1.25, key=f'max{i}'),
            )    
        )

st.divider()
        
new_config['done_configuring_priors'] =  st.selectbox("Done Configuring Priors", ("Yes", "No"), index=None)
ALL_TOMLS = []

if new_config['done_configuring_priors'] == 'Yes':
    nsamples = st.number_input('Number of samples?', 5)

    prior_set_items_pure_dict = {}
    for i, (key, value) in enumerate(selected_items.items()):
        # Access the current value stored in session state by its unique key
        prior_set_items_pure_dict[key] = dict(
            log=st.session_state[f'log{i}'],
            prior=st.session_state[f'prior{i}'],
            uniform_kwargs=dict(
                min=st.session_state[f'min{i}'],
                max=st.session_state[f'max{i}'],
            )
        )
    save_all_class_pt = []
    for i in range(nsamples):
        check_all_values = hypercube(np.random.rand(len(prior_set_items_pure_dict.keys())), dict(prior_set_items_pure_dict))
        GUESS_TOML = copy.deepcopy(config)

        for index, free_parameter in enumerate(prior_set_items_pure_dict.keys()):
            sampled_value = check_all_values[index]
            keys = free_parameter.split('.')
            write_to_guess_toml(GUESS_TOML, keys, sampled_value)

        ALL_TOMLS.append(GUESS_TOML)
        # RUNNING THROUGH SPECTRUM CLASS
        data_class = go.setup_spectrum_class(clean_dictionary(GUESS_TOML, '_options'), opacity, param_tools)
        # GETTING INFO
        t = data_class.inputs['atmosphere']['profile']['temperature']
        p = data_class.inputs['atmosphere']['profile']['pressure']
        cloud_profile = data_class.inputs['clouds']['profile']
        mixingratios = data_class.inputs['atmosphere']['profile']
        for key in mixingratios.keys():
            if key != 'pressure' or key != 'temperature':
                mixingratios[key] = mixingratios[key]
        limit = 50
        molecules = [mol for mol in mixingratios.keys() if mol not in ['pressure', 'temperature', 'kz']][:limit]
        save_all_class_pt.append({
            'temperature':t,
            'pressure':p,
            'mixingratios':mixingratios,
            'molecules': molecules,
            'cloudprofile': cloud_profile
        })

    ################################
    # MIXING RATIO GRAPH 
    ################################
    kwargs = {}
    kwargs['y_axis_label'] = kwargs.get('y_axis_label','Pressure(Bars)')
    kwargs['x_axis_label'] = kwargs.get('x_axis_label','Mixing Ratio(v/v)')
    kwargs['y_axis_type'] = kwargs.get('y_axis_type','log')
    kwargs['x_axis_type'] = kwargs.get('x_axis_type','log') 
    bokeh_fig = figure(**kwargs)

    cols = pals.magma(min([len(molecules),limit]))
    legend_it=[]

    moles = {mol:[] for mol in molecules}
    fig, axes = plt.subplots(figsize=(15, 5))
    fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    for i in range(nsamples):
        pressure = save_all_class_pt[i]['pressure']
        # st.write(pressure)
        temperature = save_all_class_pt[i]['temperature']
        # st.write(temperature)
        mixingratios = save_all_class_pt[i]['mixingratios']
        axes.semilogy(temperature,pressure, color='red', alpha=0.1)
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
        # ax1.set_xlim([0,1])
        ax1.invert_yaxis()
        ax1.set_title("Single scattering albedo vs Pressure")
        ax2.semilogy(g01d, cloud_pressure)
        # ax2.set_xlim([0,1])
        ax2.invert_yaxis()
        ax2.set_title("Asymmetry vs Pressure")
        ax3.loglog(opd1d, cloud_pressure)
        # ax3.set_xlim([1e-5,50])
        ax3.set_title("Optical Depth vs Pressure")
        ax3.invert_yaxis()

        for mol, c in zip(molecules, cols):
            # this needs to not be inside this for loop
            f = bokeh_fig.line(mixingratios[mol],pressure, color=c, line_width=2,
                muted_color=c, muted_alpha=0.05, line_alpha=0.5)
            moles[mol].append(f)
    for mol in moles.keys():
        legend_it.append((mol, moles[mol]))
    legend = Legend(items=legend_it, location=(0, -20))
    legend.click_policy="mute"
    bokeh_fig.add_layout(legend, 'left')
    bokeh_fig.y_range.flipped = True
    streamlit_bokeh(bokeh_fig)
    st.pyplot(fig2)
    # # make units accurate to what in driver.toml
    axes.set_xlabel("Temperature (K)") 
    axes.set_ylabel("Log Pressure(Bars)")
    axes.set_title(f"Pressure-Temperature Profiles ({nsamples} Samples)")
    axes.invert_yaxis()
    axes.set_yscale('log')
    st.pyplot(fig)

st.divider()

################################
# SPECTRUM GRAPHS!!!
################################
new_config['see_prior_spectrums'] =  st.selectbox("See Spectrums for Priors?", ("Yes", "No"), index=None)

if new_config['see_prior_spectrums'] == 'Yes':
    WNO_LIST = []
    ALB_LIST = []
    for prior_toml in ALL_TOMLS:
        cleaned = clean_dictionary(prior_toml, '_options')
        x_range = [0,15]
        spectral_resolution = 150

        df = go.run(driver_dict=cleaned)
        obs_key = 'thermal' if prior_toml['observation_type'] == 'thermal' else 'albedo'
        wno, alb, fpfs, full = df['wavenumber'] , df[obs_key] , df[f'fpfs_{prior_toml['observation_type']}'], df['full_output']
        wno, alb = jdi.mean_regrid(wno, alb, R=spectral_resolution)
        WNO_LIST.append(wno)
        ALB_LIST.append(alb)

    streamlit_bokeh(jpi.spectrum(WNO_LIST, ALB_LIST, palette=[(255,0,0,0.3)], plot_width=500,x_range=x_range))


# TODO --> unstringify all the np.int(64)s stored
# TODO --> hide until wanted
# TODO --> remove _options for temp profile & restore for changes
st.download_button(
    label="Download current config",
    data=toml.dumps(config),
    file_name="configured_toml.toml",
    mime="application/toml"
)