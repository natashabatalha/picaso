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

#---IMPORTS--------------------------------#
import pandas as pd
import os
import toml

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

# most if not all of this cloggy config dict will go away once we dynamically generate it from driver.toml
# I got the framework for via print(tomllib.load('path-to-config'))
config = {'observation_type': observation_type or 'reflected', 
    'irradiated': False, 
    'calc_type': None, 
    'InputOutput': {
        'observation_data': [ # not used yet
            '/Users/sjanson/Desktop/code/picaso/docs/notebooks/D_climate/spectra_logzz_9.0_teff_400.0_grav_1000.0_mh_0.0_co_1.0.nc'], 
            'retrieval_output': '/Users/sjanson/Desktop/code/picaso/docs/notebooks/J_driver_WIP/output', 
            'spectrum_output': '/Users/sjanson/Desktop/code/picaso/docs/notebooks/J_driver_WIP/output'
        }, 
    'OpticalProperties': {
        'opacity_files': path_to_opacity_DB, 
        'opacity_method': 'resampled', 
        'opacity_kwargs': {}, 
        'virga_mieff': path_to_virga_mieff
    },
    'object': {
        'radius': {'value': 1.2, 'unit': 'Rjup'}, 
        'mass': {'value': 1.2, 'unit': 'Mjup'}, 
        'gravity': {'value': 100000.0, 'unit': 'cm/s**2'}, 
        'distance': {'value': 8.3, 'unit': 'parsec'}, 
        'teff': {'value': 5400, 'unit': 'Kelvin'}, 
        'teq': {'value': 500, 'unit': 'Kelvin'}
    },
    'geometry': {
        'phase': {'value': 0, 'unit': 'radian'}
    }, 
    'star': {
        'radius': {'value': 1, 'unit': 'Rsun'}, 
        'semi_major': {'value': 200, 'unit': 'AU'}, 
        # 'type': 'grid', 
        'grid': {
            'logg': 4, 'teff': 5400, 'feh': 0, 'database': 'ck04models'
        }
        # 'userfile': {'filename': '', 'w_unit': '', 'f_unit': ''}
    }, 
    'temperature': {
        'profile': 'knots',
        'userfile': {'filename': '/Users/sjanson/Desktop/code/picaso/reference/base_cases/HJ.pt', 'pd_kwargs': {'sep': ' '}}, # not used yet
        'pressure': {'reference': {'value': 10.0, 'unit': 'bar'}, 'min': {'value': 1e-05, 'unit': 'bar'}, 'max': {'value': 1000.0, 'unit': 'bar'}, 'nlevel': 100, 'spacing': 'log'}, 
        'isothermal': {'T': 500}, 
        'knots': {'P_knots': [100.0, 10.0, 1.0, 0.1, 0.001, 0.0001], 'T_knots': [1000, 550, 300, 200, 100, 50], 'interpolation': 'brewster', 'scipy_interpolate_kwargs': {}}, 
        'madhu_seager_09_inversion': {'P_1': 10.0, 'P_2': 0.001, 'P_3': 1e-05, 'T_3': 1000, 'alpha_1': 1, 'alpha_2': 1, 'beta': 0.5}, 
        'madhu_seager_09_noinversion': {'P_1': 10.0, 'P_3': 1e-05, 'T_3': 1000, 'alpha_1': 1, 'alpha_2': 1, 'beta': 0.5}, 
        'guillot': {'Teq': 1000, 'T_int': 100, 'logg1': -1, 'logKir': -1.5, 'alpha': 0.5}, 
        'sonora_bobcat': {'sonora_path': '/Users/nbatalh1/Documents/data/sonora_bobcat/structures_m+0.0', 'teff': 1000}
    }, 
    'chemistry': {
        'method':'free', 
        'free': {
            'H2O': {'values': ['0.001'], 'unit': 'v/v', 'pressures': ['0'], 'pressure_unit': 'bar'}, 
            'CH4': {'values':['1e-4','1e-3'], 'unit': 'v/v', 'pressures': ['0.01'], 'pressure_unit': 'bar'}, 
            'NH3': {'values':['1e-05'], 'unit': 'v/v', 'pressures': ['0'], 'pressure_unit':'bar'}, 
            'background': {'gases': ['H2', 'He'], 'fraction': 5.667}
        }, 
        'visscher': {'cto_absolute': 0.55, 'log_mh': 0}, 
        'userfile': {'filename': '/Users/sjanson/Desktop/code/picaso/reference/base_cases/HJ.pt', 'pd_kwargs': {'sep': ' '}} # not used yet
    }, 
    'clouds': {
        'cloud1_type': 'brewster_mie', 
        'cloud1': {'hard_grey': {'p': 1, 'dp': 1, 'w0': 0, 'g0': 0, 'opd': 10}, 
                    'flex_fsed': {'condensate': 'Al2O3', 'base_pressure': 1, 'ndz': 2, 'fsed': 1, 'distribution': 'lognorm', 'lognorm_kwargs': {'sigma': 1, 'lograd': -4}, 'hansen_kwargs': {'b': 1, 'lograd': -4}}, 
                    'virga': {'mh': 1, 'condensates': ['Al2O3'], 'fsed': 1, 'kzz': 1000000000.0, 'mmw': 2.2, 'sig': 2}, 
                    'brewster_mie': {'condensate': 'Na2S', 'decay_type': 'slab', 'slab_kwargs': {'ptop': 1, 'dp': 1, 'reference_tau': 1}, 'deck_kwargs': {'ptop': -1, 'dp': 1}, 'distribution': 'lognorm', 'lognorm_kwargs': {'sigma': 1, 'lograd': -4}, 'hansen_kwargs': {'b': 1, 'lograd': -4}}, 
                    'brewster_grey': {'decay_type': 'slab', 'alpha': -4, 'ssa': 1, 'reference_wave': 1, 'slab_kwargs': {'ptop': -1, 'dp': 1, 'reference_tau': 1}, 'deck_kwargs': {'ptop': -1, 'dp': 1}}}, 'patchy': {'npatch': 1, 'patch1': 'clear', 'patch2': 'cloud1', 'frac_1': 0.3}
    }, 
    'retrieval': {
        'object': {'radius': {'prior': 'uniform', 'min': 0.5, 'max': 2, 'log': False}}, 
        'chemistry': {'free': {'H2O': {'prior': 'uniform', 'min': -12, 'max': 0, 'log': False}}}, 
        'clouds': {
            'cloud1': {'slab-grey': {'ptop': {'prior': 'uniform', 'min': -3, 'max': 2, 'log': True}, 'dp': {'prior': 'uniform', 'min': 0.005, 'max': 2, 'log': True}}}}, 
            'sampler': {'code': 'dynesty', 'sampler_kwargs': {'live_points': 100}, 'nlive': 100, 'resume': False}
    }, 
    'sampler': {
        'code': 'dynesty', 'sampler_kwargs': {'live_points': 100}
    }
}

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
    param_tools = Parameterize(load_cld_optical=go.find_values_for_key(config ,'condensate'), mieff_dir=config['OpticalProperties'].get('virga_mieff', None))
    opacity = jdi.opannection(
        filename_db=config['OpticalProperties']['opacity_files'], #database(s)
        method=config['OpticalProperties']['opacity_method'], #resampled, preweighted, resortrebin
        **config['OpticalProperties']['opacity_kwargs'] #additonal inputs 
    )
    return go.setup_spectrum_class(config, opacity, param_tools, stage)


# ---------------------------------------------- #
# -- BEGINNING OF APP -------------------------- #
# ---------------------------------------------- #

# HEADERS -----------------------#
st.logo('https://natashabatalha.github.io/picaso/_images/logo.png', size="large", link="https://github.com/natashabatalha/picaso")
st.header('Try out PICASO functionalities',divider='rainbow')

st.info('If this is your first time, perhaps try running with all the provided values first to see what the expected results are, and then experiment away!')
st.subheader('Administrative')
st.text('Assuming data and environment variables are set up for now, will add option to add those soon!')

# ADMINISTRATIVE OPTIONS --------------#
database_method = None # = st.selectbox("Database method",("phoenix", "ck04models"),)

config['OpticalProperties']['opacity_method'] = st.selectbox("Opacity method", ("resampled")) #, "preweighted", "resortrebin"))
# if opacity_method == 'resampled': st.warning('Warning, you are selecting resampling! This could degrade the precision of your spectral calculations so should be used with caution. If you are unsure check out this tutorial: https://natashabatalha.github.io/picaso/notebooks/10_ResamplingOpacities.html',
#     icon='⚠️')

st.subheader('Select calculation to perform')
config['calc_type'] = st.selectbox("Calculation type", ("spectrum", "climate", "retrieval"), index=None)

if config['calc_type'] == "spectrum":
    # transmission is an option too, but dependent on A.approx functionality getting added to driver.py
    observation_type = st.selectbox("Observation type", ("thermal", "reflected"), index=None, help="Reflected runs a spectrum for a planet and requires that star data is submitted. Thermal runs a spectrum for a brown dwarf and sun data is N/A")
elif config['calc_type'] != None:
    st.warning(f'The {config['calc_type']} option has not been implemented yet.')

# configure heading
if observation_type:
    st.divider()
if observation_type == 'reflected':
    st.header('Reflected Exoplanet Spectrum Config')
elif observation_type == 'thermal':
    st.header('Brown Dwarf Thermal Emission Spectrum Config')


# thermal != BD, add selection choice if want sun or not
# clarify how to prompt user for sun
# --- CONFIGURE STAR ------------------- #
if observation_type == 'reflected' or observation_type == 'transmission':
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
        if key.lower() in config['object'] and star_grid[key][0]: # why checking object & star here again?
            config['star'][key.lower()]['value'] = star_grid[key][0]

# --- SHARED QUESTIONS between thermal, transmission, and reflected ---- #
if observation_type:
    # CONFIGURE PLANET/BD ---------------------- #
    st.subheader("Object Variables")
    object_df = pd.DataFrame({
        'Radius': [1.2],
        'R Unit': 'Rjup',
        'Mass': [1.2],
        'M Unit': 'Mjup',
        'Gravity': [100000.0],
        'G Unit': 'cm/s**2'
    })
    object_grid = st.data_editor(object_df)

    # updating config file
    for key in object_grid:
        if key.lower() in config['object'] and object_grid[key][0]:
            config['object'][key.lower()]['value'] = object_grid[key][0]

    # TODO switch to degrees/ give flexibility between radians and degrees
    config['geometry']['phase']['value'] = st.number_input('Enter phase angle in radians 0-2π', min_value=0, max_value=6, value=0)


    # ATMOSPHERIC VARIABLES ---------------------- #
    st.subheader("Atmospheric Variables")
    # TODO add section for pressure
    config['temperature']['profile'] = st.selectbox(
        "Select a temperature profile",
        ('pressure (bugged)', 'isothermal', 'knots', 'madhu_seager_09_inversion', 'madhu_seager_09_noinversion', 'guillot'), # 'zj_24', 'molliere_20', 'Kitzman_20', 'sonora_bobcat, userfile'
        index=None 
    )
    if config['temperature']['profile'] == 'knots':
        config['temperature']['knots']['interpolation'] = st.selectbox(
        "Interpolation for knots temperature profile (not updating for intermediate callback, should be quickfix)",
        ('brewster', 'linear (bugged)', 'quadratic_spline', 'cubic_spline'), # OR any function from scipy.interpolate with associated kwargs passed to scipy_it scipy_interpolate_kwargs={}
    )
    
    # see PT graph
    if st.button('See Pressure-Temperature graph'):
        data_class = run_spectrum_class('temperature')
        streamlit_bokeh(jpi.pt({'layer': data_class.inputs['atmosphere']['profile']}))

    #------modelling chemistry ----------------------#
    config['chemistry']['method'] = st.selectbox(
        "How to model chemistry",
        ('free', 'visscher'), index=None # userfile, fast_chem, photo_chem
    )

    # note: if added, empty rows can show up if navigate away
    if config['chemistry']['method'] == 'free':
        gridcol, buttoncol = st.columns([6,1])
        with gridcol:
            df = pd.DataFrame({
                'Molecule': ['H2O', 'CH4', 'NH3'],
                'Values': [['1e-3'],['4e-4', '1e-3'],['1e-5']],
                'Unit': ['v/v','v/v','v/v'],
                'Pressures': [['0.1'], ['1e-2'], ['0.1']],
                'Pressure Unit': ['bar', 'bar', 'bar']
            })

            st.info('Molecules are case sensitive (ex: TiO, H2O)')
            if 'chem_free_df' not in st.session_state:
                st.session_state.chem_free_df = pd.DataFrame(df)
            grid = st.data_editor(st.session_state.chem_free_df)

            # updating config file
            for i,mole in enumerate(grid['Molecule']):
                if mole not in config['chemistry']['free']:
                    config['chemistry']['free'][mole] = {'values': [], 'unit': 'v/v', 'pressures': [], 'pressure_unit': 'bar'}
                values = [float(value) for value in grid['Values'][i]]
                if len(values) == 1:
                    # don't need a pressure point if there's only one value
                    config['chemistry']['free'][mole]['value'] = values[0]
                else:
                    config['chemistry']['free'][mole]['values'] = values
                    config['chemistry']['free'][mole]['pressures'] = [float(pressure) for pressure in grid['Pressures'][i] if pressure != 'N/A']

        with buttoncol:
            # code for figuring out how to add new rows heavily inspired by @ferdy here: https://discuss.streamlit.io/t/how-to-add-delete-a-new-row-in-data-editor-in-streamlit/70608
            if st.button("Add Row"):
                new_row_df = pd.DataFrame([{'Molecule': '', 'Values': [], 'Unit': 'v/v', 'Pressures': [], 'Pressure Unit': 'bar'}])
                st.session_state.chem_free_df = pd.concat([st.session_state.chem_free_df, new_row_df], ignore_index=True)
                st.rerun()

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
    config['irradiated'] = observation_type == 'reflected' or observation_type == 'transmission'
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