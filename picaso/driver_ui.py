# -------------------- #
# TODO: change all paths to os dynamic finds
# add validation & options for users to find necessary data /specify datapaths


# IMPORTS
import pandas as pd
import os
from PIL import Image

# PATHS AND ENV VARIABLES --> need to change for users
toml_path = '/Users/sjanson/Desktop/code/picaso/reference/input_tomls/driver.toml'
os.environ['picaso_refdata'] = "/Users/sjanson/Desktop/code/picaso/reference"
os.environ['PYSYN_CDBS'] = '/Users/sjanson/Desktop/code/picaso/reference/grp/redcat/trds' #this is for the stellar data discussed below.

import picaso.driver as go
from picaso import justdoit as jdi 
from picaso import justplotit as jpi

import streamlit as st
from streamlit_bokeh import streamlit_bokeh
from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode


# INITIALIZE VARIABLES
phase_angle = 0
chemistry_method = None
retrieval_code, cloud1_type, temperature_knots_interpolation, temperature = [None, None, 'brewster', None]
chemistry_free_config = {
    'H2O': {'value': 0.001, 'unit': 'v/v'}, 
    'CH4': {'value': 0.0004, 'unit': 'v/v', 'pressures': [0.01], 'pressure_unit': 'bar'}, 
    'NH3': {'value': 1e-05, 'unit': 'v/v'}, 
    'background': {'gases': ['H2', 'He'], 'fraction': 5.667}
}
# -- BEGINNING OF APP -------------------------- #
picaso_logo = Image.open('/Users/sjanson/Desktop/code/picaso/docs/logo.png')
st.logo(picaso_logo, size="large", link="https://github.com/natashabatalha/picaso")
st.header('Try out PICASO functionalities',divider='rainbow')

st.info('If this is your first time, perhaps try running with all the provided values first to see what the expected results are, and then experiment away!')
# run some sort of verification to see what setup users have
# --> do they have x data, what database etc.

st.subheader('Administrative')
st.text('Assuming data and environment variables are set up for now, will add option to add those after!')
database_method = None 
# st.selectbox(
#     "Database method",
#     ("phoenix", "ck04models"),
# )
opacity_method = st.selectbox(
    "Opacity method",
    ("resampled") #, "preweighted", "resortrebin")
)
# WARNING if resampled is selected --> why is this the default then
# if opacity_method == 'resampled': st.warning(
#     'Warning, you are selecting resampling! This could degrade the precision of your spectral calculations so should be used with caution. If you are unsure check out this tutorial: https://natashabatalha.github.io/picaso/notebooks/10_ResamplingOpacities.html',
#     icon='⚠️'
#     )

st.subheader('Select calculation to perform')

calculation_type = st.selectbox(
    "Calculation type",
    ("spectrum", "climate", "retrieval"), index=None
)

# CHOOSE OBSERVATION TYPE IF SPECTRUM
observation_type = None
if calculation_type == "spectrum":
    observation_type = st.selectbox(
    "Observation type",
    ("thermal", "reflected", "transmission"), 
    index=None,
    help="Reflected runs a spectrum for a planet and requires that star data is submitted. Thermal runs a spectrum for a brown dwarf and sun data is N/A"
    )
elif calculation_type != None:
    st.warning(f'The {calculation_type} option has not been implemented yet.')
# --- ACTIONS BASED ON OBSERVATION TYPE ------------ #

if observation_type:
    st.divider()

if observation_type == 'reflected':
    st.subheader('Reflected Exoplanet Spectrum Config')
    # let users configure star or see defaults
elif observation_type == 'transmission':
    st.subheader('Transmission Spectrum Config')
elif observation_type == 'thermal':
    st.subheader('Brown Dwarf Thermal Emission Spectrum Config')

# all shared
if observation_type == 'reflected' or observation_type == 'thermal':
    phase_angle = st.number_input('Enter phase angle in radians (0-180)', min_value=0, max_value=180, value=0)
    # how to model temperature
    # how to model chemistry
    st.subheader("Atmospheric Variables")
    chemistry_method = st.selectbox(
        "How to model chemistry",
        ('free', 'visscher'), index=None # userfile, fast_chem, photo_chem
    )
    if chemistry_method == 'free':
        df = pd.DataFrame({
            'Molecule (Fixed)': ['H20', 'CH4', 'NH3'],
            'Value': [1e-3,4e-4,1e-5],
            'Unit': ['v/v','v/v','v/v'],
            'Pressures': [0, 1e-2, 0],
            'Pressure Unit': ['bar', 'bar', 'bar']
        })
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_column('Molecule', editable=False)
        gb.configure_column('Value', editable=True, type=["numericColumn", "numberColumnFilter"])
        gb.configure_column('Unit', editable=False)
        gb.configure_column('Pressures', editable=True, type=["numericColumn", "numberColumnFilter"])
        gb.configure_column('Unit', editable=False)


        gridOptions = gb.build() 
        grid = AgGrid(df, gridOptions=gridOptions, fit_columns_on_grid_load=True, height=123)['data']

        for i,mole in enumerate(chemistry_free_config.keys()):
            if mole != 'background':
                chemistry_free_config[mole]['value'] = grid['Value'][i]
                if grid['Pressures'][i] > 0:
                    chemistry_free_config[mole]['pressure'] = grid['Pressures'][i]

    temperature = st.selectbox(
        "Select a temperature profile",
        ('pressure (bugged)', 'isothermal', 'knots', 'madhu_seager_09_inversion', 'madhu_seager_09_noinversion', 'guillot'), # 'zj_24', 'molliere_20', 'Kitzman_20', 'sonora_bobcat'
        index=None 
    )

    if temperature == 'knots':
        temperature_knots_interpolation = st.selectbox(
        "Interpolation for chosen temperature profile",
        ('brewster', 'linear (bugged)', 'quadratic_spline', 'cubic_spline'), # OR any function from scipy.interpolate with associated kwargs passed to scipy_it scipy_interpolate_kwargs={}
    )

# how to model clouds
# cloud1_type = st.selectbox(
#     "Cloud types", ('brewster_mie', 'brewster_grey', 'virga', 'flex_fsed', 'hard_grey')
# )

# retrieval_code = st.selectbox(
#     "Retrieval Sampler Code",
#     ("dynesty", "ultranest", "emcee", "pymultinest")
# )

# upload option for data, else show default data to be used
# or path string option


config = {'observation_type': observation_type or 'reflected', 
          'irradiated': observation_type == 'reflected' or False, 
          'calc_type': calculation_type or 'spectrum', 
          'InputOutput': {
              'observation_data': [
                  '/Users/sjanson/Desktop/code/picaso/docs/notebooks/D_climate/spectra_logzz_9.0_teff_400.0_grav_1000.0_mh_0.0_co_1.0.nc'], 
                  'retrieval_output': '/Users/sjanson/Desktop/code/picaso/docs/notebooks/J_driver_WIP/output', 
                  'spectrum_output': '/Users/sjanson/Desktop/code/picaso/docs/notebooks/J_driver_WIP/output'
                }, 
            'OpticalProperties': {
                'opacity_files': '/Users/sjanson/Desktop/code/picaso/reference/opacities/opacities.db', 
                'opacity_method': opacity_method or 'preweighted', 
                'opacity_kwargs': {}, 
                'virga_mieff': '/Users/sjanson/Desktop/code/picaso/reference/virga/'
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
                'phase': {'value': phase_angle, 'unit': 'radian'}
            }, 
            'star': {
                'radius': {'value': 1, 'unit': 'Rsun'}, 
                'semi_major': {'value': 200, 'unit': 'AU'}, 
                'type': 'grid', 
                'grid': {
                    'logg': 4, 'teff': 5400, 'feh': 0, 'database': 'phoenix'
                    }, 
                'userfile': {'filename': '', 'w_unit': '', 'f_unit': ''}
            }, 
            'temperature': {
                'profile': temperature or 'knots', 
                'userfile': {'filename': '/Users/sjanson/Desktop/code/picaso/reference/base_cases/HJ.pt', 'pd_kwargs': {'sep': ' '}}, 
                'pressure': {'reference': {'value': 10.0, 'unit': 'bar'}, 'min': {'value': 1e-05, 'unit': 'bar'}, 'max': {'value': 1000.0, 'unit': 'bar'}, 'nlevel': 100, 'spacing': 'log'}, 
                'isothermal': {'T': 500}, 
                'knots': {'P_knots': [100.0, 10.0, 1.0, 0.1, 0.001, 0.0001], 'T_knots': [1000, 550, 300, 200, 100, 50], 'interpolation': temperature_knots_interpolation, 'scipy_interpolate_kwargs': {}}, 
                'madhu_seager_09_inversion': {'P_1': 10.0, 'P_2': 0.001, 'P_3': 1e-05, 'T_3': 1000, 'alpha_1': 1, 'alpha_2': 1, 'beta': 0.5}, 
                'madhu_seager_09_noinversion': {'P_1': 10.0, 'P_3': 1e-05, 'T_3': 1000, 'alpha_1': 1, 'alpha_2': 1, 'beta': 0.5}, 
                'guillot': {'Teq': 1000, 'T_int': 100, 'logg1': -1, 'logKir': -1.5, 'alpha': 0.5}, 
                'sonora_bobcat': {'sonora_path': '/Users/nbatalh1/Documents/data/sonora_bobcat/structures_m+0.0', 'teff': 1000}
            }, 
            'chemistry': {
                'method': chemistry_method or 'free', 
                'free': chemistry_free_config, 
                'visscher': {'cto_absolute': 0.55, 'log_mh': 0}, 
                'userfile': {'filename': '/Users/sjanson/Desktop/code/picaso/reference/base_cases/HJ.pt', 'pd_kwargs': {'sep': ' '}}
            }, 
            'clouds': {
                'cloud1_type': cloud1_type or 'brewster_mie', 
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
                    'sampler': {'code': retrieval_code or 'dynesty', 'sampler_kwargs': {'live_points': 100}, 'nlive': 100, 'resume': False}
            }, 
            'sampler': {
                'code': retrieval_code or 'dynesty', 'sampler_kwargs': {'live_points': 100}
            }
        }

if calculation_type =='spectrum' and st.button(f'Run {calculation_type}'):
    df = go.run(driver_dict=config)
    if calculation_type == 'spectrum':
        wno, alb, fpfs, full = df['wavenumber'] , df['thermal'] , df['fpfs_thermal'], df['full_output']
        # mix_ratio_fig = jpi.mixing_ratio(full_output)
        # cloud_fig = jpi.cloud(full_output)
        pt_fig = jpi.pt(full)
        
        # photon attenuation depth, heatmap 

        wno, alb = jdi.mean_regrid(wno, alb , R=150)
        spec_fig = jpi.spectrum(wno, alb, plot_width=500,x_range=[0.3,1])


        # perhpas keep separate or find way to zoom out...
        col1, col2 = st.columns(2)
        with col1:
            streamlit_bokeh(spec_fig, theme="streamlit", key="spectrum")
        # move to where configure atmospheric options...
        with col2:
            streamlit_bokeh(pt_fig, theme="streamlit", key="pt")
        # streamlit_bokeh(cloud_fig, use_container_width=True, theme="streamlit", key="cloud")
        # can maybe add run without molecule functionality!