# -------------------- #
# TODO: change all paths to os dynamic finds
# add validation & options for users to find necessary data /specify datapaths
# TODO --> options should be what files users have in their directories
# add other unit options & validation


# IMPORTS
import pandas as pd
import os
from PIL import Image
import math
# PATHS AND ENV VARIABLES --> need to change for users
toml_path = '/Users/sjanson/Desktop/code/picaso/reference/input_tomls/driver.toml'
os.environ['picaso_refdata'] = "/Users/sjanson/Desktop/code/picaso/reference"
os.environ['PYSYN_CDBS'] = '/Users/sjanson/Desktop/code/picaso/reference/grp/redcat/trds' #this is for the stellar data discussed below.

import picaso.driver as go
from picaso import justdoit as jdi 
from picaso import justplotit as jpi
from picaso.parameterizations import Parameterize

import streamlit as st
from streamlit_bokeh import streamlit_bokeh
from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode

#---INITIALIZE VARIABLES--------------------------#
data_manager = jdi.inputs()
# option to set wave_range here:
opa = jdi.opannection(wave_range=[2.7,6])

observation_type = None
calculation_type = None
phase_angle = 0
chemistry_method = None
retrieval_code, cloud1_type, temperature = [None, None, None]
opacity_method = 'resampled'
chemistry_config = {
    'method':'free', 
    'free': {
        'H2O': {'value': 0.001, 'unit': 'v/v'}, 
        'CH4': {'value': 0.0004, 'unit': 'v/v', 'pressures': [0.01], 'pressure_unit': 'bar'}, 
        'NH3': {'value': 1e-05, 'unit': 'v/v'}, 
        'background': {'gases': ['H2', 'He'], 'fraction': 5.667}
    }, 
    'visscher': {'cto_absolute': 0.55, 'log_mh': 0}, 
    'userfile': {'filename': '/Users/sjanson/Desktop/code/picaso/reference/base_cases/HJ.pt', 'pd_kwargs': {'sep': ' '}}
}
object_config = {
    'radius': {'value': 1.2, 'unit': 'Rjup'}, 
    'mass': {'value': 1.2, 'unit': 'Mjup'}, 
    'gravity': {'value': 100000.0, 'unit': 'cm/s**2'}, 
    'distance': {'value': 8.3, 'unit': 'parsec'}, 
    'teff': {'value': 5400, 'unit': 'Kelvin'}, 
    'teq': {'value': 500, 'unit': 'Kelvin'}
}
star_config = {
    'radius': {'value': 1, 'unit': 'Rsun'}, 
    'semi_major': {'value': 200, 'unit': 'AU'}, 
    # 'type': 'grid', 
    'grid': {
        'logg': 4, 'teff': 5400, 'feh': 0, 'database': 'ck04models'
    }
    # 'userfile': {'filename': '', 'w_unit': '', 'f_unit': ''}
}
clouds_config = {
    'cloud1_type': cloud1_type or 'brewster_mie', 
    'cloud1': {'hard_grey': {'p': 1, 'dp': 1, 'w0': 0, 'g0': 0, 'opd': 10}, 
                'flex_fsed': {'condensate': 'Al2O3', 'base_pressure': 1, 'ndz': 2, 'fsed': 1, 'distribution': 'lognorm', 'lognorm_kwargs': {'sigma': 1, 'lograd': -4}, 'hansen_kwargs': {'b': 1, 'lograd': -4}}, 
                'virga': {'mh': 1, 'condensates': ['Al2O3'], 'fsed': 1, 'kzz': 1000000000.0, 'mmw': 2.2, 'sig': 2}, 
                'brewster_mie': {'condensate': 'Na2S', 'decay_type': 'slab', 'slab_kwargs': {'ptop': 1, 'dp': 1, 'reference_tau': 1}, 'deck_kwargs': {'ptop': -1, 'dp': 1}, 'distribution': 'lognorm', 'lognorm_kwargs': {'sigma': 1, 'lograd': -4}, 'hansen_kwargs': {'b': 1, 'lograd': -4}}, 
                'brewster_grey': {'decay_type': 'slab', 'alpha': -4, 'ssa': 1, 'reference_wave': 1, 'slab_kwargs': {'ptop': -1, 'dp': 1, 'reference_tau': 1}, 'deck_kwargs': {'ptop': -1, 'dp': 1}}}, 'patchy': {'npatch': 1, 'patch1': 'clear', 'patch2': 'cloud1', 'frac_1': 0.3}
}
opticalproperties_config = {
    'opacity_files': '/Users/sjanson/Desktop/code/picaso/reference/opacities/opacities.db', 
    'opacity_method': opacity_method or 'preweighted', 
    'opacity_kwargs': {}, 
    'virga_mieff': '/Users/sjanson/Desktop/code/picaso/reference/virga/'
}

temperature_config = {
    'profile': 'knots',
    'userfile': {'filename': '/Users/sjanson/Desktop/code/picaso/reference/base_cases/HJ.pt', 'pd_kwargs': {'sep': ' '}}, 
    'pressure': {'reference': {'value': 10.0, 'unit': 'bar'}, 'min': {'value': 1e-05, 'unit': 'bar'}, 'max': {'value': 1000.0, 'unit': 'bar'}, 'nlevel': 100, 'spacing': 'log'}, 
    'isothermal': {'T': 500}, 
    'knots': {'P_knots': [100.0, 10.0, 1.0, 0.1, 0.001, 0.0001], 'T_knots': [1000, 550, 300, 200, 100, 50], 'interpolation': 'brewster', 'scipy_interpolate_kwargs': {}}, 
    'madhu_seager_09_inversion': {'P_1': 10.0, 'P_2': 0.001, 'P_3': 1e-05, 'T_3': 1000, 'alpha_1': 1, 'alpha_2': 1, 'beta': 0.5}, 
    'madhu_seager_09_noinversion': {'P_1': 10.0, 'P_3': 1e-05, 'T_3': 1000, 'alpha_1': 1, 'alpha_2': 1, 'beta': 0.5}, 
    'guillot': {'Teq': 1000, 'T_int': 100, 'logg1': -1, 'logKir': -1.5, 'alpha': 0.5}, 
    'sonora_bobcat': {'sonora_path': '/Users/nbatalh1/Documents/data/sonora_bobcat/structures_m+0.0', 'teff': 1000}
}
param_tools = Parameterize(load_cld_optical=go.find_values_for_key(clouds_config ,'condensate'),
                                    mieff_dir=opticalproperties_config.get('virga_mieff', None))

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
# bugged...
# st.selectbox(
#     "Database method",
#     ("phoenix", "ck04models"),
# )
database_method = None
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
if calculation_type == "spectrum":
    observation_type = st.selectbox(
    "Observation type",
    ("thermal", "reflected", "transmission (bugged)"), 
    index=None,
    help="Reflected runs a spectrum for a planet and requires that star data is submitted. Thermal runs a spectrum for a brown dwarf and sun data is N/A"
    )
elif calculation_type != None:
    st.warning(f'The {calculation_type} option has not been implemented yet.')
# --- ACTIONS BASED ON OBSERVATION TYPE ------------ #

if observation_type:
    st.divider()

if observation_type == 'reflected':
    st.header('Reflected Exoplanet Spectrum Config')

elif observation_type == 'transmission':
    st.header('Transmission Spectrum Config')
elif observation_type == 'thermal':
    st.header('Brown Dwarf Thermal Emission Spectrum Config')


# thermal != BD, add selection choice if want sun or not
if observation_type == 'reflected' or observation_type == 'transmission (bugged)':
    # looks like thermal could take a sun
    # --- CONFIGURE STAR ------------------- #
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
    star_gb = GridOptionsBuilder.from_dataframe(star_df)
    star_gb.configure_column('Radius', editable=True, type=["numericColumn", "numberColumnFilter"])
    star_gb.configure_column('R Unit')
    star_gb.configure_column('Semi_Major', editable=True, type=["numericColumn", "numberColumnFilter"])
    star_gb.configure_column('Unit')
    star_gridOptions = star_gb.build() 
    star_grid = AgGrid(star_df, gridOptions=star_gridOptions, fit_columns_on_grid_load=True, height=60)['data']
    for key in star_grid:
        if key.lower() in object_config and star_grid[key][0]:
            star_config[key.lower()]['value'] = star_grid[key][0]

# all shared
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
    object_gb = GridOptionsBuilder.from_dataframe(object_df)
    object_gb.configure_column('Radius', editable=True, type=["numericColumn", "numberColumnFilter"])
    object_gb.configure_column('R Unit')
    object_gb.configure_column('Mass', editable=True, type=["numericColumn", "numberColumnFilter"])
    object_gb.configure_column('M Unit')
    object_gb.configure_column('Gravity', editable=True, type=["numericColumn", "numberColumnFilter"]) 
    object_gb.configure_column('G Unit')
    object_gridOptions = object_gb.build() 
    object_grid = AgGrid(object_df, gridOptions=object_gridOptions, fit_columns_on_grid_load=True, height=60)['data']

    for key in object_grid:
        if key.lower() in object_config and object_grid[key][0]:
            object_config[key.lower()]['value'] = object_grid[key][0]

    # SWITCH TO DEGREES TODO
    phase_angle = st.number_input('Enter phase angle in radians 0-2π', min_value=0, max_value=6, value=0)
    # how to model chemistry
    st.subheader("Atmospheric Variables")

    # how to model temperature
    temperature_config['profile'] = st.selectbox(
        "Select a temperature profile",
        ('pressure (bugged)', 'isothermal', 'knots', 'madhu_seager_09_inversion', 'madhu_seager_09_noinversion', 'guillot'), # 'zj_24', 'molliere_20', 'Kitzman_20', 'sonora_bobcat, userfile'
        index=None 
    )
    # add section to add pressure
    if temperature_config['profile'] == 'knots':
        temperature_config['knots']['interpolation'] = st.selectbox(
        "Interpolation for knots temperature profile (not updating for intermediate callback, should be quickfix)",
        ('brewster', 'linear (bugged)', 'quadratic_spline', 'cubic_spline'), # OR any function from scipy.interpolate with associated kwargs passed to scipy_it scipy_interpolate_kwargs={}
    )
    
    if st.button('See Pressure-Temperature graph'):
        data_manager.star(opa, temp=star_config['grid']['teff'], database='ck04models',
        logg=star_config['grid']['logg'], metal=star_config['grid']['feh'], 
        radius=star_config['radius']['value'], radius_unit=jdi.u.R_sun) # star_config['radius']['unit'])
        data_manager.gravity(mass=object_config['mass']['value'], mass_unit=jdi.u.M_jup,#object_config['mass']['unit'],
                             radius=object_config['radius']['value'], radius_unit=jdi.u.R_jup)#object_config['radius']['unit'])
        data_manager.add_pt(P_config=temperature_config['pressure'])
        param_tools.add_class(data_manager)
        temperature_func = getattr(param_tools, f'pt_{temperature_config['profile']}')
        pt_df = temperature_func(**temperature_config[temperature_config['profile']])
        full_output = dict({'layer':{'pressure': pt_df['pressure'], 'temperature': pt_df['temperature']}})
        # pt_df.rename(columns={'pressure',})
        # st.write(pt_df)
        data_manager.atmosphere(df=pt_df)
        param_tools.add_class(data_manager)
        streamlit_bokeh(jpi.pt(full_output))
        # CHECK for temperature value for sonora_bobcat and userfile eventually

    chemistry_config['method'] = st.selectbox(
        "How to model chemistry",
        ('free', 'visscher'), index=None # userfile, fast_chem, photo_chem
    )
    if chemistry_config['method'] == 'free':
        df = pd.DataFrame({
            'Molecule (Fixed)': ['H20', 'CH4', 'NH3'],
            'Value': [1e-3,4e-4,1e-5],
            'Unit': ['v/v','v/v','v/v'],
            'Pressures': [0, 1e-2, 0],
            'Pressure Unit': ['bar', 'bar', 'bar']
        })
            # chem_type = chemistry_free_config
            # need pressure & mixingratios...
            # needs layer pressure, layer mixingratios
            # see what is to plot

        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_column('Molecule', editable=True)
        gb.configure_column('Value', editable=True, type=["numericColumn", "numberColumnFilter"])
        gb.configure_column('Unit', editable=False)
        gb.configure_column('Pressures', editable=True, type=["numericColumn", "numberColumnFilter"])
        gb.configure_column('Unit', editable=False)

        gridOptions = gb.build() 
        grid = AgGrid(df, gridOptions=gridOptions, fit_columns_on_grid_load=True, height=123)['data']

        # st.session_state.data = grid 
        # if st.button("Add Row"):
        #     new_row = {'Molecule': '', 'Value': 0, 'Unit': 'v/v', 'Pressures': 0, 'Pressure Unit': 'bar'}
        #     st.session_state.data.loc[len(st.session_state.data)] = new_row

        #     grid = AgGrid(st.session_state.data, fit_columns_on_grid_load=True, height=150)['data']

        for i,mole in enumerate(chemistry_config['free'].keys()):
            if mole != 'background':
                chemistry_config['free'][mole]['value'] = grid['Value'][i]
                if grid['Pressures'][i] > 0:
                    chemistry_config['free'][mole]['pressure'] = grid['Pressures'][i]

        if st.button('See Mixing Ratios'):
            # getting error that an attribute doesn't exist --> this is because something differs with
            # config in add_class v pt (add_class looking for atmosphere -> profile -> pressure)
            # TODO: handle other graphing function needing to be called before this
            # add some sort of checks
            chem_type = chemistry_config.get('method', '')
            if chem_type == 'userfile':
                pass # TODO implement
            elif chem_type!='':
                chemistry_function = getattr(param_tools, f'chem_{chem_type}')
                df_mixingratio = chemistry_function(**chemistry_config[chem_type])
                full_output = dict({'layer':{'pressure': df_mixingratio['pressure'], 'mixingratios': df_mixingratio['mixingratios']}})
                streamlit_bokeh(jpi.mixing_ratio(full_output))

            # st.write(param_tools)

# how to model clouds
# cloud1_type = st.selectbox(
#     "Cloud types", ('brewster_mie', 'brewster_grey', 'virga', 'flex_fsed', 'hard_grey')
# )

# retrieval_code = st.selectbox(
#     "Retrieval Sampler Code",
#     ("dynesty", "ultranest", "emcee", "pymultinest")
# )

config = {'observation_type': observation_type or 'reflected', 
    'irradiated': observation_type == 'reflected' or observation_type == 'transmission' or False, 
    'calc_type': calculation_type or 'spectrum', 
    'InputOutput': {
        'observation_data': [
            '/Users/sjanson/Desktop/code/picaso/docs/notebooks/D_climate/spectra_logzz_9.0_teff_400.0_grav_1000.0_mh_0.0_co_1.0.nc'], 
            'retrieval_output': '/Users/sjanson/Desktop/code/picaso/docs/notebooks/J_driver_WIP/output', 
            'spectrum_output': '/Users/sjanson/Desktop/code/picaso/docs/notebooks/J_driver_WIP/output'
        }, 
    'OpticalProperties': opticalproperties_config,
    'object': object_config,
    'geometry': {
        'phase': {'value': phase_angle, 'unit': 'radian'}
    }, 
    'star': star_config, 
    'temperature': temperature_config, 
    'chemistry': chemistry_config, 
    'clouds': clouds_config, 
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
        if observation_type == 'transmission (bugged)':
            st.write(df.keys())
            # rprs2 is NaN rn
            wno, rprs2  = df['wavenumber'] , df['transit_depth']
            wno, rprs2 = jdi.mean_regrid(wno, rprs2, R=150)
            full_output = df['full_output']
            streamlit_bokeh(jpi.pt(full_output,plot_width=500))
            streamlit_bokeh(jpi.mixing_ratio(full_output,plot_width=500))
            streamlit_bokeh(jpi.spectrum(wno,rprs2*1000000,plot_width=500))
        else:
            obs_key = 'thermal' if observation_type == 'thermal' else 'albedo'
            wno, alb, fpfs, full = df['wavenumber'] , df[obs_key] , df[f'fpfs_{observation_type}'], df['full_output']
            # mix_ratio_fig = jpi.mixing_ratio(full_output)
            # cloud_fig = jpi.cloud(full_output)
            pt_fig = jpi.pt(full)
            
            # photon attenuation depth, heatmap 
            streamlit_bokeh(jpi.mixing_ratio(full,plot_width=500))

            wno, alb = jdi.mean_regrid(wno, alb , R=150)
            spec_fig = jpi.spectrum(wno, alb, plot_width=500,x_range=[0.3,1])

            streamlit_bokeh(spec_fig, theme="streamlit", key="spectrum")
            # move to where configure atmospheric options...
            streamlit_bokeh(pt_fig, theme="streamlit", key="pt")
            # streamlit_bokeh(cloud_fig, use_container_width=True, theme="streamlit", key="cloud")
            # can maybe add run without molecule functionality!