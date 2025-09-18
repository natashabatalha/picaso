import streamlit as st
import os
os.environ['picaso_refdata'] = "/Users/sjanson/Desktop/code/picaso/reference"
os.environ['PYSYN_CDBS'] = '/Users/sjanson/Desktop/code/picaso/reference/grp/redcat/trds' #this is for the stellar data discussed below.
import picaso.driver as go
toml_path = '/Users/sjanson/Desktop/code/picaso/reference/input_tomls/driver.toml'
from picaso import justdoit as jdi 
from picaso import justplotit as jpi
from streamlit_bokeh import streamlit_bokeh

st.title('Try out PICASO functionalities')

# combo
observation_type = st.selectbox(
    "Observation type",
    ("reflected", "transmission", "thermal"),
)

calculation_type = st.selectbox(
    "Calculation type",
    ("spectrum", "climate", "retrieval"),
)

opacity_method = st.selectbox(
    "Opacity method",
    ("resampled", "preweighted", "resortrebin"),
)

database_method = None
# database_method = st.selectbox(
#     "Database method",
#     ("phoenix", "ck04models"),
# )

temperature = st.selectbox(
    "Temperature Profile",
    ("knots", "isothermal", "madhu_seager_09_inversion", "madhu_seager_09_noinversion", "guillot", "zj_24", "molliere_20", "Kitzman_20", "sonora_bobcat"),
)

temperature_knots = st.selectbox(
    "Temperature interpolation",
    ("brewster", "linear", "quadratic_spline", "cubic_spline")
)

chemistry_method = st.selectbox(
    "Chemistry Method",
    ('free', 'visscher')
)

cloud1_type = st.selectbox(
    "Cloud types", ('brewster_mie', 'brewster_grey', 'virga', 'flex_fsed', 'hard_grey')
)

retrieval_code = st.selectbox(
    "Retrieval Sampler Code",
    ("dynesty", "ultranest", "emcee", "pymultinest")
)

# upload option for data, else show default data to be used
# or path string option


config = {'observation_type': observation_type or 'reflected', 
          'irradiated': False, 
          'calc_type': calculation_type or 'spectrum', 
          'InputOutput': {
              'observation_data': [
                  '/Users/sjanson/Desktop/code/picaso/docs/notebooks/D_climate/spectra_logzz_9.0_teff_400.0_grav_1000.0_mh_0.0_co_1.0.nc'], 
                  'retrieval_output': '/Users/sjanson/Desktop/code/picaso/docs/notebooks/J_driver_WIP/output', 
                  'spectrum_output': '/Users/sjanson/Desktop/code/picaso/docs/notebooks/J_driver_WIP/output'
                }, 
            'OpticalProperties': {
                'opacity_files': '/Users/sjanson/Desktop/code/picaso/reference/opacities/opacities.db', 
                'opacity_method': opacity_method or 'resampled', 
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
                'phase': {'value': 0, 'unit': 'radian'}
            }, 
            'star': {
                'radius': {'value': 1, 'unit': 'Rsun'}, 
                'semi_major': {'value': 200, 'unit': 'AU'}, 
                'type': 'grid', 
                'grid': {
                    'logg': 4, 'teff': 5400, 'feh': 0, 'database': database_method or 'phoenix'
                    }, 
                'userfile': {'filename': '', 'w_unit': '', 'f_unit': ''}
            }, 
            'temperature': {
                'profile': temperature or 'knots', 
                'userfile': {'filename': '/Users/sjanson/Desktop/code/picaso/reference/base_cases/HJ.pt', 'pd_kwargs': {'sep': ' '}}, 
                'pressure': {'reference': {'value': 10.0, 'unit': 'bar'}, 'min': {'value': 1e-05, 'unit': 'bar'}, 'max': {'value': 1000.0, 'unit': 'bar'}, 'nlevel': 100, 'spacing': 'log'}, 
                'isothermal': {'T': 500}, 
                'knots': {'P_knots': [100.0, 10.0, 1.0, 0.1, 0.001, 0.0001], 'T_knots': [1000, 550, 300, 200, 100, 50], 'interpolation': temperature_knots or 'brewster', 'scipy_interpolate_kwargs': {}}, 
                'madhu_seager_09_inversion': {'P_1': 10.0, 'P_2': 0.001, 'P_3': 1e-05, 'T_3': 1000, 'alpha_1': 1, 'alpha_2': 1, 'beta': 0.5}, 
                'madhu_seager_09_noinversion': {'P_1': 10.0, 'P_3': 1e-05, 'T_3': 1000, 'alpha_1': 1, 'alpha_2': 1, 'beta': 0.5}, 
                'guillot': {'Teq': 1000, 'T_int': 100, 'logg1': -1, 'logKir': -1.5, 'alpha': 0.5}, 
                'sonora_bobcat': {'sonora_path': '/Users/nbatalh1/Documents/data/sonora_bobcat/structures_m+0.0', 'teff': 1000}
            }, 
            'chemistry': {
                'method': chemistry_method or 'free', 
                'free': {
                    'H2O': {'value': 0.001, 'unit': 'v/v'}, 
                    'CH4': {'value': 0.0004, 'unit': 'v/v', 'pressures': [0.01], 'pressure_unit': 'bar'}, 
                    'NH3': {'value': 1e-05, 'unit': 'v/v'}, 'background': {'gases': ['H2', 'He'], 'fraction': 5.667}
                    }, 
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

if st.button('Run spectrum'):
    # st.title('Configuration')
    # st.write(config)
    df = go.run(driver_dict=config)
    # st.title('Results')
    # st.write(driver_test)
    st.write(df.keys())
    wno, alb, fpfs = df['wavenumber'] , df['thermal'] , df['fpfs_thermal'] 
    wno, alb = jdi.mean_regrid(wno, alb , R=150)
    fig = jpi.spectrum(wno, alb, plot_width=500,x_range=[0.3,1])
    streamlit_bokeh(fig, use_container_width=True, theme="streamlit", key="unique")


# options seem to be funky, make sure grabbing the right thing
# but yay spectrum!!