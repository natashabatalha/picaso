import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from picaso import information_content as ic
from picaso import justdoit as jdi
from picaso import justplotit as jpi
import os


st.set_page_config(page_title="Information Statistics Dashboard", layout="wide")

st.title("Information Statistics Dashboard")
st.markdown("""
This dashboard showcases information content statistics for your model. 
You can explore how resolution (R) affects the Degrees of Freedom (DOF) and Shannon Information Content.
""")

# =======================================
# SIDEBAR INPUTS
# =======================================
st.sidebar.header("Parameters")

r_min = st.sidebar.number_input("Minimum Resolution (R)", min_value=10, value=50, step=10)
r_max = st.sidebar.number_input("Maximum Resolution (R)", min_value=10, value=500, step=10)
r_step = st.sidebar.number_input("Resolution Step", min_value=1, value=50, step=1)

error_val = st.sidebar.number_input("Spectral Error (Absolute)", min_value=1e-6, value=0.01, format="%.4f")

# =======================================
# DATA LOADING / GENERATION
# =======================================

def get_example_jacobian():
    st.info("Generating example Jacobian (Jupiter-like case)... this may take a minute.")
    _default_ = jdi.__refdata__ 
    simple_input = {
        'OpticalProperties': {'opacity_file': os.path.join(_default_, 'opacities', 'opacities.db'),
                           'opacity_kwargs': {'wave_range':[0.3,2.0]},
                           'opacity_method': 'resampled',
                           'virga_mieff': os.path.join(_default_, 'virga/')},
        'calc_type': 'spectrum',
        'irradiated': True,
        'geometry': {'phase': {'unit': 'radian', 'value': np.pi/2}, 
                     'phase_kwargs':{'num_tangle':6, 'num_gangle':6}},
        'object': {'distance': {'unit': 'parsec', 'value': 8.3},
                'gravity': {'unit': 'cm/s**2', 'value': 100000.0},
                'mass': {'unit': 'Mjup', 'value': 1.2},
                'radius': {'unit': 'Rjup', 'value': 1.2},
                'teff': {'unit': 'Kelvin', 'value': 5400},
                'teq': {'unit': 'Kelvin', 'value': 500}},
        'observation_type': 'reflected',   
        'star': {'grid': {'database': 'ck04models',
                       'feh': 0,
                       'logg': 4,
                       'teff': 5400},
              'radius': {'unit': 'Rsun', 'value': 1},
              'semi_major': {'unit': 'AU', 'value': 200}
        },
        'temperature':{
            'profile':'guillot',
            'pressure':{
                'reference': {'value': 1e1, 'unit': 'bar'},
                'min': {'value': 1e-5, 'unit': 'bar'},
                'max': {'value': 1e3, 'unit': 'bar'},
                'nlevel': 60,
                'spacing': 'log'
            },
            'guillot': {'T_int': 100,
                                 'Teq': 200,
                                 'alpha': 0.5,
                                 'logKir': -1.5,
                                 'logg1': -1}
        }, 
        'chemistry':{
                'method': 'chemeq_on_the_fly',
                'chemeq_on_the_fly': {'cto_absolute': 0.55, 'log_mh': 2},
        },
        'clouds':{
            'cloud1_type': 'virga',
            'cloud1': {'virga':
                        {'condensates': ['H2O'],
                            'fsed': 2,
                            'kzz': 1e8,
                            'mh': 100,
                            'mmw': 2.2,
                            'sig': 2
                            }
                        },
            }
        } 
    
    spectrum = ic.run(driver_dict=simple_input)
    jac_params = ['cto_absolute','log_mh','fsed','Teq','phase']
    jac_mat = ic.jacobian(driver_dict = simple_input, params = jac_params)
    
    return spectrum['wavenumber'], jac_mat, jac_params

if 'jacobian_data' not in st.session_state:
    st.subheader("Data Initialization")
    col_init1, col_init2 = st.columns(2)
    with col_init1:
        st.text('Computes Neptune-line Chemical Equilibrium Jacobian Example')
        if st.button("Initialize Example"):
            wno, jac_mat, params = get_example_jacobian()
            st.session_state.jacobian_data = {
                'wno': wno,
                'jacobian': jac_mat,
                'params': params
            }
            st.rerun()
    with col_init2:
        uploaded_file = st.file_uploader("Upload Jacobian Data (.npz)", type="npz")
        if uploaded_file is not None:
            data_load = np.load(uploaded_file)
            if all(key in data_load for key in ['wno', 'jacobian', 'params']):
                st.session_state.jacobian_data = {
                    'wno': data_load['wno'],
                    'jacobian': data_load['jacobian'],
                    'params': list(data_load['params'])
                }
                st.success("Data uploaded successfully!")
                st.rerun()
            else:
                st.error("Uploaded file must contain 'wno', 'jacobian', and 'params' keys.")
else:
    if st.button("Reset Jacobian Data"):
        del st.session_state.jacobian_data
        st.rerun()


# =======================================
# COMPUTATION
# =======================================

if 'jacobian_data' in st.session_state:

    #GET PRIORS
    st.subheader('Input Priors for your Parameter Set')
    params = st.session_state['jacobian_data']['params']
    df_prior = pd.DataFrame([{i:'float' for i in params}])
    df_prior = st.data_editor(df_prior)
    

    if not st.button('Finalize Priors'):
        st.info("Please edit the table above and click 'Finalize Priors' to continue.")
        st.stop()
    else:
        priors = [df_prior.loc[0,i] for i in df_prior]
        passes = True
        for i,p in enumerate(priors): 
            try: 
                priors[i]=float(p)
            except: 
                st.error(fr"Each prior must be a number. Parameter {params[i]}=={p} is not")
                passes=False
            if p==0: 
                st.error(fr"Each prior must be a non-zero number. Parameter {params[i]}=={p} is not")
                passes=False
        if not passes: 
            st.stop()


    data = st.session_state.jacobian_data
    wno = data['wno']
    jac_mat = data['jacobian']
    params = data['params']

    resolutions = np.arange(r_min, r_max + r_step, r_step)
    resolutions = resolutions[resolutions <= r_max]
    
    results_svd = []
    results_shannon = []
    
    progress_bar = st.progress(0)
    for i, R in enumerate(resolutions):
        analyzer = ic.Analyze(wno, jac_mat, error_val, R=R)
        
        dfs = analyzer.degrees_of_freedom_svd()
        sic = analyzer.shannon_ic(priors)
        
        results_svd.append(dfs)
        results_shannon.append(sic)
        progress_bar.progress((i + 1) / len(resolutions))

    # =======================================
    # VISUALIZATIONS
    # =======================================
    
    st.divider()
    
    # 1) Plot of their binned Jacobian values for their highest resolution input
    st.header(f"1) Binned Jacobian (R={r_max})")
    highest_r_analyzer = ic.Analyze(wno, jac_mat, error_val, R=r_max)
    
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    rebinned_wno = highest_r_analyzer.new_wno if highest_r_analyzer.new_wno is not None else wno
    # Rebinning was done in __init__ -> rebin_jac_error. self.jacobian is now (nparams, nwno) matrix
    # wait, rebin_jac_error sets self.jacobian = np.matrix(K_rebin) which is (nparams, nwno)
    
    for i, p_name in enumerate(params):
        y_val = np.array(highest_r_analyzer.jacobian[i, :]).flatten()
        # Normalize as in notebook for better visualization? 
        # Notebook does: plt.plot(1e4/x, jdi.np.abs(y/jdi.np.max(jdi.np.abs(y))), label=ip)
        norm_y = np.abs(y_val) / np.max(np.abs(y_val)) if np.max(np.abs(y_val)) != 0 else y_val
        ax1.plot(1e4/rebinned_wno, norm_y, label=p_name)
    
    ax1.set_xlabel("Wavelength [um]")
    ax1.set_ylabel("Normalized |Jacobian|")
    ax1.set_title(f"Binned Jacobian for R={r_max}")
    ax1.legend()
    st.pyplot(fig1)

    # 2) Plot of their SVD degrees of freedom as a function of resolution
    st.divider()
    st.header("2) SVD Degrees of Freedom vs. Resolution")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(resolutions, results_svd, marker='o', linestyle='-', color='blue')
    ax2.set_xlabel("Resolution (R)")
    ax2.set_ylabel("SVD DFS")
    ax2.set_title("SVD Degrees of Freedom for Signal")
    st.pyplot(fig2)

    # 3) Plot of each of the shannon_ic dictionary outputs as a function of resolution
    st.divider()
    st.header("3) Shannon Information Statistics vs. Resolution")
    
    col1, col2 = st.columns(2)
    
    # DOF and H (Scalars)
    with col1:
        dofs = [r['DOF'] for r in results_shannon]
        hs = [r['H'] for r in results_shannon]
        
        fig3a, ax3a = plt.subplots()
        ax3a.plot(resolutions, dofs, marker='s', label='Shannon DOF', color='green')
        ax3a.set_xlabel("Resolution (R)")
        ax3a.set_ylabel("DOF")
        ax3a.legend()
        st.pyplot(fig3a)
        
    with col2:
        fig3b, ax3b = plt.subplots()
        ax3b.plot(resolutions, hs, marker='^', label='Shannon Information (H)', color='red')
        ax3b.set_xlabel("Resolution (R)")
        ax3b.set_ylabel("H [bits]")
        ax3b.legend()
        st.pyplot(fig3b)
        
    # Averaging Kernel and Constraint Interval (Vectors)
    st.subheader("Parameter-specific Shannon Stats")
    col3, col4 = st.columns(2)
    
    with col3:
        fig3c, ax3c = plt.subplots()
        for i, p_name in enumerate(params):
            ak_vals = [r['AveragingKernel'][i] for r in results_shannon]
            ax3c.plot(resolutions, ak_vals, marker='.', label=p_name)
        ax3c.set_xlabel("Resolution (R)")
        ax3c.set_ylabel("Averaging Kernel Diagonal")
        ax3c.set_title("Averaging Kernel vs. R")
        ax3c.legend()
        st.pyplot(fig3c)

    with col4:
        fig3d, ax3d = plt.subplots()
        for i, p_name in enumerate(params):
            ci_vals = [r['constraint_interval'][i] for r in results_shannon]
            ax3d.plot(resolutions, ci_vals, marker='.', label=p_name)
        ax3d.set_xlabel("Resolution (R)")
        ax3d.set_ylabel("Constraint Interval")
        ax3d.set_title("1-sigma Constraint Interval vs. R")
        ax3d.legend()
        st.pyplot(fig3d)

else:
    st.warning("Please initialize example or provide Jacobian data to begin.")
