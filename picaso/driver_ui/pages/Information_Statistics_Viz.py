import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from picaso import information_content as ic
from picaso import justdoit as jdi
from picaso import justplotit as jpi
import os
import copy


st.set_page_config(page_title="Information Statistics Dashboard", layout="wide")

st.title("Information Statistics Dashboard")
st.markdown("""
This dashboard showcases information content statistics for your model. 
You can explore how different observation cases affect the Degrees of Freedom (DOF) and Shannon Information Content.
""")

# =======================================
# SIDEBAR INPUTS
# =======================================
st.sidebar.header("Parameters")

if 'cases' not in st.session_state:
    st.session_state['cases'] = [{
            'name': "Case 1",
            'method': 'Manual',
            'min_wave': 0.3,
            'max_wave': 2.0,
            'res': 100,
            'error': 0.01,
            'csv_file': None
        }]

def add_case():
    st.session_state['cases'].append({
        'name': f"Case {len(st.session_state['cases']) + 1}",
        'method': 'Manual',
        'min_wave': 0.3,
        'max_wave': 2.0,
        'res': 100,
        'error': 0.01,
        'csv_file': None
    })

st.sidebar.button("Add Case", on_click=add_case)

for i, case in enumerate(st.session_state['cases']):
    with st.sidebar.expander(f"{case['name']}", expanded=(i==len(st.session_state['cases'])-1)):
        case['name'] = st.text_input("Name", value=case['name'], key=f"name_{i}")
        case['method'] = st.selectbox("Input Method", ["Manual", "CSV Upload"], 
                                      index=0 if case['method'] == 'Manual' else 1, key=f"method_{i}")
        if case['method'] == 'Manual':
            case['min_wave'] = st.number_input("Min Wavelength (um)", value=float(case['min_wave']), key=f"min_{i}")
            case['max_wave'] = st.number_input("Max Wavelength (um)", value=float(case['max_wave']), key=f"max_{i}")
            case['res'] = st.number_input("Resolution (R)", value=int(case['res']), key=f"res_{i}")
            case['error'] = st.number_input("Spectral Error", value=float(case['error']), key=f"err_{i}", format="%.4f")
        else:
            case['csv_file'] = st.file_uploader("Upload CSV (wavelength, error)", type="csv", key=f"csv_{i}")
        
        if st.sidebar.button("Remove Case", key=f"remove_{i}"):
            st.session_state['cases'].pop(i)
            st.rerun()

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

    cases = st.session_state.get('cases', [])
    if not cases:
        st.warning("Please add at least one case in the sidebar.")
        st.stop()

    results_svd = []
    results_shannon = []
    case_names = []
    
    first_case_analyzer = None

    progress_bar = st.progress(0)
    for i, case in enumerate(cases):
        if case['method'] == 'Manual':
            new_wno = ic.create_grid(case['min_wave'], case['max_wave'], case['res'])
            analyzer = ic.Analyze(wno, jac_mat, case['error'], new_wno=new_wno)
        else:
            if case['csv_file'] is None:
                st.error(f"Please upload a CSV file for {case['name']}")
                st.stop()
            df_csv = pd.read_csv(case['csv_file'])
            if 'wavelength' not in df_csv.columns or 'error' not in df_csv.columns:
                st.error(f"CSV for {case['name']} must have 'wavelength' and 'error' columns.")
                st.stop()
            
            csv_wavelength = df_csv['wavelength'].values
            csv_error = df_csv['error'].values
            
            # ic.Analyze expects error on original wno grid for rebinning
            # wavelength typically increasing, wno decreasing. 
            # np.interp expects increasing xp
            xp = 1e4/csv_wavelength[::-1]
            fp = csv_error[::-1]
            if np.all(np.diff(wno) < 0):
                interpolated_error = np.interp(wno[::-1], xp, fp)[::-1]
            else:
                interpolated_error = np.interp(wno, xp, fp)
            
            new_wno = 1e4/csv_wavelength
            # Sort wno descending for ic.Analyze
            idx = np.argsort(new_wno)[::-1]
            new_wno = new_wno[idx]
            final_csv_error = csv_error[idx]

            analyzer = ic.Analyze(wno, jac_mat, interpolated_error, new_wno=new_wno)
            # Ensure the error used is exactly what was in the CSV
            analyzer.error = final_csv_error

        dfs = analyzer.degrees_of_freedom_svd()
        sic = analyzer.shannon_ic(priors)
        
        results_svd.append(dfs)
        results_shannon.append(sic)
        case_names.append(case['name'])
        
        if i == 0:
            first_case_analyzer = copy.copy(analyzer)
            lossH, lossCI = first_case_analyzer.loss_by_wave()
        
        progress_bar.progress((i + 1) / len(cases))

    # =======================================
    # VISUALIZATIONS
    # =======================================
    
    st.divider()
    
    # Only display fig1, fig4a, fig4b for the first case
    st.header(f"Reference Case Analysis: {case_names[0]}")
    
    # 1) Plot of their binned Jacobian values for their first case
    st.subheader(f"Binned Jacobian")
    
    fig1 = go.Figure()
    rebinned_wno = first_case_analyzer.new_wno if first_case_analyzer.new_wno is not None else wno
    
    for i, p_name in enumerate(params):
        y_val = np.array(first_case_analyzer.jacobian[i, :]).flatten()
        norm_y = np.abs(y_val) / np.max(np.abs(y_val)) if np.max(np.abs(y_val)) != 0 else y_val
        fig1.add_trace(go.Scatter(x=1e4/rebinned_wno, y=norm_y, name=p_name, mode='lines'))
    
    fig1.update_layout(
        xaxis_title="Wavelength [um]",
        yaxis_title="Normalized |Jacobian|",
        title=f"Binned Jacobian for {case_names[0]}",
        legend_title="Parameters"
    )
    st.plotly_chart(fig1, width='stretch')

    st.divider()
    # 1) Plot of their binned Jacobian values for their first case
    st.header(f"What wavelengths are most important? (Reference Case: {case_names[0]})")
    
    fig4a = go.Figure()
    fig4a.add_trace(go.Scatter(x=1e4/rebinned_wno, y=lossH, mode='lines', name='H loss'))
    fig4a.update_layout(
        xaxis_title="Wavelength [um]",
        yaxis_title="Delta IC/um",
        title="H loss vs. W"
    )
    st.plotly_chart(fig4a, width='stretch')

    fig4b = go.Figure()
    for i, p_name in enumerate(params):
        fig4b.add_trace(go.Scatter(x=1e4/rebinned_wno, y=np.array(lossCI)[:,i], mode='lines', name=p_name))
    fig4b.update_layout(
        xaxis_title="Wavelength [um]",
        yaxis_title="Delta Constraint Interval/um",
        title="Loss in 1-sigma Constraint Interval vs. W"
    )
    st.plotly_chart(fig4b, width='stretch')


    # 3) Plot of each of the shannon_ic dictionary outputs as a function of resolution
    st.divider()
    st.header("Comparison across Cases: Shannon Information Statistics")
    
    col1, col2 = st.columns(2)
    
    # DOF and H (Scalars)
    with col1:
        dofs = [r['DOF'] for r in results_shannon]
        
        fig3a = go.Figure()
        fig3a.add_trace(go.Scatter(x=case_names, y=dofs, mode='lines+markers', name='Shannon DOF', marker=dict(symbol='square', color='green')))
        fig3a.update_layout(
            xaxis_title="Case",
            yaxis_title="DOF",
            title="Shannon DOF"
        )
        st.plotly_chart(fig3a, width='stretch')
        
    with col2:
        hs = [r['H'] for r in results_shannon]
        
        fig3b = go.Figure()
        fig3b.add_trace(go.Scatter(x=case_names, y=hs, mode='lines+markers', name='Shannon Information (H)', marker=dict(symbol='triangle-up', color='red')))
        fig3b.update_layout(
            xaxis_title="Case",
            yaxis_title="H [bits]",
            title="Shannon Information (H)"
        )
        st.plotly_chart(fig3b, width='stretch')
        
    # Averaging Kernel and Constraint Interval (Vectors)
    st.subheader("Parameter-specific Shannon Stats Comparison")
    col3, col4 = st.columns(2)
    
    with col3:
        fig3c = go.Figure()
        for i, p_name in enumerate(params):
            ak_vals = [r['AveragingKernel'][i] for r in results_shannon]
            fig3c.add_trace(go.Scatter(x=case_names, y=ak_vals, mode='lines+markers', name=p_name, marker=dict(size=6)))
        fig3c.update_layout(
            xaxis_title="Case",
            yaxis_title="Averaging Kernel Diagonal",
            title="Averaging Kernel vs. Case"
        )
        st.plotly_chart(fig3c, width='stretch')

    with col4:
        fig3d = go.Figure()
        for i, p_name in enumerate(params):
            ci_vals = [r['constraint_interval'][i] for r in results_shannon]
            fig3d.add_trace(go.Scatter(x=case_names, y=ci_vals, mode='lines+markers', name=p_name, marker=dict(size=6)))
        fig3d.update_layout(
            xaxis_title="Case",
            yaxis_title="Constraint Interval",
            title="1-sigma Constraint Interval vs. Case"
        )
        st.plotly_chart(fig3d, width='stretch')
    
    # 2) Plot of their SVD degrees of freedom as a function of case
    st.divider()
    st.header("Comparison across Cases: SVD Degrees of Freedom")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=case_names, y=results_svd, mode='lines+markers', line=dict(color='blue'), marker=dict(symbol='circle')))
    fig2.update_layout(
        xaxis_title="Case",
        yaxis_title="SVD DFS",
        title="SVD Degrees of Freedom for Signal"
    )
    st.plotly_chart(fig2, width='stretch')    
    

else:
    st.warning("Please initialize example or provide Jacobian data to begin.")
