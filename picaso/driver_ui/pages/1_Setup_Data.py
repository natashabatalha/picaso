import streamlit as st
import os
import picaso.data as data

st.logo('https://natashabatalha.github.io/picaso/_images/logo.png', size="large", link="https://github.com/natashabatalha/picaso")

st.header('Setup Data for PICASO')

# 1. Environment Variable Handling
picaso_refdata = os.environ.get('picaso_refdata', '')

if not picaso_refdata or not os.path.isdir(picaso_refdata):
    if not picaso_refdata:
        st.warning('The `picaso_refdata` environment variable is not set.')
    else:
        st.error(f'The path `{picaso_refdata}` does not appear to be a valid directory.')
        
    user_input = st.text_input('Enter the path to your picaso reference data directory:', value=picaso_refdata)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Set Environment Variable'):
            if user_input and os.path.isdir(user_input):
                os.environ['picaso_refdata'] = user_input
                st.success(f'picaso_refdata set to {user_input}')
                st.rerun()
            else:
                st.error('Please enter a valid directory path.')
    with col2:
        if st.button('Kill and Restart App'):
            st.write('Killing app...')
            os._exit(0)
    st.stop()

# 2. Display Environment Status
st.subheader('Current Environment Status')
html_output = data.check_environ(return_html=True)
st.html(html_output)

# 3. List Available Data Products
st.subheader('Available Data Downloads')
data_config = {}
try:
    input_config, data_config = data.get_data_config()
except Exception as e:
    st.error(f"Error loading data configuration: {e}")

for category, targets in data_config.items():
    with st.expander(f"Category: {category.replace('_', ' ').title()}", expanded=False):
        for target_name, info in targets.items():
            st.markdown(f"### {target_name}")
            st.write(info['description'])
            
            # Download Workflow
            if st.button(f"Download {target_name}", key=f"btn_{category}_{target_name}"):
                st.session_state[f"download_{category}_{target_name}"] = True
            
            if st.session_state.get(f"download_{category}_{target_name}"):
                st.write("---")
                st.info(f"Setting up download for **{target_name}**")
                
                default_dest = info.get('default_destination', 'cwd')
                if default_dest == 'cwd':
                    default_dest = os.getcwd()
                
                dest_option = st.radio(
                    "Select Destination:",
                    ("Default Destination", "Custom Destination"),
                    key=f"dest_opt_{category}_{target_name}"
                )
                
                final_dest = default_dest
                if dest_option == "Custom Destination":
                    final_dest = st.text_input("Enter custom path:", value=os.getcwd(), key=f"custom_path_{category}_{target_name}")
                else:
                    st.write(f"Default destination: `{default_dest}`")

                if st.button("Confirm Download", key=f"confirm_{category}_{target_name}"):
                    if os.path.isdir(final_dest) or dest_option == "Default Destination":
                        with st.spinner(f"Downloading {target_name}..."):
                            try:
                                # We need to handle cases where the folder might need to be created
                                if not os.path.isdir(final_dest) and dest_option == "Default Destination":
                                    os.makedirs(final_dest, exist_ok=True)
                                
                                data.get_data(category_download=category, target_download=target_name, final_destination_dir=final_dest)
                                st.success(f"Successfully downloaded {target_name} to {final_dest}")
                                del st.session_state[f"download_{category}_{target_name}"]
                                st.rerun()
                            except Exception as e:
                                st.error(f"Download failed: {e}")
                    else:
                        st.error("Invalid destination directory.")
                
                if st.button("Cancel", key=f"cancel_{category}_{target_name}"):
                    del st.session_state[f"download_{category}_{target_name}"]
                    st.rerun()
