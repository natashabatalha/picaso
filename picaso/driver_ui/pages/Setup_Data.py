import streamlit as st
from picaso.data import check_environ, get_data
st.logo('https://natashabatalha.github.io/picaso/_images/logo.png', size="large", link="https://github.com/natashabatalha/picaso")

st.header('Setup Data for PICASO')

if st.button('Check readiness of current environment'):
    html_output = check_environ()
    st.html(html_output)
if st.button('Download data helper'):
    get_data(is_ui=True)
# could ask what you would like to do
# --> just run spectrum
# --> do you want clouds?
# --> can download everything just in case you want more options
# --> can always download more later
# --> use data.py