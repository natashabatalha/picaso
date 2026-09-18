import streamlit as st

st.set_page_config(page_title="PICASO UI", layout="wide")

st.logo('https://natashabatalha.github.io/picaso/_images/logo.png', size="large", link="https://github.com/natashabatalha/picaso")

st.title("Welcome to the PICASO App")

st.markdown("""
**PICASO** (Planetary Intensity Code for Atmospheric Scattering Observations) is a state-of-the-art tool for exoplanet and brown dwarf atmospheric modeling. 

picaso enables the: 
1. **Computation** of exoplanet and brown dwarf spectroscopy in transmission, emission or reflected light.
2. **1D climate modeling** of brown dwarfs and exoplanets.
3. **Fitting** spectroscopic data to models.
""")

st.header("Major Features")
st.markdown("""
* **Reflected Light Spectroscopy** of planets
* **Thermal Emission Spectroscopy** of planets and brown dwarfs
* **Transit Spectroscopy** of planets
* **Phase Curves** of planets
* **1D Climate modeling** of planets and brown dwarfs
* **Fitting models** to data
""")

st.header("App Major Features")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. Setup PICASO Reference Data")
    st.write("Configure your environment variables and download necessary reference data (opacities, etc.) to get PICASO running.")

with col2:
    st.subheader("2. Run Spectral Model and/or Build Retrieval")
    st.write("The main driver for running PICASO models. Configure your star, planet, atmosphere, and clouds to generate spectra or run retrievals.")

with col3:
    st.subheader("3. Compute Information Content Statistics")
    st.write("Analyze information content, Degrees of Freedom (DOF), and Shannon Information Content for your model configurations.")

st.divider()

st.header("General PICASO Resources")
col_res1, col_res2, col_res3, col_res4 = st.columns(4)
with col_res1:
    st.link_button("Documentation", "https://natashabatalha.github.io/picaso")
with col_res2:
    st.link_button("GitHub Repository", "https://github.com/natashabatalha/picaso")
with col_res3:
    st.link_button("Installation Guide", "https://natashabatalha.github.io/picaso/installation.html")
with col_res4:
    st.link_button("Tutorials", "https://natashabatalha.github.io/picaso/tutorials.html")
