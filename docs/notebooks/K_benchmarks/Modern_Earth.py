# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Imports (wont need all these, clean up what is unneeded)
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import pickle
import astropy.units as u
from picaso import justdoit as jdi 
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

# %% [markdown]
# First, lets define the paths to the rocky planet opacity databases. If you don't have them downloaded already you can find them [here](https://zenodo.org/records/17381172). Make sure to change the filepath to the correct filepath for your download. There are currently two different databases the differences of which are discussed in the link to the download. For now we'll name them opacities_reflected_light and opacities_thermal_emission since that what we'll use each for but there is nothing inherently limiting each database for either use. 

# %%
opacities_reflected_light = jdi.opannection(wave_range=[0.1,5.5], # <- define the wavelength range we want for this database
                          filename_db='/Users/jaden/Work/Research/LAMAT/picaso/opacity_databases/better/opacities_photochem_0.1_5.5_R60000.db')
opacities_thermal_emission = jdi.opannection(wave_range=[0.1,250.0], 
                          filename_db='/Users/jaden/Work/Research/LAMAT/picaso/opacity_databases/better/opacities_photochem_0.1_250.0_R15000.db')


# %% [markdown]
# Next, let's define the functions needed to calculate Earth spectra. Since we aim to recreate the calculations done in [Robinson & Salvador 2023](https://ui.adsabs.harvard.edu/abs/2023PSJ.....4...10R/abstract), we take the ICRCCM for our pressure-temperature profile and mixing ratios as they do.

# %%
def read_atm_file(filename):
    """
    Read ICRCCM .atm file format
    
    Parameters:
    -----------
    filename : str
        Path to .atm file
    
    Returns:
    --------
    df : DataFrame
        Atmospheric profile with columns: pressure, temperature, H2O, CO2, O3, N2O, CO, CH4, O2, N2
    """
    # Read file and find where data starts
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find the header line with column names
    header_idx = None
    for i, line in enumerate(lines):
        if 'p(Pa)' in line and 'T(K)' in line:
            header_idx = i
            break
    
    if header_idx is None:
        raise ValueError("Could not find data header in .atm file")
    
    # Parse data starting from next line after header
    data = []
    for line in lines[header_idx + 1:]:
        # Skip empty lines
        if not line.strip():
            continue
        # Parse values (handle scientific notation)
        values = line.split()
        if len(values) >= 10:  # Should have p, T, and 8 species
            data.append([float(v) for v in values])
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=[
        'pressure_Pa', 'temperature', 
        'H2O', 'CO2', 'O3', 'N2O', 'CO', 'CH4', 'O2', 'N2'
    ])
    
    # Convert pressure from Pa to bars for PICASO
    df['pressure'] = df['pressure_Pa'] / 1e5
    
    # Drop the Pa column
    df = df.drop('pressure_Pa', axis=1)
    
    # Reorder columns to put pressure and temperature first
    cols = ['pressure', 'temperature', 'H2O', 'CO2', 'O3', 'N2O', 'CO', 'CH4', 'O2', 'N2']
    df = df[cols]
    
    return df



# %% [markdown]
# Next, we define our Earth function parameters.

# %%
def earth_spectrum(opacity, atm_file, atmosphere_kwargs={}):
    """
    Calculate Earth's reflected light spectrum.
    
    Parameters
    ----------
    opacity : picaso.opannection
        Opacity database object
    atm_file : str
        Path to .atm file containing atmospheric profile
    atmosphere_kwargs : dict, optional
        Optional atmospheric modifications. Supports 'exclude_mol' to remove 
        specific molecular species (e.g., {'exclude_mol': ['H2O']})
    
    Returns
    -------
    wno : numpy.ndarray
        Wavenumber grid in cm⁻¹
    fpfs : numpy.ndarray
        Planet-to-star flux ratio
    albedo : numpy.ndarray
        Geometric albedo
    """
    earth = jdi.inputs()
    
    # Phase angle 
    earth.phase_angle(0) #radians
    
    # Define gravity
    earth.gravity(radius=1, radius_unit=jdi.u.Unit('R_earth'),
                 mass=1, mass_unit=jdi.u.Unit('M_earth')) #any astropy units available
    earth.approx(raman="none")
    
    # Define star 
    earth.star(opacity,5778,0,4.0,semi_major=1, radius=1, radius_unit=jdi.u.Unit('R_sun'), 
               semi_major_unit=u.Unit('au')) 
    # Define our P-T Composition
    df_atmo_earth = read_atm_file(atm_file)
   
    # Apply molecular exclusions if requested
    if 'exclude_mol' in atmosphere_kwargs:
        sp = atmosphere_kwargs['exclude_mol'][0]
        if sp in df_atmo_earth:
            df_atmo_earth[sp] *= 0
    earth.atmosphere(df=df_atmo_earth)
    earth.surface_reflect(0.1,opacity.wno)
    # Make a cloud free spectrum
    df_cldfree = earth.spectrum(opacity,calculation='reflected',full_output=True)
    # Define clouds (won't be used here, should I keep this in or update to use clouds?)
    ptop = 0.6
    pbot = 0.7
    logdp = np.log10(pbot) - np.log10(ptop)  
    log_pbot = np.log10(pbot)
    earth.clouds(w0=[0.99], g0=[0.85], 
                 p = [log_pbot], dp = [logdp], opd=[10])
    # Cloud spectrum
    df_cld = earth.spectrum(opacity,full_output=True)
    # Average the two spectra and choose resolution
    wno, alb, fpfs = df_cldfree['wavenumber'],df_cldfree['albedo'],df_cldfree['fpfs_reflected']
    wno_c, alb_c, fpfs_c = df_cld['wavenumber'],df_cld['albedo'],df_cld['fpfs_reflected']
    _, albedo = jdi.mean_regrid(wno, 1.*alb + 0.*alb_c, R=140) # <- cloud free
    wno, fpfs = jdi.mean_regrid(wno, 1.*fpfs + 0.*fpfs_c,R=140) # <- cloud free
    return wno, fpfs, albedo

def make_case(opacity, atm_file):
    """
    Generate reflected light spectra for molecular contribution analysis.
    
    Creates spectra with individual molecules systematically removed to 
    quantify each species' contribution to the total spectrum.
    
    Parameters
    ----------
    opacity : picaso.opannection
        Opacity database object
    atm_file : str
        Path to .atm file containing atmospheric profile
    
    Returns
    -------
    dict
        Dictionary with keys 'all' (full spectrum) and molecular species names.
        Each value is a tuple of (wno, fpfs, albedo).
    """
    # Molecular species to consider
    species = ['H2O', 'CO2', 'O3', 'N2O', 'CO', 'CH4', 'O2', 'N2']

    # Calculate the spectrum(s) in a dictionary, calculating exclusions to find molecular contribution as well
    res = {}

    res['all'] = earth_spectrum(opacity, atm_file)
    for sp in species: # <- can be commented out for quicker calculations 
        res[sp] = earth_spectrum(opacity, atm_file, atmosphere_kwargs={'exclude_mol': [sp]}) # <- can be commented out for a quicker calculation

    return res


# %% [markdown]
# Now, we can calculate the spectra according to the parameters in our functions above. As mentioned above the exclusionary spectrums we calculate that find the molecular contribution can be commented out for a quicker calculation that just finds the total spec

# %%
res = make_case(opacities_reflected_light, 
                '/Users/jaden/Work/Research/LAMAT/picaso/reference/earth/tyrobinson_earth_PT_and_mixing.atm')

# %% [markdown]
# We can plot this spectrum stand alone as such.

# %%
fig, ax = plt.subplots(1, 1, figsize=[8, 5])
wno, fpfs, albedo = res['all']
ax.plot(1e4/wno, albedo, c='k', lw=2)

# Plot molecular contributions if calculated
for key in res:
    if key == 'all':
        continue
    _, fpfs1, albedo1 = res[key]
    ax.fill_between(1e4/wno, albedo, albedo1, label=key, alpha=0.5)

ax.set_xlim(0.2, 2.)
ax.set_ylim(0., 0.4)
ax.set_ylabel('Albedo')
ax.set_xlabel('Wavelength (microns)')
ax.legend()
plt.title('Earth in Reflected Light (clearsky)')
plt.show()

# %% [markdown]
# Now, we can load the rfast calculation to compare, this is stored in a .pkl file requiring the pickle import which we have above and can be found [here](placeholder). (There is no link to get to this yet, im unsure how i should imbed this in the tutorial)

# %%
with open('/Users/jaden/Work/Research/LAMAT/rfast/modernearth_rfast.pkl', 'rb') as f:
    F1_rfast, F2_rfast, lam_rfast = pickle.load(f)

# %% [markdown]
# PICASO and rfast use a slightly differen wavelength grid for calculations so it is important for a true comparison to regrid the spectra to a common grid here. (Should I give the data regridded already or should I keep this cell if its instructional?)

# %%
# Bin rfast data to same resolution as PICASO (R=140)
# Get PICASO wavelength grid
wno_picaso, fpfs_picaso, albedo_picaso = res['all']
wavelength_picaso = 1e4 / wno_picaso  # Convert to um

# Bin rfast data to R=140 using PICASO's mean_regrid function
# First convert rfast wavelength to wavenumber
wno_rfast = 1e4 / lam_rfast
#Sort in increasing wavenumber order (mean_regrid expects this)
sort_idx = np.argsort(wno_rfast)
wno_rfast_sorted = wno_rfast[sort_idx]
F1_rfast_sorted = F1_rfast[sort_idx]

# Regrid to R=140
wno_rfast_binned, F1_rfast_binned = jdi.mean_regrid(wno_rfast_sorted, F1_rfast_sorted, R=140)
wavelength_rfast_binned = 1e4 / wno_rfast_binned

print(f"Binned rfast wavelength range: {wavelength_rfast_binned.min():.3f} - {wavelength_rfast_binned.max():.3f} μm")
print(f"Number of binned points: {len(wavelength_rfast_binned)}")


# Define a common wavenumber grid
# Use the overlap region between rfast and PICASO
wl_min = max(wavelength_rfast_binned.min(), wavelength_picaso.min())
wl_max = min(wavelength_rfast_binned.max(), wavelength_picaso.max())

print(f"Common wavelength range: {wl_min:.4f} - {wl_max:.4f} μm")

# Create common wavenumber grid at R=140
# Start with PICASO's grid in the overlap region as template
mask_overlap = (wavelength_picaso >= wl_min) & (wavelength_picaso <= wl_max)
wno_common = wno_picaso[mask_overlap]

print(f"Common grid: {len(wno_common)} wavenumber points")

# Regrid PICASO albedo onto common grid
_, albedo_picaso_common = jdi.mean_regrid(wno_picaso, albedo_picaso, newx=wno_common)

# Regrid rfast albedo onto common grid
# First need rfast in wavenumber space
wno_rfast_sorted = np.sort(wno_rfast_binned)  # Make sure it's sorted
alb_rfast_sorted = F1_rfast_binned[np.argsort(wno_rfast_binned)]

_, albedo_rfast_common = jdi.mean_regrid(wno_rfast_sorted, alb_rfast_sorted, newx=wno_common)

# Convert common wavenumber grid back to wavelength for plotting
wavelength_common = 1e4 / wno_common

print(f"After regridding:")
print(f"Common wavelength points: {len(wavelength_common)}")
print(f"PICASO on common grid: {len(albedo_picaso_common)}")
print(f"rfast on common grid: {len(albedo_rfast_common)}")

# %% [markdown]
# Now we can plot the two calculations, from PICASO and rfast, and compare.

# %%
# Get original PICASO data
wno_picaso_full, fpfs, albedo_picaso_full = res['all']

# Regrid 'all' case (already done)
_, albedo_picaso_common = jdi.mean_regrid(wno_picaso_full, albedo_picaso_full, newx=wno_common)

# Regrid molecular exclusion cases
res_regridded = {}
res_regridded['all'] = albedo_picaso_common

for key in res:
    if key == 'all':
        continue
    _, fpfs1, albedo1 = res[key]
    _, albedo1_common = jdi.mean_regrid(wno_picaso_full, albedo1, newx=wno_common)
    res_regridded[key] = albedo1_common

# Now plot everything on common grid
wavelength_common = 1e4 / wno_common

fig, ax = plt.subplots(1, 1, figsize=[8, 5])

# Plot molecular contributions (all on common grid now)
for key in res_regridded:
    if key == 'all':
        continue
    albedo1_common = res_regridded[key]
    ax.fill_between(wavelength_common, albedo_picaso_common, albedo1_common, 
                    label=key, alpha=0.5)

# Plot total spectra
ax.plot(wavelength_common, albedo_picaso_common, '-', c='black', lw=2, 
        label='PICASO', zorder=10)
ax.plot(wavelength_common, albedo_rfast_common, '-', c='red', lw=2, 
        label='rfast', alpha=0.7, zorder=10)

ax.set_xlim(0.2, 2)
ax.set_ylim(0, 0.275)
ax.set_ylabel('Geometric Albedo')
ax.set_xlabel('Wavelength (μm)')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.set_title('Earth in Reflected Light, rfast Comparison (clearsky)')
plt.tight_layout()
plt.show()


# %% [markdown]
# We can perform the same exercise for Earth's thermal emission spectrum, starting here from defining functions.

# %%
def earth_spectrum_thermal(opacity_thermal, atm_file, atmosphere_kwargs={}):
    """
    Calculate Earth's thermal emission spectrum.
    
    Parameters
    ----------
    opacity_thermal : picaso.opannection
        Opacity database object for thermal wavelengths (typically 4-25 μm)
    atm_file : str
        Path to .atm file containing atmospheric profile
    atmosphere_kwargs : dict, optional
        Optional atmospheric modifications. Supports 'exclude_mol' to remove 
        specific molecular species (e.g., {'exclude_mol': ['H2O']})
    
    Returns
    -------
    wno : numpy.ndarray
        Wavenumber grid in cm⁻¹
    fpfs : numpy.ndarray
        Planet-to-star flux ratio
    fp : numpy.ndarray
        Planet thermal flux in erg/s/cm² per cm of wavelength
    """
    
    earth = jdi.inputs()
    
    # Phase angle 
    earth.phase_angle(0)
    
    # Define gravity
    earth.gravity(radius=1, radius_unit=jdi.u.Unit('R_earth'),
                 mass=1, mass_unit=jdi.u.Unit('M_earth'))
    earth.approx(raman="none")
    
    # Define star 
    earth.star(opacity_thermal, 5778, 0, 4.44, semi_major=1, radius=1, 
               radius_unit=jdi.u.Unit('R_sun'), 
               semi_major_unit=u.Unit('au'))
    
    # P-T-Composition
    df_atmo_earth = read_atm_file(atm_file)
    
    if 'exclude_mol' in atmosphere_kwargs:
        sp = atmosphere_kwargs['exclude_mol'][0]
        if sp in df_atmo_earth:
            df_atmo_earth[sp] *= 0
    
    earth.atmosphere(df=df_atmo_earth)
    earth.surface_reflect(0.0,opacity_thermal.wno)  # Albedo is set to 0.0, and thus the emissivity is set to 1
    
    # Thermal spectrum
    df_thermal = earth.spectrum(opacity_thermal, calculation='thermal', full_output=True)
    
    # Extract outputs
    wno = df_thermal['wavenumber']
    fpfs = df_thermal['fpfs_thermal']
    fp = df_thermal['thermal']
    
    # Regrid
    wno, fp = jdi.mean_regrid(wno, fp, R=140)
    wno, fpfs = jdi.mean_regrid(df_thermal['wavenumber'], fpfs, R=140)
    
    return wno, fpfs, fp

def make_case_thermal(opacity, atm_file):
    """
    Generate thermal emission spectra for molecular contribution analysis.
    
    Creates thermal spectra with individual molecules systematically removed 
    to quantify each species' contribution to the total spectrum.
    
    Parameters
    ----------
    opacity_thermal : picaso.opannection
        Opacity database object for thermal wavelengths
    atm_file : str
        Path to .atm file containing atmospheric profile
    
    Returns
    -------
    dict
        Dictionary with keys 'all' (full spectrum) and molecular species names.
        Each value is a tuple of (wno, fpfs, fp).
    """
    # Molecular species to consider
    species = ['O2', 'H2O', 'CO2', 'O3', 'CH4', 'N2', 'N2O', 'CO'] 

    # Calculate the spectrum(s) in a dictionary, calculating exclusions to find molecular contribution as well
    res = {}
    
    res['all'] = earth_spectrum_thermal(opacity, atm_file)
    for sp in species:
        res[sp] = earth_spectrum_thermal(opacity, atm_file,
                                                atmosphere_kwargs={'exclude_mol': [sp]})
    
    return res


# %% [markdown]
# Calculate the thermal emission case.

# %%
res_fix_thermal = make_case_thermal(opacities_thermal_emission,
                                    '/Users/jaden/Work/Research/LAMAT/picaso/reference/earth/tyrobinson_earth_PT_and_mixing.atm')

# %% [markdown]
# This can be plotted standalone like this.

# %%
fig, ax = plt.subplots(1, 1, figsize=[8, 5])
wno_thermal, fpfs_thermal, fp_thermal = res_fix_thermal['all']
fp_thermal_converted = fp_thermal * 1e-7

ax.plot(1e4/wno_thermal, fp_thermal_converted, c='k', lw=2)

# Plot molecular contributions if calculated
for key in res_fix_thermal:
    if key == 'all':
        continue
    _, fpfs1, fp1 = res_fix_thermal[key]
    fp1_converted = fp1 * 1e-7
    ax.fill_between(1e4/wno_thermal, fp_thermal_converted, fp1_converted, label=key, alpha=0.5)

ax.set_xlim(5, 25)
ax.set_ylim(0, 30)
ax.set_ylabel('Thermal Flux (W m⁻² μm⁻¹)')
ax.set_xlabel('Wavelength (microns)')
ax.legend()
plt.title('Earth in Thermal Emission')
plt.show()

# %% [markdown]
# Once again, lets load and regrid the rfast calculation but now for Earth's thermal emission.

# %%
# Bin rfast thermal data to same resolution as PICASO (R=140)
# Get PICASO wavelength grid
wno_picaso_thermal, fpfs_picaso_thermal, fp_picaso_thermal = res_fix_thermal['all']
wavelength_picaso_thermal = 1e4 / wno_picaso_thermal  # Convert to um

# Load rfast thermal data
with open('/Users/jaden/Work/Research/LAMAT/rfast/modernearth_rfast_thermal.pkl', 'rb') as f:
    F1_rfast_thermal, F2_rfast_thermal, lam_rfast_thermal = pickle.load(f)

# Bin rfast data to R=140 using PICASO's mean_regrid function
# First convert rfast wavelength to wavenumber
wno_rfast_thermal = 1e4 / lam_rfast_thermal

# Sort in increasing wavenumber order (mean_regrid expects this)
sort_idx = np.argsort(wno_rfast_thermal)
wno_rfast_thermal_sorted = wno_rfast_thermal[sort_idx]
F2_rfast_thermal_sorted = F2_rfast_thermal[sort_idx]

# Regrid to R=140
wno_rfast_thermal_binned, F2_rfast_thermal_binned = jdi.mean_regrid(
    wno_rfast_thermal_sorted, F2_rfast_thermal_sorted, R=140
)
wavelength_rfast_thermal_binned = 1e4 / wno_rfast_thermal_binned

print(f"Binned rfast wavelength range: {wavelength_rfast_thermal_binned.min():.3f} - {wavelength_rfast_thermal_binned.max():.3f} μm")
print(f"Number of binned points: {len(wavelength_rfast_thermal_binned)}")

# Define a common wavenumber grid
# Use the overlap region between rfast and PICASO
wl_min = max(wavelength_rfast_thermal_binned.min(), wavelength_picaso_thermal.min())
wl_max = min(wavelength_rfast_thermal_binned.max(), wavelength_picaso_thermal.max())
print(f"Common wavelength range: {wl_min:.4f} - {wl_max:.4f} μm")

# Create common wavenumber grid at R=140
# Start with PICASO's grid in the overlap region as template
mask_overlap = (wavelength_picaso_thermal >= wl_min) & (wavelength_picaso_thermal <= wl_max)
wno_common_thermal = wno_picaso_thermal[mask_overlap]
print(f"Common grid: {len(wno_common_thermal)} wavenumber points")

# Regrid PICASO fpfs onto common grid
_, fpfs_picaso_thermal_common = jdi.mean_regrid(wno_picaso_thermal, fpfs_picaso_thermal, newx=wno_common_thermal)

# Regrid rfast F2 onto common grid
# First need rfast in wavenumber space
wno_rfast_thermal_sorted = np.sort(wno_rfast_thermal_binned)  # Make sure it's sorted
F2_rfast_thermal_sorted = F2_rfast_thermal_binned[np.argsort(wno_rfast_thermal_binned)]
_, F2_rfast_thermal_common = jdi.mean_regrid(wno_rfast_thermal_sorted, F2_rfast_thermal_sorted, newx=wno_common_thermal)

# Convert common wavenumber grid back to wavelength for plotting
wavelength_common_thermal = 1e4 / wno_common_thermal

print(f"After regridding:")
print(f"Common wavelength points: {len(wavelength_common_thermal)}")
print(f"PICASO on common grid: {len(fpfs_picaso_thermal_common)}")
print(f"rfast on common grid: {len(F2_rfast_thermal_common)}")

# %%
# Get original PICASO thermal data
wno_thermal_full, fpfs_thermal, fp_thermal_full = res_fix_thermal['all']
fp_thermal_full_converted = fp_thermal_full * 1e-7

# Regrid 'all' case to common grid
_, fp_picaso_common_thermal = jdi.mean_regrid(wno_thermal_full, fp_thermal_full_converted, 
                                               newx=wno_common_thermal)

# Regrid molecular exclusion cases
res_thermal_regridded = {}
res_thermal_regridded['all'] = fp_picaso_common_thermal

for key in res_fix_thermal:
    if key == 'all':
        continue
    _, fpfs1, fp1 = res_fix_thermal[key]
    fp1_converted = fp1 * 1e-7
    _, fp1_common = jdi.mean_regrid(wno_thermal_full, fp1_converted, newx=wno_common_thermal)
    res_thermal_regridded[key] = fp1_common

# Now plot everything on common grid
wavelength_common_thermal = 1e4 / wno_common_thermal

fig, ax = plt.subplots(1, 1, figsize=[8, 5])

# Plot molecular contributions (all on common grid now)
for key in res_thermal_regridded:
    if key == 'all':
        continue
    fp1_common = res_thermal_regridded[key]
    ax.fill_between(wavelength_common_thermal, fp_picaso_common_thermal, fp1_common, 
                    label=key, alpha=0.5)

# Plot total spectra
ax.plot(wavelength_common_thermal, fp_picaso_common_thermal, '-', c='black', lw=2,
        label='PICASO', zorder=10)
ax.plot(wavelength_common_thermal, F2_rfast_thermal_common, '-', c='red', lw=2,
        label='rfast', alpha=0.7, zorder=10)

ax.set_xlim(5, 25)
ax.set_ylim(0, 30)
ax.set_ylabel('Thermal Flux (W m⁻² μm⁻¹)')
ax.set_xlabel('Wavelength (μm)')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.set_title('Earth in Thermal Emission')
plt.tight_layout()
plt.show()

# %% [markdown]
# More examples can be found [here](zenodo_will_make_later). (make zenodo link work later)

# %%
