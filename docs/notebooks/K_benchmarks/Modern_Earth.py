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
# Imports
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import os
import astropy.units as u
from picaso import justdoit as jdi 
from matplotlib import pyplot as plt

# %% [markdown]
# First, lets define the paths to the rocky planet opacity databases. If you don't have them downloaded already you can find them [here](https://zenodo.org/records/17381172) and place them in `os.path.join(jdi.__refdata__, 'opacities')`. You can check where this points on your system by running `print(jdi.__refdata__).` There are currently two different databases the differences of which are discussed in the link to the download. For now we'll name them opacities_reflected_light and opacities_thermal_emission since that what we'll use each for but there is nothing inherently limiting each database for either use. 

# %%
opacities_reflected_light = jdi.opannection(wave_range=[0.1,5.5], # <- define the wavelength range we want for this database
                          filename_db=os.path.join(jdi.__refdata__, 'opacities', 'opacities_photochem_0.1_5.5_R60000.db'))
opacities_thermal_emission = jdi.opannection(wave_range=[0.1,250.0], 
                          filename_db=os.path.join(jdi.__refdata__, 'opacities', 'opacities_photochem_0.1_250.0_R15000.db'))

# %% [markdown]
#  Next, let's define our atmospheric profile. We embed the ICRCCM mid-latitude summer sounding directly as numpy arrays, following [Robinson & Salvador 2023](https://ui.adsabs.harvard.edu/abs/2023PSJ.....4...10R/abstract).

# %%
pressure = np.array([1.0000e-07, 1.2630e-05, 2.3330e-05, 4.3110e-05, 7.9670e-05,
       1.0830e-04, 2.0010e-04, 2.7600e-04, 3.8714e-04, 5.4305e-04,
       7.6174e-04, 1.0685e-03, 1.4988e-03, 2.1024e-03, 2.9490e-03,
       4.1366e-03, 5.8025e-03, 8.1392e-03, 1.1417e-02, 1.6015e-02,
       2.2464e-02, 3.1511e-02, 4.4200e-02, 6.2000e-02, 8.5775e-02,
       1.0955e-01, 1.3333e-01, 1.5713e-01, 1.8088e-01, 2.0467e-01,
       2.2842e-01, 2.5220e-01, 2.7598e-01, 2.9975e-01, 3.2352e-01,
       3.4730e-01, 3.7108e-01, 3.9485e-01, 4.1862e-01, 4.4240e-01,
       4.6618e-01, 4.8995e-01, 5.1372e-01, 5.3750e-01, 5.6128e-01,
       5.8505e-01, 6.0882e-01, 6.3260e-01, 6.5638e-01, 6.8015e-01,
       7.0392e-01, 7.2770e-01, 7.5148e-01, 7.7525e-01, 7.9902e-01,
       8.2280e-01, 8.4658e-01, 8.7035e-01, 8.9412e-01, 9.1790e-01,
       9.4168e-01, 9.6545e-01, 9.8923e-01, 1.0130e+00])
temperature = np.array([214.7 , 214.8 , 215.8 , 216.8 , 222.4 , 228.5 , 241.5 , 248.03,
       255.67, 263.53, 271.52, 276.26, 272.5 , 266.37, 259.83, 253.39,
       247.23, 241.74, 236.46, 231.31, 226.44, 223.16, 220.49, 217.86,
       215.82, 215.74, 215.76, 215.81, 217.12, 221.44, 226.1 , 230.42,
       234.43, 238.17, 241.67, 244.97, 248.08, 251.03, 253.82, 256.45,
       258.93, 261.28, 263.51, 265.64, 267.68, 269.65, 271.56, 273.4 ,
       275.18, 276.91, 278.6 , 280.23, 281.82, 283.35, 284.83, 286.22,
       287.49, 288.62, 289.64, 290.58, 291.48, 292.34, 293.18, 294.  ])
H2O = np.array([4.0709e-06, 4.0709e-06, 4.0709e-06, 4.0709e-06, 4.0709e-06,
       4.0709e-06, 4.0709e-06, 4.0709e-06, 4.0709e-06, 4.0629e-06,
       4.0799e-06, 4.0507e-06, 4.0472e-06, 4.0565e-06, 4.0541e-06,
       4.0458e-06, 4.0202e-06, 4.0284e-06, 4.0651e-06, 4.0447e-06,
       4.0367e-06, 4.0347e-06, 4.0201e-06, 4.0244e-06, 4.0430e-06,
       4.0225e-06, 3.9623e-06, 4.3885e-06, 8.0169e-06, 1.9200e-05,
       4.1910e-05, 8.1398e-05, 1.3832e-04, 1.9636e-04, 2.5634e-04,
       3.2534e-04, 4.0545e-04, 4.9752e-04, 5.9968e-04, 7.0699e-04,
       8.1775e-04, 9.5531e-04, 1.1062e-03, 1.2628e-03, 1.4820e-03,
       1.7632e-03, 2.0838e-03, 2.4571e-03, 2.8558e-03, 3.2747e-03,
       3.7347e-03, 4.2126e-03, 4.7063e-03, 5.2331e-03, 5.8731e-03,
       6.5425e-03, 7.1289e-03, 7.7335e-03, 8.4007e-03, 9.0715e-03,
       9.6908e-03, 1.0299e-02, 1.0943e-02, 1.1600e-02])
CO2 = np.array([0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005,
       0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005,
       0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005,
       0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005,
       0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005,
       0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005,
       0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005,
       0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005])
O3 = np.array([7.7730e-07, 7.7730e-07, 7.8860e-07, 7.9660e-07, 1.1670e-06,
       1.4000e-06, 1.8000e-06, 2.2300e-06, 2.6313e-06, 3.2056e-06,
       4.0586e-06, 5.1604e-06, 6.7053e-06, 8.8747e-06, 1.1467e-05,
       1.3661e-05, 1.4482e-05, 1.3926e-05, 1.2564e-05, 1.0842e-05,
       9.0840e-06, 6.9831e-06, 4.8445e-06, 3.0733e-06, 1.6504e-06,
       1.0329e-06, 8.1811e-07, 6.6369e-07, 5.0885e-07, 3.9290e-07,
       3.2822e-07, 2.7628e-07, 2.3158e-07, 2.0252e-07, 1.8508e-07,
       1.6862e-07, 1.5344e-07, 1.4149e-07, 1.3086e-07, 1.2112e-07,
       1.1179e-07, 1.0547e-07, 1.0033e-07, 9.4776e-08, 9.0825e-08,
       8.7034e-08, 8.3000e-08, 7.9949e-08, 7.6969e-08, 7.3685e-08,
       7.0714e-08, 6.7987e-08, 6.5374e-08, 6.2877e-08, 6.1490e-08,
       6.0649e-08, 5.9085e-08, 5.7476e-08, 5.6196e-08, 5.5095e-08,
       5.3861e-08, 5.2508e-08, 5.1215e-08, 4.9970e-08])
N2O = np.array([2.1790e-09, 2.1790e-09, 2.1790e-09, 2.1790e-09, 2.1790e-09,
       2.1790e-09, 2.1790e-09, 2.1790e-09, 2.6658e-09, 3.3151e-09,
       4.2994e-09, 6.3210e-09, 9.9275e-09, 1.5305e-08, 2.3355e-08,
       3.4650e-08, 4.9657e-08, 6.8604e-08, 9.1051e-08, 1.1221e-07,
       1.2888e-07, 1.4762e-07, 1.8476e-07, 2.5507e-07, 3.4128e-07,
       3.9165e-07, 4.0902e-07, 4.2454e-07, 4.3984e-07, 4.4774e-07,
       4.5442e-07, 4.6287e-07, 4.7278e-07, 4.7722e-07, 4.8350e-07,
       4.8742e-07, 4.8794e-07, 4.8741e-07, 4.8459e-07, 4.8276e-07,
       4.8021e-07, 4.8496e-07, 4.8863e-07, 4.8670e-07, 4.8989e-07,
       4.9089e-07, 4.8866e-07, 4.9048e-07, 4.9121e-07, 4.8864e-07,
       4.8685e-07, 4.8533e-07, 4.8338e-07, 4.8111e-07, 4.8583e-07,
       4.9258e-07, 4.9181e-07, 4.9001e-07, 4.9062e-07, 4.9213e-07,
       4.9151e-07, 4.8920e-07, 4.8760e-07, 4.8661e-07])
CO = np.array([4.6432e-08, 4.6432e-08, 4.6432e-08, 4.6432e-08, 4.6432e-08,
       4.6432e-08, 4.6432e-08, 4.6432e-08, 4.1891e-08, 3.8691e-08,
       3.6961e-08, 3.5093e-08, 3.3278e-08, 3.1512e-08, 2.9625e-08,
       2.7426e-08, 2.4982e-08, 2.2831e-08, 2.0709e-08, 1.8163e-08,
       1.5956e-08, 1.3938e-08, 1.2569e-08, 1.4435e-08, 2.1315e-08,
       2.9980e-08, 3.9317e-08, 5.0871e-08, 6.3185e-08, 7.3855e-08,
       8.2564e-08, 8.9767e-08, 9.6038e-08, 1.0101e-07, 1.0638e-07,
       1.1112e-07, 1.1482e-07, 1.1758e-07, 1.1932e-07, 1.2082e-07,
       1.2171e-07, 1.2413e-07, 1.2579e-07, 1.2578e-07, 1.2700e-07,
       1.2757e-07, 1.2726e-07, 1.2833e-07, 1.2948e-07, 1.2984e-07,
       1.3040e-07, 1.3117e-07, 1.3190e-07, 1.3249e-07, 1.3503e-07,
       1.3815e-07, 1.3913e-07, 1.3976e-07, 1.4107e-07, 1.4263e-07,
       1.4353e-07, 1.4392e-07, 1.4448e-07, 1.4515e-07])
CH4 = np.array([8.6000e-08, 8.6000e-08, 8.6000e-08, 8.6000e-08, 8.6000e-08,
       8.6000e-08, 8.6000e-08, 8.6000e-08, 8.6868e-08, 9.1666e-08,
       1.0218e-07, 1.1950e-07, 1.4628e-07, 1.7696e-07, 2.0785e-07,
       2.3834e-07, 2.6739e-07, 2.9980e-07, 3.3778e-07, 3.7443e-07,
       4.1653e-07, 4.8891e-07, 5.9140e-07, 6.8341e-07, 7.4470e-07,
       7.7191e-07, 7.7957e-07, 8.0231e-07, 8.2808e-07, 8.4015e-07,
       8.5276e-07, 8.6704e-07, 8.8120e-07, 8.8785e-07, 8.9952e-07,
       9.0638e-07, 9.0813e-07, 9.1008e-07, 9.0905e-07, 9.1040e-07,
       9.1041e-07, 9.2379e-07, 9.3427e-07, 9.3332e-07, 9.4177e-07,
       9.4557e-07, 9.4292e-07, 9.4795e-07, 9.5026e-07, 9.4581e-07,
       9.4258e-07, 9.3977e-07, 9.3604e-07, 9.3153e-07, 9.4073e-07,
       9.5396e-07, 9.5240e-07, 9.4868e-07, 9.4979e-07, 9.5287e-07,
       9.5174e-07, 9.4742e-07, 9.4434e-07, 9.4212e-07])
O2 = np.array([0.23903, 0.23903, 0.23903, 0.23903, 0.23903, 0.23903, 0.23903,
       0.23903, 0.23502, 0.23456, 0.23556, 0.23385, 0.23359, 0.23412,
       0.23402, 0.23357, 0.23208, 0.23258, 0.23465, 0.23343, 0.233  ,
       0.23293, 0.23209, 0.23231, 0.23339, 0.23225, 0.22807, 0.22973,
       0.23302, 0.23273, 0.23268, 0.23303, 0.23334, 0.232  , 0.23261,
       0.23288, 0.23237, 0.23168, 0.23016, 0.22928, 0.22797, 0.23031,
       0.23212, 0.23115, 0.23259, 0.233  , 0.23192, 0.2328 , 0.23323,
       0.23202, 0.23115, 0.23049, 0.22961, 0.22852, 0.23073, 0.23389,
       0.23356, 0.23268, 0.23291, 0.23367, 0.23345, 0.23233, 0.23148,
       0.23092])
N2 = np.array([0.75833, 0.75833, 0.75833, 0.75833, 0.75833, 0.75833, 0.75833,
       0.75833, 0.75833, 0.75833, 0.75833, 0.75833, 0.75833, 0.75833,
       0.75833, 0.75833, 0.75833, 0.75833, 0.75833, 0.75833, 0.75833,
       0.75833, 0.75833, 0.75833, 0.75833, 0.75833, 0.75833, 0.75833,
       0.75833, 0.75833, 0.75833, 0.75833, 0.75833, 0.75833, 0.75833,
       0.75833, 0.75833, 0.75833, 0.75833, 0.75833, 0.75833, 0.75833,
       0.75833, 0.75833, 0.75833, 0.75833, 0.75833, 0.75833, 0.75833,
       0.75833, 0.75833, 0.75833, 0.75833, 0.75833, 0.75833, 0.75833,
       0.75833, 0.75833, 0.75833, 0.75833, 0.75833, 0.75833, 0.75833,
       0.75833])


# %% [markdown]
# Next, we define our Earth function parameters.

# %%
def earth_spectrum(opacity, atmosphere_kwargs={}):
    """
    Calculate Earth's reflected light spectrum.
    
    Parameters
    ----------
    opacity : picaso.opannection
        Opacity database object
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
    
    # Work on a copy so molecular exclusions don't modify the global DataFrame
    df_atmo = df_atmo_earth.copy()
    if 'exclude_mol' in atmosphere_kwargs:
        sp = atmosphere_kwargs['exclude_mol'][0]
        if sp in df_atmo:
            df_atmo[sp] *= 0
    earth.atmosphere(df=df_atmo)
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

def make_case(opacity):
    """
    Generate reflected light spectra for molecular contribution analysis.
    
    Creates spectra with individual molecules systematically removed to 
    quantify each species' contribution to the total spectrum.
    
    Parameters
    ----------
    opacity : picaso.opannection
        Opacity database object
    
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
    res['all'] = earth_spectrum(opacity)
    for sp in species: # <- can be commented out for quicker calculations 
        res[sp] = earth_spectrum(opacity, atmosphere_kwargs={'exclude_mol': [sp]}) # <- can be commented out for a quicker calculation
    return res


# %% [markdown]
# Now, we can calculate the spectra according to the parameters in our functions above. As mentioned above the exclusionary spectrums we calculate that find the molecular contribution can be commented out for a quicker calculation that just finds the total spectrum. 

# %%
res = make_case(opacities_reflected_light)

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
# Now, we can load the rfast calculation to compare, embedded here again as numpy arrays.

# %%
F1_rfast = np.array([9.66867957e-03, 9.58387599e-03, 9.26770071e-03, 8.80842067e-03,
       8.24216411e-03, 7.58013319e-03, 6.91486114e-03, 6.27401518e-03,
       5.66204958e-03, 5.09137404e-03, 4.54971608e-03, 4.06987515e-03,
       3.63408073e-03, 3.23922123e-03, 2.89432742e-03, 2.57922831e-03,
       2.29342145e-03, 2.06315114e-03, 1.86306423e-03, 1.69111880e-03,
       1.53848258e-03, 1.39332438e-03, 1.27201562e-03, 1.16307835e-03,
       1.07126033e-03, 9.88751714e-04, 9.15814342e-04, 8.51768139e-04,
       7.92130343e-04, 7.47297131e-04, 7.08871088e-04, 6.71522231e-04,
       6.42618319e-04, 6.14150826e-04, 5.90711647e-04, 5.81251707e-04,
       5.66095221e-04, 5.61560240e-04, 5.59455970e-04, 5.58320684e-04,
       5.63458601e-04, 5.78763937e-04, 5.98086438e-04, 6.21291484e-04,
       6.55254198e-04, 6.89892117e-04, 7.33101908e-04, 7.90129032e-04,
       8.64538771e-04, 9.42683314e-04, 1.04467214e-03, 1.16703143e-03,
       1.31410358e-03, 1.49664963e-03, 1.71415561e-03, 2.00280144e-03,
       2.37130154e-03, 2.88264244e-03, 3.63292983e-03, 4.88355834e-03,
       7.50964323e-03, 1.34195317e-02, 2.52242204e-02, 4.57307330e-02,
       7.34677707e-02, 1.07198504e-01, 1.41148175e-01, 1.66039755e-01,
       1.88305874e-01, 2.06221583e-01, 2.26143052e-01, 2.34770687e-01,
       2.44163468e-01, 2.47887869e-01, 2.51334928e-01, 2.51933556e-01,
       2.48232851e-01, 2.47867752e-01, 2.45313312e-01, 2.41317351e-01,
       2.37430126e-01, 2.34039825e-01, 2.30090993e-01, 2.27075062e-01,
       2.25014288e-01, 2.21930272e-01, 2.18663853e-01, 2.15444851e-01,
       2.12229244e-01, 2.08730561e-01, 2.05201755e-01, 2.02681142e-01,
       2.00013025e-01, 1.97103508e-01, 1.94207957e-01, 1.91360976e-01,
       1.88516933e-01, 1.85798520e-01, 1.83112445e-01, 1.80457527e-01,
       1.77821583e-01, 1.75116011e-01, 1.72534854e-01, 1.70011201e-01,
       1.67543061e-01, 1.65119924e-01, 1.62727417e-01, 1.60373827e-01,
       1.58055319e-01, 1.55765286e-01, 1.53496876e-01, 1.51203893e-01,
       1.48947031e-01, 1.46834981e-01, 1.44817237e-01, 1.42643838e-01,
       1.40445342e-01, 1.38310967e-01, 1.36290341e-01, 1.34192833e-01,
       1.31814633e-01, 1.28902395e-01, 1.26253596e-01, 1.25401291e-01,
       1.24079891e-01, 1.22311864e-01, 1.20380212e-01, 1.18295400e-01,
       1.16125028e-01, 1.13780498e-01, 1.11843468e-01, 1.10055996e-01,
       1.08614785e-01, 1.07062502e-01, 1.05311049e-01, 1.03079039e-01,
       1.00666301e-01, 9.86770537e-02, 9.70903668e-02, 9.53910194e-02,
       9.38088673e-02, 9.24695025e-02, 9.09914913e-02, 8.91144617e-02,
       8.70985235e-02, 8.49513624e-02, 8.21397785e-02, 7.96048632e-02,
       7.83653854e-02, 7.83103218e-02, 7.91900148e-02, 7.76078753e-02,
       7.30571825e-02, 7.37259042e-02, 7.60987029e-02, 7.61321612e-02,
       7.60039987e-02, 7.61329849e-02, 7.62143082e-02, 7.58696950e-02,
       7.33994466e-02, 7.49859511e-02, 7.64269550e-02, 7.74796901e-02,
       7.73572597e-02, 7.69299825e-02, 7.74801780e-02, 7.81377307e-02,
       7.86125168e-02, 7.87746319e-02, 7.88239984e-02, 7.87624524e-02,
       7.71509903e-02, 6.93164106e-02, 7.40035564e-02, 7.59394796e-02,
       7.17243837e-02, 7.64686187e-02, 7.31884909e-02, 5.39701420e-02,
       6.53418286e-02, 6.40610627e-02, 7.15053087e-02, 7.40288687e-02,
       7.67718473e-02, 7.68245814e-02, 7.45818969e-02, 3.86526440e-02,
       4.97339227e-02, 7.47975956e-02, 7.58961908e-02, 7.54506515e-02,
       7.16899809e-02, 6.89653587e-02, 7.18453355e-02, 7.35284356e-02,
       6.70403302e-02, 4.67052117e-02, 5.74900230e-02, 5.52258267e-02,
       6.40279239e-02, 7.04497074e-02, 7.30294369e-02, 7.27579753e-02,
       7.30009221e-02, 7.33225298e-02, 7.31588670e-02, 7.28499290e-02,
       7.21426900e-02, 6.98390365e-02, 5.10118235e-02, 4.45214510e-02,
       4.55796939e-02, 4.75576509e-02, 4.83410470e-02, 3.07204355e-02,
       1.40403827e-02, 2.24276754e-02, 2.03158471e-02, 2.43588575e-02,
       2.83888078e-02, 5.32739192e-02, 5.51164420e-02, 5.90426558e-02,
       6.81101610e-02, 6.95236251e-02, 6.95761936e-02, 6.95178834e-02,
       6.95165601e-02, 6.93665298e-02, 6.92954590e-02, 6.89187114e-02,
       6.81778004e-02, 6.70190497e-02, 6.58597839e-02, 6.57834065e-02,
       6.32842388e-02, 6.43999250e-02, 6.20721860e-02, 5.04224731e-02,
       3.40531771e-02, 8.17412810e-03, 1.13976483e-02, 1.18544385e-02,
       1.63295830e-02, 1.51296485e-02, 2.50617359e-02, 4.52713788e-02,
       5.06878363e-02, 5.05304882e-02, 5.51793150e-02, 5.11581838e-02,
       5.48272200e-02, 5.75399460e-02, 6.10832497e-02, 6.67055175e-02,
       6.71357810e-02, 6.61571552e-02, 6.03368494e-02, 6.22087770e-02,
       6.51016946e-02, 6.25204702e-02, 5.14327817e-02, 4.82890554e-02,
       4.42835371e-02, 2.26648801e-02, 1.26484965e-02, 4.78136479e-03,
       5.93219306e-04, 5.10367125e-04, 5.71300534e-04, 4.94298781e-04,
       5.19231042e-04, 5.17087903e-04, 5.82036519e-04, 1.22969733e-03,
       1.85772283e-03, 1.57712281e-03, 8.60833379e-03, 7.43825631e-03,
       1.04207329e-02, 1.96940956e-02, 3.03698330e-02, 4.37983640e-02,
       5.58837042e-02, 6.18665201e-02, 6.33929526e-02, 6.51781629e-02,
       6.25170532e-02, 4.96030693e-02, 6.40054829e-02, 6.41594728e-02,
       6.43913190e-02, 6.63548372e-02, 6.63532653e-02, 6.58086720e-02,
       6.55436510e-02, 6.59868849e-02, 6.64244765e-02, 6.49041530e-02,
       6.40006621e-02, 5.96375601e-02, 5.20369138e-02, 5.13432296e-02,
       4.79415002e-02, 4.33978649e-02, 2.56748469e-02, 1.20363302e-02,
       9.66424279e-04, 2.31913638e-04, 1.77338430e-04, 1.57995476e-04,
       1.58619273e-04, 1.37760720e-04, 1.67673895e-04, 1.58614899e-04,
       1.44440022e-04, 1.48940782e-04, 3.09867291e-04, 8.04816970e-04,
       6.82034079e-03, 2.16363341e-02, 2.16169732e-02])
F2_rfast = np.array([1.75740966e-11, 1.74199550e-11, 1.68452648e-11, 1.60104630e-11,
       1.49812172e-11, 1.37778889e-11, 1.25686695e-11, 1.14038477e-11,
       1.02915197e-11, 9.25424189e-12, 8.26970730e-12, 7.39753332e-12,
       6.60541964e-12, 5.88771056e-12, 5.26082071e-12, 4.68808664e-12,
       4.16859508e-12, 3.75004852e-12, 3.38636424e-12, 3.07383082e-12,
       2.79639441e-12, 2.53255030e-12, 2.31205567e-12, 2.11404786e-12,
       1.94715653e-12, 1.79718627e-12, 1.66461300e-12, 1.54820061e-12,
       1.43980107e-12, 1.35831081e-12, 1.28846643e-12, 1.22057997e-12,
       1.16804331e-12, 1.11629989e-12, 1.07369610e-12, 1.05650142e-12,
       1.02895251e-12, 1.02070959e-12, 1.01688480e-12, 1.01482127e-12,
       1.02416011e-12, 1.05197957e-12, 1.08710076e-12, 1.12927898e-12,
       1.19101068e-12, 1.25396965e-12, 1.33250913e-12, 1.43616342e-12,
       1.57141291e-12, 1.71345089e-12, 1.89882901e-12, 2.12123311e-12,
       2.38855607e-12, 2.72035753e-12, 3.11570325e-12, 3.64035501e-12,
       4.31015239e-12, 5.23958173e-12, 6.60332772e-12, 8.87650952e-12,
       1.36497642e-11, 2.43917636e-11, 4.58483379e-11, 8.31216212e-11,
       1.33537335e-10, 1.94847379e-10, 2.56555371e-10, 3.01799090e-10,
       3.42270689e-10, 3.74834846e-10, 4.11044738e-10, 4.26726599e-10,
       4.43799213e-10, 4.50568802e-10, 4.56834284e-10, 4.57922370e-10,
       4.51195852e-10, 4.50532237e-10, 4.45889206e-10, 4.38626021e-10,
       4.31560478e-10, 4.25398161e-10, 4.18220638e-10, 4.12738787e-10,
       4.08993060e-10, 4.03387455e-10, 3.97450308e-10, 3.91599349e-10,
       3.85754561e-10, 3.79395245e-10, 3.72981175e-10, 3.68399630e-10,
       3.63549977e-10, 3.58261546e-10, 3.52998503e-10, 3.47823740e-10,
       3.42654318e-10, 3.37713245e-10, 3.32830951e-10, 3.28005288e-10,
       3.23214113e-10, 3.18296379e-10, 3.13604787e-10, 3.09017715e-10,
       3.04531546e-10, 3.00127177e-10, 2.95778481e-10, 2.91500521e-10,
       2.87286328e-10, 2.83123892e-10, 2.79000758e-10, 2.74832961e-10,
       2.70730817e-10, 2.66891889e-10, 2.63224372e-10, 2.59273935e-10,
       2.55277880e-10, 2.51398372e-10, 2.47725617e-10, 2.43913121e-10,
       2.39590430e-10, 2.34297056e-10, 2.29482514e-10, 2.27933338e-10,
       2.25531519e-10, 2.22317898e-10, 2.18806867e-10, 2.15017447e-10,
       2.11072509e-10, 2.06811018e-10, 2.03290211e-10, 2.00041247e-10,
       1.97421655e-10, 1.94600178e-10, 1.91416681e-10, 1.87359709e-10,
       1.82974240e-10, 1.79358522e-10, 1.76474510e-10, 1.73385723e-10,
       1.70509953e-10, 1.68075481e-10, 1.65389002e-10, 1.61977254e-10,
       1.58313021e-10, 1.54410273e-10, 1.49299850e-10, 1.44692308e-10,
       1.42439394e-10, 1.42339308e-10, 1.43938266e-10, 1.41062519e-10,
       1.32791037e-10, 1.34006527e-10, 1.38319400e-10, 1.38380215e-10,
       1.38147263e-10, 1.38381712e-10, 1.38529528e-10, 1.37903148e-10,
       1.33413147e-10, 1.36296828e-10, 1.38916042e-10, 1.40829526e-10,
       1.40606993e-10, 1.39830360e-10, 1.40830413e-10, 1.42025601e-10,
       1.42888588e-10, 1.43183253e-10, 1.43272983e-10, 1.43161115e-10,
       1.40232071e-10, 1.25991692e-10, 1.34511196e-10, 1.38029991e-10,
       1.30368500e-10, 1.38991771e-10, 1.33029707e-10, 9.80978307e-11,
       1.18767367e-10, 1.16439406e-10, 1.29970302e-10, 1.34557204e-10,
       1.39542929e-10, 1.39638780e-10, 1.35562406e-10, 7.02562637e-11,
       9.03979449e-11, 1.35954467e-10, 1.37951308e-10, 1.37141482e-10,
       1.30305968e-10, 1.25353609e-10, 1.30588346e-10, 1.33647601e-10,
       1.21854617e-10, 8.48928645e-11, 1.04495677e-10, 1.00380203e-10,
       1.16379172e-10, 1.28051608e-10, 1.32740606e-10, 1.32247189e-10,
       1.32688776e-10, 1.33273341e-10, 1.32975862e-10, 1.32414326e-10,
       1.31128827e-10, 1.26941633e-10, 9.27206978e-11, 8.09235922e-11,
       8.28470878e-11, 8.64422848e-11, 8.78662102e-11, 5.58384315e-11,
       2.55202420e-11, 4.07652495e-11, 3.69267239e-11, 4.42754269e-11,
       5.16003915e-11, 9.68323541e-11, 1.00181382e-10, 1.07317792e-10,
       1.23799175e-10, 1.26368332e-10, 1.26463882e-10, 1.26357895e-10,
       1.26355490e-10, 1.26082790e-10, 1.25953610e-10, 1.25268821e-10,
       1.23922118e-10, 1.21815936e-10, 1.19708818e-10, 1.19569992e-10,
       1.15027426e-10, 1.17055333e-10, 1.12824361e-10, 9.16494758e-11,
       6.18961276e-11, 1.48575528e-11, 2.07167246e-11, 2.15470009e-11,
       2.96811646e-11, 2.75001258e-11, 4.55530008e-11, 8.22866845e-11,
       9.21318085e-11, 9.18458075e-11, 1.00295662e-10, 9.29867269e-11,
       9.96556827e-11, 1.04586419e-10, 1.11026839e-10, 1.21246050e-10,
       1.22028111e-10, 1.20249330e-10, 1.09670159e-10, 1.13072633e-10,
       1.18330891e-10, 1.13639177e-10, 9.34858447e-11, 8.77717087e-11,
       8.04911523e-11, 4.11964002e-11, 2.29903059e-11, 8.69075930e-12,
       1.07825410e-12, 9.27659368e-13, 1.03841385e-12, 8.98453039e-13,
       9.43770703e-13, 9.39875265e-13, 1.05792792e-12, 2.23513661e-12,
       3.37665556e-12, 2.86662812e-12, 1.56467787e-11, 1.35200090e-11,
       1.89410525e-11, 3.57966088e-11, 5.52011654e-11, 7.96092865e-11,
       1.01575982e-10, 1.12450536e-10, 1.15225028e-10, 1.18469883e-10,
       1.13632966e-10, 9.01601017e-11, 1.16338383e-10, 1.16618279e-10,
       1.17039690e-10, 1.20608643e-10, 1.20605786e-10, 1.19615916e-10,
       1.19134205e-10, 1.19939841e-10, 1.20735221e-10, 1.17971834e-10,
       1.16329620e-10, 1.08399108e-10, 9.45839342e-11, 9.33230719e-11,
       8.71399814e-11, 7.88813267e-11, 4.66674107e-11, 2.18776131e-11,
       1.75660322e-12, 4.21533536e-13, 3.22335919e-13, 2.87177556e-13,
       2.88311388e-13, 2.50398226e-13, 3.04769355e-13, 2.88303438e-13,
       2.62538735e-13, 2.70719458e-13, 5.63224551e-13, 1.46286068e-12,
       1.23968663e-11, 3.93268827e-11, 3.92916915e-11])
lam_rfast = np.array([0.2       , 0.20143369, 0.20287766, 0.20433198, 0.20579673,
       0.20727197, 0.20875779, 0.21025426, 0.21176146, 0.21327947,
       0.21480835, 0.2163482 , 0.21789908, 0.21946108, 0.22103428,
       0.22261875, 0.22421459, 0.22582186, 0.22744065, 0.22907105,
       0.23071314, 0.232367  , 0.23403271, 0.23571036, 0.23740004,
       0.23910184, 0.24081583, 0.24254211, 0.24428076, 0.24603188,
       0.24779555, 0.24957186, 0.2513609 , 0.25316277, 0.25497756,
       0.25680536, 0.25864625, 0.26050035, 0.26236774, 0.26424851,
       0.26614276, 0.2680506 , 0.26997211, 0.27190739, 0.27385655,
       0.27581968, 0.27779688, 0.27978825, 0.2817939 , 0.28381393,
       0.28584844, 0.28789753, 0.28996131, 0.29203989, 0.29413337,
       0.29624185, 0.29836545, 0.30050427, 0.30265842, 0.30482801,
       0.30701316, 0.30921397, 0.31143056, 0.31366304, 0.31591152,
       0.31817612, 0.32045695, 0.32275413, 0.32506778, 0.32739802,
       0.32974496, 0.33210872, 0.33448943, 0.3368872 , 0.33930216,
       0.34173444, 0.34418414, 0.34665141, 0.34913637, 0.35163914,
       0.35415985, 0.35669863, 0.35925561, 0.36183092, 0.36442469,
       0.36703705, 0.36966814, 0.37231809, 0.37498704, 0.37767512,
       0.38038247, 0.38310922, 0.38585553, 0.38862152, 0.39140733,
       0.39421312, 0.39703902, 0.39988518, 0.40275174, 0.40563885,
       0.40854665, 0.4114753 , 0.41442495, 0.41739574, 0.42038782,
       0.42340135, 0.42643649, 0.42949338, 0.43257219, 0.43567306,
       0.43879617, 0.44194166, 0.4451097 , 0.44830045, 0.45151407,
       0.45475073, 0.4580106 , 0.46129383, 0.46460059, 0.46793106,
       0.47128541, 0.4746638 , 0.4780664 , 0.4814934 , 0.48494497,
       0.48842128, 0.4919225 , 0.49544883, 0.49900044, 0.5025775 ,
       0.50618021, 0.50980874, 0.51346328, 0.51714402, 0.52085115,
       0.52458485, 0.52834531, 0.53213273, 0.5359473 , 0.53978922,
       0.54365868, 0.54755587, 0.551481  , 0.55543427, 0.55941588,
       0.56342603, 0.56746493, 0.57153278, 0.57562978, 0.57975616,
       0.58391212, 0.58809787, 0.59231363, 0.5965596 , 0.60083601,
       0.60514308, 0.60948103, 0.61385007, 0.61825042, 0.62268233,
       0.627146  , 0.63164167, 0.63616957, 0.64072992, 0.64532297,
       0.64994894, 0.65460807, 0.6593006 , 0.66402677, 0.66878682,
       0.67358099, 0.67840953, 0.68327268, 0.68817069, 0.69310382,
       0.6980723 , 0.7030764 , 0.70811638, 0.71319248, 0.71830497,
       0.72345411, 0.72864016, 0.73386339, 0.73912406, 0.74442244,
       0.7497588 , 0.75513342, 0.76054656, 0.76599851, 0.77148953,
       0.77701993, 0.78258996, 0.78819992, 0.7938501 , 0.79954078,
       0.80527226, 0.81104482, 0.81685876, 0.82271438, 0.82861197,
       0.83455185, 0.8405343 , 0.84655963, 0.85262816, 0.85874019,
       0.86489603, 0.871096  , 0.87734042, 0.8836296 , 0.88996386,
       0.89634353, 0.90276893, 0.90924039, 0.91575825, 0.92232282,
       0.92893445, 0.93559348, 0.94230025, 0.94905509, 0.95585835,
       0.96271038, 0.96961153, 0.97656215, 0.9835626 , 0.99061322,
       0.99771439, 1.00486647, 1.01206981, 1.01932479, 1.02663178,
       1.03399115, 1.04140327, 1.04886853, 1.0563873 , 1.06395997,
       1.07158692, 1.07926855, 1.08700524, 1.09479739, 1.1026454 ,
       1.11054967, 1.1185106 , 1.1265286 , 1.13460407, 1.14273743,
       1.1509291 , 1.15917949, 1.16748902, 1.17585811, 1.1842872 ,
       1.19277672, 1.20132709, 1.20993875, 1.21861215, 1.22734772,
       1.23614591, 1.24500717, 1.25393195, 1.26292071, 1.27197391,
       1.281092  , 1.29027546, 1.29952474, 1.30884033, 1.3182227 ,
       1.32767233, 1.33718969, 1.34677528, 1.35642958, 1.36615309,
       1.3759463 , 1.38580972, 1.39574384, 1.40574917, 1.41582623,
       1.42597552, 1.43619756, 1.44649289, 1.45686201, 1.46730547,
       1.47782379, 1.48841751, 1.49908716, 1.50983331, 1.52065649,
       1.53155725, 1.54253616, 1.55359376, 1.56473064, 1.57594734,
       1.58724446, 1.59862255, 1.61008221, 1.62162402, 1.63324857,
       1.64495644, 1.65674824, 1.66862457, 1.68058604, 1.69263325,
       1.70476682, 1.71698737, 1.72929553, 1.74169191, 1.75417716,
       1.76675191, 1.77941679, 1.79217247, 1.80501958, 1.81795879,
       1.83099075, 1.84411614, 1.85733561, 1.87064984, 1.88405952,
       1.89756532, 1.91116794, 1.92486807, 1.9386664 , 1.95256366,
       1.96656053, 1.98065774, 1.994856  ])

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
def earth_spectrum_thermal(opacity_thermal, atmosphere_kwargs={}):
    """
    Calculate Earth's thermal emission spectrum.
    
    Parameters
    ----------
    opacity_thermal : picaso.opannection
        Opacity database object for thermal wavelengths (typically 4-25 μm)
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
    
    # Work on a copy so molecular exclusions don't modify the global DataFrame
    df_atmo = df_atmo_earth.copy()
    if 'exclude_mol' in atmosphere_kwargs:
        sp = atmosphere_kwargs['exclude_mol'][0]
        if sp in df_atmo:
            df_atmo[sp] *= 0
    
    earth.atmosphere(df=df_atmo)
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

def make_case_thermal(opacity):
    """
    Generate thermal emission spectra for molecular contribution analysis.
    
    Creates thermal spectra with individual molecules systematically removed 
    to quantify each species' contribution to the total spectrum.
    
    Parameters
    ----------
    opacity : picaso.opannection
        Opacity database object for thermal wavelengths
    
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
    
    res['all'] = earth_spectrum_thermal(opacity)
    for sp in species:
        res[sp] = earth_spectrum_thermal(opacity, atmosphere_kwargs={'exclude_mol': [sp]})
    
    return res


# %% [markdown]
# Calculate the thermal emission case.

# %%
res_fix_thermal = make_case_thermal(opacities_thermal_emission)

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
# Once again, lets load and regrid the rfast calculation but now for Earth's thermal emission with the rfast calculation embedded below as numpy arrays.

# %%
F1_rfast_thermal = np.array([293.74170157, 293.03872512, 292.60413146, 292.06611595,
       291.99797755, 289.62404882, 272.66013318, 237.5802685 ,
       250.17541373, 249.02076279, 249.34805471, 242.57711015,
       227.08815566, 223.49420971, 225.2694155 , 236.5180273 ,
       243.4239924 , 252.80540518, 266.71904903, 277.92689726,
       287.70842222, 291.28947127, 288.85904331, 279.59234222,
       281.59759246, 276.72922875, 271.82822098, 277.14626526,
       276.2209216 , 281.83942539, 276.97008398, 281.73478073,
       275.66872788, 281.52498955, 274.0040812 , 268.88293356,
       271.38978898, 259.96009453, 263.68479314, 263.78739118,
       266.64193801, 258.43414776, 261.14141804, 253.55148163,
       250.84315393, 257.3773421 , 248.33602908, 245.32584963,
       243.716991  , 244.81046236, 240.17508042, 238.58803247,
       244.7348423 , 238.17602952, 235.15637534, 236.51179892,
       237.35615237, 240.63026254, 236.37762446, 239.81932014,
       242.89777729, 244.92028033, 247.52485914, 250.35288566,
       250.05882488, 238.15800689, 233.50249175, 237.72721182,
       232.0346031 , 235.9766414 , 233.36695414, 235.05167172,
       237.74352679, 242.9943764 , 239.90364372, 240.37567181,
       241.51186334, 249.3393791 , 243.09236405, 244.90806528,
       248.92716714, 248.95843078, 250.40551925, 257.01057509,
       252.85706682, 253.88830775, 259.38363716, 258.09010348,
       264.6950194 , 264.07420194, 256.86015091, 249.84705319,
       260.51725052, 263.67693452, 260.17715414, 256.87998625,
       265.45136086, 281.90031876, 277.69366809, 287.10083714,
       284.17670506, 285.88297128, 289.22860505, 289.64269158,
       286.92549452, 289.03525497, 288.77635628, 288.16810138,
       290.46105641, 290.41819756, 289.1827422 , 289.30265339,
       288.81805354, 291.93615834, 292.03696569, 292.05899689,
       288.85213308, 289.68746482, 288.09134534, 282.35780183,
       255.49397504, 238.91671579, 260.28928558, 246.16730527,
       242.50177528, 246.91903578, 260.43948261, 273.64318567,
       282.04141424, 284.77839464, 290.43785357, 291.9500761 ,
       291.32240112, 293.09216054, 293.62859922, 293.29745969,
       292.33547167, 293.37652561, 292.91728906, 292.09840084,
       293.20953167, 292.69539335, 292.49184135, 293.87901461,
       293.80469913, 293.0622877 , 293.23356252, 293.69681469,
       293.46658848, 293.52969386, 293.12063247, 292.75943728,
       293.23996683, 292.61228053, 293.66571419, 293.02924767,
       293.62075921, 292.47948075, 290.96770466, 287.78291709,
       285.26942127, 287.8299268 , 291.18835266, 289.28883056,
       286.75913631, 284.63335816, 287.75688326, 283.6527    ,
       275.8421948 , 268.15758688, 255.62751202, 250.48654973,
       248.52225693, 246.06431503, 235.3714981 , 241.96336304,
       238.34440445, 228.42023659, 225.20892163, 222.59550383,
       228.00419285, 228.33780349, 229.61597864, 238.9549675 ,
       235.09168618, 238.97481895, 234.18207064, 231.31585401,
       231.76367838, 224.94678852, 220.82427848, 221.38969338,
       227.45172908, 225.75908968, 243.02640052, 251.90470741,
       238.15232999, 244.96009218, 255.94617258, 257.68526019,
       243.71221266, 252.76335763, 253.62953957, 269.58573184,
       266.23350861, 269.48668889, 272.9591838 , 266.96038132,
       270.33121507, 279.11656089, 279.58428446, 276.13200906,
       263.70616492, 264.7388585 , 273.02101815, 272.83448046,
       275.35844195, 280.25499226, 261.20751102, 268.12904559,
       259.54751701, 262.57633352, 269.44776527, 262.51893509,
       262.11978366, 263.9122474 , 276.33798911, 268.08962829,
       266.47489165, 267.89110978, 250.89082008, 266.75806952,
       270.81723064, 253.69975126, 258.90572833, 263.43132477,
       269.31238464, 256.03359952, 244.57304817, 256.26102981,
       266.32406588, 259.87123876, 255.03324191, 256.14602423,
       264.16759648, 262.45867635, 256.67070572, 261.88551841,
       253.95452588, 244.86328179, 236.95038661, 250.99255129,
       260.96103537, 264.26061159, 263.59227155, 261.85594623,
       254.11651949])
F2_rfast_thermal = np.array([ 1.75688848,  1.79686196,  1.85966174,  1.91310432,  2.00315605,
        1.91405417,  1.19429553,  0.21583326,  0.39722705,  0.38448598,
        0.44601286,  0.34682148,  0.1329926 ,  0.10196334,  0.12999875,
        0.29153987,  0.60313334,  0.85976778,  1.59749829,  2.58741655,
        3.58599091,  4.18763965,  4.01613374,  3.10976427,  3.4641894 ,
        3.05917397,  2.84997923,  3.39811217,  3.398731  ,  4.29012361,
        3.7833424 ,  4.49339923,  3.83029048,  4.76584577,  3.81988772,
        3.31229493,  3.74037059,  2.66117092,  3.11969691,  3.13583481,
        3.5958843 ,  2.831556  ,  3.15826255,  2.47786914,  2.38249709,
        2.98146617,  2.18960263,  2.08551044,  1.97262874,  2.14341708,
        1.82530496,  1.77138784,  2.39788642,  1.90881901,  1.7378494 ,
        1.89657252,  2.0165446 ,  2.42292888,  2.13386184,  2.5364119 ,
        2.94937291,  3.25653398,  3.61079782,  4.06172537,  4.22043206,
        2.88057169,  2.42088961,  2.9361374 ,  2.50340081,  2.88298326,
        2.67785342,  2.99954179,  3.39665311,  4.23719268,  3.97319841,
        4.14156203,  4.473981  ,  5.89143926,  4.88499159,  5.28939844,
        6.30626179,  6.45199291,  6.73367835,  8.43894407,  7.65451795,
        8.01424787,  9.52006342,  9.40266725, 11.49669839, 11.66128994,
        9.86215236,  8.39707002, 11.16955828, 12.24544209, 11.52796829,
       10.6427045 , 13.89549076, 19.79070031, 18.14436314, 22.39784796,
       21.30080406, 22.09283703, 23.84975791, 24.18530242, 23.40536125,
       24.27741871, 24.21432329, 24.09870199, 25.27433438, 25.40303848,
       24.94040015, 25.16855003, 25.06936684, 26.59519074, 26.73642765,
       26.83584956, 25.37173653, 25.86100687, 25.25199147, 22.81010862,
       13.78186823,  8.84703947, 15.47454553, 11.01233227,  9.81761369,
       11.10054248, 15.32240614, 19.70766836, 22.98328175, 24.17925861,
       26.61002156, 27.2492948 , 26.94588306, 27.69510657, 27.89071087,
       27.69039341, 27.22096861, 27.6020109 , 27.33429474, 26.91919043,
       27.30188331, 27.00996921, 26.83797489, 27.2940168 , 27.15618065,
       26.74539734, 26.69708718, 26.75961693, 26.54349733, 26.43596663,
       26.14602836, 25.86849823, 25.91025052, 25.53464958, 25.75789804,
       25.37090834, 25.41997678, 24.86065511, 24.19493685, 23.01252303,
       22.04558573, 22.7021435 , 23.5878387 , 22.84740268, 21.91779345,
       21.10470636, 21.84211083, 20.516734  , 18.24020479, 16.21026154,
       13.20491212, 12.00883422, 11.63872987, 11.24941262,  9.16474083,
       10.2032063 ,  9.481389  ,  7.81105158,  7.28335647,  6.89027861,
        7.68868739,  7.68000917,  7.87413599,  9.30975873,  8.62283894,
        9.18803378,  8.34348225,  7.93536877,  7.96682213,  6.90781413,
        6.33535026,  6.37514629,  7.12785579,  6.84419092,  9.08749209,
       10.3085676 ,  8.30636875,  9.14549851, 10.56460887, 10.71378561,
        8.78972422,  9.83377017,  9.85170535, 11.92210123, 11.27094173,
       11.5929091 , 11.9183784 , 10.95827072, 11.25784184, 12.25619614,
       12.14836641, 11.5391823 ,  9.92860689,  9.85980925, 10.70211076,
       10.52319088, 10.65445097, 11.05827053,  8.9012544 ,  9.44148149,
        8.43924451,  8.60634337,  9.15517152,  8.39068228,  8.20896416,
        8.24655351,  9.25039619,  8.38626793,  8.11279005,  8.11363052,
        6.64112994,  7.75457834,  7.95599918,  6.53792421,  6.80537508,
        7.02560984,  7.34050245,  6.31797589,  5.45713953,  6.10923147,
        6.6683012 ,  6.14762406,  5.75201408,  5.72465023,  6.10171722,
        5.89662836,  5.47459454,  5.66751985,  5.14467774,  4.59340273,
        4.12855351,  4.73771835,  5.14759049,  5.21971557,  5.09530546,
        4.92325454,  4.49349912])
lam_rfast_thermal = np.array([ 4.        ,  4.02867384,  4.05755322,  4.08663962,  4.11593453,
        4.14543944,  4.17515585,  4.20508528,  4.23522926,  4.26558933,
        4.29616703,  4.32696392,  4.35798159,  4.3892216 ,  4.42068555,
        4.45237505,  4.48429172,  4.51643718,  4.54881308,  4.58142105,
        4.61426278,  4.64733993,  4.6806542 ,  4.71420728,  4.74800088,
        4.78203673,  4.81631656,  4.85084213,  4.88561519,  4.92063752,
        4.95591091,  4.99143715,  5.02721806,  5.06325547,  5.0995512 ,
        5.13610713,  5.1729251 ,  5.210007  ,  5.24735472,  5.28497017,
        5.32285526,  5.36101192,  5.39944212,  5.4381478 ,  5.47713093,
        5.51639352,  5.55593756,  5.59576507,  5.63587808,  5.67627864,
        5.71696881,  5.75795067,  5.7992263 ,  5.84079781,  5.88266733,
        5.92483699,  5.96730894,  6.01008535,  6.0531684 ,  6.09656028,
        6.14026323,  6.18427945,  6.2286112 ,  6.27326074,  6.31823036,
        6.36352233,  6.40913898,  6.45508263,  6.50135562,  6.54796032,
        6.5948991 ,  6.64217437,  6.68978852,  6.73774399,  6.78604323,
        6.8346887 ,  6.88368289,  6.93302828,  6.98272741,  7.03278281,
        7.08319702,  7.13397262,  7.18511221,  7.23661839,  7.28849379,
        7.34074106,  7.39336286,  7.44636188,  7.49974082,  7.5535024 ,
        7.60764937,  7.66218449,  7.71711055,  7.77243033,  7.82814668,
        7.88426243,  7.94078044,  7.99770359,  8.0550348 ,  8.11277699,
        8.1709331 ,  8.22950609,  8.28849897,  8.34791473,  8.40775641,
        8.46802707,  8.52872977,  8.58986762,  8.65144373,  8.71346125,
        8.77592334,  8.83883318,  8.90219399,  8.966009  ,  9.03028147,
        9.09501467,  9.16021191,  9.22587651,  9.29201183,  9.35862123,
        9.42570812,  9.49327592,  9.56132808,  9.62986806,  9.69889938,
        9.76842554,  9.83845009,  9.90897662,  9.98000871, 10.05154999,
       10.12360411, 10.19617475, 10.26926561, 10.34288041, 10.41702293,
       10.49169692, 10.56690622, 10.64265465, 10.71894608, 10.79578441,
       10.87317354, 10.95111744, 11.02962007, 11.10868545, 11.1883176 ,
       11.2685206 , 11.34929852, 11.4306555 , 11.51259569, 11.59512325,
       11.67824242, 11.76195742, 11.84627252, 11.93119204, 12.0167203 ,
       12.10286166, 12.18962053, 12.27700132, 12.3650085 , 12.45364655,
       12.54292   , 12.63283341, 12.72339135, 12.81459846, 12.90645938,
       12.9989788 , 13.09216145, 13.18601207, 13.28053545, 13.37573642,
       13.47161984, 13.56819059, 13.6654536 , 13.76341384, 13.86207631,
       13.96144603, 14.06152808, 14.16232756, 14.26384963, 14.36609944,
       14.46908224, 14.57280326, 14.6772678 , 14.78248118, 14.88844879,
       14.99517602, 15.10266832, 15.21093118, 15.31997011, 15.42979069,
       15.54039851, 15.65179921, 15.76399849, 15.87700206, 15.9908157 ,
       16.1054452 , 16.22089642, 16.33717525, 16.45428762, 16.5722395 ,
       16.69103691, 16.81068592, 16.93119263, 17.05256319, 17.17480379,
       17.29792066, 17.42192009, 17.54680841, 17.67259198, 17.79927723,
       17.92687061, 18.05537864, 18.18480788, 18.31516493, 18.44645643,
       18.5786891 , 18.71186966, 18.84600493, 18.98110174, 19.11716698,
       19.25420761, 19.3922306 , 19.53124301, 19.67125192, 19.81226448,
       19.95428788, 20.09732937, 20.24139624, 20.38649586, 20.53263561,
       20.67982297, 20.82806542, 20.97737055, 21.12774597, 21.27919935,
       21.43173841, 21.58537094, 21.74010478, 21.89594783, 22.05290803,
       22.21099339, 22.37021198, 22.53057192, 22.6920814 , 22.85474865,
       23.01858197, 23.18358973, 23.34978034, 23.51716228, 23.68574408,
       23.85553436, 24.02654178, 24.19877505, 24.37224297, 24.54695439,
       24.72291822, 24.90014344])

# %%
# Bin rfast thermal data to same resolution as PICASO (R=140)
# Get PICASO wavelength grid
wno_picaso_thermal, fpfs_picaso_thermal, fp_picaso_thermal = res_fix_thermal['all']
wavelength_picaso_thermal = 1e4 / wno_picaso_thermal  # Convert to um

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