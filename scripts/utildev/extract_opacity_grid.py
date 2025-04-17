# %%
import warnings
warnings.filterwarnings('ignore')

import picaso.justdoit as jdi
import astropy.units as u
import numpy as np
import pandas as pd

mh = '+000'
CtoO = '100'

ck_db = f"../data/kcoeff_2020/sonora_2020_feh{mh}_co_{CtoO}.data.196"
opacity_ck = jdi.opannection(ck_db=ck_db)

def get_tau_gas(temperature, pressure, gravity, opacityclass):
    bundle = jdi.inputs(calculation='brown')
    bundle.phase_angle(0,num_gangle=10, num_tangle=1)
    bundle.gravity(gravity=gravity, gravity_unit=u.Unit('m/s**2'))
    bundle.add_pt(temperature, pressure)
    bundle.premix_atmosphere(opacityclass, df = bundle.inputs['atmosphere']['profile'].loc[:,['pressure','temperature']])
    return jdi.calculate_atm(bundle, opacityclass)[1]

teff = 1600
fsed = 1
grav = 3160
sonora_df_cloudy = pd.read_csv(f"../data/sonora_diamondback/t{teff}g{grav}f{fsed}_m0.0_co1.0.pt", sep=r"\s+", skiprows=[1])
temp = np.array(sonora_df_cloudy["T"])
pressure = np.array(sonora_df_cloudy["P"])

tau_gas = get_tau_gas(temp, pressure, grav, opacity_ck)

# %%
plt.imshow(np.log10(np.mean(tau_gas, axis=2)))
plt.colorbar()
# %%
