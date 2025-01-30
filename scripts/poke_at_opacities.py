# %%
import numpy as np
import astropy.units as u
import picaso.justdoit as jdi
from matplotlib import pyplot as plt

mh = '+000'
CtoO = '100'
ck_db = f"../data/kcoeff_2020/sonora_2020_feh{mh}_co_{CtoO}.data.196"
opacity_ck = jdi.opannection(ck_db=ck_db) # grab your opacities

cl_run = jdi.inputs(calculation="brown", climate = True)
cl_run.gravity(gravity=10**(3.7), gravity_unit=u.Unit('cm/(s**2)'))
Teq = 1110 # Inglis+24, table 3
Tint = 200 # Inglis+24, table 3
Teff = (Teq ** 4 + Tint ** 4) ** (1/4)
cl_run.effective_temp(Teff)
cl_run.premix_atmosphere(opacity_ck, df=cl_run.guillot_pt(Teq, p_bottom=2, p_top=-6, nlevel=91))
atm_res = jdi.calculate_atm(cl_run, opacity_ck)
dtau = np.mean(atm_res[0], axis=2)
pressure_grid = np.array(cl_run.inputs["atmosphere"]["profile"]["pressure"])

plt.loglog(dtau[:,150], pressure_grid[1:])
plt.gca().invert_yaxis()
plt.xlabel("Differential optical depth")
plt.ylabel("Pressure (bar)")

# %%
