# %%
import numpy as np
import h5py
from matplotlib import pyplot as plt
# %%
fig = plt.figure(figsize=(8,5))
for (c, teff) in zip(["g", "b", "r", "k"], [1300, 1500, 1700, 1900]):
    with h5py.File(f"../data/four_clouds_testing/teff_{teff}_MgSiO3_browndwarf.h5", "r") as f:
        p = np.array(f["pressure"])
        temp_cloudless, temp_fixed, temp_selfconsistent = np.array(f["temp_cloudless"]), np.array(f["temp_fixed"]), np.array(f["temp_selfconsistent"])
        #plt.semilogy(temp_cloudless, p, linestyle="dotted", color=c, label=f"{teff} cloudless")
        plt.semilogy(temp_fixed, p, linestyle="dashed", color=c, label=f"{teff} fixed")
        plt.semilogy(temp_selfconsistent, p, color=c, label=f"{teff} self-consistent")
        
plt.gca().invert_yaxis()
plt.legend()
plt.show()
