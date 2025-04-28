# %%
import pickle
import numpy as np
from matplotlib import pyplot as plt

for (teff, c) in zip([1300, 1500, 1700, 1900], ['b', 'g', 'r', 'k']):
    for (cloud_mode, s) in zip(["cloudless", "fixed", "selfconsistent"], ["-", "--", "-."]):
        with open(f"../data/four_clouds_testing/pkl_outputs/teff_{teff}_fsed_1_MgSiO3_browndwarf_withstar_{cloud_mode}.pkl", "rb") as f:
            data = pickle.load(f)
            plt.semilogy(data["temperature"], data["pressure"], label=f"{teff} K, {cloud_mode}", c=c, linestyle=s)
plt.gca().invert_yaxis()
plt.legend()
# %%

