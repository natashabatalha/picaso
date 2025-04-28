# %%
import pickle

# %%
for cloudmode in ["cloudless", "fixed", "selfconsistent"]:
    with open(f"../data/four_clouds_testing/pkl_outputs/teff_300_fsed_3_MgSiO3_browndwarf_withstar_{cloudmode}.pkl", 'rb') as f:
        d = pickle.load(f)
        plt.semilogy(d["temperature"], d["pressure"], label=cloudmode)
        
plt.xlabel("Temperature (K)")
plt.ylabel("Pressure (bar)")
plt.gca().invert_yaxis()
plt.legend()
# %%
plt.loglog(d["virga_output"]["opd_by_gas"].T[0], d["pressure"][1:], )
plt.gca().invert_yaxis()
plt.xlabel("Optical depth (self-consistent)")
plt.ylabel("Pressure (bar)")
# %%
