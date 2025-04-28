# %%
import pickle
import numpy as np

for fsed in range(1, 5):
    with open(f"../data/convergence_checking/fsed{fsed}_teff1100_all_cloud.pkl", "rb") as f:
        d = pickle.load(f)
        np.save(f"../data/convergence_checking/fsed{fsed}_teff1100_all_cloud.npy", np.array([x["condensate_mmr"] for x in d]))
        np.save(f"../data/convergence_checking/fsed{fsed}_teff1100_all_profiles.npy", np.array([x["temperature"] for x in d]))
        
# %%
