# %%
import numpy as np
import matplotlib.pyplot as plt

from scripts.diamondback.diamondback_virga import read_diamondback_clouds
from scripts.diamondback.diamondback_cz import find_rcb_diamondback
from virga.justdoit import condensation_t
import h5py
import pandas as pd

def principal_condensate(df):
    max_cond = 0.0
    chosen_condensate = ""
    for c in df.columns:
        if c.endswith("qc(g/g)") and max_cond < np.max(df[c]):
            max_cond = np.max(df[c])
            chosen_condensate = c
            
    return chosen_condensate[:-8]

def find_nc_nb(teff, grav_ms2, fsed):
    df = read_diamondback_clouds(teff, grav_ms2, fsed)        
    nc = np.argmax(df[principal_condensate(df) + " qc(g/g)"])
    nb = find_rcb_diamondback(teff, grav_ms2, fsed)
    return nc, nb

def temperature_guess_for_crossing(teff, grav_ms2, fsed, nc_target):
    sonora_df_cloudy = pd.read_csv(f"../../data/sonora_diamondback/t{teff}g{grav_ms2}f{fsed}_m0.0_co1.0.pt", sep=r"\s+", skiprows=[1])
    df = read_diamondback_clouds(teff, grav_ms2, fsed)
    condensate = principal_condensate(df)
    delta_temperature = -sonora_df_cloudy["T"][nc_target] + condensation_t(condensate, 1, 2.2, df["pressure"][nc_target])[1][0]
    return np.array(sonora_df_cloudy["T"]) + delta_temperature

# %%
for teff in [1100, 1600, 2200]:
    for grav_ms2 in [316, 3160]:
        for fsed in [1, 8]:
            df = pd.read_csv(f"../../data/sonora_diamondback/t{teff}g{grav_ms2}f{fsed}_m0.0_co1.0.pt", sep=r"\s+", skiprows=[1])
            pressure = np.array(df["P"])
            nc, nb = find_nc_nb(teff, grav_ms2, fsed)
            guess_levels = [nc, max(0, min(nc, nb) - 5), min(88, max(nc, nb) + 5)]
            temperature_guess_nc = temperature_guess_for_crossing(teff, grav_ms2, fsed, nc)
            temperature_guess_radiative = temperature_guess_for_crossing(teff, grav_ms2, fsed, guess_levels[1])
            temperature_guess_convective = temperature_guess_for_crossing(teff, grav_ms2, fsed, guess_levels[2])
            output_file = f"../../data/wrongside/wrongside_guesses_teff{teff}_grav{grav_ms2}_fsed{fsed}.h5"
            with h5py.File(output_file, "w") as h5file:
                h5file.create_dataset("temperature_guess_nc", data=temperature_guess_nc)
                h5file.create_dataset("temperature_guess_radiative", data=temperature_guess_radiative)
                h5file.create_dataset("temperature_guess_convective", data=temperature_guess_convective)
                h5file.create_dataset("pressure", data=pressure)
                h5file.attrs["teff"] = teff
                h5file.attrs["grav_ms2"] = grav_ms2
                h5file.attrs["fsed"] = fsed
                h5file.attrs["nc"] = nc
                h5file.attrs["nc_radiative"] = guess_levels[1]
                h5file.attrs["nc_convective"] = guess_levels[2]
                h5file.attrs["nb"] = nb
                
# %%
with h5py.File("../../data/wrongside/wrongside_guesses_teff1100_grav316_fsed1.h5") as f:
    print(f.attrs["nb"])
# %%
