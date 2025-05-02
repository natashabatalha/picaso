# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from virga import justdoit as vdi
from astropy import units as u
import re
from os import path
from functools import reduce

from scripts.diamondback.diamondback_io import readInFile, diamondback_datapath
lmap = lambda f, x: list(map(f, x))

# %%
def read_diamondback_clouds(teff, grav_ms2, fsed):
    all_lines = readInFile(path.join(diamondback_datapath, f"diamondback_allmodels/t{teff}g{grav_ms2}f{fsed}_m0.0_co1.0.out"))
    dfs = []
    for (i, line) in enumerate(all_lines):
        if "kz(cm^2/s)" in line:
            columns = line.split()
            condensate = re.match(r" condensing gas = (\S+)", all_lines[i-2])[1]
            columns[6] = condensate + " qc(g/g)"
            columns[7] = condensate + " col_opd"
            data = lmap(lambda l: lmap(float, l.split()), all_lines[i+1:i+91])
            dfs.append(pd.DataFrame(data=data, columns=columns))
        
    df = reduce(lambda x, y: pd.merge(x, y[y.columns.difference(x.columns)], left_index=True, right_index=True), dfs)
    return df.rename(columns={"P(bar)": "pressure", "T(K)": "temperature", "kz(cm^2/s)": "kz"})

# %%
from picaso.input_pickling import retrieve_function_call

def run_virga_to_match_diamondback(teff, grav_ms2, fsed, gases, metallicity=1.0, mean_molecular_weight=2.35):
    diamondback_ptk = read_diamondback_clouds(teff, grav_ms2, fsed)
    args, kwargs, _ = retrieve_function_call("../../data/pkl_inputs/compute_2025-05-01T16.34.43.486481.pkl")
    recommended_gases = vdi.recommend_gas(diamondback_ptk["pressure"], diamondback_ptk["temperature"], metallicity, mean_molecular_weight)
    recommended_gases = np.intersect1d(recommended_gases, gases)
    sum_planet = vdi.Atmosphere(recommended_gases,fsed=fsed,mh=metallicity, mmw = mean_molecular_weight)
    sum_planet.gravity(gravity=grav_ms2, gravity_unit=u.Unit('m/(s**2)'))
    sum_planet.ptk(df = diamondback_ptk)
    
    return vdi.compute(sum_planet, "~/projects/clouds/virga/refrind")
    
    diamondback_ptk = read_diamondback_clouds(teff, grav_ms2, fsed)
    recommended_gases = vdi.recommend_gas(diamondback_ptk["pressure"], diamondback_ptk["temperature"], metallicity, mean_molecular_weight)
    recommended_gases = np.intersect1d(recommended_gases, gases)
    atm = args[0]
    # sum_planet = atm
    # sum_planet.gravity(gravity=grav_ms2, gravity_unit=u.Unit('m/(s**2)'))
    atm.ptk(df = diamondback_ptk)
    return vdi.compute(atm, **kwargs)

if __name__ == "__main__":
    grav_ms2, fsed = 316, 1
    virga_runs = {}
    for teff in [900]:
        virga_runs[teff] = [
            run_virga_to_match_diamondback(teff, grav_ms2, fsed, gases) for gases in [["Fe", "MgSiO3", "Mg2SiO4", "Al2O3"]]
        ]

    fig, axs = plt.subplots(2,2, figsize=(12,8))
    for (teff, ax) in zip([900], [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]):
        vrun = virga_runs[teff][0]
        dbclouds = read_diamondback_clouds(teff, grav_ms2, fsed)
        for (i, (c, color)) in enumerate(zip(vrun["condensibles"], ["black", "blue", "red", "green"])):
            ax.loglog(dbclouds[f"{c} qc(g/g)"], dbclouds["pressure"], label=f"Diamondback {c}", color=color)
            ax.loglog(vrun["condensate_mmr"][:,i], vrun["pressure"], label=f"virga {c}", color=color, ls="--")
        ax.set_xlabel(f"Condensate mixing ratio, {teff = }, {grav_ms2 = }, {fsed = }")
        ax.set_ylabel("Pressure (bar)")
        if teff == 900:
            ax.legend()
        ax.invert_yaxis()

    plt.savefig("../../figures/diamondback_virga_sync_bad_20250501.pdf")

    # %%
    wl_idx = 146
    fig, axs = plt.subplots(2,2, figsize=(12,8))
    for (teff, ax) in zip([900, 1400, 1900, 2400], [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]):
        diamondback_cloud = pd.read_csv(f"../../data/diamondback_optical_properties/t{teff}g{grav_ms2}f{fsed}_m0.0_co1.0.cld", sep=r"\s+")
        sonora_df_cloudy = pd.read_csv(f"../../data/sonora_diamondback/t{teff}g{grav_ms2}f{fsed}_m0.0_co1.0.pt", sep=r"\s+", skiprows=[1])
        layer_pressures = np.array(sonora_df_cloudy["P"])
        level_pressures = np.sqrt(layer_pressures[1:] * layer_pressures[:-1])
        for (virga_output, caption, alpha) in zip(virga_runs[teff], ["all gases", "only MgSiO3", "only Mg2SiO4", "only Fe and Al2O3"], [1, 0.5, 0.5, 0.5]):
            if caption == "all gases":
                ax.loglog(np.array(diamondback_cloud["tau"]).reshape((90, 196))[:,wl_idx], level_pressures, label="Diamondback", lw=2)
            ax.loglog(virga_output["opd_per_layer"][:,-wl_idx],virga_output["pressure"], label=f"virga, {caption}", ls=("--" if alpha==0.5 else "-"), alpha=alpha)
        ax.set_xlim(left=1e-10, right=1e2)
        ax.set_xlabel(f"OPD at 1.1 micron, {teff = }, {grav_ms2 = }, {fsed = }")
        ax.set_ylabel("Pressure (bar)")
        if teff == 900:
            ax.legend()
        ax.invert_yaxis()
    # %%
    for t in virga_runs.keys():
        for virga_output in virga_runs[t]:
            a = virga_output["condensate_mmr"]
            print(np.min(a[a > 0.0]))
    # %%
    vrun = virga_runs[teff][0]
    dbclouds = read_diamondback_clouds(teff, 8, 316)
    for (i, (c, color)) in enumerate(zip(vrun["condensibles"], ["black", "blue", "red", "green"])):
        plt.loglog(dbclouds[f"{c} qc(g/g)"], dbclouds["pressure"], label=f"Diamondback {c}", color=color)
        plt.loglog(vrun["condensate_mmr"][:,i], vrun["pressure"], label=f"virga {c}", color=color, ls="--")
    plt.xlabel(f"Condensate mixing ratio, {teff = }, {grav_ms2 = }, {fsed = }")
    plt.ylabel("Pressure (bar)")
    plt.gca().invert_yaxis()
    # %%
