# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from virga import justdoit as vdi
from astropy import units as u

# %%
def readInFile(filename):
	f = open(filename)
	line = f.readline()
	filelines = []
	while line != "":
		filelines.append(line)
		try: 
			line = f.readline()
		except UnicodeDecodeError: 
			line='xxx xxx'
	return filelines
# %%
def run_virga_to_match_diamondback(teff, fsed, grav_ms2, gases, metallicity=1, mean_molecular_weight=2.2):
    all_lines = readInFile(f"../../data/diamondback_allmodels/t{teff}g{grav_ms2}f{fsed}_m0.0_co1.0.out")
    for (i, line) in enumerate(all_lines):
        if "kz(cm^2/s)" in line:
            pr = np.array([float(x[1:9]) for x in all_lines[i+1:i+91]])
            tm = np.array([float(x[19:25]) for x in all_lines[i+1:i+91]])
            kz = np.array([float(x[27:34]) for x in all_lines[i+1:i+91]])
            break

    recommended_gases = vdi.recommend_gas(pr, tm, metallicity, mean_molecular_weight)
    recommended_gases = np.intersect1d(recommended_gases, gases)
    sum_planet = vdi.Atmosphere(recommended_gases,fsed=fsed,mh=metallicity, mmw = mean_molecular_weight)
    sum_planet.gravity(gravity=grav_ms2, gravity_unit=u.Unit('m/(s**2)'))
    sum_planet.ptk(df = pd.DataFrame({'pressure':pr, 'temperature':tm,'kz':kz}))
    return sum_planet.compute("~/projects/clouds/virga/refrind")

# %%
fsed, grav_ms2 = 3, 316
# %%
virga_runs = {}
for teff in [900, 1400, 1900, 2400]:
    virga_runs[teff] = [
        run_virga_to_match_diamondback(teff, fsed, grav_ms2, gases) for gases in [["Fe", "MgSiO3", "Mg2SiO4", "Al2O3"], ["MgSiO3"], ["Mg2SiO4"], ["Fe", "Al2O3"]]
    ]

# %%

fig, axs = plt.subplots(2,2, figsize=(12,8))
for (teff, ax) in zip([900, 1400, 1900, 2400], [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]):
    diamondback_cloud = pd.read_csv(f"../../data/diamondback_optical_properties/t{teff}g{grav_ms2}f{fsed}_m0.0_co1.0.cld", sep=r"\s+")
    sonora_df_cloudy = pd.read_csv(f"../../data/sonora_diamondback/t{teff}g{grav_ms2}f{fsed}_m0.0_co1.0.pt", sep=r"\s+", skiprows=[1])
    layer_pressures = np.array(sonora_df_cloudy["P"])
    level_pressures = np.sqrt(layer_pressures[1:] * layer_pressures[:-1])
    for (virga_output, caption, alpha) in zip(virga_runs[teff], ["all gases", "only MgSiO3", "only Mg2SiO4", "only Fe and Al2O3"], [1, 0.5, 0.5, 0.5]):
        if caption == "all gases":
            ax.loglog(np.array(diamondback_cloud["tau"]).reshape((90, 196))[:,146], level_pressures, label="Diamondback", lw=2)
        ax.loglog(virga_output["opd_per_layer"][:,146],virga_output["pressure"], label=f"virga, {caption}", ls=("--" if alpha==0.5 else "-"), alpha=alpha)
    ax.set_xlim(left=1e-6, right=1e2)
    ax.set_xlabel(f"OPD at 1.1 micron, {teff = }, {grav_ms2 = }, {fsed = }")
    ax.set_ylabel("Pressure (bar)")
    ax.legend()
    ax.invert_yaxis()

# %%
