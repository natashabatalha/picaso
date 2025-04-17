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
for teff in [900, 1400, 1900, 2400]:
    fsed = 3
    grav = 316
    sonora_df_cloudy = pd.read_csv(f"../../data/sonora_diamondback/t{teff}g{grav}f{fsed}_m0.0_co1.0.pt", sep=r"\s+", skiprows=[1])
    pressure_diamondback = np.array(sonora_df_cloudy["P"])
    temperature_diamondback = np.array(sonora_df_cloudy["T"])
    all_lines = readInFile(f"../../data/diamondback_allmodels/t{teff}g{grav}f{fsed}_m0.0_co1.0.out")
    for (i, line) in enumerate(all_lines):
        if "kz(cm^2/s)" in line:
            pr = np.array([float(x[1:9]) for x in all_lines[i+1:i+91]])
            tm = np.array([float(x[20:25]) for x in all_lines[i+1:i+91]])
            kz = np.array([float(x[27:34]) for x in all_lines[i+1:i+91]])
            break

    metallicity = 1
    mean_molecular_weight = 2.2 

    recommended_gases = vdi.recommend_gas(pr, tm, metallicity, mean_molecular_weight)

    recommended_gases = np.intersect1d(recommended_gases, ["Fe", "MgSiO3", "Mg2SiO4", "Al2O3"])

    sum_planet = vdi.Atmosphere(recommended_gases,fsed=fsed,mh=metallicity, mmw = mean_molecular_weight)

    #set the planet gravity
    sum_planet.gravity(gravity=grav, gravity_unit=u.Unit('cm/(s**2)'))

    #PT 
    sum_planet.ptk(
        df = pd.DataFrame(
            {'pressure':pr, 
             'temperature':tm, 
             'kz':kz
            }
        )
    )

    plt.loglog(kz, pr, label=f"{teff} K")
plt.xlabel(r"$K_{zz} (\text{cm}^2/s)$")
plt.ylabel("Pressure (bar)")
plt.legend()
plt.gca().invert_yaxis()
plt.show()
# %%
virga_output = sum_planet.compute(directory="~/projects/clouds/virga/refrind")
# %%
# now let's load the Diamondback cloud optical properties for this case

diamondback_cloud = pd.read_csv(f"../data/diamondback_optical_properties/t{teff}g{grav}f{fsed}_m0.0_co1.0.cld", sep=r"\s+")

plt.loglog(np.array(diamondback_cloud["tau"]).reshape((90, 196))[:,146], virga_output["pressure"], label="Diamondback")
plt.loglog(virga_output["opd_per_layer"][:,146],virga_output["pressure"], label="virga")
plt.xlabel("OPD at 1.1 micron")
plt.ylabel("Pressure (bar)")
plt.legend()
plt.gca().invert_yaxis()

# %%
