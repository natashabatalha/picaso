# %%
from picaso.input_pickling import retrieve_function_call
from virga import justdoit as vdi
# %%
args, kwargs, result = retrieve_function_call("../../data/pkl_inputs/compute_2025-05-01T16.34.43.486481.pkl")
# %%
atm = args[0]
atm.ptk(df=dbclouds)
vrun = vdi.compute(atm, **kwargs)
# %%
fig, ax = plt.subplots(1,1, figsize=(12,8))
dbclouds = read_diamondback_clouds(teff, grav_ms2, fsed)

for (i, (c, color)) in enumerate(zip(vrun["condensibles"], ["black", "blue", "red", "green"])):
    ax.loglog(dbclouds[f"{c} qc(g/g)"], dbclouds["pressure"], label=f"Diamondback {c}", color=color)
    ax.loglog(vrun["condensate_mmr"][:,i], vrun["pressure"], label=f"virga {c}", color=color, ls="--")
ax.set_xlabel(f"Condensate mixing ratio, {teff = }, {grav_ms2 = }, {fsed = }")
ax.set_ylabel("Pressure (bar)")
if teff == 900:
    ax.legend()
ax.invert_yaxis()
# %%
