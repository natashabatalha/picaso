# %%
import h5py
import numpy as np
from matplotlib import pyplot as plt
# %%
with h5py.File("data/convh5_oktemp/convergence_fsed3_teff2200_nstrupper10_cloudmodeselfconsistent_dt2025-03-30T12.19.18.565197.h5") as f:
    temperature_picaso = np.array(f["temperature_picaso"])
    pressure_picaso = np.array(f["pressure_picaso"])
    pressure_virga = np.array(f["pressure_virga"])
    temperature_virga = np.array(f["temperature"])
# %%
k = 40
offset = len(temperature_picaso) // 91 - temperature_virga.shape[0] + 1
i = len(temperature_picaso) // 91 - k + offset
plt.semilogy(temperature_virga[-k,:], pressure_virga, label="virga")
plt.semilogy(temperature_picaso[i*91:(i+1)*91], pressure_picaso, label="picaso")
plt.gca().invert_yaxis()
plt.legend()
# %%
