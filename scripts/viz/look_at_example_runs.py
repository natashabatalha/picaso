# %%
import numpy as np
from matplotlib import pyplot as plt
# %%
hd189_pressure = np.load("../data/silicate_test_cases/hd189_pressure.npy")
hd189_temperature = np.load("../data/silicate_test_cases/hd189_temperature.npy")
wasp17_pressure = np.load("../data/silicate_test_cases/wasp17_pressure.npy")
wasp17_temperature = np.load("../data/silicate_test_cases/wasp17_temperature.npy")

tcond_sio2_wasp17 = 1e4 / (6.14 - 0.35 * np.log10(wasp17_pressure) - 0.7 * np.log10(81))
tcond_sio2_hd189 = 1e4 / (6.14 - 0.35 * np.log10(hd189_pressure) - 0.7 * 0.17)
# %%
plt.semilogy(hd189_temperature, hd189_pressure, label="HD 189733b", c="blue")
plt.semilogy(tcond_sio2_hd189, hd189_pressure, label="HD 189733b Tcond", c="blue", ls="--")
plt.semilogy(wasp17_temperature, wasp17_pressure, label="WASP-17", c="orange")
plt.semilogy(tcond_sio2_wasp17, wasp17_pressure, label="WASP-17 Tcond", c="orange", ls="--")
plt.gca().invert_yaxis()
plt.xlabel("Temperature (K)")
plt.ylabel("Pressure (bar)")
plt.legend()
# %%
