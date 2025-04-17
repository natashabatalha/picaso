using HDF5
using Plots
using DataInterpolations

f = h5open("data/convh5_oktemp/convergence_fsed3_teff2200_nstrupper77_dt2025-03-20T13.45.55.137488.h5", "r")

Array(f["temperature"])
Array(f["temperature_picaso"])