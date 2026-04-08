from numpy import exp, zeros, where, sqrt, cumsum , pi, outer, sinh, cosh, min, dot, array,log, log10,ones, array_equal
import numpy as np
import time
import pickle as pk
from scipy.linalg import solve_banded
import ctypes
import matplotlib.pyplot as plt

import ctypes
import numpy as np
import time
import cupy as cp
from fluxes_gpu import *


def get_transit_1d(z, dz,nlevel, nwno, rstar, mmw, k_b,amu,
                    player, tlayer, colden, DTAU):
    """
    Routine to get the transmission spectrum
    Parameters
    ----------
    z : float, array
        Altitude in decreasing order (cm)
    dz : float, array
        length of each atmospheric layer
    nlevel : int
        Number of levels
    nwno : int
        Number of wavelength points
    rstar : float
        Radius of star (cm)
    mmw : float, array
        Mean molecular weight
    k_b : float
        Boltzman constant cgs
    amu : float
        Atomic mass units cgs
    player : float, array
        Pressure at layers (dyn/cm2)
    tlayer : float, array
        Temperature at layers (K)
    colden : float, array
        Column density conputed in atmsetup.get_column_density()
    DTAU : float, array
        Matrix of summed tau opacities from optics. This is
        TAUGAS + TAURAY + TAUCLD
    Returns
    -------
    array
        Rp**2 /Rs**2 as a function of wavelength

    Notes
    -----
    .. [1] Brown, Timothy M. "Transmission spectra as diagnostics of extrasolar giant planet atmospheres." The Astrophysical Journal 553.2 (2001): 1006.
    """
    mmw = mmw * amu #make sure mmw in grams

    delta_length=zeros((nlevel,nlevel))
    for i in range(nlevel):
        for j in range(i):
            reference_shell = z[i]
            inner_shell = z[i-j]
            outer_shell = z[i-j-1]
            #this is the path length between two layers
            #essentially tangent from the inner_shell and toward
            #line of sight to the outer shell
            integrate_segment=((outer_shell**2-reference_shell**2)**0.5-
                    (inner_shell**2-reference_shell**2)**0.5)
            #make sure to use the pressure and temperature
            #between inner and outer shell
            #this is the same index as outer shell because ind = 0 is the outer-
            #most layer
            delta_length[i,j]=integrate_segment*player[i-j-1]/tlayer[i-j-1]/k_b
    #remove column density and mmw from DTAU which was calculated in
    #optics because line of site integration is diff for transit
    #TAU = array([DTAU[:,i]  / colden * mmw  for i in range(nwno)])
    TAU = zeros((nwno, nlevel-1))
    for i in range(nwno):
        TAU[i,:] = DTAU[:,i]  / colden * mmw
    transmitted=zeros((nwno, nlevel))+1.0
    for i in range(nlevel):
        TAUALL=zeros(nwno)#0.
        for j in range(i):
            #two because symmetry of sphere
            TAUALL = TAUALL + 2*TAU[:,i-j-1]*delta_length[i,j]
        transmitted[:,i]=exp(-TAUALL)
    #equation 11 from Brown, T (2001)
    #https://ui.adsabs.harvard.edu/abs/2001ApJ...553.1006B/abstract
    F=(((min(z))/(rstar))**2 +
        2./(rstar)**2.*dot((1.-transmitted),z*dz))

    return F

test_ref_frame = np.load('/media/zyn/T7 Shield/PICASO_GPU_code/transmission_test_parms.npz')

nlevel = test_ref_frame['nlevel']
nwno = test_ref_frame['nwno']
atm_lv_z =test_ref_frame['atm_lv_z']
atm_lv_dz = test_ref_frame['atm_lv_dz']
DTAU_OG = test_ref_frame['DTAU_OG']
radius_star = test_ref_frame['R_Star']
atm_mmw = test_ref_frame['atm_mmw']
atm_ckb = test_ref_frame['atm_ckb']
atm_camu = test_ref_frame['atm_camu']
lvl_T = test_ref_frame['lvl_T']
lvl_P = test_ref_frame['lvl_P']
atm_colden = test_ref_frame['atm_colden']

start = time.time()
rprs2_g = get_transit_1d(atm_lv_z, atm_lv_dz,\
                  nlevel, nwno, radius_star, atm_mmw,\
                  atm_ckb, atm_camu, lvl_P,\
                  lvl_T, atm_colden,\
                  DTAU_OG)

print(atm_lv_z.shape, atm_lv_dz.shape,\
                  nlevel, nwno, radius_star, atm_mmw.shape,\
                  atm_ckb, atm_camu, lvl_P.shape,\
                  lvl_T.shape, atm_colden.shape,\
                  DTAU_OG.shape)
print(rprs2_g.shape)
# exit()
start3 = time.time()
print("compute opacity (s):", start3 - start)
cpu_runtime = start3 - start



test_ref_frame = np.load('/media/zyn/T7 Shield/PICASO_GPU_code/transmission_test_parms.npz')

nlevel = int(test_ref_frame['nlevel'])
nwno   = int(test_ref_frame['nwno'])

z_gpu      = cp.asarray(test_ref_frame['atm_lv_z']).reshape(-1)
dz_gpu     = cp.asarray(test_ref_frame['atm_lv_dz']).reshape(-1)
DTAU_gpu   = cp.asarray(test_ref_frame['DTAU_OG'], dtype=cp.float64).reshape(-1)
player_gpu = cp.asarray(test_ref_frame['lvl_P'][:nlevel-1], dtype=cp.float64).reshape(-1)
tlayer_gpu = cp.asarray(test_ref_frame['lvl_T'][:nlevel-1], dtype=cp.float64).reshape(-1)
colden_gpu = cp.asarray(test_ref_frame['atm_colden'], dtype=cp.float64).reshape(-1)
mmw_gpu    = cp.asarray(test_ref_frame['atm_mmw'], dtype=cp.float64).reshape(-1)

rstar = float(test_ref_frame['R_Star'])
k_b   = float(test_ref_frame['atm_ckb'])
amu   = float(test_ref_frame['atm_camu'])

get_transit_1d_allocate_buffers(nlevel, nwno)

get_transit_1d_set_inputs(
    z_gpu,
    dz_gpu,
    player_gpu,
    tlayer_gpu,
    colden_gpu,
    DTAU_gpu,
    mmw_gpu,
    k_b,
    amu,
    rstar,
    nlevel,
    nwno
)

n_iter = 1000
start = time.time()

for i in range(n_iter):
    print("Iteration:", i)
    rprs2_gpu = get_transit_1d_run(nwno)

cp.cuda.Stream.null.synchronize()
stop = time.time()

print("Total time:", (stop - start) / n_iter)
gpu_runtime = (stop - start) / n_iter
get_transit_1d_free()


label1 = 'GPU time: %.4f s.' % gpu_runtime
label2 = 'CPU time: %.4f s.' % cpu_runtime
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
fontsize = 20
tick_size = 14
fig = plt.figure(figsize=(15,10))
fig.subplots_adjust(hspace = 0.1)
ax1 =fig.add_subplot(2,1,1)
ax1.plot(rprs2_g,color='black',label = label2)
ax1.plot(rprs2_gpu.get(),color='red',label = label1)
ax1.set_ylabel('Spectrum',fontsize = fontsize)
# ax1.set_xlim(0.3,1.0)
plt.legend(loc='best',fontsize=fontsize)
ax1.tick_params(axis='both', labelsize=tick_size)  # Increase tick size
ax2 =fig.add_subplot(2,1,2)
ax2.plot((rprs2_gpu.get()-rprs2_g),color='black')
ax2.set_xlabel(r'wavelength ($\mu$m)',fontsize = fontsize)
ax2.set_ylabel('Residuals',fontsize = fontsize)
# ax2.set_xlim(0.3,1.0)
ax2.sharex(ax1)
ax2.tick_params(axis='both', labelsize=tick_size)  # Increase tick size
plt.show()
