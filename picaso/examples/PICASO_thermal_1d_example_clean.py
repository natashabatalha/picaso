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
from fluxes_gpu import *
import cupy as cp

test_ref_frame = np.load('/media/zyn/T7 Shield/PICASO_GPU_code/thermal_test_parms.npz')

nlevel = test_ref_frame['nlevel']
nwno= test_ref_frame['nwno']
ng= test_ref_frame['ng']
nt= test_ref_frame['nt']

wno_gpu= cp.asarray(test_ref_frame['wno'])
DTAU_OG_gpu= cp.asarray(test_ref_frame['DTAU_OG']).reshape(-1)
COSB_OG_gpu= cp.asarray(test_ref_frame['COSB_OG']).reshape(-1)
lvl_T_gpu = cp.asarray(test_ref_frame['lvl_T']).reshape(-1)
lvl_P_gpu = cp.asarray(test_ref_frame['lvl_P']).reshape(-1)
W0_no_raman_gpu = cp.asarray(test_ref_frame['W0_no_raman']).reshape(-1)

# print(test_ref_frame['DTAU_OG'].shape)
# exit()
ubar1= test_ref_frame['ubar1'].reshape(-1)
surf_reflect = test_ref_frame['surf_reflect']
hard_surface = test_ref_frame['hard_surface']
wno0_gpu = wno_gpu.copy()*0
calc_type = 1
# calc_type = 0

device_name = 'cpu'
device_name = 'gpu'

get_thermal_1d_allocate_buffers(nlevel,nwno,ng,nt,calc_type)

n_iter = 1000
start = time.time()

for i in range(n_iter):
    print("Iteration:", i)


    flux_at_top_gpu,(flux_minus_all_out_gpu,flux_plus_all_out_gpu,flux_minus_midpt_all_out_gpu,flux_plus_midpt_all_out_gpu) = get_thermal_1d(
        nlevel, wno_gpu, nwno, ng, nt,
        lvl_T_gpu, DTAU_OG_gpu, W0_no_raman_gpu, COSB_OG_gpu,
        lvl_P_gpu, ubar1,
        surf_reflect, hard_surface,
        wno0_gpu, calc_type, hardware = device_name)

stop = time.time()
print("Total time:", (stop - start)/n_iter)

get_thermal_1d_free(calc_type)


flux_at_top = flux_at_top_gpu.get().reshape(ng, nt, nwno)
if calc_type == 1:
    flux_minus_all_out = flux_minus_all_out_gpu.get().reshape(ng, nt, nlevel, nwno)
    flux_plus_all_out = flux_plus_all_out_gpu.get().reshape(ng, nt, nlevel, nwno)
    flux_minus_midpt_all_out = flux_minus_midpt_all_out_gpu.get().reshape(ng, nt, nlevel, nwno)
    flux_plus_midpt_all_out = flux_plus_midpt_all_out_gpu.get().reshape(ng, nt, nlevel, nwno)

gweight = np.array([0.01574791, 0.07390887, 0.14638699, 0.16717464, 0.09678159])
tweight = np.array([1])

gpu_runtime = 0.004143834114074707
cpu_runtime = 0.17575164794921874
check1_frame = np.load('/media/zyn/T7 Shield/PICASO_GPU_code/check_thermal_lvl.npz')

flux_cpu = check1_frame['flux'].reshape(-1)
flux_minus_all_cpu = check1_frame['flux_minus'].reshape(-1)
flux_plus_all_cpu = check1_frame['flux_plus'].reshape(-1)
flux_minus_midpt_all_cpu = check1_frame['flux_minus_mdpt'].reshape(-1)
flux_plus_midpt_all_cpu = check1_frame['flux_plus_mdpt'].reshape(-1)

if calc_type == 1:
    label1 = 'CPU time: %.4f s.' % cpu_runtime
    label2 = 'GPU time: %.4f s.' % gpu_runtime

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    fontsize = 10
    fig = plt.figure(figsize=(15,10))
    fig.subplots_adjust(hspace = 0.1)

    ax1 =fig.add_subplot(6,2,1)
    ax1.plot(flux_minus_all_cpu,color = 'black',label = label1)
    ax1.plot(flux_minus_all_out.reshape(-1),color = 'red',label = label2)
    ax1.set_title('flux_minus',fontsize=fontsize)
    plt.legend(loc='best',fontsize=fontsize)
    ax3 =fig.add_subplot(6,2,3)
    ax3.plot(flux_minus_all_cpu - flux_minus_all_out.reshape(-1),color = 'black', label='diff')
    plt.legend(loc='best',fontsize=fontsize)
    ax3.sharex(ax1)

    ax2 =fig.add_subplot(6,2,2)
    ax2.plot(flux_plus_all_cpu,color = 'black')
    ax2.plot(flux_plus_all_out.reshape(-1),color = 'red')
    ax2.set_title('flux_plus',fontsize=fontsize)
    ax4 =fig.add_subplot(6,2,4)
    ax4.plot(flux_plus_all_cpu- flux_plus_all_out.reshape(-1),color = 'black')
    ax4.sharex(ax2)


    ax5 =fig.add_subplot(6,2,5)
    ax5.plot(flux_minus_midpt_all_cpu,color = 'black')
    ax5.plot(flux_minus_midpt_all_out.reshape(-1),color = 'red')
    ax5.set_title('flux_minus_midpt',fontsize=fontsize)
    ax7 =fig.add_subplot(6,2,7)
    ax7.plot(flux_minus_midpt_all_cpu - flux_minus_midpt_all_out.reshape(-1),color = 'black')
    ax7.sharex(ax5)

    #
    ax6 =fig.add_subplot(6,2,6)
    ax6.plot(flux_plus_midpt_all_cpu,color = 'black')
    ax6.plot(flux_plus_midpt_all_out.reshape(-1),color = 'red')
    ax6.set_title('flux_plus_midpt',fontsize=fontsize)
    ax8 =fig.add_subplot(6,2,8)
    ax8.plot(flux_plus_midpt_all_cpu - flux_plus_midpt_all_out.reshape(-1),color = 'black')
    ax8.sharex(ax6)

    ax9 =fig.add_subplot(6,2,9)
    ax9.plot(flux_cpu,color = 'black')
    ax9.plot(flux_at_top.reshape(-1),color = 'red')
    ax9.set_title('flux',fontsize=fontsize)
    ax11 =fig.add_subplot(6,2,11)
    ax11.plot(flux_cpu - flux_at_top.reshape(-1),color = 'black')
    ax11.sharex(ax9)
    plt.tight_layout()
    plt.show()

thermal_gpu = compress_thermal(nwno,flux_at_top, gweight, tweight)
thermal_cpu = compress_thermal(nwno,flux_cpu.reshape((5,1,nwno)), gweight, tweight)


wno = wno_gpu.get()
label1 = 'GPU time: %.4f s.' % gpu_runtime
label2 = 'CPU time: %.4f s.' % cpu_runtime
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
fontsize = 20
tick_size = 14
fig = plt.figure(figsize=(15,10))
fig.subplots_adjust(hspace = 0.1)
ax1 =fig.add_subplot(2,1,1)
ax1.plot(1e4/wno, thermal_cpu,color='black',label = label2)
ax1.plot(1e4/wno, thermal_gpu,color='red',label = label1)
ax1.set_ylabel('Spectrum',fontsize = fontsize)
# ax1.set_xlim(0.3,1.0)
plt.legend(loc='best',fontsize=fontsize)
ax1.tick_params(axis='both', labelsize=tick_size)  # Increase tick size
ax2 =fig.add_subplot(2,1,2)
ax2.plot(1e4/wno,(thermal_gpu-thermal_cpu),color='black')
ax2.set_xlabel(r'wavelength ($\mu$m)',fontsize = fontsize)
ax2.set_ylabel('Residuals',fontsize = fontsize)
# ax2.set_xlim(0.3,1.0)
ax2.sharex(ax1)
ax2.tick_params(axis='both', labelsize=tick_size)  # Increase tick size
# plt.savefig('1d_thermal_example.png')
plt.show()
exit()
