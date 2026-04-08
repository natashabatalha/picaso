from numpy import exp, zeros, where, sqrt, cumsum , pi, outer, sinh, cosh, min, dot, array,log, log10,ones, array_equal
import numpy as np
import time
import pickle as pk
from scipy.linalg import solve_banded
import ctypes
import matplotlib.pyplot as plt

import ctypes
import numpy as np
import cupy as cp
from fluxes_gpu import *


test_ref_frame = np.load('/media/zyn/T7 Shield/PICASO_GPU_code/test_reflected_parms.npz')

nlevel = test_ref_frame['nlevel']
nwno= test_ref_frame['nwno']
ng= test_ref_frame['ng']
nt= test_ref_frame['nt']

wno_gpu= cp.asarray(test_ref_frame['wno'])
DTAU_gpu= cp.asarray(test_ref_frame['DTAU']).reshape(-1)
TAU_gpu= cp.asarray(test_ref_frame['TAU']).reshape(-1)
W0_gpu= cp.asarray(test_ref_frame['W0']).reshape(-1)
COSB_gpu= cp.asarray(test_ref_frame['COSB']).reshape(-1)
GCOS2_gpu= cp.asarray(test_ref_frame['GCOS2']).reshape(-1)
ftau_cld_gpu= cp.asarray(test_ref_frame['ftau_cld']).reshape(-1)
ftau_ray_gpu= cp.asarray(test_ref_frame['ftau_ray']).reshape(-1)
DTAU_OG_gpu= cp.asarray(test_ref_frame['DTAU_OG']).reshape(-1)
TAU_OG_gpu= cp.asarray(test_ref_frame['TAU_OG']).reshape(-1)
W0_OG_gpu= cp.asarray(test_ref_frame['W0_OG']).reshape(-1)
COSB_OG_gpu= cp.asarray(test_ref_frame['COSB_OG']).reshape(-1)
F0PI_gpu= cp.asarray(test_ref_frame['F0PI']) #array

atm_surf_reflect= test_ref_frame['atm_surf_reflect']
cos_theta= test_ref_frame['cos_theta']
single_phase= test_ref_frame['single_phase']
multi_phase= test_ref_frame['multi_phase']
frac_a= test_ref_frame['frac_a']
frac_b= test_ref_frame['frac_b']
frac_c= test_ref_frame['frac_c']
constant_back= test_ref_frame['constant_back']
constant_forward= test_ref_frame['constant_forward']
atm_lvl_flux= test_ref_frame['atm_lvl_flux']
toon_coefficients= test_ref_frame['toon_coefficients']
b_top= test_ref_frame['b_top']

ubar0= test_ref_frame['ubar0'].reshape(-1)
ubar1= test_ref_frame['ubar1'].reshape(-1)
gweight = cp.asarray(test_ref_frame['gweight'])
tweight = cp.asarray(test_ref_frame['tweight'])

get_toa_intensity = 1

device_name = 'cpu'
device_name = 'gpu'


get_reflected_1d_allocate_buffers(nlevel,nwno, ng, nt)



start = time.time()
for i in range(100):
    print(i)

    flux_at_top_gpu,flux_minus_all_out_gpu,flux_plus_all_out_gpu,flux_minus_midpt_all_out_gpu,flux_plus_midpt_all_out_gpu =  get_reflected_1d(
        nlevel,
        wno_gpu, nwno, ng, nt,
        DTAU_gpu, TAU_gpu, W0_gpu, COSB_gpu,
        GCOS2_gpu, ftau_cld_gpu, ftau_ray_gpu,
        DTAU_OG_gpu, TAU_OG_gpu, W0_OG_gpu, COSB_OG_gpu,
        atm_surf_reflect,ubar0,ubar1,
        cos_theta, F0PI_gpu,
        single_phase, multi_phase,
        frac_a, frac_b, frac_c,
        constant_back, constant_forward,
        get_toa_intensity, atm_lvl_flux,
        toon_coefficients,
        b_top,
        gweight,
        tweight, hardware = device_name)

start3 = time.time()



print("compute opacity (s):", (start3 - start)/100)
print('here')

get_reflected_1d_free()


gpu_runtime = (start3 - start)/100
cpu_runtime = 1.4660837650299072

check1_frame = np.load('/media/zyn/T7 Shield/PICASO_GPU_code/check_lvl_flux.npz')

test_out = flux_at_top_gpu.get()
flux_minus_all_out = flux_minus_all_out_gpu.get()
flux_plus_all_out = flux_plus_all_out_gpu.get()
flux_minus_midpt_all_out = flux_minus_midpt_all_out_gpu.get()
flux_plus_midpt_all_out = flux_plus_midpt_all_out_gpu.get()


flux_minus_all_cpu = check1_frame['flux_minus_all'].reshape(-1)
flux_plus_all_cpu = check1_frame['flux_plus_all'].reshape(-1)
flux_minus_midpt_all_cpu = check1_frame['flux_minus_midpt_all'].reshape(-1)
flux_plus_midpt_all_cpu = check1_frame['flux_plus_midpt_all'].reshape(-1)


label1 = 'CPU time: %.4f s.' % cpu_runtime
label2 = 'GPU time: %.4f s.' % gpu_runtime

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
fontsize = 10
fig = plt.figure(figsize=(15,10))
fig.subplots_adjust(hspace = 0.1)



ax1 =fig.add_subplot(4,2,1)
ax1.plot(flux_minus_all_cpu,color = 'black',label = label1)
ax1.plot(flux_minus_all_out,color = 'red',label = label2)
ax1.set_title('flux_minus',fontsize=fontsize)
plt.legend(loc='best',fontsize=fontsize)
ax3 =fig.add_subplot(4,2,3)
ax3.plot(flux_minus_all_cpu - flux_minus_all_out,color = 'black', label='diff')
plt.legend(loc='best',fontsize=fontsize)
ax3.sharex(ax1)

ax2 =fig.add_subplot(4,2,2)
ax2.plot(flux_plus_all_cpu,color = 'black')
ax2.plot(flux_plus_all_out,color = 'red')
ax2.set_title('flux_plus',fontsize=fontsize)
ax4 =fig.add_subplot(4,2,4)
ax4.plot(flux_plus_all_cpu- flux_plus_all_out,color = 'black')
ax4.sharex(ax2)

ax5 =fig.add_subplot(4,2,5)
ax5.plot(flux_minus_midpt_all_cpu,color = 'black')
ax5.plot(flux_minus_midpt_all_out,color = 'red')
ax5.set_title('flux_minus_midpt',fontsize=fontsize)
ax7 =fig.add_subplot(4,2,7)
ax7.plot(flux_minus_midpt_all_cpu - flux_minus_midpt_all_out,color = 'black')
ax7.sharex(ax5)


ax6 =fig.add_subplot(4,2,6)
ax6.plot(flux_plus_midpt_all_cpu,color = 'black')
ax6.plot(flux_plus_midpt_all_out,color = 'red')
ax6.set_title('flux_plus_midpt',fontsize=fontsize)
ax8 =fig.add_subplot(4,2,8)
ax8.plot(flux_plus_midpt_all_cpu - flux_plus_midpt_all_out,color = 'black')
ax8.sharex(ax6)
plt.tight_layout()
plt.show()


wno = wno_gpu.get()
check1_frame = np.load('/media/zyn/T7 Shield/PICASO_GPU_code/check_frame.npz')
flux_at_top_in = check1_frame['albedo'].reshape(-1)

label1 = 'GPU time: %.4f s.' % gpu_runtime
label2 = 'CPU time: %.4f s.' % cpu_runtime
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
fontsize = 20
tick_size = 14
fig = plt.figure(figsize=(15,10))
fig.subplots_adjust(hspace = 0.1)
ax1 =fig.add_subplot(2,1,1)
ax1.plot(1e4/wno, flux_at_top_in,color='black',label = label2)
ax1.plot(1e4/wno, test_out,color='red',label = label1)
# ax1.set_xlabel(r'wavelength ($\mu$m)',fontsize = fontsize)
ax1.set_ylabel('Spectrum',fontsize = fontsize)
ax1.set_xlim(0.3,1.0)
plt.legend(loc='best',fontsize=fontsize)
ax1.tick_params(axis='both', labelsize=tick_size)  # Increase tick size

ax2 =fig.add_subplot(2,1,2)
ax2.plot(1e4/wno,(test_out-flux_at_top_in),color='black')
ax2.set_xlabel(r'wavelength ($\mu$m)',fontsize = fontsize)
ax2.set_ylabel('Residuals',fontsize = fontsize)
ax2.set_xlim(0.3,1.0)
ax2.sharex(ax1)
ax2.tick_params(axis='both', labelsize=tick_size)  # Increase tick size
# plt.savefig('1d_reflected_example.png')
plt.show()
exit()
