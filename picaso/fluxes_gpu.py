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
# nvcc -o thermal_1d_kernel_at_top.so -shared -Xcompiler -fPIC thermal_1d_kernel_at_top.cu
# nvcc -o thermal_1d_kernel_ver1.so -shared -Xcompiler -fPIC thermal_1d_kernel_ver1.cu
cuda_lib_retrieval = ctypes.CDLL('./thermal_1d_kernel_at_top.so')

def get_thermal_1d_retrieval_allocate_buffers(nlevel,nwno,numg,numt):

    cuda_lib_retrieval.get_thermal_1d_allocate_buffers.argtypes = [ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]
    cuda_lib_retrieval.get_thermal_1d_allocate_buffers.restype  = None
    cuda_lib_retrieval.get_thermal_1d_allocate_buffers(nlevel,nwno,numg,numt)

def get_thermal_1d_retrieval_free():

    cuda_lib_retrieval.get_thermal_1d_free.argtypes = []
    cuda_lib_retrieval.get_thermal_1d_free.restype  = None


    cuda_lib_retrieval.get_thermal_1d_free()


def get_thermal_1d_retrieval(nlevel,wno, nwno,numg, numt,tlevel,dtau, w0, cosb,plevel,ubar1,
    surf_reflect,hard_surface,dwno,calc_type):

    c_double_p = ctypes.POINTER(ctypes.c_double)


    wno_p      = ctypes.c_void_p(wno.data.ptr)
    wno0_p     = ctypes.c_void_p(dwno.data.ptr)
    dtau_p     = ctypes.c_void_p(dtau.data.ptr)
    w0_p       = ctypes.c_void_p(w0.data.ptr)
    cosb_p     = ctypes.c_void_p(cosb.data.ptr)
    plevel_p   = ctypes.c_void_p(plevel.data.ptr)
    tlevel_p   = ctypes.c_void_p(tlevel.data.ptr)


    ubar1_p       = (ctypes.c_double * ubar1.size)(*ubar1)

    print('Only compute spectrum at top')


    n_out = nwno * numg * numt * nlevel

    flux_at_top              = cp.zeros(nwno * numg * numt)



    flux_at_top_p   = ctypes.c_void_p(flux_at_top.data.ptr)

    cuda_lib_retrieval.get_thermal_1d_set_inputs.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,   # wno
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,   # dtau
        ctypes.c_void_p,   # w0
        ctypes.c_void_p,   # cosb
        ctypes.c_void_p,   # plevel
        ctypes.c_void_p,   # tlevel
        ctypes.c_void_p    # wno0
    ]

    cuda_lib_retrieval.get_thermal_1d_set_inputs.restype = None

    cuda_lib_retrieval.get_thermal_1d_run.argtypes = [c_double_p,
        ctypes.c_double, ctypes.c_double, ctypes.c_int,ctypes.c_void_p]
    cuda_lib_retrieval.get_thermal_1d_run.restype = None



    cuda_lib_retrieval.get_thermal_1d_set_inputs(
        nlevel,
        wno_p,
        nwno,
        numg,
        numt,
        dtau_p,
        w0_p,
        cosb_p,
        plevel_p,
        tlevel_p,
        wno0_p,
    )


    n_iter = 1

    for i in range(n_iter):
        # print("Iteration:", i)

        cuda_lib_retrieval.get_thermal_1d_run(
            ubar1_p,
            surf_reflect,
            hard_surface,
            calc_type,
            flux_at_top_p
        )


    return flux_at_top



cuda_lib = ctypes.CDLL('./thermal_1d_kernel_ver1.so')

def get_thermal_1d_allocate_buffers(nlevel,nwno,numg,numt):

    cuda_lib.get_thermal_1d_allocate_buffers.argtypes = [ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]
    cuda_lib.get_thermal_1d_allocate_buffers.restype  = None
    cuda_lib.get_thermal_1d_allocate_buffers(nlevel,nwno,numg,numt)

def get_thermal_1d_free():

    cuda_lib.get_thermal_1d_free.argtypes = []
    cuda_lib.get_thermal_1d_free.restype  = None

    cuda_lib.get_thermal_1d_free()


def get_thermal_1d(nlevel,wno, nwno,numg, numt,tlevel,dtau, w0, cosb,plevel,ubar1,
    surf_reflect,hard_surface,dwno,calc_type):

    c_double_p = ctypes.POINTER(ctypes.c_double)


    wno_p      = ctypes.c_void_p(wno.data.ptr)
    wno0_p     = ctypes.c_void_p(dwno.data.ptr)
    dtau_p     = ctypes.c_void_p(dtau.data.ptr)
    w0_p       = ctypes.c_void_p(w0.data.ptr)
    cosb_p     = ctypes.c_void_p(cosb.data.ptr)
    plevel_p   = ctypes.c_void_p(plevel.data.ptr)
    tlevel_p   = ctypes.c_void_p(tlevel.data.ptr)


    ubar1_p       = (ctypes.c_double * ubar1.size)(*ubar1)


    n_out = nwno * numg * numt * nlevel

    flux_at_top              = cp.zeros(nwno * numg * numt)

    flux_minus_all           = cp.zeros(n_out)
    flux_plus_all            = cp.zeros(n_out)
    flux_minus_midpt_all     = cp.zeros(n_out)
    flux_plus_midpt_all      = cp.zeros(n_out)


    flux_at_top_p   = ctypes.c_void_p(flux_at_top.data.ptr)

    flux_minus_all_p   = ctypes.c_void_p(flux_minus_all.data.ptr)
    flux_plus_all_p   = ctypes.c_void_p(flux_plus_all.data.ptr)
    flux_minus_midpt_all_p   = ctypes.c_void_p(flux_minus_midpt_all.data.ptr)
    flux_plus_midpt_all_p   = ctypes.c_void_p(flux_plus_midpt_all.data.ptr)


    cuda_lib.get_thermal_1d_set_inputs.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,   # wno
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,   # dtau
        ctypes.c_void_p,   # w0
        ctypes.c_void_p,   # cosb
        ctypes.c_void_p,   # plevel
        ctypes.c_void_p,   # tlevel
        ctypes.c_void_p    # wno0
    ]

    cuda_lib.get_thermal_1d_set_inputs.restype = None

    cuda_lib.get_thermal_1d_run.argtypes = [c_double_p,
        ctypes.c_double, ctypes.c_double, ctypes.c_int,ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
    cuda_lib.get_thermal_1d_run.restype = None



    cuda_lib.get_thermal_1d_set_inputs(
        nlevel,
        wno_p,
        nwno,
        numg,
        numt,
        dtau_p,
        w0_p,
        cosb_p,
        plevel_p,
        tlevel_p,
        wno0_p,
    )


    n_iter = 1

    for i in range(n_iter):
        # print("Iteration:", i)

        cuda_lib.get_thermal_1d_run(
            ubar1_p,
            surf_reflect,
            hard_surface,
            calc_type,
            flux_at_top_p,
            flux_minus_all_p,
            flux_plus_all_p,
            flux_minus_midpt_all_p,
            flux_plus_midpt_all_p

        )


    return (flux_at_top,(flux_minus_all, flux_plus_all,flux_minus_midpt_all, flux_plus_midpt_all))

def compress_thermal(nwno, flux_at_top, gweight, tweight):
    """
    Last step in albedo code. Integrates over phase angle based on the
    Gaussian-Chebychev weights in disco.get_angles_1d or 3d

    Parameters
    ----------
    nwno : int
        Number of wavenumbers
    flux_at_top : ndarray of floats
        Thermal Flux at the top of the atmosphere with dimensions (ng, nt, nwno)
        or could also be (ng,nt,nlayer,nwno)
    gweight : ndarray of floats
        Gaussian weights for integration
    tweight : ndarray of floats
        Chebychev weights for integration
    """
    ng = len(gweight)
    nt = len(tweight)
    #flexible for something that is 3 or 4 dimensions
    flux=zeros(flux_at_top[0,0,:].shape)

    if nt==1 : sym_fac = 1
    else: sym_fac = 1/(2*pi) #azimuthal symmetry breaks down

    for ig in range(ng):
        for it in range(nt):
            flux = flux + flux_at_top[ig,it,:] * gweight[ig] * tweight[it]

    return  flux*sym_fac
