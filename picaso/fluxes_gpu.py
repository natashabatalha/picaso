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
# nvcc -o reflect_1d_kernel_ver1.so -shared -Xcompiler -fPIC reflect_1d_kernel_ver1.cu
# nvcc -o transit_1d_kernel.so -shared -Xcompiler -fPIC transit_1d_kernel.cu

cuda_lib_retrieval = ctypes.CDLL('./thermal_1d_kernel_at_top.so')
cuda_lib = ctypes.CDLL('./thermal_1d_kernel_ver1.so')
cuda_lib_reflected = ctypes.CDLL('./reflect_1d_kernel_ver1.so')



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


def get_thermal_1d_allocate_buffers(nlevel,nwno,numg,numt,calc_type):


    if calc_type == 0:
        print('Spectrum only...')
        cuda_lib_retrieval.get_thermal_1d_allocate_buffers.argtypes = [ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]
        cuda_lib_retrieval.get_thermal_1d_allocate_buffers.restype  = None
        cuda_lib_retrieval.get_thermal_1d_allocate_buffers(nlevel,nwno,numg,numt)

    else:
        print('Climate only...')
        cuda_lib.get_thermal_1d_allocate_buffers.argtypes = [ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]
        cuda_lib.get_thermal_1d_allocate_buffers.restype  = None
        cuda_lib.get_thermal_1d_allocate_buffers(nlevel,nwno,numg,numt)

def get_thermal_1d_free(calc_type):


    if calc_type == 0:
        print('Spectrum only...')
        cuda_lib_retrieval.get_thermal_1d_free.argtypes = []
        cuda_lib_retrieval.get_thermal_1d_free.restype  = None
        cuda_lib_retrieval.get_thermal_1d_free()
    else:
        print('Climate only...')
        cuda_lib.get_thermal_1d_free.argtypes = []
        cuda_lib.get_thermal_1d_free.restype  = None
        cuda_lib.get_thermal_1d_free()



def get_thermal_1d(nlevel,wno, nwno,numg, numt,tlevel,dtau, w0, cosb,plevel,ubar1,
    surf_reflect,hard_surface,dwno,calc_type, hardware = 'cpu'):

    if hardware == 'gpu':


        gpu_required = [
            ('wno',    wno),
            ('dwno',   dwno),
            ('dtau',   dtau),
            ('w0',     w0),
            ('cosb',   cosb),
            ('plevel', plevel),
            ('tlevel', tlevel),
        ]

        # Validate they are CuPy GPU arrays AND build pointers
        ptrs = {}

        for name, arr in gpu_required:
            # Must be a CuPy array with a valid device pointer
            if not hasattr(arr, "data") or not hasattr(arr.data, "ptr"):
                raise TypeError(f"ERROR: '{name}' must be a CuPy GPU array when hardware='gpu'.")



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

        if calc_type == 0:
            print('Spectrum only...')

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

            cuda_lib_retrieval.get_thermal_1d_run(
                ubar1_p,
                surf_reflect,
                hard_surface,
                calc_type,
                flux_at_top_p)

            flux_minus_all           = np.nan
            flux_plus_all            = np.nan
            flux_minus_midpt_all     = np.nan
            flux_plus_midpt_all      = np.nan

        else:
            print('Climate only...')
            calc_type = 0
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



            cuda_lib.get_thermal_1d_run(
                ubar1_p,
                surf_reflect,
                hard_surface,
                calc_type,
                flux_at_top_p,
                flux_minus_all_p,
                flux_plus_all_p,
                flux_minus_midpt_all_p,
                flux_plus_midpt_all_p)

    else:

        print('no cpu module yet...')
        flux_at_top = np.nan
        flux_minus_all = np.nan
        flux_plus_all = np.nan
        flux_minus_midpt_all = np.nan
        flux_plus_midpt_all = np.nan

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






def get_reflected_1d_allocate_buffers(nlevel,nwno,numg,numt):

    cuda_lib_reflected.get_reflected_1d_allocate_buffers.argtypes = [ctypes.c_int,ctypes.c_int,ctypes.c_int, ctypes.c_int]
    cuda_lib_reflected.get_reflected_1d_allocate_buffers.restype = None

    cuda_lib_reflected.get_reflected_1d_allocate_buffers(nlevel,nwno,numg,numt)

def get_reflected_1d(
    nlevel,
    wno, nwno, ng, nt,
    DTAU, TAU, W0, COSB,
    GCOS2, ftau_cld, ftau_ray,
    DTAU_OG, TAU_OG, W0_OG, COSB_OG,
    atm_surf_reflect,ubar0,ubar1,
    cos_theta,F0PI,
    single_phase, multi_phase,
    frac_a, frac_b, frac_c,
    constant_back, constant_forward,
    get_toa_intensity, get_lvl_flux,
    toon_coefficients,
    b_top,
    gweight,
    tweight, hardware = 'cpu'):

    if hardware == 'gpu':

        # --- required GPU inputs in a list ---
        gpu_required = [
            ('wno',      wno),
            ('DTAU',     DTAU),
            ('TAU',      TAU),
            ('W0',       W0),
            ('COSB',     COSB),
            ('GCOS2',    GCOS2),
            ('ftau_cld', ftau_cld),
            ('ftau_ray', ftau_ray),
            ('DTAU_OG',  DTAU_OG),
            ('TAU_OG',   TAU_OG),
            ('W0_OG',    W0_OG),
            ('COSB_OG',  COSB_OG),
            ('F0PI',     F0PI)
        ]


        # --- verify all inputs are CuPy arrays ---
        for name, arr in gpu_required:
            if not hasattr(arr, "data") or not hasattr(arr.data, "ptr"):
                raise TypeError(f"ERROR: Input '{name}' must be a CuPy GPU array when hardware='gpu'.")

        c_double_p = ctypes.POINTER(ctypes.c_double)

        wno_ptr      = ctypes.c_void_p(wno.data.ptr)
        F0PI_ptr     = ctypes.c_void_p(F0PI.data.ptr)
        DTAU_ptr     = ctypes.c_void_p(DTAU.data.ptr)
        TAU_ptr      = ctypes.c_void_p(TAU.data.ptr)
        W0_ptr       = ctypes.c_void_p(W0.data.ptr)
        COSB_ptr     = ctypes.c_void_p(COSB.data.ptr)
        GCOS2_ptr    = ctypes.c_void_p(GCOS2.data.ptr)
        ftau_cld_ptr = ctypes.c_void_p(ftau_cld.data.ptr)
        ftau_ray_ptr = ctypes.c_void_p(ftau_ray.data.ptr)
        DTAU_OG_ptr  = ctypes.c_void_p(DTAU_OG.data.ptr)
        TAU_OG_ptr   = ctypes.c_void_p(TAU_OG.data.ptr)
        W0_OG_ptr    = ctypes.c_void_p(W0_OG.data.ptr)
        COSB_OG_ptr  = ctypes.c_void_p(COSB_OG.data.ptr)


        c_void_p = ctypes.c_void_p

        cuda_lib_reflected.get_reflected_1d_set_inputs.argtypes = [
            ctypes.c_int,   # nlevel
            c_void_p,       # wno (device ptr)
            ctypes.c_int,   # nwno
            ctypes.c_int,   # ng
            ctypes.c_int,   # nt
            c_void_p,       # dtau
            c_void_p,       # tau
            c_void_p,       # w0
            c_void_p,       # cosb
            c_void_p,       # gcos2
            c_void_p,       # ftau_cld
            c_void_p,       # ftau_ray
            c_void_p,       # dtau_og
            c_void_p,       # tau_og
            c_void_p,       # w0_og
            c_void_p,       # cosb_og
            c_void_p,       # f0pi
            ctypes.c_int,   # atm_surf_reflect
            ctypes.c_int,   # single_phase
            ctypes.c_int,   # multi_phase
            ctypes.c_double,# frac_a
            ctypes.c_double,# frac_b
            ctypes.c_double,# frac_c
            ctypes.c_double,# constant_back
            ctypes.c_double,# constant_forward
            ctypes.c_int,   # get_toa_intensity
            ctypes.c_int,   # get_lvl_flux
            ctypes.c_double # toon_coefficients
        ]
        cuda_lib_reflected.get_reflected_1d_set_inputs.restype = None



        cuda_lib_reflected.get_reflected_1d_run.argtypes = [
            c_double_p,      # ubar0
            c_double_p,      # ubar1
            ctypes.c_double, # cos_theta
            ctypes.c_double, # b_top
            ctypes.c_double, # surf_reflect
            c_double_p,      # gweight
            c_double_p,      # tweight
            c_void_p,      # test_out
            c_void_p,      # flux_minus_all_out
            c_void_p,      # flux_plus_all_out
            c_void_p,      # flux_minus_midpt_all_out
            c_void_p       # flux_plus_midpt_all_out
        ]
        cuda_lib_reflected.get_reflected_1d_run.restype = None

        cuda_lib_reflected.get_reflected_1d_set_inputs(
            int(nlevel),
            wno_ptr,
            int(nwno),
            int(ng),
            int(nt),
            DTAU_ptr,
            TAU_ptr,
            W0_ptr,
            COSB_ptr,
            GCOS2_ptr,
            ftau_cld_ptr,
            ftau_ray_ptr,
            DTAU_OG_ptr,
            TAU_OG_ptr,
            W0_OG_ptr,
            COSB_OG_ptr,
            F0PI_ptr,
            int(atm_surf_reflect),
            int(single_phase),
            int(multi_phase),
            float(frac_a),
            float(frac_b),
            float(frac_c),
            float(constant_back),
            float(constant_forward),
            int(get_toa_intensity),
            int(get_lvl_flux),
            float(toon_coefficients)
        )

        ubar0_pointer = (ctypes.c_double*ubar0.size)(*ubar0)
        ubar1_pointer = (ctypes.c_double*ubar1.size)(*ubar1)
        gweight_pointer = (ctypes.c_double*gweight.size)(*gweight)
        tweight_pointer = (ctypes.c_double*tweight.size)(*tweight)

        flux_at_top  = cp.zeros(F0PI.size)
        flux_at_top_p   = ctypes.c_void_p(flux_at_top.data.ptr)

        flux_minus_all_out = cp.zeros(TAU.size*ng*nt)
        flux_minus_all_out_p   = ctypes.c_void_p(flux_minus_all_out.data.ptr)

        flux_plus_all_out = cp.zeros(TAU.size*ng*nt)
        flux_plus_all_out_p   = ctypes.c_void_p(flux_plus_all_out.data.ptr)

        flux_minus_midpt_all_out = cp.zeros(TAU.size*ng*nt)
        flux_minus_midpt_all_out_p   = ctypes.c_void_p(flux_minus_midpt_all_out.data.ptr)

        flux_plus_midpt_all_out = cp.zeros(TAU.size*ng*nt)
        flux_plus_midpt_all_out_p   = ctypes.c_void_p(flux_plus_midpt_all_out.data.ptr)

        cuda_lib_reflected.get_reflected_1d_run(
            ubar0_pointer,
            ubar1_pointer,
            ctypes.c_double(cos_theta),
            ctypes.c_double(b_top),
            ctypes.c_double(atm_surf_reflect),
            gweight_pointer,
            tweight_pointer,
            flux_at_top_p,
            flux_minus_all_out_p,
            flux_plus_all_out_p,
            flux_minus_midpt_all_out_p,
            flux_plus_midpt_all_out_p
         )
    else:
        print('no cpu module yet...')
        flux_at_top = np.nan
        flux_minus_all_out = np.nan
        flux_plus_all_out = np.nan
        flux_minus_midpt_all_out = np.nan
        flux_plus_midpt_all_out = np.nan


    return flux_at_top, flux_minus_all_out, flux_plus_all_out,flux_minus_midpt_all_out, flux_plus_midpt_all_out



def get_reflected_1d_free():

    cuda_lib_reflected.get_reflected_1d_free.argtypes = []
    cuda_lib_reflected.get_reflected_1d_free.restype = None
    cuda_lib_reflected.get_reflected_1d_free()


cuda_lib_transit = ctypes.CDLL("./transit_1d_kernel.so")


def get_transit_1d_allocate_buffers(nlevel, nwno):
    cuda_lib_transit.get_transit_1d_allocate_buffers.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
    ]
    cuda_lib_transit.get_transit_1d_allocate_buffers.restype = None
    cuda_lib_transit.get_transit_1d_allocate_buffers(
        int(nlevel), int(nwno)
    )


def get_transit_1d_free():
    cuda_lib_transit.get_transit_1d_free.argtypes = []
    cuda_lib_transit.get_transit_1d_free.restype = None
    cuda_lib_transit.get_transit_1d_free()


def get_transit_1d_set_inputs(
    z,
    dz,
    player,
    tlayer,
    colden,
    DTAU,
    mmw,
    k_b,
    amu,
    rstar,
    nlevel,
    nwno,
):
    gpu_required = [
        ("z", z),
        ("dz", dz),
        ("player", player),
        ("tlayer", tlayer),
        ("colden", colden),
        ("DTAU", DTAU),
        ("mmw", mmw),
    ]

    for name, arr in gpu_required:
        if not hasattr(arr, "data") or not hasattr(arr.data, "ptr"):
            raise TypeError(
                f"ERROR: '{name}' must be a CuPy GPU array."
            )

    z_p      = ctypes.c_void_p(z.data.ptr)
    dz_p     = ctypes.c_void_p(dz.data.ptr)
    player_p = ctypes.c_void_p(player.data.ptr)
    tlayer_p = ctypes.c_void_p(tlayer.data.ptr)
    colden_p = ctypes.c_void_p(colden.data.ptr)
    DTAU_p   = ctypes.c_void_p(DTAU.data.ptr)
    mmw_p    = ctypes.c_void_p(mmw.data.ptr)

    cuda_lib_transit.get_transit_1d_set_inputs.argtypes = [
        ctypes.c_void_p,  # z
        ctypes.c_void_p,  # dz
        ctypes.c_void_p,  # player
        ctypes.c_void_p,  # tlayer
        ctypes.c_void_p,  # colden
        ctypes.c_void_p,  # DTAU
        ctypes.c_void_p,  # mmw
        ctypes.c_double,  # k_b
        ctypes.c_double,  # amu
        ctypes.c_double,  # rstar
        ctypes.c_int,     # nlevel
        ctypes.c_int,     # nwno
    ]
    cuda_lib_transit.get_transit_1d_set_inputs.restype = None

    cuda_lib_transit.get_transit_1d_set_inputs(
        z_p,
        dz_p,
        player_p,
        tlayer_p,
        colden_p,
        DTAU_p,
        mmw_p,
        float(k_b),
        float(amu),
        float(rstar),
        int(nlevel),
        int(nwno),
    )


def get_transit_1d_run(nwno):
    F_out = cp.zeros(int(nwno), dtype=cp.float64)

    cuda_lib_transit.get_transit_1d_run.argtypes = [
        ctypes.c_void_p,  # F_out_dev
    ]
    cuda_lib_transit.get_transit_1d_run.restype = None

    F_out_p = ctypes.c_void_p(F_out.data.ptr)
    cuda_lib_transit.get_transit_1d_run(F_out_p)

    return F_out


def get_transit_1d(
    z,
    dz,
    nlevel,
    nwno,
    rstar,
    mmw,
    k_b,
    amu,
    player,
    tlayer,
    colden,
    DTAU,
    hardware="gpu",
):
    if hardware != "gpu":
        raise NotImplementedError("CPU path not implemented here.")

    get_transit_1d_set_inputs(
        z,
        dz,
        player,
        tlayer,
        colden,
        DTAU,
        mmw,
        k_b,
        amu,
        rstar,
        nlevel,
        nwno,
    )

    return get_transit_1d_run(nwno)
