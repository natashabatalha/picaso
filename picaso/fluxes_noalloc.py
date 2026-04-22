# Comment below helps ignore linting false-positives.
# type: ignore

import numba as nb
from numba.experimental import jitclass
from numba import types
import numpy as np

# By default we will set threads to 1
nb.set_num_threads(1)

@nb.njit(cache=True)
def setup_tri_diag_inplace(A, B, C, D, nlayer, c_plus_up, c_minus_up,
    c_plus_down, c_minus_down, b_top, b_surface, surf_reflect,
    gama, exptrm_positive, exptrm_minus):
    """
    Build the tridiagonal coefficients for one wavelength row in place.
    This routine operates on a single wavelength at a time and writes the
    coefficients directly into preallocated 1D work arrays.

    Parameters
    ----------
    A, B, C, D : ndarray of float64, shape (2 * nlayer,)
        Preallocated tridiagonal coefficient arrays. The entries are overwritten
        in place.
    nlayer : int
        Number of atmospheric layers for the current solve.
    c_plus_up : ndarray of float64, shape (nlayer,)
        Upward ``c_plus`` coefficients evaluated at the top of each layer.
    c_minus_up : ndarray of float64, shape (nlayer,)
        Upward ``c_minus`` coefficients evaluated at the top of each layer.
    c_plus_down : ndarray of float64, shape (nlayer,)
        Downward ``c_plus`` coefficients evaluated at the bottom of each layer.
    c_minus_down : ndarray of float64, shape (nlayer,)
        Downward ``c_minus`` coefficients evaluated at the bottom of each layer.
    b_top : float
        Diffuse radiation entering the atmosphere at the top boundary.
    b_surface : float
        Diffuse radiation entering the atmosphere at the lower boundary.
    surf_reflect : float
        Surface reflectivity for the current wavelength.
    gama : ndarray of float64, shape (nlayer,)
        Toon et al. coefficient from Eq. 22.
    exptrm_positive : ndarray of float64, shape (nlayer,)
        Positive exponential factors from Eq. 44.
    exptrm_minus : ndarray of float64, shape (nlayer,)
        Inverse exponential factors from Eq. 44.
    """
    A[0] = 0.0
    B[0] = gama[0] + 1.0
    C[0] = gama[0] - 1.0
    D[0] = b_top - c_minus_up[0]

    for k in range(nlayer - 1):
        e1 = exptrm_positive[k] + gama[k] * exptrm_minus[k]
        e2 = exptrm_positive[k] - gama[k] * exptrm_minus[k]
        e3 = gama[k] * exptrm_positive[k] + exptrm_minus[k]
        e4 = gama[k] * exptrm_positive[k] - exptrm_minus[k]

        g = gama[k + 1]
        cp_up = c_plus_up[k + 1]
        cp_down = c_plus_down[k]
        cm_up = c_minus_up[k + 1]
        cm_down = c_minus_down[k]

        row = 2 * k + 1
        A[row] = (e1 + e3) * (g - 1.0)
        B[row] = (e2 + e4) * (g - 1.0)
        C[row] = 2.0 * (1.0 - g * g)
        D[row] = (g - 1.0) * (cp_up - cp_down) + (1.0 - g) * (cm_down - cm_up)

        row = 2 * k + 2
        A[row] = 2.0 * (1.0 - gama[k] * gama[k])
        B[row] = (e1 - e3) * (g + 1.0)
        C[row] = (e1 + e3) * (g - 1.0)
        D[row] = e3 * (cp_up - cp_down) + e1 * (cm_down - cm_up)

    e1 = exptrm_positive[nlayer - 1] + gama[nlayer - 1] * exptrm_minus[nlayer - 1]
    e2 = exptrm_positive[nlayer - 1] - gama[nlayer - 1] * exptrm_minus[nlayer - 1]
    e3 = gama[nlayer - 1] * exptrm_positive[nlayer - 1] + exptrm_minus[nlayer - 1]
    e4 = gama[nlayer - 1] * exptrm_positive[nlayer - 1] - exptrm_minus[nlayer - 1]

    A[2 * nlayer - 1] = e1 - surf_reflect * e3
    B[2 * nlayer - 1] = e2 - surf_reflect * e4
    C[2 * nlayer - 1] = 0.0
    D[2 * nlayer - 1] = b_surface - c_plus_down[nlayer - 1] + surf_reflect * c_minus_down[nlayer - 1]


@nb.njit(cache=True)
def tri_diag_solve_inplace(l, a, b, c, d):
    """
    Solve a tridiagonal system in place with no temporary array allocations.
    The solution is written back into ``d``. The ``c`` array is also overwritten
    with the modified superdiagonal coefficients used during the forward sweep.

    Parameters
    ----------
    l : int
        System size.
    a : array-like
        Lower diagonal, with ``a[0]`` unused.
    b : array-like
        Main diagonal.
    c : array-like
        Upper diagonal, with ``c[l - 1]`` unused.
    d : array-like
        Right-hand side. Overwritten with the solution.
    """
    d[0] = d[0] / b[0]
    c[0] = c[0] / b[0]

    for i in range(1, l - 1):
        denom = b[i] - a[i] * c[i - 1]
        inv_denom = 1.0 / denom
        c[i] = c[i] * inv_denom
        d[i] = (d[i] - a[i] * d[i - 1]) * inv_denom

    if l > 1:
        denom = b[l - 1] - a[l - 1] * c[l - 2]
        d[l - 1] = (d[l - 1] - a[l - 1] * d[l - 2]) / denom

        for i in range(l - 2, -1, -1):
            d[i] = d[i] - c[i] * d[i + 1]

@jitclass
class GetReflected1DWorkspace:
    "Per-thread scratch space for the reflected-light solver."

    g1 : types.double[:]
    g2 : types.double[:]
    lamda : types.double[:]
    gama : types.double[:]
    g3 : types.double[:]
    a_minus : types.double[:]
    a_plus : types.double[:]
    c_minus_up : types.double[:]
    c_plus_up : types.double[:]
    c_minus_down : types.double[:]
    c_plus_down : types.double[:]
    exptrm : types.double[:]
    exptrm_positive : types.double[:]
    exptrm_minus : types.double[:]
    p_single : types.double[:]
    A : types.double[:]
    B : types.double[:]
    C : types.double[:]
    D : types.double[:]
    positive : types.double[:]
    negative : types.double[:]
    xint : types.double[:]

    def __init__(self, nlayer):

        self.g1 = np.empty(nlayer, dtype=np.float64)
        self.g2 = np.empty(nlayer, dtype=np.float64)
        self.lamda = np.empty(nlayer, dtype=np.float64)
        self.gama = np.empty(nlayer, dtype=np.float64)
        self.g3 = np.empty(nlayer, dtype=np.float64)
        self.a_minus = np.empty(nlayer, dtype=np.float64)
        self.a_plus = np.empty(nlayer, dtype=np.float64)
        self.c_minus_up = np.empty(nlayer, dtype=np.float64)
        self.c_plus_up = np.empty(nlayer, dtype=np.float64)
        self.c_minus_down = np.empty(nlayer, dtype=np.float64)
        self.c_plus_down = np.empty(nlayer, dtype=np.float64)
        self.exptrm = np.empty(nlayer, dtype=np.float64)
        self.exptrm_positive = np.empty(nlayer, dtype=np.float64)
        self.exptrm_minus = np.empty(nlayer, dtype=np.float64)
        self.p_single = np.empty(nlayer, dtype=np.float64)
        self.A = np.empty(2 * nlayer, dtype=np.float64)
        self.B = np.empty(2 * nlayer, dtype=np.float64)
        self.C = np.empty(2 * nlayer, dtype=np.float64)
        self.D = np.empty(2 * nlayer, dtype=np.float64)
        self.positive = np.empty(nlayer, dtype=np.float64)
        self.negative = np.empty(nlayer, dtype=np.float64)
        self.xint = np.empty(nlayer + 1, dtype=np.float64)
GetReflected1DWorkspaceType = GetReflected1DWorkspace.class_type.instance_type

@jitclass
class GetReflected1D:
    """Persistent reflected-light solver state and outputs.

    The constructor allocates the result arrays needed for the requested
    output modes and creates a typed list of per-thread
    :class:`GetReflected1DWorkspace` instances for the parallel wavelength
    loop.

    Parameters
    ----------
    nlayer : int
        Number of atmospheric layers.
    nwno : int
        Number of wavelength points.
    numg : int
        Number of Gauss angles.
    numt : int
        Number of Chebyshev angles.
    get_lvl_flux : int
        Flag indicating whether level flux outputs should be allocated.
    get_toa_intensity : int
        Flag indicating whether TOA intensity outputs should be allocated.
    """

    # Dimensions and settings
    nlayer : types.int64
    nwno : types.int64
    numg : types.int64
    numt : types.int64
    get_lvl_flux : types.int64
    get_toa_intensity : types.int64

    # Results
    flux_minus_all : types.double[:, :, :, :]
    flux_plus_all : types.double[:, :, :, :]
    flux_minus_midpt_all : types.double[:, :, :, :]
    flux_plus_midpt_all : types.double[:, :, :, :]
    xint_at_top : types.double[:, :, :]

    # Workspace
    workspace : types.ListType(GetReflected1DWorkspaceType)

    def __init__(self, nlayer, nwno, numg, numt, get_lvl_flux, get_toa_intensity):

        self._allocate_results(nlayer, nwno, numg, numt, get_lvl_flux, get_toa_intensity)
        self._allocate_workspace(nlayer)

    def _ensure(self, nlayer, nwno, numg, numt, get_lvl_flux, get_toa_intensity):
        "Re-allocates work and result arrays if necessary."

        self._ensure_results(nlayer, nwno, numg, numt, get_lvl_flux, get_toa_intensity)
        self._ensure_workspace(nlayer)
        
    def _ensure_results(self, nlayer, nwno, numg, numt, get_lvl_flux, get_toa_intensity):
        "Re-allocates result arrays if necessary."

        ok = self.nlayer == nlayer
        ok = ok and self.nwno == nwno
        ok = ok and self.numg == numg
        ok = ok and self.numt == numt
        ok = ok and self.get_lvl_flux == get_lvl_flux
        ok = ok and self.get_toa_intensity == get_toa_intensity

        if not ok:
            self._allocate_results(nlayer, nwno, numg, numt, get_lvl_flux, get_toa_intensity)

    def _allocate_results(self, nlayer, nwno, numg, numt, get_lvl_flux, get_toa_intensity):
        "Allocates result arrays."

        # Dimensions and settings
        self.nlayer = nlayer
        self.nwno = nwno
        self.numg = numg
        self.numt = numt
        self.get_lvl_flux = get_lvl_flux
        self.get_toa_intensity = get_toa_intensity

        # Results
        if get_lvl_flux:
            self.flux_minus_all = np.empty((numg, numt, nlayer + 1, nwno), dtype=np.float64)
            self.flux_plus_all = np.empty((numg, numt, nlayer + 1, nwno), dtype=np.float64)
            self.flux_minus_midpt_all = np.empty((numg, numt, nlayer + 1, nwno), dtype=np.float64)
            self.flux_plus_midpt_all = np.empty((numg, numt, nlayer + 1, nwno), dtype=np.float64)
        else:
            self.flux_minus_all = np.empty((0,0,0,0), dtype=np.float64)
            self.flux_plus_all = np.empty((0,0,0,0), dtype=np.float64)
            self.flux_minus_midpt_all = np.empty((0,0,0,0), dtype=np.float64)
            self.flux_plus_midpt_all = np.empty((0,0,0,0), dtype=np.float64)
        if get_toa_intensity:
            self.xint_at_top = np.empty((numg, numt, nwno), dtype=np.float64)
        else:
            self.xint_at_top = np.empty((0,0,0), dtype=np.float64)

    def _ensure_workspace(self, nlayer):
        "Re-allocates work list/arrays if necessary."

        if nlayer != self.nlayer or len(self.workspace) != nb.get_num_threads():
            self._allocate_workspace(nlayer)

    def _allocate_workspace(self, nlayer):
        "Allocates work list/arrays."

        nthreads = nb.get_num_threads()
        self.workspace = nb.typed.List.empty_list(GetReflected1DWorkspaceType)
        for i in range(nthreads):
            self.workspace.append(GetReflected1DWorkspace(nlayer))

    def call(self, *args):
        "Calls core solver"
        get_reflected_1d(self, *args)

# Do not cache this function, as Numba says I should not.
@nb.njit(parallel=True)
def get_reflected_1d(
    self,
    nlevel,
    wno,
    nwno,
    numg,
    numt,
    dtau,
    tau,
    w0,
    cosb,
    gcos2,
    ftau_cld,
    ftau_ray,
    dtau_og,
    tau_og,
    w0_og,
    cosb_og,
    surf_reflect,
    ubar0,
    ubar1,
    cos_theta,
    F0PI,
    single_phase,
    multi_phase,
    frac_a,
    frac_b,
    frac_c,
    constant_back,
    constant_forward,
    get_toa_intensity,
    get_lvl_flux,
    toon_coefficients,
    b_top
):
    """Compute reflected-light fluxes and intensities in place.

    The solver iterates over wavelength in parallel and writes results into the
    persistent output arrays owned by ``self``. Per-thread scratch space is
    taken from ``self.workspace`` so the wavelength loop can reuse memory
    without allocating inside the hot path.

    Parameters
    ----------
    self : GetReflected1D
        Persistent solver container holding the output arrays and per-thread
        workspace objects.
    nlevel : int
        Number of atmospheric levels.
    wno : ndarray of float64
        Wavenumber grid. Retained for API compatibility with the allocating
        solver.
    nwno : int
        Number of wavelength points.
    numg : int
        Number of Gauss angles.
    numt : int
        Number of Chebyshev angles.
    dtau : ndarray of float64, shape (nlevel - 1, nwno)
        Layer optical depths with the D-Eddington correction applied.
    tau : ndarray of float64, shape (nlevel, nwno)
        Cumulative optical depths with the D-Eddington correction applied.
    w0 : ndarray of float64, shape (nlevel - 1, nwno)
        Single-scattering albedo with the D-Eddington correction applied.
    cosb : ndarray of float64, shape (nlevel - 1, nwno)
        Asymmetry factor with the D-Eddington correction applied.
    gcos2 : ndarray of float64, shape (nlevel - 1, nwno)
        Rayleigh-scattering correction term used in the single-scattering phase
        function.
    ftau_cld : ndarray of float64, shape (nlevel - 1, nwno)
        Fraction of extinction contributed by clouds.
    ftau_ray : ndarray of float64, shape (nlevel - 1, nwno)
        Fraction of extinction contributed by Rayleigh scattering.
    dtau_og : ndarray of float64, shape (nlevel - 1, nwno)
        Uncorrected layer optical depths.
    tau_og : ndarray of float64, shape (nlevel, nwno)
        Uncorrected cumulative optical depths.
    w0_og : ndarray of float64, shape (nlevel - 1, nwno)
        Uncorrected single-scattering albedo.
    cosb_og : ndarray of float64, shape (nlevel - 1, nwno)
        Uncorrected asymmetry factor.
    surf_reflect : ndarray of float64, shape (nwno,)
        Wavelength-dependent surface reflectivity.
    ubar0 : ndarray of float64, shape (numg, numt)
        Cosine of the incident angle for each Gauss/Chebyshev pair.
    ubar1 : ndarray of float64, shape (numg, numt)
        Cosine of the observer angle for each Gauss/Chebyshev pair.
    cos_theta : float
        Cosine of the phase angle.
    F0PI : ndarray of float64, shape (nwno,)
        Incident solar flux divided by pi.
    single_phase : int
        Selector for the single-scattering phase function.
    multi_phase : int
        Selector for the multiple-scattering phase function.
    frac_a : float
        TTHG forward/back-scattering mix coefficient.
    frac_b : float
        TTHG forward/back-scattering mix coefficient.
    frac_c : float
        TTHG forward/back-scattering mix coefficient exponent.
    constant_back : float
        Back-scattering asymmetry parameter for the phase function.
    constant_forward : float
        Forward-scattering asymmetry parameter for the phase function.
    get_toa_intensity : int
        If nonzero, compute and store TOA intensities.
    get_lvl_flux : int
        If nonzero, compute and store level and midpoint fluxes.
    toon_coefficients : int
        Selects the Toon et al. coefficient set.
    b_top : float
        Diffuse radiation incident at the top of the atmosphere.

    Returns
    -------
    None
        Results are written into ``self`` in place.
    """
    
    nlayer = nlevel - 1
    
    # Ensure result + work arrays are right length
    self._ensure(nlayer, nwno, numg, numt, get_lvl_flux, get_toa_intensity)

    # Parallel loop over wavelength
    for w in nb.prange(nwno):
        # Get workspace for given thread
        wrk = self.workspace[nb.get_thread_id()]

        # Call reflected light solver for single wavelength
        get_reflected_1d_w(
            w,
            wrk,
            nlevel,
            wno,
            nwno,
            numg,
            numt,
            dtau,
            tau,
            w0,
            cosb,
            gcos2,
            ftau_cld,
            ftau_ray,
            dtau_og,
            tau_og,
            w0_og,
            cosb_og,
            surf_reflect,
            ubar0,
            ubar1,
            cos_theta,
            F0PI,
            single_phase,
            multi_phase,
            frac_a,
            frac_b,
            frac_c,
            constant_back,
            constant_forward,
            get_toa_intensity,
            get_lvl_flux,
            toon_coefficients,
            b_top,
            self.flux_minus_all,
            self.flux_plus_all,
            self.flux_minus_midpt_all,
            self.flux_plus_midpt_all,
            self.xint_at_top
        )
        
@nb.njit(cache=True)
def get_reflected_1d_w(
    w,
    wrk,
    nlevel,
    wno,
    nwno,
    numg,
    numt,
    dtau,
    tau,
    w0,
    cosb,
    gcos2,
    ftau_cld,
    ftau_ray,
    dtau_og,
    tau_og,
    w0_og,
    cosb_og,
    surf_reflect,
    ubar0,
    ubar1,
    cos_theta,
    F0PI,
    single_phase,
    multi_phase,
    frac_a,
    frac_b,
    frac_c,
    constant_back,
    constant_forward,
    get_toa_intensity,
    get_lvl_flux,
    toon_coefficients,
    b_top,
    flux_minus_all,
    flux_plus_all,
    flux_minus_midpt_all,
    flux_plus_midpt_all,
    xint_at_top
):
    "Per-wavelength reflected-light solve used by :func:`get_reflected_1d`."
    
    # Some constants
    nlayer = nlevel - 1
    sq3 = np.sqrt(3.0)

    # Unpack workspace
    g1 = wrk.g1
    g2 = wrk.g2
    lamda = wrk.lamda
    gama = wrk.gama
    g3 = wrk.g3
    a_minus = wrk.a_minus
    a_plus = wrk.a_plus
    c_minus_up = wrk.c_minus_up
    c_plus_up = wrk.c_plus_up
    c_minus_down = wrk.c_minus_down
    c_plus_down = wrk.c_plus_down
    exptrm = wrk.exptrm
    exptrm_positive = wrk.exptrm_positive
    exptrm_minus = wrk.exptrm_minus
    p_single = wrk.p_single
    A = wrk.A
    B = wrk.B
    C = wrk.C
    D = wrk.D
    positive = wrk.positive
    negative = wrk.negative
    xint = wrk.xint

    f0pi_w = F0PI[w]
    surf_reflect_w = surf_reflect[w]

    if toon_coefficients == 1:
        for i in range(nlayer):
            w0_iw = w0[i, w]
            ft_iw = ftau_cld[i, w]
            cb_iw = cosb[i, w]
            g1[i] = (7.0 - w0_iw * (4.0 + 3.0 * ft_iw * cb_iw)) / 4.0
            g2[i] = -(1.0 - w0_iw * (4.0 - 3.0 * ft_iw * cb_iw)) / 4.0
    elif toon_coefficients == 0:
        for i in range(nlayer):
            w0_iw = w0[i, w]
            ft_iw = ftau_cld[i, w]
            cb_iw = cosb[i, w]
            g1[i] = (sq3 * 0.5) * (2.0 - w0_iw * (1.0 + ft_iw * cb_iw))
            g2[i] = (sq3 * w0_iw * 0.5) * (1.0 - ft_iw * cb_iw)

    for i in range(nlayer):
        lamda_i = np.sqrt(g1[i] * g1[i] - g2[i] * g2[i])
        lamda[i] = lamda_i
        gama[i] = (g1[i] - lamda_i) / g2[i]

    if get_toa_intensity:
        for i in range(nlayer):
            g_forward = 0.0
            g_back = 0.0
            f = 0.0
            if single_phase != 1:
                g_forward = constant_forward * cosb_og[i, w]
                g_back = constant_back * cosb_og[i, w]
                f = frac_a + frac_b * g_back ** frac_c

            if single_phase == 0:
                HG_forward = (1.0 - g_forward * g_forward) / np.sqrt((1.0 + g_forward * g_forward + 2.0 * g_forward * cos_theta) ** 3)
                HG_backward = (1.0 - g_back * g_back) / np.sqrt((1.0 + g_back * g_back + 2.0 * g_back * cos_theta) ** 3)
                p_single[i] = f * HG_forward + (1.0 - f) * HG_backward + gcos2[i, w]
            elif single_phase == 1:
                cb = cosb_og[i, w]
                p_single[i] = (1.0 - cb * cb) / np.sqrt((1.0 + cb * cb + 2.0 * cb * cos_theta) ** 3)
            elif single_phase == 2:
                HG_forward = (1.0 - g_forward * g_forward) / np.sqrt((1.0 + g_forward * g_forward + 2.0 * g_forward * cos_theta) ** 3)
                HG_backward = (1.0 - g_back * g_back) / np.sqrt((1.0 + g_back * g_back + 2.0 * g_back * cos_theta) ** 3)
                p_single[i] = f * HG_forward + (1.0 - f) * HG_backward
            elif single_phase == 3:
                HG_forward = (1.0 - g_forward * g_forward) / np.sqrt((1.0 + g_forward * g_forward + 2.0 * g_forward * cos_theta) ** 3)
                HG_back = (1.0 - g_back * g_back) / np.sqrt((1.0 + g_back * g_back + 2.0 * g_back * cos_theta) ** 3)
                p_single[i] = ftau_cld[i, w] * (f * HG_forward + (1.0 - f) * HG_back) + ftau_ray[i, w] * (0.75 * (1.0 + cos_theta * cos_theta))

    for ng in range(numg):
        for nt in range(numt):
            u1 = ubar1[ng, nt]
            u0 = ubar0[ng, nt]
            inv_u0 = 1.0 / u0
            inv_u0_sq = inv_u0 * inv_u0
            inv_u1 = 1.0 / u1
            sum_u = u0 + u1
            inv_sum_u = 1.0 / sum_u
            inv_u0u1 = inv_u0 * inv_u1

            if toon_coefficients == 1:
                for i in range(nlayer):
                    g3[i] = (2.0 - 3.0 * ftau_cld[i, w] * cosb[i, w] * u0) / 4.0
            elif toon_coefficients == 0:
                for i in range(nlayer):
                    g3[i] = 0.5 * (1.0 - sq3 * ftau_cld[i, w] * cosb[i, w] * u0)

            for i in range(nlayer):
                g4 = 1.0 - g3[i]
                denom = lamda[i] * lamda[i] - inv_u0_sq
                w0_iw = w0[i, w]
                a_minus[i] = f0pi_w * w0_iw * (g4 * (g1[i] + inv_u0) + g2[i] * g3[i]) / denom
                a_plus[i] = f0pi_w * w0_iw * (g3[i] * (g1[i] - inv_u0) + g2[i] * g4) / denom

                exp_up = np.exp(-tau[i, w] / u0)
                exp_down = np.exp(-tau[i + 1, w] / u0)
                c_minus_up[i] = a_minus[i] * exp_up
                c_plus_up[i] = a_plus[i] * exp_up
                c_minus_down[i] = a_minus[i] * exp_down
                c_plus_down[i] = a_plus[i] * exp_down

                exptrm_val = lamda[i] * dtau[i, w]
                if exptrm_val > 35.0:
                    exptrm_val = 35.0
                exptrm[i] = exptrm_val
                exptrm_positive[i] = np.exp(exptrm_val)
                exptrm_minus[i] = 1.0 / exptrm_positive[i]

            b_surface = surf_reflect_w * u0 * f0pi_w * np.exp(-tau[nlevel - 1, w] * inv_u0)
            setup_tri_diag_inplace(
                A,
                B,
                C,
                D,
                nlayer,
                c_plus_up,
                c_minus_up,
                c_plus_down,
                c_minus_down,
                b_top,
                b_surface,
                surf_reflect_w,
                gama,
                exptrm_positive,
                exptrm_minus,
            )
            tri_diag_solve_inplace(2 * nlayer, A, B, C, D)

            for i in range(nlayer):
                positive[i] = D[2 * i] + D[2 * i + 1]
                negative[i] = D[2 * i] - D[2 * i + 1]

            if get_lvl_flux:
                for i in range(nlevel):
                    flux_minus_all[ng, nt, i, w] = 0.0
                    flux_plus_all[ng, nt, i, w] = 0.0
                    flux_minus_midpt_all[ng, nt, i, w] = 0.0
                    flux_plus_midpt_all[ng, nt, i, w] = 0.0

                for i in range(nlayer):
                    flux_minus_all[ng, nt, i, w] = positive[i] * gama[i] + negative[i] + c_minus_up[i]
                    flux_plus_all[ng, nt, i, w] = positive[i] + gama[i] * negative[i] + c_plus_up[i]

                flux_minus_all[ng, nt, nlayer, w] = (
                    gama[nlayer - 1] * positive[nlayer - 1] * exptrm_positive[nlayer - 1]
                    + negative[nlayer - 1] * exptrm_minus[nlayer - 1]
                    + c_minus_down[nlayer - 1]
                )
                flux_plus_all[ng, nt, nlayer, w] = (
                    positive[nlayer - 1] * exptrm_positive[nlayer - 1]
                    + gama[nlayer - 1] * negative[nlayer - 1] * exptrm_minus[nlayer - 1]
                    + c_plus_down[nlayer - 1]
                )

                u0_scale = u0 * f0pi_w
                for i in range(nlevel):
                    flux_minus_all[ng, nt, i, w] = flux_minus_all[ng, nt, i, w] + u0_scale * np.exp(-tau[i, w] * inv_u0)

                for i in range(nlayer):
                    exptrm_positive_midpt = np.exp(0.5 * exptrm[i])
                    exptrm_minus_midpt = 1.0 / exptrm_positive_midpt
                    taumid = tau[i, w] + 0.5 * dtau[i, w]
                    c_plus_mid = a_plus[i] * np.exp(-taumid / ubar0[ng, nt])
                    c_minus_mid = a_minus[i] * np.exp(-taumid / ubar0[ng, nt])

                    flux_minus_midpt_all[ng, nt, i, w] = (
                        gama[i] * positive[i] * exptrm_positive_midpt
                        + negative[i] * exptrm_minus_midpt
                        + c_minus_mid
                        + ubar0[ng, nt] * f0pi_w * np.exp(-taumid / ubar0[ng, nt])
                    )
                    flux_plus_midpt_all[ng, nt, i, w] = (
                        positive[i] * exptrm_positive_midpt
                        + gama[i] * negative[i] * exptrm_minus_midpt
                        + c_plus_mid
                    )
                flux_minus_midpt_all[ng, nt, nlayer, w] = 0.0
                flux_plus_midpt_all[ng, nt, nlayer, w] = 0.0

            if get_toa_intensity:
                flux_zero = (
                    positive[nlayer - 1] * exptrm_positive[nlayer - 1]
                    + gama[nlayer - 1] * negative[nlayer - 1] * exptrm_minus[nlayer - 1]
                    + c_plus_down[nlayer - 1]
                )
                xint[nlayer] = flux_zero / np.pi

                for i in range(nlayer - 1, -1, -1):
                    if multi_phase == 0:
                        ubar2 = 0.767
                        phase_term = 3.0 * ubar2 * ubar2 * u1 * u1 - 1.0
                        multi_plus = 1.0 + 1.5 * ftau_cld[i, w] * cosb[i, w] * u1 + gcos2[i, w] * phase_term / 2.0
                        multi_minus = 1.0 - 1.5 * ftau_cld[i, w] * cosb[i, w] * u1 + gcos2[i, w] * phase_term / 2.0
                    elif multi_phase == 1:
                        multi_plus = 1.0 + 1.5 * ftau_cld[i, w] * cosb[i, w] * u1
                        multi_minus = 1.0 - 1.5 * ftau_cld[i, w] * cosb[i, w] * u1

                    G = positive[i] * (multi_plus + gama[i] * multi_minus) * w0[i, w] * 0.5 / np.pi
                    H = negative[i] * (gama[i] * multi_plus + multi_minus) * w0[i, w] * 0.5 / np.pi
                    source_A = (multi_plus * c_plus_up[i] + multi_minus * c_minus_up[i]) * w0[i, w] * 0.5 / np.pi

                    xint[i] = (
                        xint[i + 1] * np.exp(-dtau[i, w] * inv_u1)
                        + (w0_og[i, w] * (f0pi_w * 0.25 / np.pi))
                        * p_single[i]
                        * np.exp(-tau_og[i, w] * inv_u0)
                        * (1.0 - np.exp(-dtau_og[i, w] * sum_u * inv_u0u1))
                        * (u0 * inv_sum_u)
                        + source_A * (1.0 - np.exp(-dtau[i, w] * sum_u * inv_u0u1))
                        * (u0 * inv_sum_u)
                        + G * (np.exp(exptrm[i] - dtau[i, w] * inv_u1) - 1.0) / (lamda[i] * u1 - 1.0)
                        + H * (1.0 - np.exp(-(exptrm[i] + dtau[i, w] * inv_u1))) / (lamda[i] * u1 + 1.0)
                    )

                xint_at_top[ng, nt, w] = xint[0]
