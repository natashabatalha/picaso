from pathlib import Path
import sys

import numba as nb
import numpy as np
import pytest

# Get the root of the repo and prepend to path.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Import fluxes
from picaso import fluxes
from picaso import fluxes_noalloc

def _make_reflected_case():
    return dict(
        nlevel=4,
        nwno=3,
        numg=2,
        numt=2,
        wno=np.array([1.0, 2.0, 3.0]),
        dtau=np.array(
            [[0.11, 0.12, 0.13],
             [0.21, 0.22, 0.23],
             [0.31, 0.32, 0.33]]
        ),
        tau=np.array(
            [[0.01, 0.02, 0.03],
             [0.14, 0.15, 0.16],
             [0.37, 0.38, 0.39],
             [0.70, 0.71, 0.72]]
        ),
        w0=np.array(
            [[0.61, 0.62, 0.63],
             [0.64, 0.65, 0.66],
             [0.67, 0.68, 0.69]]
        ),
        cosb=np.array(
            [[0.11, 0.12, 0.13],
             [0.21, 0.22, 0.23],
             [0.31, 0.32, 0.33]]
        ),
        gcos2=np.array(
            [[0.05, 0.06, 0.07],
             [0.08, 0.09, 0.10],
             [0.11, 0.12, 0.13]]
        ),
        ftau_cld=np.array(
            [[0.41, 0.42, 0.43],
             [0.44, 0.45, 0.46],
             [0.47, 0.48, 0.49]]
        ),
        ftau_ray=np.array(
            [[0.19, 0.20, 0.21],
             [0.22, 0.23, 0.24],
             [0.25, 0.26, 0.27]]
        ),
        dtau_og=np.array(
            [[0.09, 0.10, 0.11],
             [0.12, 0.13, 0.14],
             [0.15, 0.16, 0.17]]
        ),
        tau_og=np.array(
            [[0.02, 0.03, 0.04],
             [0.18, 0.19, 0.20],
             [0.42, 0.43, 0.44],
             [0.77, 0.78, 0.79]]
        ),
        w0_og=np.array(
            [[0.51, 0.52, 0.53],
             [0.54, 0.55, 0.56],
             [0.57, 0.58, 0.59]]
        ),
        cosb_og=np.array(
            [[0.14, 0.15, 0.16],
             [0.24, 0.25, 0.26],
             [0.34, 0.35, 0.36]]
        ),
        surf_reflect=np.array([0.23, 0.24, 0.25]),
        ubar0=np.array(
            [[0.41, 0.42],
             [0.51, 0.52]]
        ),
        ubar1=np.array(
            [[0.61, 0.62],
             [0.71, 0.72]]
        ),
        cos_theta=0.37,
        F0PI=np.array([1.11, 1.12, 1.13]),
        frac_a=0.17,
        frac_b=0.27,
        frac_c=1.3,
        constant_back=0.29,
        constant_forward=0.39,
        b_top=0.07,
    )


def _call_reflected_alloc(case, single_phase, multi_phase, get_toa_intensity, get_lvl_flux, toon_coefficients):
    return fluxes.get_reflected_1d(
        case["nlevel"],
        case["wno"].copy(),
        case["nwno"],
        case["numg"],
        case["numt"],
        case["dtau"].copy(),
        case["tau"].copy(),
        case["w0"].copy(),
        case["cosb"].copy(),
        case["gcos2"].copy(),
        case["ftau_cld"].copy(),
        case["ftau_ray"].copy(),
        case["dtau_og"].copy(),
        case["tau_og"].copy(),
        case["w0_og"].copy(),
        case["cosb_og"].copy(),
        case["surf_reflect"],
        case["ubar0"].copy(),
        case["ubar1"].copy(),
        case["cos_theta"],
        case["F0PI"],
        single_phase,
        multi_phase,
        case["frac_a"],
        case["frac_b"],
        case["frac_c"],
        case["constant_back"],
        case["constant_forward"],
        get_toa_intensity=get_toa_intensity,
        get_lvl_flux=get_lvl_flux,
        toon_coefficients=toon_coefficients,
        b_top=case["b_top"],
    )


def _assert_reflected_parity(case, single_phase, multi_phase, get_toa_intensity, get_lvl_flux, toon_coefficients):
    expected_xint, expected_fluxes = _call_reflected_alloc(
        case,
        single_phase,
        multi_phase,
        get_toa_intensity,
        get_lvl_flux,
        toon_coefficients,
    )

    reflected = fluxes_noalloc.GetReflected1D(
        case["nlevel"] - 1,
        case["nwno"],
        case["numg"],
        case["numt"],
        get_lvl_flux,
        get_toa_intensity,
    )

    fluxes_noalloc.get_reflected_1d(
        reflected,
        case["nlevel"],
        case["wno"].copy(),
        case["nwno"],
        case["numg"],
        case["numt"],
        case["dtau"].copy(),
        case["tau"].copy(),
        case["w0"].copy(),
        case["cosb"].copy(),
        case["gcos2"].copy(),
        case["ftau_cld"].copy(),
        case["ftau_ray"].copy(),
        case["dtau_og"].copy(),
        case["tau_og"].copy(),
        case["w0_og"].copy(),
        case["cosb_og"].copy(),
        case["surf_reflect"],
        case["ubar0"].copy(),
        case["ubar1"].copy(),
        case["cos_theta"],
        case["F0PI"],
        single_phase,
        multi_phase,
        case["frac_a"],
        case["frac_b"],
        case["frac_c"],
        case["constant_back"],
        case["constant_forward"],
        get_toa_intensity,
        get_lvl_flux,
        toon_coefficients,
        case["b_top"],
    )

    if get_toa_intensity:
        np.testing.assert_allclose(reflected.xint_at_top, expected_xint)
    else:
        assert reflected.xint_at_top.size == 0

    if get_lvl_flux:
        np.testing.assert_allclose(reflected.flux_minus_all, expected_fluxes[0])
        np.testing.assert_allclose(reflected.flux_plus_all, expected_fluxes[1])
        np.testing.assert_allclose(reflected.flux_minus_midpt_all, expected_fluxes[2])
        np.testing.assert_allclose(reflected.flux_plus_midpt_all, expected_fluxes[3])
    else:
        assert reflected.flux_minus_all.size == 0
        assert reflected.flux_plus_all.size == 0
        assert reflected.flux_minus_midpt_all.size == 0
        assert reflected.flux_plus_midpt_all.size == 0

def test_tri_diag_solve_matches_inplace():
    l = 5
    a = np.array([0.0, -1.0, -1.0, -1.0, -1.0])
    b = np.array([4.0, 4.0, 4.0, 4.0, 4.0])
    c = np.array([-1.0, -1.0, -1.0, -1.0, 0.0])
    d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    expected = fluxes.tri_diag_solve(l, a.copy(), b.copy(), c.copy(), d.copy())

    d_inplace = d.copy()
    result = fluxes_noalloc.tri_diag_solve_inplace(l, a.copy(), b.copy(), c.copy(), d_inplace)

    assert result is None
    np.testing.assert_allclose(d_inplace, expected)


def test_setup_tri_diag_matches_inplace():
    nlayer = 3
    nwno = 4

    c_plus_up = np.array(
        [[0.11, 0.12, 0.13, 0.14],
         [0.21, 0.22, 0.23, 0.24],
         [0.31, 0.32, 0.33, 0.34]]
    )
    c_minus_up = np.array(
        [[0.51, 0.52, 0.53, 0.54],
         [0.61, 0.62, 0.63, 0.64],
         [0.71, 0.72, 0.73, 0.74]]
    )
    c_plus_down = np.array(
        [[0.15, 0.16, 0.17, 0.18],
         [0.25, 0.26, 0.27, 0.28],
         [0.35, 0.36, 0.37, 0.38]]
    )
    c_minus_down = np.array(
        [[0.55, 0.56, 0.57, 0.58],
         [0.65, 0.66, 0.67, 0.68],
         [0.75, 0.76, 0.77, 0.78]]
    )
    b_top = 0.19
    b_surface = np.array([1.01, 1.02, 1.03, 1.04])
    surf_reflect = np.array([0.27, 0.28, 0.29, 0.30])
    gama = np.array(
        [[0.41, 0.42, 0.43, 0.44],
         [0.51, 0.52, 0.53, 0.54],
         [0.61, 0.62, 0.63, 0.64]]
    )
    dtau = np.array(
        [[0.91, 0.92, 0.93, 0.94],
         [1.01, 1.02, 1.03, 1.04],
         [1.11, 1.12, 1.13, 1.14]]
    )
    exptrm_positive = np.array(
        [[1.21, 1.22, 1.23, 1.24],
         [1.31, 1.32, 1.33, 1.34],
         [1.41, 1.42, 1.43, 1.44]]
    )
    exptrm_minus = np.array(
        [[0.81, 0.82, 0.83, 0.84],
         [0.71, 0.72, 0.73, 0.74],
         [0.61, 0.62, 0.63, 0.64]]
    )

    expected = fluxes.setup_tri_diag(
        nlayer,
        nwno,
        c_plus_up.copy(),
        c_minus_up.copy(),
        c_plus_down.copy(),
        c_minus_down.copy(),
        b_top,
        b_surface.copy(),
        surf_reflect,
        gama.copy(),
        dtau.copy(),
        exptrm_positive.copy(),
        exptrm_minus.copy(),
    )

    w = 2
    A = np.zeros(2 * nlayer)
    B = np.zeros(2 * nlayer)
    C = np.zeros(2 * nlayer)
    D = np.zeros(2 * nlayer)

    result = fluxes_noalloc.setup_tri_diag_inplace(
        A,
        B,
        C,
        D,
        nlayer,
        c_plus_up[:, w].copy(),
        c_minus_up[:, w].copy(),
        c_plus_down[:, w].copy(),
        c_minus_down[:, w].copy(),
        b_top,
        b_surface[w],
        surf_reflect[w],
        gama[:, w].copy(),
        exptrm_positive[:, w].copy(),
        exptrm_minus[:, w].copy(),
    )

    assert result is None
    np.testing.assert_allclose(A, expected[0][:, w])
    np.testing.assert_allclose(B, expected[1][:, w])
    np.testing.assert_allclose(C, expected[2][:, w])
    np.testing.assert_allclose(D, expected[3][:, w])


@pytest.mark.parametrize(
    "single_phase,multi_phase,get_toa_intensity,get_lvl_flux,toon_coefficients",
    [
        (0, 0, 1, 1, 0),
        (1, 1, 1, 1, 1),
        (2, 1, 1, 0, 0),
        (3, 0, 0, 1, 1),
        (2, 0, 0, 0, 1),
    ],
)
def test_get_reflected_1d_matches_inplace(
    single_phase,
    multi_phase,
    get_toa_intensity,
    get_lvl_flux,
    toon_coefficients,
):
    case = _make_reflected_case()
    _assert_reflected_parity(
        case,
        single_phase,
        multi_phase,
        get_toa_intensity,
        get_lvl_flux,
        toon_coefficients,
    )


def test_get_reflected_1d_matches_inplace_two_threads():
    case = _make_reflected_case()
    previous_threads = nb.get_num_threads()
    nb.set_num_threads(2)
    try:
        _assert_reflected_parity(
            case,
            single_phase=0,
            multi_phase=0,
            get_toa_intensity=1,
            get_lvl_flux=1,
            toon_coefficients=0,
        )
    finally:
        nb.set_num_threads(previous_threads)
