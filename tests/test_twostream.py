from pathlib import Path
import sys

import numpy as np

# Get the root of the repo and prepend to path.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Import fluxes
from picaso import fluxes

tri_diag_solve = fluxes.tri_diag_solve
tri_diag_solve_inplace = fluxes.tri_diag_solve_inplace
setup_tri_diag = fluxes.setup_tri_diag
setup_tri_diag_inplace = fluxes.setup_tri_diag_inplace

def test_tri_diag_solve_matches_inplace():
    l = 5
    a = np.array([0.0, -1.0, -1.0, -1.0, -1.0])
    b = np.array([4.0, 4.0, 4.0, 4.0, 4.0])
    c = np.array([-1.0, -1.0, -1.0, -1.0, 0.0])
    d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    expected = tri_diag_solve(l, a.copy(), b.copy(), c.copy(), d.copy())

    d_inplace = d.copy()
    result = tri_diag_solve_inplace(l, a.copy(), b.copy(), c.copy(), d_inplace)

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
    surf_reflect = 0.27
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

    expected = setup_tri_diag(
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

    A = np.zeros((2 * nlayer, nwno))
    B = np.zeros((2 * nlayer, nwno))
    C = np.zeros((2 * nlayer, nwno))
    D = np.zeros((2 * nlayer, nwno))

    result = setup_tri_diag_inplace(
        A,
        B,
        C,
        D,
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

    assert result is None
    np.testing.assert_allclose(A, expected[0])
    np.testing.assert_allclose(B, expected[1])
    np.testing.assert_allclose(C, expected[2])
    np.testing.assert_allclose(D, expected[3])
