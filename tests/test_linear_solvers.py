"""
Tests for the banded linear solvers in picaso.fluxes. tri_diag_solve solves the two-stream system
set up by setup_tri_diag at every wavenumber, in every thermal, reflected and climate calculation.
Each is checked against a dense numpy.linalg.solve.
"""
import numpy as np
import pytest
from numba.core.errors import TypingError

from picaso.fluxes import pent_diag_solve, setup_tri_diag, tri_diag_solve

RNG_SEED = 2026


def tridiagonal_matrix(a, b, c):
    """Dense matrix for a(i) x(i-1) + b(i) x(i) + c(i) x(i+1) = d(i); a[0] and c[-1] are unused."""
    return np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)


@pytest.mark.parametrize('n', [1, 2, 5, 50, 400])
def test_tri_diag_solve_matches_dense_solve(n):
    rng = np.random.default_rng(RNG_SEED + n)
    a, c, d = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    b = np.abs(a) + np.abs(c) + 1 + rng.random(n)  # diagonally dominant, so well posed
    x = tri_diag_solve(n, a, b, c, d)
    np.testing.assert_allclose(x, np.linalg.solve(tridiagonal_matrix(a, b, c), d), rtol=1e-12, atol=1e-14)


def test_tri_diag_solve_ignores_the_unused_corners():
    rng = np.random.default_rng(RNG_SEED)
    n = 6
    a, c, d = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    b = np.abs(a) + np.abs(c) + 1
    x = tri_diag_solve(n, a, b, c, d)
    a[0], c[-1] = 99.0, -77.0
    np.testing.assert_array_equal(tri_diag_solve(n, a, b, c, d), x)


@pytest.mark.parametrize('max_layer_dtau', [1.0, 100.0, 1e4])
def test_tri_diag_solve_on_toon_two_stream_systems(max_layer_dtau):
    """
    The systems PICASO actually solves: setup_tri_diag with Toon et al. (1989) coefficients for layers
    from optically thin to very thick. Thick layers make the matrix extremely ill conditioned (the
    exponential terms reach e^35), so this checks the solver still matches a dense solve.
    """
    rng = np.random.default_rng(RNG_SEED)
    nlayer, nwno = 60, 5
    w0 = rng.uniform(0.01, 0.99, (nlayer, nwno))
    cosb = rng.uniform(0.0, 0.9, (nlayer, nwno))
    dtau = np.logspace(-4, np.log10(max_layer_dtau), nlayer)[:, np.newaxis] * np.ones(nwno)

    # hemispheric mean coefficients, as in get_thermal_1d
    g1, g2 = 2 - w0 * (1 + cosb), w0 * (1 - cosb)
    lamda = np.sqrt(g1**2 - g2**2)
    gama = (g1 - lamda) / g2
    exptrm_positive = np.exp(np.minimum(lamda * dtau, 35.0))  # same overflow cap as the solvers
    c_plus_up, c_minus_up, c_plus_down, c_minus_down = (rng.uniform(0.1, 1, (nlayer, nwno)) for _ in range(4))

    A, B, C, D = setup_tri_diag(nlayer, nwno, c_plus_up, c_minus_up, c_plus_down, c_minus_down, 0.1,
                                rng.uniform(0.1, 1, nwno), rng.uniform(0, 0.5, nwno), gama, dtau,
                                exptrm_positive, 1 / exptrm_positive)
    for w in range(nwno):
        x = tri_diag_solve(2 * nlayer, A[:, w], B[:, w], C[:, w], D[:, w])
        matrix = tridiagonal_matrix(A[:, w], B[:, w], C[:, w])
        np.testing.assert_allclose(x, np.linalg.solve(matrix, D[:, w]), rtol=1e-10)
        residual = np.max(np.abs(matrix @ x - D[:, w])) / np.max(np.abs(D[:, w]))
        assert residual < 1e-12


@pytest.mark.xfail(strict=True, raises=TypingError, reason=(
    "pent_diag_solve calls pp.solve, but 'import pentapy as pp' was removed from fluxes.py "
    "(commit 95ca469), so it no longer compiles. Its call sites are all commented out."))
def test_pent_diag_solve_matches_dense_solve():
    """
    Bands in pentapy's flat layout: E and D are the second and first super-diagonals (stored from the
    start), C the diagonal, B and A the first and second sub-diagonals (stored to the end).
    """
    rng = np.random.default_rng(RNG_SEED)
    n = 8
    E, D, B, A = (rng.normal(size=n) for _ in range(4))
    C = np.abs(A) + np.abs(B) + np.abs(D) + np.abs(E) + 1
    F = rng.normal(size=n)
    matrix = (np.diag(C) + np.diag(D[:-1], 1) + np.diag(E[:-2], 2)
              + np.diag(B[1:], -1) + np.diag(A[2:], -2))
    x = pent_diag_solve(n, A, B, C, D, E, F)
    np.testing.assert_allclose(x, np.linalg.solve(matrix, F), rtol=1e-12)
