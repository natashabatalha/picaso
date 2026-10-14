"""
Analytic-limit and benchmark tests for the 1D reflected light solver, fluxes.get_reflected_1d
(Toon et al. 1989 two-stream with a single scattering correction), together with the full-phase
disk geometry (disco.get_angles_1d, disco.compute_disco) and disk integration
(disco.compress_disco) that spectrum(calculation='reflected') uses.

The inputs are small synthetic atmospheres (no opacity data), so these run in CI in about a second.
At each outgoing angle the reflectance is I/F = pi * xint / F0PI; compress_disco integrates that
over the visible disk to get the geometric albedo.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import REFDATA
from picaso.disco import compress_disco, compute_disco, get_angles_1d
from picaso.fluxes import get_reflected_1d

NWNO = 3
WNO = np.array([1e4, 1.5e4, 2e4])
F0PI = np.ones(NWNO)
W0_CONSERVATIVE = 0.999999  # w0 = 1 exactly divides by zero; picaso itself caps w0 just below 1
W0_ABSORBING = 1e-6
# default TTHG parameters from reference/config.json: fraction (A, B, C), constant_back, constant_forward
TTHG = (1., -1., 2., -0.5, 1.)
# single_phase: 1 = one-term Henyey-Greenstein, 3 = TTHG_ray (default); multi_phase: 0 = N=2 (default), 1 = N=1
OTHG, TTHG_RAY = 1, 3
N2, N1 = 0, 1


def geometry(ngauss=8):
    """Full-phase geometry, as inputs.phase_angle(0) sets it up for 1D spectra."""
    gangle, gweight, tangle, tweight = get_angles_1d(ngauss)
    ubar0, ubar1, cos_theta, _, _ = compute_disco(len(gangle), len(tangle), gangle, tangle, 0.0)
    return ubar0, ubar1, cos_theta, gweight, tweight


def reflected(dtau, w0, cosb=0.0, rayleigh=False, surf_reflect=0.0, nlayer=20, ngauss=8,
              single_phase=TTHG_RAY, multi_phase=N2, delta_eddington=False, get_lvl_flux=0):
    """
    Runs get_reflected_1d on a uniform atmosphere. dtau is the total optical depth (split evenly
    over nlayer layers) or an array of per-layer optical depths. rayleigh=True makes all scattering
    Rayleigh; otherwise it is a cloud with asymmetry cosb. delta_eddington scales the multiple
    scattering inputs the way optics.compute_opacity does.
    Returns (geometric albedo per wavenumber, I/F per outgoing angle, level fluxes).
    """
    ubar0, ubar1, cos_theta, gweight, tweight = geometry(ngauss)
    per_layer = np.atleast_1d(dtau) if np.ndim(dtau) else np.full(nlayer, dtau / nlayer)
    nlevel = len(per_layer) + 1
    dtau = np.repeat(per_layer[:, np.newaxis], NWNO, axis=1)
    w0 = np.full_like(dtau, w0)
    cosb = np.full_like(dtau, cosb)
    if rayleigh:
        ftau_cld, ftau_ray, gcos2 = np.zeros_like(w0), np.ones_like(w0), np.full_like(w0, 0.5)
    else:
        ftau_cld, ftau_ray, gcos2 = np.ones_like(w0), np.zeros_like(w0), np.zeros_like(w0)

    def cumulative(layer_dtau):
        tau = np.zeros((nlevel, NWNO))
        tau[1:] = np.cumsum(layer_dtau, axis=0)
        return tau

    # the multiple scattering inputs are delta-Eddington scaled; the *_og ones (used for the single
    # scattering term) never are
    if delta_eddington:
        f = cosb**2  # two-stream delta-Eddington: forward peak fraction f = g^2
        w0_ms, cosb_ms, dtau_ms = w0 * (1 - f) / (1 - w0 * f), (cosb - f) / (1 - f), dtau * (1 - w0 * f)
    else:
        w0_ms, cosb_ms, dtau_ms = w0, cosb, dtau
    xint, level_fluxes = get_reflected_1d(
        nlevel, WNO, NWNO, *ubar0.shape, dtau_ms, cumulative(dtau_ms), w0_ms, cosb_ms, gcos2, ftau_cld, ftau_ray,
        dtau, cumulative(dtau), w0, cosb,
        np.full(NWNO, surf_reflect), ubar0, ubar1, cos_theta, F0PI, single_phase, multi_phase, *TTHG,
        get_toa_intensity=1, get_lvl_flux=get_lvl_flux, toon_coefficients=0, b_top=0.)
    albedo = compress_disco(NWNO, cos_theta, xint, gweight, tweight, F0PI)
    return albedo, np.pi * xint[:, 0, :] / F0PI, level_fluxes


def test_black_planet_reflects_nothing():
    albedo, _, _ = reflected(100.0, W0_ABSORBING)
    assert np.all(albedo < 1e-6)


@pytest.mark.parametrize('surf_reflect', [0.3, 1.0])
def test_bare_lambert_surface_has_geometric_albedo_two_thirds_r(surf_reflect):
    """A Lambertian sphere with reflectivity r has geometric albedo 2r/3."""
    albedo, _, _ = reflected(1e-8, W0_ABSORBING, surf_reflect=surf_reflect)
    np.testing.assert_allclose(albedo, 2 / 3 * surf_reflect, rtol=2e-3)


@pytest.mark.parametrize('rayleigh, phase_at_180', [(True, 1.5), (False, 1.0)])
def test_thin_layer_matches_exact_single_scattering(rayleigh, phase_at_180):
    """
    For a thin layer, light scatters once. At full phase (mu0 = mu) the exact reflectance is
    I/F = w0 P(180) / 4 * mu0 / (mu0 + mu) * (1 - exp(-tau (1/mu0 + 1/mu))).
    """
    tau = 1e-4
    _, i_over_f, _ = reflected(tau, W0_CONSERVATIVE, rayleigh=rayleigh)
    mu = geometry()[1][:, 0]
    exact = W0_CONSERVATIVE * phase_at_180 / 4 * 0.5 * (1 - np.exp(-2 * tau / mu))
    np.testing.assert_allclose(i_over_f, np.broadcast_to(exact[:, np.newaxis], i_over_f.shape), rtol=1e-3)


def test_disk_integration_converges_for_a_thin_layer():
    """
    A thin layer's brightness grows toward the limb (I/F ~ 1/mu), which the Gauss quadrature over
    the disk resolves slowly. With more angles the geometric albedo approaches the exact 0.75 tau.
    """
    tau = 1e-4
    albedos = [reflected(tau, W0_CONSERVATIVE, rayleigh=True, ngauss=n)[0][0] / tau for n in [5, 6, 7, 8]]
    assert np.all(np.diff(albedos) > 0) and albedos[-1] < 0.75
    assert albedos[-1] == pytest.approx(0.75, rel=0.08)


@pytest.mark.parametrize('dtau', [0.1, 1.0, 10.0])
def test_conservative_scattering_conserves_energy(dtau):
    """With no absorption, what is not reflected is transmitted: the net flux is the same at every level."""
    _, _, (flux_minus, flux_plus, _, _) = reflected(dtau, W0_CONSERVATIVE, get_lvl_flux=1)
    net = flux_minus - flux_plus  # downward net flux, (angle, 1, level, wavenumber)
    np.testing.assert_allclose(net, np.broadcast_to(net[:, :, :1, :], net.shape), rtol=1e-3)
    reflected_fraction = flux_plus[:, :, 0, :] / flux_minus[:, :, 0, :]
    transmitted_fraction = net[:, :, -1, :] / flux_minus[:, :, 0, :]
    np.testing.assert_allclose(reflected_fraction + transmitted_fraction, 1, rtol=1e-3)


def test_absorption_removes_energy():
    _, _, (flux_minus, flux_plus, _, _) = reflected(1.0, 0.9, get_lvl_flux=1)
    net = flux_minus - flux_plus
    reflected_plus_transmitted = (flux_plus[:, :, 0, :] + net[:, :, -1, :]) / flux_minus[:, :, 0, :]
    assert np.all(reflected_plus_transmitted < 0.95)


def test_albedo_increases_with_scattering_and_thickness():
    by_w0 = [reflected(1000.0, w0)[0][0] for w0 in [0.3, 0.6, 0.9, 0.99]]
    assert np.all(np.diff(by_w0) > 0)
    by_tau = [reflected(tau, W0_CONSERVATIVE, rayleigh=True)[0][0] for tau in [0.01, 0.1, 1, 10]]
    assert np.all(np.diff(by_tau) > 0)


DLUGACH = pd.read_csv(REFDATA / 'base_cases' / 'testing' / 'DLUGACH_TEST.csv').set_index('asy')
DEEP_LAYERS = 10**np.linspace(-5, 3, 59)  # per-layer optical depth, as in model_compare.dlugach_test


@pytest.mark.parametrize('asy', DLUGACH.index)
def test_semi_infinite_geometric_albedo_matches_dlugach_yanovitskij(asy):
    """
    Geometric albedo of a semi-infinite atmosphere against Table XXI of Dlugach & Yanovitskij (1974),
    the benchmark of Batalha et al. (2019, Fig. 9), for every single scattering albedo in the table.

    Rayleigh and isotropic rows use picaso's default phase functions (TTHG_ray, N=2). The
    Henyey-Greenstein rows use one-term HG (OTHG, N=1) with delta-Eddington, because the default
    two-term HG adds a back-scattering lobe and is deliberately not the table's phase function.
    Tolerances are the two-stream accuracy as of 2026-10: <= 2.5% for Rayleigh and isotropic,
    <= 13% for forward scattering clouds (largest at intermediate w0 and g >= 0.75).
    """
    henyey_greenstein = asy not in ('Ray', '0')
    tolerance = 0.15 if henyey_greenstein else 0.03
    for column, expected in DLUGACH.loc[asy].items():
        w0 = W0_CONSERVATIVE if float(column) == 1 else float(column)
        if henyey_greenstein:
            albedo, _, _ = reflected(DEEP_LAYERS, w0, cosb=float(asy), single_phase=OTHG, multi_phase=N1,
                                     delta_eddington=True)
        else:
            albedo, _, _ = reflected(DEEP_LAYERS, w0, rayleigh=(asy == 'Ray'))
        assert albedo[0] == pytest.approx(expected, rel=tolerance), f'g={asy}, w0={column}'
