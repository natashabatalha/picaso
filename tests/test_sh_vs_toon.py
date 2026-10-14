"""
Spherical harmonics (Rooney et al. 2023; get_reflected_SH, get_thermal_SH, 2 and 4 streams) against
Toon et al. (1989) (get_reflected_1d, get_thermal_1d), in the limits where the methods should agree,
plus the checks that show where they differ and why.

All inputs are synthetic (no opacity data). Scattering is isotropic or Rayleigh (g = 0), so neither
method applies a delta-Eddington / delta-M correction and the comparison is like for like.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import REFDATA
from picaso.disco import compress_disco, compress_thermal, compute_disco, get_angles_1d
from picaso.fluxes import blackbody, get_reflected_1d, get_reflected_SH, get_thermal_1d, get_thermal_SH

TTHG = (1., -1., 2., -0.5, 1.)  # default two-term HG parameters from reference/config.json
# SH defaults from reference/config.json: TTHG forms (0) for w_single, w_multi, psingle, Rayleigh on (1)
SH_FORMS = (0, 0, 0, 1, 1, 1)
METHODS = ['toon', 2, 4]  # Toon two-stream, SH 2-stream, SH 4-stream
W0_CONSERVATIVE, W0_ABSORBING = 0.999999, 1e-6


def geometry():
    gangle, gweight, tangle, tweight = get_angles_1d(8)
    ubar0, ubar1, cos_theta, _, _ = compute_disco(len(gangle), len(tangle), gangle, tangle, 0.0)
    return ubar0, ubar1, cos_theta, gweight, tweight


def per_layer(dtau, nlayer, nwno):
    layer_dtau = np.atleast_1d(dtau) if np.ndim(dtau) else np.full(nlayer, dtau / nlayer)
    return np.repeat(np.asarray(layer_dtau, dtype=float)[:, np.newaxis], nwno, axis=1)


# ----------------------------------------------------------------------------------------------
# Reflected light
# ----------------------------------------------------------------------------------------------

def reflected(method, dtau, w0, rayleigh=False, surf_reflect=0.0):
    """Returns (geometric albedo, I/F per outgoing angle) at full phase for one wavenumber."""
    nwno = 1
    wno, F0PI = np.array([1e4]), np.ones(nwno)
    ubar0, ubar1, cos_theta, gweight, tweight = geometry()
    dtau = per_layer(dtau, 20, nwno)
    nlevel = dtau.shape[0] + 1
    tau = np.vstack([np.zeros(nwno), np.cumsum(dtau, axis=0)])
    w0, cosb = np.full_like(dtau, w0), np.zeros_like(dtau)
    if rayleigh:
        ftau_cld, ftau_ray, gcos2 = np.zeros_like(w0), np.ones_like(w0), np.full_like(w0, 0.5)
    else:
        ftau_cld, ftau_ray, gcos2 = np.ones_like(w0), np.zeros_like(w0), np.zeros_like(w0)
    surf_reflect = np.full(nwno, surf_reflect)
    if method == 'toon':
        xint, _ = get_reflected_1d(nlevel, wno, nwno, *ubar0.shape, dtau, tau, w0, cosb, gcos2, ftau_cld, ftau_ray,
                                   dtau, tau, w0, cosb, surf_reflect, ubar0, ubar1, cos_theta, F0PI, 3, 0, *TTHG,
                                   get_toa_intensity=1, get_lvl_flux=0, toon_coefficients=0, b_top=0.)
    else:
        xint, _ = get_reflected_SH(nlevel, nwno, *ubar0.shape, dtau, tau, w0, cosb, ftau_cld, ftau_ray,
                                   np.zeros_like(w0), dtau, tau, w0, cosb, surf_reflect, ubar0, ubar1, cos_theta,
                                   F0PI, *SH_FORMS, *TTHG, method, b_top=0, flx=0, single_form=0)
    albedo = compress_disco(nwno, cos_theta, xint, gweight, tweight, F0PI)[0]
    return albedo, np.pi * xint[:, 0, 0] / F0PI[0]


@pytest.mark.parametrize('method', METHODS)
def test_black_planet(method):
    assert reflected(method, 100.0, W0_ABSORBING)[0] < 1e-6


def test_bare_lambert_surface_is_identical_in_all_methods():
    albedos = [reflected(method, 1e-8, W0_ABSORBING, surf_reflect=1.0)[0] for method in METHODS]
    np.testing.assert_allclose(albedos, 2 / 3, rtol=2e-3)
    np.testing.assert_allclose(albedos, albedos[0], rtol=1e-6)


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('rayleigh, phase_at_180', [(True, 1.5), (False, 1.0)])
def test_thin_layer_exact_single_scattering(method, rayleigh, phase_at_180):
    """All three treat single scattering exactly: I/F = w0 P(180) / 8 * (1 - exp(-2 tau / mu)) at full phase."""
    tau = 1e-4
    _, i_over_f = reflected(method, tau, W0_CONSERVATIVE, rayleigh=rayleigh)
    mu = geometry()[1][:, 0]
    np.testing.assert_allclose(i_over_f, W0_CONSERVATIVE * phase_at_180 / 8 * (1 - np.exp(-2 * tau / mu)), rtol=1e-3)


@pytest.mark.parametrize('total_tau', [1e-3, 1e-2])
def test_methods_agree_while_single_scattering_dominates(total_tau):
    """They only diverge once multiple scattering matters (by tau ~ 0.1 they differ by a few percent)."""
    toon = reflected('toon', total_tau, W0_CONSERVATIVE, rayleigh=True)[0]
    for stream in [2, 4]:
        assert reflected(stream, total_tau, W0_CONSERVATIVE, rayleigh=True)[0] == pytest.approx(toon, rel=5e-3)


DLUGACH = pd.read_csv(REFDATA / 'base_cases' / 'testing' / 'DLUGACH_TEST.csv').set_index('asy')
DEEP_LAYERS = 10**np.linspace(-5, 3, 59)


@pytest.mark.parametrize('rayleigh, row', [(True, 'Ray'), (False, '0')])
def test_four_streams_beat_two_streams_in_a_deep_atmosphere(rayleigh, row):
    """
    Against the exact Dlugach & Yanovitskij (1974) geometric albedo of a semi-infinite conservative
    atmosphere, SH4 is several times closer than SH2 and within 3%.
    """
    exact = DLUGACH.loc[row, '1.000']
    error = {stream: abs(reflected(stream, DEEP_LAYERS, W0_CONSERVATIVE, rayleigh=rayleigh)[0] / exact - 1)
             for stream in [2, 4]}
    assert error[4] < 0.5 * error[2]
    assert error[4] < 0.03


# ----------------------------------------------------------------------------------------------
# Thermal emission
# ----------------------------------------------------------------------------------------------

THERMAL_WNO = np.array([500., 1000., 2000., 4000.])
NLAYER = 30


def thermal(method, temperature, dtau, w0):
    """Returns (upward flux per outgoing angle in units of 2 pi I, disk-integrated flux)."""
    nwno = len(THERMAL_WNO)
    _, ubar1, _, gweight, tweight = geometry()
    nlevel = len(temperature)
    pressure = np.logspace(1, 7, nlevel)
    dtau = per_layer(dtau, nlevel - 1, nwno) if np.ndim(dtau) else np.full((nlevel - 1, nwno), dtau)
    tau = np.vstack([np.zeros(nwno), np.cumsum(dtau, axis=0)])
    w0, cosb, surf = np.full_like(dtau, w0), np.zeros_like(dtau), np.zeros(nwno)
    if method == 'toon':
        flux, _ = get_thermal_1d(nlevel, THERMAL_WNO, nwno, *ubar1.shape, temperature, dtau, w0, cosb, pressure,
                                 ubar1, surf, 0, THERMAL_WNO * 0, 0)
    else:
        flux, _ = get_thermal_SH(nlevel, THERMAL_WNO, nwno, *ubar1.shape, temperature, dtau, tau, w0, cosb,
                                 dtau, tau, w0, w0, cosb, pressure, ubar1, surf, method, 0)
    return flux[:, 0, :], compress_thermal(nwno, flux, gweight, tweight)


def formal_solution(temperature, dtau, mu, skip_top=0.0):
    """
    Exact upward 2 pi I for a non-scattering atmosphere with B linear in tau within each layer and the
    same lower boundary as picaso (B + mu dB/dtau). skip_top > 0 evaluates it that far below the top.
    """
    planck = blackbody(temperature, 1 / THERMAL_WNO)
    dtau = np.full((len(temperature) - 1, len(THERMAL_WNO)), dtau)
    b0, b1 = planck[:-1], (planck[1:] - planck[:-1]) / dtau
    dtau = dtau.copy()
    b0[0] = b0[0] + b1[0] * skip_top
    dtau[0] = dtau[0] - skip_top
    tau_above = np.vstack([np.zeros(len(THERMAL_WNO)), np.cumsum(dtau, axis=0)])[:-1]
    intensity = np.zeros(len(THERMAL_WNO))
    for i in range(len(dtau)):
        attenuation = np.exp(-dtau[i] / mu)
        layer = b0[i] * (1 - attenuation) + b1[i] * (mu - (dtau[i] + mu) * attenuation)
        intensity += np.exp(-tau_above[i] / mu) * layer
    intensity += (planck[-1] + b1[-1] * mu) * np.exp(-dtau.sum(axis=0) / mu)
    return 2 * np.pi * intensity


@pytest.mark.parametrize('method', METHODS)
def test_isothermal_absorbing_atmosphere_is_a_blackbody(method):
    flux, disk = thermal(method, np.full(NLAYER + 1, 1000.), 1.0, W0_ABSORBING)
    planck = blackbody(np.array([1000.]), 1 / THERMAL_WNO)[0]
    np.testing.assert_allclose(flux, np.broadcast_to(2 * np.pi * planck, flux.shape), rtol=1e-6)
    np.testing.assert_allclose(disk, np.pi * planck, rtol=1e-6)


@pytest.mark.parametrize('stream', [2, 4])
@pytest.mark.parametrize('dtau', [0.3, 0.05])
def test_sh_matches_exact_non_scattering_emission(stream, dtau):
    """With no scattering there is nothing to approximate: SH gives the exact formal solution at the top."""
    temperature = np.linspace(600, 1500, NLAYER + 1)
    flux, _ = thermal(stream, temperature, dtau, W0_ABSORBING)
    mu = geometry()[1][:, 0]
    np.testing.assert_allclose(flux, [formal_solution(temperature, dtau, m) for m in mu], rtol=1e-4)


@pytest.mark.parametrize('dtau', [0.3, 0.05])
def test_toon_reports_emission_at_the_top_layer_midpoint(dtau):
    """
    get_thermal_1d returns flux_plus_mdpt at the top layer, so its non-scattering emission equals the
    exact solution half a layer below the top. This is why Toon emits more than SH when the top layer
    is not optically thin (by 3-24% here, largest toward the limb).
    """
    temperature = np.linspace(600, 1500, NLAYER + 1)
    flux, _ = thermal('toon', temperature, dtau, W0_ABSORBING)
    mu = geometry()[1][:, 0]
    # rtol allows for the token w0 = 1e-6 scattering that the exact formula leaves out
    np.testing.assert_allclose(flux, [formal_solution(temperature, dtau, m, skip_top=dtau / 2) for m in mu],
                               rtol=1e-5)


def test_toon_and_sh_emission_agree_when_the_top_layer_is_thin():
    """In real atmospheres the top layer is optically thin, so the midpoint offset is negligible."""
    temperature = np.linspace(600, 1500, NLAYER + 1)
    thin_top = np.logspace(-5, 1, NLAYER)
    toon_flux, toon_disk = thermal('toon', temperature, thin_top, W0_ABSORBING)
    for stream in [2, 4]:
        flux, disk = thermal(stream, temperature, thin_top, W0_ABSORBING)
        np.testing.assert_allclose(flux, toon_flux, rtol=1e-3)
        np.testing.assert_allclose(disk, toon_disk, rtol=1e-3)


@pytest.mark.parametrize('w0', [0.5, 0.9])
def test_scattering_emission_agrees_between_methods(w0):
    """Isothermal scattering half-space (thin top): SH2 matches Toon, SH4 is within ~1% of both."""
    temperature = np.full(NLAYER + 1, 1000.)
    thin_top = np.logspace(-5, 2, NLAYER)
    _, toon = thermal('toon', temperature, thin_top, w0)
    _, sh2 = thermal(2, temperature, thin_top, w0)
    _, sh4 = thermal(4, temperature, thin_top, w0)
    np.testing.assert_allclose(sh2, toon, rtol=3e-3)
    np.testing.assert_allclose(sh4, toon, rtol=2e-2)
