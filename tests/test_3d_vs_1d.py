"""
The 3D radiative transfer routines must reproduce the 1D ones when every column of the 3D
atmosphere is the same. Covers reflected light (get_reflected_3d), thermal emission
(get_thermal_3d) and transmission (get_transit_3d), each against its 1D counterpart.

For reflected and thermal light the 1D solvers accept the same (ng, nt) angle grid, so the two can
be compared angle by angle (they should agree to round-off), and after disk integration against the
symmetric 1D geometry that spectrum() uses at full phase (they agree to quadrature accuracy). The 3D
geometry also allows non-zero phase, checked against the analytic Lambert sphere phase curve.

Transmission uses two different algorithms (Brown 2001 chords in 1D, the TRIDENT path distribution
of MacDonald & Lewis 2022 in 3D), so they agree to within the vertical discretization, and converge
together as layers get thinner.

All inputs are synthetic (no opacity data), so these run in CI.
"""
import numpy as np
import pytest

from picaso.disco import compress_disco, compress_thermal, compute_disco, get_angles_1d, get_angles_3d
from picaso.fluxes import (blackbody, get_reflected_1d, get_reflected_3d, get_thermal_1d, get_thermal_3d,
                           get_transit_1d, get_transit_3d)

TTHG = (1., -1., 2., -0.5, 1.)  # default two-term HG parameters from reference/config.json
TTHG_RAY, N2 = 3, 0             # default single and multiple scattering phase functions


def identical_columns(array, ng, nt):
    """Repeats a 1D-atmosphere array over the (ng, nt) columns of a 3D atmosphere (last two axes)."""
    return np.repeat(np.repeat(array[..., np.newaxis, np.newaxis], ng, -2), nt, -1).copy()


def geometry_3d(ng, nt, phase=0.0):
    gangle, gweight, tangle, tweight = get_angles_3d(ng, nt)
    ubar0, ubar1, cos_theta, _, _ = compute_disco(ng, nt, gangle, tangle, phase)
    return ubar0, ubar1, cos_theta, gweight, tweight


def geometry_1d_symmetric(ngauss=8):
    """The full-phase geometry spectrum() uses for 1D: one quarter of the disk, by symmetry."""
    gangle, gweight, tangle, tweight = get_angles_1d(ngauss)
    ubar0, ubar1, cos_theta, _, _ = compute_disco(len(gangle), len(tangle), gangle, tangle, 0.0)
    return ubar0, ubar1, cos_theta, gweight, tweight


# ----------------------------------------------------------------------------------------------
# Reflected light
# ----------------------------------------------------------------------------------------------

@pytest.fixture(scope='module')
def cloudy_atmosphere():
    """A non-trivial atmosphere: mixed cloud and Rayleigh scattering over a reflective surface."""
    rng = np.random.default_rng(1)
    nlayer, nwno = 20, 3
    dtau = rng.uniform(0.01, 2, (nlayer, nwno))
    tau = np.zeros((nlayer + 1, nwno))
    tau[1:] = np.cumsum(dtau, axis=0)
    ftau_cld = rng.uniform(0, 1, (nlayer, nwno))
    return dict(nlevel=nlayer + 1, wno=np.array([1e4, 1.5e4, 2e4]), dtau=dtau, tau=tau,
                w0=rng.uniform(0.3, 0.99, (nlayer, nwno)), cosb=rng.uniform(0, 0.6, (nlayer, nwno)),
                gcos2=rng.uniform(0, 0.5, (nlayer, nwno)), ftau_cld=ftau_cld, ftau_ray=1 - ftau_cld,
                surf_reflect=np.full(nwno, 0.2), F0PI=np.ones(nwno))


def reflected_1d(atm, ubar0, ubar1, cos_theta):
    xint, _ = get_reflected_1d(
        atm['nlevel'], atm['wno'], len(atm['wno']), *ubar0.shape, atm['dtau'], atm['tau'], atm['w0'], atm['cosb'],
        atm['gcos2'], atm['ftau_cld'], atm['ftau_ray'], atm['dtau'], atm['tau'], atm['w0'], atm['cosb'],
        atm['surf_reflect'], ubar0, ubar1, cos_theta, atm['F0PI'], TTHG_RAY, N2, *TTHG,
        get_toa_intensity=1, get_lvl_flux=0, toon_coefficients=0, b_top=0.)
    return xint


def reflected_3d(atm, ubar0, ubar1, cos_theta):
    ng, nt = ubar0.shape
    cols = {key: identical_columns(atm[key], ng, nt)
            for key in ['dtau', 'tau', 'w0', 'cosb', 'gcos2', 'ftau_cld', 'ftau_ray']}
    return get_reflected_3d(
        atm['nlevel'], atm['wno'], len(atm['wno']), ng, nt, cols['dtau'], cols['tau'], cols['w0'], cols['cosb'],
        cols['gcos2'], cols['ftau_cld'], cols['ftau_ray'], cols['dtau'], cols['tau'], cols['w0'], cols['cosb'],
        atm['surf_reflect'], ubar0, ubar1, cos_theta, atm['F0PI'], TTHG_RAY, N2, *TTHG)


@pytest.mark.parametrize('phase', [0.0, 1.0, 2.5])
def test_reflected_3d_matches_1d_at_every_angle(cloudy_atmosphere, phase):
    ubar0, ubar1, cos_theta, _, _ = geometry_3d(10, 10, phase)
    expected = reflected_1d(cloudy_atmosphere, ubar0, ubar1, cos_theta)
    np.testing.assert_allclose(reflected_3d(cloudy_atmosphere, ubar0, ubar1, cos_theta), expected, rtol=1e-12)


def test_reflected_3d_disk_albedo_matches_1d_symmetric_geometry(cloudy_atmosphere):
    ubar0, ubar1, cos_theta, gweight, tweight = geometry_1d_symmetric()
    albedo_1d = compress_disco(3, cos_theta, reflected_1d(cloudy_atmosphere, ubar0, ubar1, cos_theta),
                               gweight, tweight, cloudy_atmosphere['F0PI'])
    ubar0, ubar1, cos_theta, gweight, tweight = geometry_3d(10, 10)
    albedo_3d = compress_disco(3, cos_theta, reflected_3d(cloudy_atmosphere, ubar0, ubar1, cos_theta),
                               gweight, tweight, cloudy_atmosphere['F0PI'])
    np.testing.assert_allclose(albedo_3d, albedo_1d, rtol=2e-3)


@pytest.mark.parametrize('phase', [0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
def test_reflected_3d_lambert_sphere_phase_curve(cloudy_atmosphere, phase):
    """A bare Lambertian surface (r = 1) follows A_g * Phi(a), Phi = [sin a + (pi - a) cos a] / pi."""
    nlayer, nwno = cloudy_atmosphere['dtau'].shape
    bare = dict(cloudy_atmosphere, dtau=np.full((nlayer, nwno), 1e-8), w0=np.full((nlayer, nwno), 1e-6),
                surf_reflect=np.ones(nwno))
    bare['tau'] = np.vstack([np.zeros(nwno), np.cumsum(bare['dtau'], axis=0)])
    ubar0, ubar1, cos_theta, gweight, tweight = geometry_3d(16, 16, phase)
    albedo = compress_disco(nwno, cos_theta, reflected_3d(bare, ubar0, ubar1, cos_theta), gweight, tweight,
                            bare['F0PI'])
    phase_function = (np.sin(phase) + (np.pi - phase) * np.cos(phase)) / np.pi
    np.testing.assert_allclose(albedo, 2 / 3 * phase_function, rtol=1e-3)


# ----------------------------------------------------------------------------------------------
# Thermal emission
# ----------------------------------------------------------------------------------------------

THERMAL_WNO = np.array([500., 1000., 2000., 4000.])


@pytest.fixture(scope='module')
def emitting_atmosphere():
    """Temperature rising with depth, with some scattering, so emission depends on angle."""
    rng = np.random.default_rng(2)
    nlayer, nwno = 30, len(THERMAL_WNO)
    return dict(nlevel=nlayer + 1, temperature=np.linspace(600, 1500, nlayer + 1),
                pressure=np.logspace(1, 7, nlayer + 1), dtau=rng.uniform(0.01, 3, (nlayer, nwno)),
                w0=rng.uniform(1e-6, 0.8, (nlayer, nwno)), cosb=rng.uniform(0, 0.6, (nlayer, nwno)))


def thermal_1d(atm, ubar1):
    nwno = len(THERMAL_WNO)
    flux, _ = get_thermal_1d(atm['nlevel'], THERMAL_WNO, nwno, *ubar1.shape, atm['temperature'], atm['dtau'],
                             atm['w0'], atm['cosb'], atm['pressure'], ubar1, np.zeros(nwno), 0, THERMAL_WNO * 0, 0)
    return flux


def thermal_3d(atm, ubar1):
    ng, nt = ubar1.shape
    nwno = len(THERMAL_WNO)
    return get_thermal_3d(atm['nlevel'], THERMAL_WNO, nwno, ng, nt,
                          identical_columns(atm['temperature'], ng, nt), identical_columns(atm['dtau'], ng, nt),
                          identical_columns(atm['w0'], ng, nt), identical_columns(atm['cosb'], ng, nt),
                          identical_columns(atm['pressure'], ng, nt), ubar1, np.zeros(nwno), 0)


def test_thermal_3d_matches_1d_at_every_angle(emitting_atmosphere):
    _, ubar1, _, _, _ = geometry_3d(10, 10)
    np.testing.assert_allclose(thermal_3d(emitting_atmosphere, ubar1), thermal_1d(emitting_atmosphere, ubar1),
                               rtol=1e-12)


def test_thermal_3d_disk_flux_matches_1d_symmetric_geometry(emitting_atmosphere):
    _, ubar1, _, gweight, tweight = geometry_1d_symmetric()
    flux_1d = compress_thermal(len(THERMAL_WNO), thermal_1d(emitting_atmosphere, ubar1), gweight, tweight)
    _, ubar1, _, gweight, tweight = geometry_3d(10, 10)
    flux_3d = compress_thermal(len(THERMAL_WNO), thermal_3d(emitting_atmosphere, ubar1), gweight, tweight)
    np.testing.assert_allclose(flux_3d, flux_1d, rtol=2e-3)


def test_isothermal_disk_flux_is_pi_b_in_1d_and_3d(emitting_atmosphere):
    """The 1D and 3D disk integrations normalize differently; both must give pi B(T) for a blackbody."""
    nlayer = emitting_atmosphere['nlevel'] - 1
    isothermal = dict(emitting_atmosphere, temperature=np.full(nlayer + 1, 1000.),
                      w0=np.full((nlayer, len(THERMAL_WNO)), 1e-6))
    pi_b = np.pi * blackbody(np.array([1000.]), 1 / THERMAL_WNO)[0]
    _, ubar1, _, gweight, tweight = geometry_1d_symmetric()
    np.testing.assert_allclose(compress_thermal(len(THERMAL_WNO), thermal_1d(isothermal, ubar1), gweight, tweight),
                               pi_b, rtol=1e-6)
    _, ubar1, _, gweight, tweight = geometry_3d(10, 10)
    np.testing.assert_allclose(compress_thermal(len(THERMAL_WNO), thermal_3d(isothermal, ubar1), gweight, tweight),
                               pi_b, rtol=1e-6)


# ----------------------------------------------------------------------------------------------
# Transmission
# ----------------------------------------------------------------------------------------------

K_B, AMU = 1.38064852e-16, 1.66053904e-24
TEMPERATURE, MU, GRAVITY = 1000., 2.3, 2500.
SCALE_HEIGHT = K_B * TEMPERATURE / (MU * AMU * GRAVITY)
R_PLANET, R_STAR = 7e9, 7e10


def transit_radii(cross_section, nlevel=200, nzenith=10, nsector=8):
    """
    Transit radius from get_transit_1d and get_transit_3d for an isothermal atmosphere with
    identical columns. Both get the same vertical layer optical depth, as spectrum() passes them:
    cross section times the exact particle column of each layer.
    """
    pressure = np.logspace(-8, 2, nlevel) * 1e6
    z = R_PLANET + SCALE_HEIGHT * np.log(pressure[-1] / pressure)
    dz = np.full(nlevel, SCALE_HEIGHT * np.log(pressure[1] / pressure[0]))
    density = pressure / (K_B * TEMPERATURE)
    particle_column = (density[1:] - density[:-1]) * SCALE_HEIGHT  # integral of n dz over each layer
    dtau = cross_section * particle_column[:, np.newaxis]
    nlayer = nlevel - 1
    mmw = np.full(nlayer, MU)
    depth_1d = get_transit_1d(z, dz, nlevel, 1, R_STAR, mmw, K_B, AMU, pressure, np.full(nlevel, TEMPERATURE),
                              particle_column * MU * AMU, dtau.copy())
    depth_3d = get_transit_3d(identical_columns(z, nzenith, nsector), identical_columns(dtau, nzenith, nsector),
                              np.linspace(-np.pi / 2, np.pi / 2, nzenith + 1), np.linspace(-np.pi, np.pi, nsector + 1),
                              R_STAR)
    return R_STAR * np.sqrt(depth_1d[0]), R_STAR * np.sqrt(depth_3d[0])


def test_transit_3d_matches_1d_for_a_transparent_atmosphere():
    radius_1d, radius_3d = transit_radii(0.0)
    assert radius_1d == pytest.approx(R_PLANET, rel=1e-12)
    assert radius_3d == pytest.approx(R_PLANET, rel=1e-12)


@pytest.mark.parametrize('cross_section', [1e-26, 1e-24, 1e10])
def test_transit_3d_matches_1d_for_absorbing_atmospheres(cross_section):
    """At 200 levels (0.12 H per layer) the two algorithms agree to 0.03 H (~0.6 ppm in transit depth)."""
    radius_1d, radius_3d = transit_radii(cross_section)
    assert abs(radius_3d - radius_1d) < 0.03 * SCALE_HEIGHT


def test_transit_1d_and_3d_converge_together():
    """The remaining difference comes from how each discretizes a layer, so it shrinks with layer thickness."""
    differences = [abs(np.subtract(*transit_radii(1e-24, nlevel=n))) / SCALE_HEIGHT for n in [50, 100, 200, 400]]
    assert np.all(np.diff(differences) < 0)
    assert differences[-1] < 0.01


def test_transit_3d_does_not_depend_on_its_angular_grid_for_identical_columns():
    radii = [transit_radii(1e-24, nzenith=nz, nsector=ns)[1] for nz, ns in [(2, 1), (4, 4), (10, 8), (20, 16)]]
    np.testing.assert_allclose(radii, radii[0], rtol=1e-10)


def test_transit_3d_e_times_more_opacity_raises_the_radius_one_scale_height():
    _, low = transit_radii(1e-24)
    _, high = transit_radii(np.e * 1e-24)
    assert (high - low) / SCALE_HEIGHT == pytest.approx(1, abs=0.01)
