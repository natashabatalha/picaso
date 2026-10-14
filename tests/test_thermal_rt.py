"""
Analytic-limit tests for the 1D thermal emission solver, fluxes.get_thermal_1d (Toon et al. 1989
source function method). It is used by spectrum(calculation='thermal') and by every climate
iteration.

The inputs are small synthetic atmospheres (no opacity data), so these run in CI in well under a
second. get_thermal_1d returns the upward flux at the top for each outgoing angle in units of
2*pi*intensity, so a blackbody at temperature T reads 2*pi*B(T).
"""
import numpy as np
import pytest

from picaso.fluxes import blackbody, blackbody_integrated, get_thermal_1d

WNO = np.array([500., 1000., 2000., 4000.])  # cm-1
DWNO = np.full(len(WNO), 50.)  # bin widths, used only by the climate (calc_type=1) Planck function
UBAR1 = np.array([[0.1], [0.5], [0.95]])  # outgoing cos(angle): near limb, middle, near disk center
W0_ABSORBING = 1e-6  # picaso never passes exactly 0 (Rayleigh is always present) and 0 divides by zero


def thermal(temperature, dtau, w0=W0_ABSORBING, cosb=0.0, hard_surface=0, surf_reflect=0.0, calc_type=0):
    """Runs get_thermal_1d on a uniform-in-wavenumber atmosphere; dtau is a scalar or a per-layer array."""
    temperature = np.asarray(temperature, dtype=float)
    nlevel = len(temperature)
    nlayer, nwno = nlevel - 1, len(WNO)
    dtau = np.broadcast_to(np.reshape(dtau, (-1, 1)) if np.ndim(dtau) else dtau, (nlayer, nwno)).copy()
    numg, numt = UBAR1.shape
    return get_thermal_1d(nlevel, WNO, nwno, numg, numt, temperature, dtau,
                          np.full((nlayer, nwno), w0), np.full((nlayer, nwno), cosb),
                          np.logspace(1, 6, nlevel),  # level pressures in dyne/cm2
                          UBAR1, np.full(nwno, surf_reflect), hard_surface, DWNO, calc_type)


def two_pi_planck(temperature):
    return 2 * np.pi * blackbody(np.array([float(temperature)]), 1 / WNO)[0]


# per-layer optical depths that start very thin, so the isothermal continuation that
# get_thermal_1d assumes above the top (tau_top scales with the top layer's dtau) is negligible
THIN_TOP_DTAU = np.logspace(-5, 2, 10)


@pytest.mark.parametrize('dtau', [1e-4, 0.1, 1, 10, 100])
def test_isothermal_absorbing_atmosphere_emits_a_blackbody(dtau):
    """With no scattering the source function is B(T) everywhere, so I = B(T) at any depth and angle."""
    flux_at_top, _ = thermal(np.full(11, 1000.), dtau)
    np.testing.assert_allclose(flux_at_top, np.broadcast_to(two_pi_planck(1000.), flux_at_top.shape), rtol=1e-6)


def test_isothermal_blackbody_with_climate_planck_function():
    """calc_type=1 (climate) uses the bin-integrated Planck function instead of the monochromatic one."""
    flux_at_top, _ = thermal(np.full(11, 1000.), 1.0, calc_type=1)
    expected = 2 * np.pi * blackbody_integrated(np.array([1000.]), WNO, DWNO)[0]
    np.testing.assert_allclose(flux_at_top, np.broadcast_to(expected, flux_at_top.shape), rtol=1e-6)


def test_no_net_flux_deep_in_an_isothermal_atmosphere():
    """Radiative equilibrium: deep in a thick isothermal atmosphere upward and downward fluxes cancel."""
    _, (flux_minus, flux_plus, _, _) = thermal(np.full(11, 1000.), 20.0)
    net = flux_plus[:, :, -3, :] - flux_minus[:, :, -3, :]
    np.testing.assert_allclose(net / two_pi_planck(1000.), 0, atol=1e-10)


@pytest.mark.parametrize('surf_reflect', [0.0, 0.3])
def test_transparent_atmosphere_shows_the_surface(surf_reflect):
    """An optically thin atmosphere over a hard surface emits (1 - reflectivity) * B(T_surface)."""
    temperature = np.linspace(600, 1200, 11)
    flux_at_top, _ = thermal(temperature, 1e-7, hard_surface=1, surf_reflect=surf_reflect)
    expected = (1 - surf_reflect) * two_pi_planck(temperature[-1])
    np.testing.assert_allclose(flux_at_top, np.broadcast_to(expected, flux_at_top.shape), rtol=1e-4)


def test_temperature_gradient_gives_bounded_limb_darkened_emission():
    """
    With temperature increasing with depth, emission lies between B(T_top) and B(T_bottom), is
    brighter looking straight down (we see deeper, hotter layers) and increases with temperature.
    """
    temperature = np.linspace(600, 1200, 11)
    flux_at_top, _ = thermal(temperature, 2.0)
    assert np.all(flux_at_top > two_pi_planck(temperature[0]))
    assert np.all(flux_at_top < two_pi_planck(temperature[-1]))

    limb, middle, center = flux_at_top[:, 0, :]
    assert np.all(limb < middle) and np.all(middle < center), 'no limb darkening'

    hotter, _ = thermal(temperature + 100, 2.0)
    assert np.all(hotter > flux_at_top)


def test_scattering_lowers_emission_within_the_analytic_bound():
    """
    An isothermal scattering half-space emits less than B, and less the more it scatters, but at
    least sqrt(1 - w0) * B (Chandrasekhar: I(0, mu) = sqrt(1 - w0) H(mu) B with H >= 1).
    """
    temperature = np.full(11, 1000.)
    two_pi_b = two_pi_planck(1000.)
    absorbing, _ = thermal(temperature, THIN_TOP_DTAU)
    np.testing.assert_allclose(absorbing, np.broadcast_to(two_pi_b, absorbing.shape), rtol=1e-6)

    previous = absorbing
    for w0 in [0.5, 0.9, 0.99]:
        scattering, _ = thermal(temperature, THIN_TOP_DTAU, w0=w0)
        assert np.all(scattering < previous), f'w0={w0} did not lower emission'
        assert np.all(scattering > np.sqrt(1 - w0) * two_pi_b), f'w0={w0} below sqrt(1 - w0) B'
        previous = scattering
