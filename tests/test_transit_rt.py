"""
Analytic-limit tests for the 1D transmission solver, fluxes.get_transit_1d (Brown 2001), which is
what spectrum(calculation='transmission') uses.

The atmospheres are isothermal with constant gravity, built on a log-uniform pressure grid, so the
altitude of each level is exact: z = Rp + H ln(P_bottom / P) with scale height H = k T / (mu m_u g).
get_transit_1d divides each layer's optical depth by column density over mean molecular weight to
get a cross section per particle; passing colden = mmw * amu makes DTAU that cross section directly.
No opacity data is needed, so these run in CI in well under a second.
"""
import numpy as np
import pytest
from scipy.optimize import brentq

from picaso.fluxes import get_transit_1d

K_B, AMU = 1.38064852e-16, 1.66053904e-24  # cgs, as in picaso's atmsetup constants
TEMPERATURE, MU, GRAVITY = 1000., 2.3, 2500.  # K, amu, cm/s2
SCALE_HEIGHT = K_B * TEMPERATURE / (MU * AMU * GRAVITY)  # ~145 km
R_PLANET, R_STAR = 7e9, 7e10  # cm: Jupiter around a Sun-like star


def atmosphere(nlevel=200, p_top_bar=1e-8, p_bottom_bar=1e2):
    """Levels from the top down: pressure (dyne/cm2), altitude z (cm) and spacing dz (cm)."""
    pressure = np.logspace(np.log10(p_top_bar), np.log10(p_bottom_bar), nlevel) * 1e6
    z = R_PLANET + SCALE_HEIGHT * np.log(pressure[-1] / pressure)
    dz = np.full(nlevel, SCALE_HEIGHT * np.log(pressure[1] / pressure[0]))
    return pressure, z, dz


def transit_depth(cross_section, pressure, z, dz):
    """
    (Rp/Rs)^2 per wavenumber. cross_section is per particle (cm2): a number, a per-layer array
    (nlayer,), or a full (nlayer, nwno) array.
    """
    nlevel = len(pressure)
    nlayer = nlevel - 1
    cross_section = np.asarray(cross_section, dtype=float)
    if cross_section.ndim == 0:
        cross_section = np.full((nlayer, 1), cross_section)
    elif cross_section.ndim == 1:
        cross_section = cross_section[:, np.newaxis]
    nwno = cross_section.shape[1]
    mmw = np.full(nlayer, MU)
    return get_transit_1d(z, dz, nlevel, nwno, R_STAR, mmw, K_B, AMU, pressure,
                          np.full(nlevel, TEMPERATURE), mmw * AMU, cross_section.copy())


def effective_radius(cross_section, pressure, z, dz):
    return R_STAR * np.sqrt(transit_depth(cross_section, pressure, z, dz))


def test_transparent_atmosphere_shows_only_the_planet():
    pressure, z, dz = atmosphere()
    depth = transit_depth(0.0, pressure, z, dz)
    np.testing.assert_allclose(depth, (R_PLANET / R_STAR)**2, rtol=1e-12)


def test_opaque_atmosphere_looks_as_big_as_its_top():
    """If every layer is opaque, starlight is blocked all the way up to the top level."""
    pressure, z, dz = atmosphere()
    radius = effective_radius(1e10, pressure, z, dz)
    assert abs(radius[0] - z[0]) < 0.1 * dz[0]


@pytest.mark.parametrize('deck_level', [60, 120, 180])
def test_opaque_cloud_deck_sets_the_radius(deck_level):
    """A gray, opaque deck under a clear atmosphere: the planet looks as big as the deck top."""
    pressure, z, dz = atmosphere()
    cross_section = np.zeros(len(pressure) - 1)
    cross_section[deck_level:] = 1e10
    radius = effective_radius(cross_section, pressure, z, dz)
    assert abs(radius[0] - z[deck_level]) < 0.1 * dz[0]


@pytest.mark.parametrize('cross_section', [1e-26, 1e-25, 1e-24])
def test_e_times_more_opacity_raises_the_radius_one_scale_height(cross_section):
    """
    Density falls by e every scale height, so e times more opacity per particle moves the level
    where the atmosphere turns opaque up by exactly H (isothermal, constant gravity).
    """
    pressure, z, dz = atmosphere()
    shift = effective_radius(np.e * cross_section, pressure, z, dz) - effective_radius(cross_section, pressure, z, dz)
    assert shift[0] / SCALE_HEIGHT == pytest.approx(1, abs=0.01)


@pytest.mark.parametrize('cross_section', [1e-26, 1e-25, 1e-24])
def test_radius_is_where_the_grazing_chord_optical_depth_is_0_56(cross_section):
    """
    Lecavelier des Etangs et al. (2008): the transit radius is where the optical depth along the
    grazing chord, tau = sigma n(R) sqrt(2 pi R H), reaches ~0.56.
    """
    pressure, z, dz = atmosphere()
    n_bottom = pressure[-1] / (K_B * TEMPERATURE)

    def chord_tau_minus_056(radius):
        n = n_bottom * np.exp(-(radius - R_PLANET) / SCALE_HEIGHT)
        return cross_section * n * np.sqrt(2 * np.pi * radius * SCALE_HEIGHT) - 0.56

    expected = brentq(chord_tau_minus_056, R_PLANET - 50 * SCALE_HEIGHT, R_PLANET + 200 * SCALE_HEIGHT)
    radius = effective_radius(cross_section, pressure, z, dz)
    assert abs(radius[0] - expected) < 0.05 * SCALE_HEIGHT


def test_rayleigh_slope_is_minus_four_scale_heights():
    """With sigma proportional to wavelength^-4, dR/d ln(wavelength) = -4 H (Lecavelier des Etangs et al. 2008)."""
    pressure, z, dz = atmosphere()
    wavelength = np.logspace(np.log10(0.3), np.log10(1.0), 20)  # micron
    cross_section = 1e-26 * (wavelength / 0.5)**-4
    radius = effective_radius(np.broadcast_to(cross_section, (len(pressure) - 1, len(wavelength))), pressure, z, dz)
    slope = np.polyfit(np.log(wavelength), radius, 1)[0]
    assert slope / SCALE_HEIGHT == pytest.approx(-4, abs=0.05)
