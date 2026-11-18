"""Tests for the phase-integral Bond albedo (issue #358).

Unit tests use analytic phase functions and need no reference data.

The integration tests run PICASO and are skipped unless PICASO_TEST_OPACITY_DB points to an
opacity database covering 0.3-1 um. Because picaso reads `picaso_refdata` and `PYSYN_CDBS` at
import time, also set PICASO_TEST_REFDATA (full reference directory) and, if needed,
PICASO_TEST_PYSYN_CDBS, and run this file on its own:

    PICASO_TEST_OPACITY_DB=/path/to/opacities.db PICASO_TEST_REFDATA=/path/to/reference \
        pytest tests/test_bond_albedo.py
"""
import os
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

if os.environ.get('PICASO_TEST_REFDATA'):
    os.environ['picaso_refdata'] = os.environ['PICASO_TEST_REFDATA']
if os.environ.get('PICASO_TEST_PYSYN_CDBS'):
    os.environ['PYSYN_CDBS'] = os.environ['PICASO_TEST_PYSYN_CDBS']

from picaso import justdoit as jdi


def lambert_phase_function(alpha):
    """Lambert sphere phase function, normalized to 1 at zero phase (Cahoy et al. 2010 Eq. 1)."""
    return (np.sin(alpha) + (np.pi - alpha) * np.cos(alpha)) / np.pi


def lambert_albedo(alpha, geometric_albedo):
    """Albedo spectra of a Lambert sphere at phases alpha, shape (nphase, nwno)."""
    return lambert_phase_function(np.asarray(alpha))[:, None] * np.asarray(geometric_albedo)[None, :]


# ------------------------------------------------------------------ unit tests

def test_quadrature_weights():
    phases, weights = jdi._phase_quadrature(12)
    assert np.all(np.diff(phases) > 0)
    assert phases.min() > 0 and phases.max() < np.pi
    # int_0^pi 2 sin(a) da = 4
    assert np.isclose(weights.sum(), 4.0)


@pytest.mark.parametrize('n_phase', [8, 12, 24])
def test_lambert_gauss(n_phase):
    geometric_albedo = np.array([0.1, 0.4, 2/3])
    phases, weights = jdi._phase_quadrature(n_phase)
    spherical_albedo, phase_integral_q = jdi._phase_integral(phases, lambert_albedo(phases, geometric_albedo),
                                geometric_albedo=geometric_albedo, weights=weights)
    # Lambert sphere: q = 3/2 (Cahoy et al. 2010 Sec. 4.4)
    assert np.allclose(phase_integral_q, 1.5, atol=1e-3)
    assert np.allclose(spherical_albedo, 1.5 * geometric_albedo, rtol=1e-3)


def test_lambert_trapezoid_paper_grid():
    """Phase grid used in the GJ 1214 b paper: 0-168 deg every 12 deg, plus 179 deg."""
    phases = np.radians(list(range(0, 180, 12)) + [179])
    geometric_albedo = np.array([0.3, 2/3])
    spherical_albedo, phase_integral_q = jdi._phase_integral(phases, lambert_albedo(phases, geometric_albedo))
    assert np.allclose(phase_integral_q, 1.5, rtol=1e-2)


def test_trapezoid_matches_ratio_form():
    """Integrating A(alpha) sin(alpha) then dividing by A_g equals integrating A/A_g."""
    random_generator = np.random.default_rng(0)
    phases = np.radians(np.linspace(0, 179, 30))
    albedo = random_generator.uniform(0.05, 0.6, size=(phases.size, 5))
    spherical_albedo, phase_integral_q = jdi._phase_integral(phases, albedo)
    ratio = 2 * np.trapezoid(albedo / albedo[0] * np.sin(phases)[:, None], phases, axis=0)
    assert np.allclose(phase_integral_q, ratio, rtol=1e-12)


def test_zero_geometric_albedo():
    phases, weights = jdi._phase_quadrature(12)
    geometric_albedo = np.array([0.0, 0.5])
    albedo = lambert_albedo(phases, geometric_albedo)
    albedo[:, 0] = 1e-3  # e.g. nonzero scattering away from zero phase in a line core
    spherical_albedo, phase_integral_q = jdi._phase_integral(phases, albedo, geometric_albedo=geometric_albedo, weights=weights)
    assert np.isnan(phase_integral_q[0]) and np.isfinite(phase_integral_q[1])
    assert np.all(np.isfinite(spherical_albedo))


def test_missing_zero_phase():
    phases, weights = jdi._phase_quadrature(8)
    with pytest.raises(Exception):
        jdi._phase_integral(phases, np.ones((8, 3)), weights=weights)


def test_bond_from_spherical():
    wno = np.linspace(1e4, 3e4, 200)
    # constant A_s returns A_s for any stellar spectrum
    assert np.isclose(jdi._bond_albedo_from_spherical(wno, np.full(200, 0.42), np.exp(-wno / 1e4)), 0.42)
    # NaNs in the spherical albedo do not propagate
    spherical_albedo = np.full(200, 0.42)
    spherical_albedo[10] = np.nan
    assert np.isfinite(jdi._bond_albedo_from_spherical(wno, spherical_albedo, np.ones(200)))


# ------------------------------------------------------------------ integration tests

OPA_DB = os.environ.get('PICASO_TEST_OPACITY_DB')
needs_refdata = pytest.mark.skipif(not (OPA_DB and os.path.isfile(OPA_DB)),
                                   reason='set PICASO_TEST_OPACITY_DB to run PICASO integration tests')


@pytest.fixture(scope='module')
def rayleigh_case():
    """Cloud-free Jupiter case from the issue #358 snippet, Raman off."""
    import astropy.units as u
    opacity = jdi.opannection(wave_range=[0.3, 1], filename_db=OPA_DB)
    case = jdi.inputs()
    case.phase_angle(0)
    case.gravity(gravity=25, gravity_unit=u.Unit('m/(s**2)'))
    case.star(opacity, 6000, 0.0122, 4.437)
    case.atmosphere(filename=jdi.jupiter_pt(), sep=r'\s+')
    case.approx(raman='none')
    return opacity, case


@needs_refdata
def test_rayleigh_phase_integral(rayleigh_case):
    opacity, case = rayleigh_case
    phase_before = case.inputs['phase_angle']
    disco_before = case.inputs['disco']
    bond_out = case.bond_albedo(opacity, n_phase=12)

    # the user's phase settings are untouched
    assert case.inputs['phase_angle'] == phase_before
    assert case.inputs['disco'] is disco_before

    wavelength_um = 1e4 / bond_out['wavenumber']
    shortest_wavelength_index = np.argmin(wavelength_um)  # ~0.3 um, Rayleigh dominated
    # Rayleigh phase function gives q = 4/3 (Cahoy et al. 2010 Sec. 4.4); multiple
    # scattering and residual absorption move the model value slightly off it
    assert 1.2 < bond_out['phase_integral'][shortest_wavelength_index] < 1.4
    # energy conservation: spherical albedo is the fraction reflected in all directions
    assert np.nanmax(bond_out['spherical_albedo']) <= 1.0 + 1e-2
    assert 0 < bond_out['bond_albedo'] < 1
    # phase_integral_q > 1, so the true Bond albedo exceeds the phase_integral_q=1 value from spectrum()
    assert bond_out['bond_albedo'] > case.spectrum(opacity)['bond_albedo']
