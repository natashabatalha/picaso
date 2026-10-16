"""
Tests for the step between the atmosphere and the radiative transfer solvers: looking up opacities
in the database (opannection.get_opacities) and turning them into layer optical depths, single
scattering albedos and asymmetries (optics.compute_opacity). The solver tests feed these arrays in by
hand, so this is what checks that a real atmosphere produces the right ones.

Everything runs on the mini opacity db (H2O, CO2, H2-H2 and H2-He CIA, 500-1600 K), read back
directly with sqlite3 so the expected values do not go through picaso's own lookup. The column
density each layer uses (atm.layer['colden']) is taken as given here; it is tested on its own.
"""
import io
import sqlite3

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
import scipy.constants as sc

from conftest import MINI_OPACITY_DB
from picaso import justdoit as jdi
from picaso.atmsetup import ATMSETUP
from picaso.optics import compute_opacity

AMU, K_B = 1.66053904e-24, 1.38064852e-16  # cgs
AVOGADRO = 6.02214086e23
LOSCHMIDT = 101325 / (sc.k * 273.15) * 1e-6  # cm-3, one amagat
GRAVITY = 1000.  # cm/s2
X_H2, X_H2O = 0.85, 1e-3
X_HE = 1 - X_H2 - X_H2O
NO_RAMAN = 2
FLOAT32 = 1e-6  # the db stores cross sections as float32
# Layers centred on the db pressures 1e-3 ... 10 bar (layer pressure is the geometric mean of its levels)
DECADE_LEVELS = 10**np.arange(-3.5, 2.0, 1.0)
DECADE_LAYERS = [1e-3, 1e-2, 1e-1, 1., 10.]


@pytest.fixture(scope='module')
def db():
    """Raw cross sections from the mini db: molecular in cm2/molecule, continuum in cm-1 amagat-2."""
    conn = sqlite3.connect(MINI_OPACITY_DB)
    cur = conn.cursor()

    def read(sql, *args):
        return np.load(io.BytesIO(cur.execute(sql, args).fetchone()[0])).astype(float)

    yield {
        'molecular': lambda mol, p, t: read(
            'select opacity from molecular where molecule=? and pressure=? and temperature=?', mol, p, t),
        'continuum': lambda mol, t: read(
            'select opacity from continuum where molecule=? and temperature=?', mol, t),
    }
    conn.close()


@pytest.fixture(scope='module')
def linear_opacity():
    return jdi.opannection(filename_db=str(MINI_OPACITY_DB), query_method='linear')


def atmosphere(opacity, pressure_levels, temperature, clouds=None):
    """
    The atmosphere picaso builds for a spectrum (the same setup steps as justdoit.picaso), with its
    opacities looked up. Isothermal H2/He/H2O, constant gravity.
    """
    case = jdi.inputs()
    case.gravity(gravity=GRAVITY, gravity_unit=u.Unit('cm/s**2'))
    case.phase_angle(0, num_gangle=2, num_tangle=1)
    case.atmosphere(df=pd.DataFrame({'pressure': pressure_levels, 'temperature': temperature,
                                     'H2': X_H2, 'He': X_HE, 'H2O': X_H2O}))
    if clouds is not None:
        case.clouds(**clouds)
    inputs = case.inputs
    atm = ATMSETUP(inputs)
    atm.surf_reflect, atm.hard_surface, atm.get_lvl_flux = 0, 0, False
    atm.wavenumber = opacity.wno
    atm.planet.gravity = inputs['planet']['gravity']
    atm.planet.radius = inputs['planet']['radius']
    atm.planet.mass = inputs['planet']['mass']
    atm.get_profile()
    atm.get_mmw()
    atm.get_density()
    atm.get_altitude()
    atm.get_column_density()
    atm.get_needed_continuum(opacity.rayleigh_molecules, opacity.avail_continuum)
    atm.get_clouds(opacity.wno)
    atm.molecules = np.array([m for m in atm.molecules if m in opacity.molecules])
    opacity.get_opacities(atm)
    return atm


def optical_depths(atm, opacity, **kwargs):
    """compute_opacity as justdoit calls it (test_mode=None: the default of False turns test mode on)."""
    return compute_opacity(atm, opacity, ngauss=1, stream=2, test_mode=None, raman=NO_RAMAN, **kwargs)


def per_layer_by_species(atm, opacity):
    return optical_depths(atm, opacity, return_mode=True)


def layer_arrays(atm, opacity, delta_eddington=False, **kwargs):
    """DTAU, TAU, W0, COSB, ftau_cld, ftau_ray, GCOS2 for the single gauss point."""
    out = optical_depths(atm, opacity, delta_eddington=delta_eddington, **kwargs)
    return [o[..., 0] for o in out[:7]]


# ---------------------------------------------------------------- looking up the database

@pytest.mark.parametrize('query_method', ['nearest', 'linear'])
def test_lookup_on_grid_points_returns_the_database_value(query_method, mini_opacity, linear_opacity, db):
    opacity = mini_opacity if query_method == 'nearest' else linear_opacity
    atmosphere(opacity, DECADE_LEVELS, 1000.)
    for i, p in enumerate(DECADE_LAYERS):
        expected = db['molecular']('H2O', p, 1000.) * AVOGADRO
        # the db stores float32, and linear goes through log10 and back at that precision
        rtol = FLOAT32 if query_method == 'nearest' else 1e-5
        np.testing.assert_allclose(opacity.molecular_opa['H2O'][i], expected, rtol=rtol)


def test_nearest_lookup_picks_the_closest_temperature_and_log_pressure(mini_opacity, db):
    """Layers at 1047 K: 2 bar is closest to 3 bar in log P, sqrt(10) bar is closest to 3 bar too."""
    atmosphere(mini_opacity, [1.6, 2.5, 4.0], 1047.)
    for i in range(2):
        np.testing.assert_allclose(mini_opacity.molecular_opa['H2O'][i],
                                   db['molecular']('H2O', 3., 1000.) * AVOGADRO, rtol=FLOAT32)


def test_linear_lookup_interpolates_log_opacity_in_inverse_temperature(linear_opacity, db):
    """Halfway between 1000 and 1100 K in 1/T, the opacity is the geometric mean of the two."""
    halfway = 2 / (1 / 1000. + 1 / 1100.)
    atmosphere(linear_opacity, [1 / np.sqrt(3), np.sqrt(3), 3 * np.sqrt(3)], halfway)  # layers at 1 and 3 bar
    for i, p in enumerate([1., 3.]):
        low, high = db['molecular']('H2O', p, 1000.), db['molecular']('H2O', p, 1100.)
        has_lines = (low > 0) & (high > 0)
        np.testing.assert_allclose(linear_opacity.molecular_opa['H2O'][i][has_lines],
                                   np.sqrt(low * high)[has_lines] * AVOGADRO, rtol=1e-5)


def test_linear_lookup_interpolates_log_opacity_in_log_pressure(linear_opacity, db):
    """Halfway between 1 and 3 bar in log P, the opacity is the geometric mean of the two."""
    atmosphere(linear_opacity, 3**np.array([0.25, 0.75, 1.25]), 1000.)  # first layer at sqrt(3) bar
    low, high = db['molecular']('H2O', 1., 1000.), db['molecular']('H2O', 3., 1000.)
    has_lines = (low > 0) & (high > 0)
    np.testing.assert_allclose(linear_opacity.molecular_opa['H2O'][0][has_lines],
                               np.sqrt(low * high)[has_lines] * AVOGADRO, rtol=1e-5)


@pytest.mark.parametrize('query_method', ['nearest', 'linear'])
def test_continuum_uses_the_nearest_temperature(query_method, mini_opacity, linear_opacity, db):
    opacity = mini_opacity if query_method == 'nearest' else linear_opacity
    atmosphere(opacity, DECADE_LEVELS, 1047.)
    for cia in ['H2H2', 'H2He']:
        np.testing.assert_array_equal(opacity.continuum_opa[cia],
                                      np.tile(db['continuum'](cia, 1000.), (len(DECADE_LAYERS), 1)))


# ---------------------------------------------------------------- layer optical depths

def test_line_optical_depth_is_cross_section_times_column(mini_opacity, db):
    """dtau = sigma x N, with N = colden / (mu m_u) molecules per cm2."""
    atm = atmosphere(mini_opacity, DECADE_LEVELS, 1000.)
    taus = per_layer_by_species(atm, mini_opacity)
    column = atm.layer['colden'] / (atm.layer['mmw'] * AMU)
    for i, p in enumerate(DECADE_LAYERS):
        np.testing.assert_allclose(taus['H2O'][i], db['molecular']('H2O', p, 1000.) * X_H2O * column[i],
                                   rtol=1e-6)


def test_cia_optical_depth_matches_isothermal_hydrostatic_integral(mini_opacity, db):
    """
    CIA goes as the square of density: dtau = k x1 x2 integral (n / n_L)^2 dz. In an isothermal,
    hydrostatic layer that integral is (P2^2 - P1^2) / (2 k T mu m_u g n_L^2).
    """
    temperature = 1000.
    atm = atmosphere(mini_opacity, DECADE_LEVELS, temperature)
    taus = per_layer_by_species(atm, mini_opacity)
    p = DECADE_LEVELS * 1e6
    integral = (p[1:]**2 - p[:-1]**2) / (2 * K_B * temperature * atm.layer['mmw'] * AMU * GRAVITY * LOSCHMIDT**2)
    for cia, x2 in [('H2H2', X_H2), ('H2He', X_HE)]:
        expected = db['continuum'](cia, temperature) * X_H2 * x2 * integral[:, np.newaxis]
        np.testing.assert_allclose(taus[cia], expected, rtol=1e-6)


def test_cia_optical_depth_goes_as_pressure_squared(mini_opacity):
    """Scaling every pressure by 10 keeps the layers' log thickness, and multiplies CIA by 100."""
    low = per_layer_by_species(atmosphere(mini_opacity, DECADE_LEVELS[:-1], 1000.), mini_opacity)
    high = per_layer_by_species(atmosphere(mini_opacity, DECADE_LEVELS[1:], 1000.), mini_opacity)
    # compare interior layers, away from the doubled column density in the top and bottom layers
    np.testing.assert_allclose(high['H2H2'][1:-1], 100 * low['H2H2'][1:-1], rtol=1e-10)


def test_h2_rayleigh_cross_section_is_close_to_dalgarno_williams(mini_opacity):
    """
    picaso uses the Hohm (1993) polarisability for H2; Dalgarno & Williams (1962) is the classic
    fit. The two differ by ~7%, so this checks units and wavelength dependence, not the third digit.
    """
    sigma = mini_opacity.rayleigh_opa['H2'] / AVOGADRO  # cm2 per molecule
    wave_angstrom = 1e8 / mini_opacity.wno
    dalgarno_williams = 8.14e-13 / wave_angstrom**4 + 1.28e-6 / wave_angstrom**6 + 1.61 / wave_angstrom**8
    np.testing.assert_allclose(sigma, dalgarno_williams, rtol=0.1)
    # far from the UV resonances, sigma lambda^4 is nearly constant
    flat = sigma * wave_angstrom**4
    assert np.max(flat) / np.min(flat) < 1.03


def test_rayleigh_optical_depth_sums_every_scatterer(mini_opacity):
    atm = atmosphere(mini_opacity, DECADE_LEVELS, 1000.)
    taus = per_layer_by_species(atm, mini_opacity)
    column = atm.layer['colden'] / (atm.layer['mmw'] * AMU)
    expected = sum(mini_opacity.rayleigh_opa[m] / AVOGADRO * x
                   for m, x in [('H2', X_H2), ('He', X_HE), ('H2O', X_H2O)])
    np.testing.assert_allclose(taus['rayleigh'], expected * column[:, np.newaxis], rtol=1e-6)


def test_total_optical_depth_is_the_sum_of_every_source(mini_opacity):
    atm = atmosphere(mini_opacity, DECADE_LEVELS, 1000.,
                     clouds=dict(g0=[0.6], w0=[0.9], opd=[2.], p=[0.], dp=[2.]))
    taus = per_layer_by_species(atm, mini_opacity)
    assert set(taus) == {'H2H2', 'H2He', 'H2O', 'rayleigh', 'cloud'}
    dtau, tau = layer_arrays(atm, mini_opacity)[:2]
    np.testing.assert_allclose(dtau, sum(taus.values()), rtol=1e-12)
    np.testing.assert_array_equal(tau[0], 0)
    np.testing.assert_allclose(tau[1:], np.cumsum(dtau, axis=0), rtol=1e-12)


# ---------------------------------------------------------------- scattering properties

def test_clear_atmosphere_scatters_only_by_rayleigh(mini_opacity):
    atm = atmosphere(mini_opacity, DECADE_LEVELS, 1000.)
    taus = per_layer_by_species(atm, mini_opacity)
    dtau, _, w0, cosb, ftau_cld, ftau_ray, gcos2 = layer_arrays(atm, mini_opacity)
    np.testing.assert_allclose(w0, 0.99999 * taus['rayleigh'] / dtau, rtol=1e-12)
    np.testing.assert_array_equal(cosb, 0)
    np.testing.assert_array_equal(ftau_cld, 0)
    np.testing.assert_array_equal(ftau_ray, 1)
    np.testing.assert_array_equal(gcos2, 0.5)


def test_cloud_and_rayleigh_scattering_are_weighted_by_optical_depth(mini_opacity):
    w0_cld, g0_cld = 0.9, 0.6
    atm = atmosphere(mini_opacity, DECADE_LEVELS, 1000.,
                     clouds=dict(g0=[g0_cld], w0=[w0_cld], opd=[2.], p=[0.], dp=[2.]))
    taus = per_layer_by_species(atm, mini_opacity)
    tau_ray, tau_cld = taus['rayleigh'], taus['cloud']
    assert np.any(tau_cld > 0) and np.any(tau_cld == 0)  # some cloudy layers, some clear
    dtau, _, w0, cosb, ftau_cld, ftau_ray, gcos2 = layer_arrays(atm, mini_opacity)
    cloud_scattering = atm.layer['cloud']['w0'] * tau_cld
    np.testing.assert_allclose(w0, (0.99999 * tau_ray + cloud_scattering) / dtau, rtol=1e-12)
    np.testing.assert_allclose(ftau_cld, cloud_scattering / (cloud_scattering + tau_ray), rtol=1e-12)
    np.testing.assert_allclose(ftau_ray, tau_ray / (tau_ray + cloud_scattering), rtol=1e-12)
    np.testing.assert_allclose(gcos2, 0.5 * ftau_ray, rtol=1e-12)
    np.testing.assert_array_equal(cosb, atm.layer['cloud']['g0'])
    np.testing.assert_array_equal(cosb[tau_cld > 0], g0_cld)


def test_optically_thick_cloud_sets_the_scattering_properties(mini_opacity):
    atm = atmosphere(mini_opacity, DECADE_LEVELS, 1000.,
                     clouds=dict(g0=[0.6], w0=[0.9], opd=[1e12], p=[0.], dp=[2.]))
    cloudy = atm.layer['cloud']['opd'] > 0
    _, _, w0, cosb, ftau_cld = layer_arrays(atm, mini_opacity)[:5]
    np.testing.assert_allclose(w0[cloudy], 0.9, rtol=1e-6)
    np.testing.assert_allclose(cosb[cloudy], 0.6)
    np.testing.assert_allclose(ftau_cld[cloudy], 1, rtol=1e-6)


def test_patchy_clouds_scale_only_the_cloud_optical_depth(mini_opacity):
    atm = atmosphere(mini_opacity, DECADE_LEVELS, 1000.,
                     clouds=dict(g0=[0.6], w0=[0.9], opd=[2.], p=[0.], dp=[2.]))
    taus = per_layer_by_species(atm, mini_opacity)
    dtau = layer_arrays(atm, mini_opacity, do_holes=True, fthin_cld=0.25)[0]
    clear = sum(v for k, v in taus.items() if k != 'cloud')
    np.testing.assert_allclose(dtau, clear + 0.25 * taus['cloud'], rtol=1e-12)


# ---------------------------------------------------------------- delta-Eddington scaling

@pytest.mark.parametrize('stream', [2, 4])
def test_delta_eddington_removes_the_forward_peak_and_keeps_absorption(stream, mini_opacity):
    """
    Joseph et al. (1976): with f = g^stream, the forward-scattered fraction f of the scattering is
    moved into the direct beam. Absorption optical depth dtau (1 - w0) must not change.
    """
    atm = atmosphere(mini_opacity, DECADE_LEVELS, 1000.,
                     clouds=dict(g0=[0.6], w0=[0.9], opd=[2.], p=[0.], dp=[2.]))
    dtau, _, w0, cosb = layer_arrays(atm, mini_opacity)[:4]
    out = compute_opacity(atm, mini_opacity, ngauss=1, stream=stream, test_mode=None, raman=NO_RAMAN,
                          delta_eddington=True)
    dtau_d, tau_d, w0_d, cosb_d = [o[..., 0] for o in out[:4]]
    f = out[12][..., 0]
    np.testing.assert_allclose(f, cosb**stream, rtol=1e-12)
    np.testing.assert_allclose(dtau_d * (1 - w0_d), dtau * (1 - w0), rtol=1e-12)
    np.testing.assert_allclose(dtau_d * w0_d, dtau * w0 * (1 - f), rtol=1e-12)
    np.testing.assert_allclose(cosb_d, (cosb - f) / (1 - f), rtol=1e-12)
    np.testing.assert_allclose(tau_d[1:], np.cumsum(dtau_d, axis=0), rtol=1e-12)
    # the unscaled values are returned as well, for the single scattering phase functions
    np.testing.assert_array_equal(out[7][..., 0], dtau)
    np.testing.assert_array_equal(out[9][..., 0], w0)


def test_delta_eddington_does_nothing_without_forward_scattering(mini_opacity):
    """With only Rayleigh scattering, g = 0, so the scaling is the identity."""
    atm = atmosphere(mini_opacity, DECADE_LEVELS, 1000.)
    scaled = optical_depths(atm, mini_opacity, delta_eddington=True)
    unscaled = optical_depths(atm, mini_opacity, delta_eddington=False)
    for i in range(4):
        np.testing.assert_array_equal(scaled[i], unscaled[i])
