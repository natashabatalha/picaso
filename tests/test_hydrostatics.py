"""
Tests for the atmospheric structure picaso builds before any opacity is looked up (atmsetup.ATMSETUP):
mean molecular weight, number density, altitude, gravity and column density. Column density sets
every layer's optical depth, and altitude sets the transit radius.

The atmospheres are isothermal, so hydrostatic equilibrium has exact solutions to compare against:
    constant gravity:     z = z_ref + H ln(P_ref / P),  H = k T / (mu m_u g)
    gravity G M / z^2:    1/z = 1/R + (k T / (mu m_u G M)) ln(P / P_ref)
No opacity data is needed.
"""
import astropy.units as u
import numpy as np
import pandas as pd
import pytest

from picaso import justdoit as jdi
from picaso.atmsetup import ATMSETUP

K_B, AMU, G = 1.380649e-16, 1.66053906892e-24, 6.6743e-8  # cgs
TEMPERATURE = 1000.
SOLAR = {'H2': 0.85, 'He': 0.15}
# standard atomic weights; picaso uses the mass of the most common isotope, which is within 0.1%
STANDARD_WEIGHTS = {'H2': 2.016, 'He': 4.0026, 'H2O': 18.015, 'CO2': 44.009, 'CH4': 16.043,
                    'N2': 28.014, 'CO': 28.010, 'NH3': 17.031, 'Na': 22.990}
PRESSURE = np.logspace(-6, 2, 200)  # bar


def atmosphere(mixingratios=SOLAR, pressure=PRESSURE, temperature=TEMPERATURE, gravity=None,
               p_reference=1, constant_gravity=False):
    """
    Structure for an isothermal atmosphere, built with the same calls as justdoit.picaso. Give
    gravity (cm/s2) alone, or leave it out for a Jupiter mass and radius planet.
    """
    case = jdi.inputs()
    if gravity is None:
        case.gravity(mass=1, mass_unit=u.Unit('M_jup'), radius=1, radius_unit=u.Unit('R_jup'))
    else:
        case.gravity(gravity=gravity, gravity_unit=u.Unit('cm/s**2'))
    case.phase_angle(0, num_gangle=2, num_tangle=1)
    case.atmosphere(df=pd.DataFrame({'pressure': pressure, 'temperature': temperature, **mixingratios}))
    inputs = case.inputs
    atm = ATMSETUP(inputs)
    atm.planet.gravity = inputs['planet']['gravity']
    atm.planet.radius = inputs['planet']['radius']
    atm.planet.mass = inputs['planet']['mass']
    atm.get_profile()
    atm.get_mmw()
    atm.get_density()
    atm.get_altitude(p_reference=p_reference, constant_gravity=constant_gravity)
    atm.get_column_density()
    return atm


def reference_level(pressure, p_reference):
    """picaso moves the reference pressure to the first level at or below it, or to the bottom."""
    return min(np.searchsorted(pressure, p_reference), len(pressure) - 1)


def scale_height(atm):
    return K_B * TEMPERATURE / (atm.level['mmw'] * AMU * atm.planet.gravity)


# ---------------------------------------------------------------- composition

def test_molecular_weights_match_standard_atomic_weights():
    atm = atmosphere()
    weights = atm.get_weights(list(STANDARD_WEIGHTS))
    for molecule, weight in STANDARD_WEIGHTS.items():
        assert weights[molecule] == pytest.approx(weight, rel=1e-3), molecule


@pytest.mark.parametrize('mixingratios', [{'H2': 1.0}, SOLAR, {'N2': 0.78, 'O2': 0.21, 'CO2': 0.01}])
def test_mean_molecular_weight_is_the_abundance_weighted_mass(mixingratios):
    atm = atmosphere(mixingratios)
    weights = atm.get_weights(list(mixingratios))
    expected = sum(x * weights[m] for m, x in mixingratios.items())
    np.testing.assert_allclose(atm.level['mmw'], expected, rtol=1e-12)
    np.testing.assert_allclose(atm.layer['mmw'], expected, rtol=1e-12)


def test_layer_values_are_level_averages():
    """Layer temperature and mixing ratios are arithmetic means; layer pressure is the geometric mean."""
    temperature = np.linspace(500, 1500, len(PRESSURE))
    h2o = np.logspace(-5, -3, len(PRESSURE))
    atm = atmosphere({'H2': 0.85, 'He': 0.15 - h2o, 'H2O': h2o}, temperature=temperature)
    np.testing.assert_allclose(atm.layer['temperature'], 0.5 * (temperature[1:] + temperature[:-1]))
    np.testing.assert_allclose(atm.layer['pressure'], np.sqrt(PRESSURE[1:] * PRESSURE[:-1]) * 1e6)
    np.testing.assert_allclose(atm.layer['mixingratios']['H2O'], 0.5 * (h2o[1:] + h2o[:-1]))


def test_number_density_is_ideal_gas():
    atm = atmosphere()
    np.testing.assert_allclose(atm.level['den'], PRESSURE * 1e6 / (K_B * TEMPERATURE), rtol=1e-12)


# ---------------------------------------------------------------- column density

@pytest.mark.parametrize('gravity', [2500., None], ids=['constant', 'GM/r2'])
def test_column_density_is_pressure_difference_over_gravity(gravity):
    """Every layer, including the top and bottom ones, holds dP / g grams per cm2."""
    atm = atmosphere(gravity=gravity)
    np.testing.assert_allclose(atm.layer['colden'], np.diff(PRESSURE * 1e6) / atm.layer['gravity'], rtol=1e-12)
    if gravity is not None:
        np.testing.assert_allclose(atm.layer['gravity'], gravity, rtol=1e-12)
        np.testing.assert_allclose(atm.layer['colden'].sum(), (PRESSURE[-1] - PRESSURE[0]) * 1e6 / gravity,
                                   rtol=1e-12)


def test_layer_gravity_averages_gm_over_r2_at_its_levels():
    atm = atmosphere()
    level_gravity = G * atm.planet.mass / atm.level['z']**2
    np.testing.assert_allclose(atm.layer['gravity'], 0.5 * (level_gravity[1:] + level_gravity[:-1]), rtol=1e-6)


# ---------------------------------------------------------------- altitude

@pytest.mark.parametrize('p_reference', [1., 0.5, 1e-3])
def test_constant_gravity_altitude_is_the_isothermal_log_profile(p_reference):
    atm = atmosphere(p_reference=p_reference, constant_gravity=True)
    ref = reference_level(PRESSURE, p_reference)
    H = scale_height(atm)
    expected = atm.planet.radius + H * np.log(PRESSURE[ref] / PRESSURE)
    np.testing.assert_allclose(atm.level['z'], expected, rtol=1e-12)
    np.testing.assert_allclose(atm.level['dz'], H * np.log(PRESSURE[1] / PRESSURE[0]), rtol=1e-10)
    np.testing.assert_allclose(atm.level['scale_height'], H, rtol=1e-12)


@pytest.mark.parametrize('p_reference', [0.5, 1e4])
def test_reference_pressure_moves_onto_the_pressure_grid(p_reference):
    """So that the model does not depend on how the grid falls relative to p_reference."""
    atm = atmosphere(p_reference=p_reference, constant_gravity=True)
    ref = reference_level(PRESSURE, p_reference)
    assert atm.level['z'][ref] == atm.planet.radius
    assert PRESSURE[ref] >= min(p_reference, PRESSURE[-1])


def test_inverse_square_gravity_altitude_matches_the_exact_isothermal_solution():
    """
    picaso steps upward evaluating gravity at the lower level of each layer, so it agrees with the
    exact solution to first order in the layer thickness: ~2e-4 of the atmosphere's height here.
    """
    atm = atmosphere()
    ref = reference_level(PRESSURE, 1.)
    R, M = atm.planet.radius, atm.planet.mass
    mu = atm.level['mmw'] * AMU
    exact = 1 / (1 / R + K_B * TEMPERATURE / (mu * G * M) * np.log(PRESSURE / PRESSURE[ref]))
    assert atm.level['z'][ref] == R
    height = np.max(np.abs(exact - R))
    assert np.max(np.abs(atm.level['z'] - exact)) < 1e-3 * height
    # gravity weakens with height, so the atmosphere is taller than it would be at constant gravity
    constant = R + K_B * TEMPERATURE / (mu * atm.planet.gravity) * np.log(PRESSURE[ref] / PRESSURE)
    assert atm.level['z'][0] > constant[0]


def test_altitude_is_undefined_without_a_radius():
    """With gravity alone there is no radius to start from, but the column density is still right."""
    atm = atmosphere(gravity=2500.)
    assert np.all(np.isnan(atm.level['z']))
    assert np.all(np.isfinite(atm.layer['colden']))
