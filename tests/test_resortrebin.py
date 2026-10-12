"""
Tests for the on-the-fly correlated-k mixing used by climate runs with method='resortrebin':
optics.RetrieveCKs.mix_my_opacities_gasesfly, which looks up each molecule's k-table at the four
grid neighbours of every layer, combines the gases with resort-rebin (deq_chem.mix_2_gases), and
interpolates in log pressure and 1/temperature.

These use the mini H2O/CO2 tables in reference/base_cases/testing/resortrebin_mini/ so they run in
CI. Expected values come straight from the tables, read here with h5py.
"""
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

from conftest import MINI_CK_DIR
from picaso.atmsetup import normalize_exclude_mol

AVOGADRO = 6.02214086e+23


@pytest.fixture(scope='module')
def ln_k():
    """The raw tables, ln(k) with axes [pressure, temperature, wavenumber, gauss point]."""
    tables = {}
    for molecule in ['H2O', 'CO2']:
        with h5py.File(MINI_CK_DIR / f'{molecule}_1460.hdf5', 'r') as f:
            tables[molecule] = f['kcoeffs'][:]
    return tables


def layers(pressure_bar, temperature, mixingratios):
    """The parts of picaso's atmosphere class that the mixing reads."""
    return SimpleNamespace(
        c=SimpleNamespace(nlayer=len(pressure_bar), pconv=1.0),
        molecules=list(mixingratios),
        layer={'pressure': np.asarray(pressure_bar, dtype=float),
               'temperature': np.asarray(temperature, dtype=float),
               'mixingratios': pd.DataFrame(mixingratios)})


def mixed_opacity(opacity, atmosphere, exclude_mol=1):
    opacity.mix_my_opacities_gasesfly(atmosphere, exclude_mol=exclude_mol)
    return opacity.molecular_opa.copy()


def test_single_gas_on_grid_nodes_returns_the_table(mini_ck_opacity, ln_k):
    """With one gas and layers exactly on grid nodes there is nothing to mix or interpolate."""
    ip, it = [2, 9, 15], [0, 7, 18]
    P, T = mini_ck_opacity.pressures, mini_ck_opacity.temps
    opa = mixed_opacity(mini_ck_opacity, layers(P[ip], T[it], {'H2O': [1e-3] * 3}))
    np.testing.assert_allclose(opa, np.exp(ln_k['H2O'][ip, it]) * AVOGADRO, rtol=1e-12)


def test_interpolates_in_log_pressure_and_inverse_temperature(mini_ck_opacity, ln_k):
    """Halfway between nodes in log P and 1/T, ln(k) is the mean of the four corners."""
    P, T = mini_ck_opacity.pressures, mini_ck_opacity.temps
    pressure = np.sqrt(P[5] * P[6])
    temperature = 1 / (0.5 / T[3] + 0.5 / T[4])
    opa = mixed_opacity(mini_ck_opacity, layers([pressure], [temperature], {'H2O': [1e-3]}))
    corners = ln_k['H2O'][5:7, 3:5]
    expected = np.exp(corners.mean(axis=(0, 1))) * AVOGADRO
    np.testing.assert_allclose(opa[0], expected, rtol=1e-12)


def test_two_gas_mix_is_a_valid_k_distribution(mini_ck_opacity, ln_k):
    """
    Resort-rebin sorts the combined k-coefficients, so the mix must increase with g, and it should
    preserve the band-mean (abundance-weighted) opacity apart from rebinning error.
    """
    P, T = mini_ck_opacity.pressures, mini_ck_opacity.temps
    x_h2o, x_co2 = 1e-3, 1e-4
    opa = mixed_opacity(mini_ck_opacity, layers([P[8]], [T[10]], {'H2O': [x_h2o], 'CO2': [x_co2]}))[0]
    assert np.all(np.diff(mini_ck_opacity.gauss_pts) > 0)
    assert np.all(np.diff(opa, axis=1) >= 0), 'mixed k is not sorted in g'

    weights = np.asarray(mini_ck_opacity.gauss_wts)
    k_h2o = np.exp(ln_k['H2O'][8, 10]) * AVOGADRO
    k_co2 = np.exp(ln_k['CO2'][8, 10]) * AVOGADRO
    expected_mean = ((x_h2o * k_h2o + x_co2 * k_co2) @ weights) / (x_h2o + x_co2)
    ratio = (opa @ weights) / expected_mean
    assert np.median(ratio) == pytest.approx(1, abs=0.01)
    assert np.all((ratio > 0.8) & (ratio < 1.2))


def test_negligible_or_excluded_gas_drops_out(mini_ck_opacity, ln_k):
    P, T = mini_ck_opacity.pressures, mini_ck_opacity.temps
    h2o_only = np.exp(ln_k['H2O'][8, 10]) * AVOGADRO

    trace_co2 = mixed_opacity(mini_ck_opacity, layers([P[8]], [T[10]], {'H2O': [1e-3], 'CO2': [1e-20]}))
    np.testing.assert_allclose(trace_co2[0], h2o_only, rtol=1e-9)

    excluded = mixed_opacity(mini_ck_opacity, layers([P[8]], [T[10]], {'H2O': [1e-3], 'CO2': [1e-4]}),
                             exclude_mol=normalize_exclude_mol('CO2'))
    np.testing.assert_allclose(excluded[0], h2o_only, rtol=1e-12)


@pytest.mark.needs_ck_tables
def test_mini_tables_match_full_tables(mini_ck_opacity):
    """Guards the mini tables themselves: within their temperature range they match the full set."""
    from picaso import justdoit as jdi
    full = jdi.opannection(method='resortrebin', preload_gases=['H2O', 'CO2'])
    pressure = np.logspace(-6, 2, 30)
    temperature = np.linspace(550, 1550, 30)
    mix = {'H2O': [1e-3] * 30, 'CO2': [1e-5] * 30}
    np.testing.assert_allclose(mixed_opacity(mini_ck_opacity, layers(pressure, temperature, mix)),
                               mixed_opacity(full, layers(pressure, temperature, mix)), rtol=1e-3)
