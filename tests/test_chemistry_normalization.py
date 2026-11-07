"""
Mixing ratios from the equilibrium chemistry grids sum to 1 in every level (issue #339).

The grids list only a subset of the species (and condensates) in the equilibrium calculation, so
the raw interpolated mixing ratios sum to less than 1 (to ~0.976 at 100x solar). chem_interp
renormalizes them, which must not change the ratio between any two species.

Only gas mixing ratios may take part. A chemistry table can carry other columns: the preweighted
tables' full_abunds stores a row index, the visscher grids list condensates (C-gr, graphite), and a user's
chemistry_table could have anything (kz, mmw, a saved pandas index). chem_interp ignores all of
them, so they are neither counted in the sum nor written into the atmosphere profile.

Uses the visscher 2121 grid in git.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from picaso import justdoit as jdi
from picaso.atmsetup import is_gas_species  # name rules are tested in test_species_names.py

NLEVEL = 60
NOT_GAS = ['index', 'kz', 'kzz', 'mmw', 'Unnamed: 0', 'C-gr', 'graphite']


def read_grid():
    """The 100x solar, C/O = 0.55 visscher 2121 grid, with pressure in bar."""
    grid_file = Path(jdi.__refdata__) / 'chemistry' / 'visscher_grid_2121' / 'sonora_2121grid_feh2.0_co0.55.txt'
    cols = pd.read_csv(grid_file).keys()[0].replace('T(K)', 'temperature').replace('P(bar)', 'pressure').split()
    grid = pd.read_csv(grid_file, sep=r'\s+', skiprows=1, header=None, names=cols)
    return grid.assign(pressure=10**grid['pressure'])


def interpolate(grid, T=None, P=None, profile_kz=None):
    """Runs chem_interp on `grid` for a fresh profile, returns the profile and any warnings."""
    case = jdi.inputs(calculation='browndwarf')
    case.add_pt(T=np.linspace(500, 2500, NLEVEL) if T is None else T,
                P=np.logspace(-6, 2, NLEVEL) if P is None else P)
    if profile_kz is not None:
        case.inputs['atmosphere']['profile']['kz'] = profile_kz
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        case.chem_interp(grid)
    return case.inputs['atmosphere']['profile'], [str(w.message) for w in caught]


def gases(profile):
    return profile[[c for c in profile.columns if is_gas_species(c)]]


@pytest.mark.parametrize('chem_method', ['visscher', 'visscher_1060'])
@pytest.mark.parametrize('mh', [1, 100])
def test_grid_mixing_ratios_sum_to_one(mh, chem_method):
    """Both grids in git: only gases reach the profile, they sum to 1, and nothing warns."""
    case = jdi.inputs(calculation='browndwarf')
    case.add_pt(T=np.linspace(500, 2500, NLEVEL), P=np.logspace(-6, 2, NLEVEL))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        case.atmosphere(mh=mh, cto_relative=1, chem_method=chem_method)
    assert not any('chem_interp ignored' in str(w.message) for w in caught)
    profile = case.inputs['atmosphere']['profile']
    assert set(profile.columns) - {'pressure', 'temperature'} == set(gases(profile).columns)
    np.testing.assert_allclose(gases(profile).sum(axis=1), 1, rtol=1e-12)


def test_on_a_grid_point_the_result_is_the_grid_row_renormalized():
    """Where a level sits exactly on a grid (T, P) point there is no interpolation, so the result is
    that grid row's gases divided by their sum: every gas is scaled by the same factor."""
    grid = read_grid()
    rows = grid[(grid['temperature'] == 1500.) & (grid['pressure'] > 1e-4) & (grid['pressure'] < 30)]
    profile, _ = interpolate(grid, T=rows['temperature'].values, P=rows['pressure'].values)

    raw = rows.drop(columns=['pressure', 'temperature', 'C-gr']).reset_index(drop=True)
    expected = raw.div(raw.sum(axis=1), axis=0)
    for species in ['H2', 'He', 'H2O', 'CO', 'CH4']:
        np.testing.assert_allclose(profile[species].values, expected[species].values, rtol=1e-10)


def test_columns_that_are_not_gas_are_not_counted_or_written():
    """
    Extra columns, however large, do not change the result: the gases are the same as without them,
    they sum to 1, and none of the extra columns appears in the profile. The preweighted tables'
    `index` column (the row number, up to ~1500) once divided every mixing ratio by its value.
    """
    clean = read_grid().drop(columns=['C-gr'])
    messy = clean.copy()
    messy.insert(0, 'index', np.arange(len(messy)) + 1.)
    messy['kz'] = 1e9
    messy['kzz'] = 1e10
    messy['mmw'] = 2.3
    messy['Unnamed: 0'] = np.arange(len(messy)) + 1.
    messy['C-gr'] = read_grid()['C-gr']

    expected, _ = interpolate(clean)
    profile, _ = interpolate(messy)

    assert not set(NOT_GAS) & set(profile.columns)
    pd.testing.assert_frame_equal(profile, expected, rtol=1e-12)
    np.testing.assert_allclose(gases(profile).sum(axis=1), 1, rtol=1e-12)


def test_kz_already_in_the_profile_is_kept():
    """A kz column in the chemistry table must not overwrite the profile's own kz."""
    grid = read_grid()
    grid['kz'] = 1.0
    profile_kz = np.logspace(5, 9, NLEVEL)
    profile, _ = interpolate(grid, profile_kz=profile_kz)
    np.testing.assert_array_equal(profile['kz'].values, profile_kz)


def test_ignored_columns_warn_unless_expected():
    """The grids' own non-gas columns (index, condensates) are dropped quietly; anything else warns."""
    grid = read_grid()
    grid.insert(0, 'index', np.arange(len(grid)) + 1.)
    _, caught = interpolate(grid)
    assert not any('chem_interp ignored' in m for m in caught)

    grid['mmw'] = 2.3
    _, caught = interpolate(grid)
    assert any("chem_interp ignored" in m and "'mmw'" in m for m in caught)


def test_isotopologue_columns_are_kept():
    """A custom chemistry_table can list isotopologues (e.g. 13C-16O); they are gases, so they are
    interpolated, counted in the sum and written to the profile."""
    grid = read_grid()
    grid['13C-16O'] = grid['CO'] * 0.01
    profile, caught = interpolate(grid)
    assert '13C-16O' in profile
    assert not any('chem_interp ignored' in m for m in caught)
    np.testing.assert_allclose(gases(profile).sum(axis=1), 1, rtol=1e-12)
    np.testing.assert_allclose(profile['13C-16O'] / profile['CO'], 0.01, rtol=1e-10)
