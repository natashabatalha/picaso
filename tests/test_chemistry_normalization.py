"""
Mixing ratios from the equilibrium chemistry grids sum to 1 in every level (issue #339).

The grids list only a subset of the species (and condensates) in the equilibrium calculation, so
the raw interpolated mixing ratios sum to less than 1 (to ~0.976 at 100x solar). chem_interp
renormalizes the gases, which must not change the ratio between any two species. Columns that are
not gas mixing ratios are left alone: condensates (C-gr) and the row index that the preweighted
tables store in full_abunds.

Uses the visscher 2121 grid in git.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from picaso import justdoit as jdi

NLEVEL = 60


def grid_chemistry(mh):
    case = jdi.inputs(calculation='browndwarf')
    case.add_pt(T=np.linspace(500, 2500, NLEVEL), P=np.logspace(-6, 2, NLEVEL))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        case.atmosphere(mh=mh, cto_relative=1, chem_method='visscher')
    return case.inputs['atmosphere']['profile'].drop(columns=['pressure', 'temperature', 'C-gr'])


@pytest.mark.parametrize('mh', [1, 100])
def test_grid_mixing_ratios_sum_to_one(mh):
    np.testing.assert_allclose(grid_chemistry(mh).sum(axis=1), 1, rtol=1e-12)


def read_grid():
    grid_file = Path(jdi.__refdata__) / 'chemistry' / 'visscher_grid_2121' / 'sonora_2121grid_feh2.0_co0.55.txt'
    cols = pd.read_csv(grid_file).keys()[0].replace('T(K)', 'temperature').replace('P(bar)', 'pressure').split()
    grid = pd.read_csv(grid_file, sep=r'\s+', skiprows=1, header=None, names=cols)
    return grid


def test_on_a_grid_point_the_result_is_the_grid_row_renormalized():
    """Where a level sits exactly on a grid (T, P) point there is no interpolation, so the result is
    that grid row's gases divided by their sum: every gas is scaled by the same factor."""
    grid = read_grid()
    rows = grid[(grid['temperature'] == 1500.) & (grid['pressure'] > -4) & (grid['pressure'] < 1.5)]

    case = jdi.inputs(calculation='browndwarf')
    case.add_pt(T=rows['temperature'].values, P=10**rows['pressure'].values)
    case.chem_interp(grid.assign(pressure=10**grid['pressure']))
    result = case.inputs['atmosphere']['profile']

    raw = rows.drop(columns=['pressure', 'temperature', 'C-gr']).reset_index(drop=True)
    expected = raw.div(raw.sum(axis=1), axis=0)
    for species in ['H2', 'He', 'H2O', 'CO', 'CH4']:
        np.testing.assert_allclose(result[species].values, expected[species].values, rtol=1e-10)
    np.testing.assert_allclose(result['C-gr'].values, rows['C-gr'].values, rtol=1e-10)


def test_index_and_condensate_columns_are_not_renormalized():
    """The preweighted tables' full_abunds has an `index` column (the row number). It must not be
    counted as a gas: when it was, every mixing ratio was divided by up to ~1500 and climate broke."""
    grid = read_grid().assign(pressure=lambda g: 10**g['pressure'])
    grid.insert(0, 'index', np.arange(len(grid)) + 1.)
    case = jdi.inputs(calculation='browndwarf')
    case.add_pt(T=np.linspace(500, 2500, NLEVEL), P=np.logspace(-6, 2, NLEVEL))
    case.chem_interp(grid)
    gases = case.inputs['atmosphere']['profile'].drop(columns=['pressure', 'temperature', 'index', 'C-gr'])
    np.testing.assert_allclose(gases.sum(axis=1), 1, rtol=1e-12)
    assert case.inputs['atmosphere']['profile']['index'].max() > 1  # left as interpolated, not scaled
