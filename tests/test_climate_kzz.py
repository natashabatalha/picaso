"""
Integration test: with a self-consistent kzz, climate() returns the converged kzz as the `kz`
column of ptchem_df, so it is saved with the profile (issue #405). It is the last kzz profile
of the run, the same as the last nlevel values of all_kzz.

A 600 K brown dwarf on the resortrebin tables converges in about a minute.
Run it locally with:  pytest -m integration
"""
import warnings

import astropy.units as u
import numpy as np
import pytest

from picaso import justdoit as jdi

TEFF = 600.
NLEVEL = 51

pytestmark = [pytest.mark.integration, pytest.mark.needs_ck_tables]


@pytest.fixture(scope='module')
def climate_out():
    opacity = jdi.opannection(method='resortrebin')
    case = jdi.inputs(calculation='browndwarf', climate=True)
    case.gravity(gravity=100., gravity_unit=u.Unit('m/(s**2)'))
    case.effective_temp(TEFF)
    pt = case.guillot_pt(TEFF, nlevel=NLEVEL, T_int=TEFF, p_bottom=1, p_top=-6)
    case.inputs_climate(temp_guess=pt['temperature'].values, pressure=pt['pressure'].values, rcb_guess=40)
    case.atmosphere(mh=1, cto_relative=1, chem_method='visscher')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return case.climate(opacity, save_all_kzz=True, self_consistent_kzz=True, verbose=False)


def test_ptchem_df_has_the_converged_kzz(climate_out):
    df = climate_out['ptchem_df']
    assert df.columns[2] == 'kz'  # right after pressure and temperature
    assert np.all(np.isfinite(df['kz'])) and np.all(df['kz'] > 0)
    np.testing.assert_array_equal(df['kz'].values, climate_out['all_kzz'][-NLEVEL:])


def test_flux_ratio_is_finite_or_nan(climate_out):
    """fnet/fnetir never divides by zero (issue #326)."""
    ratio = climate_out['fnet/fnetir']
    net_ir = climate_out['flux_balance']['flux_net_ir']
    assert np.all(np.isfinite(ratio[net_ir != 0]))
    assert np.all(np.isnan(ratio[net_ir == 0]))
