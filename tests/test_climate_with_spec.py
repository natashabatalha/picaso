"""
Integration test: climate(..., with_spec=True) finishes by computing a thermal spectrum of the
converged atmosphere. That spectrum should be exactly what a separate spectrum() call on the
converged profile and chemistry gives, and, since the climate model is in energy balance, it should
emit sigma Teff^4.

A 600 K brown dwarf on the resortrebin tables converges in about a minute. The mini k-tables in git
are not used: with only H2O and CO2 over 500-1600 K the deep atmosphere (~2000 K) is off the table,
and the run reports converged without reaching radiative equilibrium.
Run it locally with:  pytest -m integration
"""
import warnings

import astropy.units as u
import numpy as np
import pytest

from picaso import justdoit as jdi

TEFF = 600.
GRAVITY = 100.  # m/s2
SIGMA_SB = 5.670374419e-5  # cgs

pytestmark = [pytest.mark.integration, pytest.mark.needs_ck_tables]


@pytest.fixture(scope='module')
def opacity():
    return jdi.opannection(method='resortrebin')


@pytest.fixture(scope='module')
def climate_out(opacity):
    case = jdi.inputs(calculation='browndwarf', climate=True)
    case.gravity(gravity=GRAVITY, gravity_unit=u.Unit('m/(s**2)'))
    case.effective_temp(TEFF)
    pt = case.guillot_pt(TEFF, nlevel=51, T_int=TEFF, p_bottom=1, p_top=-6)
    case.inputs_climate(temp_guess=pt['temperature'].values, pressure=pt['pressure'].values, rcb_guess=40)
    case.atmosphere(mh=1, cto_relative=1, chem_method='visscher')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return case.climate(opacity, with_spec=True, verbose=False)


@pytest.fixture(scope='module')
def separate_spectrum(opacity, climate_out):
    """A fresh brown dwarf case built only from the converged profile and chemistry."""
    case = jdi.inputs(calculation='browndwarf')
    case.gravity(gravity=GRAVITY, gravity_unit=u.Unit('m/(s**2)'))
    case.phase_angle(0)
    case.atmosphere(df=climate_out['ptchem_df'])
    return case.spectrum(opacity, calculation='thermal', full_output=True)


def test_climate_converges(climate_out):
    assert climate_out['converged'] == 1
    assert np.all(np.isfinite(climate_out['temperature']))


def test_with_spec_is_computed_on_the_converged_profile(climate_out):
    level = climate_out['spectrum_output']['full_output']['level']
    np.testing.assert_array_equal(level['temperature'], climate_out['temperature'])
    np.testing.assert_allclose(level['pressure'], climate_out['pressure'], rtol=1e-12)  # as_dict output is in bar


def test_with_spec_matches_a_separate_spectrum_call(climate_out, separate_spectrum):
    with_spec = climate_out['spectrum_output']
    np.testing.assert_array_equal(separate_spectrum['wavenumber'], with_spec['wavenumber'])
    np.testing.assert_allclose(separate_spectrum['thermal'], with_spec['thermal'], rtol=1e-12)
    assert separate_spectrum['effective_temperature'] == pytest.approx(with_spec['effective_temperature'], rel=1e-12)


def test_converged_atmosphere_carries_sigma_teff4_through_its_radiative_zone(climate_out):
    """Above the radiative-convective boundary, all of the internal heat is carried by radiation."""
    net_ir = climate_out['flux_balance']['flux_net_ir']
    radiative = slice(0, climate_out['cvz_locs'][1])
    np.testing.assert_allclose(net_ir[radiative], SIGMA_SB * TEFF**4, rtol=0.01)


def test_with_spec_spectrum_emits_sigma_teff4(climate_out):
    """
    The spectrum uses the monochromatic blackbody and integrates with trapz, while the climate
    balances fluxes integrated over each k-table bin, so the two agree to ~0.2% here.
    """
    assert climate_out['spectrum_output']['effective_temperature'] == pytest.approx(TEFF, rel=0.01)
