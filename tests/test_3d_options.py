"""
Options that behave differently in 1d and 3d spectra (justdoit.picaso with dimension='3d').

3d honors exclude_mol like 1d, and options that only 1d supports (level fluxes and the single-column
geometry from phase_angle(num_gangle=1)) raise a clear error instead of failing later or being
silently ignored. These run on the mini opacity db and the bundled 3d hot Jupiter profile with fixed
abundances, so they need no chemistry grid or downloaded data.
"""
import warnings

import numpy as np
import pytest


@pytest.fixture(scope='module')
def gcm_profile():
    from picaso import justdoit as jdi
    ds = jdi.HJ_pt_3d(as_xarray=True)
    for molecule, abundance in [('H2', 0.85), ('He', 0.15), ('H2O', 3e-4), ('CO2', 1e-6)]:
        ds[molecule] = ds['temperature'] * 0 + abundance
    return ds


def brown_dwarf_3d(gcm_profile, **approx):
    from picaso import justdoit as jdi
    case = jdi.inputs(calculation='browndwarf')
    case.phase_angle(0, num_gangle=6, num_tangle=6)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        case.atmosphere_3d(gcm_profile, regrid=True, plot=False, verbose=False)
    case.gravity(gravity=25, gravity_unit=jdi.u.Unit('m/(s**2)'))
    case.approx(**approx)
    return case


def test_3d_honors_exclude_mol(mini_opacity, gcm_profile):
    from picaso.justdoit import normalize_exclude_mol
    with_h2o = brown_dwarf_3d(gcm_profile).spectrum(mini_opacity, calculation='thermal', dimension='3d')['thermal']
    case = brown_dwarf_3d(gcm_profile)
    case.inputs['atmosphere']['exclude_mol'] = normalize_exclude_mol(['H2O'])
    without_h2o = case.spectrum(mini_opacity, calculation='thermal', dimension='3d')['thermal']
    assert np.all(np.isfinite(with_h2o)) and not np.allclose(with_h2o, without_h2o)


def test_level_fluxes_are_1d_only(mini_opacity, gcm_profile):
    case = brown_dwarf_3d(gcm_profile, get_lvl_flux=True)
    with pytest.raises(Exception, match='only computed for 1d'):
        case.spectrum(mini_opacity, calculation='thermal', dimension='3d')


def test_single_column_geometry_is_1d_only(mini_opacity):
    from picaso import justdoit as jdi
    case = jdi.inputs(calculation='browndwarf')
    case.phase_angle(0, num_gangle=1, num_tangle=1)
    with pytest.raises(Exception, match='cannot be used for 3d'):
        jdi.picaso(case, mini_opacity, dimension='3d', calculation='reflected')
