"""
atmosphere_4d(zero_point=...) for phase curves. 'night_transit' (the default) puts transit at phase 0 by
rotating the longitudes an extra 180 degrees; 'secondary_eclipse' does not rotate them. Reflected light
phase angles are star-planet-observer angles, so phase 0 is always full phase: there 'night_transit'
cannot apply, and PICASO warns and uses 'secondary_eclipse'. Uses the HJ_pt_3d example GCM in git.
"""
import warnings

import numpy as np
import pytest

from picaso import justdoit as jdi

N_PHASES = 4


def phase_curve_case(calculation):
    case = jdi.inputs()
    phase_grid = np.linspace(0, 2 * np.pi, N_PHASES)
    if calculation == 'reflected':
        case.phase_curve_geometry('reflected', phase_grid=phase_grid, num_gangle=6, num_tangle=6)
    else:
        case.phase_angle(phase_grid=phase_grid, num_gangle=6, num_tangle=6, calculation='thermal')
    return case


def run(calculation, zero_point):
    case = phase_curve_case(calculation)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        case.atmosphere_4d(jdi.HJ_pt_3d(as_xarray=True), shift=np.zeros(N_PHASES), zero_point=zero_point,
                           plot=False, verbose=False)
    zero_point_warnings = [str(w.message) for w in caught if 'zero_point' in str(w.message)]
    return case.inputs['shift'], zero_point_warnings


def test_thermal_night_transit_rotates_by_180_without_a_warning():
    shift, caught = run('thermal', 'night_transit')
    np.testing.assert_array_equal(shift, np.full(N_PHASES, 180.))
    assert caught == []


@pytest.mark.parametrize('calculation', ['thermal', 'reflected'])
def test_secondary_eclipse_does_not_rotate_or_warn(calculation):
    shift, caught = run(calculation, 'secondary_eclipse')
    np.testing.assert_array_equal(shift, np.zeros(N_PHASES))
    assert caught == []


def test_reflected_night_transit_warns_and_uses_secondary_eclipse():
    shift, caught = run('reflected', 'night_transit')
    np.testing.assert_array_equal(shift, np.zeros(N_PHASES))
    assert len(caught) == 1
    assert "zero_point='secondary_eclipse'" in caught[0]


def test_unknown_zero_point_is_an_error():
    with pytest.raises(Exception, match='zero point'):
        run('thermal', 'periastron')
