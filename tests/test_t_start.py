"""
The modular climate solver (climate.t_start, which calls newton_raphson_solver) should reproduce
the original monolithic one (climate.t_start_legacy). Both are given the exact inputs that
climate.profile builds for its first call, for a brown dwarf with a convective zone on the mini
resortrebin H2O/CO2 tables in git, so this runs in CI.

The first call of each solver compiles it with numba, which takes ~30 s.
"""
import copy
import warnings

import astropy.units as u
import numpy as np
import pytest

from picaso import climate
from picaso import justdoit as jdi

OUTPUTS = ['temp', 'dtdp', 'all_profiles', 'flux_net_ir_layer', 'flux_net_v', 'flux_plus_ir_attop']


class _StopClimate(Exception):
    pass


@pytest.fixture(scope='module')
def solver_inputs(mini_ck_opacity):
    """Arguments of the first t_start call in a brown dwarf climate run (captured, then stopped)."""
    cl = jdi.inputs(calculation='browndwarf', climate=True)
    cl.gravity(gravity=100, gravity_unit=u.Unit('m/(s**2)'))
    cl.effective_temp(600)
    # keeps the guess mostly inside the 500-1600 K of the mini tables
    pt = cl.guillot_pt(600, nlevel=51, T_int=600, p_bottom=1, p_top=-6)
    cl.inputs_climate(temp_guess=pt['temperature'].values, pressure=pt['pressure'].values, rcb_guess=40)
    cl.atmosphere(mh=1, cto_relative=1, chem_method='visscher')

    captured = {}
    def capture(*args, **kwargs):
        captured['args'], captured['kwargs'] = args, kwargs
        raise _StopClimate

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(climate, 't_start', capture)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(_StopClimate):
                cl.climate(mini_ck_opacity)
    return captured['args'], dict(captured['kwargs'], verbose=0)


@pytest.fixture(scope='module')
def solutions(solver_inputs):
    args, kwargs = solver_inputs
    # deep copies so neither solver can see changes the other makes to its inputs
    return {name: getattr(climate, name)(*copy.deepcopy(args), **kwargs)
            for name in ['t_start_legacy', 't_start']}


def test_case_has_a_convective_zone_and_the_solver_moves_the_profile(solver_inputs, solutions):
    args, _ = solver_inputs
    nstr, atmosphere = args[1], args[6]
    assert nstr[2] > nstr[1], 'expected a convective zone so the adiabat reconstruction is exercised'
    assert np.max(np.abs(solutions['t_start_legacy'][0] - atmosphere.t_level)) > 10


@pytest.mark.parametrize('index, output', list(enumerate(OUTPUTS)))
def test_t_start_matches_t_start_legacy(solutions, index, output):
    np.testing.assert_allclose(solutions['t_start'][index], solutions['t_start_legacy'][index],
                               rtol=1e-10, atol=0, err_msg=output)
