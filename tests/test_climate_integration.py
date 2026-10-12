"""
Integration test: the same brown dwarf climate model converged with preweighted correlated-k tables
and with resortrebin (on-the-fly mixing of per-molecule tables with the same Visscher 2121
chemistry) should give the same pressure-temperature profile. It runs at the highest metallicity
with a preweighted table ([M/H] = +2.0, solar C/O), where differences in how the two methods weight
gases by abundance are largest.

Each climate run takes a few minutes and needs data that is not in git, so this is skipped in CI.
Run it locally with:  pytest -m integration
The model setup lives in reference/scripts/compare_preweighted_resortrebin.py, which also plots it.
"""
import importlib.util
import os

import numpy as np
import pytest

from conftest import REFDATA, ck_tables_available

FEH, CO_ABSOLUTE = 2.0, 0.46
PREWEIGHTED_TABLE = REFDATA / 'opacities' / 'preweighted' / f'sonora_2121grid_feh{FEH:.1f}_co{CO_ABSOLUTE:.2f}.hdf5'
BOBCAT_GUESS = REFDATA / 'sonora_grids' / 'bobcat' / 't700g316nc_m0.0.cmp.gz'
SCRIPT = REFDATA / 'scripts' / 'compare_preweighted_resortrebin.py'

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (PREWEIGHTED_TABLE.exists() and BOBCAT_GUESS.exists() and ck_tables_available()),
                       reason='needs the feh+2.0 preweighted table, resortrebin tables and Sonora Bobcat profiles'),
]


def _load_script():
    spec = importlib.util.spec_from_file_location('compare_preweighted_resortrebin', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def profiles():
    script = _load_script()
    return {method: script.run_climate(method, feh=FEH, co_absolute=CO_ABSOLUTE)
            for method in ['preweighted', 'resortrebin']}


def _temperature_difference(profiles, method):
    p_ref, t_ref = profiles['preweighted']
    p, t = profiles[method]
    return t - np.interp(np.log10(p), np.log10(p_ref), t_ref)


def test_both_methods_converge_to_physical_profiles(profiles):
    for method, (pressure, temperature) in profiles.items():
        assert np.all(np.isfinite(temperature)), method
        assert np.all(np.diff(pressure) > 0), method
        # a non-irradiated brown dwarf gets hotter with depth
        assert temperature[-1] > temperature[0], method


def test_resortrebin_matches_preweighted(profiles):
    """Tolerances are set from the converged runs on 2026-10-12: median -0.5 K, max |dT| 33 K."""
    dt = _temperature_difference(profiles, 'resortrebin')
    assert abs(np.median(dt)) < 10
    assert np.max(np.abs(dt)) < 50
