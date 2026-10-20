"""
Integration test: cloud-free brown dwarf climate models converged by picaso should reproduce the
Sonora Bobcat grid (Marley et al. 2021), which used the same opacities and Visscher chemistry with
an independent radiative-convective code. Five models span Teff = 500-1800 K and g = 316-3160 m/s2.

Each run uses Bobcat's own pressure grid and the solar preweighted table (C/O = 0.46, Bobcat's
chemistry), but starts from a Guillot profile 400-1100 K away from the answer, so agreement means
the solver found it rather than stayed put. The five runs take about 5 minutes and need data that
is not in git, so this is skipped in CI.
Run it locally with:  pytest -m integration
"""
import warnings

import astropy.units as u
import numpy as np
import pytest

from conftest import REFDATA
from picaso import justdoit as jdi

PREWEIGHTED_TABLE = REFDATA / 'opacities' / 'preweighted' / 'sonora_2121grid_feh0.0_co0.46.hdf5'
BOBCAT_DIR = REFDATA / 'sonora_grids' / 'bobcat'
SIGMA_SB = 5.670374419e-5  # cgs
# (Teff K, gravity m/s2)
MODELS = [(500, 316), (700, 1000), (1000, 1000), (1300, 316), (1800, 3160)]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (PREWEIGHTED_TABLE.exists() and BOBCAT_DIR.is_dir()),
                       reason='needs the solar preweighted table and the Sonora Bobcat profiles'),
]


def bobcat_profile(teff, gravity):
    """Pressure (bar) and temperature (K) of the Bobcat solar metallicity model."""
    return np.loadtxt(BOBCAT_DIR / f't{teff}g{gravity}nc_m0.0.cmp.gz', usecols=[1, 2], unpack=True, skiprows=1)


@pytest.fixture(scope='module')
def opacity():
    return jdi.opannection(ck_db=str(PREWEIGHTED_TABLE), method='preweighted')


@pytest.fixture(scope='module', params=MODELS, ids=[f't{t}g{g}' for t, g in MODELS])
def model(request, opacity):
    teff, gravity = request.param
    pressure, t_bobcat = bobcat_profile(teff, gravity)
    case = jdi.inputs(calculation='browndwarf', climate=True)
    case.gravity(gravity=gravity, gravity_unit=u.Unit('m/(s**2)'))
    case.effective_temp(teff)
    guillot = case.guillot_pt(teff, nlevel=len(pressure), T_int=teff,
                              p_bottom=np.log10(pressure[-1]), p_top=np.log10(pressure[0]))
    t_guess = np.interp(np.log10(pressure), np.log10(guillot['pressure'].values), guillot['temperature'].values)
    case.inputs_climate(temp_guess=t_guess, pressure=pressure, rcb_guess=len(pressure) - 12, rfacv=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = case.climate(opacity, verbose=False)
    return dict(teff=teff, pressure=pressure, t_bobcat=t_bobcat, t_guess=t_guess, out=out)


def test_climate_converges_from_a_distant_guess(model):
    out = model['out']
    assert out['converged'] == 1
    np.testing.assert_allclose(out['pressure'], model['pressure'], rtol=1e-12)
    # the starting guess is far from the answer, so matching Bobcat is not inherited from it
    assert np.max(np.abs(model['t_guess'] - model['t_bobcat'])) > 300


def test_converged_model_emits_sigma_teff4(model):
    net_ir_top = model['out']['flux_balance']['flux_net_ir'][0]
    assert (net_ir_top / SIGMA_SB)**0.25 == pytest.approx(model['teff'], rel=0.005)


def test_profile_matches_sonora_bobcat(model):
    """
    Tolerances are set from the runs on 2026-10-20: median offset +0.6 to +3.8 K, and at most 3.8%
    of the local temperature anywhere from 1e-4 to ~500 bar (largest in the deep adiabat of the
    coolest models, and at the top of the 1800 K model).
    """
    dt = model['out']['temperature'] - model['t_bobcat']
    assert abs(np.median(dt)) < 10
    assert np.max(np.abs(dt) / model['t_bobcat']) < 0.05
