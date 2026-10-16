"""
Regression tests that pin the exact output of the 1D thermal solver, fluxes.get_thermal_1d.

test_thermal_rt.py checks get_thermal_1d against analytic limits at rtol=1e-6. These tests instead
compare every output array (flux at the top and the four level fluxes) to a stored snapshot at
rtol=1e-12, so that refactors and speedups can be shown not to change the answer. The level
fluxes are only computed when get_lvl_flux=True, and skipping them must give the same flux at the
top. The synthetic
cases cover every branch: no or strong scattering, positive asymmetry, a temperature inversion,
optical depths large enough to hit the exp(35) overflow guard, a hard surface with a
wavelength-dependent reflectivity, the bin-integrated (integrated_energy=True) Planck function
used by climate, and several
outgoing angles. The end-to-end cases run spectrum(calculation='thermal') on the mini opacity db
that lives in git, with level fluxes off and on, and with patchy clouds.

The snapshot is reference/base_cases/testing/thermal_1d_regression.npz. The spectrum_level_fluxes
entries were updated when spectrum() started computing level fluxes as the climate code does (the
blackbody integrated over the real wavenumber bins) instead of with zero-width bins, and stopped
letting the level-flux request change the thermal spectrum. Regenerate it only when a change to the
thermal solver is meant to change its output:

    python tests/test_thermal_regression.py
"""
import os
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('picaso_refdata', str(REPO_ROOT / 'reference'))

from picaso.fluxes import get_thermal_1d  # noqa: E402

GOLDEN = REPO_ROOT / 'reference' / 'base_cases' / 'testing' / 'thermal_1d_regression.npz'
MINI_OPACITY_DB = (REPO_ROOT / 'reference' / 'base_cases' / 'retrieval_example' /
                   'testing_file_only_H2O_CO2_example_opacities.db')
RTOL = 1e-12
LEVEL_KEYS = ('flux_minus', 'flux_plus', 'flux_minus_mdpt', 'flux_plus_mdpt')

# outgoing cos(angle); 0.5 is avoided because lamda*ubar1 -> 1 there when w0 -> 0, which makes
# the Toon terms ill conditioned and the comparison meaningless at rtol=1e-12
UBAR1_TWO = np.array([[0.13], [0.93]])
UBAR1_SIX = np.array([[0.11, 0.21], [0.62, 0.71], [0.88, 0.97]])
UBAR1_ONE = np.array([[0.62]])


ORIGINAL_NAMES = {'climate_integrated_energy': 'climate_calc_type1'}


def synthetic_case(name):
    """Inputs for one get_thermal_1d call; every case is built from its own fixed seed."""
    # seeded from the case name; a renamed case keeps the seed of its original name so its inputs
    # (and the snapshot) stay the same
    rng = np.random.default_rng(sum(map(ord, ORIGINAL_NAMES.get(name, name))))
    nlevel, nwno = 16, 16
    nlayer = nlevel - 1
    wno = np.logspace(2.5, 4.3, nwno)  # about 0.5 to 30 micron
    dwno = np.concatenate((np.diff(wno), [np.diff(wno)[-1]]))
    plevel = np.logspace(-1, 8, nlevel)  # dyne/cm2
    case = dict(wno=wno, dwno=dwno, plevel=plevel, ubar1=UBAR1_TWO, hard_surface=0,
                surf_reflect=np.zeros(nwno), integrated_energy=False)
    lognormal_dtau = lambda lo, hi: 10**rng.uniform(np.log10(lo), np.log10(hi), (nlayer, nwno))

    if name == 'absorbing_isothermal':
        case.update(tlevel=np.full(nlevel, 1000.), dtau=lognormal_dtau(1e-4, 10),
                    w0=np.full((nlayer, nwno), 1e-6), cosb=np.zeros((nlayer, nwno)))
    elif name == 'gradient_scattering':
        case.update(tlevel=np.linspace(500, 2000, nlevel), dtau=lognormal_dtau(1e-4, 50),
                    w0=rng.uniform(0.01, 0.95, (nlayer, nwno)), cosb=rng.uniform(-0.2, 0.8, (nlayer, nwno)))
    elif name == 'inversion_high_albedo':
        tlevel = 900 + 400 * np.exp(-((np.arange(nlevel) - 4) / 3.0)**2) + np.linspace(0, 800, nlevel)
        case.update(tlevel=tlevel, dtau=lognormal_dtau(1e-3, 5),
                    w0=rng.uniform(0.9, 0.999, (nlayer, nwno)), cosb=np.full((nlayer, nwno), 0.5))
    elif name == 'optically_thick_overflow':
        case.update(tlevel=np.linspace(300, 1500, nlevel), dtau=lognormal_dtau(10, 1000),
                    w0=rng.uniform(1e-4, 0.5, (nlayer, nwno)), cosb=rng.uniform(0, 0.5, (nlayer, nwno)))
    elif name == 'hard_surface':
        case.update(tlevel=np.linspace(200, 300, nlevel), dtau=lognormal_dtau(1e-3, 0.1),
                    w0=rng.uniform(1e-3, 0.3, (nlayer, nwno)), cosb=rng.uniform(0, 0.3, (nlayer, nwno)),
                    hard_surface=1, surf_reflect=rng.uniform(0, 0.6, nwno))
    elif name == 'climate_integrated_energy':
        case.update(tlevel=np.linspace(400, 1800, nlevel), dtau=lognormal_dtau(1e-4, 30),
                    w0=rng.uniform(1e-3, 0.8, (nlayer, nwno)), cosb=rng.uniform(0, 0.6, (nlayer, nwno)),
                    integrated_energy=True)
    elif name == 'six_angles':
        case.update(tlevel=np.linspace(600, 1600, nlevel), dtau=lognormal_dtau(1e-4, 20),
                    w0=rng.uniform(1e-3, 0.7, (nlayer, nwno)), cosb=rng.uniform(0, 0.6, (nlayer, nwno)),
                    ubar1=UBAR1_SIX)
    elif name == 'one_angle':
        case.update(tlevel=np.linspace(700, 1400, nlevel), dtau=lognormal_dtau(1e-4, 20),
                    w0=rng.uniform(1e-3, 0.7, (nlayer, nwno)), cosb=rng.uniform(0, 0.6, (nlayer, nwno)),
                    ubar1=UBAR1_ONE)
    else:
        raise ValueError(name)
    return case


SYNTHETIC_CASES = ['absorbing_isothermal', 'gradient_scattering', 'inversion_high_albedo',
                   'optically_thick_overflow', 'hard_surface', 'climate_integrated_energy', 'six_angles',
                   'one_angle']


def run_synthetic(name, **kwargs):
    c = synthetic_case(name)
    nlevel, nwno = len(c['tlevel']), len(c['wno'])
    numg, numt = c['ubar1'].shape
    return get_thermal_1d(nlevel, c['wno'], nwno, numg, numt, c['tlevel'], c['dtau'], c['w0'], c['cosb'],
                          c['plevel'], c['ubar1'], c['surf_reflect'], c['hard_surface'], c['dwno'],
                          c['integrated_energy'], **kwargs)


def synthetic_outputs(name):
    flux_at_top, level = run_synthetic(name, get_lvl_flux=True)
    out = {f'{name}|flux_at_top': flux_at_top}
    out.update({f'{name}|{key}': arr for key, arr in zip(LEVEL_KEYS, level)})
    return out


E2E_CASES = ['spectrum', 'spectrum_level_fluxes', 'spectrum_patchy_clouds']


def run_spectrum(name):
    """spectrum(calculation='thermal') for a brown dwarf on the mini opacity db."""
    import astropy.units as u
    import pandas as pd
    import picaso.justdoit as jdi
    opa = jdi.opannection(filename_db=str(MINI_OPACITY_DB))
    case = jdi.inputs(calculation='browndwarf')
    case.gravity(gravity=100, gravity_unit=u.Unit('m/(s**2)'))
    pt = pd.read_csv(jdi.HJ_pt(), sep=r'\s+')
    df = pd.DataFrame({'pressure': pt['pressure'], 'temperature': pt['temperature'],
                       'H2': 0.85, 'He': 0.15, 'H2O': 3e-4, 'CO2': 1e-6})
    case.atmosphere(df=df)
    case.approx(get_lvl_flux=(name == 'spectrum_level_fluxes'), integrated_energy=(name == 'spectrum_level_fluxes'))
    if name == 'spectrum_patchy_clouds':
        case.clouds(g0=[0.5], w0=[0.8], opd=[0.5], p=[-1.0], dp=[1.5],
                    do_holes=True, fhole=0.3, fthin_cld=0.2)
    out = case.spectrum(opa, calculation='thermal', full_output=(name == 'spectrum_level_fluxes'))
    result = {f'{name}|thermal': np.asarray(out['thermal'])}
    if name == 'spectrum_level_fluxes':
        level = out['full_output']['level']['thermal_fluxes']
        # every 5th level and 20th wavenumber keeps the snapshot small
        result.update({f'{name}|{key}': np.asarray(level[key])[::5, ::20] for key in LEVEL_KEYS})
    return result


@pytest.fixture(scope='module')
def golden():
    if not GOLDEN.exists():
        pytest.fail(f'missing snapshot {GOLDEN}; run python tests/test_thermal_regression.py')
    with np.load(GOLDEN) as data:
        return {key: data[key] for key in data.files}


def assert_matches(actual, expected, key):
    actual = np.asarray(actual)
    assert actual.shape == expected.shape, f'{key}: shape {actual.shape} != {expected.shape}'
    scale = np.max(np.abs(expected)) if expected.size else 0.0
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=RTOL * scale, err_msg=key)


@pytest.mark.parametrize('name', SYNTHETIC_CASES)
def test_get_thermal_1d_matches_snapshot(golden, name):
    for key, actual in synthetic_outputs(name).items():
        assert_matches(actual, golden[key], key)


@pytest.mark.parametrize('name', SYNTHETIC_CASES)
def test_top_only_path_matches_snapshot(golden, name):
    """get_lvl_flux=False skips the level fluxes but must give the same flux at the top."""
    flux_at_top, level = run_synthetic(name, get_lvl_flux=False)
    assert_matches(flux_at_top, golden[f'{name}|flux_at_top'], f'{name}|flux_at_top')
    assert all(arr.size == 0 for arr in level)


@pytest.mark.parametrize('name', ['gradient_scattering', 'climate_integrated_energy'])
def test_level_fluxes_only_when_requested(name):
    """Left out, get_lvl_flux defaults to False whichever blackbody is used."""
    _, level = run_synthetic(name)
    assert all(arr.size == 0 for arr in level)


@pytest.mark.parametrize('get_lvl_flux', [False, True])
@pytest.mark.parametrize('name, surf_reflect', [('climate_integrated_energy', 0), ('hard_surface', 0.3)])
def test_scalar_surface_reflectivity(name, surf_reflect, get_lvl_flux):
    """climate.py passes surf_reflect as a number rather than an array; both must give the same answer."""
    c = synthetic_case(name)
    nlevel, nwno = len(c['tlevel']), len(c['wno'])
    numg, numt = c['ubar1'].shape
    args = (nlevel, c['wno'], nwno, numg, numt, c['tlevel'], c['dtau'], c['w0'], c['cosb'], c['plevel'],
            c['ubar1'])
    scalar = get_thermal_1d(*args, surf_reflect, c['hard_surface'], c['dwno'], c['integrated_energy'],
                            get_lvl_flux=get_lvl_flux)
    array = get_thermal_1d(*args, np.full(nwno, float(surf_reflect)), c['hard_surface'], c['dwno'],
                           c['integrated_energy'], get_lvl_flux=get_lvl_flux)
    np.testing.assert_array_equal(scalar[0], array[0])
    for s, a in zip(scalar[1], array[1]):
        np.testing.assert_array_equal(s, a)


def test_level_fluxes_do_not_change_the_spectrum():
    """Asking spectrum() for level fluxes must leave the thermal spectrum exactly as it was."""
    without = run_spectrum('spectrum')['spectrum|thermal']
    with_level_fluxes = run_spectrum('spectrum_level_fluxes')['spectrum_level_fluxes|thermal']
    np.testing.assert_array_equal(with_level_fluxes, without)


@pytest.mark.parametrize('name', E2E_CASES)
def test_thermal_spectrum_matches_snapshot(golden, name):
    for key, actual in run_spectrum(name).items():
        assert_matches(actual, golden[key], key)


def regenerate():
    snapshot = {}
    for name in SYNTHETIC_CASES:
        snapshot.update(synthetic_outputs(name))
    for name in E2E_CASES:
        snapshot.update(run_spectrum(name))
    np.savez_compressed(GOLDEN, **snapshot)
    print(f'wrote {len(snapshot)} arrays to {GOLDEN} ({GOLDEN.stat().st_size / 1e3:.0f} kB)')


if __name__ == '__main__':
    regenerate()
