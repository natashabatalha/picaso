"""
Regression test that pins the thermal spectrum of a realistic brown dwarf on the full resampled
opacity db: the 1270 K, g = 200 m/s2 profile (jdi.brown_dwarf_pt()), 1-14 micron, clear and with its
fsed = 1 cloud (jdi.brown_dwarf_cld()). test_thermal_regression.py does the same on the mini db in git;
this one covers every molecule and the line cores where the top layer is optically thick.

The snapshot is reference/base_cases/testing/thermal_brown_dwarf_full_db.npz. It keeps every 10th
wavenumber bin, to stay small, plus the bolometric flux of the full spectrum, which changes if any bin
does. It was made with opacities_0.3_15_R15000.db; with another db the wavenumber grid differs and the
test says so. Regenerate it only when a change is meant to change thermal spectra:

    python tests/test_thermal_regression_full_db.py

The second test documents issue #443: v5 reports the upward flux at the top of the atmosphere, where
v4 reported it at the midpoint of the top layer. The midpoint value is still computed as a level flux
(flux_plus_mdpt), so the size of the change can be checked directly.
"""
import os
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('picaso_refdata', str(REPO_ROOT / 'reference'))

import astropy.units as u  # noqa: E402
import picaso.justdoit as jdi  # noqa: E402

GOLDEN = REPO_ROOT / 'reference' / 'base_cases' / 'testing' / 'thermal_brown_dwarf_full_db.npz'
WAVE_RANGE = [1, 14]  # micron
EVERY = 10  # keep every 10th wavenumber bin in the snapshot
RTOL = 1e-10
CASES = ['clear', 'cloudy']

pytestmark = pytest.mark.needs_full_data


def brown_dwarf(cloudy, **approx):
    case = jdi.inputs(calculation='browndwarf')
    case.phase_angle(0)
    case.gravity(gravity=200, gravity_unit=u.Unit('m/s**2'))
    case.atmosphere(filename=jdi.brown_dwarf_pt(), sep=r'\s+')
    if cloudy:
        case.clouds(filename=jdi.brown_dwarf_cld(), sep=r'\s+')
    if approx:
        case.approx(**approx)
    return case


def bolometric(wavenumber, flux):
    return np.trapezoid(flux, wavenumber) if hasattr(np, 'trapezoid') else np.trapz(flux, wavenumber)


def snapshot_arrays(opacity, name):
    out = brown_dwarf(name == 'cloudy').spectrum(opacity, calculation='thermal')
    wavenumber, thermal = np.asarray(out['wavenumber']), np.asarray(out['thermal'])
    return {f'{name}|wavenumber': wavenumber[::EVERY],
            f'{name}|thermal': thermal[::EVERY],
            f'{name}|bolometric': np.array(bolometric(wavenumber, thermal))}


@pytest.fixture(scope='module')
def opacity():
    return jdi.opannection(wave_range=WAVE_RANGE)


@pytest.fixture(scope='module')
def golden():
    if not GOLDEN.exists():
        pytest.fail(f'missing snapshot {GOLDEN}; run python tests/test_thermal_regression_full_db.py')
    with np.load(GOLDEN) as data:
        return {key: data[key] for key in data.files}


@pytest.mark.parametrize('name', CASES)
def test_brown_dwarf_thermal_spectrum_matches_snapshot(opacity, golden, name):
    actual = snapshot_arrays(opacity, name)
    if actual[f'{name}|wavenumber'].shape != golden[f'{name}|wavenumber'].shape:
        pytest.fail(f"the snapshot was made with {golden['db'].item()}, but this run uses "
                    f'{os.path.basename(opacity.db_filename)}, which has a different wavenumber grid')
    for key, value in actual.items():
        np.testing.assert_allclose(value, golden[key], rtol=RTOL, err_msg=key)


@pytest.mark.parametrize('name', CASES)
def test_top_of_atmosphere_vs_top_layer_midpoint(opacity, name):
    """
    Issue #443: reporting the top level instead of the top-layer midpoint only darkens the cores of the
    strongest lines, where the top layer is optically thick. Nothing gets brighter, the bolometric flux
    changes by ~1e-5, and about 0.2% of bins change by more than 1% (at most ~4%).
    """
    out = brown_dwarf(name == 'cloudy', get_lvl_flux=True, integrated_energy=False).spectrum(
        opacity, calculation='thermal', full_output=True)
    wavenumber, top = np.asarray(out['wavenumber']), np.asarray(out['thermal'])
    midpoint = np.asarray(out['full_output']['level']['thermal_fluxes']['flux_plus_mdpt'])[0]
    change = top / midpoint - 1

    assert np.all(change <= 1e-12)
    assert abs(bolometric(wavenumber, top) / bolometric(wavenumber, midpoint) - 1) < 1e-4
    assert np.mean(np.abs(change) > 0.01) < 0.01
    assert change.min() > -0.1


def regenerate():
    opacity = jdi.opannection(wave_range=WAVE_RANGE)
    snapshot = {'db': np.array(os.path.basename(opacity.db_filename))}
    for name in CASES:
        snapshot.update(snapshot_arrays(opacity, name))
    np.savez_compressed(GOLDEN, **snapshot)
    print(f'wrote {len(snapshot)} arrays to {GOLDEN} ({GOLDEN.stat().st_size / 1e3:.0f} kB)')


if __name__ == '__main__':
    regenerate()
