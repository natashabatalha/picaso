"""
The stellar spectra that inputs.star(database=...) pulls from the stsynphot grids in PYSYN_CDBS
should carry the energy of the requested star: integrating F_lambda over wavelength and inverting
sigma*Teff^4 should give back the requested Teff to within 1%.

This catches a corrupted or mismatched grid download. Some copies of the PHOENIX grid have been
seen missing flux at the grid nodes themselves (Teff 2-10% low, e.g. the Sun at ~5310 K), which
makes every irradiated climate model using them too cold. The PHOENIX files STScI serves at
archive.stsci.edu (created 2021-5-10, MAPKEY phoenixBTS11_15) pass at 0.1-0.7% low.

The grids are too big for git, so each one is skipped when it is not in PYSYN_CDBS.
"""
import os
import warnings

import astropy.units as u
import numpy as np
import pytest
from synphot import units

from picaso import justdoit as jdi

SIGMA = 5.670374419e-5  # erg cm-2 s-1 K-4
TOLERANCE = 0.01  # fractional Teff

CASES = [  # (Teff, [M/H], logg)
    (5300, 0.0, 4.5),        # grid nodes, no interpolation
    (5400, 0.0, 4.5),
    (5300, -0.5, 4.5),
    (5400, -0.5, 4.5),
    (5326.6, -0.03, 4.38933),  # WASP-39, used in the climate tutorials
    (5778, 0.0, 4.44),       # Sun
    (6500, 0.0, 4.5),
    (4500, 0.0, 4.5),
    (3500, 0.0, 5.0),
]


def grid_available(database):
    return os.path.isdir(os.path.join(os.environ.get('PYSYN_CDBS', ''), 'grid', database))


def implied_teff(database, temp, metal, logg):
    """Teff from integrating the spectrum picaso gets for this star (same call as inputs.star)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sp = jdi.sts.grid_to_spec(database, temp, metal, logg)
    wave = sp.waveset.to(u.AA).value
    flam = sp(sp.waveset, flux_unit=units.FLAM).value  # erg cm-2 s-1 A-1
    return (np.trapezoid(flam, wave) / SIGMA) ** 0.25


@pytest.mark.parametrize('database', [
    pytest.param(db, marks=pytest.mark.skipif(not grid_available(db), reason=f'{db} grid not in PYSYN_CDBS'))
    for db in ('phoenix', 'ck04models')
])
@pytest.mark.parametrize('temp,metal,logg', CASES)
def test_stellar_spectrum_reproduces_teff(database, temp, metal, logg):
    teff = implied_teff(database, temp, metal, logg)
    assert abs(teff - temp) / temp < TOLERANCE, (
        f'{database} spectrum for Teff={temp} K, [M/H]={metal}, logg={logg} integrates to {teff:.1f} K. '
        f'Your stellar grid in PYSYN_CDBS={os.environ.get("PYSYN_CDBS")} is likely a bad copy; '
        f're-download it from STScI.')
