"""
Tests for fitting with picaso xarray grids: Parameterize loads a grid of models and interpolates
temperature and chemistry from it, and the driver uses that for xarray_grid profiles and priors.

The grid here is synthetic and tiny (2 teff x 2 gravity) and built so that temperature depends
only on teff, H2O only on teff, and CH4 only on gravity, which makes the interpolated answers
easy to predict.
"""
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from picaso.driver import hypercube, prior_finder, setup_spectrum_class
from picaso.parameterizations import Parameterize

TEFFS = [1000.0, 1500.0]
GRAVITIES = [1e4, 1e5]
PRESSURES = np.logspace(-4, 2, 5)
WAVELENGTHS = np.linspace(1.0, 2.0, 10)


def _grid_temperature(teff):
    return np.linspace(teff - 100, teff + 100, len(PRESSURES))


@pytest.fixture(scope='module')
def grid_dir(tmp_path_factory):
    """Writes one picaso-style xarray file per grid point."""
    grid_dir = tmp_path_factory.mktemp('xarray_grid')
    for teff in TEFFS:
        for gravity in GRAVITIES:
            ds = xr.Dataset(
                data_vars={
                    'transit_depth': (['wavelength'], np.linspace(0.01 * teff / 1000, 0.02 * gravity / 1e4,
                                                                  len(WAVELENGTHS))),
                    'temperature': (['pressure'], _grid_temperature(teff)),
                    'H2O': (['pressure'], np.full_like(PRESSURES, 1e-4 * teff / 1000)),
                    'CH4': (['pressure'], np.full_like(PRESSURES, 1e-5 * gravity / 1e4)),
                },
                coords={'wavelength': WAVELENGTHS, 'pressure': PRESSURES},
            )
            ds.attrs['planet_params'] = json.dumps({
                'teff': teff,
                'gravity': gravity,
                'rp': {'value': 1.0, 'unit': 'Rjup'},
                'mp': {'value': 1.0, 'unit': 'Mjup'},
            })
            ds.attrs['stellar_params'] = json.dumps({
                'rs': {'value': 1.0, 'unit': 'Rsun'},
                'steff': 5700.0,
                'feh': 0.0,
                'logg': 4.5,
                'database': 'ck04models',
            })
            ds.to_netcdf(grid_dir / f'model_t{teff}_g{gravity}.nc')
    return str(grid_dir)


@pytest.fixture(scope='module')
def param_tools(grid_dir):
    return Parameterize(model_dir=grid_dir, to_fit='transit_depth')


def test_grid_loads_with_its_parameters(param_tools):
    assert param_tools.grid_name == 'init'
    unique = param_tools.interp_params['init']['grid_parameters_unique']
    assert set(unique) == {'teff', 'gravity'}
    np.testing.assert_allclose(sorted(unique['teff']), TEFFS)
    np.testing.assert_allclose(sorted(unique['gravity']), GRAVITIES)


def test_pt_xarray_grid_interpolates_between_grid_points(param_tools):
    pt_df = param_tools.pt_xarray_grid(teff=1250.0, gravity=5.5e4)
    assert isinstance(pt_df, pd.DataFrame)
    np.testing.assert_allclose(pt_df['pressure'], PRESSURES)
    # halfway between the two teffs, and the grid temperature does not depend on gravity
    np.testing.assert_allclose(pt_df['temperature'], _grid_temperature(1250.0), rtol=1e-6)


def test_pt_xarray_grid_returns_grid_point_exactly(param_tools):
    pt_df = param_tools.pt_xarray_grid(teff=1500.0, gravity=1e4)
    np.testing.assert_allclose(pt_df['temperature'], _grid_temperature(1500.0), rtol=1e-6)


def test_chem_xarray_grid_interpolates_each_molecule(param_tools):
    from picaso.justdoit import inputs
    picaso_class = inputs(calculation='browndwarf', climate=False)
    picaso_class.add_pt(P=PRESSURES)
    picaso_class.atmosphere(df=param_tools.pt_xarray_grid(teff=1250.0, gravity=5.5e4))
    param_tools.add_class(picaso_class)

    chem_df = param_tools.chem_xarray_grid(molecules=['H2O', 'CH4'], teff=1250.0, gravity=5.5e4)
    assert {'pressure', 'temperature', 'H2O', 'CH4'} <= set(chem_df.columns)
    assert len(chem_df) == len(PRESSURES)
    # H2O depends only on teff and CH4 only on gravity, so each lies between its grid values
    assert np.all((chem_df['H2O'] > 1e-4) & (chem_df['H2O'] < 1.5e-4))
    assert np.all((chem_df['CH4'] > 1e-5) & (chem_df['CH4'] < 1e-4))


def test_driver_builds_atmosphere_from_grid(param_tools):
    grid_kwargs = {'teff': 1250.0, 'gravity': 5.5e4}
    config = {
        'irradiated': False,
        'calc_type': 'spectrum',
        'observation_type': 'transit_depth',
        'geometry': {'phase': {'value': 0, 'unit': 'radian'}},
        'object': {'gravity': {'value': 1e5, 'unit': 'cm/s**2'},
                   'radius': {'value': 1.2, 'unit': 'Rjup'}},
        'temperature': {'profile': 'xarray_grid',
                        'xarray_grid': {'grid_kwargs': grid_kwargs}},
        'chemistry': {'method': 'xarray_grid',
                      'xarray_grid': {'grid_kwargs': dict(grid_kwargs, molecules=['H2O', 'CH4'])}},
    }
    A = setup_spectrum_class(config, opacity=object(), param_tools=param_tools, stage='chemistry')
    profile = A.inputs['atmosphere']['profile']
    assert {'temperature', 'H2O', 'CH4'} <= set(profile.columns)
    np.testing.assert_allclose(profile['temperature'], _grid_temperature(1250.0), rtol=1e-6)


def test_xarray_grid_prior_spans_the_grid(param_tools):
    fitpars = prior_finder({
        'temperature': {'xarray_grid': {'grid_kwargs': {'teff': {'prior': 'xarray_grid', 'log': False}}}},
        'chemistry': {'xarray_grid': {'grid_kwargs': {'gravity': {'prior': 'xarray_grid', 'log': False}}}},
    })
    assert list(fitpars) == ['temperature.xarray_grid.grid_kwargs.teff',
                             'chemistry.xarray_grid.grid_kwargs.gravity']

    np.testing.assert_allclose(hypercube(np.array([0.0, 0.0]), fitpars, param_tools=param_tools),
                               [TEFFS[0], GRAVITIES[0]])
    np.testing.assert_allclose(hypercube(np.array([1.0, 1.0]), fitpars, param_tools=param_tools),
                               [TEFFS[1], GRAVITIES[1]])
    np.testing.assert_allclose(hypercube(np.array([0.5, 0.5]), fitpars, param_tools=param_tools),
                               [np.mean(TEFFS), np.mean(GRAVITIES)])
