import os
import json
import tempfile
import numpy as np
import pandas as pd
import xarray as xr
import pytest
from unittest.mock import MagicMock

from picaso.parameterizations import Parameterize
from picaso.driver import (
    setup_spectrum_class,
    hypercube,
    prior_finder,
    PT_handler
)

def test_xarray_grid_workflow():
    # 1. Create a temporary xarray grid
    with tempfile.TemporaryDirectory() as temp_dir:
        grid_location = temp_dir
        grid_name = "test_grid"
        
        teffs = [1000.0, 1500.0]
        gravities = [1e4, 1e5]
        wavelengths = np.linspace(1.0, 2.0, 10)
        pressures = np.logspace(-4, 2, 5)

        for t in teffs:
            for g in gravities:
                transit_depth = np.linspace(0.01 * (t / 1000.0), 0.02 * (g / 1e4), len(wavelengths))
                temperature = np.linspace(t - 100, t + 100, len(pressures))
                h2o = np.full_like(pressures, 1e-4 * (t / 1000.0))
                ch4 = np.full_like(pressures, 1e-5 * (g / 1e4))
                
                ds = xr.Dataset(
                    data_vars={
                        'transit_depth': (['wavelength'], transit_depth),
                        'temperature': (['pressure'], temperature),
                        'H2O': (['pressure'], h2o),
                        'CH4': (['pressure'], ch4)
                    },
                    coords={
                        'wavelength': wavelengths,
                        'pressure': pressures
                    }
                )
                ds.attrs['planet_params'] = json.dumps({
                    'teff': t,
                    'gravity': g,
                    'rp': {'value': 1.0, 'unit': 'Rjup'},
                    'mp': {'value': 1.0, 'unit': 'Mjup'}
                })
                ds.attrs['stellar_params'] = json.dumps({
                    'rs': {'value': 1.0, 'unit': 'Rsun'},
                    'steff': 5700.0,
                    'feh': 0.0,
                    'logg': 4.5,
                    'database': 'ck04models'
                })
                filename = os.path.join(grid_location, f"model_t{t}_g{g}.nc")
                ds.to_netcdf(filename)

        # 2. Test Parameterize initialization and load_grid
        param_tools = Parameterize()
        param_tools.load_grid(grid_name, grid_location, to_fit='transit_depth', save_chem=True)
        
        assert 'test_grid' in param_tools.interp_params
        assert param_tools.grid_name == grid_name
        assert 'teff' in param_tools.interp_params[grid_name]['grid_parameters_unique']
        assert 'gravity' in param_tools.interp_params[grid_name]['grid_parameters_unique']

        # 3. Test pt_xarray_grid interpolation
        # Let's interpolate at teff=1250, gravity=5.5e4
        pt_df = param_tools.pt_xarray_grid(
            grid_name=grid_name,
            grid_location=grid_location,
            teff=1250.0,
            gravity=5.5e4
        )
        assert isinstance(pt_df, pd.DataFrame)
        assert 'pressure' in pt_df.columns
        assert 'temperature' in pt_df.columns
        assert len(pt_df) == len(pressures)
        # Check that temperature values are in a reasonable range
        assert np.all(pt_df['temperature'] > 800)
        assert np.all(pt_df['temperature'] < 1700)

        # 4. Test chem_xarray_grid interpolation
        # First we need to simulate setup_spectrum_class or add_class setting pressure_level and temperature_level
        # Let's call param_tools.add_class using a mock or dummy class
        from picaso.justdoit import inputs
        picaso_class = inputs(calculation='browndwarf', climate=False)
        picaso_class.add_pt(P=pressures)
        picaso_class.atmosphere(df=pt_df)
        param_tools.add_class(picaso_class)

        chem_df = param_tools.chem_xarray_grid(
            grid_name=grid_name,
            grid_location=grid_location,
            molecules=['H2O', 'CH4'],
            teff=1250.0,
            gravity=5.5e4
        )
        assert isinstance(chem_df, pd.DataFrame)
        assert 'H2O' in chem_df.columns
        assert 'CH4' in chem_df.columns
        assert len(chem_df) == len(pressures)

        # 5. Test setup_spectrum_class and PT_handler integrations
        config = {
            'irradiated': False,
            'calc_type': 'spectrum',
            'observation_type': 'transit_depth',
            'geometry': {
                'phase': {'value': 0, 'unit': 'radian'}
            },
            'object': {
                'gravity': {'value': 1e5, 'unit': 'cm/s**2'},
                'radius': {'value': 1.2, 'unit': 'Rjup'},
                'mass': {'value': 1.2, 'unit': 'Mjup'}
            },
            'temperature': {
                'profile': 'xarray_grid',
                'xarray_grid': {
                    'grid_name': grid_name,
                    'grid_location': grid_location,
                    'to_fit': 'transit_depth',
                    'teff': 1250.0,
                    'gravity': 5.5e4
                }
            },
            'chemistry': {
                'method': 'xarray_grid',
                'xarray_grid': {
                    'grid_name': grid_name,
                    'grid_location': grid_location,
                    'to_fit': 'transit_depth',
                    'molecules': ['H2O', 'CH4'],
                    'teff': 1250.0,
                    'gravity': 5.5e4
                }
            }
        }
        
        A = setup_spectrum_class(config, opacity=MagicMock(), param_tools=param_tools, stage='chemistry')
        assert A is not None
        profile = A.inputs['atmosphere']['profile']
        assert 'H2O' in profile.columns
        assert 'CH4' in profile.columns
        assert 'temperature' in profile.columns

        # 6. Test prior calculations with hypercube
        retrieval_config = {
            'retrieval': {
                'temperature': {
                    'xarray_grid': {
                        'teff': {
                            'prior': 'xarray_grid',
                            'log': False
                        }
                    }
                },
                'chemistry': {
                    'xarray_grid': {
                        'gravity': {
                            'prior': 'xarray_grid',
                            'log': False
                        }
                    }
                }
            }
        }
        fitpars = prior_finder(retrieval_config['retrieval'])
        assert 'temperature.xarray_grid.teff' in fitpars
        assert 'chemistry.xarray_grid.gravity' in fitpars

        # Map u = [0.0, 1.0] (the bounds of uniform/hypercube)
        x_min = hypercube(np.array([0.0, 0.0]), fitpars, param_tools=param_tools)
        x_max = hypercube(np.array([1.0, 1.0]), fitpars, param_tools=param_tools)

        # Expected min and max for teff and gravity from unique grid parameter values
        assert np.isclose(x_min[0], 1000.0) # min teff
        assert np.isclose(x_min[1], 1e4)    # min gravity
        assert np.isclose(x_max[0], 1500.0) # max teff
        assert np.isclose(x_max[1], 1e5)    # max gravity
