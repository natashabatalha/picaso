"""
Component tests on the mini H2O/CO2 opacity db and the synthetic retrieval example in
reference/base_cases/retrieval_example/. These run in CI because all of their data is in git.
"""
import copy

import numpy as np
import pandas as pd
import pytest

from conftest import RETRIEVAL_EXAMPLE_DIR

# the parameters transit_example.csv was generated at, see reference/scripts/make_retrieval_example.py
TRUTH = {'temperature.isothermal.T': 900.0,
         'chemistry.free.H2O.value': 1e-3,
         'chemistry.free.CO2.value': 1e-4,
         'object.radius.value': 1.27}


def test_mini_opacity_db_contents(mini_opacity):
    assert sorted(mini_opacity.molecules) == ['CO2', 'H2O']
    wave = 1e4 / mini_opacity.wno
    assert wave.min() == pytest.approx(0.95, abs=0.01)
    assert wave.max() == pytest.approx(5.3, abs=0.01)


def _reduced_chi_sq(config, truth):
    import picaso.driver as go
    fitpars = go.prior_finder(config['retrieval'])
    samples = np.atleast_2d([truth[key] for key in fitpars])
    model = go.check_model_samples(config, samples=samples, full_likelihood=True)
    # check_model_samples is ordered by wavenumber, the data file by wavelength
    order = np.argsort(1e4 / np.asarray(model['xdata']))
    depth = np.asarray(model['ymodel'][0])[order]
    data = pd.read_csv(RETRIEVAL_EXAMPLE_DIR / 'transit_example.csv')
    return np.mean(((data['transit_depth'] - depth) / data['error'])**2)


def test_model_at_truth_fits_example_data(retrieval_example_config):
    """
    The example data is the model at TRUTH plus 30 ppm gaussian noise, so the model at TRUTH
    should give a reduced chi-squared near 1 (std ~0.11 for 160 points).
    """
    chi_sq = _reduced_chi_sq(copy.deepcopy(retrieval_example_config), TRUTH)
    assert 0.5 < chi_sq < 1.5


def test_model_away_from_truth_does_not_fit(retrieval_example_config):
    """Guards the test above: a 1% larger radius must be a clearly worse fit."""
    truth = dict(TRUTH, **{'object.radius.value': TRUTH['object.radius.value'] * 1.01})
    chi_sq = _reduced_chi_sq(copy.deepcopy(retrieval_example_config), truth)
    assert chi_sq > 10
