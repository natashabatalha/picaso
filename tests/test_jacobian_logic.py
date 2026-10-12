"""
Tests for information_content.jacobian.

The first group uses a stand-in for the picaso inputs class whose "spectrum" is a simple function
of its inputs, so every finite difference has an exact expected value. The last two tests take
the Jacobian of a real transmission spectrum on the mini opacity db.
"""
import copy

import numpy as np
import pandas as pd
import pytest

from picaso import information_content as ic


class StubPicaso:
    """
    Has the parts of picaso.justdoit.inputs that jacobian uses: an inputs dict, a spectrum method,
    and methods that a def.<function>.<argument> parameter reruns.
    """
    def __init__(self):
        self.inputs = {
            'planet': {'a': 3.0, 'b': 1e-3},
            'atmosphere': {'profile': pd.DataFrame({'H2O': [1e-4, 1e-3, 1e-2]})},
        }
        self.cloud(opd=2.0, scale=10.0)

    def cloud(self, opd, scale):
        self.inputs['clouds'] = opd * scale

    def spectrum(self, opacityclass, calculation=None):
        planet = self.inputs['planet']
        spec = np.array([planet['a'], planet['a']**2, planet['b'],
                         np.mean(self.inputs['atmosphere']['profile']['H2O']),
                         self.inputs['clouds']])
        return {'thermal': spec}


def stub_jacobian(params, **kwargs):
    return ic.jacobian(picaso_class=StubPicaso(), params=params, opacityclass=object(),
                       calculation='thermal', **kwargs)


@pytest.mark.parametrize('method, expected_da2', [
    ('forward', 2 * 3.0 + 0.03),   # (a+h)^2 - a^2 over h = 2a + h, with h = 1% of a
    ('backward', 2 * 3.0 - 0.03),
    ('center', 2 * 3.0),           # exact for a quadratic
])
def test_linear_scalar_derivative(method, expected_da2):
    jac = stub_jacobian(['planet.a'], method=method, d_param=0.01)
    assert jac.shape == (5, 1)
    np.testing.assert_allclose(jac[:, 0], [1.0, expected_da2, 0, 0, 0], atol=1e-9)


def test_log_scalar_derivative():
    """With is_log the derivative is with respect to log10 of the parameter."""
    jac = stub_jacobian(['planet.b'], d_param=0.01, is_log=True)
    h = 0.01 * 3  # 1% of |log10(1e-3)|
    expected = (10**(-3 + h) - 1e-3) / h
    np.testing.assert_allclose(jac[:, 0], [0, 0, expected, 0, 0], atol=1e-12)


def test_log_array_derivative():
    """A DataFrame column (a mixing ratio profile) is shifted as a whole in log space."""
    jac = stub_jacobian(['atmosphere.profile.H2O'], d_param=0.01, is_log=True)
    h2o = np.array([1e-4, 1e-3, 1e-2])
    h = 0.01 * 3  # 1% of |mean(log10(h2o))|
    expected = (np.mean(10**(np.log10(h2o) + h)) - np.mean(h2o)) / h
    np.testing.assert_allclose(jac[3, 0], expected, rtol=1e-9)


def test_multiple_params_give_one_column_each():
    jac = stub_jacobian(['planet.a', 'planet.b'], d_param=[0.01, 0.02], is_log=[False, True])
    assert jac.shape == (5, 2)
    assert jac[0, 0] == pytest.approx(1.0)
    assert jac[0, 1] == 0 and jac[2, 0] == 0


def test_shortcut_name_resolves_to_full_path():
    """A bare key is found by searching the inputs dict."""
    np.testing.assert_allclose(stub_jacobian(['a'], d_param=0.01), stub_jacobian(['planet.a'], d_param=0.01))


def test_function_argument_derivative():
    """def.cloud.opd reruns picaso_class.cloud with a perturbed opd and the other arguments fixed."""
    def_kwargs = {'def.cloud.opd': {'opd': 2.0, 'scale': 10.0}}
    jac = stub_jacobian(['def.cloud.opd'], d_param=0.01, def_kwargs=def_kwargs)
    np.testing.assert_allclose(jac[:, 0], [0, 0, 0, 0, 10.0], rtol=1e-9)


def test_does_not_modify_the_input_class_parameters():
    picaso_class = StubPicaso()
    before = copy.deepcopy(picaso_class.inputs['planet'])
    ic.jacobian(picaso_class=picaso_class, params=['planet.a'], opacityclass=object(), calculation='thermal')
    assert picaso_class.inputs['planet'] == before


@pytest.mark.parametrize('kwargs, message', [
    (dict(params=None), 'Must provide a list of parameters'),
    (dict(params=['planet.a'], d_param=[0.01, 0.02]), 'd_param must be a single value'),
    (dict(params=['planet.a'], is_log=[True, False]), 'is_log must be a single boolean'),
    (dict(params=['planet.nope']), 'not found in configuration'),
    (dict(params=['planet.a'], method='sideways'), 'Unknown derivative method'),
])
def test_bad_inputs_raise(kwargs, message):
    with pytest.raises(Exception, match=message):
        ic.jacobian(picaso_class=StubPicaso(), opacityclass=object(), calculation='thermal', **kwargs)


def test_missing_opacity_or_calculation_raises():
    with pytest.raises(Exception, match='Opacity Class was not provided'):
        ic.jacobian(picaso_class=StubPicaso(), params=['planet.a'], calculation='thermal')
    with pytest.raises(Exception, match='calculation was not provided'):
        ic.jacobian(picaso_class=StubPicaso(), params=['planet.a'], opacityclass=object())


@pytest.fixture(scope='module')
def transit_case(retrieval_example_config, mini_opacity):
    """The retrieval example at its true parameters, as a picaso inputs class, with its spectrum."""
    from picaso.driver import setup_spectrum_class
    from picaso.parameterizations import Parameterize
    picaso_class = setup_spectrum_class(copy.deepcopy(retrieval_example_config), opacity=mini_opacity,
                                        param_tools=Parameterize())
    spectrum = picaso_class.spectrum(mini_opacity, calculation='transmission')
    return picaso_class, spectrum


def test_transit_depth_radius_derivative(transit_case, mini_opacity):
    """
    For a bare disk d(depth)/dR = 2 depth / R. The atmosphere's height also grows with R, so the
    true derivative should be a little larger than that everywhere.
    """
    picaso_class, spectrum = transit_case
    jac = ic.jacobian(picaso_class=picaso_class, params=['planet.radius'], method='center',
                      opacityclass=mini_opacity, calculation='transit_depth')
    disk = 2 * spectrum['transit_depth'] / picaso_class.inputs['planet']['radius']
    ratio = jac[:, 0] / disk
    assert np.all((ratio > 1.0) & (ratio < 1.25))


def test_transit_depth_molecule_derivatives_follow_their_bands(transit_case, mini_opacity):
    """Each molecule's abundance matters most in its own bands: H2O at 1.4 micron, CO2 at 4.3 micron."""
    picaso_class, spectrum = transit_case
    jac = ic.jacobian(picaso_class=picaso_class, params=['atmosphere.profile.H2O', 'atmosphere.profile.CO2'],
                      is_log=True, opacityclass=mini_opacity, calculation='transit_depth')
    wave = 1e4 / spectrum['wavenumber']
    h2o_band = (wave > 1.35) & (wave < 1.5)
    co2_band = (wave > 4.2) & (wave < 4.4)
    d_h2o, d_co2 = jac[:, 0], jac[:, 1]
    assert np.mean(d_h2o[h2o_band]) > 10 * np.abs(np.mean(d_co2[h2o_band]))
    assert np.mean(d_co2[co2_band]) > 10 * np.abs(np.mean(d_h2o[co2_band]))
