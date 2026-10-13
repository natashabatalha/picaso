"""
Interactive corner plot of the driver UI's Retrieval Analysis page (driver_ui/core/analysis.py), on the synthetic
retrieval example in reference/base_cases/retrieval_example/. These run in CI because all of their data is in git.
"""
import copy

import numpy as np
import pytest

from conftest import REFDATA


@pytest.fixture(scope='module')
def example_info(retrieval_example_config):
    import picaso.driver as go
    from picaso import retrieval as ret
    config = retrieval_example_config
    return ret.read_retrievals(config['InputOutput']['retrieval_output'], go.prior_finder(config['retrieval']))


def test_sigma_field_of_a_gaussian():
    """For a 2D standard normal the n sigma region is the circle of radius n, so the field is the radius."""
    from picaso.driver_ui.core.analysis import _sigma_field
    samples = np.random.RandomState(1).normal(size=(2, 4000))
    xx, yy = np.meshgrid(np.linspace(-5, 5, 101), np.linspace(-5, 5, 101))
    sigma = _sigma_field(samples[0], samples[1], xx, yy)
    for radius in (1, 2):
        assert sigma[50, 50 + 10 * radius] == pytest.approx(radius, abs=0.2)
    assert sigma[50, 50] < 0.3


def test_figure_lines_start_at_the_medians(example_info, retrieval_example_config):
    from picaso.driver_ui.core import analysis
    info = example_info
    point = analysis.median_point(info)
    out = analysis.evaluate_point(retrieval_example_config, info, point)
    fig = analysis.interactive_corner_figure(info, analysis.default_corner_settings(info), point, out, out,
                                             retrieval_example_config['observation_type'])
    params = info['param_names']
    n = len(params)
    meta = fig.layout.meta

    # medians are of log10 of the log sampled parameters, as plot_pair draws them
    samples = np.array(info['samples_equal'], dtype=float)
    for i, param in enumerate(params):
        expected = np.median(np.log10(samples[:, i]) if param in info['log_params'] else samples[:, i])
        assert point[param] == pytest.approx(expected)
        assert fig.layout.shapes[i].x0 == fig.layout.shapes[i].x1 == pytest.approx(expected)
    assert meta['medians'] == pytest.approx([point[param] for param in params])

    # one crosshair per 2D panel, at (median of column parameter, median of row parameter)
    assert len(meta['crosshairs']) == n * (n - 1) // 2
    for trace, i, j in meta['crosshairs']:
        assert fig.data[trace].x[0] == pytest.approx(point[params[j]])
        assert fig.data[trace].y[0] == pytest.approx(point[params[i]])

    wavelength, model = analysis.model_spectrum(out)
    assert np.allclose(fig.data[meta['model_trace']].y, model)
    assert 'Chi-sq' in fig.layout.annotations[meta['title_annotation']].text


def test_evaluate_point_matches_check_model_samples(example_info, retrieval_example_config):
    """evaluate_point reuses the cached opacities, but must give the same model as the driver."""
    import picaso.driver as go
    from picaso.driver_ui.core import analysis
    info = example_info
    point = analysis.median_point(info)
    point[info['param_names'][0]] += 50  # away from the median, so the point is passed through
    theta = [10**point[p] if p in info['log_params'] else point[p] for p in info['param_names']]
    expected = go.check_model_samples(copy.deepcopy(retrieval_example_config), samples=np.atleast_2d(theta),
                                      full_likelihood=True)
    out = analysis.evaluate_point(retrieval_example_config, info, point)
    assert np.allclose(out['ymodel'][0], expected['ymodel'][0])
    assert out['chi_sq_per_pt'][0] == pytest.approx(expected['chi_sq_per_pt'][0])


def test_corner_point_route():
    from picaso.driver_ui.app import create_app
    app = create_app(str(REFDATA))
    client = app.test_client()
    client.get('/analysis/')
    assert client.post('/analysis/example').status_code == 200

    response = client.post('/analysis/corner/point', json={'point': {}})
    assert response.status_code == 400

    # the example's truths, with T moved 100 K away: a much worse fit than at the median
    point = {'temperature.isothermal.T': 1000.0, 'chemistry.free.H2O.value': -3.0,
             'chemistry.free.CO2.value': -4.0, 'object.radius.value': 1.27}
    response = client.post('/analysis/corner/point', json={'point': point})
    assert response.status_code == 200
    body = response.get_json()
    assert body['point'] == point
    assert len(body['wavelength']) == len(body['model']) == 160
    assert np.all(np.diff(body['wavelength']) > 0)
    assert float(body['title'].split('= ')[1].rstrip(')')) > 10

    # the selected point is kept, so a re-rendered page shows it
    html = client.post('/analysis/card/corner', data={}).get_data(as_text=True)
    assert 'value="1000"' in html
