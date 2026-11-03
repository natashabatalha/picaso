import builtins

import numpy as np
import pytest

from picaso import retrieval
from picaso.driver import active_cloud_mie_condensates, run_dynesty, run_ultranest


def _block_import(monkeypatch, blocked):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == blocked or name.startswith(f"{blocked}."):
            raise ImportError(f"blocked optional dependency: {blocked}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def test_dynesty_missing_error_names_install_extra(monkeypatch):
    _block_import(monkeypatch, "dynesty")
    with pytest.raises(ImportError, match=r"picaso\[dynesty\]"):
        run_dynesty(None, None, None, None, None, None)


def test_ultranest_missing_error_names_install_extra(monkeypatch):
    _block_import(monkeypatch, "ultranest")
    with pytest.raises(ImportError, match=r"picaso\[ultranest\]"):
        run_ultranest(None, None, None, None, None, None)


def test_prediction_band_does_not_require_ultranest():
    band = retrieval.PredictionBand(np.array([1.0, 2.0]))
    band.add(np.array([2.0, 4.0]))
    band.add(np.array([4.0, 8.0]))
    np.testing.assert_allclose(band.get_line(0.5).data, [3.0, 6.0])


def test_equal_resampling_is_sampler_independent_and_repeatable():
    samples = np.array([[0.0], [1.0], [2.0]])
    weights = np.array([0.1, 0.2, 0.7])
    first = retrieval._resample_equal(samples, weights)
    second = retrieval._resample_equal(samples, weights)
    np.testing.assert_array_equal(first, second)
    assert first.shape == samples.shape


def test_inactive_mie_cloud_options_are_not_preloaded():
    config = {
        "clouds": {
            "cloud1_type": "hard_grey",
            "cloud1": {
                "hard_grey": {"opd": [1.0]},
                "flex_fsed": {"condensate": "Al2O3"},
                "brewster_mie": {"condensate": "Na2S"},
            },
        },
    }
    assert active_cloud_mie_condensates(config) == []


@pytest.mark.parametrize("cloud_type", ["flex_fsed", "brewster_mie"])
def test_active_mie_cloud_is_preloaded(cloud_type):
    config = {
        "clouds": {
            "cloud1_type": cloud_type,
            "cloud1": {cloud_type: {"condensate": "Al2O3"}},
        },
    }
    assert active_cloud_mie_condensates(config) == ["Al2O3"]
