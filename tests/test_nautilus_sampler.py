import os
import sys
import types

import pytest

from picaso.driver import SAMPLER_REGISTRY, check_resume_exists, run_nautilus


class FakeNautilusSampler:
    def __init__(self, prior, likelihood, **kwargs):
        self.prior = prior
        self.likelihood = likelihood
        self.kwargs = kwargs
        self.run_kwargs = None

    def run(self, **kwargs):
        self.run_kwargs = kwargs
        return True


def test_nautilus_is_registered():
    assert SAMPLER_REGISTRY['nautilus'] is run_nautilus


def test_run_nautilus_passes_picaso_functions_and_options(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules, 'nautilus', types.SimpleNamespace(Sampler=FakeNautilusSampler)
    )
    prior = lambda u: u
    likelihood = lambda theta: -1.0
    pool = object()
    config = {'InputOutput': {'retrieval_output': str(tmp_path)}}
    prior_config = {
        'sampler': {
            'resume': True,
            'sampler_kwargs': {'n_live': 321, 'seed': 7},
            'run_kwargs': {'n_eff': 654, 'verbose': False},
        }
    }

    sampler = run_nautilus(
        config, likelihood, prior, ndims=3, pool=pool,
        prior_config=prior_config,
    )

    assert sampler.prior is prior
    assert sampler.likelihood is likelihood
    assert sampler.kwargs == {
        'n_dim': 3,
        'pool': pool,
        'filepath': os.path.join(tmp_path, 'nautilus.hdf5'),
        'resume': True,
        'n_live': 321,
        'seed': 7,
    }
    assert sampler.run_kwargs == {'n_eff': 654, 'verbose': False}


def test_run_nautilus_enables_progress_by_default(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules, 'nautilus', types.SimpleNamespace(Sampler=FakeNautilusSampler)
    )
    config = {'InputOutput': {'retrieval_output': str(tmp_path)}}
    prior_config = {'sampler': {'sampler_kwargs': {}, 'run_kwargs': {}}}

    sampler = run_nautilus(
        config, lambda theta: 0.0, lambda u: u, 2, None, prior_config
    )

    assert sampler.run_kwargs == {'verbose': True}


def test_run_nautilus_rejects_picaso_managed_options(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules, 'nautilus', types.SimpleNamespace(Sampler=FakeNautilusSampler)
    )
    config = {'InputOutput': {'retrieval_output': str(tmp_path)}}
    prior_config = {
        'sampler': {
            'sampler_kwargs': {'filepath': 'other.hdf5'},
            'run_kwargs': {},
        }
    }

    with pytest.raises(ValueError, match='PICASO-managed'):
        run_nautilus(
            config, lambda theta: 0.0, lambda u: u, 2, None, prior_config
        )


def test_check_resume_exists_for_nautilus(tmp_path):
    assert not check_resume_exists('nautilus', str(tmp_path))
    (tmp_path / 'nautilus.hdf5').touch()
    assert check_resume_exists('NAUTILUS', str(tmp_path))
