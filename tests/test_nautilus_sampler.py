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


def test_read_retrievals_nautilus(tmp_path):
    """A short real Nautilus run on a Gaussian likelihood is read back in physical units."""
    pytest.importorskip('nautilus')
    from functools import partial

    import numpy as np
    from nautilus import Sampler

    from picaso import retrieval as ret
    from picaso.driver import hypercube

    fitpars = {
        'a': {'prior': 'uniform', 'uniform_kwargs': {'min': -5, 'max': 5}, 'log': False},
        'b': {'prior': 'uniform', 'uniform_kwargs': {'min': -3, 'max': 1}, 'log': True},
    }
    truth = np.array([1.0, 0.01])

    def loglike(x):
        return -0.5 * (((x[0] - truth[0]) / 0.2) ** 2 + ((np.log10(x[1]) + 2) / 0.1) ** 2)

    sampler = Sampler(partial(hypercube, fitpars=fitpars), loglike, n_dim=2, n_live=200,
                      filepath=str(tmp_path / 'nautilus.hdf5'), seed=1)
    sampler.run(n_eff=1000, verbose=False)

    info = ret.read_retrievals(str(tmp_path), fitpars)

    assert info['param_names'] == ['a', 'b']
    assert info['log_params'] == ['b']
    assert info['samples_equal'].shape[1] == 2
    assert info['med_point'][0] == pytest.approx(1.0, abs=0.1)
    assert np.log10(info['med_point'][1]) == pytest.approx(-2.0, abs=0.05)
    assert info['max_logl'] == pytest.approx(0.0, abs=0.5)
    assert info['ess'] >= 1000
    assert np.isfinite(info['log_z'])

    with pytest.raises(ValueError, match='prior_finder'):
        ret.read_retrievals(str(tmp_path), ['a', 'b'])
