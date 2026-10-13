"""
The climate progress reporter (picaso.climate_log): what it prints, what it streams to watchers,
and that it does nothing at all when it is off. Fed by hand, so no climate run is needed.
"""
from collections import namedtuple

import numpy as np
import pytest

from picaso import climate_log
from picaso.climate_log import ClimateReporter, ConvergenceHistory, new_newton_diagnostics

Criteria = namedtuple('Criteria', ['max_inner_iterations', 'max_outer_iterations', 'outer_threshold'])
PRESSURE = np.logspace(-4, 2, 20)
NSTR = [0, 15, 18, 0, 0, 0]


def newton_table(rows):
    table = new_newton_diagnostics(5)
    for i, row in enumerate(rows):
        table[i] = row
    return table


def run(reporter):
    """Feeds a reporter the hooks of a short run: one stage, a zone move, two outer iterations."""
    temperature = np.linspace(500, 2000, len(PRESSURE))
    reporter.start(PRESSURE, temperature, summary=['test planet'], n_stages=1)
    reporter.stage('Relax the initial guess')
    reporter.profile_start(Criteria(5, 3, 2.0), NSTR, 1)
    # step 1 clips 3 levels to the opacity grid and backtracks; step 2 converges
    newton = newton_table([[1, 400, 2000, 0.3, 0.1, 0.2, 3, 1, climate_log.NEWTON_CONTINUE],
                           [2, 400, 2000, 1e-4, 1.0, 1e-3, 0, 0, climate_log.NEWTON_CONVERGED]])
    reporter.iteration(0, temperature, 50.0, newton, NSTR, 1)
    reporter.convection('zone top moved up 1 level(s)', [0, 14, 18, 0, 0, 0], 1)
    root = newton_table([[1, 400, 2000, 1e-6, 0, 0, 0, 0, climate_log.NEWTON_AT_ROOT]])
    reporter.iteration(1, temperature, 0.5, root, NSTR, 1, converged=True)
    reporter.profile_end(True, 2)
    reporter.finish(True, temperature)


def test_off_reporter_prints_and_streams_nothing(capsys):
    reporter = ClimateReporter(verbose=False, watch=None)
    assert not reporter.on
    run(reporter)
    assert capsys.readouterr().out == ''
    assert reporter.outer_total == 0  # returned before doing any bookkeeping


def test_printout_explains_each_iteration(capsys):
    run(ClimateReporter(verbose=True))
    out = capsys.readouterr().out
    assert 'How to read this log' in out
    assert 'STAGE 1/1  Relax the initial guess' in out
    assert '2/5  ok' in out and '1/5  root' in out
    assert '3 lvl clipped to T grid' in out and 'line search x0.10' in out
    assert 'zone top moved up 1 level(s) -> zones: levels 14-19' in out
    assert '<- converged' in out and 'CONVERGED | 2 outer iterations, 3 Newton steps' in out
    assert 'newton step' not in out  # only with verbose=2


def test_debug_printout_shows_newton_steps(capsys):
    run(ClimateReporter(verbose=2))
    assert 'newton step  1' in capsys.readouterr().out


def test_watchers_receive_events_without_printing(capsys, tmp_path):
    events = []
    path = tmp_path / 'run.jsonl'
    history = ConvergenceHistory()
    run(ClimateReporter(verbose=False, watch=[events.append, str(path), history]))
    assert capsys.readouterr().out == ''

    kinds = [e['event'] for e in events if e['event'] != 'log']
    assert kinds == ['start', 'stage', 'iteration', 'convection', 'iteration', 'finish']
    iteration = next(e for e in events if e['event'] == 'iteration')
    assert iteration['newton_steps'] == 2 and iteration['zones'] == [(15, 19)]

    logged = climate_log.read_climate_log(path)
    assert [e['event'] for e in logged] == [e['event'] for e in events]

    assert len(history.profiles) == 2 and history.finished['converged']
    assert [dt for _, dt, _, _ in history.dts] == [50.0, 0.5]


def test_unknown_watch_is_an_error():
    with pytest.raises(ValueError):
        ClimateReporter(watch=42)
