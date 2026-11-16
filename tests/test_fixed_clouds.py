"""
Fixed (static) clouds in climate: a cloud added with clouds() after inputs_climate() and without
virga() is included in the opacities at every step but is never recomputed (PR #279).

The checks on the cloud grid run on their own. The climate runs use the solar preweighted table,
which is not in git, so they are skipped in CI.
Run them locally with:  pytest -m integration tests/test_fixed_clouds.py
"""
import copy
import warnings

import matplotlib
matplotlib.use('Agg')
import astropy.units as u
import numpy as np
import pytest

from conftest import REFDATA
from picaso import justdoit as jdi
from picaso import justplotit as jpi

PREWEIGHTED_TABLE = REFDATA / 'opacities' / 'preweighted' / 'sonora_2121grid_feh0.0_co0.46.hdf5'
TEFF = 1000.
GRAVITY = 1000.  # m/s2
NLEVEL = 51

needs_table = pytest.mark.skipif(not PREWEIGHTED_TABLE.exists(), reason='needs the solar preweighted table')


def brown_dwarf(nlevel=NLEVEL):
    case = jdi.inputs(calculation='browndwarf', climate=True)
    case.gravity(gravity=GRAVITY, gravity_unit=u.Unit('m/(s**2)'))
    case.effective_temp(TEFF)
    pt = case.guillot_pt(TEFF, nlevel=nlevel, T_int=TEFF, p_bottom=2, p_top=-4)
    case.inputs_climate(temp_guess=pt['temperature'].values, pressure=pt['pressure'].values,
                        rcb_guess=nlevel - 10, rfacv=0.0)
    return case


def add_deck(case, opd, g0=0.8, w0=0.9):
    """Grey deck between 1 and 10 bar with optical depth `opd` in each layer."""
    case.clouds(p=[1.0], dp=[1.0], opd=[opd], g0=[g0], w0=[w0])


def test_no_cloud_is_not_a_fixed_cloud():
    case = brown_dwarf()
    assert not case._fixed_cloud(case.inputs['climate'].get('cloudy', False))


def test_clouds_after_inputs_climate_is_a_fixed_cloud():
    case = brown_dwarf()
    add_deck(case, 1.0)
    assert case._fixed_cloud(False)
    case._check_fixed_cloud(case.inputs['climate']['pressure'])
    assert 'clouds: fixed' in case._climate_summary(False, False, False, 0.0)


def test_virga_takes_precedence_over_a_fixed_cloud():
    case = brown_dwarf()
    add_deck(case, 1.0)
    assert not case._fixed_cloud(True)


def test_cloud_made_on_another_pressure_grid_is_rejected():
    case = brown_dwarf(nlevel=61)
    add_deck(case, 1.0)
    #a new climate grid with a different number of levels
    pt = case.guillot_pt(TEFF, nlevel=NLEVEL, T_int=TEFF, p_bottom=2, p_top=-4)
    with pytest.raises(Exception, match='after inputs_climate'):
        case._check_fixed_cloud(pt['pressure'].values)


def test_cloud_shifted_in_pressure_is_rejected():
    case = brown_dwarf()
    add_deck(case, 1.0)
    #same number of layers but a pressure grid one decade deeper
    with pytest.raises(Exception, match='do not match'):
        case._check_fixed_cloud(case.inputs['climate']['pressure']*10)


@pytest.fixture(scope='module')
def runs():
    """Cloud-free, zero-opacity fixed cloud and opaque fixed deck, all from the same start."""
    opacity = jdi.opannection(ck_db=str(PREWEIGHTED_TABLE), method='preweighted')
    out = {}
    #the empty cloud also needs g0=0: optics uses the cloud g0 as the layer asymmetry even where opd=0
    for name, cloud in [('clear', None), ('zero', dict(opd=0.0, g0=0.0, w0=0.0)), ('deck', dict(opd=1.0))]:
        case = brown_dwarf()
        if cloud is not None:
            add_deck(case, **cloud)
        cloud_in = None if cloud is None else case.inputs['clouds']['profile'].copy()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out[name] = dict(out=case.climate(opacity, verbose=False), cloud_in=cloud_in,
                             cloud_after=case.inputs['clouds']['profile'], case=case, opacity=opacity)
    return out


@pytest.mark.integration
@needs_table
def test_fixed_cloud_runs_converge(runs):
    for run in runs.values():
        assert run['out']['converged'] == 1


@pytest.mark.integration
@needs_table
def test_fixed_cloud_is_never_changed(runs):
    """The cloud returned (and still on the bundle) is exactly the cloud that went in."""
    for name in ('zero', 'deck'):
        run = runs[name]
        jdi.pd.testing.assert_frame_equal(run['out']['cld_df'], run['cloud_in'])
        jdi.pd.testing.assert_frame_equal(run['cloud_after'], run['cloud_in'])
    assert 'cld_df' not in runs['clear']['out']


@pytest.mark.integration
@needs_table
def test_zero_opacity_fixed_cloud_is_cloud_free(runs):
    np.testing.assert_allclose(runs['zero']['out']['temperature'], runs['clear']['out']['temperature'], rtol=0)


@pytest.mark.integration
@needs_table
def test_opaque_deck_warms_the_atmosphere_below_it(runs):
    """With no star, a cloud only traps the thermal emission from below (a greenhouse effect)."""
    pressure = runs['clear']['out']['pressure']
    below = pressure > 10
    dT = runs['deck']['out']['temperature'] - runs['clear']['out']['temperature']
    assert np.all(dT[below] > 0)
    assert np.max(dT[below]) > 50


def fake_climate_out(temperature_shift=0., cloud=False):
    """Minimal climate() output: 21 levels, flat fluxes, optionally a cloud between 0.1 and 1 bar."""
    pressure = np.logspace(-4, 2, 21)
    p_layer = np.sqrt(pressure[1:]*pressure[:-1])
    wno = np.linspace(100, 30000, 50)
    out = dict(pressure=pressure, temperature=np.linspace(800, 2000, 21) + temperature_shift,
               flux_balance=dict(flux_net_v=-np.linspace(1e8, 0, 21), flux_net_ir=np.linspace(5e7, 1e5, 21),
                                 rfacv=0.5, rfaci=1),
               flux_ir_attop=np.full(len(wno), 1e6),
               spectrum_output=dict(wavenumber=wno))
    if cloud:
        rows = [(p, w, 1.0 if 0.1 <= p <= 1 else 0.0) for p in p_layer for w in wno]
        out['cld_df'] = jdi.pd.DataFrame(rows, columns=['pressure', 'wavenumber', 'opd']).assign(g0=0.8, w0=0.9)
    return out


@pytest.mark.parametrize('backend', ['bokeh', 'matplotlib', 'plotly'])
def test_pt_plots_several_climate_profiles(backend):
    outs = [fake_climate_out(), fake_climate_out(100.)]
    fig = jpi.pt(outs, legend=['a', 'b'], backend=backend)
    if backend == 'plotly':
        assert len(fig.data) == 2
        np.testing.assert_array_equal(fig.data[1].x, outs[1]['temperature'])
    elif backend == 'matplotlib':
        assert len(fig.axes[0].lines) == 2


def test_flux_analyzer_finds_the_cloud_layers():
    clear, cloudy = jpi.flux_analyzer(fake_climate_out(), fake_climate_out(cloud=True), plot=False)
    assert clear['cloud_range'] is None
    p_layer = np.sqrt(cloudy['pressure'][1:]*cloudy['pressure'][:-1])
    in_cloud = p_layer[(p_layer >= 0.1) & (p_layer <= 1)]
    assert cloudy['cloud_range'] == (in_cloud.min(), in_cloud.max())


def converged_bundle(run):
    """A copy of the climate bundle of `run` (so the shared fixture is not changed), set to its converged profile."""
    case = copy.deepcopy(run['case'])
    case.atmosphere(df=run['out']['ptchem_df'])
    return case


@pytest.mark.integration
@needs_table
def test_energy_budget_reproduces_the_converged_climate(runs):
    """At the converged temperature, the level fluxes give back the climate fluxes and an energy balance."""
    for name in ('clear', 'deck'):
        run = runs[name]
        case = converged_bundle(run)
        profile_before = case.inputs['atmosphere']['profile']
        budget = case.energy_budget(run['opacity'])
        np.testing.assert_allclose(budget['flux_balance']['flux_net_ir'], run['out']['flux_balance']['flux_net_ir'], rtol=0.01)
        assert abs(budget['imbalance']) < 0.01*budget['outgoing_thermal']
        assert ('cld_df' in budget) == (name == 'deck')
        #energy_budget does not change the bundle
        assert case.inputs['atmosphere']['profile'] is profile_before


@pytest.mark.integration
@needs_table
def test_energy_budget_of_a_cloud_added_at_fixed_temperature(runs):
    """Adding the deck to the cloud-free profile traps thermal radiation (less escapes); clouds_reset removes it."""
    run = runs['clear']
    case = converged_bundle(run)
    clear = case.energy_budget(run['opacity'])
    case.clouds(df=runs['deck']['out']['cld_df'])
    instant = case.energy_budget(run['opacity'])
    assert instant['outgoing_thermal'] < clear['outgoing_thermal']
    assert instant['imbalance'] > 0.01*clear['outgoing_thermal']
    np.testing.assert_array_equal(instant['temperature'], clear['temperature'])

    case.clouds_reset()
    reset = case.energy_budget(run['opacity'])
    np.testing.assert_allclose(reset['flux_balance']['flux_net_ir'], clear['flux_balance']['flux_net_ir'], rtol=1e-10)
    #clouds_reset does not zero the cloud saved in the climate output
    assert runs['deck']['out']['cld_df']['opd'].max() > 0


def fake_energy_budget(tint=200., shift=0.):
    """Minimal energy_budget output in equilibrium (constant net flux) with a convective zone at the bottom."""
    sigma = 5.670374419e-5
    pressure = np.logspace(-4, 2, 21)
    net = np.full(21, sigma*tint**4) + shift
    return dict(pressure=pressure, temperature=np.linspace(800, 2000, 21), internal_heat=sigma*tint**4,
                flux_net_radiative=net, tint_computed=np.sign(net)*(np.abs(net)/sigma)**0.25)


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_equilibrium_check_plots_profiles_and_computed_tint(backend):
    budgets = [fake_energy_budget(), fake_energy_budget(shift=1e5)]
    fig = jpi.equilibrium_check(budgets, labels=['a', 'b'], cvz_locs=[0, 16, 19, 0, 0, 0], backend=backend)
    if backend == 'plotly':
        assert len(fig.data) == 4  # T(P) and computed Tint for each budget
        np.testing.assert_allclose(fig.data[1].x, 200.)
        assert len(fig.layout.shapes) == 3  # convective zone in both panels + target line
    else:
        assert len(fig.axes) == 2 and len(fig.axes[1].lines) == 4  # 2 profiles + target + zero line
        assert len(fig.axes[1].patches) == 1  # convective zone
    with pytest.raises(ValueError):
        jpi.equilibrium_check(budgets, backend='bokeh')


def test_equilibrium_check_target_tint():
    budgets = [fake_energy_budget(tint=200.)]
    #the same target as the energy budget: no warning
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        fig = jpi.equilibrium_check(budgets, target_tint=200., backend='plotly')
    assert fig.layout.shapes[-1].x0 == 200.
    #a different target is plotted, with a warning
    with pytest.warns(UserWarning, match='differs'):
        fig = jpi.equilibrium_check(budgets, target_tint=300., backend='plotly')
    assert fig.layout.shapes[-1].x0 == 300.


@pytest.mark.integration
@needs_table
def test_energy_budget_of_any_profile_and_computed_tint(runs):
    """A converged brown dwarf radiates sigma Teff^4 through its radiative zone; a shifted profile does not."""
    run = runs['clear']
    case = converged_bundle(run)
    converged = case.energy_budget(run['opacity'])
    radiative = converged['pressure'] < converged['pressure'][run['out']['cvz_locs'][1]]
    np.testing.assert_allclose(converged['tint_computed'][radiative], TEFF, rtol=0.02)
    #no star: the radiative flux is all thermal
    np.testing.assert_array_equal(converged['flux_net_radiative'], converged['flux_balance']['flux_net_ir'])

    hotter = run['out']['ptchem_df'].copy()
    hotter['temperature'] = hotter['temperature'] + 200
    case.atmosphere(df=hotter)
    shifted = case.energy_budget(run['opacity'])
    np.testing.assert_array_equal(shifted['temperature'], hotter['temperature'].values)
    assert np.median(shifted['tint_computed'][radiative]) > 1.1*TEFF
