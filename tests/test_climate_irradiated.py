"""
Integration test: energy balance of converged climate models of an irradiated planet (WASP-39b-like:
5327 K star at 0.0486 AU, g = 4.5 m/s2, 10x solar, the setup of the 2_Exoplanet_PreW tutorial).

At convergence the atmosphere must emit, in net, exactly the starlight it absorbs plus its own
internal heat: net IR flux - absorbed stellar flux = sigma T_int^4 at the top. picaso carries the
internal heat as tidal = -sigma T_int^4, so the total net flux (rfacv F_v + F_ir + tidal) is zero.
The internal flux here is only 1/60 to 1/2000 of the absorbed starlight, so this is a stringent
check of the climate solver and of how its stellar and thermal fluxes are combined.

Three runs (internal temperature and the fraction of starlight received, rfacv) take ~4 minutes and
need data that is not in git, so this is skipped in CI.
Run it locally with:  pytest -m integration
"""
import warnings

import astropy.units as u
import numpy as np
import pytest
import scipy.constants as sc

from conftest import REFDATA
from picaso import justdoit as jdi

PREWEIGHTED_TABLE = REFDATA / 'opacities' / 'preweighted' / 'sonora_2121grid_feh1.0_co0.46.hdf5'
SIGMA_SB = sc.sigma * 1e3  # W/(m2 K4) to erg/(cm2 s K4)
R_SUN, AU = 6.957e10, 1.495978707e13  # cm
STAR = dict(temp=5326.6, metal=-0.03, logg=4.38933, radius=0.932, semi_major=0.0486)
# (T_int K, rfacv): planet-average and dayside (all of the starlight)
CASES = [(200, 0.5), (400, 0.5), (200, 1.0)]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.needs_full_data,
    pytest.mark.skipif(not PREWEIGHTED_TABLE.exists(), reason='needs the 10x solar preweighted table'),
]


@pytest.fixture(scope='module')
def opacity():
    return jdi.opannection(ck_db=str(PREWEIGHTED_TABLE), method='preweighted')


@pytest.fixture(scope='module', params=CASES, ids=[f'tint{t}-rfacv{r}' for t, r in CASES])
def model(request, opacity):
    t_int, rfacv = request.param
    case = jdi.inputs(calculation='planet', climate=True)
    case.gravity(gravity=4.5, gravity_unit=u.Unit('m/(s**2)'))
    case.effective_temp(t_int)
    case.star(opacity, temp=STAR['temp'], metal=STAR['metal'], logg=STAR['logg'], radius=STAR['radius'],
              radius_unit=u.R_sun, semi_major=STAR['semi_major'], semi_major_unit=u.AU, database='ck04models')
    pt = case.guillot_pt(1000, nlevel=91, T_int=t_int, p_bottom=2, p_top=-6)
    case.inputs_climate(temp_guess=pt['temperature'].values, pressure=pt['pressure'].values,
                        rcb_guess=85, rfacv=rfacv)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = case.climate(opacity, verbose=False)
    return dict(t_int=t_int, rfacv=rfacv, out=out, fluxes=out['flux_balance'])


def internal_flux(model):
    return SIGMA_SB * model['t_int']**4


def absorbed_starlight(model):
    """flux_net_v is the net (down-minus-up) stellar flux, negative when energy goes in."""
    return -model['rfacv'] * model['fluxes']['flux_net_v'][0]


def test_climate_converges(model):
    assert model['out']['converged'] == 1
    assert np.all(np.isfinite(model['out']['temperature']))


def test_internal_heat_is_carried_as_the_tidal_term(model):
    np.testing.assert_allclose(model['fluxes']['tidal'], -internal_flux(model), rtol=1e-9)


def test_net_flux_at_the_top_equals_the_internal_flux(model):
    """Converged runs on 2026-10-20 balanced to 0.03% of sigma T_int^4 in all three cases."""
    emitted = model['fluxes']['flux_net_ir'][0]
    assert emitted - absorbed_starlight(model) == pytest.approx(internal_flux(model), rel=0.01)
    assert model['fluxes']['flux_net'][0] == pytest.approx(0, abs=0.01 * internal_flux(model))


def test_planet_absorbs_no_more_starlight_than_it_receives(model):
    """The incident flux is approximated by a blackbody star, sigma T*^4 (R*/a)^2."""
    incident = SIGMA_SB * STAR['temp']**4 * (STAR['radius'] * R_SUN / (STAR['semi_major'] * AU))**2
    assert 0 < absorbed_starlight(model) < model['rfacv'] * incident


def test_radiative_zone_is_close_to_radiative_equilibrium(model):
    """
    Above the radiative-convective boundary the total net flux should vanish at every level. It is
    within 3.4% of the local net IR flux in the converged runs, but in units of sigma T_int^4 that is
    up to 0.6 (T_int = 200 K) and 2.6 (dayside), where absorbed starlight dominates the budget.
    """
    radiative = slice(0, model['out']['cvz_locs'][1])
    net, net_ir = model['fluxes']['flux_net'][radiative], model['fluxes']['flux_net_ir'][radiative]
    assert np.max(np.abs(net / net_ir)) < 0.05
