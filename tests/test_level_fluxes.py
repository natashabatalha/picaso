"""
Tests for the level fluxes that spectrum() returns when approx(get_lvl_flux=True).

At the top of the atmosphere nothing diffuse comes in from above, so the downward reflected flux
there is only the direct stellar beam, mu0 * F0PI, whatever the opacities or the number of
correlated-k gauss points. For the single column that energy budgets use (phase_angle(0,
num_gangle=1), lit at mu0 = 0.5 like the climate code) that is F0PI/2; for the usual disk geometry
the level fluxes are averaged per unit area over the visible hemisphere. These run on the mini opacity db and the mini
resortrebin k-tables in git, with a blackbody star written to a file, so no stellar grids are needed.
"""
import numpy as np
import pandas as pd
import pytest


def blackbody_star_file(path, temperature=6000.):
    """Two column file: wavelength (micron) and blackbody flux (erg/s/cm2/Angstrom)."""
    h, c, k = 6.62607004e-27, 2.99792458e10, 1.38064852e-16
    wave_um = np.linspace(0.1, 40, 20000)
    wave_cm = wave_um * 1e-4
    flux_per_cm = np.pi * 2 * h * c**2 / wave_cm**5 / np.expm1(h * c / (wave_cm * k * temperature))
    np.savetxt(path, np.column_stack([wave_um, flux_per_cm * 1e-8]))
    return str(path)


def reflected_level_fluxes(opa, star_file, num_gangle=1):
    from picaso import justdoit as jdi
    case = jdi.inputs()
    case.approx(get_lvl_flux=True)
    case.phase_angle(0, num_gangle=num_gangle, num_tangle=1)
    case.gravity(radius=1, radius_unit=jdi.u.Unit('R_jup'), mass=1, mass_unit=jdi.u.Unit('M_jup'))
    case.star(opa, filename=star_file, w_unit='um', f_unit='erg/(s cm2 Angstrom)',
              radius=1, radius_unit=jdi.u.Unit('R_sun'), semi_major=0.05, semi_major_unit=jdi.u.Unit('au'))
    pt = pd.read_csv(jdi.HJ_pt(), sep=r'\s+')
    case.atmosphere(df=pd.DataFrame({'pressure': pt['pressure'], 'temperature': pt['temperature'],
                                     'H2': 0.85, 'He': 0.15, 'H2O': 3e-4, 'CO2': 1e-6}))
    out = case.spectrum(opa, calculation='reflected', full_output=True)
    return out['full_output']['level']['reflected_fluxes'], case.inputs['star']['relative_flux'], case.inputs['disco']


@pytest.fixture(scope='module')
def star_file(tmp_path_factory):
    return blackbody_star_file(tmp_path_factory.mktemp('star') / 'blackbody_6000K.txt')


@pytest.mark.parametrize('method', ['monochromatic', 'correlated-k'])
def test_top_downward_reflected_flux_is_half_the_stellar_flux(mini_opacity, mini_ck_opacity, star_file, method):
    """
    Monochromatic opacities have one gauss point and correlated-k has eight. The level fluxes must be
    summed over all eight with their weights; keeping only one makes the ck fluxes far too small.
    """
    opa = mini_opacity if method == 'monochromatic' else mini_ck_opacity
    level, relative_flux, _ = reflected_level_fluxes(opa, star_file)
    np.testing.assert_allclose(np.asarray(level['flux_minus'])[0] / relative_flux, 0.5, rtol=1e-10)


def test_disk_geometry_averages_level_fluxes_over_the_visible_hemisphere(mini_opacity, star_file):
    """At full phase each facet's direct beam is mu1 * F0PI, so the area-weighted (w / mu1) average is
    sum(w) / sum(w / mu1) times F0PI."""
    level, relative_flux, disco = reflected_level_fluxes(mini_opacity, star_file, num_gangle=10)
    w = np.asarray(disco['gweight'])[:, None] * np.asarray(disco['tweight'])[None, :]
    expected = np.sum(w) / np.sum(w / np.asarray(disco['ubar1']))
    np.testing.assert_allclose(np.asarray(level['flux_minus'])[0] / relative_flux, expected, rtol=1e-10)


def test_single_column_cannot_be_used_for_thermal_emission(mini_opacity):
    from picaso import justdoit as jdi
    case = jdi.inputs(calculation='browndwarf')
    case.phase_angle(0, num_gangle=1, num_tangle=1)
    case.gravity(gravity=100, gravity_unit=jdi.u.Unit('m/(s**2)'))
    pt = pd.read_csv(jdi.HJ_pt(), sep=r'\s+')
    case.atmosphere(df=pd.DataFrame({'pressure': pt['pressure'], 'temperature': pt['temperature'],
                                     'H2': 0.85, 'He': 0.15, 'H2O': 3e-4, 'CO2': 1e-6}))
    with pytest.raises(Exception, match='single column'):
        case.spectrum(mini_opacity, calculation='thermal')


def test_monochromatic_level_fluxes_are_in_spectrum_units(mini_opacity):
    """With integrated_energy=False the level fluxes are monochromatic: the upward flux at the midpoint
    of the top layer is exactly the thermal spectrum."""
    from picaso import justdoit as jdi
    case = jdi.inputs(calculation='browndwarf')
    case.approx(get_lvl_flux=True, integrated_energy=False)
    case.phase_angle(0)
    case.gravity(gravity=100, gravity_unit=jdi.u.Unit('m/(s**2)'))
    pt = pd.read_csv(jdi.HJ_pt(), sep=r'\s+')
    case.atmosphere(df=pd.DataFrame({'pressure': pt['pressure'], 'temperature': pt['temperature'],
                                     'H2': 0.85, 'He': 0.15, 'H2O': 3e-4, 'CO2': 1e-6}))
    out = case.spectrum(mini_opacity, calculation='thermal', full_output=True)
    top_up = np.asarray(out['full_output']['level']['thermal_fluxes']['flux_plus_mdpt'])[0]
    np.testing.assert_array_equal(top_up, np.asarray(out['thermal']))


def test_climate_always_uses_level_fluxes_and_integrated_energy():
    from picaso import justdoit as jdi
    case = jdi.inputs(calculation='planet', climate=True)
    assert case.inputs['approx']['get_lvl_flux'] and case.inputs['approx']['integrated_energy']
    case.approx()  # a later approx() call cannot turn them off for climate
    assert case.inputs['approx']['get_lvl_flux'] and case.inputs['approx']['integrated_energy']


@pytest.mark.parametrize('integrated_energy', [False, True])
def test_star_flux_units_follow_integrated_energy(mini_opacity, star_file, integrated_energy):
    """Only integrated_energy, not get_lvl_flux, decides whether star() returns energy per bin."""
    from picaso import justdoit as jdi
    case = jdi.inputs()
    case.approx(get_lvl_flux=True, integrated_energy=integrated_energy)
    case.star(mini_opacity, filename=star_file, w_unit='um', f_unit='erg/(s cm2 Angstrom)',
              radius=1, radius_unit=jdi.u.Unit('R_sun'), semi_major=0.05, semi_major_unit=jdi.u.Unit('au'))
    per_bin = 'ergs cm^{-2} s^{-1}'
    assert (case.inputs['star']['flux_unit'] == per_bin) == integrated_energy
