"""
output_xarray / input_xarray round trip (issue #449). A model saved to xarray and read back with
input_xarray gives the same spectrum. A star from a user file (star(filename=...)) is stored in the
xarray itself, so the round trip works without the original file or any stellar grid.

Uses the mini H2O/CO2 opacity db in git.
"""
import astropy.units as u
import numpy as np
import pytest

from picaso import justdoit as jdi

H, C, K_B = 6.62607015e-27, 2.99792458e10, 1.380649e-16  # cgs


def write_blackbody_star(path, teff=5000.):
    """Two-column (micron, erg/cm2/s/cm) blackbody surface flux."""
    wave_um = np.linspace(0.3, 30, 4000)
    wave_cm = wave_um * 1e-4
    flux = np.pi * 2 * H * C**2 / wave_cm**5 / np.expm1(H * C / (wave_cm * K_B * teff))
    np.savetxt(path, np.column_stack([wave_um, flux]))
    return path


def planet(mini_opacity, star_file):
    case = jdi.inputs()
    case.phase_angle(0)
    case.gravity(mass=1, mass_unit=u.Unit('M_jup'), radius=1, radius_unit=u.Unit('R_jup'))
    star_kwargs = dict(radius=1, radius_unit=u.Unit('R_sun'), semi_major=0.05, semi_major_unit=u.AU)
    if star_file is None:
        case.star(mini_opacity, 5000, 0, 4.5, database='phoenix', **star_kwargs)
    else:
        case.star(mini_opacity, filename=str(star_file), w_unit='um', f_unit='erg/(cm**2*s*cm)', **star_kwargs)
    pressure = np.logspace(-6, 2, 60)
    case.atmosphere(df=jdi.pd.DataFrame(dict(pressure=pressure, temperature=np.full(60, 1200.),
                                             H2O=np.full(60, 1e-3), CO2=np.full(60, 1e-5),
                                             H2=np.full(60, 0.85), He=np.full(60, 0.149))))
    return case


def test_user_stellar_file_round_trips(mini_opacity, tmp_path):
    star_file = write_blackbody_star(tmp_path / 'my_star.txt')
    case = planet(mini_opacity, star_file)
    out = case.spectrum(mini_opacity, calculation='thermal', full_output=True)
    savefile = str(tmp_path / 'model.nc')
    jdi.output_xarray(out, case, savefile=savefile)
    star_file.unlink()  # after saving, the xarray alone must be enough to rerun the model
    reloaded = jdi.input_xarray(jdi.xr.load_dataset(savefile), mini_opacity)
    again = reloaded.spectrum(mini_opacity, calculation='thermal', full_output=True)

    np.testing.assert_allclose(reloaded.inputs['star']['flux'], case.inputs['star']['flux'], rtol=1e-10)
    np.testing.assert_allclose(again['thermal'], out['thermal'], rtol=1e-10)
    np.testing.assert_allclose(again['fpfs_thermal'], out['fpfs_thermal'], rtol=1e-10)
    assert reloaded.inputs['star']['filename'] == 'my_star.txt'


def test_user_stellar_file_is_stored_in_the_xarray(mini_opacity, tmp_path):
    star_file = write_blackbody_star(tmp_path / 'my_star.txt')
    case = planet(mini_opacity, star_file)
    ds = jdi.output_xarray(case.spectrum(mini_opacity, calculation='thermal', full_output=True), case)
    wave, flux = np.loadtxt(star_file, unpack=True)
    np.testing.assert_array_equal(ds['stellar_wavelength'].values, wave)
    np.testing.assert_array_equal(ds['stellar_flux'].values, flux)
    assert ds['stellar_flux'].attrs['units'] == 'erg/(cm**2*s*cm)'
    assert 'database' not in ds.attrs['stellar_params']


@pytest.mark.needs_full_data
def test_stellar_grid_round_trips(mini_opacity, tmp_path):
    case = planet(mini_opacity, None)
    out = case.spectrum(mini_opacity, calculation='thermal', full_output=True)
    savefile = str(tmp_path / 'model.nc')
    jdi.output_xarray(out, case, savefile=savefile)
    again = jdi.input_xarray(jdi.xr.load_dataset(savefile), mini_opacity).spectrum(
        mini_opacity, calculation='thermal', full_output=True)
    np.testing.assert_allclose(again['fpfs_thermal'], out['fpfs_thermal'], rtol=1e-10)


def test_clouds_round_trip_with_their_coordinates(mini_opacity, tmp_path):
    """Cloud optical properties are saved on pressure_layer / wavenumber_layer coordinates and reload unchanged."""
    case = planet(mini_opacity, write_blackbody_star(tmp_path / 'my_star.txt'))
    case.clouds(g0=[0.5], w0=[0.8], opd=[0.5], p=[-1.0], dp=[1.5])
    out = case.spectrum(mini_opacity, calculation='thermal', full_output=True)
    savefile = str(tmp_path / 'model.nc')
    ds = jdi.output_xarray(out, case, savefile=savefile)

    np.testing.assert_array_equal(ds['wavenumber_layer'].values, case.inputs['clouds']['wavenumber'])
    assert len(ds['pressure_layer']) == case.nlevel - 1
    reloaded = jdi.input_xarray(jdi.xr.load_dataset(savefile), mini_opacity)
    for key in ['opd', 'w0', 'g0']:
        np.testing.assert_allclose(reloaded.inputs['clouds']['profile'][key].values,
                                   case.inputs['clouds']['profile'][key].values, rtol=1e-12)
    again = reloaded.spectrum(mini_opacity, calculation='thermal', full_output=True)
    np.testing.assert_allclose(again['thermal'], out['thermal'], rtol=1e-10)
