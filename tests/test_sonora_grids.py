"""
inputs.sonora() for the Diamondback, Elf Owl and Flame Skimmer grids (Bobcat keeps its own reader).
Each test writes a few small files in the grid's real file format and naming scheme, so nothing
needs to be downloaded: Diamondback .pt (species header, gravity line, one row per level) and .cld
(eddysed, 196 wavenumbers per layer); Elf Owl and Flame Skimmer xarray .nc files.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from picaso import justdoit as jdi

NLEVEL = 91
PRESSURE = np.logspace(-4, 2, NLEVEL)
DB_SPECIES = ['e-', 'H2', 'H', 'He', 'H2O', 'CH4', 'CO', 'NH3']


def temperature(teff):
    return teff * (1 + np.log10(PRESSURE / PRESSURE[0]) / 6)


def write_diamondback(path, teff, grav, fsed, mh='0.0', co='1.0', cloud=True):
    name = f't{teff}g{grav}{fsed}_m{mh}_co{co}'
    with open(path / f'{name}.pt', 'w') as f:
        f.write(' n P T ' + ' '.join(DB_SPECIES) + ' \t\n')
        f.write(f'   {grav:.14f}     \n')
        for i, (p, t) in enumerate(zip(PRESSURE, temperature(teff))):
            f.write(f'{i+1:3d} {p:.5E} {t:.2f} ' + ' '.join(['0.100E-03'] * len(DB_SPECIES)) + '\n')
    if cloud and fsed != 'nc':
        nlayer, nwave = NLEVEL - 1, 196
        df = pd.DataFrame({'level_number': np.repeat(np.arange(1, nlayer + 1), nwave),
                           'spec_number': np.tile(np.arange(1, nwave + 1), nlayer),
                           'tau': 0.1, 'g0': 0.5, 'w0': 0.9, 'sigma(cm2/molec)': 1e-25})
        df.to_csv(path / f'{name}.cld', sep=' ', index=False)


def write_nc(filename, teff):
    ds = xr.Dataset({'temperature': ('pressure', temperature(teff)),
                     'H2O': ('pressure', np.full(NLEVEL, 1e-3)),
                     'H2': ('pressure', np.full(NLEVEL, 0.84)),
                     'He': ('pressure', np.full(NLEVEL, 0.16)),
                     'flux': ('wavelength', np.ones(5))},
                    coords={'pressure': PRESSURE, 'wavelength': np.linspace(1, 5, 5)})
    filename.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(filename)


def bd(grav_ms2):
    case = jdi.inputs(calculation='browndwarf')
    case.gravity(gravity=grav_ms2, gravity_unit=jdi.u.Unit('m/s**2'))
    return case


@pytest.fixture
def diamondback(tmp_path):
    tmp_path = tmp_path / 'diamondback'
    tmp_path.mkdir()
    for fsed in ['nc', 'f1', 'f3']:
        for grav in [100, 1000]:
            write_diamondback(tmp_path, 1500, grav, fsed)
    write_diamondback(tmp_path, 1600, 100, 'f1', mh='+0.5')
    write_diamondback(tmp_path, 1600, 316, 'f2', cloud=False)
    return tmp_path


@pytest.fixture
def elfowl(tmp_path):
    tmp_path = tmp_path / 'elfowl'
    tmp_path.mkdir()
    for kzz in ['2.0', '4.0']:
        for teff in ['275.0', '300.0']:
            write_nc(tmp_path / 'output_275.0_325.0' /
                     f'spectra_logzz_{kzz}_teff_{teff}_grav_1000.0_mh_0.0_co_1.0.nc', float(teff))
    return tmp_path


@pytest.fixture
def flameskimmer(tmp_path):
    tmp_path = tmp_path / 'flameskimmer'
    tmp_path.mkdir()
    for R in [3000, 30000]:  #quickstart and full files of the same model
        write_nc(tmp_path / 'deq' / 'mh_+0.0' / 'co_0.5' /
                 f'spectra_logkzz_2_teff_1000_grav_100_mh_+0.0_co_0.5_R{R}.nc', 1000)
    write_nc(tmp_path / 'eq' / 'mh_+0.0' / 'co_0.5' /
             'spectra_teff_1000_grav_100_mh_+0.0_co_0.5_R30000.nc', 1000)
    return tmp_path


def test_grid_options_parse_names(diamondback, elfowl, flameskimmer):
    db = jdi.sonora_grid_options(diamondback)
    assert len(db) == 8
    assert db['fsed'].isna().sum() == 2
    assert set(db['log_mh']) == {0.0, 0.5}

    eo = jdi.sonora_grid_options(elfowl)
    assert set(eo['logkzz']) == {2.0, 4.0} and set(eo['teff']) == {275.0, 300.0}

    fs = jdi.sonora_grid_options(flameskimmer)
    #the R=3000 and R=30000 copies of the deq model count once, the higher resolution is kept
    assert len(fs) == 2 and fs['logkzz'].isna().sum() == 1
    assert fs.loc[fs['logkzz'].notna(), 'filename'].iloc[0].endswith('R30000.nc')


def test_diamondback_cloudy(diamondback):
    case = bd(100)
    case.sonora(diamondback, 1500, fsed=1)
    profile = case.inputs['atmosphere']['profile']
    assert case.inputs['atmosphere']['sonora_filename'] == 't1500g100f1_m0.0_co1.0.pt'
    assert profile.shape[0] == NLEVEL
    np.testing.assert_allclose(profile['temperature'], temperature(1500), atol=0.01)
    assert 'H2O' in profile and 'n' not in profile
    cld = case.inputs['clouds']['profile']
    assert cld.shape == ((NLEVEL - 1) * 196, 3)
    np.testing.assert_allclose(cld['opd'], 0.1)


def test_diamondback_clear_is_default(diamondback):
    case = bd(100)
    case.sonora(diamondback, 1500)
    assert case.inputs['atmosphere']['sonora_filename'] == 't1500g100nc_m0.0_co1.0.pt'
    assert case.inputs['clouds']['profile'] is None


def test_diamondback_nearest_neighbor(diamondback):
    case = bd(900)  #nearest in log g is 1000
    case.sonora(diamondback, 1520, fsed=2.6)
    assert case.inputs['atmosphere']['sonora_filename'] == 't1500g1000f3_m0.0_co1.0.pt'
    assert case.inputs['atmosphere']['sonora_params']['fsed'] == 3.0


def test_diamondback_missing_cld(diamondback):
    with pytest.raises(Exception, match='cloud_optical_properties'):
        bd(316).sonora(diamondback, 1600, fsed=2)
    case = bd(316)
    case.sonora(diamondback, 1600, fsed=2, add_clouds=False)
    assert case.inputs['clouds']['profile'] is None


def test_elfowl(elfowl):
    case = bd(1000)
    case.sonora(elfowl, 290, logkzz=4)
    assert case.inputs['atmosphere']['sonora_filename'] == 'spectra_logzz_4.0_teff_300.0_grav_1000.0_mh_0.0_co_1.0.nc'
    profile = case.inputs['atmosphere']['profile']
    np.testing.assert_allclose(profile['temperature'], temperature(300))
    assert 'flux' not in profile
    with pytest.raises(ValueError, match='logkzz'):
        bd(1000).sonora(elfowl, 290)


def test_flameskimmer_eq_and_deq(flameskimmer):
    case = bd(100)
    case.sonora(flameskimmer, 1000, cto_relative=0.5)
    assert case.inputs['atmosphere']['sonora_filename'].startswith('spectra_teff_1000')
    case.sonora(flameskimmer, 1000, cto_relative=0.5, logkzz=2)
    assert case.inputs['atmosphere']['sonora_filename'] == 'spectra_logkzz_2_teff_1000_grav_100_mh_+0.0_co_0.5_R30000.nc'


def test_mixed_directory_needs_grid(diamondback):
    write_nc(diamondback / 'spectra_logzz_2.0_teff_300.0_grav_1000.0_mh_0.0_co_1.0.nc', 300)
    with pytest.raises(Exception, match='several Sonora grids'):
        bd(100).sonora(diamondback, 1500)
    case = bd(100)
    case.sonora(diamondback, 1500, grid='diamondback')
    assert case.inputs['atmosphere']['sonora_filename'] == 't1500g100nc_m0.0_co1.0.pt'
