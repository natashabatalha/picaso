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


def test_parent_directory_and_blank_toml_values(diamondback):
    #sonora_path can be the folder that holds one folder per grid; '' means "not set", as in the TOML templates
    case = bd(100)
    case.sonora(diamondback.parent, 1500, grid='diamondback', fsed='1', logkzz='', chem='')
    assert case.inputs['atmosphere']['sonora_filename'] == 't1500g100f1_m0.0_co1.0.pt'
    assert case.inputs['clouds']['profile'] is not None
    case = bd(100)
    case.sonora(diamondback.parent, 1500, grid='diamondback', fsed='nc')
    assert case.inputs['clouds']['profile'] is None


def test_driver_pt_handler(diamondback):
    import picaso.driver as go
    pt_config = {'profile': 'sonora',
                 'sonora': {'sonora_path': str(diamondback.parent), 'grid': 'diamondback',
                            'grid_options': ['bobcat', 'diamondback', 'elfowl', 'flameskimmer'],
                            'teff': 1500.0, 'log_mh': 0.0, 'cto_relative': 1.0,
                            'logkzz': '', 'fsed': 1, 'chem': ''}}
    case = bd(100)
    df = go.PT_handler(pt_config, case, None)
    assert 'H2O' in df and df.shape[0] == NLEVEL
    assert case.inputs['clouds']['profile'] is not None
    #the old name of the option still works
    case = bd(100)
    df_old = go.PT_handler({'profile': 'sonora_bobcat', 'sonora_bobcat': pt_config['sonora']}, case, None)
    pd.testing.assert_frame_equal(df, df_old)


def test_driver_templates_have_sonora_blocks():
    import picaso.driver as go
    config = go.load_template_config()
    assert 'sonora' in go.pt_options and 'sonora' in go.climate_pt_options and 'sonora' in go.chem_options
    assert 'sonora' in config['temperature'] and 'sonora' in config['chemistry']
    assert 'sonora' in go.load_climate_template_config()['temperature']


def write_spectra(diamondback_dir):
    """Published spectra in the Diamondback (.spec, W/m2/m) and Bobcat (sp_*.gz, erg/cm2/s/Hz) formats"""
    import gzip
    wave = np.linspace(20, 1, 50)  #micron, decreasing as in the published files
    with open(diamondback_dir / 't1500g100f1_m0.0_co1.0.spec', 'w') as f:
        f.write('wavelength (micron) \t flux (W/m2/m)\t\ncalculated using grid spacing: std \t\nmolecules included: [] \t\n')
        f.writelines(f'{w}\t1.0\n' for w in wave)
    bobcat = diamondback_dir.parent / 'bobcat'
    (bobcat / 'spectra').mkdir(parents=True)
    with gzip.open(bobcat / 'spectra' / 'sp_t900g100nc_m0.0.gz', 'wt') as f:
        f.write('  900.  100.  0.28  0.00  1.0000E+05  0.00  1.00  1.00  Teff, grav (MKS), Y, f_rain, Kzz, [Fe/H], C/O, f_hole\n')
        f.write('    microns  Flux (erg/cm^2/s/Hz)\n')
        f.writelines(f'  {w:.7f}  1.0E-05\n' for w in wave)
    with open(bobcat / 't900g100nc_m0.0.dat', 'w') as f:
        f.write(' header\n')
        f.writelines(f'{i+1} {p:.4e} {t:.1f}\n' for i, (p, t) in enumerate(zip(PRESSURE, temperature(900))))
    return bobcat


def test_published_spectra_and_models(diamondback):
    from picaso import sonora
    bobcat = write_spectra(diamondback)
    pt = str(diamondback / 't1500g100f1_m0.0_co1.0.pt')
    wave, flux = sonora.read_sonora_spectrum(sonora.find_sonora_spectrum(pt))
    assert np.all(np.diff(wave) > 0)
    np.testing.assert_allclose(flux, 10.0)  #1 W/m2/m = 10 erg/cm2/s/cm
    spec = sonora.find_sonora_spectrum(str(bobcat / 't900g100nc_m0.0.dat'), str(bobcat))
    assert spec.endswith('sp_t900g100nc_m0.0.gz')
    wave, flux = sonora.read_sonora_spectrum(spec)
    np.testing.assert_allclose(flux, 1e-5 * 2.99792458e10 / (wave * 1e-4)**2)  #F_nu to F_lambda
    assert sonora.find_sonora_spectrum(str(diamondback / 't1500g100nc_m0.0_co1.0.pt')) is None
    assert sonora.find_sonora_cloud(pt).endswith('.cld')
    assert sonora.find_sonora_cloud(str(diamondback / 't1500g100nc_m0.0_co1.0.pt')) is None
    df = sonora.read_sonora_model(str(bobcat / 't900g100nc_m0.0.dat'))
    assert list(df.columns) == ['pressure', 'temperature'] and len(df) == NLEVEL


def test_sonora_viewer_page(diamondback, elfowl, flameskimmer):
    from picaso.driver_ui.app import create_app
    write_spectra(diamondback)
    root = diamondback.parent  #holds one folder per grid, like $picaso_refdata/sonora_grids
    client = create_app(str(jdi.__refdata__)).test_client()
    page = client.get(f'/sonora/?root={root}').get_data(as_text=True)
    for grid in ['Bobcat', 'Diamondback', 'Elf Owl', 'Flame Skimmer']:
        assert f'value="{grid.lower().replace(" ", "")}"' in page
    assert 'diamondback-spectra' in page and '8.56 GB' in page  #every download is listed with its size
    assert 'class="on-disk"' in page  #downloads on disk get a check mark
    form = {'root': str(root), 'loaded_root': str(root), 'grid': 'diamondback', 'loaded_grid': 'diamondback',
            'teff': '1500.0', 'grav': '100.0', 'log_mh': '0.0', 'cto_relative': '1.0', 'fsed': '1.0', 'action': 'add'}
    page = client.post('/sonora/plot', data=form).get_data(as_text=True)
    for plot in ['pt', 'chemistry', 'clouds', 'spectrum']:
        assert f'id="plot-sonora-{plot}"' in page
    assert 'Spectra not downloaded' not in page
    compare = str(diamondback / 't1500g100f1_m0.0_co1.0.pt')
    assert f'name="compare" value="{compare}"' in page
    #the cloud-free model has no published spectrum on disk
    page = client.post('/sonora/plot', data={**form, 'fsed': '', 'action': ''}).get_data(as_text=True)
    assert 'Spectra not downloaded' in page and "diamondback-spectra" in page
    assert 'id="plot-sonora-pt"' in page
    #the comparison is kept when browsing another grid
    page = client.post('/sonora/plot', data={'root': str(root), 'loaded_root': str(root), 'grid': 'elfowl',
                                             'loaded_grid': 'diamondback', 'compare': compare}).get_data(as_text=True)
    assert f'name="compare" value="{compare}"' in page and 'id="plot-sonora-spectrum"' in page
    page = client.get('/sonora/?root=/no/such/folder').get_data(as_text=True)
    assert 'Sonora grid folder not found' in page
    assert 'diamondback-spectra' in page  #the downloads are still listed
