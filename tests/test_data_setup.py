"""
Finding and downloading the PICASO data: picaso.paths (environment variable -> config file ->
default folder), picaso.data.setup, and get_data when nobody can answer its questions.

Nothing here downloads anything (the download functions are replaced with fakes), except
test_real_stellar_download_works_with_stsynphot, which runs only with PICASO_NETWORK_TESTS=1.
"""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest

import picaso.justdoit as jdi
from picaso import data, paths

from conftest import REFDATA


def _run_isolated(code, home, extra_env=None):
    """Run python code in a fresh process with no picaso variables and HOME set to a temp folder."""
    env = {k: v for k, v in os.environ.items()
           if k not in ('picaso_refdata', 'PYSYN_CDBS') and not k.startswith('XDG_')}
    env.update(HOME=str(home), **(extra_env or {}))
    return subprocess.run([sys.executable, '-W', 'always', '-c', code], env=env,
                          capture_output=True, text=True, timeout=600)


@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    """Point the config file and default data folder at tmp_path, and restore the environment after."""
    monkeypatch.setattr(paths, 'config_file', lambda: str(tmp_path / 'config' / 'config.toml'))
    monkeypatch.setattr(paths, 'default_data_dir', lambda: str(tmp_path / 'data'))
    for name in ('picaso_refdata', 'PYSYN_CDBS'):
        monkeypatch.setenv(name, os.environ.get(name, ''))
        monkeypatch.delenv(name)
    return tmp_path


def test_import_without_data_warns_and_opannection_explains(tmp_path):
    result = _run_isolated(
        'import picaso.justdoit as jdi\n'
        'try:\n'
        '    jdi.opannection()\n'
        'except FileNotFoundError as e:\n'
        '    print("RAISED", e)\n', tmp_path)
    assert result.returncode == 0, result.stderr
    assert 'PICASO could not find its reference data' in result.stderr
    assert 'RAISED' in result.stdout and 'picaso-data setup --bundle default-planet' in result.stdout


def test_config_file_is_used_when_no_environment_variable(tmp_path):
    code = ('from picaso import paths; import json; paths.write_config(picaso_refdata=%r)\n' % str(REFDATA))
    assert _run_isolated(code, tmp_path).returncode == 0
    result = _run_isolated('import picaso.justdoit as jdi; print("REFDATA", jdi.__refdata__)', tmp_path)
    assert f'REFDATA {REFDATA}' in result.stdout, result.stderr


def test_environment_variable_wins_over_config_file(isolated_dirs, monkeypatch):
    paths.write_config(picaso_refdata=str(isolated_dirs / 'from_config'))
    monkeypatch.setenv('picaso_refdata', str(REFDATA))
    assert paths.resolve()['picaso_refdata'] == (str(REFDATA), 'environment variable')


def test_default_folder_is_used_last(isolated_dirs):
    assert paths.resolve()['picaso_refdata'] == (None, 'not found')
    ref = isolated_dirs / 'data' / 'reference'
    (ref / 'stellar_grids' / 'grid').mkdir(parents=True)
    (ref / 'config.json').write_text('{}')
    found = paths.resolve()
    assert found['picaso_refdata'] == (str(ref), 'default data folder')
    assert found['PYSYN_CDBS'][0] == str(ref / 'stellar_grids')


def test_get_data_does_not_prompt_when_not_interactive():
    # pytest replaces stdin, so get_data must raise instead of calling input()
    with pytest.raises(ValueError, match='category_download'):
        data.get_data()
    with pytest.raises(ValueError, match='target_download'):
        data.get_data('stellar_grids')
    with pytest.raises(ValueError, match='Unknown category_download'):
        data.get_data('not_a_category', 'default')


@pytest.fixture
def fake_downloads(monkeypatch):
    """Replace the network downloads with functions that create the files they would create."""
    calls = []

    def fake_reference(save_to_dir, **kwargs):
        calls.append('reference')
        os.makedirs(os.path.join(save_to_dir, 'opacities'), exist_ok=True)
        with open(os.path.join(save_to_dir, 'config.json'), 'w') as f:
            json.dump(json.load(open(REFDATA / 'config.json')), f)

    def fake_get_data(category, target, final_destination_dir=None, keep_archive=True):
        calls.append((category, target))
        if category == 'resampled_opacity':
            open(os.path.join(final_destination_dir, 'opacities_fake.db'), 'w').close()
        elif category == 'stellar_grids':
            os.makedirs(os.path.join(final_destination_dir, target))

    monkeypatch.setattr(data, 'download_github_folder_api', fake_reference)
    monkeypatch.setattr(data, 'get_data', fake_get_data)
    return calls


def test_setup_browndwarf_skips_stellar_and_saves_config(isolated_dirs, fake_downloads):
    out = data.setup('default-browndwarf')
    ref = str(isolated_dirs / 'data' / 'reference')
    assert out == {'picaso_refdata': ref, 'PYSYN_CDBS': None}
    assert fake_downloads == ['reference', ('resampled_opacity', 'default')]
    assert paths.read_config() == {'picaso_refdata': ref}


def test_setup_planet_adds_phoenix_and_reruns_skip_what_is_there(isolated_dirs, fake_downloads):
    data.setup('default-browndwarf')
    fake_downloads.clear()
    out = data.setup('default-planet')
    assert fake_downloads == [('stellar_grids', 'phoenix')]
    assert os.path.isdir(os.path.join(out['PYSYN_CDBS'], 'grid', 'phoenix'))
    assert paths.read_config()['PYSYN_CDBS'] == out['PYSYN_CDBS']
    fake_downloads.clear()
    data.setup('default-planet')
    assert fake_downloads == []


def test_setup_rejects_unknown_bundle(isolated_dirs):
    with pytest.raises(ValueError, match='default-planet'):
        data.setup('everything')


def test_star_defaults_to_phoenix_and_explains_missing_grid(mini_opacity, tmp_path, monkeypatch):
    (tmp_path / 'grid' / 'ck04models').mkdir(parents=True)
    monkeypatch.setenv('PYSYN_CDBS', str(tmp_path))
    case = jdi.inputs()
    with pytest.raises(FileNotFoundError, match="database='ck04models'"):
        case.star(mini_opacity, temp=5000, metal=0, logg=4.0)


# --- where get_data puts the stsynphot stellar grids ---------------------------------------
#
# The STScI tarballs unpack to grp/redcat/trds/grid/<database>/ (checked for synphot3 =
# ck04models and synphot5 = phoenix). stsynphot reads $PYSYN_CDBS/grid/<database>/catalog.fits,
# so get_data must end with the models in $PYSYN_CDBS/grid/<database>/, whatever folder the
# archive was downloaded to.

STSCI_PREFIX = os.path.join('grp', 'redcat', 'trds', 'grid')
MODEL_FILES = {'ck04models': ['AA_README', 'catalog.fits', 'ckp00/ckp00_5000.fits'],
               'phoenix': ['AA_README', 'catalog.fits', 'phoenixm00/phoenixm00_5000.fits']}


def _make_stsci_archive(folder, database, kind):
    """Archive with the same layout as the STScI synphot tarballs, holding a few placeholder files."""
    src = folder / 'src'
    for name in MODEL_FILES[database]:
        f = src / STSCI_PREFIX / database / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(name)
    archive = folder / f'synphot_test.{kind}'
    if kind == 'tar.gz':
        with tarfile.open(archive, 'w:gz') as tar:
            tar.add(src / 'grp', arcname='grp')
    else:
        with zipfile.ZipFile(archive, 'w') as z:
            for f in src.rglob('*'):
                if f.is_file(): z.write(f, f.relative_to(src))
    return archive


@pytest.fixture
def fake_stellar_download(monkeypatch, tmp_path):
    """
    Serve a local archive instead of downloading from STScI. pooch.retrieve is replaced by a
    function that copies the archive into place and then runs the real pooch Untar/Unzip
    processor that get_data chose, so unpacking and moving are exactly what users get.
    """
    def serve(database, kind):
        archive = _make_stsci_archive(tmp_path / 'archive', database, kind)
        fname = f'synphot_test.{kind}'
        real_config = data.get_data_config

        def config_with_test_archive():
            inputs, config = real_config()
            config['stellar_grids'][database]['url'] = {fname: 'https://example.invalid/' + fname}
            return inputs, config

        def fake_retrieve(url, known_hash, fname, path, processor, progressbar):
            target = os.path.join(path, fname)
            shutil.copy(archive, target)
            return processor(target, 'download', None) if processor else target

        monkeypatch.setattr(data, 'get_data_config', config_with_test_archive)
        monkeypatch.setattr(data.pooch, 'retrieve', fake_retrieve)
        return fname
    return serve


@pytest.mark.parametrize('kind', ['tar.gz', 'zip'])
@pytest.mark.parametrize('database', ['phoenix', 'ck04models'])
@pytest.mark.parametrize('destination', ['default', 'grid', 'elsewhere'])
@pytest.mark.parametrize('keep_archive', [True, False])
def test_stellar_grid_lands_in_pysyn_cdbs_grid(fake_stellar_download, tmp_path, monkeypatch,
                                               kind, database, destination, keep_archive):
    stellar = tmp_path / 'stellar'
    grid = stellar / 'grid'
    grid.mkdir(parents=True)
    monkeypatch.setenv('PYSYN_CDBS', str(stellar))
    fname = fake_stellar_download(database, kind)

    # default: no destination, so get_data uses $PYSYN_CDBS/grid (what users get from get_data)
    # grid: $PYSYN_CDBS/grid passed explicitly (what setup does)
    # elsewhere: downloaded somewhere else, and the models must still be moved into $PYSYN_CDBS/grid
    download_dir = {'default': grid, 'grid': grid, 'elsewhere': tmp_path / 'downloads'}[destination]
    download_dir.mkdir(exist_ok=True)
    data.get_data('stellar_grids', database,
                  final_destination_dir=None if destination == 'default' else str(download_dir),
                  keep_archive=keep_archive)

    for name in MODEL_FILES[database]:
        assert (grid / database / name).read_text() == name, f'{name} is not in $PYSYN_CDBS/grid/{database}'
    # nothing is left nested where stsynphot will not look
    assert not (grid / database / 'grp').exists()
    assert not (grid / 'grid').exists()

    if keep_archive:
        assert (download_dir / fname).exists()
    else:
        assert not (download_dir / fname).exists(), 'archive should be deleted'
        assert not (download_dir / 'grp').exists(), 'empty grp/redcat/trds/grid folders should be deleted'
        assert sorted(os.listdir(grid)) == [database]


def test_second_stellar_download_does_not_clobber_existing_grid(fake_stellar_download, tmp_path, monkeypatch, capsys):
    stellar = tmp_path / 'stellar'
    (stellar / 'grid' / 'phoenix').mkdir(parents=True)
    (stellar / 'grid' / 'phoenix' / 'my_file.fits').write_text('keep me')
    monkeypatch.setenv('PYSYN_CDBS', str(stellar))
    fake_stellar_download('phoenix', 'tar.gz')

    data.get_data('stellar_grids', 'phoenix', final_destination_dir=str(stellar / 'grid'))

    assert (stellar / 'grid' / 'phoenix' / 'my_file.fits').read_text() == 'keep me'
    assert 'manually move it' in capsys.readouterr().out


@pytest.mark.skipif(os.environ.get('PICASO_NETWORK_TESTS') != '1',
                    reason='downloads 18 MB from STScI; run with PICASO_NETWORK_TESTS=1')
def test_real_stellar_download_works_with_stsynphot(tmp_path):
    """Download the real ck04models tarball and check stsynphot can build a spectrum from where it lands."""
    stellar = tmp_path / 'stellar'
    env = {k: v for k, v in os.environ.items() if not k.startswith('XDG_')}
    env.update(HOME=str(tmp_path), picaso_refdata=str(REFDATA), PYSYN_CDBS=str(stellar))
    (stellar / 'grid').mkdir(parents=True)
    code = ("from picaso import data\n"
            "data.get_data('stellar_grids', 'ck04models', keep_archive=False)\n"
            "import stsynphot as sts\n"
            "sp = sts.grid_to_spec('ck04models', 5000, 0.0, 4.0)\n"
            "print('NPOINTS', len(sp.waveset))\n")
    result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=600)
    assert 'NPOINTS' in result.stdout, result.stdout + result.stderr
    assert (stellar / 'grid' / 'ck04models' / 'catalog.fits').exists()
    assert sorted(os.listdir(stellar / 'grid')) == ['ck04models']
