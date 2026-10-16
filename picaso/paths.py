"""
Where PICASO finds its reference data and stellar grids.

PICASO needs two folders: the reference data (``picaso_refdata``) and the
stsynphot stellar grids (``PYSYN_CDBS``, only needed for exoplanets). Each is
looked up in this order:

1. the environment variable
2. the PICASO config file written by ``picaso.data.setup`` (see ``config_file``)
3. the default data folder (see ``default_data_dir``), if data was downloaded there

``picaso/__init__.py`` calls ``apply`` before any other picaso module is imported,
so the rest of the code can keep reading ``os.environ``.
"""
import json
import os
import tomllib
import warnings

import platformdirs

REFDATA = 'picaso_refdata'
STELLAR = 'PYSYN_CDBS'


def config_file():
    """Path of the PICASO config file (it may not exist yet)."""
    return os.path.join(platformdirs.user_config_dir('picaso'), 'config.toml')


def default_data_dir():
    """Folder that ``picaso.data.setup`` downloads into when no destination is given."""
    return platformdirs.user_data_dir('picaso')


def default_refdata():
    return os.path.join(default_data_dir(), 'reference')


def read_config():
    """Contents of the config file as a dict, or {} if there is none."""
    try:
        with open(config_file(), 'rb') as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}
    except tomllib.TOMLDecodeError as e:
        warnings.warn(f'Ignoring the PICASO config file {config_file()} because it could not be read: {e}')
        return {}


def write_config(**paths):
    """
    Save paths to the config file, keeping any keys already there.

    Parameters
    ----------
    **paths : str
        e.g. picaso_refdata='/data/picaso/reference', PYSYN_CDBS='/data/picaso/reference/stellar_grids'
    """
    config = read_config()
    config.update({k: os.path.abspath(os.path.expanduser(v)) for k, v in paths.items()})
    os.makedirs(os.path.dirname(config_file()), exist_ok=True)
    with open(config_file(), 'w') as f:
        f.write('# Written by picaso.data.setup. Environment variables of the same name take precedence.\n')
        for key, value in config.items():
            # JSON string escaping is valid TOML basic-string escaping
            f.write(f'{key} = {json.dumps(value)}\n')


def resolve():
    """
    Find the reference data and stellar grid folders.

    Returns
    -------
    dict
        {'picaso_refdata': (path, source), 'PYSYN_CDBS': (path, source)}, where path is None
        if nothing was found and source says where the path came from.
    """
    config = read_config()

    if os.environ.get(REFDATA):
        refdata = (os.environ[REFDATA], 'environment variable')
    elif config.get(REFDATA):
        refdata = (config[REFDATA], f'config file {config_file()}')
    elif os.path.isfile(os.path.join(default_refdata(), 'config.json')):
        refdata = (default_refdata(), 'default data folder')
    else:
        refdata = (None, 'not found')

    bundled = os.path.join(refdata[0], 'stellar_grids') if refdata[0] else None
    if os.environ.get(STELLAR):
        stellar = (os.environ[STELLAR], 'environment variable')
    elif config.get(STELLAR):
        stellar = (config[STELLAR], f'config file {config_file()}')
    elif bundled and os.path.isdir(os.path.join(bundled, 'grid')):
        stellar = (bundled, 'stellar_grids folder in the reference data')
    else:
        stellar = (None, 'not found')

    return {REFDATA: refdata, STELLAR: stellar}


def apply():
    """Set picaso_refdata and PYSYN_CDBS for this process when they are not set but were found."""
    found = resolve()
    for name, (path, _) in found.items():
        if path and not os.environ.get(name):
            os.environ[name] = path
    return found


SETUP_HELP = (
    'Download it with one command:\n'
    '    picaso-data setup --bundle default-planet       # exoplanets: adds the PHOENIX stellar grids\n'
    '    picaso-data setup --bundle default-browndwarf   # brown dwarfs: no stellar grids\n'
    "or in Python: picaso.data.setup(bundle='default-planet'). "
)


def missing_refdata_message():
    path = os.environ.get(REFDATA)
    if path:
        found = f'picaso_refdata is set to {path}, but that folder does not exist or has no config.json.'
    else:
        found = (f'picaso_refdata is not set, the config file {config_file()} does not name a folder, '
                 f'and there is no data in the default folder {default_refdata()}.')
    return (f'PICASO could not find its reference data. {found}\n' + SETUP_HELP +
            'If you already have the data, set picaso_refdata to that folder before importing picaso. '
            'Details: https://natashabatalha.github.io/picaso/installation.html')


def missing_stellar_message(database=None):
    path = os.environ.get(STELLAR)
    if not path:
        found = 'PYSYN_CDBS is not set.'
    else:
        found = f'PYSYN_CDBS is {path}, but {os.path.join(path, "grid", database or "")} does not exist.'
        grid = os.path.join(path, 'grid')
        have = sorted(d for d in os.listdir(grid) if os.path.isdir(os.path.join(grid, d))) if os.path.isdir(grid) else []
        if have:
            found += (f" You do have {', '.join(have)}: to use one of those, pass database='{have[0]}'"
                      " (PICASO v5 changed the default from 'ck04models' to 'phoenix').")
    return (f'The {database or "stellar"} grid needed for this star was not found. {found} '
            'Brown dwarf models do not need stellar grids, exoplanet models do.\n' + SETUP_HELP +
            (f"To add only this grid: picaso.data.get_data('stellar_grids', '{database}'). " if database else '') +
            'Or pass your own spectrum with star(filename=...).')


def refdata_ok(path):
    return bool(path) and os.path.isfile(os.path.join(path, 'config.json'))


def require_refdata():
    """Return the reference data folder, or raise a FileNotFoundError that says how to get it."""
    path = os.environ.get(REFDATA)
    if not refdata_ok(path):
        raise FileNotFoundError(missing_refdata_message())
    return path
