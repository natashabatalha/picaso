import glob
import json
import os
from pathlib import Path

import pytest

# Point picaso at the reference data in this checkout. This must happen before picaso is
# imported, which is why it lives at the top of conftest.py rather than in a fixture.
# The stellar grids are too big for git, so an existing PYSYN_CDBS is kept.
REPO_ROOT = Path(__file__).resolve().parents[1]
REFDATA = REPO_ROOT / 'reference'
os.environ['picaso_refdata'] = str(REFDATA)
os.environ.setdefault('PYSYN_CDBS', str(REFDATA / 'stellar_grids'))

# Mini opacity db (H2O, CO2, H2-H2 and H2-He CIA, 0.95-5.3 um, R~1000, 500-1600 K) that is
# small enough to live in git, so tests that use it can run on GitHub Actions.
# Built by reference/scripts/make_retrieval_example.py.
RETRIEVAL_EXAMPLE_DIR = REFDATA / 'base_cases' / 'retrieval_example'
MINI_OPACITY_DB = RETRIEVAL_EXAMPLE_DIR / 'testing_file_only_H2O_CO2_example_opacities.db'


def full_data_available():
    """
    True if the data that is too big for git has been downloaded: the default (multi-GB)
    resampled opacity db in reference/opacities/ and the ck04models stellar grids in PYSYN_CDBS.
    """
    with open(REFDATA / 'config.json') as f:
        pattern = json.load(f)['opacities']['files']['opacity']
    has_opacities = len(glob.glob(str(REFDATA / pattern))) > 0
    has_stars = os.path.isdir(os.path.join(os.environ['PYSYN_CDBS'], 'grid', 'ck04models'))
    return has_opacities and has_stars


def pytest_collection_modifyitems(config, items):
    if full_data_available():
        return
    skip = pytest.mark.skip(reason='needs the full opacity db and stellar grids (not in git)')
    for item in items:
        if 'needs_full_data' in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope='session')
def mini_opacity():
    """opannection to the mini opacity db, opened once and shared by every test in the run."""
    from picaso import justdoit as jdi
    return jdi.opannection(filename_db=str(MINI_OPACITY_DB))


@pytest.fixture(scope='session')
def retrieval_example_config():
    """The retrieval example toml with its _default_ paths resolved against this checkout."""
    import tomllib
    import picaso.driver as go
    with open(RETRIEVAL_EXAMPLE_DIR / 'retrieval_example.toml', 'rb') as f:
        return go.resolve_default_paths(tomllib.load(f), str(REFDATA))
