"""
The README Quickstart is what new users (and AI assistants) copy first, so its code is run
here exactly as written: the python blocks between the quickstart markers in README.md.
"""
import re
import warnings

import matplotlib
import numpy as np
import pytest

from picaso import data
from conftest import REPO_ROOT


def _quickstart_blocks(language):
    readme = (REPO_ROOT / 'README.md').read_text()
    section = re.search(r'<!-- quickstart:start -->(.*?)<!-- quickstart:end -->', readme, re.S)
    assert section, 'README.md is missing the quickstart markers'
    return re.findall(rf'```{language}\n(.*?)```', section.group(1), re.S)


def _run(code):
    matplotlib.use('Agg')
    namespace = {}
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'FigureCanvasAgg is non-interactive')  # from plt.show()
        exec(compile(code, 'README.md', 'exec'), namespace)
    return namespace


def test_quickstart_commands_exist():
    """Bundle names and get targets in the README match what picaso-data accepts. Runs without data."""
    commands = ' '.join(_quickstart_blocks('bash')) + ' '.join(
        re.findall(r'`(picaso-data [^`]+)`', (REPO_ROOT / 'README.md').read_text()))
    for bundle in re.findall(r'--bundle (\S+)', commands):
        assert bundle in data.BUNDLES
    categories = data.get_data_config()[1]
    for category, target in re.findall(r'picaso-data get (\S+) (\S+)', commands):
        assert target in categories[category]


def test_quickstart_has_all_examples():
    assert len(_quickstart_blocks('python')) == 3


@pytest.mark.needs_full_data
def test_quickstart_transmission_runs():
    ns = _run(_quickstart_blocks('python')[0])
    wavelength, depth = 1e4 / ns['wno'], ns['depth']
    assert wavelength.min() < 0.7 and wavelength.max() > 4.9
    assert np.all(np.isfinite(depth))
    # 1.2 R_jup around 1 R_sun is (Rp/Rs)^2 ~ 1.5%, plus a few scale heights of atmosphere
    assert 0.014 < depth.min() < depth.max() < 0.02


@pytest.mark.needs_full_data
def test_quickstart_brown_dwarf_runs():
    ns = _run(_quickstart_blocks('python')[1])
    wavelength, fnu = 1e4 / ns['wno'], ns['fnu']
    assert wavelength.min() < 1.1 and wavelength.max() > 4.9
    assert np.all(np.isfinite(fnu)) and np.all(fnu > 0)


@pytest.mark.needs_full_data
def test_quickstart_terrestrial_reflected_runs():
    ns = _run(_quickstart_blocks('python')[2])
    wavelength, albedo, fpfs = 1e4 / ns['wno'], ns['albedo'], ns['fpfs']
    assert wavelength.min() < 0.31 and wavelength.max() > 0.99
    assert np.all(np.isfinite(albedo)) and 0 < albedo.min() < albedo.max() < 1
    # albedo * (R_earth / 1 au)^2 is at most ~1.8e-9
    assert 0 < fpfs.min() < fpfs.max() < 1.8e-9
