"""
Loading per-molecule correlated-k tables (method='resortrebin'). PICASO v5 reads only the HDF5 tables
({molecule}_1460.hdf5). A folder with only the old .npy tables raises an error that says what to
download, instead of the v4 warning. Uses the mini H2O/CO2 HDF5 tables in git.
"""
import shutil

import numpy as np
import pytest

from conftest import MINI_CK_DIR
from picaso.opacity_factory import get_ck_tables


def test_npy_only_tables_raise_and_say_what_to_download(tmp_path):
    np.save(tmp_path / 'H2O_1460.npy', np.zeros((2, 2, 2, 2)))
    with pytest.raises(FileNotFoundError) as err:
        get_ck_tables(str(tmp_path), preload_gases=['H2O'])
    message = str(err.value)
    assert 'H2O' in message and '.npy' in message
    assert 'picaso-data get ck_tables by-molecule' in message
    assert "get_data('ck_tables', 'by-molecule')" in message


def test_hdf5_tables_load_even_when_old_npy_files_are_also_there(tmp_path):
    for f in MINI_CK_DIR.glob('*_1460.hdf5'):
        shutil.copy(f, tmp_path)
    np.save(tmp_path / 'H2O_1460.npy', np.zeros((2, 2, 2, 2)))
    output = get_ck_tables(str(tmp_path), preload_gases=['H2O', 'CO2'])
    assert sorted(output['molecules']) == ['CO2', 'H2O']


def test_a_molecule_with_only_npy_fails_even_if_others_have_hdf5(tmp_path):
    shutil.copy(MINI_CK_DIR / 'H2O_1460.hdf5', tmp_path)
    np.save(tmp_path / 'CH4_1460.npy', np.zeros((2, 2, 2, 2)))
    with pytest.raises(FileNotFoundError, match='CH4'):
        get_ck_tables(str(tmp_path), preload_gases=['H2O', 'CH4'])
