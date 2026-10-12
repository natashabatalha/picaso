"""
Builds the mini correlated-k tables in reference/base_cases/testing/resortrebin_mini/ that let the
climate (method='resortrebin') unit tests run on GitHub Actions without the 2 GB of full tables.

Each table is cut from the full per-molecule table in the same folder format picaso reads
(<molecule>_1460.hdf5; the name is kept so the loader finds it even though the grid is smaller):
    molecules    : H2O, CO2 (the same two as the mini resampled db)
    pressures    : all 20 (1e-6 to 3000 bar)
    temperatures : 500-1600 K (19 of the 73 grid temperatures)
    wavenumbers  : all 661, which must match the climate continuum db
    gauss points : all 8
k-coefficients are rounded to float32 precision but stored as float64 (picaso's numba mixing needs
float64), so with shuffle + gzip the zeroed low bytes compress away: ~10 MB for both files.
These tables are for testing only, not for science.

    python make_mini_resortrebin.py --source_dir /path/to/reference/opacities/resortrebin
"""
import argparse
import os

import h5py
import numpy as np

REFDATA = os.environ.get('picaso_refdata', os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = os.path.join(REFDATA, 'base_cases', 'testing', 'resortrebin_mini')

MOLECULES = ['H2O', 'CO2']
TEMPERATURE_RANGE = (500, 1600)  # K, inclusive


def make_table(source_dir, molecule):
    source = os.path.join(source_dir, f'{molecule}_1460.hdf5')
    out = os.path.join(OUT_DIR, f'{molecule}_1460.hdf5')
    with h5py.File(source, 'r') as src, h5py.File(out, 'w') as dst:
        temps = np.unique(src['temperatures'][:])
        keep = np.where((temps >= TEMPERATURE_RANGE[0]) & (temps <= TEMPERATURE_RANGE[1]))[0]
        # pressures and temperatures are the flattened (P, T) pairs of the grid
        pairs = np.isin(src['temperatures'][:], temps[keep])

        for name in ['wno', 'delta_wno', 'gauss_pts', 'gauss_wts']:
            dst.create_dataset(name, data=src[name][:])
        dst.create_dataset('pressures', data=src['pressures'][:][pairs])
        dst.create_dataset('temperatures', data=src['temperatures'][:][pairs])
        dst.create_dataset('nc_p', data=src['nc_p'][:][keep])
        # axes are [pressure, temperature, wavenumber, gauss point], values are ln(k)
        kcoeffs = src['kcoeffs'][:, keep].astype(np.float32).astype(np.float64)
        dst.create_dataset('kcoeffs', data=kcoeffs,
                           compression='gzip', compression_opts=9, shuffle=True)
    print(f'{out}: {len(keep)} temperatures, {os.path.getsize(out) / 1e6:.2f} MB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source_dir', required=True, help='folder with the full <molecule>_1460.hdf5 tables')
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    for molecule in MOLECULES:
        make_table(args.source_dir, molecule)
