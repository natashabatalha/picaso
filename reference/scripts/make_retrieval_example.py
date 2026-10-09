"""
Builds the small synthetic transit retrieval in reference/base_cases/retrieval_example/ that the
Retrieval Analysis page of picaso-app loads with its "Load example" button.

The example is a WASP-39b-like clear transit with four free parameters: isothermal temperature,
constant H2O and CO2 volume mixing ratios (log sampled) and planet radius. The fake data is the
model at the true parameters, binned to R=100 from 1-5 micron with 30 ppm gaussian noise.

Steps (run all, or pick with --steps):
    opacity  : tiny opacity db (H2O, CO2, H2-H2 and H2-He CIA, 0.95-5.3 um, R~1000, 400-1700 K)
               cut from the full default resampled db, which must be given with --source_db
    inputs   : stellar userfile, retrieval toml and the fake data
    retrieve : ultranest run (minutes on one core), keeping only the files picaso reads

    python make_retrieval_example.py --source_db /path/to/opacities_0.3_15_R15000.db
"""
import argparse
import io
import os
import shutil
import sqlite3

import numpy as np
import pandas as pd

import picaso.driver as go
from picaso import opacity_factory as opa_fac

REFDATA = os.environ.get('picaso_refdata', os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = os.path.join(REFDATA, 'base_cases', 'retrieval_example')
DEFAULT_DIR = '_default_/base_cases/retrieval_example'  # how the toml points at this folder

DB_NAME = 'testing_file_only_H2O_CO2_example_opacities.db'
STAR_NAME = 'star_5400K.txt'
DATA_NAME = 'transit_example.csv'
TOML_NAME = 'retrieval_example.toml'
SAMPLES_DIR = 'ultranest'

# opacity db
MOLECULES = ['H2O', 'CO2']
CONTINUUM = ['H2H2', 'H2He']
WAVE_RANGE = (0.95, 5.3)  # micron, a bit beyond the data so the edge bins are fully covered
RESAMPLE = 15  # take every 15th point of the R=15000 grid -> R=1000
TEMPERATURES = np.arange(500, 1601, 100)  # brackets the 500-1500 K temperature prior

# truths and data
TRUTH = {'temperature.isothermal.T': 900.0,
         'chemistry.free.H2O.value': 1e-3,
         'chemistry.free.CO2.value': 1e-4,
         'object.radius.value': 1.27}
DATA_RANGE = (1.0, 5.0)  # micron
DATA_R = 100
NOISE_PPM = 30
SEED = 42

TOML = f"""# Synthetic WASP-39b-like transit used as the picaso-app Retrieval Analysis example.
# Built by reference/scripts/make_retrieval_example.py. Truths: T=900 K, H2O=1e-3, CO2=1e-4, radius=1.27 Rjup.
observation_type = 'transit_depth'
irradiated = true
calc_type = 'retrieval'

[InputOutput]
retrieval_output = '{DEFAULT_DIR}/{SAMPLES_DIR}'

[ObservationData]
filenames = ['{DEFAULT_DIR}/{DATA_NAME}']
data = 'transit_depth'
data_unit = 'cm**2/cm**2'
coord = 'wavelength'
coord_unit = 'um'
error = 'error'
instruments = []

[OpticalProperties]
opacity_file = '{DEFAULT_DIR}/{DB_NAME}'
opacity_method = 'resampled'
opacity_kwargs = {{}}
virga_mieff = '_default_/virga/'

[object]
radius = {{value={TRUTH['object.radius.value']}, unit='Rjup'}}
mass = {{value=0.28, unit='Mjup'}}
gravity = {{value=430.0, unit='cm/s**2'}} # unused: mass and radius set the gravity

[geometry]
phase = {{value=0, unit='radian'}}

[star]
radius = {{value=0.939, unit='Rsun'}}
semi_major = {{value=0.0486, unit='AU'}}
type = 'userfile'

[star.userfile]
filename = '{DEFAULT_DIR}/{STAR_NAME}'
w_unit = 'um'
f_unit = 'erg/(s*cm**2*AA)'

[temperature]
profile = 'isothermal'

[temperature.pressure]
reference = {{value=1.0, unit='bar'}}
min = {{value=1e-6, unit='bar'}}
max = {{value=1e2, unit='bar'}}
nlevel = 60
spacing = 'log'

[temperature.isothermal]
T = {TRUTH['temperature.isothermal.T']}

[chemistry]
method = 'free'

[chemistry.free]
background = {{gases=['H2','He'], fraction=5.667}}
species = ['H2O', 'CO2']

[chemistry.free.H2O]
profile = 'constant'
value = {TRUTH['chemistry.free.H2O.value']}

[chemistry.free.CO2]
profile = 'constant'
value = {TRUTH['chemistry.free.CO2.value']}

[retrieval]
mpi = false
processes = 1

[retrieval.sampler]
code = 'ultranest'
resume = false
sampler_kwargs = {{}}
run_kwargs = {{min_num_live_points=200, show_status=false}}

[retrieval.temperature.isothermal.T]
prior = 'uniform'
uniform_kwargs = {{min=500.0, max=1500.0}}
log = false

[retrieval.chemistry.free.H2O.value]
prior = 'uniform'
uniform_kwargs = {{min=-8.0, max=-1.0}}
log = true

[retrieval.chemistry.free.CO2.value]
prior = 'uniform'
uniform_kwargs = {{min=-8.0, max=-1.0}}
log = true

[retrieval.object.radius.value]
prior = 'uniform'
uniform_kwargs = {{min=1.0, max=1.5}}
log = false
"""


def out(name):
    return os.path.join(OUT_DIR, name)


def load_config():
    import tomllib
    with open(out(TOML_NAME), 'rb') as f:
        return go.resolve_default_paths(tomllib.load(f), REFDATA)


def blob(arr):
    buffer = io.BytesIO()
    np.save(buffer, arr)
    return sqlite3.Binary(buffer.getvalue())


def unblob(raw):
    return np.load(io.BytesIO(raw))


def make_opacity(source_db):
    """Copies a small slice of the source db, stored as float32 to halve the size."""
    new_db = out(DB_NAME)
    if os.path.exists(new_db):
        os.remove(new_db)
    opa_fac.build_skeleton(new_db)

    src = sqlite3.connect(source_db)
    wno = unblob(src.execute('SELECT wavenumber_grid FROM header').fetchone()[0])
    wave = 1e4 / wno
    keep = np.where((wave >= WAVE_RANGE[0]) & (wave <= WAVE_RANGE[1]))[0][::RESAMPLE]

    dst = sqlite3.connect(new_db)
    dst.execute('INSERT INTO header (pressure_unit, temperature_unit, wavenumber_grid, continuum_unit, molecular_unit) '
                'VALUES (?,?,?,?,?)', ('bar', 'kelvin', blob(wno[keep]), 'cm-1 amagat-2', 'cm2/molecule'))
    temps = ','.join(str(float(t)) for t in TEMPERATURES)
    for mol in MOLECULES:
        rows = src.execute(f'SELECT ptid, molecule, pressure, temperature, opacity FROM molecular '
                           f'WHERE molecule=? AND temperature IN ({temps}) ORDER BY ptid', (mol,)).fetchall()
        dst.executemany('INSERT INTO molecular (ptid, molecule, pressure, temperature, opacity) VALUES (?,?,?,?,?)',
                        [(*row[:4], blob(unblob(row[4])[keep].astype(np.float32))) for row in rows])
    for mol in CONTINUUM:
        rows = src.execute(f'SELECT molecule, temperature, opacity FROM continuum '
                           f'WHERE molecule=? AND temperature IN ({temps}) ORDER BY temperature', (mol,)).fetchall()
        dst.executemany('INSERT INTO continuum (molecule, temperature, opacity) VALUES (?,?,?)',
                        [(*row[:2], blob(unblob(row[2])[keep].astype(np.float32))) for row in rows])
    dst.commit()
    dst.execute('VACUUM')
    dst.close()
    src.close()

    resolution = round(1 / np.median(np.abs(np.diff(np.log(wno[keep])))))
    opa_fac.add_all_metadata(new_db, 'example', False, str(resolution), str(WAVE_RANGE[0]), str(WAVE_RANGE[1]),
                             '10.5281/zenodo.14861730')
    print(f'{new_db}: {len(keep)} wavenumbers (R~{resolution}), {os.path.getsize(new_db) / 1e6:.2f} MB')


def make_inputs():
    # stellar userfile: only its radius matters for transit depth, the spectrum is a 5400 K blackbody
    wave = np.logspace(np.log10(0.3), np.log10(30), 300)
    h, c, k = 6.62607e-27, 2.99792e10, 1.380649e-16
    wcm = wave * 1e-4
    flam = 2 * np.pi * h * c ** 2 / wcm ** 5 / np.expm1(h * c / (wcm * k * 5400)) * 1e-8  # erg/s/cm2/A
    np.savetxt(out(STAR_NAME), np.column_stack([wave, flam]), fmt='%.6e')

    with open(out(TOML_NAME), 'w') as f:
        f.write(TOML)

    # R=100 bins; write a placeholder so get_data gives the model grid, then fill in the noisy model
    nbins = int(np.log(DATA_RANGE[1] / DATA_RANGE[0]) * DATA_R)
    wave = DATA_RANGE[0] * np.exp((np.arange(nbins) + 0.5) / DATA_R)
    error = np.full(nbins, NOISE_PPM * 1e-6)
    pd.DataFrame({'wavelength': wave, 'transit_depth': 0.0, 'error': error}).to_csv(out(DATA_NAME), index=False)

    config = load_config()
    fitpars = go.prior_finder(config['retrieval'])
    truth = np.array([TRUTH[key] for key in fitpars])
    model = go.check_model_samples(config, samples=np.atleast_2d(truth), full_likelihood=True)
    order = np.argsort(1e4 / np.asarray(model['xdata']))  # check_model_samples is ordered by wavenumber
    depth = np.asarray(model['ymodel'][0])[order]
    noisy = depth + np.random.default_rng(SEED).normal(0, NOISE_PPM * 1e-6, nbins)
    pd.DataFrame({'wavelength': wave, 'transit_depth': noisy, 'error': error}).to_csv(out(DATA_NAME), index=False)
    print(f'{out(DATA_NAME)}: {nbins} points, depth {depth.min() * 1e2:.3f}-{depth.max() * 1e2:.3f} %')


def run_retrieval():
    config = load_config()
    samples_dir = config['InputOutput']['retrieval_output']
    shutil.rmtree(samples_dir, ignore_errors=True)
    go.retrieve(config=config)
    # picaso only reads results/points.hdf5 and info/; drop the plots, chains and logs to keep the folder small
    for name in os.listdir(samples_dir):
        if name not in ('results', 'info'):
            path = os.path.join(samples_dir, name)
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, files in os.walk(samples_dir) for f in files)
    print(f'{samples_dir}: {size / 1e6:.2f} MB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source_db', help='full resampled opacity db to cut the example db from')
    parser.add_argument('--steps', nargs='+', default=['opacity', 'inputs', 'retrieve'],
                        choices=['opacity', 'inputs', 'retrieve'])
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if 'opacity' in args.steps:
        if not args.source_db:
            parser.error('the opacity step needs --source_db')
        make_opacity(args.source_db)
    if 'inputs' in args.steps:
        make_inputs()
    if 'retrieve' in args.steps:
        run_retrieval()
