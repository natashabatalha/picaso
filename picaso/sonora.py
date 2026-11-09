"""
Reading the Sonora brown dwarf and giant planet model grids: Bobcat (Marley et al. 2021), Diamondback
(Morley et al. 2024), Elf Owl (Mukherjee et al. 2024) and Flame Skimmer (Mang et al. 2026).

Download the grids with ``picaso.data.get_data('sonora_grids', ...)`` and load a model into PICASO with
``justdoit.inputs.sonora``. This module only reads files (no PICASO reference data is needed), so it is
also used by the Sonora viewer in picaso-app.
"""
import gzip
import os
import re

import numpy as np
import pandas as pd
import xarray as xr

C_CGS = 2.99792458e10 #speed of light, cm/s

SONORA_GRIDS = ['bobcat', 'diamondback', 'elfowl', 'flameskimmer']

_num = r'[-+]?\d+(?:\.\d+)?'
_SONORA_PATTERNS = {
    #t1300g3160f4_m-0.5_co1.0.pt (fsed=nc is cloud-free)
    'diamondback': re.compile(rf'^t(?P<teff>{_num})g(?P<grav>{_num})(?P<fsed>nc|f{_num})_m(?P<log_mh>{_num})_co(?P<cto_relative>{_num})\.pt$'),
    #spectra_logzz_2.0_teff_275.0_grav_1000.0_mh_0.0_co_0.5.nc
    'elfowl': re.compile(rf'^spectra_logzz_(?P<logkzz>{_num})_teff_(?P<teff>{_num})_grav_(?P<grav>{_num})_mh_(?P<log_mh>{_num})_co_(?P<cto_relative>{_num})\.nc$'),
    #spectra_logkzz_2_teff_425_grav_100_mh_+0.0_co_0.5_R30000.nc (no logkzz for chemical equilibrium)
    'flameskimmer': re.compile(rf'^spectra_(?:logkzz_(?P<logkzz>{_num})_)?teff_(?P<teff>{_num})_grav_(?P<grav>{_num})_mh_(?P<log_mh>{_num})_co_(?P<cto_relative>{_num})_R(?P<R>\d+)\.nc$'),
    #t1000g100nc_m0.0.cmp.gz or t200g10nc_m+0.0.dat
    'bobcat': re.compile(rf'^t(?P<teff>{_num})g(?P<grav>{_num})nc_m(?P<log_mh>{_num})[^/]*\.(?:cmp\.gz|dat)$'),
}

def _list_sonora_files(sonora_path, grid):
    if not os.path.isdir(sonora_path):
        raise FileNotFoundError(f'Sonora path {sonora_path} is not a directory')
    if grid == 'bobcat': #bobcat is read from the top directory only
        return [os.path.join(sonora_path, i) for i in os.listdir(sonora_path) if not i.startswith('._')]
    out = []
    for root, _, files in os.walk(sonora_path):
        out += [os.path.join(root, i) for i in files if not i.startswith('._')]
    return out

def _detect_sonora_grid(sonora_path):
    found = []
    files = [os.path.basename(i) for i in _list_sonora_files(sonora_path, 'all')]
    for grid in ['diamondback', 'elfowl', 'flameskimmer']:
        if any(_SONORA_PATTERNS[grid].match(i) for i in files): found.append(grid)
    if any(i.endswith(('.cmp.gz', '.dat')) for i in os.listdir(sonora_path)): found.append('bobcat')
    if len(found) == 1: return found[0]
    if len(found) == 0:
        raise Exception(f'Could not find any Sonora grid files in {sonora_path}. Download a grid with '
            "picaso.data.get_data('sonora_grids') and point sonora_path to the unpacked files.")
    raise Exception(f'Found files from several Sonora grids {found} in {sonora_path}. Pass grid= to pick one.')

def sonora_grid_options(sonora_path, grid=None):
    """
    Lists the Sonora grid models found in a directory. 

    Parameters
    ----------
    sonora_path : str 
        Directory with the unpacked grid files (subdirectories are searched for all grids but Bobcat)
    grid : str 
        'bobcat', 'diamondback', 'elfowl' or 'flameskimmer'. Default None detects it from the files

    Returns
    -------
    pandas.DataFrame
        One row per model with teff (K), grav (m/s2), log_mh, cto_relative (relative to solar=0.458),
        logkzz (nan for chemical equilibrium), fsed (nan for cloud-free) and filename
    """
    if grid is None: grid = _detect_sonora_grid(sonora_path)
    grid = grid.lower().replace(' ','').replace('_','')
    pattern = _SONORA_PATTERNS[grid]
    rows = []
    for f in _list_sonora_files(sonora_path, grid):
        m = pattern.match(os.path.basename(f))
        if m is None: continue
        d = m.groupdict()
        fsed = d.get('fsed')
        rows.append({'teff':float(d['teff']), 'grav':float(d['grav']),
                     'log_mh':float(d['log_mh']),
                     'cto_relative':float(d['cto_relative']) if d.get('cto_relative') else 1.0,
                     'logkzz':float(d['logkzz']) if d.get('logkzz') else np.nan,
                     'fsed':np.nan if fsed in (None, 'nc') else float(fsed[1:]),
                     'R':float(d['R']) if d.get('R') else np.nan,
                     'filename':f})
    cols = ['teff','grav','log_mh','cto_relative','logkzz','fsed','filename']
    if len(rows) == 0:
        raise Exception(f'No {grid} files found in {sonora_path}')
    df = pd.DataFrame(rows)
    #flame skimmer quickstart (R=3000) and full (R=30000) files hold the same model, keep the highest resolution
    df = df.sort_values('R', ascending=False).drop_duplicates(cols[:-1])
    return df[cols].sort_values(cols[:-1]).reset_index(drop=True)

def _nearest_sonora_model(options, grid, teff, grav, log_mh=0.0, cto_relative=1.0, logkzz=None, fsed=None, verbose=True):
    """Picks the nearest model one parameter at a time: teff, log gravity, M/H, C/O, Kzz, fsed"""
    df = options
    if grid == 'elfowl' and logkzz is None:
        raise ValueError(f'Elf Owl only has disequilibrium chemistry models, so logkzz must be given. Options: {sorted(df["logkzz"].unique())}')
    #categorical choices first: chemical equilibrium (no kzz) and cloud-free (no fsed)
    df = df[df['logkzz'].isna()] if logkzz is None else df[df['logkzz'].notna()]
    df = df[df['fsed'].isna()] if fsed is None else df[df['fsed'].notna()]
    if len(df) == 0:
        raise Exception(f"No {grid} models with logkzz={logkzz} and fsed={fsed} were found. "
                        "Use sonora_grid_options to see the models on disk.")
    request = {'teff':teff, 'grav':grav, 'log_mh':log_mh, 'cto_relative':cto_relative, 'logkzz':logkzz, 'fsed':fsed}
    for key in ['teff', 'grav', 'log_mh', 'cto_relative', 'logkzz', 'fsed']:
        if request[key] is None: continue
        vals = df[key].values
        dist = np.abs(np.log10(vals) - np.log10(request[key])) if key == 'grav' else np.abs(vals - request[key])
        df = df[dist == dist.min()]
    row = df.iloc[0]
    if verbose:
        off = {k:(request[k], row[k]) for k in request if (request[k] is not None) and (not np.isclose(row[k], request[k]))}
        if len(off) > 0:
            print(f'Sonora {grid}: no exact match, using nearest model {os.path.basename(row["filename"])}. '
                  + ', '.join(f'{k}: asked {a}, using {b}' for k, (a, b) in off.items()))
    return row

def read_diamondback_pt(filename):
    """
    Reads a Sonora Diamondback .pt file (header of species names, then gravity, then one row per level). 

    Returns
    -------
    pandas.DataFrame with pressure (bars), temperature (K) and volume mixing ratios
    """
    with open(filename) as f:
        names = f.readline().split()
    df = pd.read_csv(filename, sep=r'\s+', skiprows=2, header=None, names=names)
    return df.drop(columns='n').rename(columns={'P':'pressure', 'T':'temperature'})

def read_diamondback_cld(filename):
    """
    Reads a Sonora Diamondback .cld file (eddysed format, 196 wavenumber points per layer) into the 
    opd, g0, w0 columns that `inputs.clouds(df=)` expects
    """
    df = pd.read_csv(filename, sep=r'\s+')
    return df.rename(columns={'tau':'opd'})[['opd','g0','w0']]

def find_sonora_spectrum(model_file, sonora_path=None):
    """
    Finds the published spectrum of a Sonora grid model. Elf Owl and Flame Skimmer save it in the 
    model file itself; Diamondback (`<model>.spec`, from the diamondback-spectra download) and 
    Bobcat (`sp_<model>` or `sp_<model>.gz`, from the bobcat-spectra download) save it in separate files, searched for in sonora_path (default: the 
    folder above the model file) and its subfolders. 

    Returns
    -------
    str or None 
        Filename of the published spectrum, or None if it was not found
    """
    base = os.path.basename(model_file)
    if base.endswith('.nc'):
        return model_file
    if base.endswith('.pt'):
        target = base[:-len('.pt')] + '.spec'
    else: #bobcat
        target = 'sp_' + re.sub(r'\.(cmp\.gz|dat)$', '', base)
    if sonora_path is None:
        sonora_path = os.path.dirname(os.path.dirname(os.path.abspath(model_file)))
    for root, _, files in os.walk(sonora_path):
        for name in (target, target + '.gz'):
            if name in files: return os.path.join(root, name)
    return None

def read_sonora_spectrum(filename):
    """
    Reads a published Sonora spectrum: Elf Owl or Flame Skimmer xarray files, Diamondback .spec files
    (W/m2/m) or Bobcat sp_ files (erg/cm2/s/Hz). 

    Returns
    -------
    wavelength, flux 
        Wavelength (micron, increasing) and flux at the top of the atmosphere in erg/cm2/s/cm, the units 
        of the PICASO `thermal` output. Multiply by (R/D)^2 to compare with an observed flux.
    """
    if filename.endswith('.nc'):
        with xr.open_dataset(filename) as ds:
            flux_name = 'flux_emission' if 'flux_emission' in ds else 'flux'
            wave, flux = ds['wavelength'].values, ds[flux_name].values
    else:
        opener = gzip.open if filename.endswith('.gz') else open
        with opener(filename, 'rt') as f:
            header = f.readline() + f.readline()
        if 'W/m2/m' in header: #diamondback
            wave, flux = np.loadtxt(filename, skiprows=3, unpack=True)
            flux = flux * 10 #W/m2/m to erg/cm2/s/cm
        elif 'erg/cm^2/s/Hz' in header: #bobcat
            wave, flux = np.loadtxt(filename, skiprows=2, unpack=True)
            flux = flux * C_CGS / (wave*1e-4)**2 #F_nu to F_lambda
        else:
            raise Exception(f'Did not recognize the Sonora spectrum format of {filename}')
    order = np.argsort(wave)
    return wave[order], flux[order]

def find_sonora_cloud(model_file, sonora_path=None):
    """
    Finds the cloud optical properties (.cld) of a cloudy Diamondback model, searched for next to the 
    model file and in sonora_path (default: the folder above the model file) and its subfolders. 
    Returns None for cloud-free models or if the file was not downloaded. 
    """
    base = os.path.basename(model_file)
    if (not base.endswith('.pt')) or ('nc_m' in base):
        return None
    target = base[:-len('.pt')] + '.cld'
    if os.path.exists(os.path.join(os.path.dirname(model_file), target)):
        return os.path.join(os.path.dirname(model_file), target)
    if sonora_path is None:
        sonora_path = os.path.dirname(os.path.dirname(os.path.abspath(model_file)))
    for root, _, files in os.walk(sonora_path):
        if target in files: return os.path.join(root, target)
    return None

def read_sonora_model(filename):
    """
    Reads the atmosphere of any Sonora grid model file. 

    Returns
    -------
    pandas.DataFrame 
        pressure (bars, increasing), temperature (K) and the volume mixing ratios saved in the file 
        (Bobcat .dat structure files only have pressure and temperature)
    """
    base = os.path.basename(filename)
    if base.endswith('.nc'):
        with xr.open_dataset(filename) as ds:
            cols = [i for i in ds.data_vars if ds[i].dims == ('pressure',)]
            df = ds[cols].to_dataframe().reset_index()
    elif base.endswith('.pt'):
        df = read_diamondback_pt(filename)
    elif base.endswith('.cmp.gz'):
        df = pd.read_csv(filename, sep=r'\s+', compression='gzip')
        df = df.rename(columns={'P(BARS)':'pressure', 'TEMP':'temperature', 'HE':'He'}).drop(columns=['LVL', 'MU'], errors='ignore')
    elif base.endswith('.dat'):
        pressure, temperature = np.loadtxt(filename, usecols=[1, 2], unpack=True, skiprows=1)
        df = pd.DataFrame({'pressure':pressure, 'temperature':temperature})
    else:
        raise Exception(f'Did not recognize the Sonora model file {filename}')
    return df.sort_values('pressure').reset_index(drop=True)
