import numpy as np
import pandas as pd
import astropy.units as u
import pickle
import os, sys
from scipy import interpolate
import virga.justdoit as vd
import virga.justplotit as vp
import time

from picaso import build_3d_input as threed
from picaso import disco
from scipy.spatial import cKDTree

import linecache
import matplotlib
import matplotlib.pyplot as plt
import math
import xarray as xr

import astropy.units as u

from picaso import justdoit as jdi
from picaso import justplotit as jpi

import sys

from collections import defaultdict
from operator import itemgetter

from picaso.disco import get_angles_3d

def regrid_and_chem_3D(planet=None, input_file=None, orb_phase=None, time_after_pa=None,
                       n_gauss_angles=None, n_chebychev_angles=None, nlon=None, nlat=None, nz=None, CtoO=None, mh=None):
    '''
    Function to regrid MITgcm input data into the chosen low resolution grid in PICASO. Selects the visiblle hemisphere
    at each point in the orbit by rotating the grid. It computes the 3D chemistry for the full planet.

    Parameters
    ----------
    planet : str
        Name of planet
    input_file : str
        input MITgcm file, see necessary format at https://natashabatalha.github.io/picaso/notebooks/9_Adding3DFunctionality.htm
    chempath : str
        Path to store chemistry files
    newpt_path : str
        Path to store regridded PTK data
    orb_phase : float
        orbital phase angle (Earth facing longitude), in degrees from -180 to 180, for planet rotation
    time_after_pa : float
        Time after periapse, for eccentric planets
    n_gauss_angles : int
        number of Gauss angles
    n_chebychev_angles : int
        number of Chebyshev angles
    nlon : int
        number of longitude points in MITgcm grid
    nlat : int
        number of latitude points in MITgcm grid
    nz : int
        number of pressure layers in MITgcm grid
    CtoO : float
        C to O ratio
    mh : int
        metallicity, in NON log units (1 is solar)

    Returns
    -------
    newdat : dict
        Dictionary with regridded pressure-temperature-kzz profiles
    chem3d : dict
        Dictionary with computed chemistry, to be added into 3D atmosphere
    '''

    if time_after_pa != None:  # for eccentric planets
        infile = open(input_file, 'r')

        # skip headers and blank line -- see GCM output
        line0 = infile.readline().split()
        line1 = infile.readline().split()

    else:
        infile = open(input_file, 'r')

    gangle, gweight, tangle, tweight = disco.get_angles_3d(n_gauss_angles, n_chebychev_angles)
    ubar0, ubar1, cos_theta, latitude, longitude = disco.compute_disco(n_gauss_angles, n_chebychev_angles, gangle, tangle, phase_angle=0)

    all_lon = np.zeros(nlon * nlat)  # 128 x 64 -- first, check size of GCM output
    all_lat = np.zeros(nlon * nlat)
    p = np.zeros(nz)
    t = np.zeros((nlon, nlat, nz))
    kzz = np.zeros((nlon, nlat, nz))

    total_pts = nlon * nlat

    ctr = -1

    for ilon in range(0, nlon):
        for ilat in range(0, nlat):
            ctr += 1

            # skip blank line -- check GCM output formatting

            temp = infile.readline().split()

            if planet == 'wasp-43b':
                all_lon[ctr] = float(temp[0]) + orb_phase #- 45  # this is hard coded for rotated WASP-43b grid from TK, remove later
            else:
                all_lon[ctr] = float(temp[0]) + orb_phase

            all_lat[ctr] = float(temp[1])

            # read in data for each grid point
            for iz in range(0, nz):
                temp = infile.readline().split()
                # print(temp)
                p[iz] = float(temp[0])

                t[ilon, ilat, iz] = float(temp[1])

                # print(t)

                kzz[ilon, ilat, iz] = float(temp[2])
            temp = infile.readline()

    lon = np.unique(all_lon)
    lat = np.unique(all_lat)

    # REGRID PTK

    lon2d, lat2d = np.meshgrid(longitude, latitude)
    lon2d = lon2d.flatten() * 180 / np.pi
    lat2d = lat2d.flatten() * 180 / np.pi

    xs, ys, zs = jpi.lon_lat_to_cartesian(np.radians(all_lon), np.radians(all_lat))
    xt, yt, zt = jpi.lon_lat_to_cartesian(np.radians(lon2d), np.radians(lat2d))

    nn = int(total_pts / (n_gauss_angles * n_chebychev_angles))

    tree = cKDTree(list(zip(xs, ys, zs)))              # these are the original 128x64 angles
    d, inds = tree.query(list(zip(xt, yt, zt)), k=nn)  # this grid is for ngxnt angles, regridding done here

    new_t = np.zeros((n_gauss_angles * n_chebychev_angles, nz))
    new_kzz = np.zeros((n_gauss_angles * n_chebychev_angles, nz))

    for iz in range(0, nz):
        new_t[:, iz] = np.sum(t[:, :, iz].flatten()[inds], axis=1) / nn
        new_kzz[:, iz] = np.sum(kzz[:, :, iz].flatten()[inds], axis=1) / nn

    newdat = {'lon': lon2d, 'lat': lat2d, 'temp': new_t, 'P': p, 'kzz': new_kzz}

    print('T, Kzz ready')

    # START CHEMISTRY

    input3d = {i: {} for i in latitude}

    print('starting chem')
    for ilat in range(n_chebychev_angles):
        for ilon in range(n_gauss_angles):
            case1 = jdi.inputs(chemeq=True)
            df = pd.DataFrame({'temperature': new_t[ilat * n_chebychev_angles + ilon, :], 'pressure': p})
            case1.inputs['atmosphere']['profile'] = df
            case1.chemeq(CtoO, mh)

            # Save as df within dictionary
            df2 = case1.inputs['atmosphere']['profile']
            df2['kzz'] = new_kzz[ilat * n_chebychev_angles + ilon, :]

            input3d[latitude[ilat]][longitude[ilon]] = df2

    print('chem ready')

    return (newdat, input3d)


# CLOUDS WITH VIRGA

def virga_calc(optics_dir=None, mmw=None, fsed=None, radius=None, mass=None, p=None, t=None, kz=None, metallicity=1):

    '''

    Compute 1D cloud profile and optical properties given a 1D pressure temperature ad kzz profile

    Parameters
    ----------

    optics_dir : directory for virga optical properties files
    mmw : float
        mean molecular weight of atm
    fsed : float
        sedimentation efficiency
    radius : float
        radius of planet in Rjup
    mass :  float
        mass of planet in Mjup
    p : ndarray
        pressure (1D)
    t : ndarray
        temperature (1D)
    kz : ndarray
        mixing coefficient (1D)
    metallicity : int
        metallicity in NON log units, default 1 (solar)

    Returns
    -------
    all_out : dict
        1D cloud profile and optical properties
    '''

    mean_molecular_weight = mmw
    gases = vd.recommend_gas(p, t, metallicity, mean_molecular_weight)

    if 'KCl' in gases:
        gases.remove('KCl')
    print('Virga Gases:', gases)

    sum_planet = vd.Atmosphere(gases, fsed=fsed, mh=metallicity, mmw=mean_molecular_weight)
    sum_planet.gravity(radius=radius, radius_unit=u.Unit('R_jup'), mass=mass, mass_unit=u.Unit('M_jup'))
    sum_planet.ptk(df=pd.DataFrame({'pressure': p, 'temperature': t, 'kz': kz}))  # will add to this dict from MITgcm file

    all_out = sum_planet.compute(as_dict=True, directory=optics_dir)
    return (all_out)


def virga_3D(optics_dir=None, mmw=None, fsed=None, radius=None, mass=None, ptk_dat=None):
    '''

    This function runs Virga for a 3D atmosphere, by computing cloud profiles for every 1D column in the atmosphere.
    It formats the result to be compatible with picaso, and outputs a dictionary with all the necessary cloud
    parameters in 3D. Written by Danica Adams (Caltech)

    Parameters
    ----------

    optics_dir : str
        directory for Virga optical properties files
    mmw : float
        mean molecular weight
    fsed : float
        sedimentation efficiency
    radius : float
        planet radius in Jupiter radius
    mass : float
        planet mass in Jupiter mass
    ptk_dat : dictionary
        input PTK profile database or dict, use regridded by PICASO.

    Returns
    -------
    cld3d : dict
        3D dictionary with cloud properties that can be added to 3D atmosphere in PICASO
    '''

    shortdat = ptk_dat

    # CREATE EMPTY DICTIONARY
    cld3d = {i: {} for i in np.unique(shortdat['lat'])}  # dict for each lat, all lons,

    for ilat in np.unique(shortdat['lat']):

        #for each lon/lat, will take outputs from virga run

        cld3d[ilat] = {i: {} for i in np.unique(shortdat['lon'])}  # and make a 3D dictionary, cld3d, with them

    # VIRGA CALCULATIONS

    for ipos in range(0, len(shortdat['lat'])):
        outdat = virga_calc(optics_dir, mmw, fsed, radius, mass, shortdat['P'], shortdat['temp'][ipos, :],
                            shortdat['kzz'][ipos, :])
        cld3d[shortdat['lat'][ipos]][shortdat['lon'][ipos]] = outdat  # add results from virga run to dict
        cld3d[shortdat['lat'][ipos]][shortdat['lon'][ipos]]['g0'] = cld3d[shortdat['lat'][ipos]][
            shortdat['lon'][ipos]].pop('asymmetry')
        cld3d[shortdat['lat'][ipos]][shortdat['lon'][ipos]]['opd'] = cld3d[shortdat['lat'][ipos]][
            shortdat['lon'][ipos]].pop('opd_per_layer')
        cld3d[shortdat['lat'][ipos]][shortdat['lon'][ipos]]['w0'] = cld3d[shortdat['lat'][ipos]][
            shortdat['lon'][ipos]].pop('single_scattering')
        cld3d[shortdat['lat'][ipos]][shortdat['lon'][ipos]]['wavenumber'] = cld3d[shortdat['lat'][ipos]][
            shortdat['lon'][ipos]].pop('wave')

    return (cld3d)


# THERMAL SPECTRUM

def spectrum_3D(calc=None, res=None, ng=None, nt=None, waverange=None,
                radius=None, mass=None, s_radius=None, sma=None, chemdata=None, starfile=None,
                s_w_units=None, s_f_units=None, logg=None, s_temp=None, s_met=None, clouds_dat=None, fsed=None):
    '''
    Parameters
    ----------

    planet : str
        name of planet
    orb_phase : float
        orbital phase determining visible hemisphere
    calc : str
        thermal or reflected
    time_after_pa : float
        time after periapse, for eccentric planets
    res : int
        resample opacities for higher speed, default is 0.1
    ng : int
        number of gauss angles
    nt : int
        number of cheby angles
    waverange : list of floats
        wavelength range, must be [wv1,wv2]
    radius : float
        radius of planet in units Rjup
    mass : float
        mass of planet in units Mjup
    s_temp : float
        temperature of star
    logg : float
        log g of star
    s_met : float
        metallicity of star
    sma : float
        semi-major axis
    chempath : str
        path to 3D chemistry file, if reusing chemistry for increased speed
    outpath : str
        path to output spectrum, saving disabled for now
    outfile : str
        name of output file where spectrum is stored, saving disabled for now
    save : bool
        if True, save spectrum to file, disabled for now
    starfile : str
        path to input stellar file (if you want to use your own), units must be erg/cm2/s/Hz and um
    chemdata : dict
        dictionary containing chemistry, if None, use file containing chemistry
    clouds: bool
        if True, adds Virga clouds to atmosphere, names of Virga files depend on phase and is hardcoded here
    fsed : float
        sedimentation efficiency for clouds

    Returns
    -------
    out: dict
        1D planet spectrum and 3d thermal flux
    '''

    t1 = time.time()

    # Initialize dictionary where spectral data will be stored
    star_dict = defaultdict(dict)

    opacity = jdi.opannection(wave_range=waverange, resample=res)  # change for each run depending on instrument/spectrum type

    # Planet properties, star properties

    start_case = jdi.inputs()
    start_case.phase_angle(phase=0, num_tangle=nt, num_gangle=ng)  # radians
    start_case.gravity(radius=radius, radius_unit=u.Unit('R_jup'), mass=mass, mass_unit=u.Unit('M_jup'))  # any astropy units available

    if starfile != None:
        start_case.star(opacity, radius=s_radius, radius_unit=u.Unit('R_sun'), semi_major=sma,
                        semi_major_unit=u.Unit('au'), database=None, filename=starfile, w_unit=s_w_units,
                        f_unit=s_f_units)
        print('Using input stellar file')
    else:
        print('Using CK04 stellar model database')
        start_case.star(opacity, s_temp, s_met, logg, radius=s_radius, radius_unit=u.Unit('R_sun'), semi_major=sma,
                        semi_major_unit=u.Unit('au'))  # opacity db, pysynphot database, temp, metallicity, logg

    # Create stellar flux dictionary and file

    star_flux = start_case.inputs['star']['flux']
    star_wno = start_case.inputs['star']['wno']

    star_dict['flux'] = star_flux
    star_dict['wno'] = star_wno
    
    # Add rebinned PTK

    df = chemdata

    # This makes sure that all lons and lats are on the same grid

    lats = sorted([int(i * 180 / np.pi) for i in start_case.inputs['disco']['latitude']])
    lons = sorted([int(i * 180 / np.pi) for i in start_case.inputs['disco']['longitude']])

    new_df = {}
    k = list(df.keys())
    for i in range(0, len(k)):
        if type(k[i]) == str:
            del df[k[i]]

    for i, new_la in zip(sorted(list(df.keys())), lats):
        new_df[new_la] = {}
        for j, new_lo in zip(sorted(list(df[i].keys())), lons):
            new_df[new_la][new_lo] = df[i][j]

    start_case.atmosphere_3d(dictionary=new_df)

    # Add Virga clouds to atmosphere

    if clouds_dat != None:

        df = clouds_dat

        lats = sorted([int(i * 180 / np.pi) for i in start_case.inputs['disco']['latitude']])
        lons = sorted([int(i * 180 / np.pi) for i in start_case.inputs['disco']['longitude']])

        new_df = {}
        for i, new_la in zip(sorted(list(df.keys())), lats):
            new_df[new_la] = {}
            # print(new_df)
            for j, new_lo in zip(sorted(list(df[i].keys())), lons):
                dicttemp = vd.picaso_format(df[i][j]['opd'], df[i][j]['g0'], df[i][j]['g0'])
                new_df[new_la][new_lo] = dicttemp

        start_case.clouds_3d(dictionary=new_df)

    # Compute spectrum

    print('Computing spectrum')

    out = start_case.spectrum(opacity, dimension='3d', calculation=calc, full_output=True)

    print('Spectrum done')

    t2 = time.time()

    return (out, star_dict)


def correct_star_units(star_model, w_unit, f_unit):

    '''
    Correct from stellar units of original input file into PICASO units of ergs/cm3/s without regridding

    Parameters
    ----------
    star_model : str
        path to input stellar file
    w_unit : str
        units of wavelength in input stellar file
    f_units : str
        units of flux in input stellar file

    Returns
    -------
    wv_star : ndarray
        Stellar wavelength in corrected units
    flux_star : ndarray
        Stellar flux in corrected units
    '''

    # Correct from stellar units of original file into PICASO units of ergs/cm3/s

    star = np.genfromtxt(star_model, dtype=(float, float), names='w, f')
    flux = star['f']
    wave = star['w']

    # sort if not in ascending order
    sort = np.array([wave, flux]).T
    sort = sort[sort[:, 0].argsort()]
    wave = sort[:, 0]
    flux = sort[:, 1]
    if w_unit == 'um':
        WAVEUNITS = 'um'
    elif w_unit == 'nm':
        WAVEUNITS = 'nm'
    elif w_unit == 'cm':
        WAVEUNITS = 'cm'
    elif w_unit == 'Angs':
        WAVEUNITS = 'angstrom'
    elif w_unit == 'Hz':
        WAVEUNITS = 'Hz'
    else:
        raise Exception('Stellar units are not correct. Pick um, nm, cm, hz, or Angs')

        # http://www.gemini.edu/sciops/instruments/integration-time-calculators/itc-help/source-definition
    if f_unit == 'Jy':
        FLUXUNITS = 'jy'
    elif f_unit == 'FLAM':
        FLUXUNITS = 'FLAM'
    elif f_unit == 'erg/cm2/s/Hz':
        flux = flux * 1e23
        FLUXUNITS = 'jy'
    else:
        raise Exception('Stellar units are not correct. Pick FLAM or Jy or erg/cm2/s/Hz')

    sp = jdi.psyn.ArraySpectrum(wave, flux, waveunits=WAVEUNITS,
                                fluxunits=FLUXUNITS)  # Convert everything to nanometer for conversion based on gemini.edu
    sp.convert("um")
    sp.convert('flam')  # ergs/cm2/s/ang
    wno_star = 1e4 / sp.wave[::-1]  # convert to wave number and flip
    wv_star = 1e4 / wno_star
    flux_star = sp.flux[::-1] * 1e8  # flip and convert to ergs/cm3/s here to get correct order

    return (wv_star, flux_star)


def mask_star(wv_star, flux_star, choose_wave):
    '''

    Select wavelengths from stellar grid that correspond to range of filter wavelengths

    Parameters
    ----------
    wv_star : ndarray
        High resolution stellar wavelength grid with correct units
    flux_star : ndarray
        High resolution stellar flux grid with correct units
    choose_wave : list of floats
        Wavelength range of interest corresponding to filter wavelengths, in microns. e.g. [0.95,1.77] for HST WFC3 G141

    Returns
    -------
    wv_star_masked : ndarray
        Masked stellar wavelength array
    flux_star_masked : ndarray
        Masked stellar flux array
    '''

    l1 = choose_wave[0]
    l2 = choose_wave[1]

    mask_s = [(wv_star > l1) & (wv_star < l2)]
    wv_star_masked = wv_star[mask_s]
    flux_star_masked = flux_star[mask_s]

    return (wv_star_masked, flux_star_masked)


def interpolate_filter(filter_file, wv_star):
    '''
    Interpolate transmission function (e.g. HST WFC3 filter) onto high resolution stellar grid

    Parameters
    ----------
    filter_file : str
        Path to instrument filter transmission function (must be 2 columns: wavelength (in meters) and sensitivity)
    wv_star : ndarray
        High resolution stellar wavelength grid with correct units (microns and erg/cm3/s)

    Returns
    -------
    resp : ndarray
        Interpolated response function, onto higher resolution stellar wavelength grid
    '''

    wv_filter = np.genfromtxt(filter_file, usecols=0) * 1e6
    sns_filter = np.genfromtxt(filter_file, usecols=1)

    interpfunc = interpolate.interp1d(wv_filter, sns_filter, bounds_error=False, fill_value=0)
    resp = interpfunc(wv_star)

    ########## Make transmission 0 where is is supposed to be 0 #############
    start = wv_filter[0]
    end = wv_filter[len(wv_filter) - 1]

    wws = np.where(wv_star <= start)
    wwe = np.where(wv_star >= end)
    resp[wwe] = 0.0
    resp[wws] = 0.0

    return (resp)


def transmitted_fpfs(planet_flux, star_flux, wv_star, resp, rprs, R):
    '''
    Calculate fraction of flux transmitted by instrument response.

    Parameters
    ----------
    planet_flux: ndarray
        1D array of planetary flux interpolated onto high resolution masked stellar grid
    star_flux : ndarray
        High resolution stellar flux
    wv_star : ndarray
        High resolution stellar wavelength
    resp : ndarray
        Instrument response function interpolated over high resolution masked stellar grid
    rprs : float
        Planet to star radius ratio

    Returns
    -------
    planet_trans :  ndarray
        Transmitted planetary thermal flux
    star_trans : ndarray
        Transmitted stellar flux
    '''

    h = 6.626e-27
    c = 2.998e14  # dist. units=microns

    #diff = np.diff(wv_star)
    #diff = (np.append(diff, diff[-1:]) + np.append(diff[1:], diff[-2:])) / 2

    planet_trans = planet_flux * resp * wv_star / (h * c)  # apply weighting factor (filter) to every flux
    star_trans = star_flux * resp * wv_star / (h * c)

    return (planet_trans, star_trans)


def whitelight_flux(planet_trans, star_trans, resp):
    '''
    Sum up fluxes from each wavelength band to obtain white light flux

    Parameters
    ----------
    star_trans : ndarray
        Stellar flux transmitted through filter (output of transmitted fpfs)
    planet_trans : ndarray
        Planet flux transmitted through filter (output of transmitted fpfs)
    resp : ndarray
        Instrument response function interpolated over high resolution masked stellar grid

    Returns
    -------
    sum_pflux : float
        White light planetary thermal flux
    sum_sflux : float
        White light stellar flux
    sum_weights : float
        Sum of filter response weighting function
    '''

    sum_pflux = 0
    sum_sflux = 0
    sum_weights = 0

    for j in range(len(planet_trans)):  # planet_trans is array of len = # wavelengths

        sum_pflux += planet_trans[j]
        sum_sflux += star_trans[j]
        sum_weights += resp[j]

    return (sum_pflux, sum_sflux, sum_weights)


##### Main Function to compute 3D fluxes at different phases (e.g. phase curves) #####

def thermal_phasecurve(planet=None, in_ptk=None, filt_path=None, wv_range=None,
                       res=None, nphases=None,
                       p_range=None, ng=None, nt=None, nlon=None, nlat=None, nz=None, CtoO=None, mh=None, mmw=None,
                       rp=None, mp=None, rs=None, rprs=None, sma=None, R=None,
                       sw_units=None, sf_units=None, in_star=None, Ts=None,
                       logg_s=None, met_s=None, cloudy=False, fsed=None, optics_dir=None):
    
    '''
    Compute thermal phase curves from 3D pressure-temperature input (MITgcm). 
    This function rotates the input 3D grid to select the visible hemisphere at 
    each orbital phase angle, it computes the 3D chemistry profile and thermal spectrum 
    at each orbital point. If clouds are turned on, it will also run virga to compute 
    the 3D cloud profile, and use it to compute the spectrum.
    The 3D flux at each orbital point is then integrated vertically and for all angles,
    and then integrated for desired wavelength range to obtain one flux value per orbital 
    phase.

    Parameters
    ----------
    planet : str
        Name of planet
    in_ptk : str
        Path to MITgcm pressure-temperature-Kzz profile file
    newpt_path : str
        Path to store regridded P-T-Kzz profiles
    filt_path : str
        Path to instrument response function
    wv_range : list
        Wavelength range of interest in microns (e.g. [0.95, 1.77] for HST WFC3 G141
    res : int
        Resampling value, default=1(increasing values reduce wavelength resolution)
    nphases : int
        Number of phases around the orbit (orbital phase resolution)
    p_range : list of floats
        Orbital phase range, in degrees from 0 to 360
    ng : int
        Number of Gauss angles
    nt : int
         Number of Chebyshev angles
    nlon : int
        Number of longitude points in MITgcm input file
    nlat : int
        Number of latitude points in MITgcm input file
    nz : int
        Number of pressure layers in MITgcm input file
    CtoO : float
        Carbon to Oxygen ratio
    mh : float
        Metallicity of planet
    mmw : float
        Atmospheric mean molecular weight
    rp : float
        Radius of planet in Jupiter radius
    mp : float
        Mass of planet in Jupiter mass
    rs : float
        Radius of star in Sun radius
    rprs : float
        Planet to star radius ratio
    sma : float
        Semi-major axis
    R : int
        Final wavelength resolution for plotting spectrum
    sw_units : str
        Stellar wavelength units from input stellar file (e.g. 'microns')
    sf_units : str
        Stellar flux units from input stellar file (e.g. erg/cm2/Hz/s)
    in_star : str
        Path to input stellar file
    Ts : float
        Star temperature
    logg_s : float
        Star Log G
    met_s : float
        Star metallicity
    cloudy : bool
        If True, clouds added into the calculation. Default = False
    fsed : float
        Sedimentation efficiency for cloud calculation (Virga)
    cld_path : str
        Path to save Virga cloud dictionaries
    optics_dir : str
        Path to virga Mie parameters and refractive indices for all species
    reuse_pt : bool
        If True, this function will search existing regridded chemistry (with the same resolution) in chempath to save time. Default=False
    reuse_cld : bool
        If True, this function will search existing Virga cloud files (with the same resolution) in cld_path to save time. Default=False

    Returns
    -------
    fluxes : dict
        Dictionary with thermal flux at every orbital phase.
    '''

    phases360 = np.linspace(p_range[0], p_range[1], nphases)
    phases = phases360 - 180  # go from 0-360 to -180-180

    flag = 0
    fluxes = defaultdict(dict)  # inititate dictionary where phase curve data will be stored
    all_fpfs = np.zeros(nphases)
    all_phases = np.zeros(nphases)

    for iphase in range(0, nphases):

        # Regrid data and compute chemistry

        phase360 = phases360[iphase]
        phase = phases[iphase]

        print('PHASE:', round(phase360, 1))

        newptk, chem = regrid_and_chem_3D(planet=planet, input_file=in_ptk, orb_phase=phase, n_gauss_angles=ng,
                                              n_chebychev_angles=nt, nlon=nlon, nlat=nlat, nz=nz, CtoO=CtoO, mh=mh)

        # Compute cloud profiles with Virga

        if cloudy == True:
            cld_dat = virga_3D(optics_dir=optics_dir, mmw=mmw, fsed=fsed,
                                radius=rp, mass=mp, ptk_dat=newptk)

        else:
            cld_dat = None

        # Compute spectrum

        spectrum, star_dict = spectrum_3D(calc='thermal', res=res, ng=ng, nt=nt,
                                          waverange=wv_range, radius=rp, mass=mp, s_radius=rs, logg=logg_s, s_temp=Ts,
                                          s_met=met_s, sma=sma, chemdata=chem, starfile=in_star,
                                          s_w_units=sw_units, s_f_units=sf_units, clouds_dat=cld_dat, fsed=fsed)

        # Compute average flux over hemisphere

        thermal_3d = spectrum['full_output']['thermal_3d']
        fpfs_pic = spectrum['fpfs_thermal']
        wno = spectrum['wavenumber']
        wvl_pic = 1e4 / wno  # in microns

        if flag == 0:
            # write stellar flux into file only once
            fluxes['star_flux_picaso'] = star_dict['flux']

        ### Calculate Fp/Fs for each phase and wavelength ###

        pflux = spectrum['thermal']  # get directly from picaso output spectrum
               
        # get stellar flux directly from the star file, but need to make sure units are in erg/cm2/s/cm

        if in_star != None:
            ws, fs = correct_star_units(in_star, sw_units, sf_units)

        else:
            ws, fs = 1e4 / star_dict['wno'], star_dict['flux']
                                   
        ### Apply mask with wavelengths of interest ###

        wv_star, flux_star = mask_star(ws, fs, wv_range)

        ### sensitivity function, interpolate over stellar wavelength ###

        resp = interpolate_filter(filt_path, wv_star)

        ### Calculate transmission ###

        #avg_pflux = np.zeros(len(all_phases))  # to plot phase curves
        #avg_sflux = np.zeros(len(all_phases))  # to plot phase curves

        # Interpolate planet flux onto stellar grid
        interpfunc = interpolate.interp1d(wvl_pic, pflux, bounds_error=False, fill_value=0)
        fplan = interpfunc(wv_star)

        # Calculate transmission in the filter band and FpFs for every phase

        planet_trans, star_trans = transmitted_fpfs(fplan, flux_star, wv_star, resp, rprs=rprs, R=R)

        ### integrate flux over wavelength for white light phasecurve ###

        sum_pflux, sum_sflux, sum_weights = whitelight_flux(planet_trans, star_trans, resp)

        #avg_pflux[flag] = sum_pflux  # array with weighted average planet flux
        #avg_sflux[flag] = sum_sflux  # array with weighted average stellar flux

        #fpfs_pc = (avg_pflux / avg_sflux) * (rprs ** 2)
        
        all_fpfs[flag] = (sum_pflux/sum_sflux)*(rprs**2)
        print('fpfs', all_fpfs[flag])
        all_phases[flag] = round(phase360, 2)

        fluxes[round(phase360, 2)] = {'fpfs_spect': fpfs_pic, 'thermal': pflux, 'pflux_trans': planet_trans, 'sflux_trans': star_trans, 'wvl_spect': wvl_pic, 'wvl_star_masked':wv_star, 'thermal_3d': thermal_3d}

        print('Thermal flux added to phasecurve dictionary')
        
        flag += 1

    fluxes['all_fpfs'] = all_fpfs
    fluxes['all_phases'] = all_phases

    return (fluxes)  # this is a dictionary
