#from .justdoit import inputs 
from .disco import get_angles_3d, compute_disco

import json
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import astropy.units as u
import pickle as pk
import xarray as xr

def regrid_xarray(xarray_original, num_gangle=None, num_tangle=None,phase_angle=None, 
                latitude=None, longitude=None): 
    """
    This regrid functionality uses the xarray and xesmf method. It requires to additional 
    packages that are not required by the rest of PICASO (xesmf). See 
    tutorial for additional tips on how to structure your xarray input.  

    Parameters
    ----------
    xarray_original : xarray
        xarray of all 3d input you would like regridded 
    num_gangle : int 
        Number of gauss angles (not needed if directly supply lat and lons)
    num_tangle : int 
        Number of cheychev angles (not needed if directly supply lat and lons)
    phase_angle : int 
        Radians phase angle (not needed if directly supply lat and lons)
    latitude : array 
        array of latitudes in degrees (not needed if supplied num_gangle, num_tangle, phase_angle)
    longitude : array 
        array of longitudes in degrees (not needed if supplied num_gangle, num_tangle, phase_angle)

    Output
    ------
        Dictionary of regridded data and new latitude longitude grid 
    """
    import xesmf as xe
    if ((not isinstance(num_gangle, type(None))) & (not isinstance(num_gangle, type(None)))):
        gangle, gweight, tangle, tweight = get_angles_3d(num_gangle, num_tangle)
        ubar0, ubar1, cos_theta, latitude, longitude = compute_disco(num_gangle, num_tangle,
        gangle, tangle, phase_angle=phase_angle)
        
        longitude=longitude*180/np.pi #to degrees
        latitude=latitude*180/np.pi #to degrees
    elif ((not isinstance(longitude, type(None))) & (not isinstance(latitude, type(None)))):
        assert isinstance(latitude, np.ndarray), 'latitude must be numpy array'
        assert isinstance(longitude, np.ndarray), 'longitude must be numpy array'
    else: 
        raise Exception('Need to either supply (num_tangle, num_gangle,phase_angle) OR (latitude and longitude)')

    ds_out = xr.Dataset({'lon': (['lon'], longitude),
                     'lat': (['lat'], latitude),
                    }
                   )

    #xesmf regrids over the last two dimensions (lat, lon). If the input is stored in a different
    #order (e.g. lon, lat, pressure) xesmf transposes it, which creates a non C_CONTIGUOUS array and
    #raises a performance warning. So here we put lat, lon last and make the data contiguous in memory
    xarray_original = xarray_original.transpose(..., 'lat', 'lon')
    xarray_original = xarray_original.copy(data={v: np.ascontiguousarray(xarray_original[v].values)
                                                 for v in xarray_original.data_vars})

    regridder = xe.Regridder(xarray_original, ds_out, 'bilinear')
    ds_out = regridder(xarray_original,keep_attrs=True)

    dict_out = ds_out.to_dict()

    return ds_out

def regrid_transit(xarray_original, geom):
    """
    Regrids a lat/lon xarray onto the columns of a 3D transit geometry
    (see `inputs.transit_geometry`). Uses linear interpolation (periodic in longitude).

    The columns are not a rectilinear grid in latitude/longitude. Therefore, the output
    is relabeled so that the coordinate 'lon' holds the zenith angle of each column and
    'lat' holds the azimuthal angle of each column (both in degrees). These are the
    values stored in geom['longitude'] and geom['latitude'], so the rest of the 3D
    machinery can read each column as usual. Here we store the "real" latitude/longitude 
    of each column are in inputs['disco']['transit_latitude'] and ['transit_longitude'].

    Parameters
    ----------
    xarray_original : xarray
        xarray with 'lat' and 'lon' coordinates in degrees, with longitude=0 at the
        substellar point
    geom : dict
        Transit geometry dictionary, inputs['disco'], built by `inputs.transit_geometry`

    Returns
    -------
    xarray regridded onto the transit columns
    """
    ds = xarray_original.sortby('lon').sortby('lat')
    lon_in = ds.coords['lon'].values
    lat_in = ds.coords['lat'].values

    lat2d = geom['transit_latitude']*180/np.pi
    lon2d = geom['transit_longitude']*180/np.pi
    #match the longitude convention of the input (-180 to 180 or 0 to 360)
    if np.max(lon_in) > 180:
        lon2d = np.mod(lon2d, 360)
    #pad one column on either side so longitude interpolation is periodic
    ds = xr.concat([ds.isel(lon=[-1]).assign_coords(lon=[lon_in[-1]-360]), ds,
                    ds.isel(lon=[0]).assign_coords(lon=[lon_in[0]+360])], dim='lon')
    #columns poleward of the input grid take the value of the closest latitude
    lat2d = np.clip(lat2d, np.min(lat_in), np.max(lat_in))

    ds_out = ds.interp(lon=xr.DataArray(lon2d, dims=['lon_transit','lat_transit']),
                       lat=xr.DataArray(lat2d, dims=['lon_transit','lat_transit']),
                       kwargs={'fill_value':None})
    ds_out = ds_out.drop_vars(['lon','lat']).rename({'lon_transit':'lon','lat_transit':'lat'})
    ds_out = ds_out.assign_coords(lon=geom['longitude']*180/np.pi, lat=geom['latitude']*180/np.pi)
    ds_out.coords['lon'].attrs['units'] = 'degrees (transit zenith angle)'
    ds_out.coords['lat'].attrs['units'] = 'degrees (transit azimuthal angle)'

    return ds_out

def rebin_mitgcm_pt(ng, nt, phase_angle, input_file, output_file,p_unit='Pa', kzz_unit = 'm*m/s',run_chem=False, MH=None, CtoO=None):
    """
    Rebin GCM grid to a smaller grid. This function creates a pickle file 
    that can be directly input into `atmosphere_3d`.

    Parameters 
    ----------
    ng : int
        Number of gauss angles
    nt : int
        Number of Tchebysehv angles
    phase_angle : float
        Geometry of phase angle (radians)
    input_file : str
        PT input file you want to create the 3d grid from. Currently takes in a single 1d PT profile.
        Must have labeled, whitespace delimeted columns. In the columns, must have `temperature`, 
        `pressure`, and at least some molecular species (which are ALL case-sensitive)
    output_file : str
        output file location
    p_unit : str,optional
        Pressure Unit (default Pascal)
    kzz_unit : str,optional 
        Kzz Unit (default is m*m/s)
    run_chem : bool , optional
        (Optional) runs chemistry and adds to json file
    run_chem : bool , optional
        (Optional) MIT gcm output doesn't usually come with chemistry in the files 
        If this is the case then you can post process chemistry by setting this to True. 
    MH : float , optional
        (Optional) This is only used if run_chem is set to True. It is the Metallicity of 
        the planet in NON log units. MH = 1 is solar. 
    CtoO : float , optional
        (Optional) This is the C/O ratio of the planet also in NON log units. 
        CtoO = 0.55 is solar.  
    """ 

    gangle,gweight,tangle,tweight = get_angles_3d(ng,nt)
    ubar0, ubar1, cos_theta ,lat_picaso,lon_picaso = compute_disco(ng,nt, gangle, tangle, phase_angle)    


    threed_grid = pd.read_csv(input_file,sep=r'\s+',names=['p','t','k'])
    all_lon= threed_grid.loc[np.isnan(threed_grid['k'])]['p'].values
    all_lat=  threed_grid.loc[np.isnan(threed_grid['k'])]['t'].values
    latlong_ind = np.concatenate((np.array(threed_grid.loc[np.isnan(threed_grid['k'])].index),[threed_grid.shape[0]] ))
    threed_grid = threed_grid.dropna() 

    lon = np.unique(all_lon)
    lat = np.unique(all_lat)

    nlon = len(lon)
    nlat = len(lat)
    total_pts = nlon*nlat
    nz = latlong_ind[1] - 1 

    p = np.zeros((nlon,nlat,nz))
    t = np.zeros((nlon,nlat,nz))
    kzz = np.zeros((nlon,nlat,nz))

    for i in range(len(latlong_ind)-1):

        ilon = list(lon).index(all_lon[i])
        ilat = list(lat).index(all_lat[i])

        p[ilon, ilat, :] = threed_grid.loc[latlong_ind[i]:latlong_ind[i+1]]['p'].values
        t[ilon, ilat, :] = threed_grid.loc[latlong_ind[i]:latlong_ind[i+1]]['t'].values
        kzz[ilon, ilat, :] = threed_grid.loc[latlong_ind[i]:latlong_ind[i+1]]['k'].values


    lon2d, lat2d = np.meshgrid(lon_picaso, lat_picaso)
    lon2d = lon2d.flatten()
    lat2d = lat2d.flatten()

    xs, ys, zs = lon_lat_to_cartesian(all_lon,all_lat)
    xt, yt, zt = lon_lat_to_cartesian(lon2d,lat2d)

    tree = cKDTree(list(zip(xs,ys,zs)))
    nn = int(total_pts / (ng*nt))
    d,inds = tree.query(list(zip(xt,yt,zt)),k=nn)

    new_t = np.zeros((ng*nt,nz))
    new_kzz = np.zeros((ng*nt,nz))
    new_p = np.zeros((ng*nt,nz))

    for iz in range(0,nz):
        new_p[:,iz] = np.sum(p[:,:,iz].flatten()[inds],axis=1)/nn
        new_t[:,iz] = np.sum(t[:,:,iz].flatten()[inds],axis=1)/nn
        new_kzz[:,iz] = np.sum(kzz[:,:,iz].flatten()[inds],axis=1)/nn


    df = {} 
    df={int(i*180/np.pi):{} for i in lat_picaso}


    outfile = open(output_file,'w')
    if run_chem == True: case = inputs()
    
    for ipos in range(0,ng*nt):
        
        #make sure units are always bars
        new_p = (new_p*u.Unit(p_unit)).to(u.bar).value
        new_kzz = (new_kzz*u.Unit(kzz_unit)).to('cm*cm/s').value

        if run_chem:
            case.inputs['atmosphere']['profile'] = pd.DataFrame({'temperature':new_t[ipos,0:nz], 'pressure':new_p[ipos,0:nz], 'kzz':new_kzz[ipos,0:nz]})
            case.channon_grid_high() #solar metallicity/solar C/O
            data = case.inputs['atmosphere']['profile']
        else : 
            data = pd.DataFrame({'temperature':new_t[ipos,0:nz], 'pressure':new_p[ipos,0:nz], 'kzz':new_kzz[ipos,0:nz]})

        df[int(lat2d[ipos]*180/np.pi)][int(lon2d[ipos]*180/np.pi)] = data

    df['phase_angle'] = phase_angle

    pk.dump(df,open(output_file, 'wb'))


def rebin_mitgcm_cld(ng, nt, phase_angle, input_file, output_file,names=['i','j','opd','g0','w0']):
    """
    Rebin post processed GCM cloud grid to a smaller grid. This function creates a pickle file 
    that can be directly input into `clouds_3d`.

    Parameters 
    ----------
    ng : int
        Number of gauss angles
    nt : int
        Number of Tchebysehv angles
    phase_angle : float
        Geometry of phase angle (radians)
    input_file : str
        PT input file you want to create the 3d grid from. Currently takes in a single 1d PT profile.
        Must have labeled, whitespace delimeted columns. In the columns, must have `temperature`, 
        `pressure`, and at least some molecular species (which are ALL case-sensitive)
    output_file : str
        output file location
    names : list,str,optional
        List of names for the post processed file. The Default value is what generallly 
        comes out of A&M code.. But double check before running!! 
    """ 

    gangle,gweight,tangle,tweight = get_angles_3d(ng,nt)
    ubar0, ubar1, cos_theta ,lat_picaso,lon_picaso = compute_disco(ng,nt, gangle, tangle, phase_angle)    


    threed_grid = pd.read_csv(input_file,sep=r'\s+',names=names)
    #get the lat and lon points by looking at the locations that the last 
    #index is nan but the first two have values. 
    #this assumes that the MIT GCM person has created their file by printing
    #Lon, Lat before the dump of output.cld
    all_lon= threed_grid.loc[np.isnan(threed_grid[names[-1]])][names[0]].values
    all_lat=  threed_grid.loc[np.isnan(threed_grid[names[-1]])][names[1]].values
    latlong_ind = np.concatenate((np.array(threed_grid.loc[np.isnan(threed_grid['k'])].index),[threed_grid.shape[0]] ))
    threed_grid = threed_grid.dropna() 

    lon = np.unique(all_lon)
    lat = np.unique(all_lat)

    nlon = len(lon)
    nlat = len(lat)
    total_pts = nlon*nlat
    nznw = latlong_ind[1] - 1 #number of wavepoints * number of levels

    opd = np.zeros((nlon,nlat,nznw))
    g0 = np.zeros((nlon,nlat,nznw))
    w0 = np.zeros((nlon,nlat,nznw))

    for i in range(len(latlong_ind)-1):

        ilon = list(lon).index(all_lon[i])
        ilat = list(lat).index(all_lat[i])

        opd[ilon, ilat, :] = threed_grid.loc[latlong_ind[i]:latlong_ind[i+1]]['opd'].values
        g0[ilon, ilat, :] = threed_grid.loc[latlong_ind[i]:latlong_ind[i+1]]['g0'].values
        w0[ilon, ilat, :] = threed_grid.loc[latlong_ind[i]:latlong_ind[i+1]]['w0'].values


    lon2d, lat2d = np.meshgrid(lon_picaso, lat_picaso)
    lon2d = lon2d.flatten()
    lat2d = lat2d.flatten()

    xs, ys, zs = lon_lat_to_cartesian(all_lon,all_lat)
    xt, yt, zt = lon_lat_to_cartesian(lon2d,lat2d)

    tree = cKDTree(list(zip(xs,ys,zs)))
    nn = int(total_pts / (ng*nt))
    d,inds = tree.query(list(zip(xt,yt,zt)),k=nn)

    new_opd = np.zeros((ng*nt,nznw))
    new_g0 = np.zeros((ng*nt,nznw))
    new_w0 = np.zeros((ng*nt,nznw))

    for iz in range(0,nznw):
        new_opd[:,iz] = np.sum(opd[:,:,iz].flatten()[inds],axis=1)/nn
        new_g0[:,iz] = np.sum(g0[:,:,iz].flatten()[inds],axis=1)/nn
        new_w0[:,iz] = np.sum(w0[:,:,iz].flatten()[inds],axis=1)/nn


    df = {} 
    df={int(i*180/np.pi):{} for i in lat_picaso}


    outfile = open(output_file,'w')
    if run_chem == True: case = inputs(chemeq=True)
    
    for ipos in range(0,ng*nt):
        
        data = pd.DataFrame({'opd':new_opd[ipos,0:nznw], 'g0':new_g0[ipos,0:nznw], 'w0':new_w0[ipos,0:nznw]})

        df[int(lat2d[ipos]*180/np.pi)][int(lon2d[ipos]*180/np.pi)] = data

    df['phase_angle'] = phase_angle

    pk.dump(df,open(output_file, 'wb'))

def make_3d_pt_input(ng,nt,phase_angle,input_file,output_file,**kwargs):
    """Discontinued
    Program to create 3d PT input. Used to feed GCM input into disco ball.

    Parameters
    ----------
    ng : int
        Number of gauss angles
    nt : int
        Number of Tchebysehv angles
    phase_angle : float
        Geometry of phase angle
    input_file : str
        PT input file you want to create the 3d grid from. Currently takes in a single 1d PT profile.
        Must have labeled, whitespace delimeted columns. In the columns, must have `temperature`, 
        `pressure`, and at least some molecular species (which are ALL case-sensitive)
    output_file : str
        Output file location

    Returns
    -------
    Creates output file. No other returns.

    Examples
    --------
    Basic use of creating a 3D file: 

    >>> phase_angle = 0
    >>> number_gauss_angles = 2
    >>> number_tcheby_angles = 3
    >>> phase_angle = 0
    >>> input_file=jdi.jupiter_pt()
    >>> output_file='jupiter3D.hdf5'
    >>>make_3d_pt_input(number_gauss_angles, number_tcheby_angles,
    >>>         phase_angle, input_file,output_file)
    
    Can also use `**kwargs` to control how pandas reads in input file 

    >>>make_3d_pt_input(number_gauss_angles, number_tcheby_angles,
    >>>         phase_angle, input_file,output_file,usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21])

    """

    h5db = h5py.File(output_file,'w')
    #get geometry
    gangle,gweight,tangle,tweight = disco.get_angles_3d(ng, nt)
    ubar0, ubar1, cos_theta,lat,lon = disco.compute_disco(ng, nt, gangle, tangle, phase_angle)



    first = True
    dset = {}
    for g in gangle:
        for t in tangle:
            data = pd.read_csv(input_file, sep=r'\s+',**kwargs)
            if first:
                dset['header'] = ','.join(list(data.keys()))
                first = False
            dset[str(g)][str(t)] = data

def make_3d_cld_input(ng,nt,phase_angle,input_file,output_file, lat_range=None, lon_range=None,rand_coverage=1):
    """Discontinued
    Program to create 3d CLOUD input. Used to feed GCM input into disco ball.

    Parameters
    ----------
    ng : int
        Number of gauss angles
    nt : int
        Number of Tchebysehv angles
    phase_angle : float
        Geometry of phase angle
    input_file : str
        CLD input file you want to create the 3d grid from. Currently takes in a single 1d CLD profile.
        Input file must be structured with the following columns in the correct order: ['lvl', 'wv','opd','g0','w0','sigma']
        input file 
    output_file : str
        Output file location
    lat_range : list
        (Optional)Range of latitudes to exclude in the CLD file
    lon_range : list
        (Optional)Range of longitudes to exclud in the CLD file
    rand_coverage : float 
        (Optional)Fractional cloud coverage. rand_cov=1 introduces full cloud coverage. rand_cov=0 has no
        coverage. 
    **kwargs : dict 
        Key word arguments used for `panads.read_csv`

    Returns
    -------
    Creates output file. No other returns.

    Examples
    --------
    Basic use of creating a 3D file: 
        
    >>> phase_angle = 0
    >>> number_gauss_angles = 2
    >>> number_tcheby_angles = 3
    >>> phase_angle = 0
    >>> input_file=jdi.jupiter_cld()
    >>> output_file='jupiter3D.hdf5'
    >>>make_3d_pt_input(number_gauss_angles, number_tcheby_angles,
    >>>         phase_angle, input_file,output_file)
    """

    h5db = h5py.File(output_file,'w')
    #get geometry
    gangle,gweight,tangle,tweight = disco.get_angles_3d(ng, nt)
    ubar0, ubar1, cos_theta,lat,lon = disco.compute_disco(ng, nt, gangle, tangle, phase_angle)
    first = True

    for g, lg in zip(gangle,lon):
        for t, lt in zip(tangle, lat):
            data = pd.read_csv(input_file, sep=r'\s+',
                    header=None, skiprows=1, names = ['lvl', 'wv','opd','g0','w0','sigma'],
                    dtype='f8')
            if not isinstance(lat_range,type(None)):
                if (lt > np.min(lat_range)) and (lt<np.max(lat_range)):
                    data = data*0
            if not isinstance(lon_range,type(None)):
                if (lg > np.min(lon_range)) and (lg < np.max(lon_range)):
                    data = data*0
            if np.random.rand() > rand_coverage:
                data = data*0

            if first:
                h5db.attrs['header'] = ','.join(list(data.keys()))
                first = False
            dset = h5db.create_dataset(str(g)+'/'+str(t), data=data, chunks=True)

def lon_lat_to_cartesian(lon, lat, R = 1):
    """
    Calculates lon, lat coordinates of a point on a sphere with
    radius R

    Parameters 
    ----------
    lon : array
        Longitude
    lat : array 
        Latitude 
    R : float   
        radius of sphere 

    Returns
    -------
    x, y, z 
    """
    x =  R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    return x,y,z

def variable_pressure_example(): 
    # create coords
    longitude = np.linspace(-180,177,10)
    latitude = np.linspace(-90, 90,10)
    pressure =  np.logspace(-6,2,50)
    lon3d,lat3d, pres3d = np.meshgrid(longitude,latitude,pressure, indexing='ij')

    # create fake data for example 
    data = pres3d*0 + 1800#

    # put data into a dataset
    ds_variable_pressure = xr.Dataset(
        data_vars=dict(
            temperature=(["x","y",'z'], data,{'units': 'Kelvin'})
        ),
        coords=dict(
            lon=(["x","y",'z'], lon3d,{'units': 'degrees'}),
            lat=(["x","y",'z'], lat3d,{'units': 'degrees'}),
            pressure=(["x","y",'z'], pres3d,{'units': 'bar'}),
        ),
        attrs=dict(description="coords with matrices"),
    )    
    return ds_variable_pressure