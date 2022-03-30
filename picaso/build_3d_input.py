import json
from .disco import get_angles_3d, compute_disco
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import astropy.units as u
from .justdoit import inputs 
import pickle as pk
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


	threed_grid = pd.read_csv(input_file,delim_whitespace=True,names=['p','t','k'])
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


	threed_grid = pd.read_csv(input_file,delim_whitespace=True,names=names)
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
			data = pd.read_csv(input_file, delim_whitespace=True,**kwargs)
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
			data = pd.read_csv(input_file, delim_whitespace = True,
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
