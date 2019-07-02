import json
from .disco import get_angles, compute_disco
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import astropy.units as u
def rebin_mitgcm(ng, nt, phase_angle, input_file, output_file,p_unit='Pa', run_chem=False, MH=None, CtoO=None):
	"""
	Rebin GCM grid to a smaller grid. Function does not yet work!! 

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
	p_unit : str
		Pressure Unit (default Pascal)
	run_chem : bool 
		(Optional) runs chemistry and adds to json file

	TO DO
	-----
	- add run chemistry to get 3d chem
	"""	

	gangle,gweight,tangle,tweight = get_angles(ng,nt)
	ubar0, ubar1, cos_theta ,lat_picaso,lon_picaso = compute_disco(ng,nt, gangle, tangle, phase_angle)    


	infile = open(input_file,'r')

	temp = infile.readline().split()
	nlon = int(temp[0])
	nlat = int(temp[1])
	nz = int(temp[2])
	total_pts = nlon * nlat
	all_lon = np.zeros(total_pts) # 128 x 64
	all_lat = np.zeros(total_pts)
	p = np.zeros(nz)
	t = np.zeros((nlon,nlat,nz))
	kzz = np.zeros((nlon,nlat,nz))

	# skip line of header
	temp = infile.readline()
	ctr = -1

	for ilon in range(0,nlon):
		for ilat in range(0,nlat):
			ctr += 1

			#skip blank line
			temp = infile.readline()
			temp = infile.readline().split()

			all_lon[ctr] = float(temp[0])
			all_lat[ctr] = float(temp[1])

			# read in data for each grid point
			for iz in range(0,nz):
				temp = infile.readline().split()
				p[iz] = float(temp[0])
				t[ilon,ilat,iz] = float(temp[1])
				kzz[ilon,ilat,iz] = float(temp[2])

	lon = np.unique(all_lon)
	lat = np.unique(all_lat)

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
	for iz in range(0,nz):
	    new_t[:,iz] = np.sum(t[:,:,iz].flatten()[inds],axis=1)/nn
	    new_kzz[:,iz] = np.sum(kzz[:,:,iz].flatten()[inds],axis=1)/nn


	df = {} 
	df={i:{} for i in lat_picaso}


	outfile = open(output_file,'w')
	for ipos in range(0,ng*nt):

		#make sure units are always bars
		p = (p*u.Unit(p_unit)).to(u.bar).value

		if run_chem:
			case = jdi.inputs()
			case.inputs['atmosphere']['profile'] = pd.DataFrame({'temperature':new_t[ipos,0:nz], 'pressure':p, 'kzz':new_kzz[ipos,0:nz]})
			case.chemeq(CtoO, MH)
			data = case.inputs['atmosphere']['profile']
		else : 
			data = pd.DataFrame({'temperature':new_t[ipos,0:nz], 'pressure':p, 'kzz':new_kzz[ipos,0:nz]})

		df[lat2d[ilat]][lon2d[ipos]] = data

	df['phase_angle'] = phase_angle

def make_3d_pt_input(ng,nt,phase_angle,input_file,output_file,**kwargs):
	"""
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
	gangle,gweight,tangle,tweight = disco.get_angles(ng, nt)
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
	"""
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
	gangle,gweight,tangle,tweight = disco.get_angles(ng, nt)
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
	x =  R * np.cos(lat_r) * np.cos(lon_r)
	y = R * np.cos(lat_r) * np.sin(lon_r)
	z = R * np.sin(lat_r)
	return x,y,z
