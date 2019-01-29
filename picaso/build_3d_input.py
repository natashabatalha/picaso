import h5py
from picaso import disco
import pandas as pd
import numpy as np




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
	for g in gangle:
		for t in tangle:

			data = pd.read_csv(input_file, delim_whitespace=True,**kwargs)
			if first:
				h5db.attrs['header'] = ','.join(list(data.keys()))
				first = False
			dset = h5db.create_dataset(str(g)+'/'+str(t), data=data, chunks=True)

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

