import h5py
from picaso import disco
import pandas as pd



def make_3d_pt_input(ng,nt,phase_angle,input_file,output_file):
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
	output_file : str 
		Output file location 

	Returns
	-------
	Creates output file. No other returns. 
	"""

	h5db = h5py.File(output_file,'w')
	#get geometry
	gangle,gweight,tangle,tweight = disco.get_angles(ng, nt) 
	ubar0, ubar1, cos_theta,lat,lon = disco.compute_disco(ng, nt, gangle, tangle, phase_angle)
	


	first = True
	for g in gangle: 
		for t in tangle:

			data = pd.read_csv(input_file, delim_whitespace=True)
				if first:
					h5db.attrs['header'] = ','.join(list(data.keys()))
					first = False
				dset = h5db.create_dataset(i'/'+str(g)+'/'+str(t), data=data, chunks=True)

def make_3d_cld_input(ng,nt,phase_angle,input_file,output_file):
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
	output_file : str 
		Output file location 

	Returns
	-------
	Creates output file. No other returns. 
	"""
	
	h5db = h5py.File(output_file,'w')
	#get geometry
	gangle,gweight,tangle,tweight = disco.get_angles(ng, nt) 
	ubar0, ubar1, cos_theta,lat,lon = disco.compute_disco(ng, nt, gangle, tangle, phase_angle)
	first = True

	for g in gangle: 
		for t in tangle:
			data = pd.read_csv(input_file, delim_whitespace = True,
					header=None, skiprows=1, names = ['lvl', 'wv','opd','g0','w0','sigma'],
					dtype='f8')
			data = data*0
			if first:
				h5db.attrs['header'] = ','.join(list(data.keys()))
				first = False
			dset = h5db.create_dataset(str(g)+'/'+str(t), data=data, chunks=True)	

if __name__ == "__main__":

	phase_angle = 0 
	ng = 10 
	nt = 10

	input_file='/Users/natashabatalha/Documents/picaso/reference/base_cases/jupiter.pt'
	output_file='../3d_pt_test.hdf5'

	make_3d_pt_input(ng, nt,phase_angle, input_file,output_file)	

	input_file='/Users/natashabatalha/Documents/picaso/reference/base_cases/jupiterf3.cld'
	output_file='../3d_cld_test.hdf5'

	make_3d_cld_input(ng, nt,phase_angle, input_file,output_file)	