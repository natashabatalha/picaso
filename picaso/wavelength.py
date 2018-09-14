import pandas as pd 
import os 
from io_utils import read_hdf
import numpy as np
__refdata__ = os.environ.get('picaso_refdata')
import pickle as pk
import astropy.constants as c

def get_output_grid(filename, wmin=0.3, wmax=1, R=5000):
	"""
	Defines wavelength grid from HDF5 file.(comparable to SETSPV)

	Currently, only single option for wavelength grid, could be expanded to include more 
	grids, more opacities. Currently there is only one grid, so all values are defaulted to 
	0.3-1 at R=5000

	Parameters
	----------
	filename : name of file  
		filename
	wmin : float
		minimum wavelength 
	wmax : float 
		maximum wavelength 
	R : float
		Resolution 
	"""
	pfile = os.path.join(__refdata__, 'opacities',filename)
	requires = {'wmin':wmin, 'wmax':wmax,'R':R}
	grid = read_hdf(pfile,  requires)
	return grid

def get_input_grid(filename):
	"""
	The albedo code relies on the cloud code input, which is traditionally on a 196 wavelength grid. 
	This method is to retrieve that grid. This file should be kept in the package reference data. Alternatively, 
	it might be useful to add it directly to the input.cld file. 

	Parameters
	----------
	filename : str
		filename of the input grid 

	Returns 
	-------
	array 
		array of wave numbers in increasing order 

	To Do 
	-----
	Add this directly to reference data 
	"""
	grid = pd.read_csv(os.path.join(__refdata__, 'opacities',filename), delim_whitespace=True, header=None, names=['micron','wavenumber'], usecols=[1,2])
	return grid

def regrid(matrix, old_wno, new_wno):
	"""
	This takes in a matrix that is (number of something versus number of wavelength points) and regrids 
	to a new wave grid. 

	Parameters
	----------
	matrix : ndarray
		matrix that is (number of something versus number of wavelength points)
	old_wno : array
		array that represents the old wavelength grid of the matrix 
	new_wno : array
		array that represents the desired wavelength grid 

	Returns
	-------
	matrix 
		matrix that has been reinterpolated to new wave number grid 
	"""
	new = np.zeros((matrix.shape[0],len(new_wno)))
	for i in range(matrix.shape[0]): 
		new[i, :] = np.interp(new_wno, old_wno, matrix[i,:])
	return new