import pandas as pd 
import os 
from .io_utils import read_hdf
import numpy as np
__refdata__ = os.environ.get('picaso_refdata')
import pickle as pk
import astropy.constants as c
import scipy.interpolate as sci

def get_cld_input_grid(filename_or_grid='wave_EGP.dat'):
	"""
	The albedo code relies on the cloud code input, which is traditionally on a 196 wavelength grid. 
	This method is to retrieve that grid. This file should be kept in the package reference data. Alternatively, 
	it might be useful to add it directly to the input.cld file. 

	Parameters
	----------
	filename_or_grid : str or ndarray
		filename of the input grid OR a numpy array of wavenumber corresponding to cloud 
		input

	Returns 
	-------
	array 
		array of wave numbers in increasing order 
	"""
	if filename_or_grid == 'wave_EGP.dat':
		grid = pd.read_csv(os.path.join(__refdata__, 'opacities',filename_or_grid), delim_whitespace=True)
		grid = grid.sort_values('wavenumber')['wavenumber'].values
	elif isinstance(filename_or_grid, np.ndarray):
		grid = np.sort(filename_or_grid)
	elif (isinstance(filename_or_grid, str) & (filename_or_grid != 'wave_EGP.dat') & 
		os.path.exists(filename_or_grid)):	
		grid = pd.read_csv(os.path.join(filename_or_grid), delim_whitespace=True)
		if 'wavenumber' in grid.keys():
			grid = grid.sort_values('wavenumber')['wavenumber'].values
		else: 
			raise Exception('Please make sure there is a column named "wavenumber" in your cloud wavegrid file')
	else:
		raise Exception("Please enter valid cloud wavegrid filepath, or numpy array. Or use default in reference file.")

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
		#f = sci.interp1d(old_wno, matrix[i,:],kind='cubic')
		#new[i, :] = f(new_wno)
	return new