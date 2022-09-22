from .opacity_factory import ContinuumFactory ,MolecularFactory
import pandas as pd
import numpy as np
import os 
__refdata__ = os.environ.get('picaso_refdata')
def continuum(new_wno ='newwave_2000.coverage', original_file = 'CIA_DS_aug_2015.dat',
	colnames = ['wno','H2H2','H2He','H2H','H2CH4','H2N2'], new_filename=os.path.join(__refdata__, 'opacities','continuum.json')):
	"""
	This is a top level function to run the opacity facotry to compute the continuum. 
	It leverages D. Saumom's continuum opacity file and regrids it to the wavelength 
	that the user is attempting to create a spectrum on. 

	Parameters
	----------
	new_wno : str , array
		This is a pointer to the filename that is whitespace delimeted with at least one column 
		called "wavenumber". Can also be a numpy array
	original_file : str 
		This is a pointer to the original continuum opacity file that is structured according to 
		D. Saumon. This will eventually be replaced with the opacity db structure. 
	colnames : list, str
		This identifies the continuum absorbers that are located within the original_file with 
		opacities. In future updates, the file might be updated to include more continuum 
		absorbers. 
	"""
	if isinstance(new_wno ,str):
		new_wno = np.sort(pd.read_csv(new_wno, delim_whitespace=True)['wavenumber'].values)
	else: 
		new_wno = np.sort(new_wno)
	factory = ContinuumFactory(original_file,colnames, new_wno,new_filename=new_filename)
	factory.restructure_opacity()

def molecular(original_dir = '/Users/natashabatalha/Documents/AlbedoCodeWC/opacities/', 
	new_wno ='newwave_2000.coverage',new_filename=os.path.join(__refdata__, 'opacities','molecular.hdf5')):
	"""
	Currently, picaso is operating on a BYO opacities. Therefore, this function should be used 
	to go through a large directory of opacities and restructure them to the hdf5 databse that 
	picaso uses. 

	Warning
	-------
	This function is **not** yet general enough to be used by the public. It is highly tied to 
	the opacity grid calculated by R. Freedman and used by ExoARC group. There are several 
	subleties to how this database is created. For example, alkalis are added through a 
	separate script. Please contact 
	natasha.e.batalha@gmail.com if you have questions about opacities. 

	Parameters
	----------
	new_wno : str, array
		This is a pointer to the filename that is whitespace delimeted with at least one column 
		called "wavenumber". Can also be a numpy array.
	original_dier : str 
		This is a pointer to the a file that should have a directory for each molecule. 

	Todo
	----
	-Right now users have to add Na K CH4 from function `addNa&K&CH4.py` in scripts. Come up with a better method than when we 
	have a better opacity database
	"""
	
	if isinstance(new_wno ,str):
		new_wno = np.sort(pd.read_csv(new_wno, delim_whitespace=True)['wavenumber'].values)
	else: 
		new_wno = np.sort(new_wno)
	factory = MolecularFactory(original_dir, new_wno,new_filename=new_filename)
	factory.restructure_opacity()

#