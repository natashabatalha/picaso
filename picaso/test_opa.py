from opacity_factory import ContinuumFactory ,MolecularFactory
#from continuum_factory_v2 import ContinuumFactory as CFV2
import pandas as pd
import numpy as np

def continuum():
	original_file = '../notebooks/CIA_DS_aug_2015.dat'
	colnames = ['wno','H2H2','H2He','H2H','H2CH4','H2N2']
	new_wno = '../scripts/newwave_2000.coverage'
	new_wno = np.sort(pd.read_csv(new_wno, delim_whitespace=True, header=None)[2].values)
	factory = ContinuumFactory(original_file,colnames, new_wno)
	factory.restructure_opacity()

def molecular():
	original_dir = '/Users/natashabatalha/Documents/AlbedoCodeWC/opacities/'
	new_wno = '../scripts/newwave_2000.coverage'
	new_wno = np.sort(pd.read_csv(new_wno, delim_whitespace=True, header=None)[2].values)
	factory = MolecularFactory(original_dir, new_wno)
	factory.restructure_opacity()

#molecular()
continuum()