import pandas as pd 
import os 
from io_utils import read_hdf
import numpy as np
__refdata__ = os.environ.get('albedo_refdata')
import pickle as pk
import astropy.constants as c

def get_wave_grid(filename, wmin=0.3, wmax=1, R=5000):
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

def set_PIA(filename, wno):
	"""
	Computes pressure induced absorption, which currently includes 
	(H2H2, H2He, H2H, H2CH4). Comparable to SETPIAI

	Parameters: 
	-----------
	filename : str 
		filename of H2CIA to read in 
	wno : numpy.array 
		wavelength dictionary with {'wave':[array of microns]}
	"""

	pfile = os.path.join(__refdata__, 'opacities',filename)

	#read in all h2cia data (currently there is only one hdf5 table with all data)
	h2cia = pd.read_hdf(pfile,'d')

	#slice of dataframe with only values less than the freqmax 
	freqmax = np.max(h2cia['wno'])
	freqlo = np.min(h2cia['wno'])

	#xmean, xmeanhe, xmeanh, xmeanch combined to single dictionary

	#first set up empty dataframe with right temperatures
	columns = [i for i in h2cia.keys() if (i != 'wno')]
	index = list(wno)*int(len(h2cia['temperature'].unique()))
	xmean = pd.DataFrame(index=index, columns=columns)
	ts=[]
	for i in h2cia['temperature'].unique():
		ts+=[i]*len(wno)
	xmean['temperature'] = pd.Series(ts, index = xmean.index)

	#now fill dataframe with interpolated H2 data and set wno (spectral wave) as the index\
	#note here that the pd.index will be repeat for values of temperature
	for i in h2cia['temperature'].unique():
		cia = h2cia.loc[(h2cia['temperature']==i)]
		cia.pop('temperature')
		cia = pandas_interpolate(cia, wno, 'wno', method='linear')
		cia=cia.set_index('wno')
		xmean.loc[xmean['temperature']==i,cia.keys()] = cia
	#set everything to greater than freqmax to -33 to make sure interpolation doesn't do anything bad
	xmean.loc[xmean.index>freqmax,cia.keys()] = -33

	#Use Linsky parameterization to fill in -33 gaps, if they exist 
	df['h2h2'].loc[df['h2h2']==-33] = np.vectorize(fit_linsky)(df['h2h2'].loc[df['h2h2']==-33], 
														df['temperature'].loc[df['h2h2']==-33],
														df.index[df['h2h2']==-33])

	return xmean


def pandas_interpolate(df,at, interp_column, method='linear'):
	"""
	Interpolates in a pandas dictionary 
	Parameters
	----------
	df : DataFrame
		pandas dataframe 
	at : numpy.array
		array with wavelength grid to bin to
	interp_column : str
		String of the key you want to interpolate to 
	method : str
		(Optional) Method of interpolation, Default='linear' (linear, cubic, quadratic)
	"""
	df = df.set_index(interp_column)
	# previously it was the next line. Change it to take the union of new and old
	# df = df.reindex(numpy.arange(df.index.min(), df.index.max(), 0.0005))
	df = df.reindex(df.index | at)
	df = df.interpolate(method=method).loc[at]
	df = df.reset_index()
	df = df.rename(columns={'index': interp_column})
	return df

def fit_linsky(h2h2, t, wno, va=3):
	"""
	Adds H2-H2 opacity from Linsky (1969) and Lenzuni et al. (1991) 
	to values that were set to -33 as place holders. 

	Parameters
	----------
	h2h2 : numpy.array
		h2h2 opacity 
	t : numpy.array
		Temperature 
	wno : numpy.array
		wave number
	va : int or float
		(Optional) 1,2 or 3 (depending on what overtone to compute 

	Returns
	-------
	numpy.array 
		H2-H2 absorption in cm-1 amagat-2 
	"""

	#these numbers are hard coded from Lenuzi et al 1991 Table 8. 
	sig0 = np.array([4162.043,8274.650,12017.753]) #applicable sections in wavelength 

	d1 = np.array([1.2750e5,1.32e6,1.32e6])
	d2 = np.array([2760.,2760.,2760.])
	d3 = np.array([0.40,0.40,0.40])

	a1 = np.array([-7.661,-9.70,-11.32])
	a2 = np.array([0.5725,0.5725,0.5725])

	b1 = np.array([0.9376,0.9376,0.9376])
	b2 = np.array([0.5616,0.5616,0.5616])

	w = sig0[va]

	d=d3[va]*np.sqrt(d1[va]+d2[va]*t)
	a=10**(a1[va]+a2[va]*np.log10(t))
	b=10**(b1[va]+b2[va]*np.log10(t))
	aa=4.0/13.0*a/d*np.exp(1.5*d/b)

	if wno<w: kappa = a*d*wno*np.exp((wno-w)/0.6952/t)/((wno-w)**2+d*d)
	elif wno<w+1.5*d: kappa=a*d*wno/((wno-w)**2+d*d)
	else: kappa = aa*wno*np.exp(-(wno-w)/b)

	return np.log10(kappa)

def get_h2minus(wno):
	"""
	This results gives the H2 minus opacity, needed for temperatures greater than 600 K. 
	K L Bell 1980 J. Phys. B: At. Mol. Phys. 13 1859, Table 1 
	theta=5040/T(K)

	The result is given in cm4/dyn, which is why there is a multiplication by nh2*ne*k*T
	where:
	nh2: number of h2 molecules/cm3
	ne: number of electrons/cm3
	
	Parameters:
	-----------
	wno : wave number cm-1

	Returns
	-------
	dict
		index=wno, columns = 'h2minus','temperature'
	"""
	fname = os.path.join(__refdata__, 'opacities',filename)
	df = pd.read_csv(fname, skiprows=5, header=0)

	#Per Bell's definition of temperature
	t = 5040.0/df['theta']

