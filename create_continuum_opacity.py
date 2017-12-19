import numpy as np
import datetime 
import os 
import pandas as pd

__refdata__ = os.environ.get('albedo_refdata')

class Continuum_Factory():
	"""
	The continuum factory takes the CIA opacity file and adds in extra sources of 
	opacity from other references to fill in empty bands. It assumes that the original file is 
	structured as following,with the first column wavenumber and the rest molecules: 
	  1000 198
		75.
 	 0.0  -33.0000  -33.0000  -33.0000  -33.0000  -33.0000
 	20.0   -7.4572   -7.4518   -6.8038   -6.0928   -5.9806
 	40.0   -6.9547   -6.9765   -6.6322   -5.7934   -5.4823
	... ... ... etc 

	Where 1000 corresponds to the number of wavelengths, 198 corresponds to the number of temperatures
	and 75 is the first temperature. If this structure changes, we will need to restructure this top 
	level __init__ function that reads it in the original opacity file. 
	
	Parameters:
	-----------
	original_file : str
		Filepath that points to original opacity file (see description above)
	colnames : list 
		defines the sources of opacity in the original file. For the example file above, 
		colnames would be ['wno','h2h2','h2he','h2h','h2ch4','h2n2']
	new_wave : numpy.ndarray, list 
		wave grid to interpolate onto (units of inverse cm)
	"""
	def __init__(self, original_file,colnames, new_wave):
		self.extras = ''
		self.references = ''
		self.notes = ''
		self.date = now.strftime("%Y-%m-%d %H:%M")
		self.og_opacity = pd.read_csv(original_file,delim_whitespace=True,names=colnames)
		self.num_temp = self.og_opacity['h2h2'][0]
		self.num_wno = self.og_opacity['wno'][0]
		self.w_unit = 'cm-1'
		self.abs_unit = 'log(cm-1 amagat^-2)'
		self.molecules = ','.join(colnames[1:])
		self.hdf = pd.HDFStore('continuum'+self.date+'.h5')
		self.min_wave = np.min(new_wave)
		self.max_wave = np.max(new_wave)


	def create_opacity(linsky=True,h2minus=True):
		"""
		Creates master continuum opacity file 
		Everything is returned in units of amagat
		"""
		
		#first add in temperature as a separate column 
		self.og_opacity['temperature'] = pd.Series(self.og_opacity.index)
		ind = self.og_opacity.loc[np.isnan(self.og_opacity['h2h2'])]

		ntemp = self.og_opacity['h2h2'][0]
		for i in range(int(ntemp)): 
			if i == ntemp-1:
				end = self.og_opacity.shape[0]
			else:
				end = ind.index[i+1]
			new = self.og_opacity.loc[ind.index[i]:end]
			self.og_opacity['temperature'][new.index] = pd.Series([int(self.og_opacity['wno'][ind.index[i]])]*len(new.index))	
		self.og_opacity=self.og_opacity.dropna()

		#slice of dataframe with only values less than the freqmax 
		freqmax = np.max(self.og_opacity['wno'])
		freqlo = np.min(self.og_opacity['wno'])

		#xmean, xmeanhe, xmeanh, xmeanch combined to single dictionary

		#first set up empty dataframe with right temperatures
		columns = [i for i in self.og_opacity.keys() if (i != 'wno')]
		index = list(wno)*int(len(self.og_opacity['temperature'].unique()))
		self.master_opacity = pd.DataFrame(index=index, columns=columns)
		ts=[]
		for i in self.og_opacity['temperature'].unique():
			ts+=[i]*len(wno)
		self.master_opacity['temperature'] = pd.Series(ts, index = self.master_opacity.index)

		#now fill dataframe with interpolated H2 data and set wno (spectral wave) as the index\
		#note here that the pd.index will be repeat for values of temperature
		for i in self.og_opacity['temperature'].unique():
			cia = self.og_opacity.loc[(self.og_opacity['temperature']==i)]
			cia.pop('temperature')
			cia = pandas_interpolate(cia, wno, 'wno', method='linear')
			cia=cia.set_index('wno')
			self.master_opacity.loc[self.master_opacity['temperature']==i,cia.keys()] = cia
		#set everything to greater than freqmax to -33 to make sure interpolation doesn't do anything bad
		self.master_opacity.loc[self.master_opacity.index>freqmax,cia.keys()] = -33

		if linsky:
			#Use Linsky parameterization to fill in -33 gaps, if they exist 
			#remember that the dataframe master_opacity.index is wavenumber
			#this is ONLY adding in the opacity for indices that were previously -33
			#this is a fance way of vectorizing dataframes df['new_column']= np.vectorize(function)(input1, input2.. etc)

			self.master_opacity['h2h2'].loc[self.master_opacity['h2h2']==-33] = np.vectorize(fit_linsky)(self.master_opacity['h2h2'].loc[self.master_opacity['h2h2']==-33], 
														self.master_opacity['temperature'].loc[self.master_opacity['h2h2']==-33],
														self.master_opacity.index[self.master_opacity['h2h2']==-33])
			self.extras += 'H2H2 ;'
			self.references += 'H2H2: Linsky(1969),Lenzuni et al.(1991);'
			self.notes +='H2H2 opacity only added to previous opacities set to -33.;'

			print('ADDING IN LINSKY OPACITY FOR H2H2')

		if h2minus:
			#adding new column to master opacities for H2-
			self.master_opacity['h2m'] = np.vectorize(get_h2minus)(self.master_opacity.index) 

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

	def get_h2minus(wno, temperature):
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
		wno : numpy.ndarray
			wave number cm-1
		temperature :; numpy.ndarray
			temperature in K

		Returns
		-------
		H2 minus opacity in units of log(cm-1 amagat-2)
		"""
		fname = os.path.join(__refdata__, 'opacities','h2minus.csv')
		df = pd.read_csv(fname, skiprows=5, header=0)

		#Per Bell's definition of temperature
		t = 5040.0/df['theta']

		self.extras += 'H2-,'
		self.references += 'H2-: K L Bell 1980 J. Phys. B: At. Mol. Phys. 13 1859, Table 1;'

		return

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
