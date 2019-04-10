import numpy as np
import os
import json
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
__refdata__ = os.environ.get('picaso_refdata')
import scipy.signal as sig


class ContinuumFactory():
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
	
	Parameters
	----------
	original_file : str
		Filepath that points to original opacity file (see description above)
	colnames : list 
		defines the sources of opacity in the original file. For the example file above, 
		colnames would be ['wno','h2h2','h2he','h2h','h2ch4','h2n2']
	new_wno : numpy.ndarray, list 
		wavenumber grid to interpolate onto (units of inverse cm)
	overwrite : bool 
		Default is set to False as to not overwrite any existing files. This parameter controls overwriting 
		cia database 

	Todo 
	----
	- Change this to be able to input the filename specified by the input config file
	"""
	def __init__(self, original_file,colnames, new_wno, overwrite=False,new_filename=os.path.join(__refdata__, 'opacities','continuum.json')):
		#original opacity database from freedman, hopefully we can discontinue this soon
		self.new_filename = new_filename
		self.og_opacity = pd.read_csv(original_file,delim_whitespace=True,names=colnames)
		self.temperatures = self.og_opacity['wno'].loc[np.isnan(self.og_opacity[colnames[1]])].values
		self.ntemp = int(self.og_opacity[colnames[1]][0])
		self.nwno = int(self.og_opacity['wno'][0])
		self.og_opacity = self.og_opacity.dropna()
		self.old_wno = self.og_opacity['wno'].unique()
		#define units
		self.w_unit = 'cm-1'
		self.opacity_unit = 'cm-1 amagat^-2'
		self.molecules = colnames[1:]
		
		#this will be what everything is rebinned to
		self.new_wno = new_wno

		#create database file
		dbfile = new_filename
		if os.path.exists(dbfile):
			if overwrite:
				raise Exception("Overwrite is set to false to save db's from being overwritten.")
		self.db_cia = {i:{} for i in self.molecules}# h5py.File(dbfile, driver='mpio', comm=MPI.COMM_WORLD)#, 'w')
		self.db_cia['wave_unit'] = self.w_unit
		self.db_cia['opacity_unit'] = self.opacity_unit
		self.db_cia['wavenumber_grid'] = list(new_wno)
		self.db_cia['temperature_unit'] ='K'


	def restructure_opacity(self):
		for i in range(self.ntemp): 
			for m in self.molecules:
				opa_bundle = self.og_opacity.iloc[ i*self.nwno : (i+1) * self.nwno][m].values
				new_bundle = 10**(np.interp(self.new_wno,  self.old_wno, opa_bundle,right=-33,left=-33))
				#now for anywhere that doesn't have opacity (-33) replace with linsky
				if m=='H2H2':
					h2h2, loc = self.h2h2_overtone(self.temperatures[i],self.new_wno)
					new_bundle[loc] = h2h2
					new_bundle[np.where(new_bundle==1e-33)] = self.fit_linsky(self.temperatures[i],self.new_wno[np.where(new_bundle==1e-33)])
					new_bundle = sig.medfilt(np.array(new_bundle), kernel_size=5) #this is to smooth the discontinuous parts 

				self.db_cia[m][str(self.temperatures[i])] = list(new_bundle) #, chunks=True)
		#get h2minus opacity, currently returns -33 for h2minus opacity regions that are out of bounds
		allw = np.array(list(self.new_wno)*len(self.temperatures))
		allt = np.concatenate([[i]*len(self.new_wno) for i in self.temperatures])

		h2minus = self.get_h2minus(allt, allw)
		self.db_cia['H2-'] = {}
		for i in range(self.ntemp): 
			bundle = h2minus[i*len(self.new_wno) : (i+1) * len(self.new_wno)]
			self.db_cia['H2-'][str(self.temperatures[i])]=list(bundle)#), chunks=True)

		#get hminusbf 
		self.db_cia['H-bf'] = {}
		for i in range(self.ntemp):
			if self.temperatures[i]<600.0:
				bundle = np.zeros(len(self.new_wno)) 
			else:
				bundle = self.get_hminusbf(self.new_wno)
			self.db_cia['H-bf'][str(self.temperatures[i])] = list(bundle)#), chunks=True)

		#get hminusff 
		self.db_cia['H-ff'] = {}
		for i in range(self.ntemp):
			bundle = self.get_hminusff(self.temperatures[i], self.new_wno) 
			self.db_cia['H-ff'][str(self.temperatures[i])]=list(bundle)#), chunks=True)

		with open(self.new_filename, 'w') as fp:
			json.dump(self.db_cia, fp)
	def h2h2_overtone(self, t, wno):
		"""
		Add in special CIA h2h2 band at 0.8 microns

		Parameters
		---------- 
		t : float
			Temperature 
		wno : numpy.array
			wave number

		Returns
		-------
		H2-H2 absorption in cm-1 amagat-2 		
		"""
		fname = os.path.join(__refdata__, 'opacities','H2H2_ov2_eq.tbl')
		df = pd.read_csv(fname, delim_whitespace=True).set_index('wavenumber').apply(np.log10)		
		temps = [ float(i) for i in df.keys()]
		it = find_nearest(temps, t)
		placeholder_temp = temps[it]
		loc = np.where((wno>=df.index.min()) & (wno<=df.index.max()))
		new_opa = 10**(np.interp(wno[loc],  np.array(df.index), df[list(df.keys())[it]].values,right=-33,left=-33))
		return new_opa, loc

	def fit_linsky(self, t, wno, va=3):
		"""
		Adds H2-H2 opacity from Linsky (1969) and Lenzuni et al. (1991) 
		to values that were set to -33 as place holders. 

		Parameters
		---------- 
		t : float
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
		va = va-1
		w = sig0[va]

		d=d3[va]*np.sqrt(d1[va]+d2[va]*t)
		a=10**(a1[va]+a2[va]*np.log10(t))
		b=10**(b1[va]+b2[va]*np.log10(t))
		aa=4.0/13.0*a/d*np.exp(1.5*d/b)
		kappa = aa*wno*np.exp(-(wno-w)/b)
		smaller = np.where(wno<w)

		if len(smaller)>0:
			kappa[smaller] = a*d*wno[smaller]*np.exp((wno[smaller]-w)/0.6952/t)/((wno[smaller]-w)**2+d*d)
		even_smaller = np.where(wno<w+1.5*d)
		if len(even_smaller)>0:
			kappa[even_smaller]=a*d*wno[even_smaller]/((wno[even_smaller]-w)**2+d*d)

		return kappa

	def get_h2minus(self, t, wno):
		"""
		This results gives the H2 minus opacity, needed for temperatures greater than 600 K. 
		K L Bell 1980 J. Phys. B: At. Mol. Phys. 13 1859, Table 1 
		theta=5040/T(K)

		The result is given in cm4/dyn, which is why will will  multiply by nh2*ne*k*T
		where:
		nh2: number of h2 molecules/cm3
		ne: number of electrons/cm3
		This will happen when we go to sum opacities. For now returns will be in cm4/dyn
		
		Parameters
		----------
		t : numpy.ndarray
			temperature in K
		wno : numpy.ndarray
			wave number cm-1

		Returns
		-------
		H2 minus opacity in units of cm4/dyn
		"""
		fname = os.path.join(__refdata__, 'opacities','h2minus.csv')
		df = pd.read_csv(fname, skiprows=5, header=0).set_index('theta').apply(np.log10)

		#Bell+1980 wavenumber
		wno_bell = 1e8/df.columns.astype(float).values


		#if lower than our grid 
		new_t = 5040.0/t
		new_w = wno

		h2m_interpolator =  RegularGridInterpolator((df.index.values,wno_bell), df.as_matrix(), bounds_error=False,fill_value=None)#)bounds_error=False,fill_value=None)

		new = np.c_[new_t, new_w]
		pts = 10**h2m_interpolator(new)*1e-26 #units from Bell 1980
		#zeros for out of bound temperatures
		pts[np.where(new_t>np.max(df.index.values))] = 0 
		return pts

	def get_hminusbf(self,wno):
		"""
		H- bound free opacity, which is only dependent on wavelength. From John 1988 http://adsabs.harvard.edu/abs/1988A%26A...193..189J

		Parameters
		----------
		wno :  numpy.ndarray 
			Wavenumber in cm-1

		Returns 
		-------
		array of floats 
			Absorption coefficient in units of cm2
		"""
		coeff = np.array([152.519,49.534,-118.858,92.536, -34.194,4.982])[::-1]
		lambda_0 = 1.6419
		wave =1e4/wno
		x=np.sqrt(1.0/wave-1.0/lambda_0)
		f = np.zeros(np.size(wave))
		nonzero = np.where(wno > 1e4/lambda_0)
		for i in coeff: 
			f[nonzero] = f[nonzero]*x[nonzero] + i 
		result = (wave*x)**3*f*1e-18
		result[np.isnan(result)] = 1e-33
		return result
		
	def get_hminusff(self, t, wno):
		"""
		H- free free opacity, which is both wavelength and temperature dependent. 
		From Bell & Berrington (1987)
		Also includes factor for simulated emission 
		
		Parameters
		----------
		t :  float
			Temperature in K 
		wno :  numpy.ndarray
			Wavenumber in cm-1

		Returns
		-------
		array of float
			Gives cross section in cm^5
		"""
		AJ1= [0.e0, 2483.346, -3449.889, 2200.040, -696.271, 88.283]
		BJ1= [0.e0, 285.827, -1158.382, 2427.719, -1841.400, 444.517]
		CJ1= [0.e0, -2054.291, 8746.523, -13651.105, 8624.970, -1863.864]
		DJ1= [0.e0, 2827.776, -11485.632, 16755.524, -10051.530, 2095.288]
		EJ1= [0.e0, -1341.537, 5303.609, -7510.494, 4400.067,  -901.788]
		FJ1= [0.e0, 208.952, -812.939, 1132.738, -655.020, 132.985]
		AJ2= [518.1021, 473.2636, -482.2089, 115.5291, 0.0,0.0]
		BJ2= [-734.8666, 1443.4137, -737.1616, 169.6374, 0.0,0.0]
		CJ2= [1021.1775, -1977.3395, 1096.8827, -245.649, 0.0,0.0]
		DJ2= [-479.0721, 922.3575, -521.1341, 114.243, 0.0,0.0]
		EJ2= [93.1373, -178.9275, 101.7963, -21.9972,0.0,0.0]
		FJ2= [-6.4285, 12.3600, -7.0571, 1.5097, 0.0,0.0	]

		wave = 1e4/wno 

		nwave = np.size(wave)

		if t<800 : 
			return np.zeros(nwave ) 

		t_coeff = 5040.0/t



		hj = np.zeros((6, nwave))
		longw = np.where(wave>0.3645)
		midw = np.where((wave<=0.3645))
		shortw = np.where(wave<0.1823)
		wave[shortw] = 0.1823
		for i in range(6):
			hj[i,longw] = 1e-29*(wave[longw]*wave[longw]*AJ1[i] + BJ1[i] + (CJ1[i] + (
							DJ1[i] + (EJ1[i] + FJ1[i]/wave[longw])/wave[longw])/wave[longw])/wave[longw])
			hj[i,midw] = 1e-29*(wave[midw]*wave[midw]*AJ2[i] + BJ2[i] + (
							CJ2[i] + (DJ2[i] +(EJ2[i] + FJ2[i]/wave[midw])/wave[midw])/wave[midw])/wave[midw])

		hm_cx = np.zeros(nwave)
		for i in range(6):
			hm_cx  += t_coeff**((i+1)/2.0)*hj[i, :]

		#this parameterization is not valid past 20 micron..
		past20 = np.where(wave>20.0)

		if np.size(past20) > 0 :
			hm_cx[past20] = np.zeros(np.size(past20)) 

		return	hm_cx * 1.380658e-16 * t 

class MolecularFactory():
	"""
	This class contains everything needed to take Richard's Opacities and turn them into 
	usable json database for fast querying.

	This directly translates Richards old opacities from the original albedo code to the new 
	json format. Each molecule has the 1060 files with the following header structure: 

	ch4_2014 opacities from 0.3 to 1 microns at R~5000 resolution 	
	wavelength 	 opacity (cm2/g)  

	Warnings
	--------
	When this routine puts everything into the json database it will reorder everything 
	by wavenumber!! This way molcular opacity and continuum opacity are on the same grid 

	Parameters
	----------
	original_dir : str 
		Pointer to the original directory that contains all the opacities. This folder should 
		contain a bunch of different folders (one for each molecule). 
	
	"""
	def __init__(self, original_dir, new_wno, overwrite=False,new_filename=os.path.join(__refdata__, 'opacities','molecular.json')):
		#define units
		self.w_unit = 'cm-1'
		self.opacity_unit = 'cm2/g'

		#this will be what everything is rebinned to
		self.new_wno = new_wno

		self.original_dir = original_dir

		#create database file
		dbfile = new_filename
		if os.path.exists(dbfile):
			if overwrite:
				raise Exception("Overwrite is set to false to save db's from being overwritten.")
		self.db_mole = {}

		self.db_mole.attrs['wave_unit'] = w_unit
		self.db_mole.attrs['opacity_unit'] = opacity_unit
		self.db_mole.attrs['wavenumber_grid'] = new_wno
		self.db_molecular.attrs['prssure_unit'] ='bars'
		self.db_molecular.attrs['temperature_unit'] ='K'

	def restructure_opacity(self):
		dset = self.db_mole.create_dataset('wavenumber', data=self.new_wno)#, chunks=True)

		for fold in next(os.walk(self.original_dir))[1]:
			print(fold)
			file = list(listdir(os.path.join(self.original_dir, fold)))
			#if (fold == 'Na') or (fold =='K'):
			#	print('skip')
			#	continue
			if len(file)==1060:
				grid = pd.read_csv(os.path.join(self.original_dir, 'PTgrid1060.txt'),delim_whitespace=True,skiprows=1,
					header=None, names=['i','p','t'], dtype=str)
			elif len(file)==736:
				grid = pd.read_csv(os.path.join(self.original_dir, 'PTgrid736.txt'),delim_whitespace=True,skiprows=1,
					header=None, names=['i','p','t'],dtype=str)
			else:
				raise Exception('Not on 736 or 1060 grid')

			pressures=grid['p'].values
			temperatures=grid['t'].values

			file1 = file[0][0:file[0].find('.')+1]
			file2 = file[0][file[0].rfind('.'):]
			for i in np.linspace(1,len(file),len(file),dtype=int):
				a = pd.read_csv(os.path.join(self.original_dir, fold,file1+str(i)+file2), delim_whitespace=True, skiprows=2, header=None, names=['wave','cm2/g'])
				a['wno'] = 1e4/a['wave']
				a = a.sort_values('wno')
				new_bundle = a['cm2/g'].values
				dset = self.db_mole.create_dataset(fold+'/'+temperatures[i-1]+'/'+pressures[i-1], data=new_bundle)#, chunks=True)
				
def find_nearest(array,value):
	#small program to find the nearest neighbor in temperature  
	idx = (np.abs(array-value)).argmin()
	return idx

def listdir(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f