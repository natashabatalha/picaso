from elements import ELEMENTS as ele 
import json 
import os
from io_utils import read_json
import astropy.units as u
import astropy.constants as c
import pandas as pd
import warnings 
import numpy as np
import wavelength
from numba import jit

__refdata__ = os.environ.get('picaso_refdata')

class ATMSETUP():
	"""
	Reads in default source onfiguration from JSON 
	No parameters yet. Parameters would come in if multiple configs were 
	eventually defined. Currently there is only a single. 
	"""
	def __init__(self, config):
		if __refdata__ is None:
			warnings.warn("Reference data has not been initialized, some functionality will be crippled", UserWarning)
		else: 
			self.ref_dir = os.path.join(__refdata__)
		self.input = config
		self.warnings = []
		self.planet = type('planet', (object,),{})
		self.layer = {}
		self.level = {}
		self.input_wno = np.sort(wavelength.get_input_grid(self.input['opacities']['files']['cld_input_grid'])['wavenumber'].values)
		self.get_constants()

	def get_constants(self):
		"""
		This function gets all conversion factors based on input units and all 
		constants. Goal here is to get everything to cgs and make sure we are only 
		call lengths of things once (e.g. number of wavelength points, number of layers etc)

		Some stuff will be added to this in other routines (e.g. number of layers)
		"""
		self.c = type('c', (object,),{})
		#conversion units
		self.c.pconv = 1e6 #convert bars to cgs 

		#physical constants
		self.c.k_b = (c.k_B.to(u.erg / u.K).value) 		
		self.c.G = c.G.to(u.cm*u.cm*u.cm/u.g/u.s/u.s).value 
		self.c.amu = c.u.to(u.g).value #grams
		self.c.rgas = c.R.value
		self.c.pi = np.pi

		#code constants
		self.c.input_npts_wave = len(self.input_wno) 

		return

	def get_profile(self):
		"""
		Get profile from file or from user input file or direct pandas dataframe. If PT profile 
		is not given, use parameterization 

		Currently only needs inputs from config file

		TO DO
		-----
		- Add regridding to this by having users be able to set a different nlevel than the input cloud code is
		"""

		#get chemistry input from configuration
		chemistry_input = self.input['atmosphere']
		if chemistry_input['profile']['type'] == 'user':

			if chemistry_input['profile']['filepath'] != None:

				read = pd.read_csv(chemistry_input['profile']['filepath'], delim_whitespace=True)

			elif (isinstance(chemistry_input['profile']['profile'],pd.core.frame.DataFrame) or 
					isinstance(chemistry_input['profile']['profile'], dict)): 
					read = chemistry_input['profile']['profile']
			else:
				raise Exception("Provide dictionary or a pointer to pointer to filepath")
		else: 
			raise Exception("TODO: only capability for user is included right now")

		#if a subset is not specified, 
		#determine which of the columns is a molecule by trying to get it's weight 
		
		#COMPUTE THE MOLECULAT WEIGHTS OF THE MOLECULES
		weights = pd.DataFrame({})

		#users have the option of specifying a subset of specified file or pd dataframe
		if isinstance(chemistry_input['molecules']['whichones'], list):
			#make sure that the whichones list has more than one molecule
			num_mol = len(chemistry_input['molecules']['whichones'])
			self.molecules = np.array(chemistry_input['molecules']['whichones'])
			if num_mol >= 1:
				#go through list and compute molecular weights for each one
				for i in chemistry_input['molecules']['whichones']:
					try: 
						weights[i] = pd.Series([self.get_weights([i])[i]])
					except:
						if i == 'e-':
							self.level['electrons'] = read['e-'].values
							self.layer['electrons'] = 0.5*(self.level['electrons'][1:] + self.level['electrons'][:-1])
						else:
							raise Exception("Molecule %s in Subset is not recognized, check list and resubmit" %i)
		else: 
			#if one big file was uploaded, then cycle through each column
			self.molecules = np.array([],dtype=str)
			for i in read.keys():
				try:
					weights[i] = pd.Series([self.get_weights([i])[i]])
					self.molecules = np.concatenate((self.molecules ,np.array([i])))
				except:
					if i == 'e-':
						self.level['electrons'] = read['e-'].values
						self.layer['electrons'] = 0.5*(self.level['electrons'][1:] + self.level['electrons'][:-1])
					else:					#don't raise exception, instead add user warning that a column has been automatically skipped
						self.add_warnings("Ignoring %s in input file, not recognized molecule" % i)
			

		#DEFINE MIXING RATIOS
		self.level['mixingratios'] = read[list(weights.keys())].as_matrix()
		self.layer['mixingratios'] = 0.5*(self.level['mixingratios'][1:,:] + self.level['mixingratios'][:-1,:])
		self.weights = weights

		#GET TP PROFILE 
		#define these to see if they are floats check to see that they are floats 
		T = chemistry_input['PT']['T']
		logg1 = chemistry_input['PT']['logg1']
		logKir = chemistry_input['PT']['logKir']
		logPc = chemistry_input['PT']['logPc']

		#from file
		if ('temperature' in read.keys()) and ('pressure' in read.keys()):
			self.level['temperature'] = read['temperature'].values
			self.level['pressure'] = read['pressure'].values*self.c.pconv #CONVERTING BARS TO DYN/CM2
			self.layer['temperature'] = 0.5*(self.level['temperature'][1:] + self.level['temperature'][:-1])
			self.layer['pressure'] = np.sqrt(self.level['pressure'][1:] * self.level['pressure'][:-1])
		#from parameterization
		elif (isinstance(T,(float,int)) and isinstance(logg1,(float,int)) and 
							isinstance(logKir,(float,int)) and isinstance(logPc,(float,int))): 
			self.profile = calc_TP(T,logKir, logg1, logPc)
		#no other options supported so raise error 
		else:
			raise Exception("There is not adequte information to compute PT profile")

		#Define nlevel and nlayers after profile has been built
		self.c.nlevel = self.level['mixingratios'].shape[0]

		self.c.nlayer = self.c.nlevel - 1		

	def calc_PT(self, T, logKir, logg1, logPc):
		"""
		Calculates parameterized PT profile from Guillot. This isntance is here 
		primary for the retrieval scheme, so this can be updated

		Parameters
		----------
		T : float 
		logKir : float
		logg1 : float 
		logPc : float 
		"""
		raise Exception('TODO: Temperature parameterization option not included yet')
		return pd.DataFrame({'temperature':[], 'pressure':[], 'den':[], 'mu':[]})

	def get_needed_continuum(self):
		"""
		This will define which molecules are needed for the continuum opacities. THis is based on 
		temperature and molecules. Eventually CIA's will expand but we may not necessarily 
		want all of them. 
		'wno','h2h2','h2he','h2h','h2ch4','h2n2']

		To Do
		-----
		- Add in temperature dependent to negate h- and things when not necessary
		"""
		self.continuum_molecules = []
		if "H2" in self.molecules:
			self.continuum_molecules += [['H2','H2']]
		if ("H2" in self.molecules) and ("He" in self.molecules):
			self.continuum_molecules += [['H2','He']]
		if ("H2" in self.molecules) and ("N2" in self.molecules):
			self.continuum_molecules += [['H2','N2']]	
		if 	("H2" in self.molecules) and ("H" in self.molecules):
			self.continuum_molecules += [['H2','H']]
		if 	("H2" in self.molecules) and ("CH4" in self.molecules):
			self.continuum_molecules += [['H2','CH4']]
		if ("H-" in self.molecules):
			self.continuum_molecules += [['H-','bf']]
		if ("H" in self.molecules) and ("electrons" in self.level.keys()):
			self.continuum_molecules += [['H-','ff']]
		if ("H2" in self.molecules) and ("H2-" in self.molecules) and ("e-" in self.molecules):
			self.continuum_molecules += [['H2','H2-']]
		#now we can remove continuum molecules from self.molecules to keep them separate 
		self.molecules = np.array([ x for x in self.molecules if x not in ['H','H2-','H2','H-','He','N2'] ])

	def get_weights(self, molecule):
		"""
		Automatically gets mean molecular weights of any molecule. Requires that 
		user inputs case sensitive molecules i.e. TiO instead of TIO. 

		Parameters
		----------
		molecule : str or list
			any molecule string e.g. "H2O", "CH4" etc "TiO" or ["H2O", 'CH4']

		Returns
		-------
		dict 
			molecule as keys with molecular weights as value
		"""
		weights = {}
		if not isinstance(molecule,list):
			molecule = [molecule]

		for i in molecule:
			molecule_list = []
			for j in range(0,len(i)):
				try:
					molecule_list += [float(i[j])]
				except: 
					if i[j].isupper(): molecule_list += [i[j]] 
					elif i[j].islower(): molecule_list[j-1] =  molecule_list[j-1] + i[j]
			totmass=0
			for j in range(0,len(molecule_list)): 
	            
				if isinstance(molecule_list[j],str):
					elem = ele[molecule_list[j]]
					try:
						num = float(molecule_list[j+1])
					except: 
						num = 1 
					totmass += elem.mass * num
			weights[i] = totmass
		return weights


	#@jit(nopython=True)
	def get_mmw(self):
		"""
		Returns the mean molecular weight of the atmosphere 
		"""
		weighted_matrix = self.level['mixingratios'] @ self.weights.values[0]
		#levels are the edges
		self.level['mmw'] = weighted_matrix
		#layer is the midpoint
		self.layer['mmw'] = 0.5*(weighted_matrix[:-1]+weighted_matrix[1:])
		return 

	#@jit(nopython=True)
	def get_density(self):
		"""
		Calculates density of atmospheres used on TP profile: LEVEL
		"""
		self.level['den'] = self.level['pressure'] / (self.c.k_b * self.level['temperature']) 
		return

	#@jit(nopython=True)
	def get_column_density(self):
		"""
		Calculates the column desntiy based on TP profile: LAYER
		"""
		self.layer['colden'] = (self.level['pressure'][1:] - self.level['pressure'][:-1] ) / self.planet.gravity
		return

	def get_gravity(self):
		"""
		Get gravity based on mass and radius, or gravity inputs 
		"""
		planet_input = self.input['planet']
		if planet_input['gravity'] is not None:
			g = (planet_input['gravity']*u.Unit(planet_input['gravity_unit'])).to('cm/(s**2)')
			g = g.value
		elif (planet_input['mass'] is not None) and (planet_input['radius'] is not None):
			m = (planet_input['mass']*u.Unit(planet_input['mass_unit'])).to(u.g)
			r = ((planet_input['radius']*u.Unit(planet_input['radius_unit'])).to(u.cm))
			g = (self.c.G * m /  (r**2)).value
		else: 
			raise Exception('Need to specify gravity or radius and mass + additional units')
		self.planet.gravity = g 
		return


	def get_clouds(self, wno):
		"""
		Get cloud properties from .cld input returned from eddysed

		Input
		-----
		wno : array
			Array in ascending order of wavenumbers. This is used to regrid the cloud output

		TO DO 
		-----
		- Allow users to add different kinds of "simple" cloud options like "isotropic scattering" or grey 
		opacity at certain pressure. 
		"""

		self.c.output_npts_wave = np.size(wno)

		if self.input['atmosphere']['clouds']['filepath'] != None:
			if os.path.exists(self.input['atmosphere']['clouds']['filepath']):			
				cld_input = pd.read_csv(self.input['atmosphere']['clouds']['filepath'], delim_whitespace = True,
					header=None, skiprows=1, names = ['lvl', 'wv','opd','g0','w0','sigma'])
				#make sure cloud input is has the correct number of waves and PT points
				assert cld_input.shape[0] is not self.c.nlayer*self.c.input_npts_wave, "Cloud input file is not on the same grid as the input PT profile:"
				opd = np.reshape(cld_input['opd'].values, (self.c.nlayer,self.c.input_npts_wave))
				opd = wavelength.regrid(opd, self.input_wno, wno)
				self.layer['cloud'] = {'opd': opd}
				g0 = np.reshape(cld_input['g0'].values, (self.c.nlayer,self.c.input_npts_wave))
				g0 = wavelength.regrid(g0, self.input_wno, wno)
				self.layer['cloud']['g0'] = g0
				w0 = np.reshape(cld_input['w0'].values, (self.c.nlayer,self.c.input_npts_wave))
				w0 = wavelength.regrid(w0, self.input_wno, wno)
				self.layer['cloud']['w0'] = w0  
			else: 
				raise Except('Cld file specified does not exist. Replace with None or find real file')            
		elif self.input['atmosphere']['clouds']['filepath'] == None:
			zeros = np.zeros((self.c.nlayer,self.c.output_npts_wave))
			self.layer['cloud'] = {'w0': zeros}
			self.layer['cloud']['g0'] = zeros
			self.layer['cloud']['opd'] = zeros
		else:
			raise Except("CLD input not recognized. Either input a filepath, or input None")

		return


	def add_warnings(self, warn):
		"""
		Accumulate warnings from ATMSETUP
		"""
		self.warnings += [warn]
		return


