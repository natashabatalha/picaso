from elements import ELEMENTS as ele 
import json 
import os
from io_utils import read_json
import astropy.units as u
import astropy.constants as c
import pandas as pd
import warnings 
import numpy as np
#CALL ATMSETUP(NLEVEL,Z,TEMP,PRESS,DEN,XMU,
#     & CH4,H2,HE,XN2,CO,H2O,CO2,XNH3,H,Hmin,elec,xNa,xK,RB,CS,FEH,CRH,
#     & TIO, VO, x736,x1060,IPRINT)

__refdata__ = os.environ.get('albedo_refdata')

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

	def get_profile(self):
		"""
		Get profile from file or from user input file or direct pandas dataframe. If PT profile 
		is not given, use parameterization 

		Currently only needs inputs from config file
		"""

		#get chemistry input from configuration
		chemistry_input = self.input['chemistry']
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
		self.weights = pd.DataFrame({})

		#users have the option of specifying a subset of specified file or pd dataframe
		if isinstance(chemistry_input['molecules']['whichones'], list):
			#make sure that the whichones list has more than one molecule
			num_mol = len(chemistry_input['molecules']['whichones'])
			if num_mol >= 1:
				#go through list and compute molecular weights for each one
				for i in chemistry_input['molecules']['whichones']:
					try: 
						self.weights[i] = pd.Series([self.get_weights([i])[i]])
					except:
						raise Exception("Molecule %s in Subset is not recognized, check list and resubmit" %i)

				#GET THE NECESSARY MIXING RATIO COLUMNS  
				self.mixingratios = read[list(self.weights.keys())]

		else: 
			#if one big file was uploaded, then cycle through each column
			for i in read.keys():
				try:
					self.weights[i] = pd.Series([self.get_weights([i])[i]])
				except:
					#don't raise exception, instead add user warning that a column has been automatically skipped
					self.add_warnings("Ignoring %s in input file, not recognized molecule" % i)
			
			#get the mixing ratios dependending on which files made sense 
			#GET THE NECESSARY MIXING RATIO COLUMNS
			self.mixingratios = read[list(self.weights.keys())]

		#GET TP PROFILE 

		#define these to see if they are floats check to see that they are floats 
		T = chemistry_input['PT']['T']
		logg1 = chemistry_input['PT']['logg1']
		logKir = chemistry_input['PT']['logKir']
		logPc = chemistry_input['PT']['logPc']		

		#from file
		if ('temperature' in read.keys()) and ('pressure' in read.keys()):
			self.profile = read[['temperature', 'pressure']].copy()
		#from parameterization
		elif (isinstance(T,(float,int)) and isinstance(logg1,(float,int)) and 
							isinstance(logKir,(float,int)) and isinstance(logPc,(float,int))): 
			self.profile = calc_TP(T,logKir, logg1, logPc)
		#no other options supported so raise error 
		else:
			raise Exception("There is not adequte information to compute PT profile")

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
		return pd.DataFrame({'temperature':{}, 'pressure':[], 'den':[], 'mu':[]})


	def get_weights(self, molecule):
		"""
		Automatically gets weights of any molecule. Requires that 
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



	def get_mmw(self):
		"""
		Returns the mean molecular weight of the atmosphere 
		"""
		weighted_matrix = np.dot(self.weights.as_matrix(),self.mixingratios.as_matrix().transpose())
		self.profile['mu'] = pd.Series(weighted_matrix[0])
		return 

	def get_density(self):
		"""
		Calculates density of atmospheres ased on TP profile
		"""
		self.profile['den'] = self.profile['pressure'] * 1e6 / (c.k_B.to(u.erg / u.K).value * self.profile['temperature']) 
		return

	def get_gravity(self):
		"""
		Get gravity based on mass and radius, or gravity inputs 
		"""
		planet_input = self.input['planet']
		if planet_input['gravity'] is not None:
			g = (planet_input['gravity']*u.Unit(planet_input['gravity_unit'])).to('m/(s**2)')
			g = g.value
		elif (planet_input['mass'] is not None) and (planet_input['radius'] is not None):
			m = (planet_input['mass']*u.Unit(planet_input['mass_unit'])).to(u.kg)
			r = ((planet_input['radius']*u.Unit(planet_input['radius_unit'])).to(u.m))
			g = (c.G * m /  (r**2)).value
		else: 
			raise Exception('Need to specify gravity or radius and mass + additional units')
		self.planet.gravity = g 
		return

	def add_warnings(self, warn):
		"""
		Accumulate warnings from ATMSETUP
		"""
		self.warnings += [warn]
		return

	def get_layer(self):
		"""
		Calculates the 
		"""
