from .atmsetup import ATMSETUP
from .fluxes import get_reflected_1d, get_reflected_3d , get_thermal_1d
from .wavelength import get_cld_input_grid
import numpy as np
import pandas as pd
from .optics import RetrieveOpacities,compute_opacity
import os
import pickle as pk
from .disco import get_angles, compute_disco, compress_disco, compress_thermal
import copy
import json
import pysynphot as psyn
import astropy.units as u
import astropy.constants as c
__refdata__ = os.environ.get('picaso_refdata')

def picaso(bundle,opacityclass, dimension = '1d',calculation='reflected', full_output=False, plot_opacity= False):
	"""
	Currently top level program to run albedo code 

	Parameters 
	----------
	bundle : dict 
		This input dict is built by loading the input = `justdoit.load_inputs()` 
	dimension : str 
		(Optional) Dimensions of the calculation. Default = '1d'. But '3d' is also accepted. 
		In order to run '3d' calculations, user must build 3d input (see tutorials)
	full_output : bool 
		(Optional) Default = False. Returns atmosphere class, which enables several 
		plotting capabilities. 
	plot_opacity : bool 
		(Optional) Default = False, Creates pop up of the weighted opacity

	Return
	------
	Wavenumber, albedo if full_output=False 
	Wavenumber, albedo, atmosphere if full_output = True 
	"""
	inputs = bundle.inputs

	wno = opacityclass.wno
	nwno = opacityclass.nwno

	#check to see if we are running in test mode
	test_mode = inputs['test_mode']

	#set approx numbers options (to be used in numba compiled functions)
	single_phase = inputs['approx']['single_phase']
	multi_phase = inputs['approx']['multi_phase']
	raman_approx =inputs['approx']['raman']

	#phase angle 
	phase_angle = inputs['phase_angle']

	#parameters needed for the two term hg phase function. 
	#Defaults are set in config.json
	f = inputs['approx']['TTHG_params']['fraction']
	frac_a = f[0]
	frac_b = f[1]
	frac_c = f[2]
	constant_back = inputs['approx']['TTHG_params']['constant_back']
	constant_forward = inputs['approx']['TTHG_params']['constant_forward']

	#get geometry
	ng = inputs['disco']['num_gangle']
	nt = inputs['disco']['num_tangle']

	gangle,gweight,tangle,tweight = get_angles(ng, nt) 
	#planet disk is divided into gaussian and chebyshev angles and weights for perfoming the 
	#intensity as a function of planetary pahse angle 
	ubar0, ubar1, cos_theta,lat,lon = compute_disco(ng, nt, gangle, tangle, phase_angle)

	#set star 
	radius_star = inputs['star']['radius']
	F0PI = np.zeros(nwno) + 1.0 

	#define approximinations 
	delta_eddington = inputs['approx']['delta_eddington']

	#begin atm setup
	atm = ATMSETUP(inputs)

	################ From here on out is everything that would go through retrieval or 3d input##############
	atm.planet.gravity = inputs['planet']['gravity']
	atm.planet.radius = inputs['planet']['radius']

	if dimension == '1d':
		atm.get_profile()
	elif dimension == '3d':
		atm.get_profile_3d()

	#now can get these 
	atm.get_mmw()
	atm.get_density()
	atm.get_column_density()
	#get needed continuum molecules 
	atm.get_needed_continuum()

	#get cloud properties, if there are any and put it on current grid 
	atm.get_clouds(wno)

	#determine surface reflectivity as function of wavelength (set to zero here)
	#TODO: Should be an input
	atm.get_surf_reflect(nwno) 

	#Make sure that all molecules are in opacityclass. If not, remove them and add warning
	no_opacities = [i for i in atm.molecules if i not in opacityclass.molecules]
	atm.add_warnings('No computed opacities for: '+','.join(no_opacities))
	atm.molecules = np.array([ x for x in atm.molecules if x not in no_opacities ])

	#lastly grab needed opacities for the problem
	opacityclass.get_opacities(atm)

	if dimension == '1d':
		#only need to get opacities for one pt profile

		#There are two sets of dtau,tau,w0,g in the event that the user chooses to use delta-eddington
		#We use HG function for single scattering which gets the forward scattering/back scattering peaks 
		#well. We only really want to use delta-edd for multi scattering legendre polynomials. 
		DTAU, TAU, W0, COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG= compute_opacity(
			atm, opacityclass,delta_eddington=delta_eddington,test_mode=test_mode,raman=raman_approx,
			full_output=full_output, plot_opacity=plot_opacity)

		if  'reflected' in calculation:
			#use toon method (and tridiagonal matrix solver) to get net cumulative fluxes 
			xint_at_top  = get_reflected_1d(atm.c.nlevel, wno,nwno,ng,nt,
													DTAU, TAU, W0, COSB,GCOS2,ftau_cld,ftau_ray,
													DTAU_OG, TAU_OG, W0_OG, COSB_OG ,
													atm.surf_reflect, ubar0,ubar1,cos_theta, F0PI,
													single_phase,multi_phase,
													frac_a,frac_b,frac_c,constant_back,constant_forward)
		if 'thermal' in calculation:
			#use toon method (and tridiagonal matrix solver) to get net cumulative fluxes 
			flux_at_top  = get_thermal_1d(atm.c.nlevel, wno,nwno,ng,nt,atm.level['temperature'],
													DTAU_OG, W0_OG, COSB_OG, atm.level['pressure'],ubar1)
			
	elif dimension == '3d':

		#setup zero array to fill with opacities
		TAU_3d = np.zeros((atm.c.nlevel, nwno, ng, nt))
		DTAU_3d = np.zeros((atm.c.nlayer, nwno, ng, nt))
		W0_3d = np.zeros((atm.c.nlayer, nwno, ng, nt))
		COSB_3d = np.zeros((atm.c.nlayer, nwno, ng, nt))
		GCOS2_3d = np.zeros((atm.c.nlayer, nwno, ng, nt))
		FTAU_CLD_3d = np.zeros((atm.c.nlayer, nwno, ng, nt))
		FTAU_RAY_3d = np.zeros((atm.c.nlayer, nwno, ng, nt))
		#these are the unchanged values from delta-eddington
		TAU_OG_3d = np.zeros((atm.c.nlevel, nwno, ng, nt))
		DTAU_OG_3d = np.zeros((atm.c.nlayer, nwno, ng, nt))
		W0_OG_3d = np.zeros((atm.c.nlayer, nwno, ng, nt))
		COSB_OG_3d = np.zeros((atm.c.nlayer, nwno, ng, nt))

		#get opacities at each facet
		for g in range(ng):
			for t in range(nt): 

				#edit atm class to only have subsection of 3d stuff 
				atm_1d = copy.deepcopy(atm)

				#diesct just a subsection to get the opacity 
				atm_1d.disect(g,t)

				dtau, tau, w0, cosb,ftau_cld, ftau_ray, gcos2, DTAU_OG, TAU_OG, W0_OG, COSB_OG = compute_opacity(
					atm_1d, opacityclass,delta_eddington=delta_eddington,test_mode=test_mode,raman=raman_approx)

				DTAU_3d[:,:,g,t] = dtau
				TAU_3d[:,:,g,t] = tau
				W0_3d[:,:,g,t] = w0 
				COSB_3d[:,:,g,t] = cosb
				GCOS2_3d[:,:,g,t]= gcos2 
				FTAU_CLD_3d[:,:,g,t]= ftau_cld
				FTAU_RAY_3d[:,:,g,t]= ftau_ray
				#these are the unchanged values from delta-eddington
				TAU_OG_3d[:,:,g,t] = TAU_OG
				DTAU_OG_3d[:,:,g,t] = DTAU_OG
				W0_OG_3d[:,:,g,t] = W0_OG
				COSB_OG_3d[:,:,g,t] = COSB_OG



		#use toon method (and tridiagonal matrix solver) to get net cumulative fluxes 
		xint_at_top  = get_reflected_3d(atm.c.nlevel, wno,nwno,ng,nt,
											DTAU_3d, TAU_3d, W0_3d, COSB_3d,GCOS2_3d, FTAU_CLD_3d,FTAU_RAY_3d,
											DTAU_OG_3d, TAU_OG_3d, W0_OG_3d, COSB_OG_3d,
											atm.surf_reflect, ubar0,ubar1,cos_theta, F0PI,
											single_phase,multi_phase,
											frac_a,frac_b,frac_c,constant_back,constant_forward)

	#now compress everything based on the weights 
	if  ('reflected' in calculation) & ('thermal' not in calculation):
		albedo = compress_disco(nwno, cos_theta, xint_at_top, gweight, tweight,F0PI)
		returns = (wno, albedo)

	elif ('reflected' not in calculation) & ('thermal' in calculation):
		thermal = compress_thermal(nwno,ubar1, flux_at_top, gweight, tweight)
		fpfs_thermal = thermal/(opacityclass.unshifted_stellar_spec)*(atm.planet.radius/radius_star)**2.0
		returns = wno,fpfs_thermal,thermal

	elif ('reflected' in calculation) & ('thermal' in calculation):
		albedo = compress_disco(nwno, cos_theta, xint_at_top, gweight, tweight,F0PI)
		thermal = compress_thermal(nwno,ubar1,flux_at_top, gweight, tweight)
		fpfs_thermal = thermal/(opacityclass.unshifted_stellar_spec)*(atm.planet.radius/radius_star)**2.0
		returns = wno,albedo, fpfs_thermal,thermal

	if full_output:	
		#add full solution and latitude and longitudes to the full output
		#atm.flux_at_top = flux_at_top
		atm.xint_at_top = xint_at_top
		atm.latitude = lat
		atm.longitude = lon
		return wno, albedo , atm.as_dict()
	else: 
		return returns

def opannection(filename_db = None, raman_db = None):
	"""
	Sets up database connection to opacities. 
	"""

	inputs = json.load(open(os.path.join(__refdata__,'config.json')))

	if isinstance(filename_db,type(None) ): filename_db = os.path.join(__refdata__, 'opacities', inputs['opacities']['files']['opacity'])
	if isinstance(raman_db,type(None) ): raman_db = os.path.join(__refdata__, 'opacities', inputs['opacities']['files']['raman'])

	opacityclass=RetrieveOpacities(
				filename_db, 
				raman_db
				)
	return opacityclass

class inputs():
	"""Class to setup planet to run

	Parameters
	----------
	continuum_db: str 
		(Optional)Filename that points to the HDF5 contimuum database (see notebook on swapping out opacities).
		This database pointer will also define the wavelength grid to run on. So, wavenumber grid 
		should be specified in the database as an attribute!
		Default will pull the zenodo defualt opacities 
	molecular_db: str 
		(Optional)Filename that points to the HDF5 molecular database (see notebook on swapping out opacities).
		This database pointer will also define the wavelength grid to run on. So, wavenumber grid 
		should be specified in the database as an attribute!
		Default will pull the zenodo defualt opacities 
	raman_db: str 
		(Optional)Filename that points to the raman scattering database. Default is the text file from Ocklopcic+2018 paper 
		on exoplanet raman scattering. 

	Attributes
	----------
	phase_angle() : set phase angle
	gravity() : set gravity
	star() : set stellar spec
	atmosphere() : set atmosphere composition and PT
	clouds() : set cloud profile
	approx()  : set approximation
	spectrum() : create spectrum
	"""
	def __init__(self):#continuum_db = None, molecular_db = None, raman_db = None):

		self.inputs = json.load(open(os.path.join(__refdata__,'config.json')))

		#if isinstance(continuum_db,type(None) ): continuum_db = os.path.join(__refdata__, 'opacities', self.inputs['opacities']['files']['continuum'])
		#if isinstance(molecular_db,type(None) ): molecular_db = os.path.join(__refdata__, 'opacities', self.inputs['opacities']['files']['molecular'])
		#if isinstance(raman_db,type(None) ): raman_db = os.path.join(__refdata__, 'opacities', self.inputs['opacities']['files']['raman'])

		#self.opacityclass=RetrieveOpacities(
		#			continuum_db,
		#			molecular_db, 
		#			raman_db
		#			)


	def phase_angle(self, phase=0,num_gangle=10, num_tangle=10):
		"""Define phase angle and number of gauss and tchebychev angles to compute. 
		
		phase : float,int
			Phase angle in radians 
		num_gangle : int 
			Number of Gauss angles to integrate over facets (Default is 10).
			 Higher numbers will slow down code. 
		num_tangle : int 
			Number of Tchebyshev angles to integrate over facets (default is 10)
		"""
		if (num_gangle < 2 ) or (num_tangle < 2 ): raise Exception("length of gangle and tangle must be > than 2")
		self.inputs['phase_angle'] = phase
		self.inputs['disco']['num_gangle'] = int(num_gangle)
		self.inputs['disco']['num_tangle'] = int(num_tangle)

	def gravity(self, gravity=None, gravity_unit=None, 
		              radius=None, radius_unit=None, 
		              mass = None, mass_unit=None):
		"""
		Get gravity based on mass and radius, or gravity inputs 

		Parameters
		----------
		gravity : float 
			(Optional) Gravity of planet 
		gravity_unit : astropy.unit
			(Optional) Unit of Gravity
		radius : float 
			(Optional) radius of planet MUST be specified for thermal emission!
		radius_unit : astropy.unit
			(Optional) Unit of radius
		mass : float 
			(Optional) mass of planet 
		mass_unit : astropy.unit
			(Optional) Unit of mass	
		"""
		if (mass is not None) and (radius is not None):
			m = (mass*mass_unit).to(u.g)
			r = (radius*radius_unit).to(u.cm)
			g = (c.G * m /  (r**2)).value
			self.inputs['planet']['radius'] = r.value
			self.inputs['planet']['radius_unit'] = 'cm'
			self.inputs['planet']['mass'] = m.value
			self.inputs['planet']['mass_unit'] = 'g'
			self.inputs['planet']['gravity'] = g
			self.inputs['planet']['gravity_unit'] = 'cm/(s**2)'
		elif gravity is not None:
			g = (gravity*gravity_unit).to('cm/(s**2)')
			g = g.value
			self.inputs['planet']['gravity'] = g
			self.inputs['planet']['gravity_unit'] = 'cm/(s**2)'
			self.inputs['planet']['radius'] = np.nan
			self.inputs['planet']['radius_unit'] = 'Radius not specified'
		else: 
			raise Exception('Need to specify gravity or radius and mass + additional units')

	def star(self, opannection,temp=None, metal=None, logg=None ,radius = None, radius_unit=None,
		database='ck04models',filename=None, w_units=None, f_units=None):
		"""
		Get the stellar spectrum using pysynphot and interpolate onto a much finer grid than the 
		planet grid. 

		Parameters
		----------
		opannection : class picaso.RetrieveOpacities
			This is the opacity class and it's needed to get the correct wave info and raman scattering cross sections
		temp : float 
			Teff of the stellar model 
		metal : float 
			Metallicity of the stellar model 
		logg : float 
			Logg cgs of the stellar model
		radius : float 
			Radius of the star 
		radius_unit : astropy.unit
			Any astropy unit (e.g. `radius_unit=astropy.unit.Unit("R_sun")`)
		database : str 
			(Optional)The database to pull stellar spectrum from. See documentation for pysynphot. 
		filename : str 
			(Optional) Upload your own stellar spectrum. File format = two column white space (wave, flux)
		wunits : str 
			(Optional) Used for stellar file wave units 
		funits : str 
			(Optional) Used for stellar file flux units 
		"""
		#most people will just upload their thing from a database
		if (not isinstance(radius, type(None))):
			r = (radius*radius_unit).to(u.cm).value
			radius_unit='cm'
		else :
			r = np.nan
			radius_unit = "Radius not supplied"

		if (not isinstance(temp, type(None))):
			sp = psyn.Icat(database, temp, metal, logg)
			sp.convert("um")
			sp.convert('flam') 
			wno_star = 1e4/sp.wave[::-1] #convert to wave number and flip
			flux_star = sp.flux[::-1]*1e8	 #flip here and convert to ergs/cm3/s to get correct order

		#but you can also upload a stellar spec of your own 
		elif (not isinstance(filename),type(none)):
			star = np.genfromtxt(filename, dtype=(float, float), names='w, f')
			flux = star['f']
			wave = star['w']
			#sort if not in ascending order 
			sort = np.array([wave,flux]).T
			sort= sort[sort[:,0].argsort()]
			wave = sort[:,0]
			flux = sort[:,1] 
			if w_unit == 'um':
				WAVEUNITS = 'um' 
			elif w_unit == 'nm':
				WAVEUNITS = 'nm'
			elif w_unit == 'cm' :
				WAVEUNITS = 'cm'
			elif w_unit == 'Angs' :
				WAVEUNITS = 'angstrom'
			elif w_unit == 'Hz' :
				WAVEUNITS = 'Hz'
			else: 
				raise Exception('Stellar units are not correct. Pick um, nm, cm, hz, or Angs')        

			#http://www.gemini.edu/sciops/instruments/integration-time-calculators/itc-help/source-definition
			if f_unit == 'Jy':
				FLUXUNITS = 'jy' 
			elif f_unit == 'FLAM' :
				FLUXUNITS = 'FLAM'
			elif f_unit == 'erg/cm2/s/Hz':
				flux = flux*1e23
				FLUXUNITS = 'jy' 
			else: 
				raise Exception('Stellar units are not correct. Pick FLAM or Jy or erg/cm2/s/Hz')

			sp = psyn.ArraySpectrum(wave, flux, waveunits=WAVEUNITS, fluxunits=FLUXUNITS)        #Convert evrything to nanometer for converstion based on gemini.edu  
			sp.convert("um")
			sp.convert('flam') #ergs/cm2/s/ang
			wno_star = 1e4/sp.wave[::-1] #convert to wave number and flip
			flux_star = sp.flux[::-1]*1e8 #flip and convert to ergs/cm3/s here to get correct order			


		wno_planet = opannection.wno
		max_shift = np.max(wno_planet)+6000 #this 6000 is just the max raman shift we could have 
		min_shift = np.min(wno_planet) -2000 #it is just to make sure we cut off the right wave ranges

		#do a fail safe to make sure that star is on a fine enough grid for planet case 
		fine_wno_star = np.linspace(min_shift, max_shift, len(wno_planet)*5)
		fine_flux_star = np.interp(fine_wno_star,wno_star, flux_star)

		#this adds stellar shifts 'self.raman_stellar_shifts' to the opacity class
		#the cross sections are computed later 
		opannection.compute_stellar_shits(fine_wno_star, fine_flux_star)

		self.inputs['star']['database'] = database
		self.inputs['star']['temp'] = temp
		self.inputs['star']['logg'] = logg
		self.inputs['star']['metal'] = metal
		self.inputs['star']['radius'] = r 
		self.inputs['star']['radius_unit'] = radius_unit 
		self.inputs['star']['flux'] = fine_flux_star 
		self.inputs['star']['wno'] = fine_wno_star 

	def atmosphere(self, df=None, filename=None, pt_params = None,exclude_mol=None, **pd_kwargs):
		"""
		Builds a dataframe and makes sure that minimum necessary parameters have been suplied. 

		Parameters
		----------
		df : pandas.DataFrame
			(Optional) Dataframe with volume mixing ratios and pressure, temperature profile. 
			Must contain pressure (bars) at least one molecule
		filename : str 
			(Optional) Filename with pressure, temperature and volume mixing ratios.
			Must contain pressure at least one molecule
		exclude_mol : list of str 
			(Optional) List of molecules to ignore from file
		pt_params : list of float 
			(Optional) list of [T, logKir, logg1,logg2, alpha] from Guillot+2010
		pd_kwargs : kwargs 
			Key word arguments for pd.read_csv to read in supplied atmosphere file 
		"""

		if not isinstance(df, type(None)):
			if ((not isinstance(df, dict )) & (not isinstance(df, pd.core.frame.DataFrame ))): 
				raise Exception("df must be pandas DataFrame or dictionary")
			else:
				self.nlevel=df.shape[0] 
		elif not isinstance(filename, type(None)):
			df = pd.read_csv(filename, **pd_kwargs)
			self.nlevel=df.shape[0] 
		elif not isinstance(pt_params, type(None)): 
			self.inputs['atmosphere']['pt_params'] = pt_params
			self.nlevel=61 #default n of levels for parameterization
			raise Exception('Pt parameterization not in yet')

		if 'pressure' not in df.keys(): 
			raise Exception("Check column names. `pressure` must be included.")

		if (('temperature' not in df.keys()) and (isinstance(pt_params, type(None)))):
			raise Exception("`temperature` not specified and pt_params not given. Do one or the other.")



		if not isinstance(exclude_mol, type(None)):
			df = df.drop(exclude_mol, axis=1)
			self.inputs['atmosphere']['exclude_mol'] = exclude_mol

		self.inputs['atmosphere']['profile'] = df.sort_values('pressure').reset_index(drop=True)


	def clouds(self, filename = None, g0=None, w0=None, opd=None,p=None, dp=None,df =None,**pd_kwargs):
		"""
		Cloud specification for the model. Clouds are parameterized by a single scattering albedo (w0), 
		an assymetry parameter (g0), and a total extinction per layer (opd).

		g0,w0, and opd are both wavelength and pressure dependent. Our cloud models come 
		from eddysed. Their output looks like this (where level 1=TOA, and wavenumber1=Smallest number)

		level wavenumber opd w0 g0
		1.	 1.   ... . .
		1.	 2.   ... . .
		1.	 3.   ... . .
		.	  .	... . .
		.	  .	... . .
		1.	 M.   ... . .
		2.	 1.   ... . .
		.	  .	... . .
		N.	 .	... . .

		If you are creating your own file you have to make sure that you have a 
		**pressure** (bars) and **wavenumber**(inverse cm) column. We will use this to make sure that your cloud 
		and atmospheric profiles are on the same grid. **If there is no pressure or wavelength parameter
		we will assume that you are on the same grid as your atmospheric input, and on the 
		eddysed wavelength grid! **

		Users can also input their own fixed cloud parameters, by specifying a single value 
		for g0,w0,opd and defining the thickness and location of the cloud. 

		Parameters
		----------
		filename : str 
			(Optional) Filename with info on the wavelength and pressure-dependent single scattering
			albedo, asymmetry factor, and total extinction per layer. Input associated pd_kwargs 
			so that the resultant output has columns named : `g0`, `w0` and `opd`. If you are not 
			using the eddysed output, you will also need a `wavenumber` and `pressure` column in units 
			of inverse cm, and bars. 
		g0 : float, list of float
			(Optional) Asymmetry factor. Can be a single float for a single cloud. Or a list of floats 
			for two different cloud layers 
		w0 : list of float 
			(Optional) Single Scattering Albedo. Can be a single float for a single cloud. Or a list of floats 
			for two different cloud layers 		
		opd : list of float 
			(Optional) Total Extinction in `dp`. Can be a single float for a single cloud. Or a list of floats 
			for two different cloud layers 
		p : list of float 
			(Optional) Center location of cloud deck (bars). Can be a single float for a single cloud. Or a list of floats 
			for two different cloud layers 
		dp : list of float 
			(Optional) Total thickness cloud deck (bars). Can be a single float for a single cloud or a list of floats 
			for two different cloud layers 
			Cloud will span 10**(np.log10(p) +- np.log10(dp)/2)
		df : pd.DataFrame, dict
			(Optional) Same as what would be included in the file, but in DataFrame or dict form
		"""

		#first complete options if user inputs dataframe or dict 
		if (not isinstance(filename, type(None)) & isinstance(df, type(None))) or (isinstance(filename, type(None)) & (not isinstance(df, type(None)))):

			if not isinstance(filename, type(None)):
				df = pd.read_csv(filename, **pd_kwargs)

			cols = df.keys()

			assert 'g0' in cols, "Please make sure g0 is a named column in cld file"
			assert 'w0' in cols, "Please make sure w0 is a named column in cld file"
			assert 'opd' in cols, "Please make sure opd is a named column in cld file"

			#CHECK SIZES

			#if it's a user specified pressure and wavenumber
			if (('pressure' in cols) & ('wavenumber' in cols)):
				df = df.sort_values(['wavenumber','pressure']).reset_index(drop=True)
				self.inputs['clouds']['wavenumber'] = df['wavenumber'].unique()
				nwave = len(self.inputs['clouds']['wavenumber'])
				nlayer = len(df['pressure'].unique())
				assert df.shape[0] == (self.nlevel-1)*nwave, "There are {0} rows in the df, which does not equal {1} layers previously specified x {2} wave pts".format(df.shape[0], self.nlevel-1, nwave) 
			
			#if its eddysed, make sure there are 196 wave points 
			else: 
				assert df.shape[0] == (self.nlevel-1)*196, "There are {0} rows in the df, which does not equal {1} layers x 196 eddysed wave pts".format(df.shape[0], self.nlevel-1) 
				
				self.inputs['clouds']['wavenumber'] = get_cld_input_grid('wave_EGP.dat')

			#add it to input
			self.inputs['clouds']['profile'] = df

		#first make sure that all of these have been specified
		elif None in [g0, w0, opd, p,dp]:
			raise Exception("Must either give dataframe/dict, OR a complete set of g0, w0, opd,p,dp to compute cloud profile")
		else:
			pressure_level = self.inputs['atmosphere']['profile']['pressure'].values
			pressure = np.sqrt(pressure_level[1:] * pressure_level[0:-1])#layer

			w = get_cld_input_grid('wave_EGP.dat')

			self.inputs['clouds']['wavenumber'] = w

			pressure_all =[]
			for i in pressure: pressure_all += [i]*len(w)
			wave_all = list(w)*len(pressure)

			df = pd.DataFrame({'pressure':pressure_all,
								'wavenumber': wave_all })


			zeros=np.zeros(196*(self.nlevel-1))

			#add in cloud layers 
			df['g0'] = zeros
			df['w0'] = zeros
			df['opd'] = zeros
			#loop through all cloud layers and set cloud profile
			for ig, iw, io , ip, idp in zip(g0,w0,opd,p,dp):
				minp = 10**(np.log10(ip) - np.log10(idp/2))
				maxp = 10**(np.log10(ip) + np.log10(idp/2))
				df.loc[((df['pressure'] >= minp) & (df['pressure'] <= maxp)),'g0']= ig
				df.loc[((df['pressure'] >= minp) & (df['pressure'] <= maxp)),'w0']= iw
				df.loc[((df['pressure'] >= minp) & (df['pressure'] <= maxp)),'opd']= io

			self.inputs['clouds']['profile'] = df

	def approx(self,single_phase='TTHG_ray',multi_phase='N=2',delta_eddington=True,raman='oklopcic',
				tthg_frac=[1,-1,2], tthg_back=-0.5, tthg_forward=1):
		"""
		This function sets all the default approximations in the code. It transforms the string specificatons
		into a number so that they can be used in numba nopython routines. 

		For `str` cases such as `TTHG_ray` users see all the options by using the function `single_phase_options`
		or `multi_phase_options`, etc. 

		single_phase : str 
			Single scattering phase function approximation 
		multi_phase : str 
			Multiple scattering phase function approximation 
		delta_eddington : bool 
			Turns delta-eddington on and off
		raman : str 
			Uses various versions of raman scattering 
		tthg_frac : list 
			Functional of forward to back scattering with the form of polynomial :
			tthg_frac[0] + tthg_frac[1]*g_b^tthg_frac[2]
			See eqn. 6 in picaso paper 
		tthg_back : float 
			Back scattering asymmetry factor gf = g_bar*tthg_back
		tthg_forward : float 
			Forward scattering asymmetry factor gb = g_bar * tthg_forward 
		"""

		self.inputs['approx']['single_phase'] = single_phase_options(printout=False).index(single_phase)
		self.inputs['approx']['multi_phase'] = multi_phase_options(printout=False).index(multi_phase)
		self.inputs['approx']['delta_eddington'] = delta_eddington
		self.inputs['approx']['raman'] =  raman_options().index(raman)

		if isinstance(tthg_frac, (list, np.ndarray)):
			if len(tthg_frac) == 3:
				self.inputs['approx']['TTHG_params']['fraction'] = tthg_frac
			else:
				raise Exception('tthg_frac should be of length=3 so that : tthg_frac[0] + tthg_frac[1]*g_b^tthg_frac[2]')
		else: 
			raise Exception('tthg_frac should be a list or ndarray of length=3')

		self.inputs['approx']['TTHG_params']['constant_back'] = tthg_back
		self.inputs['approx']['TTHG_params']['constant_forward']=tthg_forward


	def spectrum(self,opacityclass,dimension = '1d', calculation='reflected', full_output=False, plot_opacity= False):
		"""Run Spectrum"""
		if ('thermal' in calculation) and (np.isnan(self.inputs['star']['radius']) or np.isnan(self.inputs['planet']['radius'])):
			raise Exception("Stellar or Planet radius not supplied but thermal flux was requested. See options in `star()` `gravity()`")
			
		return picaso(self, opacityclass,dimension=dimension,calculation=calculation,
			full_output=full_output, plot_opacity=plot_opacity)


def jupiter_pt():
	"""Function to get Jupiter's PT profile"""
	return os.path.join(__refdata__, 'base_cases','jupiter.pt')
def jupiter_cld():
	"""Function to get rough Jupiter Cloud model with fsed=3"""
	return os.path.join(__refdata__, 'base_cases','jupiterf3.cld')
def single_phase_options(printout=True):
	"""Retrieve all the options for direct radation"""
	if printout: print("Can also set functional form of forward/back scattering in approx['TTHG_params']")
	return ['cahoy','OTHG','TTHG','TTHG_ray']
def multi_phase_options(printout=True):
	"""Retrieve all the options for multiple scattering radiation"""
	if printout: print("Can also set delta_eddington=True/False in approx['delta_eddington']")
	return ['N=2','N=1']
def raman_options():
	"""Retrieve options for raman scattering approximtions"""
	return ["oklopcic","pollack","none"]

