from .atmsetup import ATMSETUP
from .fluxes import get_flux_geom_1d, get_flux_geom_3d 
from .wavelength import get_output_grid,get_cld_input_grid
import numpy as np
import pandas as pd
from .optics import RetrieveOpacities,compute_opacity
import os
import pickle as pk
from .disco import get_angles, compute_disco, compress_disco
import copy
import json
import pysynphot as psyn
__refdata__ = os.environ.get('picaso_refdata')

def picaso(input,dimension = '1d', full_output=False, plot_opacity= False):
	"""
	Currently top level program to run albedo code 

	Parameters 
	----------
	input : dict 
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

	#check to see if we are running in test mode
	test_mode = input['test_mode']

	#set approx numbers options (to be used in numba compiled functions)
	single_phase, multi_phase, raman_approx, = set_approximations(input) 

	#phase angle 
	phase_angle = input['phase_angle']

	#parameters needed for the two term hg phase function. 
	#Defaults are set in config.json
	f = input['approx']['TTHG_params']['fraction']
	frac_a = f[0]
	frac_b = f[1]
	frac_c = f[2]
	constant_back = input['approx']['TTHG_params']['constant_back']
	constant_forward = input['approx']['TTHG_params']['constant_forward']

	#get wavelength grid and order it
	grid = get_output_grid(input['opacities']['files']['wavegrid'])
	wno = np.sort(grid['wavenumber'].values)
	nwno = len(wno)

	#check to see if new wavelength grid is necessary .. assuming its always the same here
	#this is where continuum_factory would run, if it was a brand new grid
	#TAG:TODO

	#get opacity class
	opacityclass=RetrieveOpacities(wno, 1e4/wno,
					os.path.join(__refdata__, 'opacities', input['opacities']['files']['continuum']),
					os.path.join(__refdata__, 'opacities', input['opacities']['files']['molecular']), 
					os.path.join(__refdata__, 'opacities', input['opacities']['files']['raman']) 
					)

	#get geometry
	ng = input['disco']['num_gangle']
	nt = input['disco']['num_tangle']
	gangle,gweight,tangle,tweight = get_angles(ng, nt) 
	#planet disk is divieded into gaussian and chebyshev angles and weights for perfoming the 
	#intensity as a function of planetary pahse angle 
	ubar0, ubar1, cos_theta,lat,lon = compute_disco(ng, nt, gangle, tangle, phase_angle)

	#set star 
	F0PI = np.zeros(nwno) + 1.0 

	#define approximinations 
	delta_eddington = input['approx']['delta_eddington']

	#begin atm setup
	atm = ATMSETUP(input)

	############### To stuff needed for the star ###############
	#get star 
	star = input['star']
	wno_star, flux_star, hires_wno_star, hires_flux_star = atm.get_stellar_spec(opacityclass.wno, star['wno'],star['flux']) 
	#this is added to the opacity class
	stellar_raman_shifts = opacityclass.compute_stellar_shits(hires_wno_star, hires_flux_star)


	################ From here on out is everything that would go through retrieval or 3d input##############
	atm.planet.gravity = input['planet']['gravity']

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


	if dimension == '1d':
		#only need to get opacities for one pt profile

		#There are two sets of dtau,tau,w0,g in the event that the user chooses to use delta-eddington
		#We use HG function for single scattering which gets the forward scattering/back scattering peaks 
		#well. We only really want to use delta-edd for multi scattering legendre polynomials. 
		DTAU, TAU, W0, COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG= compute_opacity(
			atm, opacityclass,delta_eddington=delta_eddington,test_mode=test_mode,raman=raman_approx,
			full_output=full_output, plot_opacity=plot_opacity)

		#use toon method (and tridiagonal matrix solver) to get net cumulative fluxes 
		xint_at_top  = get_flux_geom_1d(atm.c.nlevel, wno,nwno,ng,nt,
													DTAU, TAU, W0, COSB,GCOS2,ftau_cld,ftau_ray,
													DTAU_OG, TAU_OG, W0_OG, COSB_OG ,
													atm.surf_reflect, ubar0,ubar1,cos_theta, F0PI,
													single_phase,multi_phase,
													frac_a,frac_b,frac_c,constant_back,constant_forward)
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
		xint_at_top  = get_flux_geom_3d(atm.c.nlevel, wno,nwno,ng,nt,
											DTAU_3d, TAU_3d, W0_3d, COSB_3d,GCOS2_3d, FTAU_CLD_3d,FTAU_RAY_3d,
											DTAU_OG_3d, TAU_OG_3d, W0_OG_3d, COSB_OG_3d,
											atm.surf_reflect, ubar0,ubar1,cos_theta, F0PI,
											single_phase,multi_phase,
											frac_a,frac_b,frac_c,constant_back,constant_forward)

	#now compress everything based on the weights 
	albedo = compress_disco(ng, nt, nwno, cos_theta, xint_at_top, gweight, tweight,F0PI)
	
	if full_output:
		return wno, albedo , atm
	else: 
		return wno, albedo

def set_approximations(input):
	"""
	This is a function devoded to defining all the approximations within the code. 
	This was built to make the code slightly more transparent in terms of options it 
	is making. A lof of the functions are numba, which means they cannot compare strings. 
	Therefore, this function returns integers for each option based on placement in a string. 
	The functions in the rest of the code follow from this.  

	Parameters
	----------
	input : dict 
		Input dictionary from main config.json structure 

	Returns
	-------
	Integer option for single scattering phase function, multiple scattering phase function, 
	and raman scattering 
	"""

	#define all possible options
	single_phase = single_phase_options(printout=False).index(input['approx']['single_phase'])

	multi_phase = multi_phase_options(printout=False).index(input['approx']['multi_phase'])

	raman_approx = raman_options().index(input['approx']['raman'])

	return single_phase, multi_phase, raman_approx

class inputs():
	"""Class to setup planet to run"""
	def __init__(self):

		self.inputs = json.load(open(os.path.join(__refdata__,'config.json')))
	def phase_angle(self, phase=0)	:
		"""Define phase angle"""
		self.inputs['phase_angle'] = phase
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
			(Optional) radius of planet 
		radius_unit : astropy.unit
			(Optional) Unit of radius
		mass : float 
			(Optional) mass of planet 
		mass_unit : astropy.unit
			(Optional) Unit of mass	
		"""
		if gravity is not None:
			g = (gravity*gravity_unit).to('cm/(s**2)')
			g = g.value
			self.inputs['planet']['gravity'] = g
			self.inputs['planet']['gravity_unit'] = 'cm/(s**2)'
		elif (mass is not None) and (radius is not None):
			m = (mass*mass_unit).to(u.g)
			r = (radius*radius_unit).to(u.cm)
			g = (self.c.G * m /  (r**2)).value
			self.inputs['planet']['radius'] = r
			self.inputs['planet']['radius_unit'] = 'cm'
			self.inputs['planet']['mass'] = m
			self.inputs['planet']['mass_unit'] = 'g'
			self.inputs['planet']['gravity'] = g
			self.inputs['planet']['gravity_unit'] = 'cm/(s**2)'
		else: 
			raise Exception('Need to specify gravity or radius and mass + additional units')

	def star(self, temp, metal, logg ,database='ck04models'):
		"""
		Get the stellar spectrum using pysynphot and interpolate onto a much finer grid than the 
		planet grid. 

		Parameters
		----------
		temp : float 
			Teff of the stellar model 
		metal : float 
			Metallicity of the stellar model 
		logg : float 
			Logg cgs of the stellar model
		database : str 
			(Optional)The database to pull stellar spectrum from. See documentation for pysynphot. 
		"""
		sp = psyn.Icat(database, temp, metal, logg)
		sp.convert("um")
		sp.convert('flam') 
		wno_star = 1e4/sp.wave[::-1] #convert to wave number and flip
		flux_star = sp.flux[::-1]	 #flip here to get correct order 

		self.inputs['star']['database'] = database
		self.inputs['star']['temp'] = temp
		self.inputs['star']['logg'] = logg
		self.inputs['star']['metal'] = metal
		self.inputs['star']['wno'] = wno_star 
		self.inputs['star']['flux'] = flux_star 

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
		if not isinstance(filename, type(None)):
			df = pd.read_csv(filename, **pd_kwargs)
			self.nlevel=df.shape[0] 

		if 'pressure' not in df.keys(): 
			raise Exception("Check column names. `pressure` must be included.")

		if (('temperature' not in df.keys()) and (isinstance(pt_params, type(None)))):
			raise Exception("`temperature` not specified and pt_params not given. Do one or the other.")

		else: 
			self.inputs['atmosphere']['pt_params'] = pt_params
			self.nlevel=61 #default n of levels for parameterization

		if not isinstance(exclude_mol, type(None)):
			df = df.drop(exclude_mol, axis=1)
			self.inputs['atmosphere']['exclude_mol'] = exclude_mol

		self.inputs['atmosphere']['profile'] = df.sort_values('pressure').reset_index(drop=True)


	def clouds(self, filename = None, g0=None, w0=None, opd=None,p=None, dp=None,**pd_kwargs):
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
		"""

		#first complete options if user inputs dataframe or dict 
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

	def spectrum(self,dimension = '1d', full_output=False, plot_opacity= False):
		"""Run Spectrum"""
		w, a = picaso(self.inputs, full_output=full_output, plot_opacity=plot_opacity)
		return w, a


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

if __name__ == "__main__":
	
	import pandas as pd
	from bokeh.plotting import figure, show, output_file
	from bokeh.palettes import inferno
	colors = inferno(19)

	__refdata__ = os.environ.get('picaso_refdata')
	
	a = load_inputs()

	#paths to pt and cld files
	a['atmosphere']['profile']['filepath'] = jupiter_pt()
	#a['atmosphere']['clouds']['filepath'] =  jupiter_cld()
	#define gravity
	a['planet']['gravity'] = 25
	a['planet']['gravity_unit'] = 'm/(s**2)' 
	#this is default. But set eddington approx
	a['approx']['delta_eddington']=True
	#define number of integration angles
	a['disco']['num_gangle']= 10
	a['disco']['num_tangle'] = 10
	#add star properties for raman scattering
	a['star']['temp'] = 6000
	a['star']['metal'] = 0.0122
	a['star']['logg'] = 4.437

	fig = figure(width=1200, height=800)
	ii = 0 
	for phase in [0.0]:
		print(phase)
		wno, alb = picaso(a, phase,dimension='1d')
		fig.line(1e4/wno, alb, line_width = 4, color = colors[ii], legend = 'Phase='+str(phase))
		ii+=1 
		show(fig)
