import pandas as pd
import numpy as np
import h5py 
import os
from numba import jit
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import inferno
debug = False 

#@jit(nopython=True)
def optc(atmosphere, opacityclass, delta_eddington=True,test_mode=False):
	"""
	Returns total optical depth per slab layer including molecular opacity, continuum opacity. 
	It should automatically select the molecules needed
	
	Parameters
	----------
	atmosphere : class ATMSETUP
		This inherets the class from atmsetup.py 
	opacityclass : class opacity
		This inherets the class from optics.py. It is done this way so that the opacity db doesnt have 
		to be reloaded in a retrieval 
	delta_eddington : bool 
		(Optional) Default=True, With Delta-eddington on, it incorporates the forward peak 
		contribution by adjusting optical properties such that the fraction of scattered energy
		in the forward direction is removed from the scattering parameters 

	Returns
	-------
	DTAU : ndarray 
		This is a matrix with # layer by # wavelength. It is the opacity contained within a layer 
		including the continuum, scattering, cloud (if specified), and molecular opacity
	TAU : ndarray
		This is a matrix with # level by # wavelength. It is the cumsum of opacity contained 
		including the continuum, scattering, cloud (if specified), and molecular opacity

	WBAR : ndarray
		This is the single scattering albedo that includes rayleigh, raman and user input scattering sources. 
		It has dimensions: # layer by # wavelength

	COSB : ndarray
		This is the asymettry factor which accounts for rayleigh and user specified values 
		It has dimensions: # layer by # wavelength

	Notes
	-----
	This was baselined against jupiter with the old fortran code. It matches 100% for all cases 
	except for hotter cases where Na & K are present. This differences is not a product of the code 
	but a product of the different opacities (1060 grid versus old 736 grid)

	To Do 
	-----
		- Replace detla-function adjustment with better approximation (e.g. Cuzzi)
	"""

	atm = atmosphere
	tlevel = atm.level['temperature']
	plevel = atm.level['pressure']/atm.c.pconv #think of a better solution for this later when mark responds
	
	tlayer = atm.layer['temperature']
	player = atm.layer['pressure']/atm.c.pconv #think of a better solution for this later when mark responds

	gravity = atm.planet.gravity / 100.0 #this too... need to have consistent units.

	if debug: 
		plot_layer=0#np.size(tlayer)-1
		opt_figure = figure(x_axis_label = 'Wavelength', y_axis_label='TAUGAS in optics.py', 
		title = 'Opacity at T='+str(tlayer[plot_layer])+' Layer='+str(plot_layer)
		,y_axis_type='log',height=800, width=1200)

	#====================== INITIALIZE TAUGAS#======================
	TAUGAS = 0 
	c=1
	#set color scheme.. adding 2 for rayleigh, and total
	if debug: colors = inferno(2+len(atm.continuum_molecules) + len(atm.molecules))

	#====================== ADD CONTIMUUM OPACITY======================	
	#Set up coefficients needed to convert amagat to a normal human unit
	#these COEF's are only used for the continuum opacity. 
	ACOEF = (tlayer/(tlevel[:-1]*tlevel[1:]))*(
	 		tlevel[1:]*plevel[1:] - tlevel[:-1]*plevel[:-1])/(plevel[1:]-plevel[:-1]) #UNITLESS

	BCOEF = (tlayer/(tlevel[:-1]*tlevel[1:]))*(
			tlevel[:-1] - tlevel[1:])/(plevel[1:]-plevel[:-1]) #INVERSE PRESSURE

	COEF1 = atm.c.rgas*273.15**2*.5E5* (
		ACOEF* (plevel[1:]**2 - plevel[:-1]**2) + BCOEF*(
			2./3.)*(plevel[1:]**3 - plevel[:-1]**3) ) / (
		1.01325**2 *gravity*tlayer*atm.layer['mmw'])

	#go through every molecule in the continuum first 
	for m in atm.continuum_molecules:
		#H- Bound-Free
		if (m[0] == "H-") and (m[1] == "bf"):
			h_ =np.where(m[0]==np.array(atm.weights.keys()))[0][0]
			ADDTAU = (opacityclass.get_continuum_opac(tlayer, 'H-bf').T*( 		#[(nlayer x nwno).T *(
							atm.layer['mixingratios'][:,h_]*      				#nlayer
			               	atm.layer['colden']/ 								#nlayer
			               	(atm.layer['mmw']*atm.c.amu)) 	).T					#nlayer)].T
			#testing['H-bf'] = ADDTAU
			TAUGAS += ADDTAU
			if debug: opt_figure.line(1e4/opacityclass.wno, ADDTAU[plot_layer,:], alpha=0.7,legend=m[0]+m[1], line_width=3, color=colors[c],
			muted_color=colors[c], muted_alpha=0.2)
		#H- Free-Free
		elif (m[0] == "H-") and (m[1] == "ff"):
			h_ = np.where('H'==np.array(atm.weights.keys()))[0][0]
			ADDTAU = (opacityclass.get_continuum_opac(tlayer, 'H-ff').T*( 		        #[(nlayer x nwno).T *(
							atm.layer['pressure']* 								  		#nlayer
							atm.layer['mixingratios'][:,h_]*atm.layer['electrons']*     #nlayer
			               	atm.layer['colden']/ 										#nlayer
			               	(tlayer*atm.layer['mmw']*atm.c.amu*atm.c.k_b)) 	).T			#nlayer)].T
			#testing['H-ff'] = ADDTAU
			TAUGAS += ADDTAU
			if debug: opt_figure.line(1e4/opacityclass.wno, ADDTAU[plot_layer,:], alpha=0.7,legend=m[0]+m[1], line_width=3, color=colors[c],
			muted_color=colors[c], muted_alpha=0.2)
		#H2- 
		elif (m[0] == "H2") and (m[1] == "H2-"): 
			#make sure you know which index of matrix is h2 and e-
			h2_ = np.where(m[0]==np.array(atm.weights.keys()))[0][0]
			#calculate opacity
			#this is a hefty matrix multiplication to make sure that we are 
			#multiplying each column of the opacities by the same 1D vector (as opposed to traditional 
			#matrix multiplication). This is the reason for the transposes.
			ADDTAU = (opacityclass.get_continuum_opac(tlayer, 'H2-').T*( 		        #[(nlayer x nwno).T *(
							atm.layer['pressure']* 								  		#nlayer
							atm.layer['mixingratios'][:,h2_]*atm.layer['electrons']*    #nlayer
			               	atm.layer['colden']/ 										#nlayer
			               	(atm.layer['mmw']*atm.c.amu)) 	).T							#nlayer)].T
			#testing['H2-'] = ADDTAU

			TAUGAS += ADDTAU
			if debug: opt_figure.line(1e4/opacityclass.wno, ADDTAU[plot_layer,:], alpha=0.7,legend=m[0]+m[1], line_width=3, color=colors[c],
			muted_color=colors[c], muted_alpha=0.2)
		#everything else.. e.g. H2-H2, H2-CH4. Automatically determined by which molecules were requested
		else:
			m0 = np.where(m[0]==np.array(atm.weights.keys()))[0][0]
			m1 = np.where(m[1]==np.array(atm.weights.keys()))[0][0]
			#calculate opacity

			ADDTAU = (opacityclass.get_continuum_opac(tlayer, m[0]+m[1]).T * ( #[(nlayer x nwno).T *(
								COEF1*											#nlayer
								atm.layer['mixingratios'][:,m0]*				#nlayer
								atm.layer['mixingratios'][:,m1] )  ).T 			#nlayer)].T
			#testing[m[0]+m[1]] = ADDTAU
			TAUGAS += ADDTAU
			if debug: opt_figure.line(1e4/opacityclass.wno, ADDTAU[plot_layer,:], alpha=0.7,legend=m[0]+m[1], line_width=3, color=colors[c],
			muted_color=colors[c], muted_alpha=0.2)
		c+=1
	
	#====================== ADD MOLECULAR OPACITY======================	
	for m in atm.molecules:
		ind = np.where(m==np.array(atm.weights.keys()))[0][0]
		ADDTAU = (opacityclass.get_molecular_opac(tlayer,player, m).T * ( 
					atm.layer['colden']*
					atm.layer['mixingratios'][:,ind]*atm.weights[m].values[0]/ 
			        atm.layer['mmw']) ).T
		TAUGAS += ADDTAU
		#testing[m] = ADDTAU
		if debug: opt_figure.line(1e4/opacityclass.wno, ADDTAU[int(np.size(tlayer)/2),:], alpha=0.7,legend=m, line_width=3, color=colors[c],
			muted_color=colors[c], muted_alpha=0.2)
		c+=1

	#====================== ADD RAYLEIGH OPACITY======================	
	ich4 = np.where('CH4'==np.array(atm.weights.keys()))[0][0]
	ih2 = np.where('H2'==np.array(atm.weights.keys()))[0][0]
	ihe = np.where('He'==np.array(atm.weights.keys()))[0][0]
	TAURAY = rayleigh(atm.layer['colden'],atm.layer['mixingratios'][:,ih2], 
					atm.layer['mixingratios'][:,ihe], atm.layer['mixingratios'][:,ich4], 
					opacityclass.wave, atm.layer['mmw'],atm.c.amu )
	#testing['ray'] = TAURAY
	if debug: opt_figure.line(1e4/opacityclass.wno, TAURAY[int(np.size(tlayer)/2),:], alpha=0.7,legend='Rayleigh', line_width=3, color=colors[c],
			muted_color=colors[c], muted_alpha=0.2)	


	#====================== ADD RAMAN OPACITY======================
	raman_factor = 0.9999#raman(opacityclass.wno) #CURRENTLY ONLY RETURNS ONES 


	#====================== ADD CLOUD OPACITY======================	
	TAUCLD = atm.layer['cloud']['opd'] #TAUCLD is the total extinction from cloud = (abs + scattering)
	#testing['cld'] = TAUCLD
	asym_factor_cld = atm.layer['cloud']['g0'] 
	single_scattering_cld = atm.layer['cloud']['w0'] #scatter / (abs + scattering) from cloud only 


	#====================== ADD EVERYTHING TOGETHER PER LAYER======================	
	#formerly DTAUV
	DTAU = TAUGAS + TAURAY + TAUCLD 
	#formerly COSBV
	COSB = (TAUCLD*asym_factor_cld)/(TAUCLD + TAURAY)
	#formerly GCOSB2 
	GCOS2 = 0.5*TAURAY/(TAURAY + TAUCLD) #Hansen & Travis 1974 for Rayleigh scattering 
	#formerly WBARV.. change name later
	W0 = (TAURAY*raman_factor + TAUCLD*single_scattering_cld) / (TAUGAS + TAURAY + TAUCLD) #TOTAL single scattering 

	#sum up taus starting at the top, going to depth
	shape = DTAU.shape
	TAU = np.zeros((shape[0]+1, shape[1]))
	TAU[1:,:]=numba_cumsum(DTAU)

	if debug:
		opt_figure.line(1e4/opacityclass.wno, DTAU[int(np.size(tlayer)/2),:], legend='TOTAL', line_width=4, color=colors[0],
			muted_color=colors[c], muted_alpha=0.2)
		opt_figure.legend.click_policy="mute"
		output_file('OpacityDebug.html')
		show(opt_figure)

	if test_mode != None:  
		#this is to check against Dlugach & Yanovitskij 
		#https://www.sciencedirect.com/science/article/pii/0019103574901675?via%3Dihub
		if test_mode=='rayleigh':
			DTAU = TAURAY 
			GCOS2 = 0.5
		else: 
			DTAU = TAURAY*0+0.5
			GCOS2 = 0.0
		COSB = atm.layer['scattering']['g0']
		W0 = atm.layer['scattering']['w0']
		TAU = np.zeros((shape[0]+1, shape[1]))
		TAU[1:,:]=numba_cumsum(DTAU)

	#====================== D-Eddington Approximation or TEST mode======================
	if delta_eddington:
		#First thing to do is to use the delta function to icorporate the forward 
		#peak contribution of scattering by adjusting optical properties such that 
		#the fraction of scattered energy in the forward direction is removed from 
		#the scattering parameters 

		#Joseph, J.H., W. J. Wiscombe, and J. A. Weinman, 
		#The Delta-Eddington approximation for radiative flux transfer, J. Atmos. Sci. 33, 2452-2459, 1976.

		#also see these lecture notes are pretty good
		#http://irina.eas.gatech.edu/EAS8803_SPRING2012/Lec20.pdf
		w0_dedd=W0*(1.-COSB**2)/(1.0-W0*COSB**2)
		cosb_dedd=COSB/(1.+COSB)
		dtau_dedd=DTAU*(1.-W0*COSB**2) 

		#sum up taus starting at the top, going to depth
		tau_dedd = np.zeros((shape[0]+1, shape[1]))
		tau_dedd[1:,:]=numba_cumsum(dtau_dedd)
    
    	#returning the terms used in 
		return dtau_dedd, tau_dedd,  w0_dedd, cosb_dedd ,GCOS2

	else: 
		return DTAU, TAU, W0, COSB,GCOS2

#'f8[:,:](f8[:],f8[:],f8[:],f8[:],f8[:],f8[:],f8)',
@jit(nopython=True, cache=True)
def rayleigh(colden,H2,He,CH4,wave,xmu,amu):
	"""
	Rayleigh function taken from old albedo code. Keeping this modular, as we may want 
	to swap out different methods to calculate rayleigh opacity 

	Parameters
	----------
	colden : array of float 
		Column Density in CGS units 
	H2 : array of float 
		Mixing ratio as function of altitude for H2 
	He : array of flaot 
		Mixing ratio as a function of altitude for He 
	CH4 : array of float 
		Mixing ratio as a function of altitude fro CH4 
	wave : array of float 
		wavelength (microns) of grid. This should be in DECENDING order to comply with 
		everything else being in wave numberes 
	xmu : array of float
		mean molecular weight of atmosphere in amu 
	amu : float 
		amu constant in grams 
	"""
	#define all rayleigh constants
	TAURAY = np.zeros((colden.size, wave.size))
	dpol= np.zeros(3)
	dpol[0], dpol[1], dpol[2]=1.022 , 1.0, 1.0
	gnu = np.zeros((2,3))
	gnu[0,0]=1.355e-4
	gnu[1,0]=1.235e-6
	gnu[0,1]=3.469e-5
	gnu[1,1]=8.139e-8
	gnu[0,2]=4.318e-4
	gnu[1,2]=3.408e-6
	XN0 = 2.687E19
	cfray = 32.0*np.pi**3*1.e21/(3.0*2.687e19)
	cold = colden / (xmu * amu) #nlayers
	gasmixing = np.zeros((cold.size, 3))
	gasmixing[:,0] = H2
	gasmixing[:,1] = He
	gasmixing[:,2] = CH4
	#add rayleigh from each contributing gas using corresponding mixing 
	for i in np.arange(0,3,1):
		tec = cfray*(dpol[i]/wave**4)*(gnu[0,i]+gnu[1,i]/   #nwave
                     wave**2)**2 
		TAUR = (cold*gasmixing[:,i]).reshape((-1, 1)) * tec * 1e-5 / XN0
		
		TAURAY += TAUR

	return TAURAY

@jit(nopython=True, cache=True)
def raman(wavelength):
	"""
	The Ramam scattering will alter the rayleigh scattering. The returned value is 
	modified single scattering albedo. 

	This method is described in Pollack+1986. Albeit not the best method. Sromovsky+2005 
	pointed out the inconsistencies in this method. You can see from his comparisons 
	that the Pollack approximations don't accurately capture the depths of the line centers. 
	Since then, OKLOPCIC+2016 did a much
	better model of raman scattring (with ghost lines). Might be worth it to consider a more 
	sophisticated version of Raman scattering. 

	Will be added to the rayleigh scattering as : TAURAY*RAMAN
	"""
	#constants 
	h = 6.6252e-27
	c = 2.9978e10
	bohrd = 5.2917e-9
	hmass = 1.6734e-24
	rmu = .5 * hmass

	#set wavelength shift of the ramam scatterer
	shift_v0 = 4161.0 

	facip = h * c / ( 1.e-4 * 27.2 * 1.602e-12 ) 
	facray = 1.e16 * bohrd ** 3 * 128. * np.pi ** 5 * bohrd ** 3 / 9. 
	facv = 2.08 / 2.38 * facray / bohrd ** 2 / ( 8. * np.pi * np.pi * rmu * c * shift_v0 ) * h

	#cross section of the unshifted rayleigh and the vibrationally shifted rayleigh
	gli = np.zeros(5)
	wli = gli 
	gri = gli 
	wri = gli 
	gli[:] = [1.296, .247, .297,  .157,  .003]
	wli[:] = [.507, .628, .733, 1.175, 2.526]
	gri[:] = [.913, .239, .440,  .344,  .064]
	wri[:] = [.537, .639, .789, 1.304, 3.263]

	alp = np.zeros(7)
	arp = alp
	alp[:] = [6.84, 6.96, 7.33, 8.02, 9.18, 11.1, 14.5 ]
	arp[:] = [3.66, 3.71, 3.88, 4.19, 4.70, 5.52, 6.88 ]

	omega = facip / wavelength

	#first compute extinction cross section for unshifted component 
	#e.g. rayleigh
	alpha_l=0
	alpha_r=0

	for i in range(5):
		alpha_l += gli[i] / ( wli[i] ** 2 - omega ** 2 ) 
		alpha_r += gri[i] / ( wri[i] ** 2 - omega ** 2 )

	alpha2 = (( 2. * alpha_r + alpha_l ) / 3. ) ** 2
	gamma2 = ( alpha_l - alpha_r ) ** 2
	qray = facray * ( 3. * alpha2 + 2./3. * gamma2 ) / wavelength ** 4

	#next, compute the extinction cross section for vibrationally 
	#shifted component 
	ip = np.min([int(omega/0.05), 5.0]) + 1 
	f = omega / 0.5 - (ip-1)
	alpha_pl = ( 1. - f ) * alp[ip] + f * alp[ip+1]
	alpha_pr = ( 1. - f ) * arp[ip] + f * arp[ip+1]
	alpha_p2 = (( 2. * alpha_pr + alpha_pl ) / 3. ) ** 2
	gamma_p2 = ( alpha_pl - alpha_pr ) ** 2
	qv = facv / SHIFT( WAVEL, -SHIFTV0 ) ** 4 * ( 3. * alpha_p2 + 2./3. * gamma_p2 )	



	return 0.9999999 

class RetrieveOpacities():
	"""
	This will be the class that will retrieve the opacities from the h5 database. Right 
	now we are just employing nearest neighbors to grab the respective opacities. 
	Eventually, we will want to do something like correlated K. 

	Attributes
	----------
	get_continuum_atlayer 
		given a temperature and molecule, retrieve the wavelength depedent opacity 
	get_molopac_atlayer 
		given a pressure and temperature, retrieve the wavelength dependent opacity

	"""
	def __init__(self, wno, wave, continuum_data, molecular_data):
		self.cia_db = h5py.File(continuum_data)
		self.cia_temps = np.array([float(i) for i in self.cia_db['H2H2'].keys()])
		#self.cia_mols = [i for i in self.cia_db.keys()]

		self.mol_db = h5py.File(molecular_data)

		self.molecules = np.array([i for i in self.mol_db.keys()])

		self.mol_temps = np.array([float(i) for i in self.mol_db['H2O'].keys()])
		self.mol_press = {}
		for i in self.mol_temps:
			self.mol_press[str(i)] = np.array([float(i) for i in self.mol_db['H2O'][str(i)].keys()])
		#self.cia_mols = [i for i in self.cia_db.keys()]

		self.wno = wno
		self.wave = wave
		self.nwno = np.size(wno)


	def get_continuum_opac(self, temperature, molecule): 
		"""
		Based on a temperature, this retrieves the continuum opacity for 

		Parameters
		----------
		temperature : float array
			An array of temperatures to retrieve the continuum opacity at

		molecule : str 
			Which opacity source to query. currently available: 'h2h2','h2he','h2h','h2ch4','h2n2'

		Return
		------
		matrix 
			number of layers by number of wave points 
		"""
		nearestT = [self.cia_temps[find_nearest(self.cia_temps, t)].astype(str) for t in temperature]
		sizeT = np.size(nearestT)
		a = np.zeros((sizeT, self.nwno))
		for i,t in zip(range(sizeT) ,nearestT): 
			a[i,:] = np.array(self.cia_db[molecule][t])
		return a

	def get_molecular_opac(self, temperature, pressure, molecule):
		"""
		Based on temperature, and pressure, retrieves the molecular opacity for 
		certain molecule 

		Parameters
		----------
		temperature : array of float 
			An array of temperatures to retrieve the continuum opacity at 
		pressure : array of float 
			Pressures (IN BARS) to retrieve the continuum opacity at 
		molecule : str 
			Which opacity source to query. Will get error if not in the db
		"""
		if (molecule == 'Na') or (molecule == 'K'):
			mol_temps = np.array([float(i) for i in self.mol_db['Na'].keys()])
			mol_press = {}
			for i in mol_temps:
				mol_press[str(i)] = np.array([float(i) for i in self.mol_db['Na'][str(i)].keys()])
			nearestT = [mol_temps[find_nearest(mol_temps, t)].astype(str) for t in temperature]
			nearestP = [mol_press[str(t)][find_nearest(mol_press[str(t)], p)].astype(str) for p,t in zip(pressure,nearestT)]
		else:
			nearestT = [self.mol_temps[find_nearest(self.mol_temps, t)].astype(str) for t in temperature]
			nearestP = [self.mol_press[str(t)][find_nearest(self.mol_press[str(t)], p)].astype(str) for p,t in zip(pressure,nearestT)]

		sizeT = np.size(nearestT)
		a = np.zeros((sizeT, self.nwno))
		for i,t,p in zip(range(sizeT) ,nearestT,nearestP): 
			a[i,:] = np.array(self.mol_db[molecule][t][p])
		return a

@jit(nopython=True, cache=True)
def find_nearest(array,value):
	#small program to find the nearest neighbor in temperature  
	idx = (np.abs(array-value)).argmin()
	return idx
@jit(nopython=True, cache=True)
def numba_cumsum(mat):
	"""Function to compute cumsum along axis=0 to bypass numba not allowing kwargs in 
	cumsum 
	"""
	new_mat = np.zeros(mat.shape)
	for i in range(mat.shape[1]):
		new_mat[:,i] = np.cumsum(mat[:,i])
	return new_mat
