import pandas as pd
import numpy as np
import h5py 
import os
from numba import jit
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import inferno
debug = True 

#@jit(nopython=True)
def optc(atmosphere, opacityclass):
	"""
	Returns total optical depth per slab layer including molecular opacity, continuum opacity 
	"""

	atm = atmosphere
	tlevel = atm.level['temperature']
	plevel = atm.level['pressure']/atm.c.pconv #think of a better solution for this later when mark responds

	tlayer = atm.layer['temperature']
	player = atm.layer['pressure']/atm.c.pconv #think of a better solution for this later when mark responds

	gravity = atm.planet.gravity * 100.0 #this too... need to have consistent units.

	if debug: opt_figure = figure(x_axis_label = 'Wavenumber', y_axis_label='TAUGAS in optics.py', 
		title = 'Opacity at nlayers/2, T='+str(tlayer[int(np.size(tlayer)/2)])
		,y_axis_type='log',height=800, width=1200)

	#====================== INITIALIZE TAUGAS#======================
	TAUGAS = 0 
	c=1
	#set color scheme.. adding 2 for rayleigh, and total
	if debug: colors = inferno(2+len(atm.continuum_molecules) + len(atm.molecules))

	#====================== ADD CONTIMUUM OPACITY======================	
	#coefficients from McKay TO DO: Check with Mark where this comes from
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
			ADDTAU = (opacityclass.get_continuum_opac(tlayer, 'H-bf').T*( 		          #[(nlayer x nwno).T *(
							atm.layer['mixingratios'][:,h_]*      							#nlayer
			               	atm.layer['colden']/ 											  #nlayer
			               	(atm.layer['mmw']*atm.c.amu)) 	).T								  #nlayer)].T
			TAUGAS += ADDTAU
			if debug: opt_figure.line(opacityclass.wno, ADDTAU[int(np.size(tlayer)/2),:], alpha=0.7,legend=m[0]+m[1], line_width=3, color=colors[c],
			muted_color=colors[c], muted_alpha=0.2)
		#H- Free-Free
		elif (m[0] == "H-") and (m[1] == "ff"):
			h_ = np.where('H'==np.array(atm.weights.keys()))[0][0]
			ADDTAU = (opacityclass.get_continuum_opac(tlayer, 'H-ff').T*( 		              #[(nlayer x nwno).T *(
							atm.layer['pressure']* 								  			  #nlayer
							atm.layer['mixingratios'][:,h_]*atm.layer['electrons']*           #nlayer
			               	atm.layer['colden']/ 											  #nlayer
			               	(tlayer*atm.layer['mmw']*atm.c.amu*atm.c.k_b)) 	).T				  #nlayer)].T

			TAUGAS += ADDTAU
			if debug: opt_figure.line(opacityclass.wno, ADDTAU[int(np.size(tlayer)/2),:], alpha=0.7,legend=m[0]+m[1], line_width=3, color=colors[c],
			muted_color=colors[c], muted_alpha=0.2)
		#H2- 
		elif (m[0] == "H2") and (m[1] == "H2-"): 
			#make sure you know which index of matrix is h2 and e-
			h2_ = np.where(m[0]==np.array(atm.weights.keys()))[0][0]
			#calculate opacity
			#this is a hefty matrix multiplication to make sure that we are 
			#multiplying each column of the opacities by the same 1D vector (as opposed to traditional 
			#matrix multiplication). This is the reason for the transposes.
			ADDTAU = (opacityclass.get_continuum_opac(tlayer, 'H2-').T*( 		          #[(nlayer x nwno).T *(
							atm.layer['pressure']* 								  			  #nlayer
							atm.layer['mixingratios'][:,h2_]*atm.layer['electrons']* #nlayer
			               	atm.layer['colden']/ 											  #nlayer
			               	(atm.layer['mmw']*atm.c.amu)) 	).T								  #nlayer)].T

			TAUGAS += ADDTAU
			if debug: opt_figure.line(opacityclass.wno, ADDTAU[int(np.size(tlayer)/2),:], alpha=0.7,legend=m[0]+m[1], line_width=3, color=colors[c],
			muted_color=colors[c], muted_alpha=0.2)
		#everything else.. e.g. H2-H2, H2-CH4. Automatically determined by which molecules were requested
		else:
			m0 = np.where(m[0]==np.array(atm.weights.keys()))[0][0]
			m1 = np.where(m[1]==np.array(atm.weights.keys()))[0][0]
			#calculate opacity

			ADDTAU = (opacityclass.get_continuum_opac(tlayer, m[0]+m[1]).T * (#[(nlayer x nwno).T *(
								COEF1*													#nlayer
								atm.layer['mixingratios'][:,m0]*						#nlayer
								atm.layer['mixingratios'][:,m1] )  ).T 					#nlayer)].T
			TAUGAS += ADDTAU
			if debug: opt_figure.line(opacityclass.wno, ADDTAU[int(np.size(tlayer)/2),:], alpha=0.7,legend=m[0]+m[1], line_width=3, color=colors[c],
			muted_color=colors[c], muted_alpha=0.2)
		c+=1
	
	#====================== ADD MOLECULAR OPACITY======================	
	for m in atm.molecules:
		if (m =='CH4') or (m =='CO'):
			#METHANE IS WHACK FIX THIS LATER
			continue
		ind = np.where(m==np.array(atm.weights.keys()))[0][0]
		ADDTAU = (opacityclass.get_molecular_opac(tlayer,player, m).T * ( 
					atm.layer['colden']*
					atm.layer['mixingratios'][:,ind]*atm.weights[m].values[0]/ 
			        atm.layer['mmw']) ).T
		TAUGAS += ADDTAU

		if debug: opt_figure.line(opacityclass.wno, ADDTAU[int(np.size(tlayer)/2),:], alpha=0.7,legend=m, line_width=3, color=colors[c],
			muted_color=colors[c], muted_alpha=0.2)
		c+=1

	#====================== ADD RAYLEIGH OPACITY======================	
	ih2 = np.where('H2'==np.array(atm.weights.keys()))[0][0]
	ihe = np.where('He'==np.array(atm.weights.keys()))[0][0]
	#ich4 = np.where('CH4'==np.array(atm.weights.keys()))[0][0]
	TAURAY = rayleigh(atm.layer['colden'],atm.layer['mixingratios'][:,ih2], 
					atm.layer['mixingratios'][:,ihe], atm.layer['mixingratios'][:,ihe], #ich4], 
					opacityclass.wave, atm.layer['mmw'],atm.c.amu )
	if debug: opt_figure.line(opacityclass.wno, TAURAY[int(np.size(tlayer)/2),:], alpha=0.7,legend='Rayleigh', line_width=3, color=colors[c],
			muted_color=colors[c], muted_alpha=0.2)	


	#====================== ADD RAMAN OPACITY======================
	raman_factor = raman(opacityclass.wno) #CURRENTLY ONLY RETURNS ONES 


	#====================== ADD CLOUD OPACITY======================	
	TAUCLD = atm.layer['cloud']['opd'] #TAUCLD is the total extinction from cloud = (abs + scattering)
	asym_factor_cld = atm.layer['cloud']['g0'] 
	single_scattering_cld = atm.layer['cloud']['w0'] #scatter / (abs + scattering) from cloud only 


	#====================== ADD EVERYTHING TOGETHER PER LAYER======================	
	DTAU = TAUGAS + TAURAY + TAUCLD 
	g_tot = (TAUCLD*asym_factor_cld)/(TAUCLD + TAURAY)
	w_tot = (TAURAY*raman_factor + TAUCLD*single_scattering_cld) / (TAUGAS + TAURAY + TAUCLD) #TOTAL single scattering 


	if debug:
		opt_figure.line(opacityclass.wno, DTAU[int(np.size(tlayer)/2),:], legend='TOTAL', line_width=4, color=colors[0],
			muted_color=colors[c], muted_alpha=0.2)
		opt_figure.legend.click_policy="mute"
		output_file('OpacityDebug.html')
		show(opt_figure)

	#====================== TOTAL INTEGRATED EXTINCTION OPTICAL DEPTH======================
	
	#this extra zero is because the total integrated extinction should have nlevel and not, nlayer 
	#Later, the zero will be replaced with the lower boundary condition
	tau_tot =np.concatenate(([0], np.cumsum(DTAU)))

	return tau_tot, g_tot, w_tot

@jit('f8[:,:](f8[:],f8[:],f8[:],f8[:],f8[:],f8[:],f8)')
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
	gasmixing[:,2] = CH4*0 #FIX THISISIS
	#add rayleigh from each contributing gas using corresponding mixing 
	for i in np.arange(0,3,1):
		tec = cfray*(dpol[i]/wave**4)*(gnu[0,i]+gnu[1,i]/   #nwave
                     wave**2)**2 
		TAUR = (cold*gasmixing[:,i]).reshape((-1, 1)) * tec * 1e-5 / XN0
		
		TAURAY += TAUR

	return TAURAY

def raman(wavelength):
	"""
	The Ramam scattering will alter the rayleigh scattering. The returned value is 
	modified single scattering albedo

	Will be added to the rayleigh scattering as : TAURAY*RAMAN
	"""
	return 1.0 


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

		self.mol_temps = np.array([float(i) for i in self.mol_db['H2O'].keys()])
		self.mol_press = {}
		for i in self.mol_temps:
			self.mol_press[str(i)] = np.array([float(i) for i in self.mol_db['H2O'][str(i)].keys()])
		#self.cia_mols = [i for i in self.cia_db.keys()]

		self.wno = wno
		self.wave = wave
		self.nwno = np.size(wno)


	#@jit(nopython=True)
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
		return 10**(a)

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
		nearestT = [self.mol_temps[find_nearest(self.mol_temps, t)].astype(str) for t in temperature]
		nearestP = [self.mol_press[str(t)][find_nearest(self.mol_press[str(t)], p)].astype(str) for p,t in zip(pressure,nearestT)]
		sizeT = np.size(nearestT)
		a = np.zeros((sizeT, self.nwno))
		for i,t,p in zip(range(sizeT) ,nearestT,nearestP): 
			a[i,:] = np.array(self.mol_db[molecule][t][p])
		return a

@jit(nopython=True)
def find_nearest(array,value):
	#small program to find the nearest neighbor in temperature  
	idx = (np.abs(array-value)).argmin()
	return idx

