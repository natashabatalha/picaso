from .atmsetup import ATMSETUP
from .fluxes import get_flux_geom_1d, get_flux_geom_3d 
from .wavelength import get_output_grid
import numpy as np
from .optics import RetrieveOpacities,compute_opacity
import os
import pickle as pk
from .disco import get_angles, compute_disco, compress_disco
import copy
import json
__refdata__ = os.environ.get('picaso_refdata')

def picaso(input,phase_angle, dimension = '1d', full_output=False):
	"""
	Currently top level program to run albedo code 

	Parameters 
	----------
	input : dict 
		This input dict is built by loading the input = `justdoit.load_inputs()` 
	phase_angle : int 	
		Phase angle of the planet in radians 
	dimension : str 
		(Optional) Dimensions of the calculation. Default = '1d'. But '3d' is also accepted. 
		In order to run '3d' calculations, user must build 3d input (see tutorials)
	full_output : bool 
		(Optional) Default = False. Returns atmosphere class, which enables several 
		plotting capabilities. 

	Return
	------
	Wavenumber, albedo if full_output=False 
	Wavenumber, albedo, atmosphere if full_output = True 
	"""

	#check to see if we are running in test mode
	test_mode = input['test_mode']

	#set approx numbers options (to be used in numba compiled functions)
	single_phase, multi_phase, raman_approx = set_approximations(input) 

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
	wno_star, flux_star, hires_wno_star, hires_flux_star = atm.get_stellar_spec(opacityclass.wno, star['database'],star['temp'], star['metal'], star['logg']) 
	#this is added to the opacity class
	stellar_raman_shifts = opacityclass.compute_stellar_shits(hires_wno_star, hires_flux_star)


	################ From here on out is everything that would go through retrieval or 3d input##############
	atm.get_gravity()
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
			full_output=full_output)

		#use toon method (and tridiagonal matrix solver) to get net cumulative fluxes 
		xint_at_top  = get_flux_geom_1d(atm.c.nlevel, wno,nwno,ng,nt,
													DTAU, TAU, W0, COSB,GCOS2,ftau_cld,ftau_ray,
													DTAU_OG, TAU_OG, W0_OG, COSB_OG ,
													atm.surf_reflect, ubar0,ubar1,cos_theta, F0PI,
													single_phase,multi_phase)
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
											single_phase,multi_phase)
	
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
	single_phase = single_phase_options().index(input['approx']['single_phase'])

	multi_phase = multi_phase_options().index(input['approx']['multi_phase'])

	raman_approx = raman_options().index(input['approx']['raman'])

	return single_phase, multi_phase, raman_approx

def load_inputs():
	"""Function to load empty inputs"""
	return json.load(open(os.path.join(__refdata__,'config.json')))
def jupiter_pt():
	"""Function to get Jupiter's PT profile"""
	return os.path.join(__refdata__, 'base_cases','jupiter.pt')
def jupiter_cld():
	"""Function to get rough Jupiter Cloud model with fsed=3"""
	return os.path.join(__refdata__, 'base_cases','jupiterf3.cld')
def single_phase_options():
	"""Retrieve all the options for direct radation"""
	return ['cahoy','OTHG','TTHG','TTGH_ray']
def multi_phase_options():
	"""Retrieve all the options for multiple scattering radiation"""
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
