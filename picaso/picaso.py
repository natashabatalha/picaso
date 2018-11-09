from atmsetup import ATMSETUP
import fluxes 
import wavelength
import numpy as np
import optics 
import os
import pickle as pk
import disco 
import copy

__refdata__ = os.environ.get('picaso_refdata')

def picaso(input,phase_angle, dimension = '1d'):
	"""
	Currently top level program to run albedo code 
	"""

	#check to see if we are running in test mode
	test_mode = input['test_mode']

	#get wavelength grid and order it
	grid = wavelength.get_output_grid(input['opacities']['files']['wavegrid'])
	wno = np.sort(grid['wavenumber'].values)
	nwno = len(wno)

	#check to see if new wavelength grid is necessary .. assuming its always the same here
	#this is where continuum_factory would run, if it was a brand new grid
	#TAG:TODO

	#get opacity class
	opacityclass=optics.RetrieveOpacities(wno, 1e4/wno,
					os.path.join(__refdata__, 'opacities', input['opacities']['files']['continuum']),
					os.path.join(__refdata__, 'opacities', input['opacities']['files']['molecular']) 
					)

	#get geometry
	ng = input['disco']['num_gangle']
	nt = input['disco']['num_tangle']
	gangle,gweight,tangle,tweight = disco.get_angles(ng, nt) 
	#planet disk is divieded into gaussian and chebyshev angles and weights for perfoming the 
	#intensity as a function of planetary pahse angle 
	ubar0, ubar1, cos_theta,lat,lon = disco.compute_disco(ng, nt, gangle, tangle, phase_angle)

	#set star 
	F0PI = np.zeros(nwno) + 1.0 

	#define approximinations 
	delta_eddington = input['approx']['delta_eddington']

	#begin atm setup
	atm = ATMSETUP(input)
	#get star 
	atm.get_stellar_spec(wno) 

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
		DTAU, TAU, W0, COSB,GCOS2= optics.optc(atm, opacityclass,delta_eddington=delta_eddington,test_mode=test_mode)


		#use toon method (and tridiagonal matrix solver) to get net cumulative fluxes 
		xint_at_top  = fluxes.get_flux_geom_1d(atm.c.nlevel, wno,nwno,ng,nt,
													DTAU, TAU, W0, COSB,GCOS2,# dtau_dedd, tau_dedd, w0_dedd, cosb_dedd ,
													atm.surf_reflect, ubar0,ubar1,cos_theta, F0PI)
	elif dimension == '3d':

		#setup zero array to fill with opacities
		TAU_3d = np.zeros((atm.c.nlevel, nwno, ng, nt))
		DTAU_3d = np.zeros((atm.c.nlayer, nwno, ng, nt))
		W0_3d = np.zeros((atm.c.nlayer, nwno, ng, nt))
		COSB_3d = np.zeros((atm.c.nlayer, nwno, ng, nt))
		GCOS2_3d = np.zeros((atm.c.nlayer, nwno, ng, nt))
		#get opacities at each facet
		for g in range(ng):
			for t in range(nt): 

				#edit atm class to only have subsection of 3d stuff 
				atm_1d = copy.deepcopy(atm)

				#diesct just a subsection to get the opacity 
				atm_1d.disect(g,t)

				dtau, tau, w0, cosb, gcos2 = optics.optc(
					atm_1d, opacityclass,delta_eddington=delta_eddington,test_mode=test_mode) 

				DTAU_3d[:,:,g,t] = dtau
				TAU_3d[:,:,g,t] = tau
				W0_3d[:,:,g,t] = w0 
				COSB_3d[:,:,g,t] = cosb
				GCOS2_3d[:,:,g,t]= gcos2 

		#use toon method (and tridiagonal matrix solver) to get net cumulative fluxes 
		xint_at_top  = fluxes.get_flux_geom_3d(atm.c.nlevel, wno,nwno,ng,nt,
													DTAU_3d, TAU_3d, W0_3d, COSB_3d,GCOS2_3d,# dtau_dedd, tau_dedd, w0_dedd, cosb_dedd ,
													atm.surf_reflect, ubar0,ubar1,cos_theta, F0PI)		
	
	#now compress everything based on the weights 
	albedo = disco.compress_disco(ng, nt, nwno, cos_theta, xint_at_top, gweight, tweight,F0PI)
	
	return wno, albedo 


if __name__ == "__main__":
	import json
	import pandas as pd
	from bokeh.plotting import figure, show, output_file
	from bokeh.palettes import inferno
	colors = inferno(19)
	a = json.load(open('../reference/config.json'))
	a['atmosphere']['profile']['filepath'] = '../3d_pt_test.hdf5'#'../reference/base_cases/jupiter.pt'#
	a['planet']['gravity'] = 25
	a['planet']['gravity_unit'] = 'm/(s**2)' 
	a['atmosphere']['clouds']['filepath'] =  '../3d_cld_test.hdf5'#'../reference/base_cases/jupiterf3.cld'#
	#a['approx']['delta_eddington']=False
	a['disco']['num_gangle']= 10
	a['disco']['num_tangle'] = 10
	#prof = pd.read_csv('../test.profile',delim_whitespace=True)
	#prof['temperature'] =prof['temperature']*10.0
	#prof['H-'] =prof['H-']*1e30
	#a['atmosphere']['profile']['profile'] = prof
	#a['planet']['gravity'] = 25
	#a['planet']['gravity_unit'] = 'm/(s**2)' 
	fig = figure(width=1200, height=800)
	ii = 0 
	for phase in [0.0, 0.1745]:#, 0.3491, 0.5236, 0.6981, 0.8727, 1.0472, 1.2217, 1.3963,1.5708, 1.7453, 1.9199, 2.0944, 2.2689, 2.4435, 2.6180, 2.7925, 2.9671, 3.139]:
		print(phase)
		wno, alb = picaso(a, phase,dimension='3d')
		pk.dump([1e4/wno, alb], open('../first_run/phase'+str(phase)+'.pk','wb'))
		fig.line(1e4/wno, alb, line_width = 4, color = colors[ii], legend = 'Phase='+str(phase))
		ii+=1 
		show(fig)
