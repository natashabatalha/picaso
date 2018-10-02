from atmsetup import ATMSETUP
import fluxes 
import wavelength
import numpy as np
import optics 
import os
import pickle as pk
import disco 

__refdata__ = os.environ.get('picaso_refdata')

def picaso(input,phase_angle):
	"""
	Currently top level program to run albedo code 
	"""

	#setup atmosphere 
	#phase_angle = 0

	atm = ATMSETUP(input)

	atm.get_profile()
	atm.get_gravity()
	#now can get these 
	atm.get_mmw()
	atm.get_density()
	atm.get_column_density()
	#get needed continuum molecules 
	atm.get_needed_continuum()


	#get wavelength grid and order it
	grid = wavelength.get_output_grid(atm.input['opacities']['files']['wavegrid'])
	wno = np.sort(grid['wavenumber'].values)
	nwno = len(wno)

	#get cloud properties, if there are any and put it on current grid 
	atm.get_clouds(wno)
	#get star 
	atm.get_stellar_spec(wno) 

	#check to see if new wavelength grid is necessary .. assuming its always the same here
	#this is where continuum_factory would run, if it was a brand new grid
	#TAG:TODO

	#get opacity class
	opacityclass=optics.RetrieveOpacities(wno, 1e4/wno,
					os.path.join(__refdata__, 'opacities', atm.input['opacities']['files']['continuum']),
					os.path.join(__refdata__, 'opacities', atm.input['opacities']['files']['molecular']) 
					)

	#Make sure that all molecules are in opacityclass. If not, remove them and add warning
	no_opacities = [i for i in atm.molecules if i not in opacityclass.molecules]
	atm.add_warnings('No computed opacities for: '+','.join(no_opacities))
	atm.molecules = np.array([ x for x in atm.molecules if x not in no_opacities ])

	#get geometry
	#. 100 of these, 10 of these
	gangle,gweight,tangle,tweight = disco.get_angles(__refdata__) 
	ng = len(gangle)
	nt = len(tangle)

	#set star 
	F0PI = np.zeros(nwno) + 1.0 

    ################ From here on out is everything that must go through retrieval##############

	#get optics (need D-eddington approximations and uncorrected )
	DTAU, TAU, W0, COSB,GCOS2, dtau_dedd, tau_dedd, w0_dedd, cosb_dedd = optics.optc(atm, opacityclass)

	#determine surface reflectivity as function of wavelength (set to zero here)
	surf_reflect = np.zeros(nwno)

	#now get the geometric albedo given the layer opacities for each single wavelength
	#planet disk is divieded into gaussian and chebyshev angles and weights for perfoming the 
	#intensity as a function of planetary pahse angle 
	ubar0, ubar1, cos_theta = disco.compute_disco(ng, nt, gangle, tangle, phase_angle)

	#use toon method (and tridiagonal matrix solver) to get net cumulative fluxes 
	xint_at_top  = fluxes.get_flux_geom(atm.c.nlevel, wno,nwno,ng,nt,
													DTAU, TAU, W0, COSB,GCOS2, dtau_dedd, tau_dedd, w0_dedd, cosb_dedd ,
													surf_reflect, ubar0,ubar1,cos_theta, F0PI)

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
	a['atmosphere']['profile']['filepath'] = '../reference/base_cases/jupiter.pt'
	a['planet']['gravity'] = 25
	a['planet']['gravity_unit'] = 'm/(s**2)' 
	a['atmosphere']['clouds']['filepath'] = '../reference/base_cases/jupiterf3.cld' 
	#prof = pd.read_csv('../test.profile',delim_whitespace=True)
	#prof['temperature'] =prof['temperature']*10.0
	#prof['H-'] =prof['H-']*1e30
	#a['atmosphere']['profile']['profile'] = prof
	#a['planet']['gravity'] = 25
	#a['planet']['gravity_unit'] = 'm/(s**2)' 
	fig = figure(width=1200, height=800)
	ii = 0 
	for phase in [0.0, 0.1745, 0.3491, 0.5236, 0.6981, 0.8727, 1.0472, 1.2217, 1.3963,1.5708, 1.7453, 1.9199, 2.0944, 2.2689, 2.4435, 2.6180, 2.7925, 2.9671, 3.139]:
		print(phase)
		wno, alb = picaso(a, phase)
		pk.dump([1e4/wno, alb], open('../first_run/phase'+str(phase)+'.pk','wb'))
		fig.line(1e4/wno, alb, line_width = 4, color = colors[ii], legend = 'Phase='+str(phase))
		ii+=1 
		show(fig)
