from atmsetup import ATMSETUP
import fluxes 

import wavelength
import numpy as np
import optics 
import os
import pickle as pk
__refdata__ = os.environ.get('picaso_refdata')

def picaso(input):
	"""
	Currently top level program to run albedo code 
	"""

	#setup atmosphere 
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

	#get opacity
	opacityclass=optics.RetrieveOpacities(wno, 1e4/wno,
					os.path.join(__refdata__, 'opacities', atm.input['opacities']['files']['continuum']),
					os.path.join(__refdata__, 'opacities', atm.input['opacities']['files']['molecular']) 
					)

	#Make sure that all molecules are in opacityclass. If not, remove them and add warning
	no_opacities = [i for i in atm.molecules if i not in opacityclass.molecules]
	atm.add_warnings('No computed opacities for: '+','.join(no_opacities))
	atm.molecules = np.array([ x for x in atm.molecules if x not in no_opacities ])

	#get optics 
	DTAU,  WBAR, COSB  = optics.optc(atm, opacityclass)

	#use toon method (and tridiagonal matrix solver) to get net cumulative fluxes 
	ubar0 = 0.5 #hemispheric constant, sqrt(3) = guass quadrature
	F0PI = np.zeros(nwno) + 1.0 
	surf_reflect = np.zeros(nwno)

	flux_plus, flux_minus  = fluxes.get_flux_toon(atm.c.nlevel, wno,nwno,
														DTAU, WBAR, COSB, surf_reflect, ubar0, F0PI)

	

	return atm


if __name__ == "__main__":
	import json
	import pandas as pd
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
	picaso(a)