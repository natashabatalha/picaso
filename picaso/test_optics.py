from .atmsetup import ATMSETUP
from .wavelength import get_output_grid
import numpy as np
from .optics import compute_opacity,RetrieveOpacities
import os
import pickle as pk
__refdata__ = os.environ.get('picaso_refdata')

def test_optc(input):
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
	grid = get_output_grid(atm.input['opacities']['files']['wavegrid'])
	wno = np.sort(grid['wavenumber'].values)

	#get cloud properties, if there are any and put it on current grid 
	atm.get_clouds(wno)
	#get star 
	atm.get_stellar_spec(wno) 

	#check to see if new wavelength grid is necessary .. assuming its always the same here
	#this is where continuum_factory would run, if it was a brand new grid
	#TAG:TODO

	#get opacity
	opacityclass=RetrieveOpacities(wno, 1e4/wno,
					os.path.join(__refdata__, 'opacities', atm.input['opacities']['files']['continuum']),
					os.path.join(__refdata__, 'opacities', atm.input['opacities']['files']['molecular']) 
					)

	#Make sure that all molecules are in opacityclass. If not, remove them and add warning
	no_opacities = [i for i in atm.molecules if i not in opacityclass.molecules]
	atm.add_warnings('No computed opacities for: '+','.join(no_opacities))
	atm.molecules = np.array([ x for x in atm.molecules if x not in no_opacities ])



	#get optics 
	DTAU,  WBAR, COSB  = compute_opacity(atm, opacityclass)
	pk.dump([DTAU, WBAR, COSB], open('../testing_notebooks/OPTC_Sep2018.pk','wb'))
	#solve for coefficients of two stream: upward flux and downward flux at each layer
	#fluxes.get_flux_toon(nlevel, wno, DTAU, TAU, WBAR, COSB)


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
	test_optc(a)