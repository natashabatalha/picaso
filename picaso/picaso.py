from atmsetup import ATMSETUP
import wavelength
import numpy as np
import optics 
import os
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

	#get cloud properties, if there are any and put it on current grid 
	atm.get_clouds(wno)

	#check to see if new wavelength grid is necessary .. assuming its always the same here
	#this is where continuum_factory would run, if it was a brand new grid
	#TAG:TODO

	#get opacity
	opacityclass=optics.RetrieveOpacities(wno, 1e4/wno,
					os.path.join(__refdata__, 'opacities', atm.input['opacities']['files']['continuum']),
					os.path.join(__refdata__, 'opacities', atm.input['opacities']['files']['molecular']) 
					)

	#get optics 
	tau_tot, g_tot, w_tot = optics.optc(atm, opacityclass)

	#calculate fluxes 
	

	return atm


if __name__ == "__main__":
	import json
	import pandas as pd
	a = json.load(open('../reference/config.json'))
	a['atmosphere']['profile']['filepath'] = '../test.profile'
	a['planet']['gravity'] = 25
	a['planet']['gravity_unit'] = 'm/(s**2)' 
	a['atmosphere']['clouds']['filepath'] = '../test.cld' 
	#prof = pd.read_csv('../test.profile',delim_whitespace=True)
	#prof['temperature'] =prof['temperature']*10.0
	#prof['H-'] =prof['H-']*1e30
	#a['atmosphere']['profile']['profile'] = prof
	#a['planet']['gravity'] = 25
	#a['planet']['gravity_unit'] = 'm/(s**2)' 	
	picaso(a)