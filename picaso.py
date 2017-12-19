from atmsetup import ATMSETUP
import opacity
import numpy as np
def picaso(input):
	"""
	Currently top level program to run albedo code 
	"""

	#setup atmosphere 
	atm = ATMSETUP(input)
	atm.get_profile()
	atm.get_gravity()
	atm.get_mmw()
	atm.get_density()

	#get wavelength grid and order it
	file = input['opacities']['files']['wavegrid']
	grid = opacity.get_wave_grid(file)
	wno = np.array(grid['wavenumber'])[::-1]

	#get H2 PIA 
	file = input['opacities']['files']['h2cia']
	h2cia = opacity.set_PIA(file, wno)

	#add h2- opacity 
	file = input['opacities']['files']['h2minus']
	h2minus = opacity.get_h2minus(file, wno)