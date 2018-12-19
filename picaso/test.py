import json
import pandas as pd
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import inferno
import numpy as np
from .justdoit import picaso 
import os 

__refdata__ = os.environ.get('picaso_refdata')
 
def dlugach_test(single_phase = 'TTHG_ray', output_dir = None, rayleigh=True, constant_tau=True):
	"""
	Test the flux against against Dlugach & Yanovitskij 
	https://www.sciencedirect.com/science/article/pii/0019103574901675?via%3Dihub

	There are two tests: rayleigh and constant_tau. This will run through the entire 
	table of values (XXI). Users can run both or specify one. Constant_tau picks 
	single value for the asymmetry factors g = [0.5,0.75,0.8,0.85,0.9] and tests 
	single scattering abled. Rayleigh only varies the ssa (with rayleigh phase fun) 

	This routine contains the functionality to reproduce table XXI and determine the % 
	error in various realms

	Parameters
	----------
	single_phase : str 
		Single phase function to test with the constant_tau test. Can either be: 
		"cahoy","TTHG_ray","TTHG", and "OTHG"
	output_dir : str 
		Output directory for results of test. Default is to just return the dataframe (None). 
	rayleigh : bool
		Default is True. Turns on and off the rayleigh phase function test 
	constant_tau : float
		Default=True. Turns on and off the constant tau phase function test

	Retuns
	------
	DataFrame of % deviation from Dlugach & Yanovitskij Table XXI

	"""

	#read in table from reference data with the test values
	real_answer = pd.read_csv(os.path.join(__refdata__,'base_cases', 'DLUGACH_TEST.csv'))
	real_answer = real_answer.set_index('Unnamed: 0')

	perror = real_answer.copy()

	a = json.load(open(os.path.join(__refdata__,'config.json')))

	p = np.logspace(-5,4,60)
	t = p*0+300
	h2o = p*0 +0.01
	h2 = p*0 + 0.99

	a['atmosphere']['profile']['profile'] = pd.DataFrame({'pressure':p,
														'temperature':t,
														'CH4':h2o,
														'H2':h2/2,
														'He':h2/2})
	a['planet']['gravity'] = 10
	a['planet']['gravity_unit'] = 'm/(s**2)' 
	a['star']['temp'] = 6000 #kelvin
	a['star']['metal'] = 0.0122 #log metal
	a['star']['logg'] = 4.437 #log cgs
	a['atmosphere']['scattering']['g0'] = 0.0

	a['approx']['delta_eddington']=True
	a['approx']['raman']='pollack'
	a['approx']['single_phase'] = 'TTHG_ray'

	a['test_mode']='rayleigh'

	if rayleigh:
		#first test Rayleigh
		for w in real_answer.keys():#[0.0, 0.1745, 0.3491, 0.5236, 0.6981, 0.8727, 1.0472, 1.2217, 1.3963,1.5708, 1.7453, 1.9199, 2.0944, 2.2689, 2.4435, 2.6180, 2.7925, 2.9671, 3.139]:
			if float(w) ==1.000:
				w0 = 0.999999
			else: 
				w0 = float(w)
			a['atmosphere']['scattering']['w0'] = w0
			wno, alb = picaso(a, 0.0)
			perror.loc[-1][w] = (100*(alb[-1]-real_answer.loc[-1][w])/real_answer.loc[-1][w])
			print("rayleigh",(100*(alb[-1]-real_answer.loc[-1][w])/real_answer.loc[-1][w]))

	a['test_mode']='constant_tau'
	a['approx']['single_phase'] = single_phase

	#first test Rayleigh
	#from bokeh.plotting import figure, show
	#fig = figure()
	if constant_tau:
		for g0 in real_answer.index[1:]:#[6:]:
			for w in real_answer.keys():#[7:]:
				if float(w) ==1.000:
					w0 = 0.999999
				else: 
					w0 = float(w)
				a['atmosphere']['scattering']['g0'] = g0
				a['atmosphere']['scattering']['w0'] = w0
				wno, alb = picaso(a, 0.0)
				#fig.line(1e4/wno, alb, legend=str(g0)+str(w0))
				perror.loc[g0][w] = (100*(alb[-1]-real_answer.loc[g0][w])/real_answer.loc[g0][w])
				print(g0,w0,(100*(alb[-1]-real_answer.loc[g0][w])/real_answer.loc[g0][w]))
				#show(fig)
		if output_dir!=None: perror.to_csv(os.path.join(output_dir,'test_results.csv'))
	return perror

if __name__ == "__main__":
	run_all_test()
