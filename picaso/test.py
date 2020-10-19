import json
import pandas as pd
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import inferno
import numpy as np
from .justdoit import opannection, inputs
import os 
import astropy.units as u

__refdata__ = os.environ.get('picaso_refdata')
 
def dlugach_test(single_phase = 'OTHG', output_dir = None, rayleigh=True, phase=True, approximation='2steam', stream=2):
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
	phase : float
		Default=True. Turns on and off the constant g phase function test

	Retuns
	------
	DataFrame of % deviation from Dlugach & Yanovitskij Table XXI
	"""

	#read in table from reference data with the test values
	real_answer = pd.read_csv(os.path.join(__refdata__,'base_cases', 'DLUGACH_TEST.csv'))
	real_answer = real_answer.set_index('Unnamed: 0')

	perror = real_answer.copy()

	nlevel = 20 

	opa = opannection(wave_range=[0.3,0.5], resample=10)
	start_case=inputs()
	start_case.phase_angle(0) #radians
	start_case.gravity(gravity = 25, gravity_unit=u.Unit('m/(s**2)'))
	start_case.approx(raman='none', )
	start_case.star(opa, 6000,0.0122,4.437) #kelvin, log metal, log cgs
	start_case.atmosphere(df=pd.DataFrame({'pressure':np.logspace(-6,3,nlevel),
	                                    'temperature':np.logspace(-6,3,nlevel)*0+1000 ,
	                                    'H2':np.logspace(-6,3,nlevel)*0+0.99, 
	                                     'H2O':np.logspace(-6,3,nlevel)*0+0.01}))

	start_case.inputs['approx']['delta_eddington']=False

	if rayleigh: 
		#first test Rayleigh
		for w in real_answer.keys():#[0.0, 0.1745, 0.3491, 0.5236, 0.6981, 0.8727, 1.0472, 1.2217, 1.3963,1.5708, 1.7453, 1.9199, 2.0944, 2.2689, 2.4435, 2.6180, 2.7925, 2.9671, 3.139]:
			if float(w) ==1.000:
				w0 = 0.999999
			else: 
				w0 = float(w)		
			start_case.inputs['test_mode'] = 'rayleigh'

			start_case.clouds(df=pd.DataFrame({'opd':sum([[i]*196 for i in 10**np.linspace(-5, 3, nlevel-1)],[]),
			                                    'w0':np.zeros(196*(nlevel-1)) + w0 ,
			                                    'g0':np.zeros(196*(nlevel-1)) + 0}))
			allout = start_case.spectrum(opa, calculation='reflected', approximation=approximation, stream=stream)

			alb = allout['albedo']
			perror.loc[-1][w] = alb[-1]#(100*(alb[-1]-real_answer.loc[-1][w])/real_answer.loc[-1][w])

	start_case.inputs['test_mode']='constant_tau'
	start_case.approx(single_phase = 'OTHG') 

	#first test Rayleigh
	if phase:
		for g0 in real_answer.index[1:]:#[6:]:
			for w in real_answer.keys():#[7:]:
				if float(w) ==1.000:
					w0 = 0.999999
				else: 
					w0 = float(w)

				start_case.clouds(df=pd.DataFrame({'opd':sum([[i]*196 for i in 10**np.linspace(-5, 3, nlevel-1)],[]),
				                                    'w0':np.zeros(196*(nlevel-1)) + w0 ,
				                                    'g0':np.zeros(196*(nlevel-1)) + g0}))
				allout = start_case.spectrum(opa, calculation='reflected', approximation=approximation, stream=stream)

				alb = allout['albedo']
				perror.loc[g0][w] = alb[-1]#(100*(alb[-1]-real_answer.loc[-1][w])/real_answer.loc[-1][w])
	
	if output_dir!=None: perror.to_csv(os.path.join(output_dir))
	return perror

def madhu_test(rayleigh=True, isotropic=True, asymmetric=True, single_phase = 'TTHG_ray', output_dir = None):
	"""
	Test the flux against against Madhu and Burrows  
	https://arxiv.org/pdf/1112.4476.pdf

	There are three tests: rayleigh, isotropic, and lambert. Reproduces Figure 2 of 
	Madhu and Burrows paper. 

	Parameters
	----------
	rayleigh : bool 
		Tests rayleigh phase function 
	isotropic : bool 
		Tests isotropic phase function 
	single_phase : str 
		Single phase function to test with the constant_tau test. Can either be: 
		"cahoy","TTHG_ray","TTHG", and "OTHG"
	output_dir : str 
		Output directory for results of test. Default is to just return the dataframe (None). 


	Returns
	-------
	DataFrame of % deviation from Dlugach & Yanovitskij Table XXI

	"""

	#read in table from reference data with the test values
	real_answer = pd.read_csv(os.path.join(__refdata__,'base_cases', 'MADHU_isotropic.csv'))
	
	#perror = real_answer.copy()

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
	a['atmosphere']['cloud']['g0'] = 0.0
	a['phase_angle']=0
	if rayleigh:
		#SCALAR RAYLEIGH
		a['approx']['delta_eddington']=True
		a['approx']['raman']='pollack'
		a['approx']['single_phase'] ='TTHG_ray'
		a['test_mode']='rayleigh'
		a['atmosphere']['cloud']['g0'] = 0

		for i in real_answer.index:
			a['atmosphere']['cloud']['w0'] = real_answer['ssa'][i]
			wno, alb = picaso(a)
			real_answer.loc[i,'rayleigh']=alb[-1] 

	if isotropic:
		#constant tau
		a['approx']['delta_eddington'] = True
		a['approx']['raman']='pollack'
		a['approx']['single_phase'] = 'OTHG'
		a['test_mode']='constant_tau'
		a['atmosphere']['cloud']['g0'] = 0

		for i in real_answer.index:
			a['atmosphere']['cloud']['w0'] = real_answer['ssa'][i]
			wno, alb = picaso(a)
			real_answer.loc[i,'0.0']=alb[-1] 

	if asymmetric:
		#constant tau
		a['approx']['delta_eddington']=True
		a['approx']['raman']='pollack'
		a['approx']['single_phase'] = single_phase

		a['test_mode']='constant_tau'

		for g in [0.2,0.4,0.6,0.8]: 
			a['atmosphere']['cloud']['g0'] = g
			for i in real_answer.index:
				a['atmosphere']['cloud']['w0'] = real_answer['omega'][i]
				wno, alb = picaso(a)
				real_answer.loc[i,str(g)]=alb[-1] 
	return real_answer
