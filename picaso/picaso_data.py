import json
import pandas as pd
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import inferno
import numpy as np
from .justdoit import opannection, inputs
import os 
import astropy.units as u

__refdata__ = os.environ.get('picaso_refdata')
 
def picaso_albedos(single_phase = 'OTHG', output_dir = None, rayleigh=True, phase=True, 
	method="Toon", stream=2, Toon_coefficients="quadrature", delta_eddington=False,
	disort_data=False, phangle=0):
	"""
	Generate data to compare against DISORT. Must also run picaso_compare.py in pyDISORT
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
	DataFrame 
	"""

	#read in table from reference data with the test values
	real_answer = pd.read_csv(os.path.join(__refdata__,'base_cases', 'DLUGACH_TEST.csv'))
	real_answer = real_answer.set_index('Unnamed: 0')
	df = pd.DataFrame(columns=list(real_answer.keys())+['0.65','0.60','0.55','0.50'], 
					index=real_answer.index)
	perror = df.copy()#real_answer.copy()

	nlevel = 20

	opa = opannection(wave_range=[0.3,0.5], resample=100)
	start_case=inputs()
	start_case.phase_angle(phangle) #radians
	start_case.gravity(gravity = 25, gravity_unit=u.Unit('m/(s**2)'))
	start_case.star(opa, 6000,0.0122,4.437) #kelvin, log metal, log cgs
	start_case.atmosphere(df=pd.DataFrame({'pressure':np.logspace(-6,3,nlevel),
	                                    'temperature':np.logspace(-6,3,nlevel)*0+1000 ,
	                                    'H2':np.logspace(-6,3,nlevel)*0+0.99, 
	                                     'H2O':np.logspace(-6,3,nlevel)*0+0.01}))

	start_case.inputs['approx']['delta_eddington']=delta_eddington
	start_case.approx(raman='none', method = method, stream = stream, 
				Toon_coefficients = Toon_coefficients,
				delta_eddington=delta_eddington)

	if method=="Toon":
		#disort_dir = '/Users/crooney/Documents/codes/pyDISORT/test/picaso_data/'
		disort_dir = '/Users/crooney/Documents/codes/picaso/picaso/cdisort_comparison/picaso_data/'
	else:
		if stream==2:
			#disort_dir = '/Users/crooney/Documents/codes/pyDISORT/test/picaso_data/SH2/'
			disort_dir = '/Users/crooney/Documents/codes/picaso/picaso/cdisort_comparison/picaso_data/SH2/'
		elif stream==4:
			#disort_dir = '/Users/crooney/Documents/codes/pyDISORT/test/picaso_data/SH4/'
			disort_dir = '/Users/crooney/Documents/codes/picaso/picaso/cdisort_comparison/picaso_data/SH4/'

	if rayleigh: 
		#first test Rayleigh
		for w in df.keys():
			if float(w) ==1.000:
				w0 = 0.999999
			else: 
				w0 = float(w)		
			start_case.inputs['test_mode'] = 'rayleigh'

			start_case.clouds(df=pd.DataFrame({'opd':sum([[i]*196 for i in 10**np.linspace(-5, 3, nlevel-1)],[]),
			                                    'w0':np.zeros(196*(nlevel-1)) + w0 ,
			                                    'g0':np.zeros(196*(nlevel-1)) + 0}))

			if disort_data:
			    #disort_dir_ = disort_dir + 'data_rayleigh_%.3f.pk' % w0
			    disort_dir_ = disort_dir + 'data_rayleigh_%.3f.pk' % w0
			    start_case.inputs['approx']['input_dir']=disort_dir_
			allout = start_case.spectrum(opa, calculation='reflected')#, full_output=True)

			alb = allout['albedo']
			perror.loc[-1][w] = alb[-1]


	start_case.inputs['test_mode']='constant_tau'
	start_case.approx(single_phase = 'OTHG', method = method, stream = stream, 
				Toon_coefficients = Toon_coefficients,
				delta_eddington=delta_eddington)

	if phase:
		for g0 in df.index[1:]:
			for w in df.keys():
				if float(w) ==1.000:
					w0 = 0.999999
				else: 
					w0 = float(w)

				start_case.clouds(df=pd.DataFrame({'opd':sum([[i]*196 for i in 10**np.linspace(-4, 2, nlevel-1)],[]),
				                                    'w0':np.zeros(196*(nlevel-1)) + w0 ,
				                                    'g0':np.zeros(196*(nlevel-1)) + g0}))

				if disort_data:
					disort_dir_ = disort_dir + 'data_%.3f_%.3f.pk' % (g0, w0)
					start_case.inputs['approx']['input_dir']=disort_dir_
				allout = start_case.spectrum(opa, calculation='reflected')#, full_output=True)

				alb = allout['albedo']
				perror.loc[g0][w] = alb[-1]
	
	if output_dir!=None: perror.to_csv(os.path.join(output_dir))
	if disort_data:
	    print('RUN DISORT CODE NOW')
	else:
	    print('NO DATA GENERATED FOR DISORT')
	#import IPython; IPython.embed()
	return perror

