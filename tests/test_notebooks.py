import json
import pandas as pd
#from bokeh.plotting import figure, show, output_file
#from bokeh.palettes import inferno
#from bokeh.palettes import RdGy
import numpy as np
from picaso import justdoit as jdi
#from picaso import justplotit as jpi
# from .justdoit import opannection, inputs, brown_dwarf_pt,brown_dwarf_cld
import os 
import astropy.units as u


__refdata__ = os.environ.get('picaso_refdata')


# def example_test(wave_range=[0.3,1], phase_angle=0, gravity=25, create_data=False):
# 	"""
# 	This tests the functionality of BLAH based on notebook BLAH 
	
# 	Parameters
# 	----------
# 	add parameters here 
# 	"""

# 	opacity = jdi.opannection(wave_range=wave_range) #lets just use all defaults

# 	start_case = jdi.inputs()

# 	#phase angle
# 	start_case.phase_angle(0) #radians

# 	#define gravity
# 	start_case.gravity(gravity=25, gravity_unit=jdi.u.Unit('m/(s**2)')) #any astropy units available

# 	#define star
# 	start_case.star(opacity, 5000,0,4.0) #opacity db, pysynphot database, temp, metallicity, logg	

	
# 	start_case.atmosphere(filename=jdi.jupiter_pt(), delim_whitespace=True)

# 	df = start_case.spectrum(opacity, calculation='reflected') 

# 	#add informative name here for your file
# 	#this file should be ascii, hdf5, or json
# 	benchmark_spectrum = pd.read_csv(os.path.join(__refdata__,'base_cases','testing','example_test.csv'))

# 	#determine what you think is "good enough"
# 	assert np.allclose(benchmark_spectrum , df['albedo'], .001),'Failed example_test unit test'

# 	if create_data: 
# 		pd.DataFrame(dict(benchmark_spectrum=benchmark_spectrum)).to_csv('example_test.csv')


def test_reflected_1d(create_data=False):
	"""
	This tests the functionality of 1D reflected light spectra from notebook #1 'Getting Started'

	"""

	opacity = jdi.opannection(wave_range=[0.3,1])

	start_case = jdi.inputs()

	#phase angle
	start_case.phase_angle(0) #radians

	#define gravity
	start_case.gravity(gravity=25, gravity_unit=jdi.u.Unit('m/(s**2)')) #any astropy units available

	#define star
	start_case.star(opacity, 5000,0,4.0) #opacity db, pysynphot database, temp, metallicity, logg

	start_case.atmosphere(filename=jdi.jupiter_pt(), delim_whitespace=True)

	df = start_case.spectrum(opacity)

	wno, alb, fpfs = df['wavenumber'] , df['albedo'] , df['fpfs_reflected']
	wno, alb = jdi.mean_regrid(wno, alb , R=150)

	if create_data: 
		# Create a new dataframe
		new_df = pd.DataFrame()

		# Add new columns to the dataframe
		new_df['wavenumber'] = wno
		new_df['albedo'] = alb

	# compare to benchmark
	benchmark_path = os.path.join(__refdata__,'base_cases','testing','reflected_1d_basecase.csv')
	assert os.path.isfile(benchmark_path), 'Benchmark file not found'
	benchmark_spectrum = pd.read_csv(benchmark_path)
	assert np.allclose(benchmark_spectrum['albedo'], alb, atol=0.01), 'Failed albedo reflected_1d test'

	#get relative flux as well
	start_case.star(opacity, 5000,0,4.0,semi_major=1, semi_major_unit=jdi.u.Unit('au'))
	start_case.gravity(radius=1, radius_unit=jdi.u.Unit('R_jup'),
					mass = 1, mass_unit=jdi.u.Unit('M_jup'))
	df = start_case.spectrum(opacity)
	wno, alb, fpfs = df['wavenumber'] , df['albedo'] , df['fpfs_reflected']
	wno2, fpfs = jdi.mean_regrid(wno, fpfs , R=150)

	if create_data:
		# Add new columns to the dataframe
		new_df['fpfs'] = fpfs

	# compare to benchmark
	assert np.allclose(benchmark_spectrum['fpfs'], fpfs, atol=0.01), 'Failed fpfs reflected_1d test'

	# method 2
	start_case.atmosphere( df = jdi.pd.DataFrame({'pressure':np.logspace(-6,2,60),
                                                 'temperature':np.logspace(-6,2,60)*0+200,
                                                 "H2":np.logspace(-6,2,60)*0+0.837,
                                                 "He":np.logspace(-6,2,60)*0+0.163,
                                                 "CH4":np.logspace(-6,2,60)*0+0.000466})
                     )
	
	df = start_case.spectrum(opacity)
	wno_ch4, alb_ch4, fpfs = df['wavenumber'] , df['albedo'] , df['fpfs_reflected']
	wno_ch4, alb_ch4 = jdi.mean_regrid(wno_ch4, alb_ch4 , R=150)

	if create_data:
		# Add new columns to the dataframe
		new_df['albedo_ch4'] = alb_ch4

	# compare to benchmark
	assert np.allclose(benchmark_spectrum['albedo_ch4'], alb_ch4, atol=0.01), 'Failed CH4 albedo reflected_1d test'

	# exclude water molecule
	start_case.atmosphere(filename=jdi.jupiter_pt(), exclude_mol='H2O', delim_whitespace=True)

	df = start_case.spectrum(opacity)
	wno_nowater, alb_nowater, fpfs = df['wavenumber'] , df['albedo'] , df['fpfs_reflected']
	wno_nowater, alb_nowater= jdi.mean_regrid(wno_ch4, alb_ch4 , R=150)

	if create_data:
		# Add new columns to the dataframe
		new_df['albedo_nowater'] = alb_nowater
		savepath = os.path.join(__refdata__,'base_cases','testing','reflected_1d_basecase.csv')
		new_df.to_csv(savepath, index = False)

	# compare to benchmark, added equal_nan=True to account for NaN values present in the benchmark
	assert np.allclose(benchmark_spectrum['albedo_nowater'], alb_nowater, atol=0.01, equal_nan=True), 'Failed no H2O albedo reflected_1d test'


def test_model_storage():
	"""
	This tests the functionality of model storage and read-in from notebook 'Model Storage'

	"""
	opa = jdi.opannection(wave_range=[0.3,1])
	pl = jdi.inputs()#calculation='brown')
	pl.gravity(radius=1, radius_unit=u.Unit('R_jup'),
			mass=1, mass_unit=u.Unit('M_jup'))
	pl.atmosphere(filename=jdi.jupiter_pt(), sep='\s+')
	pl.phase_angle(0)
	pl.clouds(filename=jdi.jupiter_cld(), sep='\s+')
	pl.star(opa, 5000,0,4, radius=1, radius_unit=u.Unit("R_sun"), semi_major=1, semi_major_unit=1*u.AU)
	#MUST USE full output=True for this functionality
	df= pl.spectrum(opa, calculation='reflected', full_output=True)

	preserve = jdi.output_xarray(df,pl,savefile=os.path.join(__refdata__,'base_cases','testing','test_storage_spec.nc'))

	#test read in
	opa = jdi.opannection(wave_range=[0.3,14])
	ds = jdi.xr.load_dataset(os.path.join(__refdata__,'base_cases','testing','test_storage_spec.nc'))
	reuse = jdi.input_xarray(ds, opa)

	#NOTE: error will pop up in newdev because of having the gravity saved instead of having mp and rp saved. Will investigate later
	new_model = reuse.spectrum(opa, calculation='reflected+thermal', full_output=True)
	new_output= jdi.output_xarray(new_model, reuse)


	# test metadata
	# didn't copy block of code from notebook since it's just a repeat of the first block
	mh = 0
	cto = 1

	preserve = jdi.output_xarray(df,pl,add_output={
		'author':"Awesome Scientist",
		'contact' : "awesomescientist@universe.com",
		'code' : "picaso",
		'planet_params':{'mh':mh, 'co':cto},
		'cloud_params':str({'fsed':3})
	})

	# compare to benchmark
	benchmark_model = jdi.xr.load_dataset(os.path.join(__refdata__,'base_cases','testing','model_storage_basecase.nc'))

	#compare numerical values
	assert np.allclose(benchmark_model['albedo'], preserve['albedo'], atol=0.01), 'Failed save and read-in model_storage test'
	#compare metadata
	assert benchmark_model.attrs == preserve.attrs, 'Failed model_storage metadata test'
	#compare dimensions and variables
	assert benchmark_model['albedo'].shape == preserve['albedo'].shape, 'Failed model_storage test: shape mismatch'
	
def test_it_all(): 
	#reflected 1d 

	#thermal 1d 

	#reflected 3d 

	#thermal 3d 

	#climate tests 

	#transit 1d 
	return 
	
def thermal_sh_test(single_phase = 'OTHG', output_dir = None, 
	phase=True, method="toon", stream=2, toon_coefficients="quadrature", delta_eddington=True,
	disort_data=False, phangle=0, tau=0.2):
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
	tau : float 
		constant per layer opacity, 0.2 used as default in rooney+2023 tests
	Retuns
	------
	DataFrame 
	"""
	calculation = 'thermal'

	# high single scattering 
	df = pd.DataFrame(columns = ['1.0','0.999','0.995','0.990','0.980','0.950','0.90','0.8','0.7','0.6','0.5','0.4','0.3','0.2','0.1'],
                   index = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.999])
	perror = df.copy()

	nlevel = 20

	opa = opannection(wave_range=[1,10], resample=100,verbose=False)
	start_case=inputs(calculation='browndwarf')
	start_case.phase_angle(0)
	#start_case.inputs['test_mode']='test'
	start_case.gravity(gravity=200 , gravity_unit=u.Unit('m/s**2'))
	start_case.surface_reflect(0,opa.wno)
	
	cloudy_file = brown_dwarf_pt()
	atmos=pd.read_csv(cloudy_file,delim_whitespace=True)
	start_case.atmosphere(df=atmos)
	cld=pd.read_csv(brown_dwarf_cld(),delim_whitespace=True)

	start_case.inputs['approx']['rt_params']['common']['delta_eddington']=delta_eddington
	

	start_case.inputs['test_mode']='constant_tau'
	start_case.surface_reflect(0,opa.wno)

	if phase:
		for g in perror.index:
			for w in perror.keys():
				if float(w) ==1.000:
					w0 = 0.999999
				else: 
					w0 = float(w)
				g0 = float(g)

				#start_case.clouds(df=pd.DataFrame({'opd':sum([[i]*196 for i in 10**np.linspace(-4, 2, nlevel-1)],[]),
				#                                    'w0':np.zeros(196*(nlevel-1)) + w0 ,
				#                                    'g0':np.zeros(196*(nlevel-1)) + g0}))
				CLD = cld
				if tau is not None:
					CLD['opd'] = 0*CLD['opd'] + tau
				CLD['w0']=0*CLD['w0']+w0
				CLD['g0']=0*CLD['g0']+g0
				start_case.clouds(df=CLD)

				#if disort_data:
				#	disort_dir_ = disort_dir + 'data_%.3f_%.3f.pk' % (g0, w0)
				#	start_case.inputs['approx']['input_dir']=disort_dir_
				#if disort_data:
				#	disort_dir_ = disort_dir + 'data_%.3f_%.3f.pk' % (g0, w0)
				#	start_case.inputs['output_dir']=disort_dir_
				start_case.approx(single_phase = 'OTHG', rt_method = method, stream = stream, 
						toon_coefficients = toon_coefficients,
						delta_eddington=delta_eddington)

				allout = start_case.spectrum(opa, full_output=True, calculation=calculation, as_dict=True)

				alb = allout['thermal']
				perror.loc[g][w] = np.mean(alb)
	
	perror.index.name = 'asy'
	if output_dir!=None: perror.to_csv(os.path.join(output_dir))

	return perror


def dlugach_test(single_phase = 'OTHG', multi_phase='N=1',
	output_dir = None, rayleigh=True, phase=True, 
	method="Toon", stream=2, opd=0.2,
	toon_coefficients="quadrature", delta_eddington=False):
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
	real_answer = pd.read_csv(os.path.join(__refdata__,'base_cases', 'testing','DLUGACH_TEST.csv'))
	#real_answer = pd.read_csv('new_dlug.csv')
	real_answer = real_answer.set_index('asy')

	perror = real_answer.copy()

	nlevel = 60

	opa = opannection(wave_range=[0.5,1], resample=10, verbose=False)
	start_case=inputs()
	start_case.phase_angle(0) #radians
	start_case.gravity(gravity = 25, gravity_unit=u.Unit('m/(s**2)'))
	start_case.star(opa, 6000,0.0122,4.437) #kelvin, log metal, log cgs
	start_case.atmosphere(df=pd.DataFrame({'pressure':np.logspace(-6,3,nlevel),
	                                    'temperature':np.logspace(-6,3,nlevel)*0+1000 ,
	                                    'H2':np.logspace(-6,3,nlevel)*0+0.99, 
	                                     'H2O':np.logspace(-6,3,nlevel)*0+0.01}))

	start_case.approx(raman='none', rt_method = method, stream = stream, 
				toon_coefficients = toon_coefficients,multi_phase=multi_phase)
	start_case.inputs['approx']['rt_params']['common']['delta_eddington']=delta_eddington

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

			allout = start_case.spectrum(opa, calculation='reflected')

			alb = allout['albedo']
			perror.loc['Ray',w] = alb[-1]#(100*(alb[-1]-real_answer.loc[-1][w])/real_answer.loc[-1][w])#


	start_case.inputs['test_mode']='constant_tau'
	start_case.approx(single_phase = 'OTHG', rt_method = method, stream = stream, 
				toon_coefficients = toon_coefficients,multi_phase=multi_phase)
	start_case.inputs['approx']['rt_params']['common']['delta_eddington']=delta_eddington

	#then test constant tau approx
	if phase:
		for g0 in real_answer.index[1:]:#[6:]:
			for w in real_answer.keys():#[7:]:
				if float(w) ==1.000:
					w0 = 0.999999
				else: 
					w0 = float(w)
				start_case.clouds(df=pd.DataFrame({'opd':opd,#sum([[i]*196 for i in 10**np.linspace(-4, 2, nlevel-1)],[]),
				                                    'w0':np.zeros(196*(nlevel-1)) + w0 ,
				                                    'g0':np.zeros(196*(nlevel-1)) + float(g0)}))
				allout = start_case.spectrum(opa, calculation='reflected')

				alb = allout['albedo']
				perror.loc[g0][w] = alb[-1]#(100*(alb[-1]-real_answer.loc[-1][w])/real_answer.loc[-1][w])#
	
	if output_dir!=None: perror.to_csv(os.path.join(output_dir))
	perror.index.name = 'asy'
	return real_answer, perror

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
		a['approx']['rt_params']['common']['delta_eddington']=True
		a['approx']['rt_params']['common']['raman']='pollack'
		a['approx']['rt_params']['common']['single_phase'] ='TTHG_ray'
		a['test_mode']='rayleigh'
		a['atmosphere']['cloud']['g0'] = 0

		for i in real_answer.index:
			a['atmosphere']['cloud']['w0'] = real_answer['ssa'][i]
			wno, alb = picaso(a)
			real_answer.loc[i,'rayleigh']=alb[-1] 

	if isotropic:
		#constant tau
		a['approx']['rt_params']['common']['delta_eddington'] = True
		a['approx']['rt_params']['common']['raman']='pollack'
		a['approx']['rt_params']['common']['single_phase'] = 'OTHG'
		a['test_mode']='constant_tau'
		a['atmosphere']['cloud']['g0'] = 0

		for i in real_answer.index:
			a['atmosphere']['cloud']['w0'] = real_answer['ssa'][i]
			wno, alb = picaso(a)
			real_answer.loc[i,'0.0']=alb[-1] 

	if asymmetric:
		#constant tau
		a['approx']['rt_params']['common']['delta_eddington']=True
		a['approx']['rt_params']['common']['raman']='pollack'
		a['approx']['rt_params']['common']['single_phase'] = single_phase

		a['test_mode']='constant_tau'

		for g in [0.2,0.4,0.6,0.8]: 
			a['atmosphere']['cloud']['g0'] = g
			for i in real_answer.index:
				a['atmosphere']['cloud']['w0'] = real_answer['omega'][i]
				wno, alb = picaso(a)
				real_answer.loc[i,str(g)]=alb[-1] 
	return real_answer

