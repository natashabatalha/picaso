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

	
# 	start_case.atmosphere(filename=jdi.jupiter_pt(), sep=r'\s+')

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

	start_case.atmosphere(filename=jdi.jupiter_pt(), sep=r'\s+')

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
	start_case.atmosphere(filename=jdi.jupiter_pt(), exclude_mol='H2O', sep=r'\s+')

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
	
