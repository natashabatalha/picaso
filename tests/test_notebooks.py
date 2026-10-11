import json
import pytest
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


MINI_OPACITY_DB = os.path.join(__refdata__,'base_cases','retrieval_example','testing_file_only_H2O_CO2_example_opacities.db')
STAR_FILE = os.path.join(__refdata__,'base_cases','retrieval_example','star_5400K.txt')
REFLECTED_BENCHMARK = os.path.join(__refdata__,'base_cases','testing','reflected_1d_mini_db.csv')

def _reflected_case(opacity, h2o=1e-3, exclude_mol=None):
	"""
	Warm (800 K) clear H2/He atmosphere with H2O and CO2, the only molecules in the mini opacity db,
	around a userfile star so that no stellar grids are needed
	"""
	start_case = jdi.inputs()
	start_case.phase_angle(0) #radians
	start_case.gravity(radius=1, radius_unit=jdi.u.Unit('R_jup'),
					mass=1, mass_unit=jdi.u.Unit('M_jup'))
	start_case.star(opacity, radius=1, radius_unit=jdi.u.Unit('R_sun'),
					semi_major=0.05, semi_major_unit=jdi.u.Unit('au'),
					filename=STAR_FILE, w_unit='um', f_unit='erg/(s*cm**2*AA)')
	pressure = np.logspace(-6,2,60)
	start_case.atmosphere(df=pd.DataFrame({'pressure':pressure,
										   'temperature':pressure*0+800,
										   'H2':pressure*0+0.837-h2o,
										   'He':pressure*0+0.163,
										   'H2O':pressure*0+h2o,
										   'CO2':pressure*0+1e-4}),
						  exclude_mol=exclude_mol)
	df = start_case.spectrum(opacity, calculation='reflected')
	return start_case, df

def test_reflected_1d(create_data=False):
	"""
	This tests the functionality of 1D reflected light spectra from notebook #1 'Getting Started'
	using the mini H2O/CO2 opacity db so that it can run in CI.

	Rerun with create_data=True to rewrite the benchmark after an intended change in the output.
	"""
	opacity = jdi.opannection(filename_db=MINI_OPACITY_DB)

	start_case, df = _reflected_case(opacity)
	wno, alb, fpfs = df['wavenumber'], df['albedo'], df['fpfs_reflected']
	wave = 1e4/wno
	water_band = (wave > 1.35) & (wave < 1.5)
	continuum = (wave > 1.0) & (wave < 1.1)

	#physical checks that do not depend on the benchmark
	assert np.all((alb >= 0) & (alb <= 1)), 'albedo outside of 0-1'
	#at full phase the planet-star flux ratio is the geometric albedo times (Rp/a)^2
	rp = start_case.inputs['planet']['radius']
	semi_major = start_case.inputs['star']['semi_major']
	assert np.allclose(fpfs, alb*(rp/semi_major)**2, rtol=1e-10), 'fpfs is not albedo*(Rp/a)^2 at phase 0'
	assert np.mean(alb[water_band]) < 0.1*np.mean(alb[continuum]), 'water band is not dark'

	#less water, or no water opacity at all, brightens the water band
	_, df_low = _reflected_case(opacity, h2o=1e-5)
	_, df_nowater = _reflected_case(opacity, exclude_mol='H2O')
	assert np.mean(df_low['albedo'][water_band]) > np.mean(alb[water_band]), 'less H2O did not brighten water band'
	assert np.mean(df_nowater['albedo'][water_band]) > np.mean(df_low['albedo'][water_band]), 'exclude_mol H2O did not remove water'

	wno_r, alb_r = jdi.mean_regrid(wno, alb, R=150)
	_, fpfs_r = jdi.mean_regrid(wno, fpfs, R=150)
	_, alb_nowater_r = jdi.mean_regrid(df_nowater['wavenumber'], df_nowater['albedo'], R=150)

	if create_data:
		pd.DataFrame({'wavenumber':wno_r, 'albedo':alb_r, 'fpfs':fpfs_r,
					  'albedo_nowater':alb_nowater_r}).to_csv(REFLECTED_BENCHMARK, index=False)

	# compare to benchmark
	assert os.path.isfile(REFLECTED_BENCHMARK), 'Benchmark file not found'
	benchmark_spectrum = pd.read_csv(REFLECTED_BENCHMARK)
	assert np.allclose(benchmark_spectrum['wavenumber'], wno_r), 'Failed wavenumber grid reflected_1d test'
	assert np.allclose(benchmark_spectrum['albedo'], alb_r, rtol=1e-3, atol=1e-8), 'Failed albedo reflected_1d test'
	assert np.allclose(benchmark_spectrum['fpfs'], fpfs_r, rtol=1e-3, atol=1e-14), 'Failed fpfs reflected_1d test'
	assert np.allclose(benchmark_spectrum['albedo_nowater'], alb_nowater_r, rtol=1e-3, atol=1e-8), 'Failed no H2O albedo reflected_1d test'


@pytest.mark.needs_full_data
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
	
def test_citations_and_references():
	from picaso.references import cite, get_citations, all_citations
	import picaso.driver as go

	@cite('10.0000/dummy_doi_1', '10.0000/dummy_doi_2')
	def sample_func():
		pass

	assert get_citations('sample_func') == ('10.0000/dummy_doi_1', '10.0000/dummy_doi_2')
	assert 'sample_func' in all_citations()

	dummy_config = {
		'temperature': {'profile': 'zj23'},
		'chemistry': {'method': 'visscher'},
		'clouds': {'cloud1_type': 'virga'}
	}

	ref_dict = go.references(driver_dict=dummy_config)
	assert isinstance(ref_dict, dict)
	assert 'temperature' in ref_dict
	assert 'chemistry' in ref_dict
	assert 'clouds' in ref_dict

	assert 'pt_zj23' in ref_dict['temperature']
	assert isinstance(ref_dict['temperature']['pt_zj23'], list)
	assert '10.3847/1538-3881/acf768' in ref_dict['temperature']['pt_zj23']

	assert 'chem_visscher' in ref_dict['chemistry']
	assert isinstance(ref_dict['chemistry']['chem_visscher'], list)

	assert 'cloud_virga' in ref_dict['clouds']
	assert isinstance(ref_dict['clouds']['cloud_virga'], list)


def test_it_all(): 
	#reflected 1d 

	#thermal 1d 

	#reflected 3d 

	#thermal 3d 

	#climate tests 

	#transit 1d 
	return 
	
