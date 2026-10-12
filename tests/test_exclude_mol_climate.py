import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pytest
import astropy.units as u
import matplotlib
matplotlib.use('Agg')

from picaso import justdoit as jdi
from picaso import justplotit as jpi
from picaso.climate import calculate_atm


def _setup_climate_bundle(opa, exclude_mol=None):
	#brown dwarf so that no stellar grids are needed 
	cl = jdi.inputs(calculation="browndwarf", climate=True)
	cl.gravity(gravity=100, gravity_unit=u.Unit('m/(s**2)'))
	cl.effective_temp(1000)
	pt = cl.guillot_pt(1000, nlevel=51, T_int=1000, p_bottom=2, p_top=-6)
	cl.inputs_climate(temp_guess=pt['temperature'].values, pressure=pt['pressure'].values,
					rcb_guess=45)
	cl.atmosphere(mh=10, cto_relative=1, chem_method='visscher', exclude_mol=exclude_mol)
	return cl


@pytest.mark.needs_ck_tables
def test_exclude_mol_climate_opacity():
	"""
	exclude_mol should remove a molecule's opacity in the climate opacity calculation
	(climate.calculate_atm), which is what the climate solver uses at every iteration.
	"""
	opa = jdi.opannection(method='resortrebin')

	cl_all = _setup_climate_bundle(opa)
	dtau_all = calculate_atm(cl_all, opa)[0].DTAU

	cl_noco2 = _setup_climate_bundle(opa, exclude_mol='CO2')
	dtau_noco2 = calculate_atm(cl_noco2, opa)[0].DTAU

	assert not np.allclose(dtau_all, dtau_noco2), 'exclude_mol had no effect on climate opacities'
	#resort-rebin re-sorts the k-distribution so individual gauss points can shift, 
	#but overall removing a gas should decrease the optical depth
	assert np.sum(dtau_noco2) < np.sum(dtau_all), 'excluding CO2 increased optical depth'


def _fake_climate_out(temp_offset, nlevel=11, nwno=5):
	pressure = np.logspace(-6, 2, nlevel)
	wno = np.linspace(100, 10000, nwno)
	return dict(pressure=pressure,
				temperature=np.linspace(800, 2000, nlevel)+temp_offset,
				cvz_locs=[0, nlevel-3, nlevel-2, 0, 0, 0],
				flux_ir_attop=np.ones(nwno)*1e7,
				flux_balance=dict(flux_net_ir=np.linspace(1e8, 1e5, nlevel),
								  flux_net_v=-np.linspace(2e8, 0, nlevel),
								  rfacv=0.5, rfaci=1)), wno


class _FakeOpa:
	def __init__(self, wno):
		self.wno = wno
		self.delta_wno = np.gradient(wno)


def test_flux_analyzer():
	out_a, wno = _fake_climate_out(0)
	out_b, _ = _fake_climate_out(10)
	opa = _FakeOpa(wno)

	#single model, no gravity: flux divergence per dex
	res = jpi.flux_analyzer(out_a, opacityclass=opa, plot=False)
	assert isinstance(res, dict)
	assert len(res['heating_stellar']) == len(out_a['pressure'])-1
	#stellar flux absorbed is everywhere positive (heating) for a monotonic downward flux
	assert np.all(res['heating_stellar'] > 0)
	assert np.all(res['cooling_ir'] < 0)
	assert np.isclose(res['olr'], 5e7)

	#single model figure 
	res, fig = jpi.flux_analyzer(out_a, opacityclass=opa, gravity=4.5*u.Unit('m/s**2'))
	assert len(fig.axes) == 3

	#comparison: list of two diagnostics and a figure with the difference row 
	res, fig = jpi.flux_analyzer(out_a, out_b, opacityclass=opa,
								 gravity=4.5*u.Unit('m/s**2'), labels=('a','b'))
	assert isinstance(res, list) and len(res) == 2
	assert len(fig.axes) == 6
