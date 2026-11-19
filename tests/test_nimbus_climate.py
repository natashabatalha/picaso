"""
Nimbus clouds in climate (PR #406): nimbus() on a climate case stages a self-consistent cloud, and
update_clouds() reruns it at each cloud step, the same way it does for virga.

A converged climate run with Nimbus is at least 10x slower than one with virga, so it is too slow
for the tests. These tests replace the Nimbus class with a fake that returns a grey cloud
instantly, so they check the PICASO side (the staged kwargs, the run summary, the call to Nimbus,
and the cloud passed back to the climate solver) without Nimbus, MieNet or reference data.
"""
from collections import namedtuple
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pandas as pd
import pytest

from picaso import climate
from picaso import justdoit as jdi

NLEVEL = 31
NWNO_NIMBUS = 196  # nimbus returns its cloud on the 196-wavenumber eddysed grid
GAS_MMR = {'H2O': 7.54e-4*18/2.34}


class FakeNimbus:
    """Stands in for nimbus.Nimbus: records its inputs and returns a grey cloud between 0.1 and 1 bar."""
    calls = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs

    def set_up_atmosphere(self, **kwargs):
        self.atmosphere = kwargs
        FakeNimbus.calls.append(kwargs)

    def set_up_solver(self):
        pass

    def compute(self, typ='full'):
        return 'ds'

    def picaso_formater(self, ds_prev=None):
        pressure = self.atmosphere['pressure']
        p_layer = np.sqrt(pressure[1:]*pressure[:-1])
        wno = np.linspace(200, 30000, NWNO_NIMBUS)
        rows = [(p, w, 1.0 if 0.1 <= p <= 1 else 0.0, 0.9, 0.8) for p in p_layer for w in wno]
        return pd.DataFrame(rows, columns=['pressure', 'wavenumber', 'opd', 'w0', 'g0'])


@pytest.fixture
def fake_nimbus(monkeypatch):
    FakeNimbus.calls = []
    monkeypatch.setattr(jdi, 'Nimbus', FakeNimbus)
    return FakeNimbus


def brown_dwarf():
    case = jdi.inputs(calculation='browndwarf', climate=True)
    case.gravity(gravity=1000, gravity_unit=u.Unit('m/(s**2)'))
    case.effective_temp(400)
    pt = case.guillot_pt(400, nlevel=NLEVEL, T_int=400, p_bottom=2, p_top=-4)
    case.inputs_climate(temp_guess=pt['temperature'].values, pressure=pt['pressure'].values,
                        rcb_guess=NLEVEL - 8, rfacv=0.0)
    return case


def test_nimbus_not_installed_raises_with_install_instructions(monkeypatch):
    monkeypatch.setattr(jdi, 'Nimbus', None)
    with pytest.raises(ImportError, match=r'picaso\[nimbus\]'):
        brown_dwarf().nimbus(['H2O'], gas_mmr=GAS_MMR)


def test_nimbus_needs_gas_mmr(fake_nimbus):
    with pytest.raises(Exception, match='gas_mmr'):
        brown_dwarf().nimbus(['H2O'])


def test_nimbus_stages_a_self_consistent_climate_cloud(fake_nimbus):
    """On a climate case, nimbus() only stores its inputs: the cloud is computed inside climate()."""
    case = brown_dwarf()
    case.nimbus(['H2O'], mh=1, mmw=2.34, gas_mmr=GAS_MMR, verbose=False)
    climate_inputs = case.inputs['climate']
    assert climate_inputs['cloudy'] is True
    assert climate_inputs['cld_model'] == 'nimbus'
    #exactly the arguments nimbus() takes (it used to also pass an undefined fsed)
    assert climate_inputs['nimbus_kwargs'] == dict(condensates=['H2O'], mh=1, mmw=2.34,
                                                   gas_mmr=GAS_MMR, verbose=False)
    assert climate_inputs['cloud_kwargs'] == dict(patchy_do_holes=False, patchy_fthin_cld=None,
                                                  patchy_fhole=None)
    assert fake_nimbus.calls == []  # no kz yet, so no cloud was computed
    #a Nimbus cloud is self-consistent, not a fixed cloud, and is labelled as Nimbus
    assert not case._fixed_cloud(climate_inputs['cloudy'])
    assert 'clouds: nimbus' in case._climate_summary(False, True, False, 0.0)


def test_virga_climate_cloud_is_still_labelled_virga():
    case = brown_dwarf()
    case.inputs['climate']['cld_model'] = 'virga'
    assert 'clouds: virga' in case._climate_summary(False, True, False, 0.0)


def test_update_clouds_reruns_nimbus(fake_nimbus):
    """One cloud step of the climate solver: Nimbus gets the current T, kz and mmw, and its cloud comes back."""
    case = brown_dwarf()
    case.nimbus(['H2O'], mh=1, mmw=2.34, gas_mmr=GAS_MMR, verbose=False)
    nimbus_kwargs = dict(case.inputs['climate']['nimbus_kwargs'])

    nlayer = NLEVEL - 1
    zeros = np.zeros((nlayer, NWNO_NIMBUS, 4))
    CloudParameters = namedtuple('CloudParameters', ['cloudy', 'model', 'OPD', 'G0', 'W0'])(
        True, 'nimbus', zeros.copy(), zeros.copy(), zeros.copy())
    atmosphere = SimpleNamespace(mmw_layer=np.full(nlayer, 2.3))
    kzz = np.full(NLEVEL, 1e8)

    cld_out, df_cld, taudif, taudif_tol, all_opd, new_params = climate.update_clouds(
        case, CloudParameters, atmosphere, kzz, virga_kwargs={}, nimbus_kwargs=nimbus_kwargs,
        save_profile=True, all_opd=[])

    #Nimbus ran once, on this T(P), with the kz and mean molecular weight of this step
    assert len(fake_nimbus.calls) == 1
    call = fake_nimbus.calls[0]
    np.testing.assert_array_equal(call['kzz'], kzz)
    np.testing.assert_array_equal(call['temperature'], case.inputs['atmosphere']['profile']['temperature'].values)
    assert call['mmw'] == pytest.approx(2.3)
    assert call['species'] == ['H2O'] and call['deep_mmr'] == GAS_MMR

    #the new cloud goes in the first of the four slots the solver averages over
    assert cld_out['opd_per_layer'].shape == (nlayer, NWNO_NIMBUS)
    np.testing.assert_array_equal(new_params.OPD[:, :, 0], cld_out['opd_per_layer'])
    assert np.all(new_params.OPD[:, :, 1:] == 0)
    #the cloud passed back to the solver is the 4-step average: a quarter of the new cloud
    in_cloud = (df_cld['pressure'] >= 0.1) & (df_cld['pressure'] <= 1)
    np.testing.assert_allclose(df_cld.loc[in_cloud, 'opd'], 0.25)
    assert np.all(df_cld.loc[~in_cloud, 'opd'] == 0)
    assert taudif == pytest.approx(0.25)
    assert len(all_opd) == nlayer
