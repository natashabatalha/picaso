"""
Species names: molecular weights, isotopologues, and telling gas species apart from other columns.

Isotopologues give the mass number before each element, with a dash (or underscore) between the
isotope groups: 12C-18O2, 13C-16O, 1H2-16O, 12C_16O2, and 12C-O2 for one specified isotope.
Deuterium is D (HDO). molecular_weight is what ATMSETUP.get_weights uses, and is_gas_species uses it
to check that every element and isotope exists.
"""
import pytest

from picaso.atmsetup import ATMSETUP, convert_to_simple, is_gas_species, isotope_parts, molecular_weight
from picaso.elements import ELEMENTS as ele


def mass(el, iso):
    return ele[el].isotopes[iso].mass


def test_isotopologue_weights_use_the_given_isotopes():
    assert molecular_weight('12C-18O2') == pytest.approx(mass('C', 12) + 2 * mass('O', 18), rel=1e-12)
    assert molecular_weight('13C-16O') == pytest.approx(mass('C', 13) + mass('O', 16), rel=1e-12)
    assert molecular_weight('1H2-16O') == pytest.approx(2 * mass('H', 1) + mass('O', 16), rel=1e-12)
    assert molecular_weight('2H-1H-16O') == pytest.approx(mass('H', 2) + mass('H', 1) + mass('O', 16), rel=1e-12)


def test_unspecified_elements_use_the_main_isotope():
    assert molecular_weight('12C-O2') == pytest.approx(mass('C', 12) + 2 * mass('O', 16), rel=1e-12)
    assert molecular_weight('H2O') == pytest.approx(2 * mass('H', 1) + mass('O', 16), rel=1e-12)


@pytest.mark.parametrize('dash, underscore', [('12C-16O2', '12C_16O2'), ('13C-18O2', '13C_18O2'), ('12C-O2', 'C_O2')])
def test_dash_and_underscore_are_the_same_isotopologue(dash, underscore):
    assert molecular_weight(dash) == molecular_weight(underscore)
    assert convert_to_simple(dash) == convert_to_simple(underscore) == 'CO2'


def test_ions_weigh_the_same_as_the_neutral():
    assert molecular_weight('H3+') == molecular_weight('H3')
    assert molecular_weight('H2-') == molecular_weight('H2')


def test_get_weights_matches_molecular_weight():
    names = ['H2O', 'HDO', 'KCl', '12C-18O2', 'C_16O2', 'H3+']
    assert ATMSETUP.get_weights(None, names) == {n: molecular_weight(n) for n in names}


@pytest.mark.parametrize('name, parts', [('12C-16O2', ['12C', '16O2']), ('12C_16O2', ['12C', '16O2']),
                                         ('12C-O2', ['12C', 'O2']), ('CO2', ['CO2']), ('H-', ['H-']),
                                         ('H2-', ['H2-']), ('C-gr', ['C-gr'])])
def test_isotope_parts(name, parts):
    assert isotope_parts(name) == parts


@pytest.mark.parametrize('name, simple', [('13C-16O', 'CO'), ('1H2-16O', 'H2O'), ('C_16O2', 'CO2'),
                                          ('HDO', 'HDO'), ('CO2', 'CO2'), ('H-', 'H-')])
def test_convert_to_simple(name, simple):
    assert convert_to_simple(name) == simple


@pytest.mark.parametrize('name', ['H2O', 'CO', 'CO2', 'TiO', 'VO', 'FeH', 'H2', 'He', 'H', 'Na', 'K', 'KCl',
                                  'Cs', 'Rb', 'C2H6', 'LiOH', 'HDO', '1H-16O2', '12C-18O2', '12C-16O2',
                                  '13C-16O', '1H2-16O', '12C-O2', '12C_16O2', 'C_16O2',
                                  'H+', 'H-', 'H2-', 'H3+', 'Li+', 'Fe+', 'e-'])
def test_gas_species_are_recognized(name):
    assert is_gas_species(name)


@pytest.mark.parametrize('name', ['index', 'kz', 'kzz', 'mmw', 'Unnamed: 0', 'C-gr', 'graphite', 'H2O(s)',
                                  'pressure', 'temperature', 'Xy2', '99C-16O', '', 0, None])
def test_other_columns_are_not_gas_species(name):
    assert not is_gas_species(name)
