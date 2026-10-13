import pytest

from picaso.driver import _validate_object_gravity_config


@pytest.mark.parametrize(
    'object_config',
    [
        {'gravity': {'value': 1e5}, 'radius': {'value': 1.0}},
        {'mass': {'value': 20.0}, 'radius': {'value': 1.0}},
        {'gravity': {'value': 1e5}, 'mass': {'value': 20.0}},
    ],
)
def test_two_bulk_properties_are_allowed(object_config):
    _validate_object_gravity_config(object_config)


@pytest.mark.parametrize(
    'object_config',
    [
        {'gravity': {'value': 1e5}},
        # TOML has no null, so a blank value ('') means "not given"
        {'gravity': {'value': ''}, 'mass': {'value': 20.0}, 'radius': {'value': 1.0}},
        {'gravity': {'value': 1e5}, 'mass': {'value': ''}, 'radius': {'value': ''}},
    ],
)
def test_blank_or_missing_bulk_properties_are_allowed(object_config):
    _validate_object_gravity_config(object_config)


@pytest.mark.parametrize(
    'object_config',
    [
        {},
        {'gravity': {'value': ''}, 'mass': {'value': 20.0}, 'radius': {'value': ''}},
        {'radius': {'value': 1.0}},
    ],
)
def test_gravity_without_gravity_or_mass_and_radius_is_rejected(object_config):
    with pytest.raises(ValueError, match='under-specified'):
        _validate_object_gravity_config(object_config)


def test_gravity_mass_and_radius_are_rejected_together():
    object_config = {
        'gravity': {'value': 1e5},
        'mass': {'value': 20.0},
        'radius': {'value': 1.0},
    }

    with pytest.raises(ValueError, match='gravity, mass, and radius are all set'):
        _validate_object_gravity_config(object_config)


def test_mass_and_radius_alone_set_gravity(retrieval_example_config, mini_opacity):
    """A config with no gravity entry at all must still run, with gravity from mass and radius."""
    import copy
    import astropy.constants as const
    import astropy.units as u
    from picaso.driver import setup_spectrum_class

    config = copy.deepcopy(retrieval_example_config)
    assert 'gravity' not in config['object']
    picaso_class = setup_spectrum_class(config, opacity=mini_opacity, param_tools=None, stage='object')

    expected = (const.G * 0.28 * u.M_jup / (1.27 * u.R_jup)**2).to(u.cm / u.s**2).value
    assert picaso_class.inputs['planet']['gravity'] == pytest.approx(expected, rel=1e-6)
