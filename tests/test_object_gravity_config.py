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


def test_gravity_mass_and_radius_are_rejected_together():
    object_config = {
        'gravity': {'value': 1e5},
        'mass': {'value': 20.0},
        'radius': {'value': 1.0},
    }

    with pytest.raises(ValueError, match='gravity, mass, and radius are all set'):
        _validate_object_gravity_config(object_config)
