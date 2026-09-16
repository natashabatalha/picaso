"""Regression tests for the astronomical radial-velocity sign convention."""

import numpy as np
import pytest

from picaso.driver import process_model


@pytest.mark.parametrize('velocity', [116.0, -116.0])
def test_process_model_rv_shifts_wavelength_in_physical_direction(velocity):
    wavenumber = np.linspace(2800.0, 3300.0, 50001)
    rest_wavenumber = 3050.0
    flux = np.exp(-0.5 * ((wavenumber - rest_wavenumber) / 0.15) ** 2)
    config = {'object': {'RV': {'value': velocity, 'unit': 'km/s'}}}

    shifted = process_model(wavenumber, flux, config=config)['model'][1]
    peak_wavelength = 1e4 / wavenumber[np.argmax(shifted)]
    expected_wavelength = 1e4 / rest_wavenumber * (1 + velocity / 299792.458)

    assert peak_wavelength == pytest.approx(expected_wavelength, abs=1.2e-5)


def test_process_model_zero_rv_does_not_shift_spectrum():
    wavenumber = np.linspace(2800.0, 3300.0, 101)
    flux = np.sin(wavenumber / 100)

    shifted = process_model(
        wavenumber, flux, config={'object': {'RV': {'value': 0.0, 'unit': 'km/s'}}}
    )['model'][1]

    np.testing.assert_array_equal(shifted, flux)
