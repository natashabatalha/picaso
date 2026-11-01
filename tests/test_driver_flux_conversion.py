import numpy as np

from picaso.driver import convert_flux_units


def test_convert_flux_units_preserves_negative_flux_and_uncertainty():
    wavenumber = np.array([10_000.0, 20_000.0, 30_000.0])
    flux = np.array([-2.0e-19, 0.0, 4.0e-19])
    uncertainty = np.array([3.0e-20, 5.0e-20, 7.0e-20])

    converted_flux, converted_uncertainty = convert_flux_units(
        wavenumber,
        flux,
        "erg/(s*cm**2*cm)",
        flux_err=uncertainty,
        xgrid_unit="cm^(-1)",
        f_unit="erg/(s*cm**2*Angstrom)",
    )

    assert converted_flux.value[0] < 0
    assert converted_flux.value[1] == 0
    assert np.all(converted_uncertainty.value > 0)

    conversion_factor = convert_flux_units(
        wavenumber,
        np.ones_like(flux),
        "erg/(s*cm**2*cm)",
        xgrid_unit="cm^(-1)",
        f_unit="erg/(s*cm**2*Angstrom)",
    )
    np.testing.assert_allclose(converted_flux.value, flux * conversion_factor.value)
    np.testing.assert_allclose(
        converted_uncertainty.value,
        uncertainty * conversion_factor.value,
    )
