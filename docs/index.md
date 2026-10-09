---
html_theme.sidebar_secondary.remove: true
---

# PICASO

```{image} logo.png
:alt: PICASO
:width: 360px
:align: center
:class: landing-logo dark-light
```

```{container} landing-tagline
Spectra, climate, and fits for exoplanets and brown dwarfs.
```

PICASO is an open-source Python code for computing reflected-light, thermal-emission and transmission spectra, running 1D radiative–convective climate models, and fitting models to observations with grids and retrievals.

::::{container} landing-buttons
```{button-ref} getting_started
:ref-type: doc
:color: primary

Get started
```
```{button-ref} tutorials
:ref-type: doc
:color: primary
:outline:

Browse tutorials
```
::::

## What you can do

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} {fas}`sun` Reflected light
:img-bottom: _static/landing/reflected.svg
:img-alt: PICASO reflected spectrum
:class-img-bottom: landing-spectrum dark-light
:link: notebooks/A_basics/1_GetStarted
:link-type: doc

Albedo spectra of planets, with clouds, surfaces and full phase dependence.
:::

:::{grid-item-card} {fas}`fire` Thermal emission
:img-bottom: _static/landing/thermal.svg
:img-alt: PICASO thermal spectrum
:class-img-bottom: landing-spectrum dark-light
:link: notebooks/A_basics/5_AddingThermalFlux
:link-type: doc

Emission spectra of planets and brown dwarfs.
:::

:::{grid-item-card} {fas}`circle-half-stroke` Transmission
:img-bottom: _static/landing/transmission.svg
:img-alt: PICASO transmission spectrum
:class-img-bottom: landing-spectrum dark-light
:link: notebooks/A_basics/6_AddingTransitSpectrum
:link-type: doc

Transit spectra for interpreting JWST and other observations.
:::

:::{grid-item-card} {fas}`temperature-half` 1D climate
:img-bottom: _static/landing/climate.svg
:img-alt: PICASO pressure-temperature profile
:class-img-bottom: landing-spectrum dark-light
:link: notebooks/D_climate/1_BrownDwarf_PreW
:link-type: doc

Radiative–convective equilibrium with clouds, disequilibrium chemistry and photochemistry.
:::

:::{grid-item-card} {fas}`earth-americas` 3D and phase curves
:img-bottom: _static/landing/phases.png
:img-alt: PICASO planet at several phase angles
:class-img-bottom: landing-spectrum dark-light
:link: notebooks/E_3dmodeling/2_3DInputsWithPICASOandXarray
:link-type: doc

Post-process GCM output into 3D spectra and thermal or reflected phase curves.
:::

:::{grid-item-card} {fas}`chart-line` Fit data
:img-bottom: _static/landing/fit.svg
:img-alt: PICASO model fit to spectral data
:class-img-bottom: landing-spectrum dark-light
:link: notebooks/F_fitdata/1_GridSearch
:link-type: doc

Grid searches and Bayesian retrievals against observed spectra.
:::

::::

## Install

::::{tab-set}

:::{tab-item} pip
```bash
pip install picaso
```
:::

:::{tab-item} conda
```bash
conda install conda-forge::picaso
```
:::

:::{tab-item} from source
```bash
git clone https://github.com/natashabatalha/picaso.git
cd picaso
pip install .
```
:::

::::

PICASO also needs its reference data and an opacity file before it can compute a spectrum. The {doc}`Quickstart <notebooks/Quickstart>` downloads them and checks your setup; see {doc}`installation` for manual and optional data.

## Using PICASO in your research

If PICASO contributes to a publication, please see {doc}`credit`. Questions and bug reports are welcome on [GitHub Issues](https://github.com/natashabatalha/picaso/issues).

```{toctree}
:hidden:
:maxdepth: 2

getting_started
tutorials
workshops
howto
theory
picaso
community
```
