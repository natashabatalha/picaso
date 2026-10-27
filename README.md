
![Logo](docs/logo.png)


`picaso` enables the: 1) computation of exoplanet and brown dwarf spectroscopy in transmission, emission or reflected light, 2) 1D climate modeling of brown dwarfs and exoplanets, 3) fitting spectroscopic data to models with grids and retrievals.

- Website: https://natashabatalha.github.io/picaso/index.html 
- Installation: https://natashabatalha.github.io/picaso/installation.html
- Tutorials: https://natashabatalha.github.io/picaso/tutorials.html
- Workshops: https://natashabatalha.github.io/picaso/workshops.html
- Source code: https://github.com/natashabatalha/picaso
- Contributing: https://natashabatalha.github.io/picaso/contribution.html
- Bug reports: https://github.com/natashabatalha/picaso/issues
- Physics help: https://natashabatalha.github.io/picaso_dev 
- What to cite: https://natashabatalha.github.io/picaso/credit.html 
- Upgrading from v4: https://natashabatalha.github.io/picaso/migration_v5.html
- For AI assistants and LLM tools: https://natashabatalha.github.io/picaso/llms.txt (page index with install and usage notes) and https://natashabatalha.github.io/picaso/llms-full.txt (all docs in one file)

## Major Features

- [Reflected Light Spectroscopy of planets](https://natashabatalha.github.io/picaso/notebooks/A_basics/1_GetStarted.html)
- Thermal Emission Spectroscopy of [planets](https://natashabatalha.github.io/picaso/notebooks/A_basics/5_AddingThermalFlux.html) and [brown dwarfs](https://natashabatalha.github.io/picaso/notebooks/A_basics/7_BrownDwarfs.html)
- [Transit Spectroscopy of planets](https://natashabatalha.github.io/picaso/notebooks/A_basics/6_AddingTransitSpectrum.html)
- [Phase Curves of planets](https://natashabatalha.github.io/picaso/notebooks/E_3dmodeling/6_PhaseCurves.html)
- 1D Climate modeling of [brown dwarfs](https://natashabatalha.github.io/picaso/notebooks/D_climate/1_BrownDwarf_PreW.html) and [planets](https://natashabatalha.github.io/picaso/notebooks/D_climate/2_Exoplanet_PreW.html)
- Fitting models to data with [grids](https://natashabatalha.github.io/picaso/notebooks/F_fitdata/1_GridSearch.html) and [retrievals](https://natashabatalha.github.io/picaso/notebooks/J_driver%26retrievals/4_Retrieval_Setup.html)

## Quickstart

<!-- quickstart:start -->
Install PICASO, then download its data with one command. Pick the bundle for what you are modeling:

```bash
pip install picaso
picaso-data setup --bundle default-planet       # exoplanets: reference data, default opacities and PHOENIX stellar grids (~7.3 GB)
picaso-data setup --bundle default-browndwarf   # brown dwarfs: the same without stellar grids (~5 GB)
```

PICASO remembers where the data went, so no environment variables are needed. Run `picaso-data check` to confirm everything is in place.

**Example Transmission spectrum of a hot Jupiter** (needs `default-planet`):

```python
import picaso.justdoit as jdi

opa = jdi.opannection(wave_range=[0.6, 5])                # default opacity database, 0.6-5 um

planet = jdi.inputs()                                     # an exoplanet with a host star
planet.phase_angle(0)
planet.gravity(mass=1, mass_unit=jdi.u.Unit('M_jup'), radius=1.2, radius_unit=jdi.u.Unit('R_jup'))
planet.star(opa, temp=5500, metal=0.0, logg=4.5, radius=1, radius_unit=jdi.u.Unit('R_sun'))  # PHOENIX grid
planet.guillot_pt(Teq=1200)                               # parametric P-T profile (Guillot 2010)
planet.chemeq_visscher_2121(cto_absolute=0.549, log_mh=0) # chemical equilibrium: solar metallicity and C/O
planet.approx(p_reference=10)                             # the radius above is at 10 bar

out = planet.spectrum(opa, calculation='transmission', full_output=True)
wno, depth = jdi.mean_regrid(out['wavenumber'], out['transit_depth'], R=150)  # wavenumber (cm-1), (Rp/Rs)^2

import matplotlib.pyplot as plt
plt.plot(1e4 / wno, depth * 1e6)
plt.xlabel('Wavelength (um)'); plt.ylabel('Transit depth (ppm)'); plt.show()
```

**Example Thermal emission of a brown dwarf** (needs `default-browndwarf`):

```python
import picaso.justdoit as jdi

opa = jdi.opannection(wave_range=[1, 5])

bd = jdi.inputs(calculation='browndwarf')                 # no host star
bd.phase_angle(0)
bd.gravity(gravity=200, gravity_unit=jdi.u.Unit('m/s**2'))
bd.atmosphere(filename=jdi.brown_dwarf_pt(), sep=r'\s+')  # Teff=1270 K profile and chemistry from the reference data

out = bd.spectrum(opa, full_output=True)
fnu = jdi.convert_flux_units(out['wavenumber'], out['thermal'], to_f_unit='erg*cm^(-2)*s^(-1)*Hz^(-1)')
wno, fnu = jdi.mean_regrid(out['wavenumber'], fnu, R=300)

import matplotlib.pyplot as plt
plt.semilogy(1e4 / wno, fnu)
plt.xlabel('Wavelength (um)'); plt.ylabel('Flux (erg/cm2/s/Hz)'); plt.show()
```

**Example of Terrestrial Planet in Reflected Light** (needs `default-planet`):

```python
import numpy as np
import pandas as pd
import picaso.justdoit as jdi

opa = jdi.opannection(wave_range=[0.3, 1])                # reflected light, 0.3-1 um

rocky = jdi.inputs()
rocky.phase_angle(0)                                      # full phase
rocky.gravity(mass=1, mass_unit=jdi.u.Unit('M_earth'), radius=1, radius_unit=jdi.u.Unit('R_earth'))
rocky.star(opa, temp=5800, metal=0.0, logg=4.4, radius=1, radius_unit=jdi.u.Unit('R_sun'),
           semi_major=1, semi_major_unit=jdi.u.Unit('au'))  # Sun-like star at 1 au, for contrast units
nlevel = 40
rocky.atmosphere(df=pd.DataFrame({
    'pressure': np.logspace(-6, 0, nlevel),               # bar: the surface is at 1 bar
    'temperature': np.linspace(200, 288, nlevel),         # simple profile with a 288 K surface
    'N2': np.full(nlevel, 0.98), 'CO2': np.full(nlevel, 0.01), 'H2O': np.full(nlevel, 0.01)}))
rocky.surface_reflect(0.3, opa.wno)                       # surface albedo, and tells PICASO there is a hard surface
rocky.approx(p_reference=1)                               # the radius above is at the surface

out = rocky.spectrum(opa, calculation='reflected')
wno, albedo = jdi.mean_regrid(out['wavenumber'], out['albedo'], R=150)          # geometric albedo
wno, fpfs = jdi.mean_regrid(out['wavenumber'], out['fpfs_reflected'], R=150)    # planet/star contrast

import matplotlib.pyplot as plt
plt.plot(1e4 / wno, fpfs * 1e10)
plt.xlabel('Wavelength (um)'); plt.ylabel('Planet/star contrast (x 1e-10)'); plt.show()
```

The default opacities are broadened by H2/He and do not include O2 or O3. For Earth-like atmospheres, use the rocky-planet opacities (`picaso-data get resampled_opacity photochem-R15000,0.1-250um`, 6 GB) by passing that file to `jdi.opannection(filename_db=...)`, and see the [Modern Earth benchmark](https://natashabatalha.github.io/picaso/notebooks/K_benchmarks/Modern_Earth.html).

`full_output=True` also returns the atmosphere, opacities and contribution functions (`out['full_output']`), which the plotting tools in `picaso.justplotit` use.

**Explore more functionality**

- 1D climate models: download the correlated-k tables with `picaso-data get ck_tables pre-weighted`, then follow the [brown dwarf climate tutorial](https://natashabatalha.github.io/picaso/notebooks/D_climate/1_BrownDwarf_PreW.html)
- Spectra, climate models and retrievals from a TOML file: `pip install "picaso[driver]"` and the [driver tutorial](https://natashabatalha.github.io/picaso/notebooks/J_driver%26retrievals/1_Driver_Tutorial.html)
- Everything `picaso-data` can download: `picaso-data list`
<!-- quickstart:end -->

## Contributing

Contributions are always welcome no matter the code level you are at. We value anything from small typos in our documentations page to larger feature changes. We follow the [all-contributors](https://github.com/all-contributors/all-contributors) specification. This means that contributions of any kind welcome! If you are a new to GitHub and conda environments you can read our instructions here: https://natashabatalha.github.io/picaso/contribution.html

Additionally all contributors agree to adhere to this project's [`code of conduct`](https://github.com/natashabatalha/picaso/blob/master/code_of_conduct.md)

Anyone is free to use the [contributor bot to add their contribution](https://allcontributors.org/docs/en/bot/usage). Simply comment on an issue or pull request with: 

```
@all-contributors please add @<username> for <contributions>
```

## Contributors

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-21-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://natashabatalha.github.io"><img src="https://avatars.githubusercontent.com/u/6554465?v=4?s=100" width="100px;" alt="Natasha Batalha"/><br /><sub><b>Natasha Batalha</b></sub></a><br /><a href="https://github.com/natashabatalha/picaso/commits?author=natashabatalha" title="Code">💻</a> <a href="#mentoring-natashabatalha" title="Mentoring">🧑‍🏫</a> <a href="#maintenance-natashabatalha" title="Maintenance">🚧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Rangertreaty33"><img src="https://avatars.githubusercontent.com/u/106419391?v=4?s=100" width="100px;" alt="Roman St. Gerard"/><br /><sub><b>Roman St. Gerard</b></sub></a><br /><a href="https://github.com/natashabatalha/picaso/commits?author=Rangertreaty33" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.thinkkappi.com"><img src="https://avatars.githubusercontent.com/u/4071244?v=4?s=100" width="100px;" alt="Kappibw"/><br /><sub><b>Kappibw</b></sub></a><br /><a href="https://github.com/natashabatalha/picaso/issues?q=author%3AKappibw" title="Bug reports">🐛</a> <a href="https://github.com/natashabatalha/picaso/commits?author=Kappibw" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/lcmayor"><img src="https://avatars.githubusercontent.com/u/13788703?v=4?s=100" width="100px;" alt="Laura C. Mayorga"/><br /><sub><b>Laura C. Mayorga</b></sub></a><br /><a href="https://github.com/natashabatalha/picaso/issues?q=author%3Alcmayor" title="Bug reports">🐛</a> <a href="https://github.com/natashabatalha/picaso/commits?author=lcmayor" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.linkedin.com/in/ehsan-gharib-nezhad/"><img src="https://avatars.githubusercontent.com/u/22139918?v=4?s=100" width="100px;" alt="Ehsan Gharib-Nezhad"/><br /><sub><b>Ehsan Gharib-Nezhad</b></sub></a><br /><a href="#data-EhsanGharibNezhad" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/caoimherooney11"><img src="https://avatars.githubusercontent.com/u/24413015?v=4?s=100" width="100px;" alt="Caoimhe Rooney"/><br /><sub><b>Caoimhe Rooney</b></sub></a><br /><a href="#research-caoimherooney11" title="Research">🔬</a> <a href="https://github.com/natashabatalha/picaso/commits?author=caoimherooney11" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/sagnickm"><img src="https://avatars.githubusercontent.com/u/55552916?v=4?s=100" width="100px;" alt="Sagnick Mukherjee"/><br /><sub><b>Sagnick Mukherjee</b></sub></a><br /><a href="#research-sagnickm" title="Research">🔬</a> <a href="https://github.com/natashabatalha/picaso/commits?author=sagnickm" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ninarobbins"><img src="https://avatars.githubusercontent.com/u/43654406?v=4?s=100" width="100px;" alt="Nina Robbins Blanch"/><br /><sub><b>Nina Robbins Blanch</b></sub></a><br /><a href="https://github.com/natashabatalha/picaso/commits?author=ninarobbins" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/MartianColonist"><img src="https://avatars.githubusercontent.com/u/22718554?v=4?s=100" width="100px;" alt="Ryan MacDonald"/><br /><sub><b>Ryan MacDonald</b></sub></a><br /><a href="https://github.com/natashabatalha/picaso/issues?q=author%3AMartianColonist" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://epl.carnegiescience.edu/people/peter-gao"><img src="https://avatars.githubusercontent.com/u/20616506?v=4?s=100" width="100px;" alt="Peter Gao"/><br /><sub><b>Peter Gao</b></sub></a><br /><a href="https://github.com/natashabatalha/picaso/issues?q=author%3Asirpetergao" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/nklewis"><img src="https://avatars.githubusercontent.com/u/16822846?v=4?s=100" width="100px;" alt="Nikole Lewis"/><br /><sub><b>Nikole Lewis</b></sub></a><br /><a href="https://github.com/natashabatalha/picaso/issues?q=author%3Anklewis" title="Bug reports">🐛</a> <a href="#mentoring-nklewis" title="Mentoring">🧑‍🏫</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/exoBD"><img src="https://avatars.githubusercontent.com/u/38989139?v=4?s=100" width="100px;" alt="Mark Marley"/><br /><sub><b>Mark Marley</b></sub></a><br /><a href="#mentoring-exobd" title="Mentoring">🧑‍🏫</a> <a href="#ideas-exobd" title="Ideas, Planning, & Feedback">🤔</a> <a href="#research-exobd" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/drtkat"><img src="https://avatars.githubusercontent.com/u/10516244?v=4?s=100" width="100px;" alt="Tiffany Kataria"/><br /><sub><b>Tiffany Kataria</b></sub></a><br /><a href="https://github.com/natashabatalha/picaso/issues?q=author%3Adrtkat" title="Bug reports">🐛</a> <a href="#mentoring-drtkat" title="Mentoring">🧑‍🏫</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jjfplanet"><img src="https://avatars.githubusercontent.com/u/42284609?v=4?s=100" width="100px;" alt="jjfplanet"/><br /><sub><b>jjfplanet</b></sub></a><br /><a href="#ideas-jjfplanet" title="Ideas, Planning, & Feedback">🤔</a> <a href="#mentoring-jjfplanet" title="Mentoring">🧑‍🏫</a> <a href="#fundingFinding-jjfplanet" title="Funding Finding">🔍</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Ziva18t"><img src="https://avatars.githubusercontent.com/u/107640508?v=4?s=100" width="100px;" alt="Ziva18t"/><br /><sub><b>Ziva18t</b></sub></a><br /><a href="https://github.com/natashabatalha/picaso/commits?author=Ziva18t" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://jamesmang.wixsite.com/jamesmang"><img src="https://avatars.githubusercontent.com/u/33335954?v=4?s=100" width="100px;" alt="James"/><br /><sub><b>James</b></sub></a><br /><a href="https://github.com/natashabatalha/picaso/commits?author=James-Mang" title="Code">💻</a> <a href="#research-James-Mang" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/astrocaroline"><img src="https://avatars.githubusercontent.com/u/10456842?v=4?s=100" width="100px;" alt="astrocaroline"/><br /><sub><b>astrocaroline</b></sub></a><br /><a href="#mentoring-astrocaroline" title="Mentoring">🧑‍🏫</a> <a href="#ideas-astrocaroline" title="Ideas, Planning, & Feedback">🤔</a> <a href="#fundingFinding-astrocaroline" title="Funding Finding">🔍</a> <a href="#research-astrocaroline" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/rewaclarkbush"><img src="https://avatars.githubusercontent.com/u/85771931?v=4?s=100" width="100px;" alt="rewaclarkbush"/><br /><sub><b>rewaclarkbush</b></sub></a><br /><a href="https://github.com/natashabatalha/picaso/commits?author=rewaclarkbush" title="Code">💻</a> <a href="https://github.com/natashabatalha/picaso/commits?author=rewaclarkbush" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://nicholaswogan.github.io/"><img src="https://avatars.githubusercontent.com/u/18489667?v=4?s=100" width="100px;" alt="Nick Wogan"/><br /><sub><b>Nick Wogan</b></sub></a><br /><a href="#tutorial-Nicholaswogan" title="Tutorials">✅</a> <a href="https://github.com/natashabatalha/picaso/commits?author=Nicholaswogan" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kelle"><img src="https://avatars.githubusercontent.com/u/1444249?v=4?s=100" width="100px;" alt="Kelle Cruz"/><br /><sub><b>Kelle Cruz</b></sub></a><br /><a href="https://github.com/natashabatalha/picaso/commits?author=kelle" title="Tests">⚠️</a> <a href="https://github.com/natashabatalha/picaso/commits?author=kelle" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/benjaminliberles"><img src="https://avatars.githubusercontent.com/u/89707153?v=4?s=100" width="100px;" alt="Benjamin Liberles"/><br /><sub><b>Benjamin Liberles</b></sub></a><br /><a href="https://github.com/natashabatalha/picaso/commits?author=benjaminliberles" title="Documentation">📖</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->



[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code_of_conduct.md)

