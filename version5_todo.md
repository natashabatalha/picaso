# PICASO v5.0 release plan

Release v5.0 from `gridtree`

## Highlights of v5

1. Driver functionality (TOML-driven runs)
2. Retrieval framework (including the Nautilus sampler)
3. 3D transmission spectroscopy with validation against trident
4. Radiative Transfer 101 models and tutorials
5. Unit tests and integration tests
6. picaso-app for: setting up retrievals, creating spectra, climate models, viewing opacities or sonora grids, analyzing retrievals
7. Easy data downloading (`picaso-data`)
8. Optional enviornment setting for easier setup
9. Modern docs with LLM docs builds
10. Bond albedo function from Nagpal et al. 2026

### Minor but useful changes 

1. big get_thermal_1d speed ups (by eliminating per level calcs when needed)
2. compute_opacity speed ups (by using numba decorators)
3. better energy balance tutorials (and bug fix in how level fluxes were returned for reflected light) and functions to compute radiative heating/cooling. will help diagnose how off RCE retrieved profiels are 
4. 

v5 already changes results relative to v4 (PHOENIX default stellar grid, gravity / Boltzmann constant / CIA unit fixes). Because this is a major version, the other fixes that change results also go in v5. Each one gets an entry in `docs/migration_v5.md`.

---

## 1. Issues to address in v5

### Small 

- [x] **#326** Divide by zero when one net IR flux value is zero. `fnet/fnetir` is now nan where the net IR flux is exactly zero (`tests/test_climate_kzz.py`).
  - [x] Ask @logan-pearce for a reproducer, then close
- [x] **#405** The converged self-consistent Kzz is now the `kz` column of `ptchem_df` whenever a Kzz is computed (clouds, `diseq_chem` or `save_all_kzz`). Also fixed: `save_all_kzz=True` now returns `all_kzz` without `save_all_profiles=True` (`tests/test_climate_kzz.py`).
- [x] **#449** A star from `star(filename=...)` is now stored in the xarray (`stellar_flux` on `stellar_wavelength`) and rebuilt by `input_xarray`; duplicate lines removed. Also fixed: `output_xarray` now keeps planet mass and radius (it used to keep only gravity, so reloaded models lost GM/r² gravity and `fpfs`) (`tests/test_xarray_io.py`). Migration-guide entry added.
- [x] **#443** `get_thermal_1d` (both the spectrum and level-flux paths) and `get_thermal_3d` now report the top level. Toon now matches the exact non-scattering solution at the top (`tests/test_sh_vs_toon.py`); thermal regression snapshot regenerated (mini-db brown dwarf: median change 4e-9, max 1.5e-4; level fluxes unchanged). Migration-guide entry added.
  - [x] Brown dwarf snapshot on the full db (#181): `tests/test_thermal_regression_full_db.py`, clear and fsed=1 cloudy, 1–14 µm. Also checks the #443 change against the old midpoint value: never brighter, bolometric −3e-6 (clear) / −1.2e-5 (cloudy), 0.22% of bins >1%, largest −3.9%.
- [x] **#339** `chem_interp` renormalizes grid gas mixing ratios to sum to 1 (condensates like C-gr and the preweighted tables' `index` column are left out) (they were 0.976–0.990 at 100× solar) (`tests/test_chemistry_normalization.py`). Migration-guide entry added.

### Medium

- [x] **#344** Removed the 32 functions/classes labelled `_deprecate` (none were used) and the `WIP_justplotit` module; the old `.npy` k-tables now raise an error naming the HDF5 download. Kept everything labelled legacy (`compute_opacity_legacy`, `t_start_legacy`, `get_legacy_data_1460`, `get_legacy_data_1060`) and the 1060 code. Migration-guide sections added.
  - [x] `atmosphere_4d(zero_point='night_transit')`: kept. Dropped the "will be deprecated" message; reflected light now gets a real `UserWarning` and uses `secondary_eclipse` (`tests/test_zero_point.py`)
- [ ] **#151 + #175 + #224** Sonora grids, see section 2.

### Under consideration: cloud-hosted opacities (#255)

Only if beta cloud resources come through in time. If not, move it to v5.1.

- [ ] Confirm cloud resources and the timeline
- [ ] Decide the access model: streaming queries against a hosted database vs. on-demand download of wavelength/molecule subsets. The HDF5 move (#403) probably makes chunked remote reads easier than SQLite.
- [ ] Make it usable through `opannection` / `picaso-data` without changing the user's code
- [ ] Document it, and add it to the "easy data downloading" highlight if it ships

### Close or tidy before tagging

- [ ] **#353** Close as wontfix (two people tested it and found no difference).
- [ ] **#355** Move "patchiness between cloud decks" to its own v5.x issue and close this one.
- [ ] **#352** `test_t_start.py` exists. Close the parts that are done and keep only `get_fluxes` / `find_strat` open.
- [ ] **#407** Ask @sirpetergao to retest with v5 damping parameter (PR #417 may have fixed it).
- [ ] **#181 / #447** Move these checklists to a v5.x milestone. Do one item now: **PICASO vs CDISORT thermal comparison** (files already in `reference/base_cases/testing/`). A comparison against an independent code is a strong point for the JOSS review.

### Open PRs

- [ ] **#450** bibtexparser 1.4 → 2.1: see "References tools on bibtexparser 2.1" below. Merge (or close in favour of our own PR) once the port is done.
- [ ] **#279** Fixed clouds: the PR can't be merged as is (see "Fixed clouds in climate" below). Close it with a note pointing to the v5 work, and credit @aditya-sengupta.
- [x] **#360** Optimize opacity reading: closed.
- [ ] **#406** Nimbus integration: merge into v5

### Medium: fixed clouds in climate (from PR #279)

The goal: the user sets a cloud profile at the start of a climate run, and the cloud does not change while the climate calculation iterates (it is not recomputed with virga as the temperature profile changes). PR #279 adds this, but it also moves a lot of code to different files, so the feature can't be separated from that refactor. Re-implement it on `gridtree`, using #279 as the reference.

- [ ] Read #279 and list what the fixed-cloud feature actually needs (inputs, where clouds are updated in `climate.py`, outputs)
- [ ] API: e.g. `inputs_climate(..., cloudy=True)` plus `clouds(df=...)` / `clouds(p=, dp=, opd=, ...)` before `climate()`, with an explicit flag like `fixed_clouds=True` so the cloud update step is skipped
- [ ] Skip the virga update in `profile()` / `find_strat()` when clouds are fixed, and make sure the fixed cloud is re-gridded if the pressure grid changes
- [ ] Return the cloud that was used in the climate output (`cld_df`) so it can be saved with the profile
- [ ] Tests: the cloud optical properties are identical at every iteration; a zero-opacity fixed cloud gives the same profile as a cloud-free run; an opaque fixed deck warms the atmosphere below it
- [ ] Tutorial section in the climate notebooks, and the climate page in picaso-app if it's easy
- [ ] Credit @aditya-sengupta in HISTORY.rst

### Medium: references tools on bibtexparser 2.1 (#450)

bibtexparser 2.x is a rewrite with a different API (`bibtexparser.load` / `BibTexWriter` / `BibDatabase` become `bibtexparser.parse_file` / `parse_string`, `Library`, `write_string`, and entries are `Entry` objects with `fields` instead of dicts). We've done this port before, so there are examples to follow.

- [ ] Collect the earlier port as the reference
- [ ] Port `picaso/references.py` (`cite()` and the bib-writing helpers) to the 2.x API
- [ ] Check every caller of `picaso.references` (`justdoit` imports `cite`) and any notebooks that use it
- [ ] Pin `bibtexparser>=2.0` in `pyproject.toml`
- [ ] Unit test: parsing `reference/` bib data and writing citations gives the same entries as the 1.x version

---

## 2. Sonora grids + Sonora viewer in picaso-app (#151, #175, #224)

### Code

- [ ] Elf Owl entry point in `sonora()` (#175)
- [ ] Diamondback support, including the `.pt` file format and cloud information (#224)
- [ ] `method='nearest_neighbor' | 'interp'` to interpolate to exact Teff / log g (#151)
- [ ] Make each grid downloadable with `picaso-data` (ties into the "easy data downloading" highlight)
- [ ] Unit tests: interpolation at a grid point returns the grid profile; each grid's file format parses
- [ ] Tutorial update (the brown dwarf notebooks)

### Web app: Sonora viewer (Explore section)

New page in the **Explore** group of `picaso/driver_ui/navigation.py`, next to "View opacities". Follow the same pattern as the opacities page (`views/opacities.py`, `core/opacities.py`, `templates/opacities/`).

- [ ] Choose a grid (Bobcat, Elf Owl, Diamondback) from the grids found in the reference data
- [ ] Choose grid parameters (Teff, log g, and M/H, C/O, Kzz, fsed where the grid has them), using nearest-neighbour or interpolation
- [ ] Plot pressure–temperature profiles, overlaying several models to compare them
- [ ] Plot chemistry profiles, and spectra where the grid includes them
- [ ] Show which parts of the grid exist (available parameter combinations)
- [ ] "Use this profile" button that sends the profile into the spectrum or climate page as a P–T input or starting guess
- [ ] If no Sonora data is found, link to the download page
- [ ] Add a GIF of it (section 3, Phase C)

---

## 3. Leave for v5.1 or later

- Convergence: #110, #384, #407
- Memory and GPU: #263 (Nick's chunked branch `big_memory_fix`), #319
- HDF5 migration: #383, #403 (beyond removing deprecated code)
- New physics: #342, #343, #349, #358 (bond albedo; merge if Vighnesh sends a PR), #415
- Cloud-hosted opacities #255, if it doesn't make v5

- [ ] Create a v5.1 milestone and assign these to it

---

## 4. Release phases

### Phase A: Code freeze prep (1–2 weeks)

- [ ] Section 1 fixes on `gridtree`, each with a unit test
- [ ] Sonora grids + viewer (section 2)
- [ ] Fixed clouds in climate, bibtexparser 2.1 port, merge Nimbus (#406)
- [ ] Run the full integration suite locally (`needs_full_data` tests), the notebook tests and `test_readme_quickstart.py`
- [ ] Clean-install check in a new Python 3.12 environment: `pip install "picaso[app]"` → `picaso-data setup --bundle default-planet` → `picaso-app`

### Phase B: Docs (about 1 week)

- [ ] Finish `docs/migration_v5.md`: #443, #339, the deprecated-code removals, the gravity / k_B / CIA fixes, and remove the "still in development" note
- [ ] `5.0` entry in `HISTORY.rst`, organized around the seven highlights, with tutorial links
- [ ] Check that each highlight has a tutorial page: driver, retrievals (incl. Nautilus), 3D transits, Radiative Transfer 101, running the tests, picaso-app, `picaso-data`
- [ ] README: list the highlights, show the hero GIF, link the migration guide

### Phase C: picaso-app GIFs (in parallel with B, a few days)

Script the recordings with Playwright so they can be regenerated with one command when the UI changes. Convert them to GIF with ffmpeg (`palettegen` / `paletteuse`), keep each under ~5 MB at ~1000 px wide, and save them in `docs/_static/gifs/`.

- [ ] Playwright recording script
- [ ] GIF 1, hero (README): home → planets → spectrum. Pull a planet from the NASA Exoplanet Archive, prefill the form, get a spectrum.
- [ ] GIF 2, get the data (refdata): download reference data and stellar grids from inside the app
- [ ] GIF 3, spectrum: switch between transmission / emission / reflected, add a cloud deck, unit conversion
- [ ] GIF 4, climate: live progress bar and convergence log (fast brown dwarf case, sped up)
- [ ] GIF 5, retrieval: upload data, pick free parameters and priors, choose a sampler (Nautilus)
- [ ] GIF 6, analysis: interactive corner plot
- [ ] GIF 7, information content: Jacobians and information statistics
- [ ] GIF 8, Sonora viewer: compare grid profiles, send one into a climate run
- [ ] Hero GIF in the README; the others on their tutorial pages. Static screenshot for the JOSS paper.

### Phase D: JOSS paper (about 2 weeks of writing; submit soon after tagging)

- [ ] Add `paper/paper.md` and `paper/paper.bib`, plus the `openjournals-draft-action` workflow to build a draft PDF on every push
- [ ] Framing: PICASO is already described in Batalha et al. 2019 and Mukherjee et al. 2023. Present v5 as the new software contribution (driver/TOML interface, retrieval framework, 3D transits, picaso-app, data tooling, test suite), not "PICASO"
- [ ] Sections: Summary, Statement of need, State of the field (petitRADTRANS, TauREx, POSEIDON, PLATON, ...), v5 features, acknowledgements, references. About 1000 words and one figure (app screenshot or feature overview)
- [ ] Check the current JOSS submission requirements. They have changed recently and may now require a statement on generative AI use, which applies to this branch.
- [ ] JOSS checklist: OSI license ✓, `docs/contribution.rst` and `code_of_conduct.md` ✓, CI tests ✓, docs ✓, tagged release archived on Zenodo (Phase E)
- [ ] Decide the author list early (Sagnick, Nick, James, Sierra, Francisco, Aditya, Ben, ...) and collect ORCIDs and approvals

### Phase E: Tag and ship

- [ ] Bump `__version__` in `picaso/_version.py` **and** `version` in `reference/config.json` (they must change together)
- [ ] Update the version and date in `CITATION.cff`
- [ ] If `reference/` changed, publish a new reference-data release on Zenodo and point `picaso-data` at it
- [ ] Merge `gridtree` → `dev` → `master`, tag `v5.0.0`, create the GitHub release (Zenodo archives it with a DOI)
- [ ] Publish to PyPI, deploy the docs
- [ ] Submit to JOSS with the release DOI
- [ ] Announce with the hero GIF, linking the migration guide and the app tutorial
