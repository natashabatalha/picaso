# Upgrading from v4 to v5

This page lists the changes in PICASO v5 that can change results or break code written for v4, and what to do about each one. Most v4 code runs unchanged.

:::{note}
v5 is still in development, so this list is not complete yet.
:::

## Python 3.11 or newer is required

v4 declared support for Python 3.7 and later. v5 requires Python 3.11, because its dependencies (`virga-exo` 2.0.2 and current `astropy`) need it.

**What to do:** create an environment with Python 3.11 or newer (3.12 is recommended), for example `conda create -n picaso python=3.12`.

## The default stellar grid is now PHOENIX

In v4, `inputs.star(opa, temp, metal, logg)` used the Castelli & Kurucz (2004) `ck04models` grid unless you passed `database`. In v5 the default is `phoenix`, which is also what `picaso-data setup --bundle default-planet` downloads. The same default now applies to `load_planet` (`stellar_db`), to the driver when the TOML file does not set `star.grid.database`, and to `reference/input_tomls/driver.toml`.

**What changes:** spectra that use a stellar grid are computed with a PHOENIX spectrum, so reflected light and contrast (`fpfs_*`) spectra differ slightly from v4 results. Spectra of brown dwarfs, and of stars given with `star(filename=...)`, are unchanged.

**What to do:**

- To reproduce v4 results, pass the old grid explicitly: `star(opa, temp, metal, logg, database='ck04models')`.
- If you only downloaded `ck04models`, calls without `database` now raise a `FileNotFoundError` that lists the grids you have. Either pass `database='ck04models'`, or download PHOENIX (2.3 GB) with `picaso-data setup --bundle default-planet` or `picaso.data.get_data('stellar_grids', 'phoenix')`.

## Importing PICASO without reference data warns instead of failing

In v4, `import picaso.justdoit` raised an exception if `picaso_refdata` did not point to the reference data. In v5 the import succeeds with a warning that explains how to get the data, so that `picaso.data` can download it in the same session. The error now comes from `opannection()` and `inputs()`, as a `FileNotFoundError` with the same instructions.

PICASO also finds the data without environment variables: it looks at `picaso_refdata` and `PYSYN_CDBS` first, then the config file written by `picaso-data setup`, then the default PICASO data folder. Run `picaso-data where` to see what it found.

**What to do:** nothing, unless your code relied on the import failing (for example, a `try: import picaso.justdoit` check for whether the data is installed). Check with `picaso.data.check_environ(return_messages=True)` or `picaso-data check` instead.

## `get_data` no longer asks questions outside a notebook or terminal

In v4, `picaso.data.get_data()` asked for any argument you left out, using `input()`. In a script, a batch job or a pipeline nobody can answer, so it waited forever. In v5 it only asks in a notebook or an interactive terminal. Anywhere else:

- `category_download` and `target_download` are required, and a `ValueError` lists the options if one is missing;
- the data goes to its usual destination unless you pass `final_destination_dir`;
- an unknown category or target raises a `ValueError` that lists the options, in every mode.

**What to do:** in scripts, pass both names, for example `get_data('stellar_grids', 'phoenix')`, or use `picaso.data.setup(bundle=...)` for everything needed to compute spectra. `picaso-data list` shows every category and target.

## Thermal emission is reported at the top of the atmosphere

In v4, the Toon thermal solver (`get_thermal_1d`, `get_thermal_3d`) returned the upward flux at the midpoint of the top layer instead of at the top of the atmosphere. Light leaving the middle of the top layer has not yet crossed its upper half, so wherever the top layer has some optical depth the emission came out too bright, most toward the limb. In v5 the solver reports the top level (issue #443).

**What changes:** thermal spectra (`thermal`, `fpfs_thermal`, brightness temperatures) get slightly darker only where the top layer is not optically thin, which in practice means the cores of the strongest lines. For a brown dwarf on the full opacity database the bolometric flux changes by about 0.001% and the largest change in any wavelength bin is about 5%; most bins do not change. Converged climate profiles are not affected, because climate uses the level fluxes, which have not changed.

**What to do:** nothing. If you compare against v4 spectra, expect small differences in strong line cores. To limit how much the top layer matters in either version, put the top of your pressure grid where the atmosphere is optically thin.

## `output_xarray` keeps the planet mass and radius, and user stellar files

In v4, `output_xarray` saved only the gravity when it was known, and dropped the planet mass and radius. A model read back with `input_xarray` then had constant gravity instead of GM/r², and could not compute `fpfs_*`. It also could not save a model whose star came from `star(filename=...)`: it raised an error about missing stellar parameters (issue #449).

In v5, `output_xarray` saves the mass and radius whenever both are known, as well as the gravity. A star from a user file is saved in the xarray itself, as the `stellar_flux` variable on a `stellar_wavelength` coordinate, so `input_xarray` can rebuild it without the original file.

**What to do:** nothing. xarray files written by v4 still load in v5.

## Grid chemistry mixing ratios sum to 1

The Visscher equilibrium chemistry grids (`chem_method='visscher'`, `'visscher_1060'`, or a custom `chemistry_table`) list only a subset of the species in the equilibrium calculation, and the condensates are left out. In v4 the interpolated mixing ratios were used as they were, so they summed to less than 1: by about 0.02% at solar metallicity and by up to about 2.5% at 100× solar. In v5 they are renormalized to sum to 1 in every level (issue #339). The ratio between any two species does not change. `chem_method='on-the-fly'` was already normalized.

**What changes:** every species is scaled up by the same small factor in each level, most at high metallicity. This shifts spectra and climate profiles slightly at high metallicity, and barely at all near solar.

Only gas-phase species from a chemistry table now reach `inputs['atmosphere']['profile']`. In v4, every column of the table was interpolated and copied into the profile, so the profile also had an `index` column (the row number stored in the preweighted tables) and the graphite condensate (`C-gr`, or `graphite` in the 1060 grid), both treated as gases. They are now left out. A custom `chemistry_table` with other columns that are not mixing ratios (for example `kz` or `mmw`) gets a warning naming them, and they no longer overwrite the profile's own columns.

**What to do:** nothing. To reproduce v4 results exactly, multiply the mixing-ratio columns of `inputs['atmosphere']['profile']` by their v4 sums after calling `atmosphere()`.

## Isotopologue names with dashes are recognized

PICASO names isotopologues by giving the mass number before each element, with a dash between the isotope groups: `12C-16O2`, `13C-18O2`, `1H2-16O`, or `12C-O2` to specify only one isotope. In v4 only the underscore form (`12C_16O2`) was actually parsed. A dash-named column in an atmosphere dataframe was dropped with an "Ignoring … not recognized molecule" warning, and `convert_to_simple('C_16O2')` returned `16` instead of `CO2`.

In v5 both forms work and give the same molecular weight, and `convert_to_simple` returns the parent molecule for both (`CO2`).

**What to do:** nothing. If you had dash-named isotopologue columns that were being ignored, they are now included in the atmosphere, so check that you meant to include them.
