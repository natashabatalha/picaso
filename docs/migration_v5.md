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
