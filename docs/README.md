# Building and deploying the PICASO docs

The docs are built with Sphinx ([pydata-sphinx-theme](https://pydata-sphinx-theme.readthedocs.io), [myst-nb](https://myst-nb.readthedocs.io)) and hosted on GitHub Pages from the `gh-pages` branch:
https://natashabatalha.github.io/picaso/

Tutorials in `notebooks/` are jupytext percent-format `.py` files. They run during the build, so a full build needs the complete reference data and opacity database. That is why science builds and deploys happen on a local machine, not in CI.

## One-time setup

```bash
pip install -e ".[docs]"           # from the repo root
export picaso_refdata=/path/to/reference
export PYSYN_CDBS=/path/to/grp/redcat/trds
```

## Everyday commands (run from `docs/`)

| Command | What it does |
|---|---|
| `make html` | Build into `_build/html`. Notebooks run, but only ones whose code changed since the last build (cached in `_build/.jupyter_cache`). |
| `make html-noexec` | Build without running notebooks. Fast check of text and layout. |
| `make html-force` | Rerun every notebook (e.g. after updating the opacity database). |
| `make deploy` | `make html`, then publish to the root of the site ("latest"). |
| `make deploy-release` | Same, plus a frozen copy at `/v<version>/` that appears in the version switcher. Run this once per PICASO release. |
| `make clean-cache` | Delete the notebook execution cache. |

Preview locally with `python -m http.server -d _build/html`.

## How deploying works

`deploy.sh` checks out `gh-pages` into `_build/gh-pages` (a separate git worktree, so your branch is untouched), copies the build in, rewrites `switcher.json` from the `v*/` folders that exist, commits, and pushes. Options:

- `./deploy.sh --no-push` commits locally only; publish later with `git push origin gh-pages`.
- Existing `v*/` folders are never modified by a normal deploy.

The page footer and `llms.txt` record the picaso and reference-data versions used for the build.

## AI-readable output

Each build also writes `llms.txt` (an index of every page) and `llms-full.txt` (all page sources in one file) to the site root, via `sphinx-llms-txt`.

## CI

`.github/workflows/docs.yml` builds the docs on pull requests with `PICASO_DOCS_EXECUTE=off`, using the small reference data in the repo. It catches broken markup and API pages that fail to import, and uploads the HTML as a downloadable artifact. It does not deploy.
