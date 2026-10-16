# Sphinx configuration for the PICASO documentation.
#
# Tutorials are jupytext percent-format .py files that myst-nb executes at
# build time. Execution needs the full reference data (opacities etc.), so
# science builds happen on a local machine and are pushed to gh-pages with
# `make deploy` (see docs/README.md).
#
# Environment variables:
#   PICASO_DOCS_EXECUTE  cache (default) | force | off
#                        "off" skips notebook execution, used by the CI check.

import json
import os

import picaso.justdoit as jdi

# -- Project information -----------------------------------------------------

project = 'picaso'
author = 'Natasha E. Batalha'
copyright = 'PICASO Dev Team'

# justdoit.__version__ is what users see; editable-install metadata can be stale
release = str(jdi.__version__)
version = release

def _refdata_version():
    try:
        with open(os.path.join(os.environ['picaso_refdata'], 'config.json')) as f:
            return str(json.load(f).get('version', 'unknown'))
    except (KeyError, OSError, ValueError):
        return 'unknown'

refdata_version = _refdata_version()

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'sphinx.ext.githubpages',
    'myst_nb',
    'sphinx_design',
    'sphinx_copybutton',
    'sphinx_llms_txt',
]

templates_path = ['_templates']
master_doc = 'index'
language = 'en'
exclude_patterns = [
    '_build', '_scripts', 'conf.py', 'README.md',
    '**.ipynb_checkpoints', '**/*WIP*', '**/*WIP*ipynb',
]
todo_include_todos = True

# -- Notebooks (myst-nb) -----------------------------------------------------

nb_custom_formats = {
    '.py': ['jupytext.reads', {'fmt': 'py:percent'}],
}
nb_execution_mode = os.environ.get('PICASO_DOCS_EXECUTE', 'cache')
nb_execution_timeout = -1          # climate tutorials can run for a long time
nb_execution_raise_on_error = True # same as the old nbsphinx_allow_errors = False
nb_execution_show_tb = True
nb_merge_streams = True

myst_enable_extensions = [
    'amsmath',
    'colon_fence',
    'deflist',
    'dollarmath',
]
# Tutorials link to other pages by their built .html paths (as nbsphinx
# allowed), so emit markdown links as-is instead of resolving them.
myst_all_links_external = True

# -- HTML output -------------------------------------------------------------

html_theme = 'pydata_sphinx_theme'
html_title = 'PICASO'
html_logo = 'logo.png'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_baseurl = 'https://natashabatalha.github.io/picaso/'
html_show_sourcelink = True
html_sidebars = {'index': []}   # landing page: no left sidebar

html_context = {
    'github_user': 'natashabatalha',
    'github_repo': 'picaso',
    'github_version': 'master',
    'doc_path': 'docs',
    'refdata_version': refdata_version,
}

html_theme_options = {
    'logo': {'alt_text': f'PICASO {release} - Home'},
    'navbar_align': 'left',
    'header_links_before_dropdown': 8,
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/natashabatalha/picaso',
            'icon': 'fa-brands fa-github',
        },
    ],
    'use_edit_page_button': True,
    'secondary_sidebar_items': ['page-toc', 'edit-this-page', 'sourcelink'],
    'footer_start': ['copyright'],
    'footer_end': ['build-info'],
    # switcher.json lives at the root of gh-pages and is rewritten by deploy.sh
    'switcher': {
        'json_url': 'https://natashabatalha.github.io/picaso/switcher.json',
        'version_match': os.environ.get('PICASO_DOCS_VERSION', release),
    },
    'check_switcher': False,
    'show_version_warning_banner': True,
    'navbar_end': ['version-switcher', 'theme-switcher', 'navbar-icon-links'],
}

# -- llms.txt ----------------------------------------------------------------

llms_txt_title = 'PICASO'
# every line goes into the blockquote under the title; keep it short and current,
# since AI assistants read this before (or instead of) the pages listed below it
llms_txt_summary = (
    'PICASO is a Python code for computing exoplanet and brown dwarf spectra '
    '(reflected light, thermal emission, transmission), 1D radiative-convective climate models, '
    'and fitting models to data with grids and retrievals.\n'
    '\n'
    'Install: `pip install picaso`, then download the data with one command: '
    '`picaso-data setup --bundle default-planet` (exoplanets: reference data, default opacities '
    'and PHOENIX stellar grids) or `picaso-data setup --bundle default-browndwarf` (no stellar grids). '
    'PICASO then finds the data without environment variables; `picaso-data check` verifies it '
    'and `picaso-data list` shows optional data (climate correlated-k tables, Sonora grids, other opacities).\n'
    '\n'
    'Usage (`import picaso.justdoit as jdi`): `opa = jdi.opannection(wave_range=[...])` loads opacities; '
    '`jdi.inputs()` (or `jdi.inputs(calculation="browndwarf")`) sets up a case with `.gravity()`, '
    '`.star()`, an atmosphere (`.atmosphere(df=...)`, or `.guillot_pt()` plus `.chemeq_visscher_2121()`), '
    'and `.approx()`; then `.spectrum(opa, calculation="transmission" | "thermal" | "reflected")`. '
    'Copy-and-run examples for a transmission spectrum, brown dwarf emission and a terrestrial planet '
    'in reflected light: https://natashabatalha.github.io/picaso/getting_started.html\n'
    '\n'
    'Upgrading from v4 (e.g. the default stellar grid is now PHOENIX): '
    'https://natashabatalha.github.io/picaso/migration_v5.html\n'
    '\n'
    f'These docs were built with picaso {release} and reference data {refdata_version}.'
)
# getting_started.md pulls the Quickstart from the README with a MyST {include}, which
# sphinx-llms-txt does not expand, so append the README itself to llms-full.txt
llms_txt_code_files = ['+:../README.md']

# -- Setup -------------------------------------------------------------------

def _write_all_pages(app, env):
    # The navbar on every page comes from the root toctree, but Sphinx only
    # rewrites pages whose own source changed. Rewrite all pages each build so
    # navigation never goes stale; notebooks are not re-executed (cached).
    return list(env.found_docs)

def setup(app):
    app.connect('env-get-updated', _write_all_pages)
