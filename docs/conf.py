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
llms_txt_summary = (
    'PICASO is a Python code for computing exoplanet and brown dwarf spectra '
    '(reflected light, thermal emission, transmission), 1D climate models, '
    'and fitting models to data. Requires the PICASO reference data '
    '(environment variable picaso_refdata) and an opacity database. '
    f'These docs were built with picaso {release} and reference data {refdata_version}.'
)

# -- Setup -------------------------------------------------------------------

def _write_all_pages(app, env):
    # The navbar on every page comes from the root toctree, but Sphinx only
    # rewrites pages whose own source changed. Rewrite all pages each build so
    # navigation never goes stale; notebooks are not re-executed (cached).
    return list(env.found_docs)

def setup(app):
    app.connect('env-get-updated', _write_all_pages)
