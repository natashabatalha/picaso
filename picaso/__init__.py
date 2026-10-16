try:
    from pkg_resources import declare_namespace
    declare_namespace(__name__)
except ImportError:
    from pkgutil import extend_path
    __path__ = extend_path(__path__, __name__)

# Find the reference data and stellar grids (environment variables, then the config
# file, then the default data folder) before any module reads them from os.environ.
from . import paths as _paths
_paths.apply()
