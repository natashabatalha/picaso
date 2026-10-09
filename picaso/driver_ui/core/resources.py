"""
Expensive PICASO objects (opacity connections, parameterization tools), built
once per process and shared by all sessions.

Only one opacity database is kept in memory, since one can take several GB.
The objects keep per-calculation state, so calculations using them run one at
a time (see `serialized`).
"""
import functools
import json
import os
import threading

from picaso import justdoit as jdi
from picaso.parameterizations import Parameterize

# virga lists these, but their optical properties do not load
_UNLOADABLE_CONDENSATES = ("CaAl12O19", "CaTiO3", "SiO2")


def opacity(optical):
    """Opacity connection for a config's [OpticalProperties] section."""
    kwargs = json.dumps(optical.get("opacity_kwargs", {}), sort_keys=True)
    return _opacity(optical["opacity_file"], optical["opacity_method"], kwargs)


@functools.lru_cache(maxsize=1)
def _opacity(filename, method, kwargs_json):
    return jdi.opannection(filename_db=filename, method=method, **json.loads(kwargs_json))


def param_tools_for(config):
    """Parameterization tools, loaded with the xarray model grid when the temperature profile uses one."""
    mieff_dir = config["OpticalProperties"].get("virga_mieff")
    temperature = config["temperature"]
    if temperature["profile"] == "xarray_grid":
        model_dir = str(temperature["xarray_grid"]["model_dir"])
        if os.path.isdir(model_dir):
            return param_tools(mieff_dir, model_dir)
    return param_tools(mieff_dir)


@functools.lru_cache(maxsize=4)
def param_tools(mieff_dir, model_dir=None):
    condensates = [c for c in jdi.vj.available() if c not in _UNLOADABLE_CONDENSATES]
    return Parameterize(load_cld_optical=condensates, mieff_dir=mieff_dir, model_dir=model_dir)


def grid_parameters(param_tools):
    """{parameter name: unique values} of a loaded xarray model grid."""
    return param_tools.interp_params[param_tools.grid_name]["grid_parameters_unique"]


def possible_molecules(opacity):
    """Molecules available in the opacity database, plus continuum-only species (H2, He, H, H2-, H-)."""
    continuum = {species for pair in opacity.avail_continuum
                 for species in ("H2", "He", "H", "H2-", "H-") if species in pair}
    return sorted(set(opacity.molecules) | continuum)


_lock = threading.RLock()


def serialized(func):
    """Runs `func` while holding the lock on the shared PICASO objects."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _lock:
            return func(*args, **kwargs)
    return wrapper
