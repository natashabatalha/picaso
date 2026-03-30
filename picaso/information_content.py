import numpy as np
import copy 

#picaso imports 
from .driver import *

def _get_dict_value(data, path_string):
    """
    Retrieves the value of a key in a nested dictionary using a dot-separated path string.
    """
    keys = path_string.split('.')
    current_level = data
    for key in keys:
        if isinstance(current_level, dict) and key in current_level:
            current_level = current_level[key]
        else:
            return None
    return current_level

def _find_path_to_key(data, target_key, current_path=""):
    """
    Recursively finds the dot-separated path to a key in a nested dictionary.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            new_path = f"{current_path}.{key}" if current_path else key
            if key == target_key:
                return new_path
            if isinstance(value, dict):
                res = _find_path_to_key(value, target_key, new_path)
                if res:
                    return res
    return None

def _resolve_param_path(config, param_path):
    """
    Resolves a parameter shortcut or path to its full path in the configuration.
    """
    # If it's already a path that exists, use it
    val = _get_dict_value(config, param_path)
    if val is not None:
        # If it's a dict with 'value' key (like object variables), append '.value'
        if isinstance(val, dict) and 'value' in val:
            return param_path + '.value'
        return param_path
    
    # Try common shortcuts
    shortcuts = {
        'mh': 'chemistry.visscher.log_mh',
        'cto': 'chemistry.visscher.cto_absolute',
        'teq': 'object.teq.value'
    }
    if param_path in shortcuts:
        resolved = shortcuts[param_path]
        if _get_dict_value(config, resolved) is not None:
            return resolved
            
    # Fallback: search for the key
    path = _find_path_to_key(config, param_path)
    if path:
        val = _get_dict_value(config, path)
        if isinstance(val, dict) and 'value' in val:
            return path + '.value'
        return path
        
    return param_path

def jacobian(driver_file=None, driver_dict=None, params=None, method='forward', d_param=1e-2, is_log=False):
    """
    Computes a Jacobian matrix for a set of model parameters.

    The Jacobian is defined as the partial derivative of the model output spectrum 
    (reflected light albedo, thermal emission, or transmission transit depth) 
    with respect to the input parameters.

    Parameters
    ----------
    driver_file : str, optional
        Path to the driver.toml configuration file.
    driver_dict : dict, optional
        Configuration dictionary.
    params : list of str, optional
        List of parameter names or dot-separated paths (e.g., 'mh', 'cto', 'teq').
    method : str, optional
        Finite difference method: 'forward', 'backward', or 'center'. Default is 'forward'.
    d_param : float or list of float, optional
        The perturbation amount for each parameter. If a single float, the same amount 
        is used for all parameters. Default is 1e-2.
    is_log : bool or list of bool, optional
        If True, the parameter is perturbed logarithmically (i.e., multiplied/divided by 10^d_param)
        and the derivative is computed with respect to the base-10 logarithm of the parameter.
        Set to False if the parameter in the configuration is already in log units.
        Default is False.

    Returns
    -------
    numpy.ndarray
        A matrix of shape (N_wavelengths, N_parameters) containing the Jacobian.
    """
    if isinstance(driver_file, str):
        with open(driver_file, "rb") as f:
            config = tomllib.load(f)
    elif isinstance(driver_dict, dict):
        config = driver_dict
    else:
        raise Exception('Could not interpret either driver file or dictionary input')

    if params is None:
        raise Exception('Must provide a list of parameters for the Jacobian')

    if isinstance(d_param, (int, float)):
        d_params = [d_param] * len(params)
    else:
        d_params = d_param
        if len(d_params) != len(params):
            raise Exception('d_param must be a single value or have the same length as params')

    if isinstance(is_log, bool):
        is_logs = [is_log] * len(params)
    else:
        is_logs = is_log
        if len(is_logs) != len(params):
            raise Exception('is_log must be a single boolean or have the same length as params')

    # Resolve parameter paths
    resolved_paths = [_resolve_param_path(config, p) for p in params]

    # Check for numeric parameters
    for i, path in enumerate(resolved_paths):
        val = _get_dict_value(config, path)
        if val is None:
            raise Exception(f'Parameter path {path} not found in configuration')
        if not isinstance(val, (int, float, np.int64, np.float64)):
            raise Exception(f'Jacobian can only be computed for numeric parameters. {path} is {type(val)}')
        if is_logs[i] and val <= 0:
            raise Exception(f'Cannot compute logarithmic Jacobian for non-positive parameter: {path}={val}')

    obs_type = config['observation_type']
    observation_key_mapping = {
        'thermal': 'thermal',
        'reflected': 'albedo',
        'transmission': 'transit_depth'
    }
    spec_key = observation_key_mapping.get(obs_type)
    if not spec_key:
        raise Exception(f'Observation type {obs_type} not supported for Jacobian')

    def get_spec(cfg):
        out = run(driver_dict=cfg)
        return out[spec_key]

    jacobian_cols = []
    
    # Base spectrum for forward/backward
    if method in ['forward', 'backward']:
        base_spec = get_spec(config)

    for i, path in enumerate(resolved_paths):
        dp = d_params[i]
        val = _get_dict_value(config, path)
        log_pert = is_logs[i]

        if method == 'forward':
            cfg_plus = copy.deepcopy(config)
            new_val = val * 10**dp if log_pert else val + dp
            set_dict_value(cfg_plus, path, new_val)
            spec_plus = get_spec(cfg_plus)
            deriv = (spec_plus - base_spec) / dp
        elif method == 'backward':
            cfg_minus = copy.deepcopy(config)
            new_val = val / 10**dp if log_pert else val - dp
            set_dict_value(cfg_minus, path, new_val)
            spec_minus = get_spec(cfg_minus)
            deriv = (base_spec - spec_minus) / dp
        elif method == 'center':
            cfg_plus = copy.deepcopy(config)
            new_val_plus = val * 10**dp if log_pert else val + dp
            set_dict_value(cfg_plus, path, new_val_plus)
            spec_plus = get_spec(cfg_plus)
            
            cfg_minus = copy.deepcopy(config)
            new_val_minus = val / 10**dp if log_pert else val - dp
            set_dict_value(cfg_minus, path, new_val_minus)
            spec_minus = get_spec(cfg_minus)
            
            deriv = (spec_plus - spec_minus) / (2 * dp)
        else:
            raise Exception(f"Unknown derivative method: {method}")
        
        jacobian_cols.append(deriv)

    return np.column_stack(jacobian_cols)