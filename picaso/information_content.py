import numpy as np
import copy 
from spectres import spectres 
import pandas as pd

#picaso imports 
from .driver import *
from .opacity_factory import create_grid
from .justplotit import mean_regrid 

def _get_dict_value(data, path_string):
    """
    Retrieves the value of a key in a nested dictionary or a 
    column in a DataFrame using a dot-separated path string.
    """
    keys = path_string.split('.')
    current_level = data
    
    for i, key in enumerate(keys):
        # Case 1: Standard dictionary lookup
        if isinstance(current_level, dict) and key in current_level:
            current_level = current_level[key]
            
        # Case 2: Pandas DataFrame lookup
        elif isinstance(current_level, pd.DataFrame) and key in current_level.columns:
            current_level = current_level[key]
            # If this was the last key in the path, return the .values
            if i == len(keys) - 1:
                return current_level.values
            
        # Case 3: Path not found
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

def jacobian(driver_file=None, driver_dict=None, picaso_class = None, params=None, method='forward', d_param=1e-2, is_log=False, 
             opacityclass=None, calculation=None):
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
    picaso_class : picaso.justdoit.inputs
        Standard picaso class. 
        If this is passed then required inputs become opacityclass and calculation (normal picaso .spectrum arguments)
    params : list of str, optional
        List of parameter names or dot-separated paths (e.g., 'log_mh', 'cto', 'teq') or ('chemistry.free.visscher.log_mh').
        Full dot-separated paths are recommended to ensure the correct parameter is being adjusted. 
        Parameter names can only be supplied if using driver_file or driver_dict setup. 
        For picaso_class input must specify dot-separated path considering the .inputs dict structure. 
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
    opacityclass : picaso.justdoit.opannection
        results of justdoit.opannection
    calculation : str 
        Regular calculation params for picaso.justdoit.spectrum call. reflected, thermal, or transmission or a combination thereof e.g. reflected+thermal

    Returns
    -------
    numpy.ndarray
        A matrix of shape (N_wavelengths, N_parameters) containing the Jacobian.
    """
    if isinstance(driver_file, str):
        with open(driver_file, "rb") as f:
            config = tomllib.load(f)
        runtype='driver'
    elif isinstance(driver_dict, dict):
        config = driver_dict
        runtype='driver'
    elif not isinstance(picaso_class,type(None)):
        config = picaso_class.inputs
        runtype='picasoclass'
        if isinstance(opacityclass,type(None)): 
            raise Exception('Opacity Class was not provided but is needed if inputting a picaso class as your starting case')
        if isinstance(calculation,type(None)): 
            raise Exception('calculation was not provided but is needed if inputting a picaso class as your starting case')
        else: 
            config['observation_type']=calculation
    else:
        raise Exception('Could not interpret either driver file or dictionary input or picaso class')

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
        #if not isinstance(val, (int, float, np.int64, np.float64)):
        #    raise Exception(f'Jacobian can only be computed for numeric parameters. {path} is {type(val)}')
        if is_logs[i] and np.any(val <= 0):
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
        if runtype == 'driver':
            out = run(driver_dict=cfg)
        else: 
            picaso_class.inputs = cfg 
            out = picaso_class.spectrum(opacityclass, calculation=cfg['observation_type'])
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

class Analyze():
    def __init__(self, wno_grid, jacobian, error ,new_wno=None, R=None):
        """
        Parameters 
        ----------
        jacobian : numpy.matrix 
            number of molecules by number of parameters
        error : numpy.array or float 
            absolute error on spectrum. could be constant or an input array
            input array must be on same grid as wno_grid. Any binning requests 
            will also bin the error according to spectres.spectres error binning. 
        newx : numpy.array
            new x axis
        R : float 
            resolution to bin the data to
        """
        self.og_wno_grid = wno_grid 
        self.jacobian = jacobian 
        self.error = error 
        self.new_wno = new_wno
        self.R = R
        self.nparams = jacobian.shape[1]
        #rebin the jacobian and error according to user input 
        #if error is singular value it will return a matrix
        self.rebin_jac_error()
    
    def rebin_jac_error(self):
        """
        Rebins the error and jacobian according to R or new_wno
        """
        jacobian = self.jacobian
        nwno = len(self.og_wno_grid)
        error = self.error
        #bin jacobian and error
        if (not isinstance(self.new_wno , type(None))) or (not isinstance(self.R , type(None))): 
            
            K_rebin = []
            for i in range(jacobian.shape[1]):
                newx,y = mean_regrid(self.og_wno_grid, jacobian[:,i], newx=self.new_wno, R=self.R)
                K_rebin += [y]
            jacobian = np.matrix(K_rebin)

            if not isinstance(error,(float,int)):
                assert len(error)==len(self.og_wno_grid), 'Error and wavenumber grid are not on the same axis'
                _,error = spectres(newx, self.og_wno_grid, self.og_wno_grid, spec_errs = error)
            
            nwno = len(newx)
            self.new_wno = newx
        else: 
            jacobian=jacobian.T

        if isinstance(error,(float,int)):
            error = np.zeros(nwno)+error   

        self.error = error 
        self.jacobian = jacobian       

    def degrees_of_freedom_svd(self):
        """
        Degrees of Freedom for Signal (DFS) derived from Singular Value Decomposition 

        Tells you how many independent pieces of information you are actually extracting from each of the jacobian elements. 

        Not dependent on prior information. This DOF derived from SVD is only asking: "How orthogonal are these jacobian vectors in a vacuum?"

        High DFS (~number of jacobian elements): You can perfectly distinguish all jacobian elements

        Low DFS (<half of your jacobian elements): Your resolution or wavelength range is insufficient; spectral information is "blending" together.
        """
        S_epsilon_inv = np.diag(1.0 / self.error**2 )

        self.FIM =  getattr(self, 'FIM', self.jacobian @ S_epsilon_inv @ self.jacobian.T)
        # DFS = trace of the Hat Matrix (simplifying for the 'elbow' search)
        # Using SVD on the FIM to assess rank/independence
        u, s, vh = np.linalg.svd(self.FIM)
        dfs = np.sum(s / (s + 1)) # Normalized info contribution
        
        return dfs 

    def shannon_ic(self,prior):
        """
        Shannon Information Criteria 

        Returns 
        -------
        dict 
            AveragingKernel : Averaging Kernel, 𝐴, tells us which of the parameters have the greatest impact on the retrieval
            DOF : Degrees of Freedom, is the sum of the diagonal elements of A tells us how many independent parameters can be determined from the observations
            H : Total information content, 𝐻, is measured in bits and describes how our state of knowledge has increased due to a measurement, relative to the prior. This quantity is best used to compare and contrast modes.
            constraint_interval : gives predicted 1 sigma constraint interval on each of the parmaeters 
        Note
        ----
        Why is this DOF different from the SVD derived DOF? 
        The SVD is a "Stress Test" for your physics—it tells you if your jacobian vectors are fundamentally too similar to ever tell apart. The Shannon DFS tells you how well a real instrument (depending on the validity of your error input) will perform **relative** to your prior knowledge. 
        Therefore if your prior here is large then your DOF might be very optimistic. 
        """
        assert len(prior)==self.nparams, 'length of prior array does not match number of parameters from jacobian'
        self.prior=prior
        S_prior = np.diag(np.array(prior)**2)
        S_prior_inv = np.linalg.inv(S_prior)
        
        S_epsilon_inv = np.diag(1.0 / self.error**2 )

        #if this has already been computed then grab it 
        self.FIM = self.jacobian @ S_epsilon_inv @ self.jacobian.T

        S_Hat = np.linalg.inv(self.FIM + S_prior_inv ) #nparams x nparamers where the sqrt(diagonal elements) are error on parameters e.g. 
        error_on_params = np.sqrt(np.diag(S_Hat))

        #Gain, 𝐺, describes the sensitivity of the retrieval to the observation (e.g. if 𝐺 =0the measurements contribute no new information
        G = S_Hat @ self.jacobian @ S_epsilon_inv

        # Averaging Kernel, 𝐴, tells us which of the parameters have the greatest impact on the retrieval
        A = G @ self.jacobian.T
        
        #Degrees of Freedom, is the sum of the diagonal elements of A tells us how many independent parameters can be determined from the observations
        A_diag = np.diag(A)
        DOF = np.sum(A_diag)
        sign, H = np.linalg.slogdet(np.linalg.inv(S_Hat) @ S_prior)

        #Total information 
        H = 0.5*H

        return {'AveragingKernel':A_diag, 'DOF':DOF, 'H':H,'constraint_interval':error_on_params}

    def loss_by_wave(self,prior=None):
        """
        Computes information loss over spectral ranges to understand what wavelength space is important
        """
        prior = getattr(self, 'prior', prior)
        if isinstance(prior, type(None)):
            raise Exception('The Analyze() class has not yet received a prior so need to pass one to the ic loss by wave function')
        else: 
            self.prior=prior 
        
        jacobian_og = copy.deepcopy(self.jacobian)
        error_og = copy.deepcopy(self.error)
        wno_og = copy.deepcopy(self.new_wno)
        H_og = self.shannon_ic(prior)

        lossH_all = []
        lossCI_all = []

        for i in range(len(wno_og)):
            #remove single element of jac and error 
            self.jacobian =  np.delete(jacobian_og, i, axis=1)
            self.error = np.delete(error_og, i)
            new_H = self.shannon_ic(prior)

            lossH = - H_og['H'] + new_H['H'] 
            lossCI = [ H_og['constraint_interval'][i] - ival for i,ival in enumerate(new_H['constraint_interval'])]
        
            lossH_all += [lossH]
            lossCI_all += [lossCI]
        
        return lossH_all, lossCI_all



