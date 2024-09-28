# %%
import os
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def equal_check(arr1, arr2):
    if isinstance(arr1, np.ndarray):
        return arr1.shape == arr2.shape and np.all(arr1 == arr2)
    else:
        return arr1 == arr2

from picaso.input_pickling import retrieve_function_call
from picaso.justdoit import calculate_atm, t_start, profile, get_fluxes, find_strat

# %%
for subfolder in ["cloudless", "selfconsistent"]:
    pkl_files = [f for f in os.listdir(f"../data/pkl_inputs/{subfolder}") if f.endswith('.pkl')]
    for f in ["profile", "find_strat"]:
        pkl_file = next(iter(filter(lambda x: x.startswith(str(f)), pkl_files)))
        args, kwargs, result = retrieve_function_call(f"../data/pkl_inputs/{subfolder}/{pkl_file}")
        if "verbose" in kwargs:
            kwargs["verbose"] = False
        res_new = eval(f)(*args, **kwargs)
        names = ["pressure", "temp", "dtdp", "conv_flag", "all_profiles", "opd_cld_climate","g0_cld_climate","w0_cld_climate", "cld_out","flux_net_ir_layer", "flux_plus_ir_attop", "all_opd"]
        for (n, rnew, rsaved) in zip(names, res_new, result):
            this_match = False
            if isinstance(rnew, dict):
                this_match = np.all([equal_check(rnew[k], rsaved[k]) for k in rsaved.keys()])
            else:
                this_match = equal_check(rnew, rsaved)
            
            print(f, n, ": ", this_match)
            
        print("\n")
        
        
# %%
subfolder = "selfconsistent"
pkl_files = [f for f in os.listdir(f"../data/pkl_inputs/{subfolder}") if f.endswith('.pkl')]
pkl_file = next(iter(filter(lambda x: x.startswith("profile"), pkl_files)))
args, kwargs, result = retrieve_function_call(f"../data/pkl_inputs/{subfolder}/{pkl_file}")
if "verbose" in kwargs:
    kwargs["verbose"] = False
res_new = profile(*args, **kwargs)
# %%
names = ["pressure", "temp", "dtdp", "conv_flag", "all_profiles", "opd_cld_climate","g0_cld_climate","w0_cld_climate", "cld_out","flux_net_ir_layer", "flux_plus_ir_attop", "all_opd"]
# %%
for (n, rnew, rsaved) in zip(names, res_new, result):
    this_match = False
    if isinstance(rnew, dict):
        this_match = np.all([equal_check(rnew[k], rsaved[k]) for k in rsaved.keys()])
    else:
        this_match = equal_check(rnew, rsaved)
    
    print("profile", n, ": ", this_match)

        
# %%
