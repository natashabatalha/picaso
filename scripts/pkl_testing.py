# %%
import os
import pickle
import numpy as np

from picaso.input_pickling import retrieve_function_call
from picaso.justdoit import calculate_atm, t_start, profile, get_fluxes, find_strat

for subfolder in ["selfconsistent"]:
    pkl_files = [f for f in os.listdir(f"../data/pkl_inputs/{subfolder}") if f.endswith('.pkl')]
    for f in ["calculate_atm", "t_start", "profile", "get_fluxes", "find_strat"]:
        pkl_file = next(iter(filter(lambda x: x.startswith(str(f)), pkl_files)))
        args, kwargs, result = retrieve_function_call(f"../data/pkl_inputs/{subfolder}/{pkl_file}")
        if "verbose" in kwargs:
            kwargs["verbose"] = False
        res_new = eval(f)(*args, **kwargs)
        matches = True
        for (rnew, rsaved) in zip(res_new, result):
            if isinstance(rnew, dict):
                matches = matches and np.all([np.all(rnew[k] == rsaved[k]) for k in rsaved.keys()])
            else:
                matches = matches and np.all(rnew == rsaved)
            
        print(f, matches)
# %%
