# DELETE BEFORE MERGING
# this is only a debugging tool
from datetime import datetime
import pickle

def datetime_now():
    return datetime.now().isoformat().replace(":", ".")

def cache_function_call(f, *args, **kwargs):
    # Generate a unique filename using the function name and current datetime
    filename = f"/Users/adityasengupta/projects/clouds/picaso/data/pkl_inputs/{f.__name__}_{datetime_now()}.pkl"
    with open(filename, 'wb') as file:
        pickle.dump((args, kwargs), file)
    
    result = f(*args, **kwargs)
    with open(filename, 'ab') as file:
        pickle.dump(result, file)
    
    return result

def retrieve_function_call(fname):
    with open(fname, 'rb') as file:
        args, kwargs = pickle.load(file)
        result = pickle.load(file)
    return args, kwargs, result

# %%
