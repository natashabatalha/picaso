import os 
import json 
import pandas as pd
import warnings 

def read_json(filename, **kwargs):
    """
    read in a JSON format file.  return None if the file is not there.

    Parameters
    ----------
    filename: string
        name of the JSON file to read
    except: bool
        if true, raise exception if file is missing. if false, return empty dict.

    Returns
    -------
    d: python object
        data from JSON file decoded into python object
    """
    try:
        with open(filename, 'r') as f:
            json_data = json.load(f, **kwargs)
    except:
        msg = "Missing JSON file: %s" % filename
        raise Exception(msg)
    d = json_data
    return d


def read_hdf(filename, requires, raise_except=False, **kwargs):
    """
    read in H5 format file and pull out specific table based on specific
    header information in the requires field. return None if file is not there 

    Parameters
    ----------
    filename: str
        name of H5 file to read 
    requires : dict 
        dictionary with keys and values to get correct table 
        e.g. {'wmin':0.3, 'wmax':1, 'R':5000} for opacity grid

    Returns
    -------
    d : python object 
        data from H5 file decoded into python object 
    """
    requires_list = [i+'='+str(requires[i]) for i in requires.keys()]

    hdf_name = pd.read_hdf(filename,'header',where=requires_list, columns=['table'],**kwargs)

    if len(hdf_name['table'].values) > 1:
        msg = "HDF requirements satify two grids. Using %s in %s" % (hdf_name.values[0],filename)
        warnings.warn(msg, UserWarning)
        d = pd.read_hdf(filename, hdf_name['table'].values[0])
    else: 
        d = pd.read_hdf(filename, hdf_name['table'].values[0])
    return d
