import os 
import json 
import pandas as pd
import warnings 
import h5py

def read_visscher_2121(filename):
    """
    Explicit function to read and reformat channon's files 
    
    Parameters
    -----------
    filename: chem filename 

    Returns
    -------
    pandas df 
        columns include temperature (Kelvin), pressure (bar), and all molecules (v/v mixing ratio)
    """
    header = pd.read_csv(filename).keys()[0]
    cols = header.replace('T(K)','temperature').replace('P(bar)','pressure').split()
    a = pd.read_csv(filename,sep=r'\s+',skiprows=1,header=None, names=cols)
    a['pressure']=10**a['pressure']
    return a

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

def write_all_profiles(save_all_profiles, all_profiles):
    """
    Saves all T(P) profiles to the file in "save_all_profiles"
    """
    if isinstance(save_all_profiles, str):
        profiles_file = h5py.File(save_all_profiles, 'w')
        gp = profiles_file.create_group("all_profiles")
        num_digits = len(str(len(all_profiles)))
        for (i, profile_to_save) in enumerate(all_profiles):
            i_str = str(i).zfill(num_digits)
            gp.create_dataset(f"pt_{i_str}", data=profile_to_save)
        profiles_file.close()