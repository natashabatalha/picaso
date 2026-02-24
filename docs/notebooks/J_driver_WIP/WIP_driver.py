# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: pic312
#     language: python
#     name: python3
# ---

# %%
import picaso.driver as go
import os


# %%
#two functions to directly pull the master config and prune it of options that are not needed for the driver notebook. This is to avoid confusion with the large number of options in the master config that are not relevant to the driver notebook.
def prune_dict_by_key(obj, sequence):
    # 1. Identify keys at the current level that contain the sequence
    keys_to_delete = [k for k in obj.keys() if sequence in str(k)]
    
    # 2. Delete those keys
    for k in keys_to_delete:
        del obj[k]
    
    # 3. Recurse into remaining dictionary values
    for v in obj.values():
        if isinstance(v, dict):
            prune_dict_by_key(v, sequence)
            
    return obj

def prep_master_config():
    master_driver = os.path.join(os.getenv('picaso_refdata'), 'input_tomls', 'driver.toml')
    with open(master_driver, "rb") as f:
        config = go.tomllib.load(f)
    config = prune_dict_by_key(config,'_options')
    config['OpticalProperties']['opacity_file']=config['OpticalProperties']['opacity_file'].replace('_default_',os.getenv('picaso_refdata'))
    config['OpticalProperties']['virga_mieff']=config['OpticalProperties']['virga_mieff'].replace('_default_',os.getenv('picaso_refdata'))
    config['temperature']['userfile']['filename'] =config['temperature']['userfile']['filename'].replace('_default_',os.getenv('picaso_refdata'))
    config['temperature']['sonora_bobcat']['sonora_path'] =config['temperature']['sonora_bobcat']['sonora_path'].replace('_default_',os.getenv('picaso_refdata'))
    config['chemistry']['userfile']['filename'] =config['chemistry']['userfile']['filename'].replace('_default_',os.getenv('picaso_refdata'))
    config['clouds']['cloud1']['userfile']['filename'] =config['clouds']['cloud1']['userfile']['filename'].replace('_default_',os.getenv('picaso_refdata'))

    return config


# %%
#do all the PT options work?
for i in go.pt_options:
    config = prep_master_config()
    config['temperature']['PT_method'] = i
    test1 = go.run(driver_dict =config)

# %%
#do all the chem options work?
for i in go.chem_options:
    config = prep_master_config()
    config['chemistry']['method']=i
    print(i)
    test1 = go.run(driver_dict =config)

# %%
#do all the cloud options work?
for i in go.cloud_options:#running everything except userfile since the pressures won't exaclty line up with this flex pressure option
    config = prep_master_config()
    config['clouds']['cloud1_type']=i
    print(i)
    test1 = go.run(driver_dict =config)

# %%
