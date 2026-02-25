# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: pic312
#     language: python
#     name: python3
# ---

# %%
import picaso.driver as go

# %%
test1 = go.run('test_case1.toml')

# %%
#do all the PT options work?
for i in go.pt_options:
    with open('test_case1.toml', "rb") as f:
        config = go.tomllib.load(f)
    config['temperature']['profile']=i
    print(i)
    test1 = go.run(driver_dict =config)

# %%
#do all the chem options work?
for i in go.chem_options:
    with open('test_case1.toml', "rb") as f:
        config = go.tomllib.load(f)
    config['chemistry']['method']=i
    print(i)
    test1 = go.run(driver_dict =config)

# %%
#do all the cloud options work?
for i in go.cloud_options:
    with open('test_case1.toml', "rb") as f:
        config = go.tomllib.load(f)
    config['clouds']['cloud1_type']=i
    print(i)
    test1 = go.run(driver_dict =config)
