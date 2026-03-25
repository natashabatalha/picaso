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

# %% [markdown]
# # What do I reference?
#
# This notebook shows you how to get the references you need based on:
# - your `full_output` data bundle
# - specified molecules
# - specified methods (e.g. 1d spectra vs climate)

# %%
import picaso.justdoit as jdi
import picaso.references as pref

# %% [markdown]
# Let's set up a quick little model run so that we can see how the reference function works

# %%
opa = jdi.opannection(wave_range=[0.3,1]) #lets just use all defaults
planet=jdi.inputs()
planet.phase_angle(0) #radians
planet.gravity(gravity=25, gravity_unit=jdi.u.Unit('m/(s**2)')) #any astropy units available
planet.star(opa, 5000,0,4.0) #opacity db, pysynphot database, temp, metallicity, logg
planet.atmosphere(filename=jdi.jupiter_pt(), sep=r'\s+')
full_output=planet.spectrum(opa, full_output=True)

# %% [markdown]
# ## Get opacity data references based on model output

# %%
refs = pref.References()
opa_latex, bibdb = refs.get_opa(full_output=full_output['full_output'])

# %%
print(opa_latex)

# %%
bibdb.entries[0:2]

# %% [markdown]
# ### Write to bibtex file

# %%
pref.create_bib(bibdb, 'molecule.bib')

# %% [markdown]
# ## Get opacity data references for certain molecules

# %%
opa_latex, bibdb = refs.get_opa(molecules=['H2O','CO2'])

# %%
print(opa_latex)

# %%
bibdb.entries
