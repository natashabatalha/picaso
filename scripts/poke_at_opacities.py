# %%
import sys
import os
import warnings
warnings.filterwarnings('ignore')
import picaso.justdoit as jdi
# %%
import picaso.justdoit as jdi
import picaso.justplotit as jpi
import virga.justdoit as vj
import virga.justplotit as cldplt
import astropy.units as u
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import pickle
import xarray
from copy import deepcopy
from bokeh.plotting import show
import h5py

cloud_species = "MgSiO3"

#1 ck tables from roxana
mh = '+000'#'+0.0' #log metallicity
CtoO = '100'#'1.0' # CtoO ratio

ck_db = f"../data/kcoeff_2020/sonora_2020_feh{mh}_co_{CtoO}.data.196"
opacity_ck = jdi.opannection(ck_db=ck_db) # grab your opacities

# %%
