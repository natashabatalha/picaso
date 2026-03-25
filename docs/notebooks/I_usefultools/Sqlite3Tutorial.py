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
# # SQLITE Tutorial
#
#
# This tutorial shows you how query the existing `opacity.db` and also shows you how to customize your own `opacity.db`
#
# A lot of this code is embedded in PICASO.

# %%
import sqlite3
import io
import numpy as np
import os
import picaso.justdoit as jdi
opa = jdi.opannection()

# %% [markdown]
# ## Establishing a Connection to a Database

# %%
#this is where your opacity file should be located if you've set your environments correctly
db_filename = opa.db_filename

#these functions are so that you can store your float arrays as bytes to minimize storage
def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

#tell sqlite what to do with an array
sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

# %%
#this will be how we execute commands to grab chunks of data
#this is how you establish a connection to the db
conn = sqlite3.connect(db_filename, detect_types=sqlite3.PARSE_DECLTYPES)
cur = conn.cursor()

#usually you want to close your database right after you open it using
#conn.close()

#for now, we will keep it open for the tutorial

# %% [markdown]
# ## Using `SELECT * FROM` to Query Items
#
# ### Get from header

# %%
#let's start by just grabbing all the info from the header
header = cur.execute('SELECT * FROM header')
cols = [description[0] for description in header.description]
data = cur.fetchall()


# %% [markdown]
# ### Get Continuum Opacity

# %%
#what molecules exist?
cur.execute('SELECT molecule FROM continuum')
print(np.unique(cur.fetchall()))

# %%
#what temperatures exist?
cur.execute('SELECT temperature FROM continuum')
cia_temperatures = np.unique(cur.fetchall())
cia_temperatures[0:10]

# %%
#wavenumber grid from header
cur.execute('SELECT wavenumber_grid FROM header')
wave_grid = cur.fetchone()[0]

# %%
#grab H2H2 at 300 K
cur.execute('SELECT opacity FROM continuum WHERE molecule=? AND temperature=?',('H2H2',300))
data = cur.fetchall()
data

# %%
#grab all opacity at 300 K
cur.execute('SELECT molecule,opacity FROM continuum WHERE temperature=300')
data = cur.fetchall()
data

# %% [markdown]
# ### Get Molecular Opacity
#
# Molecular opacities are on a specific P-T grid so we book keep them by assigning indices to each pair e.g (1: 1e-6 bar, 75 K, 2:1e-6, 80K.. and so on)

# %%
#get the PT grid with the corresponding grid
cur.execute('SELECT ptid, pressure, temperature FROM molecular')
data= cur.fetchall()
pt_pairs = sorted(list(set(data)),key=lambda x: (x[0]) )
pt_pairs[0:10]#example of the first 10 PT pairs

# %%
#what molecules exist?
cur.execute('SELECT molecule FROM molecular')
print(np.unique(cur.fetchall()))

# %%
# grab the opacity at a specific temp and pressure
grab_p = 0.1 # bar
grab_t = 100 # kelvin
import math

#here's a little code to get out the correct pair (so we dont have to worry about getting the exact number right)
ind_pt = [min(pt_pairs, key=lambda c: math.hypot(c[1]- coordinate[0], c[2]-coordinate[1]))[0]
          for coordinate in  zip([grab_p],[grab_t])]

cur.execute("""SELECT molecule,ptid,opacity
            FROM molecular
            WHERE molecule = ?
            AND ptid = ?""",('H2O',ind_pt[0]))
data= cur.fetchall()
data #gives you the molecule, ptid, and the opacity

# %%
grab_moles = ['H2O','CO2']
grab_p = [0.1,1,100] # bar
grab_t = [100,200,700] # kelvin

#here's a little code to get out the correct pair (so we dont have to worry about getting the exact number right)
ind_pt = [min(pt_pairs, key=lambda c: math.hypot(c[1]- coordinate[0], c[2]-coordinate[1]))[0]
          for coordinate in  zip(grab_p,grab_t)]

cur.execute("""SELECT molecule,ptid,opacity
            FROM molecular
            WHERE molecule in {}
            AND ptid in {}""".format(str(tuple(grab_moles)), str(tuple(ind_pt))))
data= cur.fetchall()
data #gives you the molecule, ptid, and the opacity

# %%
#Dont forget to close your connection!!!!
conn.close()

# %% [markdown]
# ## Creating a New Database from Scratch
#
# **Note on molecule names**: Because ``picaso`` uses dict formatting to handle opacities, users can easily swap in different molecules.
#
# For example, if I wanted to include CO2-H2 CIA absorption, I can add ``CO2H2`` to the molecules list below. However, it is only quasi-automated in this regaurd. Please contact natasha.e.batalha@gmail.com if you are adding new CIA to the code.
#
# **Exceptions**: The exceptions to this are non-CIA continuum opacities. Right now, the other sources of continuum enabled are ``H2-``, ``H-bf`` and ``H-ff`` which have odd-ball formatting since they aren't simple two molecules. _Please let me know if you want to see another continuum source added_.
#
# **Careful** with case sensitive molecules like **TiO**, **Na**. Make sure you get these right.

# %%
db_filename = '/data/picaso_dbs/new_fake_opacity.db'
conn = sqlite3.connect(db_filename, detect_types=sqlite3.PARSE_DECLTYPES)
#same story with bytes and arrays
def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

#tell sqlite what to do with an array
sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

cur = conn.cursor()

# %% [markdown]
# ### Build Tables: header, continuum, and molecular tables
#
# It is **VERY** important that these tables are structured the same way. If you think something should be edited, ping natasha.e.batalha@gmail.com

# %%
#header
command="""DROP TABLE IF EXISTS header;
CREATE TABLE header (
    id INTEGER PRIMARY KEY,
    pressure_unit VARCHAR,
    temperature_unit VARCHAR,
    wavenumber_grid array,
    continuum_unit VARCHAR,
    molecular_unit VARCHAR
    );"""

cur.executescript(command)
conn.commit()
#molecular data table, note the existence of PTID which will be very important
command = """DROP TABLE IF EXISTS molecular;
CREATE TABLE molecular (
    id INTEGER PRIMARY KEY,
    ptid INTEGER,
    molecule VARCHAR ,
    pressure FLOAT,
    temperature FLOAT,
    opacity array);"""

cur.executescript(command)
conn.commit()
#continuum data table
command = """DROP TABLE IF EXISTS continuum;
CREATE TABLE continuum (
    id INTEGER PRIMARY KEY,
    molecule VARCHAR ,
    temperature FLOAT,
    opacity array);"""

cur.executescript(command)
conn.commit() #this commits the changes to the database

# %% [markdown]
# ### Insert header info (unit and wave grid info!)
#
# The units **MUST** be the same. The wave grid can be whatever as long as it's consistent between continuum and molecular tables.

# %%
wave_grid = np.linspace(1e4/2, 1e4/0.5, 1000) #fake inverse cm wavenumber grid

cur.execute('INSERT INTO header (pressure_unit, temperature_unit, wavenumber_grid, continuum_unit,molecular_unit) values (?,?,?,?,?)',
            ('bar','kelvin', np.array(wave_grid), 'cm-1 amagat-2', 'cm2/molecule'))
conn.commit()

# %% [markdown]
# ### Insert continuum opacity to database

# %%
cia_temperature_grid = [100,300,500,700]
#insert continuum
for mol in ['H2H2', 'H2He', 'H2H', 'H2CH4', 'H2N2','H2-', 'H-bf', 'H-ff']:
    for T in cia_temperature_grid:
        OPACITY = wave_grid *0 + 1e-33 #INSERT YOUR OPACITY HERE
        cur.execute('INSERT INTO continuum (molecule, temperature, opacity) values (?,?,?)', (mol,float(T), OPACITY))
        conn.commit()

# %% [markdown]
# ### Insert molecular opacity to database
#
# Again, make sure that your molecules are **case-sensitive**: e.g. Sodium should be `Na` not `NA`

# %%
#create a fake PT grid
pts=[]
for temp in [100,200,400]:
    for pres in [0.1, 1, 100]:
        pts += [[temp,pres]]
pts

# %%
#insert molecular
for mol in ['H2O','CO2','CH4']:
    i = 1 #NOTE THIS INDEX HERE IS CRUCIAL! It will be how we quickly locate opacities
    for T,P in pts:
        OPACITY = wave_grid *0 + 1e-33 #INSERT YOUR OPACITY HERE
        cur.execute('INSERT INTO molecular (ptid, molecule, temperature, pressure,opacity) values (?,?,?,?,?)', (i,mol,float(T),float(P), OPACITY))
        conn.commit()
        i+=1

# %%
#ALL DONE!!!
conn.close()
