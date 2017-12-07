from atmsetup import ATMSETUP
import json 
import pandas as pd
import numpy as np
import picaso as mp
#CASE 1 file path for molecules with weird things that aren't molecules as columns 
a = json.load(open('reference/config.json'))
a['chemistry']['profile']['filepath'] = 'test.profile'
a['planet']['gravity'] = 25
a['planet']['gravity_unit'] = 'm/(s**2)' 
atm = ATMSETUP(a)
atm.get_profile()
atm.mixingratios
atm.get_gravity()
atm.planet.gravity
if len(atm.mixingratios.keys()) ==14:
	print("PASS CASE 1")
else: 
	raise Exception('FAIL CASE 1')

#CASE 2 radius and mass for gravity 

a = json.load(open('reference/config.json'))
a['planet']['gravity'] = None
a['planet']['gravity_unit'] = None 
a['planet']['radius'] = 1 
a['planet']['radius_unit'] = 'R_earth'
a['planet']['mass'] = 1 
a['planet']['mass_unit'] = 'M_earth'
atm1 = ATMSETUP(a)
atm1.get_gravity()
atm1.planet.gravity
if "%.1f" % atm1.planet.gravity == '9.8' : 
	print('PASS CASE 2')
else: 
	raise Exception("FAIL CASE 2")

#CASE 3 subset of molecuels in file 
a = json.load(open('reference/config.json'))
a['chemistry']['profile']['filepath'] = 'test.profile'
a['chemistry']['molecules']['whichones'] = ['H2O', 'H2', 'CH4']
atm = ATMSETUP(a)
atm.get_profile()
if a['chemistry']['molecules']['whichones'] == list(atm.mixingratios.keys()):
	print('PASS CASE 3')
else: 
	raise Exception('FAIL CASE 3')


#Case 4 add profile directly to chemistry 
a = json.load(open('reference/config.json'))
prof = pd.read_csv('test.profile',delim_whitespace=True)
a['chemistry']['profile']['profile'] = prof
a['planet']['gravity'] = 25
a['planet']['gravity_unit'] = 'm/(s**2)' 
atm = ATMSETUP(a)
atm.get_profile()
atm.get_gravity()
atm.profile.keys()
atm.get_mmw()
atm.get_density()
if len(atm.mixingratios.keys()) ==14:
	print("PASS CASE 4")
else: 
	raise Exception('FAIL CASE 4')

#case 5 testing opacity 

a = json.load(open('reference/config.json'))
prof = pd.read_csv('test.profile',delim_whitespace=True)
a['chemistry']['profile']['profile'] = prof
a['planet']['gravity'] = 25
a['planet']['gravity_unit'] = 'm/(s**2)' 
mp.mapdexo(a)
