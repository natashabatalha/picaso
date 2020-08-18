import picaso.opacity_factory as opa_fac
import time
import os

#Define wavelength region you want 

min_wavelength = 0.3#0.3 #micron 
max_wavelength = 30#30 #micron

#Define resolution. NOTE!!!! This should be ~100x greater than the final planet 
#spectrum you want. For example, for HST data (R=100) you would want R~10,000 models.

#if new_R is set to None it will create a grid with the same grid spacing as the original 1060 grid
new_R = None#10000 

#this is the directory where the original LBL opacities are. 

og_directory ='/Users/nbatalh1/Documents/data/all_opacities/'

molecules=['Na','K','Li','Cs','Rb','CH4','CO','CO2','H2O','H2S','N2O','NH3','O2','O3','PH3','TiO','VO']

new_db = '/Users/nbatalh1/Documents/data/all_opacities_noregrid.db'
original_continuum = 'CIA_DS_aug_2015.dat'

#hack locations
dir_kark = 'KarkCH4TempDependent.csv'
dir_o3 = 'O3_visible.txt'

#two extra files for karkoshka methane and optical ozone. This is 
#just the directory of where they are located 

opa_fac.build_skeleton(new_db)

for molecule in molecules:
	start_time = time.time()
	print('Inserting: '+molecule)
	new_waveno_grid = opa_fac.resample_and_insert_molecular(molecule, min_wavelength, max_wavelength, new_R, 
        	og_directory, new_db,
        	dir_kark_ch4=dir_kark, dir_optical_o3=dir_o3) #these two parameters are used to hack in extra cross sections into the db 
	print(molecule+ ' inserts finished in :' +str((time.time() - start_time)/60.0)[0:3]+' minutes')

start_time = time.time()
#new_waveno_grid = opa_fac.get_molecular(new_db,['H2O'],[500],[1])['wavenumber']
opa_fac.restruct_continuum(original_continuum,['wno','H2H2','H2He','H2H','H2CH4','H2N2']
                               ,new_waveno_grid, overwrite=False,
                               new_db = new_db)
print('Continuum inserts finished in :' +str((time.time() - start_time)/60.0)[0:3]+' minutes')



