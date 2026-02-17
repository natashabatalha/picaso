# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Creating Custom Opacity Database
#
# In this notebook we will take users through how to compute an opacity database file using the full high resolution 1460 opacity grid.
#
# Roxana Lupu has posted a version of this 1460 grid online on Zenodo: https://zenodo.org/record/6600976#.YtcFguzMI6E
#
# Roxana's file at 1 $\mu$m is R\~100k. This makes it suitable for data up to R\~10K at 1 $\mu$m.
#
# Coming soon, we will also be posting original LBL calculations via the MAESTRO collaboration on a MAESTRO Zenodo community and NASA hosted website.
#
# In the meantime, please contact the developers for access to certain raw data.

# %%
import picaso.opacity_factory as opa_fac#
import time
import os

# %% [markdown]
# ## First Define Database Specs (min, max wavelength, R)
#
# Opacities are computed "line-by-line" (LBL) grids that are R~1e6 accross a wide wavelength range (e.g. 0.3-300um). In order to make our calculations computationally feasible, we need to create databases that are a resampled subset of this full calculation.
#
# **CAUTIONS regarding resampling**: when you resample an opacity database you must resample to **at least 100x higher than your expected data**. For example:
#
# My data is at R=100...
# I will need to resample to R=10000
#
# My data is at R=3000...
# I will need to resample to R=300000
#
# If you want a native line-by-line calculations you don't need to resample. In this case you just make sure that `oldR`=`newR`.
#
# The general procedure of the opacity factory is:
#
# 1. Interpolate the given opacity bundle to a common wavelength solution that is comparable R (`oldR` below) to the original LBL calculation
# 2. Resample the opacity accordingly to `newR`. For example, if `oldR`=1e6 and `newR`=1e4, then every 100th point is taken for the final opacity specturm
# 3. Insert that resampled opacity to the databse
#
# ### Things to consider when choosing minw, maxw, and R
#
# Here are some typical file sizes to guide your choosing:
#
# 1. 43G : all_opacities_0.3_15_R50000.db
# 2. 28G : all_opacities_0.3_3_R30000.db
# 3. 12G : all_opacities_0.3_5.3_R10000.db
# 4. 565G : all_opacities_0,3_5,3_R500k.db
# 5. 4.1G : all_opacities_5_14_R10000.db

# %%
#this is where your opacity file should be located if you've set your environments correctly
min_wavelength = 1
max_wavelength = 3
old_R=1e6 #this should be near the value of the LBL calculations
#the routine works by first interpolating the LBL calculations uniformly to this opacity
#then they are resampled to the new resolution requested by the user

#what molecules would you like to include
#Lupu et als set set includes
#C2H2, C2H4, C2H6, CH4, CO, CO2, CrH, Fe,
#FeH, H2, H3+, H2O, H2S, HCN, LiCl,
#LiF, LiH, MgH, N2, NH3, OCS, PH3, SiO,
#TiO, and VO, in addition to alkali metals (Li, Na, K, Rb, Cs)

#let's choose a subset of these for purposes of demonstration (note I usually include
#as many as are available)
molecules_1460 = ['CH4', 'CO2' , 'H2O' ,  'CO'  ,'H2' ,'Na','K']



# %% [markdown]
# There are a three additional opacities files that are used to compute the spectral database:
#
# - Optical CH4: Cool optical CH4 is still missing from state of the art CH4 calculations. [Therefore we have to "stitch in" optical CH4 at cool temperatures from Karkoschka](https://www.sciencedirect.com/science/article/abs/pii/S0019103598959139)
# - Optical O3: similar to CH4, we have to stitch in the famous O3 band needed to compute Earth like planets
# - Continuum induced absorption (e.g. H2-H2, H2-He, H2-N2, H2-CH4)
#
# In addition to these continuum sources, the PICASO opacity factory also adds in:
# - [H2-](https://github.com/natashabatalha/picaso/blob/af8dfef83f507c27d947c93c6d09a8a87c040b98/picaso/opacity_factory.py#L253)
# - [H-bf](https://github.com/natashabatalha/picaso/blob/af8dfef83f507c27d947c93c6d09a8a87c040b98/picaso/opacity_factory.py#L292)
# - [H-ff](https://github.com/natashabatalha/picaso/blob/af8dfef83f507c27d947c93c6d09a8a87c040b98/picaso/opacity_factory.py#L321)

# %%
#additional files, if needed
#these are located in the picaso_refdata folder
dir_extras = os.path.join(os.environ['picaso_refdata'],'opacities')

original_continuum = os.path.join(dir_extras,'CIA_DS_aug_2015.dat')
dir_kark = os.path.join(dir_extras,'KarkCH4TempDependent.csv')
dir_o3 = os.path.join(dir_extras,'O3_visible.txt')


# %% [markdown]
# ## Data Formats
#
# There are only specific file formats that `opacity_factory` will read from. If you would like to be added to the list please contact the developers.
#
# 1. p_1, p_2, p_3 file formats from legacy Richard Freedman calculations
# 2. [.npy file formats from Ehsan Gharib-Nezhad](https://zenodo.org/record/4458189#.Ytc6R-zMLvU)
# 3. Roxana Lupu .txt file formats
#
# ### Directory Assumption Requirement
#
# - folder name needs to be the same as the molecule name (e.g. H2O/ would have all the opacity defined above in the molecule_1460 variable name
#
#
# For example, if each individual opacity file is `p_1`:
#
# `/data/weighted_cxs_1460/`
#
#     |--> H2O/
#     |----|--> p_1
#     |----|--> p_2
#     |----|--> p_3
#     .....
#     |----|--> p_N
#     |--> CH4/
#     |----|--> p_1
#     |----|--> p_2
#     |----|--> p_3
#     .....
#     |----|--> p_N
#     |--> NO2/
#
# ### For alkalis
#
# Historically the alkalis have come in different formats. Possible inputs for `alkali_dir` below:
#
# - For Roxana's Files from Zenodo: set alkalis_dir to `individual_file`, which will use the  alkali name. E.g., will look for "Na" in the "Na" folder
# - For Natasha's processed alkalis file: use `alkali` or point to the directory of the alkali folder
#

# %%
#original deirectory of the 1460 grid
#for Lupu files remember to unzip!!
og_directory ="/data/lupu"
#og_directory='/data/weighted_cxs_1460/'
#alkalis_dir = '/data/weighted_cxs_1460/alkalis'#this is technically the default (a folder called alkalis in og_directory) but if your alkalis are located somewhere else you can specificy the full path and add it below
alkalis_dir = 'individual_file'

# %% [markdown]
# ## Step 1: Build New Database
#
# Note: Do not run if you want to insert to existing database

# %%
opa_fac.build_skeleton('/data/picaso_dbs/test.db')

# %% [markdown]
# ## Step 2: Insert Molecular opacity
#
# ### Option 1: my data is low resolution R<100
#
# If your data is at low resolution then you can proceed with what is below by resamplig the Lupu files to a lower resolution (R=10k)

# %%
newR=10000
#new database name
new_db = f'/data/picaso_dbs/lupu_{min_wavelength}_{max_wavelength}_R{newR}.db'
opa_fac.build_skeleton(new_db)
for molecule in molecules_1460:#molecules_1460:
    start_time = time.time()
    print('Inserting: '+molecule)
    new_waveno_grid = opa_fac.insert_molecular_1460(molecule, min_wavelength, max_wavelength, og_directory, new_db,
                                                    #SEE CHOICE HERE!!!!
                                                    new_R=newR, #new_dwno=new_dwno,
                                                    alkali_dir=alkalis_dir,
            dir_kark_ch4=dir_kark, dir_optical_o3=dir_o3, old_R=old_R) #these two parameters are used to hack in extra cross sections into the db
    print(molecule+ ' inserts finished in :' +str((time.time() - start_time)/60.0)[0:3]+' minutes')

# %% [markdown]
# ### Option 2: my data is at high resolution R=10k
#
# At high resolution you will need to use the Lupu files with their direct interpolated wavelength solution. This will not do any resampling and will just insert their opacity data into the picaso database.

# %%
new_db = f'/data/picaso_dbs/lupu_{min_wavelength}_{max_wavelength}_OG_R.db'
opa_fac.build_skeleton(new_db)
for molecule in molecules_1460:#molecules_1460:
    start_time = time.time()
    print('Inserting: '+molecule)
    new_waveno_grid = opa_fac.insert_molecular_1460(molecule, min_wavelength, max_wavelength, og_directory, new_db,
                                                    #SEE CHOICE HERE!!!!
                                                    insert_direct=True,
                                                    alkali_dir=alkalis_dir,
            dir_kark_ch4=dir_kark, dir_optical_o3=dir_o3) #these two parameters are used to hack in extra cross sections into the db
    print(molecule+ ' inserts finished in :' +str((time.time() - start_time)/60.0)[0:3]+' minutes')

# %% [markdown]
# ## Step 3) Insert Continuum Opacity

# %%
start_time = time.time()
new_waveno_grid = opa_fac.get_molecular(new_db,[molecules_1460[0]],[500],[1])['wavenumber']

opa_fac.restruct_continuum(original_continuum,['wno','H2H2','H2He','H2H','H2CH4','H2N2']
                               ,new_waveno_grid, overwrite=False,
                               new_db = new_db)
print('Continuum inserts finished in :' +str((time.time() - start_time)/60.0)[0:3]+' minutes')
