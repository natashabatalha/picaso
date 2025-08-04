import pooch
import os
import shutil
import glob
import json
import tempfile

def get_data_config():
    __refdata__ = os.environ.get('picaso_refdata','')
    if __refdata__=='':
        print('It does not look like have created picaso_refdata environment variable yet. Please do so by following these instructions first: https://natashabatalha.github.io/picaso/installation.html#create-environment-variable')
        print('After you have created the environment variable I can help download the file and put it in the right place')
        raise Exception('Cannot get data config until picaso_refdata environment is set')
    elif not os.path.isdir(__refdata__):
        print('It looks like you have set picaso_refdata path as: ',__refdata__)
        print('However, this is not a real directory. You might have a typo and should make sure it is the correct path.')  
        print('After you have created the environment variable I can help download the file and put it in the right place')
        raise Exception('Cannot get data config until picaso_refdata environment is created')
    
    inputs = json.load(open(os.path.join(__refdata__,'config.json')))   

    __stellar_refdata__ = os.environ.get('PYSYN_CDBS','$PYSYN_CDBS')
    if __stellar_refdata__ =='$PYSYN_CDBS':print("Stellar environment variable PYSYN_CDBS not yet set, which may impact exoplanet functionality")

    return inputs,{
    "resampled_opacity":{
        'default':{
           'url':{'opacities_0.3_15_R15000.db.tar.gz':'https://zenodo.org/records/14861730/files/opacities_0.3_15_R15000.db.tar.gz'},
            'description':'7.34 GB file resampled at R=15,000 from 0.3-15um. This is sufficient for doing R=100 JWST calculations and serves as a good default opacity database for exploration.',
            'default_destination':os.path.join(__refdata__, inputs['opacities']['files']['resampled-default']) 
        },
        'R60000,0.6-6um':{
            'url':{'all_opacities_0.6_6_R60000.db.tar.gz':'https://zenodo.org/records/6928501/files/all_opacities_0.6_6_R60000.db.tar.gz'},
            'description':'38.3 GB file resampled at R=60,000 from 0.6-6um. This is sufficient for doing moderate resolution JWST calculations.',
            'default_destimation':inputs['opacities']['files']['resampled']
            },
        'R20000,4.8-15um':{
            'url':{'all_opacities_4.8_15_R20000.db.tar.gz':'https://zenodo.org/records/6928501/files/all_opacities_4.8_15_R20000.db.tar.gz'},
            'description':'7.0 GB file resampled at R=20,000 from 4.8-15um. This is sufficient for doing low resolution JWST calculations.',
            'default_destimation':inputs['opacities']['files']['resampled']
            }
        },
    'stellar_grids':{
        'phoenix':{
            'url':{'synphot5.tar.gz':'http://ssb.stsci.edu/trds/tarfiles/synphot5.tar.gz'},
            'description':'Phoenix stellar atlas',
            'default_destination':os.path.join(__stellar_refdata__, 'grid')
            },
        'ck04models':{
            'url':{'synphot3.tar.gz':'http://ssb.stsci.edu/trds/tarfiles/synphot3.tar.gz'},
            'description':'Castelli & Kurucz (2004) stellar atlas',
            'default_destination':os.path.join(__stellar_refdata__, 'grid')
            },
        },
    'virga_mieff':{
        'default':{
            'url':{'virga.zip':'https://zenodo.org/records/5179187/files/virga.zip'},
            'description':'Virga refractive index and Mie files on standard 196 grid',
            'default_destination':os.path.join(__refdata__, 'virga')
            }
        },
    'sonora_grids':{
        'elfowl-Ytype-v2':{
            'url':{'output_275.0_325.0.tar.gz': 'https://zenodo.org/records/15150865/files/output_275.0_325.0.tar.gz',
                    'output_350.0_400.0.tar.gz':'https://zenodo.org/records/15150865/files/output_350.0_400.0.tar.gz',
                    'output_500.0_550.0.tar.gz':'https://zenodo.org/records/15150865/files/output_500.0_550.0.tar.gz',
                    'output_425.0_475.0.tar.gz':'https://zenodo.org/records/15150865/files/output_425.0_475.0.tar.gz'},
            'description':'The models between Teff of 275 to 550 K (applicable to Y-type objects). V2 corrects quenched CO2. Total: ~40 Gb.',
            'default_destination':os.path.join(__refdata__, 'sonora_grids', 'elfowl')
            },
        'elfowl-Ttype-v2':{
            'url':{'output_700.0_800.tar.gz':'https://zenodo.org/records/15150874/files/output_700.0_800.tar.gz',
                    'output_850.0_950.tar.gz':'https://zenodo.org/records/15150874/files/output_850.0_950.tar.gz',
                    'output_1000.0_1200.tar.gz':'https://zenodo.org/records/15150874/files/output_1000.0_1200.tar.gz',
                    'output_575.0_650.tar.gz':'https://zenodo.org/records/15150874/files/output_575.0_650.tar.gz'},
            'description':'The models for Teff between 575 to 1200 K (applicable for T-type objects). V2 corrects quenched CO2. Total: ~40 Gb.',
            'default_destination':os.path.join(__refdata__, 'sonora_grids', 'elfowl')
            },
        'elfowl-Ltype-v2':{
            'url':{'output_1600.0_1800.tar.gz':'https://zenodo.org/records/15150881/files/output_1600.0_1800.tar.gz',
                    'output_1900.0_2100.tar.gz':'https://zenodo.org/records/15150881/files/output_1900.0_2100.tar.gz',
                    'output_2200.0_2400.tar.gz':'https://zenodo.org/records/15150881/files/output_2200.0_2400.tar.gz',
                    'output_1300.0_1400.tar.gz':'https://zenodo.org/records/15150881/files/output_1300.0_1400.tar.gz'
                },
            'description':'Models for Teff between 1300 to 2400 K (applicable for L-type objects).  V2 corrects quenched CO2.Total: ~40 Gb.',
            'default_destination':os.path.join(__refdata__, 'sonora_grids', 'elfowl')
            },
        'elfowl-Ytype-v1':{
            'url':{'output_275.0_325.0.tar.gz': 'https://zenodo.org/records/10381250/files/output_275.0_325.0.tar.gz',
                    'output_350.0_400.0.tar.gz':'https://zenodo.org/records/10381250/files/output_350.0_400.0.tar.gz',
                    'output_500.0_550.0.tar.gz':'https://zenodo.org/records/10381250/files/output_500.0_550.0.tar.gz',
                    'output_425.0_475.0.tar.gz':'https://zenodo.org/records/10381250/files/output_425.0_475.0.tar.gz'},
            'description':'The models between Teff of 275 to 550 K (applicable to Y-type objects). Total: ~40 Gb.',
            'default_destination':os.path.join(__refdata__, 'sonora_grids', 'elfowl')
            },
        'elfowl-Ttype-v1':{
            'url':{'output_700.0_800.tar.gz':'https://zenodo.org/records/10385821/files/output_700.0_800.tar.gz',
                    'output_850.0_950.tar.gz':'https://zenodo.org/records/10385821/files/output_850.0_950.tar.gz',
                    'output_1000.0_1200.tar.gz':'https://zenodo.org/records/10385821/files/output_1000.0_1200.tar.gz',
                    'output_575.0_650.tar.gz':'https://zenodo.org/records/10385821/files/output_575.0_650.tar.gz'},
            'description':'The models for Teff between 575 to 1200 K (applicable for T-type objects). Total: ~40 Gb.',
            'default_destination':os.path.join(__refdata__, 'sonora_grids', 'elfowl')
            },
        'elfowl-Ltype-v1':{
            'url':{'output_1600.0_1800.tar.gz':'https://zenodo.org/records/10385987/files/output_1600.0_1800.tar.gz',
                    'output_1900.0_2100.tar.gz':'https://zenodo.org/records/10385987/files/output_1900.0_2100.tar.gz',
                    'output_2200.0_2400.tar.gz':'https://zenodo.org/records/10385987/files/output_2200.0_2400.tar.gz',
                    'output_1300.0_1400.tar.gz':'https://zenodo.org/records/10385987/files/output_1300.0_1400.tar.gz'
                },
            'description':'Models for Teff between 1300 to 2400 K (applicable for L-type objects). Total: ~40 Gb.',
            'default_destination':os.path.join(__refdata__, 'sonora_grids', 'elfowl')
            },
        'bobcat':{
            'url':{'profile.tar':'https://zenodo.org/records/1309035/files/profile.tar'},
            'description':'Sonora bobcat pressure-temperature profiles',
            'default_destination':os.path.join(__refdata__, 'sonora_grids', 'bobcat')

            },
        'diamondback':{
            'url':'',
            'description':'',
            },
        },
    'ck_tables':{
        'by-molecule':{
            'url':{
                'H2S_1460.npy':'https://zenodo.org/records/10895826/files/H2S_1460.npy',
                'MgH_1460.npy':'https://zenodo.org/records/10895826/files/MgH_1460.npy',
                'O2_1460.npy':'https://zenodo.org/records/10895826/files/O2_1460.npy',
                'FeH_1460.npy':'https://zenodo.org/records/10895826/files/FeH_1460.npy',
                'TiO_1460.npy':'https://zenodo.org/records/10895826/files/TiO_1460.npy',
                'SO2_1460.npy':'https://zenodo.org/records/10895826/files/SO2_1460.npy',
                'Fe_1460.npy':'https://zenodo.org/records/10895826/files/Fe_1460.npy',
                'C2H4_1460.np':'https://zenodo.org/records/10895826/files/C2H4_1460.npy',
                'OCS_1460.npy':'https://zenodo.org/records/10895826/files/OCS_1460.npy',
                'C2H6_1460.npy':'https://zenodo.org/records/10895826/files/C2H6_1460.npy',
                'SiO_1460.np':'https://zenodo.org/records/10895826/files/SiO_1460.npy',
                'C2H2_1460.npy':'https://zenodo.org/records/10895826/files/C2H2_1460.npy',
                'LiCl_1460.npy':'https://zenodo.org/records/10895826/files/LiCl_1460.npy',
                'CO2_1460.npy':'https://zenodo.org/records/10895826/files/CO2_1460.npy',
                'CrH_1460.npy':'https://zenodo.org/records/10895826/files/CrH_1460.npy',
                'Na_1460.npy':'https://zenodo.org/records/10895826/files/Na_1460.npy',
                'Rb_1460.npy':'https://zenodo.org/records/10895826/files/Rb_1460.npy',
                'H3+_1460.npy':'https://zenodo.org/records/10895826/files/H3+_1460.npy',
                'O3_1460.npy':'https://zenodo.org/records/10895826/files/O3_1460.npy',
                'H2O_1460.npy':'https://zenodo.org/records/10895826/files/H2O_1460.npy',
                'H2_1460.npy':'https://zenodo.org/records/10895826/files/H2_1460.npy',
                'VO_1460.npy':'https://zenodo.org/records/10895826/files/VO_1460.npy',
                'CO_1460.npy':'https://zenodo.org/records/10895826/files/CO_1460.npy',
                'LiF_1460.npy':'https://zenodo.org/records/10895826/files/LiF_1460.npy',
                'N2_1460.npy':'https://zenodo.org/records/10895826/files/N2_1460.npy',
                'CaH_1460.npy':'https://zenodo.org/records/10895826/files/CaH_1460.npy',
                'LiH_1460.npy':'https://zenodo.org/records/10895826/files/LiH_1460.npy',
                'K_1460.npy':'https://zenodo.org/records/10895826/files/K_1460.npy',
                'CH4_1460.npy':'https://zenodo.org/records/10895826/files/CH4_1460.npy',
                'Li_1460.npy':'https://zenodo.org/records/10895826/files/Li_1460.npy',
                'HCN_1460.npy':'https://zenodo.org/records/10895826/files/HCN_1460.npy',
                'TiH_1460.np':'https://zenodo.org/records/10895826/files/TiH_1460.npy',
                'Cs_1460.npy':'https://zenodo.org/records/10895826/files/Cs_1460.npy',
                'NH3_1460.npy':'https://zenodo.org/records/10895826/files/NH3_1460.npy',
                'PH3_1460.npy':'https://zenodo.org/records/10895826/files/PH3_1460.npy',
                'AlH_1460.npy':'https://zenodo.org/records/10895826/files/AlH_1460.npy'
        },
            'description':'By molecule CK-Table on the 1460 grid with 661 wavenumber points.',
            'default_destination':os.path.join(__refdata__, inputs['opacities']['files']['ktable_by_molecule']) ,
        },
            
        'pre-weighted':{
            'url':{"sonora_2020_feh+130_co_100_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_100_noTiOVO.data.196.tar.gz", "sonora_2020_feh-030_co_025_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_025_noTiOVO.data.196.tar.gz", "sonora_2020_feh+030_co_150.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_150.data.196.tar.gz", "sonora_2020_feh-030_co_050.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_050.data.196.tar.gz", "sonora_2020_feh+130_co_200.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_200.data.196.tar.gz", "sonora_2020_feh+070_co_200.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_200.data.196.tar.gz", "sonora_2020_feh+130_co_200_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_200_noTiOVO.data.196.tar.gz", "sonora_2020_feh-030_co_025.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_025.data.196.tar.gz", "sonora_2020_feh+150_co_025.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_025.data.196.tar.gz", "sonora_2020_feh+150_co_050_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_050_noTiOVO.data.196.tar.gz", "sonora_2020_feh-030_co_200_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_200_noTiOVO.data.196.tar.gz", "sonora_2020_feh+070_co_025_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_025_noTiOVO.data.196.tar.gz", "sonora_2020_feh-070_co_100_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_100_noTiOVO.data.196.tar.gz", "sonora_2020_feh-030_co_150_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_150_noTiOVO.data.196.tar.gz", "sonora_2020_feh-070_co_250.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_250.data.196.tar.gz", "sonora_2020_feh+050_co_100_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_100_noTiOVO.data.196.tar.gz", "sonora_2020_feh+100_co_250_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_250_noTiOVO.data.196.tar.gz", "sonora_2020_feh+000_co_025.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_025.data.196.tar.gz", "sonora_2020_feh+050_co_200_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_200_noTiOVO.data.196.tar.gz", "sonora_2020_feh-100_co_050.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_050.data.196.tar.gz", "sonora_2020_feh+130_co_250_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_250_noTiOVO.data.196.tar.gz", "sonora_2020_feh+000_co_200_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_200_noTiOVO.data.196.tar.gz", "sonora_2020_feh+050_co_050.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_050.data.196.tar.gz", "sonora_2020_feh-050_co_025_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_025_noTiOVO.data.196.tar.gz", "sonora_2020_feh-030_co_250_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_250_noTiOVO.data.196.tar.gz", "sonora_2020_feh-050_co_150.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_150.data.196.tar.gz", "sonora_2020_feh+100_co_100_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_100_noTiOVO.data.196.tar.gz", "sonora_2020_feh+050_co_200.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_200.data.196.tar.gz", "sonora_2020_feh+100_co_025.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_025.data.196.tar.gz", "sonora_2020_feh-050_co_200_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_200_noTiOVO.data.196.tar.gz", "sonora_2020_feh-070_co_050_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_050_noTiOVO.data.196.tar.gz", "sonora_2020_feh-100_co_250.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_250.data.196.tar.gz", "sonora_2020_feh+200_co_200.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_200.data.196.tar.gz", "sonora_2020_feh-030_co_100.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_100.data.196.tar.gz", "sonora_2020_feh+100_co_050_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_050_noTiOVO.data.196.tar.gz", "sonora_2020_feh+000_co_050_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_050_noTiOVO.data.196.tar.gz", "sonora_2020_feh+030_co_025.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_025.data.196.tar.gz", "sonora_2020_feh+170_co_025_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_025_noTiOVO.data.196.tar.gz", "sonora_2020_feh-070_co_250_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_250_noTiOVO.data.196.tar.gz", "sonora_2020_feh-070_co_025.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_025.data.196.tar.gz", "sonora_2020_feh-100_co_150.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_150.data.196.tar.gz", "sonora_2020_feh+030_co_200_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_200_noTiOVO.data.196.tar.gz", "sonora_2020_feh+070_co_050.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_050.data.196.tar.gz", "sonora_2020_feh+200_co_100_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_100_noTiOVO.data.196.tar.gz", "sonora_2020_feh+030_co_100_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_100_noTiOVO.data.196.tar.gz", "sonora_2020_feh+200_co_200_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_200_noTiOVO.data.196.tar.gz", "sonora_2020_feh-100_co_025.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_025.data.196.tar.gz", "sonora_2020_feh+200_co_025.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_025.data.196.tar.gz", "sonora_2020_feh+130_co_050_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_050_noTiOVO.data.196.tar.gz", "sonora_2020_feh+100_co_025_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_025_noTiOVO.data.196.tar.gz", "sonora_2020_feh-100_co_200.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_200.data.196.tar.gz", "sonora_2020_feh+070_co_250.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_250.data.196.tar.gz", "sonora_2020_feh+150_co_025_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_025_noTiOVO.data.196.tar.gz", "sonora_2020_feh+050_co_050_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_050_noTiOVO.data.196.tar.gz", "sonora_2020_feh+050_co_150.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_150.data.196.tar.gz", "sonora_2020_feh+130_co_025_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_025_noTiOVO.data.196.tar.gz", "sonora_2020_feh+070_co_250_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_250_noTiOVO.data.196.tar.gz", "sonora_2020_feh-100_co_250_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_250_noTiOVO.data.196.tar.gz", "sonora_2020_feh-100_co_100.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_100.data.196.tar.gz", "sonora_2020_feh+070_co_100.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_100.data.196.tar.gz", "sonora_2020_feh+150_co_150.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_150.data.196.tar.gz", "sonora_2020_feh+150_co_200.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_200.data.196.tar.gz", "sonora_2020_feh-050_co_250_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_250_noTiOVO.data.196.tar.gz", "sonora_2020_feh-070_co_200_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_200_noTiOVO.data.196.tar.gz", "sonora_2020_feh+100_co_050.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_050.data.196.tar.gz", "sonora_2020_feh+100_co_200.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_200.data.196.tar.gz", "sonora_2020_feh+130_co_150_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_150_noTiOVO.data.196.tar.gz", "sonora_2020_feh-070_co_100.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_100.data.196.tar.gz", "sonora_2020_feh+170_co_250.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_250.data.196.tar.gz", "sonora_2020_feh+070_co_150.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_150.data.196.tar.gz", "sonora_2020_feh+030_co_250_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_250_noTiOVO.data.196.tar.gz", "sonora_2020_feh+070_co_100_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_100_noTiOVO.data.196.tar.gz", "sonora_2020_feh+130_co_025.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_025.data.196.tar.gz", "sonora_2020_feh-030_co_200.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_200.data.196.tar.gz", "sonora_2020_feh+200_co_150_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_150_noTiOVO.data.196.tar.gz", "sonora_2020_feh+170_co_200.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_200.data.196.tar.gz", "sonora_2020_feh-050_co_050.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_050.data.196.tar.gz", "sonora_2020_feh-070_co_150.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_150.data.196.tar.gz", "sonora_2020_feh+170_co_100.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_100.data.196.tar.gz", "sonora_2020_feh+130_co_050.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_050.data.196.tar.gz", "sonora_2020_feh+150_co_100_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_100_noTiOVO.data.196.tar.gz", "sonora_2020_feh+000_co_100.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_100.data.196.tar.gz", "sonora_2020_feh+100_co_150_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_150_noTiOVO.data.196.tar.gz", "sonora_2020_feh-070_co_200.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_200.data.196.tar.gz", "sonora_2020_feh+030_co_025_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_025_noTiOVO.data.196.tar.gz", "sonora_2020_feh+000_co_250_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_250_noTiOVO.data.196.tar.gz", "sonora_2020_feh+150_co_150_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_150_noTiOVO.data.196.tar.gz", "sonora_2020_feh+200_co_150.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_150.data.196.tar.gz", "sonora_2020_feh-100_co_150_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_150_noTiOVO.data.196.tar.gz", "sonora_2020_feh+170_co_200_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_200_noTiOVO.data.196.tar.gz", "sonora_2020_feh-050_co_200.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_200.data.196.tar.gz", "sonora_2020_feh+200_co_250_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_250_noTiOVO.data.196.tar.gz", "sonora_2020_feh+000_co_150_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_150_noTiOVO.data.196.tar.gz", "sonora_2020_feh-030_co_050_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_050_noTiOVO.data.196.tar.gz", "sonora_2020_feh+050_co_150_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_150_noTiOVO.data.196.tar.gz", "sonora_2020_feh-030_co_100_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_100_noTiOVO.data.196.tar.gz", "sonora_2020_feh+100_co_100.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_100.data.196.tar.gz", "sonora_2020_feh-050_co_050_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_050_noTiOVO.data.196.tar.gz", "sonora_2020_feh+200_co_100.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_100.data.196.tar.gz", "sonora_2020_feh-030_co_150.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_150.data.196.tar.gz", "sonora_2020_feh+030_co_150_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_150_noTiOVO.data.196.tar.gz", "sonora_2020_feh-100_co_025_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_025_noTiOVO.data.196.tar.gz", "sonora_2020_feh-050_co_100.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_100.data.196.tar.gz", "sonora_2020_feh+150_co_050.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_050.data.196.tar.gz", "sonora_2020_feh+200_co_050_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_050_noTiOVO.data.196.tar.gz", "sonora_2020_feh+070_co_200_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_200_noTiOVO.data.196.tar.gz", "sonora_2020_feh+170_co_150.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_150.data.196.tar.gz", "sonora_2020_feh-100_co_200_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_200_noTiOVO.data.196.tar.gz", "sonora_2020_feh-070_co_150_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_150_noTiOVO.data.196.tar.gz", "sonora_2020_feh-070_co_050.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_050.data.196.tar.gz", "sonora_2020_feh+030_co_200.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_200.data.196.tar.gz", "sonora_2020_feh+030_co_250.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_250.data.196.tar.gz", "sonora_2020_feh+050_co_025.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_025.data.196.tar.gz", "sonora_2020_feh+050_co_250.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_250.data.196.tar.gz", "sonora_2020_feh-050_co_100_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_100_noTiOVO.data.196.tar.gz", "sonora_2020_feh+170_co_150_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_150_noTiOVO.data.196.tar.gz", "sonora_2020_feh+170_co_250_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_250_noTiOVO.data.196.tar.gz", "sonora_2020_feh+050_co_025_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_025_noTiOVO.data.196.tar.gz", "sonora_2020_feh+050_co_250_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_250_noTiOVO.data.196.tar.gz", "sonora_2020_feh-050_co_150_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_150_noTiOVO.data.196.tar.gz", "sonora_2020_feh+100_co_250.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_250.data.196.tar.gz", "sonora_2020_feh+000_co_100_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_100_noTiOVO.data.196.tar.gz", "sonora_2020_feh+170_co_025.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_025.data.196.tar.gz", "sonora_2020_feh+150_co_200_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_200_noTiOVO.data.196.tar.gz", "sonora_2020_feh+170_co_050.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_050.data.196.tar.gz", "sonora_2020_feh+070_co_050_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_050_noTiOVO.data.196.tar.gz", "sonora_2020_feh+170_co_100_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_100_noTiOVO.data.196.tar.gz", "sonora_2020_feh+150_co_250.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_250.data.196.tar.gz", "sonora_2020_feh+000_co_250.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_250.data.196.tar.gz", "sonora_2020_feh+200_co_050.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_050.data.196.tar.gz", "sonora_2020_feh+200_co_025_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_025_noTiOVO.data.196.tar.gz", "sonora_2020_feh+070_co_025.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_025.data.196.tar.gz", "sonora_2020_feh-070_co_025_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_025_noTiOVO.data.196.tar.gz", "sonora_2020_feh-100_co_050_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_050_noTiOVO.data.196.tar.gz", "sonora_2020_feh-100_co_100_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_100_noTiOVO.data.196.tar.gz", "sonora_2020_feh+150_co_250_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_250_noTiOVO.data.196.tar.gz", "sonora_2020_feh+170_co_050_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_050_noTiOVO.data.196.tar.gz", "sonora_2020_feh-050_co_025.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_025.data.196.tar.gz", "sonora_2020_feh+130_co_100.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_100.data.196.tar.gz", "sonora_2020_feh+000_co_200.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_200.data.196.tar.gz", "sonora_2020_feh-050_co_250.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_250.data.196.tar.gz", "sonora_2020_feh+050_co_100.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_100.data.196.tar.gz", "sonora_2020_feh+150_co_100.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_100.data.196.tar.gz", "sonora_2020_feh+000_co_150.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_150.data.196.tar.gz", "sonora_2020_feh+030_co_100.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_100.data.196.tar.gz", "sonora_2020_feh+100_co_200_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_200_noTiOVO.data.196.tar.gz", "sonora_2020_feh+030_co_050.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_050.data.196.tar.gz", "sonora_2020_feh+130_co_150.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_150.data.196.tar.gz", "sonora_2020_feh+000_co_050.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_050.data.196.tar.gz", "sonora_2020_feh-030_co_250.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_250.data.196.tar.gz", "sonora_2020_feh+000_co_025_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_025_noTiOVO.data.196.tar.gz", "sonora_2020_feh+130_co_250.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_250.data.196.tar.gz", "sonora_2020_feh+200_co_250.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_250.data.196.tar.gz", "sonora_2020_feh+070_co_150_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_150_noTiOVO.data.196.tar.gz", "sonora_2020_feh+100_co_150.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_150.data.196.tar.gz", "sonora_2020_feh+030_co_050_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_050_noTiOVO.data.196.tar.gz"},
            'description':'196 CK Tables computed by Roxana Lupu in legacy file formats.',
            'default_destination':os.path.join(__refdata__, inputs['opacities']['files']['preweighted']) ,

            }
        },
        
}

def check_environ(): 
	picaso_refdata = os.environ.get('picaso_refdata',0)
	PYSYN_CDBS = os.environ.get('PYSYN_CDBS',0)
	if picaso_refdata!=0: 
		print('I have found a picaso environment here', picaso_refdata)
		if not os.path.isdir(picaso_refdata):
			print('Uh no. However, I dont recognize this as a directory. You may need to download the reference data folder from Github: https://github.com/natashabatalha/picaso/tree/master/reference')
		else: 
			inside = [i.split('/')[-1] for i in glob.glob(os.path.join(picaso_refdata,'*'))]
			if 'opacities' in inside: 
				ref_v = json.load(open(os.path.join(picaso_refdata,'config.json'))).get('version',2.3)

				print('Fantastic. Looks like you have your basic reference data downloaded. Proceed with grabbing opacities.db with the help of get_data function')
				print(f'Your reference version number {ref_v} has these files:')
				print(inside)
			else: 
				print('I cannot find the standard data in this reference folder, maybe you still need to add the contents to the directory. You should have these folders: https://github.com/natashabatalha/picaso/tree/master/reference')
	else: 
		print('Error: no picaso_refdata is set. Please follow the instructions to create your environment variables: https://natashabatalha.github.io/picaso/installation.html#create-environment-variable')
	if PYSYN_CDBS!=0: 
		print('I have found a stellar environment here', PYSYN_CDBS)
		inside = [i.split('/')[-1] for i in glob.glob(os.path.join(PYSYN_CDBS,'*'))]
		if 'grid' in inside: 
			print('You have the correct subfolder called grid')
			inside_grid = [i.split('/')[-1] for i in glob.glob(os.path.join(PYSYN_CDBS,'grid','*'))]
			print('You have downloaded these stellar grids:')
			print(inside_grid)
		else: 
			print('PYSYN_CDBS does not contain a subfolder called grid which contains your stellar grids. More information on setting this up can be found here: https://natashabatalha.github.io/picaso/installation.html#download-and-link-pysynphot-stellar-data')

	else: 
		print('Error: no PYSYN_CDBS is set which means exoplanet modeling will be hindered. Please follow the instructions to create your environment variables: https://natashabatalha.github.io/picaso/installation.html#create-environment-variable')

def get_data(category_download=None,target_download=None, final_destination_dir=None):
    input_config, data_config=get_data_config()
    __refdata__=os.environ.get('picaso_refdata')
    if ((category_download==None) and (target_download==None)):
        print('What data can I help you download? Options include:')
        options = [i for i in data_config.keys()]
        print(options)
        category_download = input()
        while category_download not in options:
            print('I dont recognize that category. Try again:',options)
            category_download = input()

    if target_download==None: 
        print(f'Great. I found these options for {category_download}. Select one:')
        options = [i for i in data_config[category_download].keys()]
        for ii, i in enumerate(options): 
            print(f"{ii} - '{i}': {data_config[category_download][i]['description']}")
        
        target_download = input()
        while target_download not in options:
            print('I dont recognize that target. Try again:',options)
            target_download = input()    
    
    url_download = list( data_config[category_download][target_download]['url'].values())
    download_name = list(data_config[category_download][target_download]['url'].keys())
    
    if final_destination_dir==None: 
        print('No destination has been specified. Let me help put this in the right place.')

        if 'stellar' in category_download: 
            print("""Stellar gird models should go to the environment variable directory called PYSYN_CDBS. 
            """)
            PYSYN_CDBS = os.environ.get('PYSYN_CDBS','')
            if os.path.isdir(PYSYN_CDBS):
                print('It looks like you have set PYSYN_CDBS path as: ',PYSYN_CDBS)
            elif PYSYN_CDBS=='':
                print('It does not look like have created PYSYN_CDBS environment variable yet. Please do so by following these instructions first: https://natashabatalha.github.io/picaso/installation.html#create-environment-variable')
                print('After you have created the environment variable I can help download the file and put it in the right place')
                return 
            elif not os.path.isdir(PYSYN_CDBS):
                print('It looks like you have set PYSYN_CDBS path as: ',PYSYN_CDBS)
                print('However, this is not a real directory. You might have a typo and should make sure it is the correct path.')  
                print('After you have created the environment variable I can help download the file and put it in the right place')
                return 

            final_destination_dir = os.path.join(PYSYN_CDBS,'grid')
            if not os.path.isdir(final_destination_dir): 
                os.mkdir(final_destination_dir)
            
        elif 'opacity' in category_download: 
            print("""PICASO uses one default opacity database that is placed here: $picaso_refdata/opacities which is auto called when opannection() is run.
            Any other reampled opacity files are recommended you store in a the "extra" directory:  $picaso_refdata/opacities/resampled. 
            Since no destination was specified, can you tell me, would you like to make this your one "default" or add to "extra" or speciy your "ownpath"? "default" or "extra" or "ownpath". 
            """)
            make_default= input() 

            while make_default not in ['default', 'extra','ownpath']:
                print('I did not understand. Please enter: default or extra or ownpath')
                make_default= input() 
                        
            if make_default.lower()=='default':
                final_destination_dir = os.path.join(__refdata__, input_config['opacities']['files']['resampled-default']) 
            elif make_default.lower()=='extra': 
                final_destination_dir = os.path.join(__refdata__, input_config['opacities']['files']['resampled']) 

            else: 
                print('Please enter where you would like to store this file. Default (cwd)=',os.getcwd())
                final_destination_dir = input() 
                if final_destination_dir=='':
                    final_destination_dir=os.getcwd()
                while not os.path.isdir(final_destination_dir):
                    print('I dont recognize that directory. Please enter a valid directory or press enter to keep in current working directory')
                    final_destination_dir = input() 
        elif 'default_destination' in data_config[category_download][target_download].keys():
            default = data_config[category_download][target_download]['default_destination']
            print(f'I found this suggested default destination: {default}. Would you like to add them to this destination? yes or no')
            make_default= input() 
            while make_default not in ['yes', 'no']:
                print('I did not understand. Please enter, yes or no')
                make_default= input() 
            if ((make_default.lower()=='y') or (make_default.lower()=='yes')):
                final_destination_dir = default 
                #in some cases like if "bobcat" is being added to "sonora" directory we want to create an additional 
                #embedded directory
                #so if the final destimation does not exist lets try to make it. 
                #if it does not exist and this fails then we have a problem 
                if not os.path.isdir(final_destination_dir):
                    try: 
                        os.mkdir(final_destination_dir) 
                    except: 
                        raise Exception(f'Failed to make directory for specified data. Make sure python has permission to create new directories or create this directory yourself and try again: {final_destination_dir}') 
            else: 
                print('No Problem. Please enter where you would like to store this file. Default=',os.getcwd())
                final_destination_dir = input() 
                if final_destination_dir=='':
                    final_destination_dir=os.getcwd()
                while not os.path.isdir(final_destination_dir):
                    print('I dont recognize that directory. Please enter a existing  directory or press enter to keep in current working directory')
                    final_destination_dir = input()              
        else: 
            raise Exception('Internal PICASO issue: default destimation not supplied in data config. Contact develoepers.')

    #with tempfile.TemporaryDirectory() as temp_dir:
    allzips = []
    totalfiles = len(url_download)
    for ii, iurl, iname in zip(range(totalfiles),url_download,download_name):
        print(f'Downloading file {ii}/{totalfiles} which is saving to location: {final_destination_dir}')
        
        processor = None
        
        if iname.endswith('.zip'):
            processor = pooch.Unzip(extract_dir='')
            
        elif iname.endswith('.tar.gz') or iname.endswith('.tar'):
            processor = pooch.Untar(extract_dir='')

        # Using pooch to download and unpack
        
        try:
            files = pooch.retrieve(
                url=iurl,
                known_hash=None,
                fname=iname,
                path=final_destination_dir,
                processor=processor,
                progressbar=True
            )

        except Exception as e:
            print(f"Error downloading or processing {iurl}: {e}")
            continue


        if 'stellar_grid' in category_download: 
            og = os.path.join(final_destination_dir, 'grp','redcat','trds','grid',target_download)
            new = os.path.join(PYSYN_CDBS,'grid')
            try:
                shutil.move(og ,new)
            except: 
                print(f'I tried to move stellar grids from here {og} to here {new} but it failed. Likely because the file already exists. Find your file here {og} and manually move it if you want to overwrite. ')

