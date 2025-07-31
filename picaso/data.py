from urllib.request import urlretrieve
import tarfile
import os
import shutil
import glob
import uuid
import json

try:
    from IPython.display import display, HTML
except ImportError:
    def display(data):
        """A dummy display function for non-IPython environments."""
        print(data)
    def HTML(data):
        """A dummy HTML class for non-IPython environments."""
        return data

def remove_directory(dir_path):
    """Removes a directory and all its contents.

    Args:
        dir_path: The path to the directory to remove.
    """
    try:
        shutil.rmtree(dir_path)
        print(f"Directory '{dir_path}' and its contents have been removed.")
    except FileNotFoundError:
        print(f"Directory '{dir_path}' not found.")
    except OSError as e:
        print(f"Error removing directory '{dir_path}': {e}")
def download_with_progress(url, filename):
    """Downloads a file from a URL and displays a progress bar."""

    def progress_bar(count, block_size, total_size):
        """Displays a simple progress bar."""
        percent = int(count * block_size * 100 / total_size)
        print(f"\rDownloading: {percent}% [{('#' * percent)}{(' ' * (100 - percent))}]", end="")

    try:
        urlretrieve(url, filename, reporthook=progress_bar)
        print("\nDownload complete!")  # Add a newline after the progress bar finishes.
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def get_data_config():
    __refdata__ = os.environ.get('picaso_refdata','')
    if __refdata__ =='':raise Exception('Cannot get data config until picaso_refdata environment is set')
    inputs = json.load(open(os.path.join(__refdata__,'config.json')))   

    __stellar_refdata__ = os.environ.get('PYSYN_CDBS','$PYSYN_CDBS')
    if __stellar_refdata__ =='$PYSYN_CDBS':print("Stellar environment variable PYSYN_CDBS not yet set, which may impact exoplanet functionality")

    return {
    "resampled_opacity":{
        'default':{
           'url':'https://zenodo.org/records/14861730/files/opacities_0.3_15_R15000.db.tar.gz?download=1',
            'filename':'opacities_0.3_15_R15000.db.tar.gz',
            'description':'7.34 GB file resampled at R=15,000 from 0.3-15um. This is sufficient for doing R=100 JWST calculations and serves as a good default opacity database for exploration.',
            'default_destination':os.path.join(__refdata__, inputs['opacities']['files']['opacity']) 
        },
        'R60000,0.6-6um':{
            'url':'https://zenodo.org/records/6928501/files/all_opacities_0.6_6_R60000.db.tar.gz?download=1',
            'filename':'all_opacities_0.6_6_R60000.db.tar.gz',
            'description':'38.3 GB file resampled at R=60,000 from 0.6-6um. This is sufficient for doing moderate resolution JWST calculations.',
            },
        'R20000,4.8-15um':{
            'url':'https://zenodo.org/records/6928501/files/all_opacities_4.8_15_R20000.db.tar.gz?download=1',
            'filename':'all_opacities_4.8_15_R20000.db.tar.gz',
            'description':'7.0 GB file resampled at R=20,000 from 4.8-15um. This is sufficient for doing low resolution JWST calculations.',
            }
        },
    'stellar_grids':{
        'phoenix':{
            'url':'http://ssb.stsci.edu/trds/tarfiles/synphot5.tar.gz',
            'filename':'synphot5.tar.gz',
            'description':'Phoenix stellar atlas',
            'default_destination':os.path.join(__stellar_refdata__, 'grid')
            },
        'ck04models':{
            'url':'http://ssb.stsci.edu/trds/tarfiles/synphot3.tar.gz',
            'filename':'synphot3.tar.gz',
            'description':'Castelli & Kurucz (2004) stellar atlas',
            'default_destination':os.path.join(__stellar_refdata__, 'grid')
            },
        },
    'virga_mieff':{
        'default':{
            'url':'https://zenodo.org/records/5179187/files/virga.zip?download=1',
            'filename':'virga.zip',
            'description':'Virga refractive index and Mie files on standard 196 grid',
            }
        },
    'sonora_grids':{
        'elfowl-Ytype':{
            'url':['https://zenodo.org/records/10381250/files/output_275.0_325.0.tar.gz',
                    'https://zenodo.org/records/10381250/files/output_350.0_400.0.tar.gz',
                    'https://zenodo.org/records/10381250/files/output_500.0_550.0.tar.gz',
                    'https://zenodo.org/records/10381250/files/output_425.0_475.0.tar.gz'],
            'filename':['output_275.0_325.0.tar.gz',
                        'output_350.0_400.0.tar.gz',
                        'output_500.0_550.0.tar.gz',
                        'output_425.0_475.0.tar.gz'],
            'description':'The models between Teff of 275 to 550 K (applicable to Y-type objects). Total: ~40 Gb.',
            },
        'elfowl-Ttype':{
            'url':['https://zenodo.org/records/10385821/files/output_700.0_800.tar.gz',
                    'https://zenodo.org/records/10385821/files/output_850.0_950.tar.gz',
                    'https://zenodo.org/records/10385821/files/output_1000.0_1200.tar.gz',
                    'https://zenodo.org/records/10385821/files/output_575.0_650.tar.gz'],
            'filename':['output_700.0_800.tar.gz',
                        'output_850.0_950.tar.gz',
                        'output_1000.0_1200.tar.gz',
                        'output_575.0_650.tar.gz'],
            'description':'The models for Teff between 575 to 1200 K (applicable for T-type objects). Total: ~40 Gb.',
            },
        'elfowl-Ltype':{
            'url':['https://zenodo.org/records/10385987/files/output_1600.0_1800.tar.gz',
                    'https://zenodo.org/records/10385987/files/output_1900.0_2100.tar.gz',
                    'https://zenodo.org/records/10385987/files/output_2200.0_2400.tar.gz',
                    'https://zenodo.org/records/10385987/files/output_1300.0_1400.tar.gz'
                    ],
            'filename':[
                    'output_1600.0_1800.tar.gz',
                    'output_1900.0_2100.tar.gz',
                    'output_2200.0_2400.tar.gz',
                    'output_1300.0_1400.tar.gz'
            ],
            'description':'Models for Teff between 1300 to 2400 K (applicable for L-type objects). Total: ~40 Gb.',
            },
        'bobcat':{
            'url':'https://zenodo.org/records/1309035/files/profile.tar?download=1',
            'filename':'profile.tar',
            'description':'Sonora bobcat pressure-temperature profiles',
            },
        'diamondback':{
            'url':'',
            'description':'',
            },
        },
    'ck_tables':{
        #'default-pre-mixed':{
        #    'url':'',
        #    'description':'',
        #    },
        'by-molecule':{
            'url':[
                'https://zenodo.org/records/10895826/files/H2S_1460.npy',
                'https://zenodo.org/records/10895826/files/MgH_1460.npy',
                'https://zenodo.org/records/10895826/files/O2_1460.npy',
                'https://zenodo.org/records/10895826/files/FeH_1460.npy',
                'https://zenodo.org/records/10895826/files/TiO_1460.npy',
                'https://zenodo.org/records/10895826/files/SO2_1460.npy',
                'https://zenodo.org/records/10895826/files/Fe_1460.npy',
                'https://zenodo.org/records/10895826/files/C2H4_1460.npy',
                'https://zenodo.org/records/10895826/files/OCS_1460.npy',
                'https://zenodo.org/records/10895826/files/C2H6_1460.npy',
                'https://zenodo.org/records/10895826/files/SiO_1460.npy',
                'https://zenodo.org/records/10895826/files/C2H2_1460.npy',
                'https://zenodo.org/records/10895826/files/LiCl_1460.npy',
                'https://zenodo.org/records/10895826/files/CO2_1460.npy',
                'https://zenodo.org/records/10895826/files/CrH_1460.npy',
                'https://zenodo.org/records/10895826/files/Na_1460.npy',
                'https://zenodo.org/records/10895826/files/Rb_1460.npy',
                'https://zenodo.org/records/10895826/files/H3+_1460.npy',
                'https://zenodo.org/records/10895826/files/O3_1460.npy',
                'https://zenodo.org/records/10895826/files/H2O_1460.npy',
                'https://zenodo.org/records/10895826/files/H2_1460.npy',
                'https://zenodo.org/records/10895826/files/VO_1460.npy',
                'https://zenodo.org/records/10895826/files/CO_1460.npy',
                'https://zenodo.org/records/10895826/files/LiF_1460.npy',
                'https://zenodo.org/records/10895826/files/N2_1460.npy',
                'https://zenodo.org/records/10895826/files/CaH_1460.npy',
                'https://zenodo.org/records/10895826/files/LiH_1460.npy',
                'https://zenodo.org/records/10895826/files/K_1460.npy',
                'https://zenodo.org/records/10895826/files/CH4_1460.npy',
                'https://zenodo.org/records/10895826/files/Li_1460.npy',
                'https://zenodo.org/records/10895826/files/HCN_1460.npy',
                'https://zenodo.org/records/10895826/files/TiH_1460.npy',
                'https://zenodo.org/records/10895826/files/Cs_1460.npy',
                'https://zenodo.org/records/10895826/files/NH3_1460.npy',
                'https://zenodo.org/records/10895826/files/PH3_1460.npy',
                'https://zenodo.org/records/10895826/files/AlH_1460.npy'
            ],
            'description':'By molecule CK-Table on the 1460 grid with 661 wavenumber points.',
            'default_destination':os.path.join(__refdata__, inputs['opacities']['files']['ktable_by_molecule']) ,
            'filename':[
                'H2S_1460.npy',
                'MgH_1460.npy',
                'O2_1460.npy',
                'FeH_1460.npy',
                'TiO_1460.npy',
                'SO2_1460.npy',
                'Fe_1460.npy',
                'C2H4_1460.npy',
                'OCS_1460.npy',
                'C2H6_1460.npy',
                'SiO_1460.npy',
                'C2H2_1460.npy',
                'LiCl_1460.npy',
                'CO2_1460.npy',
                'CrH_1460.npy',
                'Na_1460.npy',
                'Rb_1460.npy',
                'H3+_1460.npy',
                'O3_1460.npy',
                'H2O_1460.npy',
                'H2_1460.npy',
                'VO_1460.npy',
                'CO_1460.npy',
                'LiF_1460.npy',
                'N2_1460.npy',
                'CaH_1460.npy',
                'LiH_1460.npy',
                'K_1460.npy',
                'CH4_1460.npy',
                'Li_1460.npy',
                'HCN_1460.npy',
                'TiH_1460.npy',
                'Cs_1460.npy',
                'NH3_1460.npy',
                'PH3_1460.npy',
                'AlH_1460.npy'
            ]
            },
        'pre-weighted':{
            'url':['https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_050.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_250_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_100.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_050_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_050.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_100.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_025_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_100_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_100.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_025.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_150.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_050_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_250_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_050_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_100_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_150.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_050.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_025_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_150.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_025_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_025.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_150_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_025.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_200.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_250_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_250_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_250.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_200_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_050_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_050.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_250.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_150_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_200_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_200_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_250.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_025.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_100_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_200.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_200.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_200_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_150_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_250.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_250_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_050_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_200.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_150_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_150_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_150.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_200_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_150.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_200.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_100_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_050.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_025_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_100_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_150_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_250.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_150.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_050.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_100_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_050.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_025_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_200_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_100.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_025_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_200.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_150_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_150.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_200.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_100.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_025_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_100_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_025.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_200.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_200.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_100.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_050_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_050.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_025.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_025_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_250_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_100.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_250_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_200_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_250.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_025.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_250_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_250.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_025.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_150_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_025_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_025_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_250.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_150_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_200_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_200_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_200_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_050_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_200.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_200.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_050_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_100.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_050_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_100.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_050.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_025_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_025.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_150_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_200.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_050.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_050_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_100_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_150_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_200_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_150.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_150.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_150_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_250.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_150.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_050.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_250.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_100_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_150.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_100_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_250_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_100.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_100.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_100_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_150.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_150.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_050.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_100.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_025_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_150_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_200.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_025.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_250_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_250_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_100_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_100_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_250.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_250.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_200_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_200_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_250.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_250_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_025.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_050_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_250_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_050_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_100.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_025.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_050.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_025_noTiOVO.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_025.data.196.tar.gz',
                    'https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_050_noTiOVO.data.196.tar.gz'
                    ],
            'description':'196 CK Tables computed by Roxana Lupu in legacy file formats.',
            'filename':[
                    'sonora_2020_feh+050_co_050.data.196.tar.gz',
                    'sonora_2020_feh+150_co_250_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+050_co_100.data.196.tar.gz',
                    'sonora_2020_feh+150_co_050_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+150_co_050.data.196.tar.gz',
                    'sonora_2020_feh+150_co_100.data.196.tar.gz',
                    'sonora_2020_feh+150_co_025_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+150_co_100_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh-050_co_100.data.196.tar.gz',
                    'sonora_2020_feh+150_co_025.data.196.tar.gz',
                    'sonora_2020_feh+150_co_150.data.196.tar.gz',
                    'sonora_2020_feh+050_co_050_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+130_co_250_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh-050_co_050_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+050_co_100_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh-050_co_150.data.196.tar.gz',
                    'sonora_2020_feh-050_co_050.data.196.tar.gz',
                    'sonora_2020_feh+050_co_025_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+050_co_150.data.196.tar.gz',
                    'sonora_2020_feh-050_co_025_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+050_co_025.data.196.tar.gz',
                    'sonora_2020_feh+150_co_150_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh-050_co_025.data.196.tar.gz',
                    'sonora_2020_feh+150_co_200.data.196.tar.gz',
                    'sonora_2020_feh+030_co_250_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh-030_co_250_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+030_co_250.data.196.tar.gz',
                    'sonora_2020_feh+150_co_200_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+100_co_050_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+100_co_050.data.196.tar.gz',
                    'sonora_2020_feh-030_co_250.data.196.tar.gz',
                    'sonora_2020_feh-050_co_150_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+030_co_200_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh-030_co_200_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+150_co_250.data.196.tar.gz',
                    'sonora_2020_feh+030_co_025.data.196.tar.gz',
                    'sonora_2020_feh-050_co_100_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+030_co_200.data.196.tar.gz',
                    'sonora_2020_feh-030_co_200.data.196.tar.gz',
                    'sonora_2020_feh-100_co_200_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+050_co_150_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+130_co_250.data.196.tar.gz',
                    'sonora_2020_feh+200_co_250_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh-100_co_050_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh-050_co_200.data.196.tar.gz',
                    'sonora_2020_feh+030_co_150_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh-030_co_150_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+030_co_150.data.196.tar.gz',
                    'sonora_2020_feh+130_co_200_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh-030_co_150.data.196.tar.gz',
                    'sonora_2020_feh+130_co_200.data.196.tar.gz',
                    'sonora_2020_feh+030_co_100_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh-070_co_050.data.196.tar.gz',
                    'sonora_2020_feh+070_co_025_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh-030_co_100_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+130_co_150_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+200_co_250.data.196.tar.gz',
                    'sonora_2020_feh+130_co_150.data.196.tar.gz',
                    'sonora_2020_feh-030_co_050.data.196.tar.gz',
                    'sonora_2020_feh+130_co_100_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh-100_co_050.data.196.tar.gz',
                    'sonora_2020_feh-070_co_025_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+200_co_200_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh-100_co_100.data.196.tar.gz',
                    'sonora_2020_feh+100_co_025_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+200_co_200.data.196.tar.gz',
                    'sonora_2020_feh+200_co_150_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+200_co_150.data.196.tar.gz',
                    'sonora_2020_feh+100_co_200.data.196.tar.gz',
                    'sonora_2020_feh+030_co_100.data.196.tar.gz',
                    'sonora_2020_feh-100_co_025_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+200_co_100_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+100_co_025.data.196.tar.gz',
                    'sonora_2020_feh-100_co_200.data.196.tar.gz',
                    'sonora_2020_feh+050_co_200.data.196.tar.gz',
                    'sonora_2020_feh+200_co_100.data.196.tar.gz',
                    'sonora_2020_feh+200_co_050_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+200_co_050.data.196.tar.gz',
                    'sonora_2020_feh-100_co_025.data.196.tar.gz',
                    'sonora_2020_feh+200_co_025_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+070_co_250_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh-030_co_100.data.196.tar.gz',
                    'sonora_2020_feh-070_co_250_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh-050_co_200_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+070_co_250.data.196.tar.gz',
                    'sonora_2020_feh+200_co_025.data.196.tar.gz',
                    'sonora_2020_feh+170_co_250_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh-070_co_250.data.196.tar.gz',
                    'sonora_2020_feh+170_co_025.data.196.tar.gz',
                    'sonora_2020_feh+100_co_150_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh-030_co_025_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+030_co_025_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+170_co_250.data.196.tar.gz',
                    'sonora_2020_feh-100_co_150_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+070_co_200_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+170_co_200_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh-070_co_200_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+030_co_050_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+070_co_200.data.196.tar.gz',
                    'sonora_2020_feh-070_co_200.data.196.tar.gz',
                    'sonora_2020_feh-030_co_050_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+000_co_100.data.196.tar.gz',
                    'sonora_2020_feh+000_co_050_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+130_co_100.data.196.tar.gz',
                    'sonora_2020_feh+000_co_050.data.196.tar.gz',
                    'sonora_2020_feh+000_co_025_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+000_co_025.data.196.tar.gz',
                    'sonora_2020_feh+070_co_150_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+170_co_200.data.196.tar.gz',
                    'sonora_2020_feh+030_co_050.data.196.tar.gz',
                    'sonora_2020_feh+130_co_050_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+170_co_100_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh-070_co_150_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+050_co_200_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+100_co_150.data.196.tar.gz',
                    'sonora_2020_feh+070_co_150.data.196.tar.gz',
                    'sonora_2020_feh+170_co_150_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh-050_co_250.data.196.tar.gz',
                    'sonora_2020_feh-070_co_150.data.196.tar.gz',
                    'sonora_2020_feh+070_co_050.data.196.tar.gz',
                    'sonora_2020_feh+050_co_250.data.196.tar.gz',
                    'sonora_2020_feh+070_co_100_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+170_co_150.data.196.tar.gz',
                    'sonora_2020_feh-070_co_100_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh-050_co_250_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+070_co_100.data.196.tar.gz',
                    'sonora_2020_feh-070_co_100.data.196.tar.gz',
                    'sonora_2020_feh+000_co_100_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh-100_co_150.data.196.tar.gz',
                    'sonora_2020_feh+000_co_150.data.196.tar.gz',
                    'sonora_2020_feh+130_co_050.data.196.tar.gz',
                    'sonora_2020_feh+100_co_100.data.196.tar.gz',
                    'sonora_2020_feh+130_co_025_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+000_co_150_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+000_co_200.data.196.tar.gz',
                    'sonora_2020_feh+130_co_025.data.196.tar.gz',
                    'sonora_2020_feh+100_co_250_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh-100_co_250_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+100_co_100_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh-100_co_100_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+100_co_250.data.196.tar.gz',
                    'sonora_2020_feh-100_co_250.data.196.tar.gz',
                    'sonora_2020_feh+100_co_200_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+000_co_200_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+000_co_250.data.196.tar.gz',
                    'sonora_2020_feh+000_co_250_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh-030_co_025.data.196.tar.gz',
                    'sonora_2020_feh-070_co_050_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+050_co_250_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+170_co_050_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+170_co_100.data.196.tar.gz',
                    'sonora_2020_feh-070_co_025.data.196.tar.gz',
                    'sonora_2020_feh+170_co_050.data.196.tar.gz',
                    'sonora_2020_feh+170_co_025_noTiOVO.data.196.tar.gz',
                    'sonora_2020_feh+070_co_025.data.196.tar.gz',
                    'sonora_2020_feh+070_co_050_noTiOVO.data.196.tar.gz'
            ]
            }
        }
}


def _is_notebook():
    """
    Checks if the code is being run in a Jupyter notebook.
    This is used to determine whether to display output as HTML.
    """
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except (NameError, ImportError):
        return False      # Probably standard Python interpreter or IPython not installed
    except Exception:
        return False      # Another exception, assume not a notebook.


def check_environ():
    """
    Checks user environment variables and provides feedback.
    This function will display its output as formatted HTML if run in a Jupyter notebook.
    """
    messages = []
    picaso_refdata = os.environ.get('picaso_refdata')
    PYSYN_CDBS = os.environ.get('PYSYN_CDBS')

    # --- Check for picaso_refdata ---
    if picaso_refdata:
        messages.append(('info', f'Found <code>picaso_refdata</code> environment variable: <code>{picaso_refdata}</code>'))
        if not os.path.isdir(picaso_refdata):
            messages.append(('error', 'This path does not appear to be a valid directory. You may need to download the reference data folder from <a href="https://github.com/natashabatalha/picaso/tree/master/reference" target="_blank">GitHub</a>.'))
        else:
            inside = [os.path.basename(i) for i in glob.glob(os.path.join(picaso_refdata, '*'))]
            if 'opacities' in inside:
                messages.append(('success', 'Basic picaso reference data seems to be in place.'))
                try:
                    config_path = os.path.join(picaso_refdata, 'config.json')
                    with open(config_path) as f:
                        ref_v = json.load(f).get('version', 'N/A')
                    messages.append(('info', f'Reference data version: {ref_v}'))
                    messages.append(('info', 'Files in reference directory:'))
                    messages.append(('list', inside))
                except FileNotFoundError:
                    messages.append(('warning', 'Could not find <code>config.json</code> to check version number.'))
                except json.JSONDecodeError:
                    messages.append(('warning', 'Could not parse <code>config.json</code>.'))
            else:
                messages.append(('error', 'The "opacities" folder was not found in your reference directory. Please ensure you have all the folders from <a href="https://github.com/natashabatalha/picaso/tree/master/reference" target="_blank">GitHub</a>.'))
    else:
        messages.append(('error', 'The <code>picaso_refdata</code> environment variable is not set. Please see the <a href="https://natashabatalha.github.io/picaso/installation.html#create-environment-variable" target="_blank">Installation Guide</a>.'))

    # --- Check for PYSYN_CDBS ---
    if PYSYN_CDBS:
        messages.append(('info', f'Found <code>PYSYN_CDBS</code> environment variable for <code>stsynphot</code> was found: <code>{PYSYN_CDBS}</code>'))
        inside = [os.path.basename(i) for i in glob.glob(os.path.join(PYSYN_CDBS, '*'))]
        if 'grid' in inside:
            messages.append(('success', 'Found stellar "grid" subfolder.'))
            inside_grid = [os.path.basename(i) for i in glob.glob(os.path.join(PYSYN_CDBS, 'grid', '*'))]
            if inside_grid:
                messages.append(('info', 'Downloaded stellar grids:'))
                messages.append(('list', inside_grid))
            else:
                messages.append(('warning', 'The stellar "grid" subfolder is empty. You may need to download stellar grids which can you can do by running <code>data.get_data</code>'))
        else:
            messages.append(('error', '<code>PYSYN_CDBS</code> does not contain the required "grid" subfolder. See the <a href="https://natashabatalha.github.io/picaso/installation.html#download-and-link-pysynphot-stellar-data" target="_blank">Installation Guide</a>.'))
    else:
        messages.append(('warning', 'The <code>PYSYN_CDBS</code> environment variable is not set. This will hinder any modeling that sets stellar parameters with the star function. See the <a href="https://natashabatalha.github.io/picaso/installation.html#create-environment-variable" target="_blank">Installation Guide</a>.'))

    # --- Display Messages ---
    if _is_notebook():
        # Generate and display HTML output for Jupyter notebooks
        html_output = '<div style="border: 1px solid #e0e0e0; padding: 15px; border-radius: 5px; background-color: #f9f9f9; font-family: sans-serif; max-width: 800px; margin: auto;">'
        html_output += '<h3 style="margin-top: 0; border-bottom: 2px solid #e0e0e0; padding-bottom: 10px; font-size: 1.2em;">PICASO Environment Check</h3>'

        style_map = {
            'info': 'color: #00529B; background-color: #BDE5F8;',
            'success': 'color: #4F8A10; background-color: #DFF2BF;',
            'warning': 'color: #9F6000; background-color: #FEEFB3;',
            'error': 'color: #D8000C; background-color: #FFBABA;',
        }

        icon_map = {
            'info': '&#8505;',      # Info
            'success': '&#10004;',  # Check
            'warning': '&#9888;',   # Warning
            'error': '&#10006;',    # Cross
        }

        for msg_type, msg in messages:
            if msg_type == 'list':
                html_output += '<ul style="margin: 5px 0 10px 20px; padding-left: 20px; border: 1px solid #e0e0e0; background-color: #fff; border-radius: 4px;">'
                for item in msg:
                    html_output += f'<li style="margin: 4px 0;"><code>{item}</code></li>'
                html_output += '</ul>'
            else:
                style = style_map.get(msg_type, 'color: #000; background-color: #fff;')
                icon = icon_map.get(msg_type, '')
                html_output += f'<div style="padding: 10px; margin: 5px 0; border-radius: 4px; {style}">{icon} {msg}</div>'

        html_output += '</div>'
        display(HTML(html_output))
    else:
        # Keep the original text-based output for other environments
        print("--- PICASO Environment Check ---")
        for msg_type, msg in messages:
            if msg_type == 'list':
                for item in msg:
                    print(f"  - {item}")
            else:
                import re
                clean_msg = re.sub('<[^<]+?>', '', msg) # Strip HTML tags for console
                print(f"[{msg_type.upper()}] {clean_msg}")



def get_data(category_download=None,target_download=None, final_destination_dir=None):
    data_config=get_data_config()
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
    
    url_download = data_config[category_download][target_download]['url']
    download_name = data_config[category_download][target_download]['filename']
    if isinstance(url_download,str) :
        url_download = [url_download]
        download_name = [download_name]
    
        
    if final_destination_dir==None: 
        print('No destination has been specified. Let me help put this in the right place.')

        #STELLAR GRID HELP
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
            
        #OPACITY HELP            
        elif 'opacity' in category_download: 
            print("""PICASO uses one default opacity database that is placed here: $picaso_refdata/opacities/opacities.db .
            Any other files are recommended you store in a separate directory. 
            Since no destination was specified, can you tell me, would you like to make this your default PICASO file? yes or no. 
            """)
            make_default= input() 

            while make_default not in ['yes', 'no']:
                print('I did not understand. Please enter, yes or no')
                make_default= input() 
        
            if ((make_default.lower()=='y') or (make_default.lower()=='yes')):
                picaso_refdata = os.environ.get('picaso_refdata','')
                if os.path.isdir(picaso_refdata):
                    print('It looks like you have set picaso_refdata path as: ',picaso_refdata)
                    final_destination_dir = data_config['resampled_opacity']['default']['default_destination']#os.path.join(picaso_refdata,'opacities','opacities.db')
                elif picaso_refdata=='':
                    print('It does not look like have created picaso_refdata environment variable yet. Please do so by following these instructions first: https://natashabatalha.github.io/picaso/installation.html#create-environment-variable')
                    print('After you have created the environment variable I can help download the file and put it in the right place')
                    return 
                elif not os.path.isdir(picaso_refdata):
                    print('It looks like you have set picaso_refdata path as: ',picaso_refdata)
                    print('However, this is not a real directory. You might have a typo and should make sure it is the correct path.')  
                    print('After you have created the environment variable I can help download the file and put it in the right place')
                    return 
            else: 
                print('No Problem. Please enter where you would like to store this file. Default=',os.getcwd())
                final_destination_dir = input() 
                if final_destination_dir=='':
                    final_destination_dir=os.getcwd()
                while not os.path.isdir(final_destination_dir):
                    print('I dont recognize that directory. Please enter a valid directory or press enter to keep in current working directory')
                    final_destination_dir = input() 
        elif 'default_destination' in data_config[category_download][target_download].keys():
            default = data_config[category_download][target_download]['default_destination']
            print(f'I found this suggested default destination: {default}. Would you like to make these your default files and add them to this destination? yes or no')
            make_default= input() 
            while make_default not in ['yes', 'no']:
                print('I did not understand. Please enter, yes or no')
                make_default= input() 
            if ((make_default.lower()=='y') or (make_default.lower()=='yes')):
                final_destination_dir = default   
            else: 
                print('No Problem. Please enter where you would like to store this file. Default=',os.getcwd())
                final_destination_dir = input() 
                if final_destination_dir=='':
                    final_destination_dir=os.getcwd()
                while not os.path.isdir(final_destination_dir):
                    print('I dont recognize that directory. Please enter a valid directory or press enter to keep in current working directory')
                    final_destination_dir = input()              
        else: 
            print(f'When running the code you will have to point to this directory. Therefore, keep it somewhere you will remember. My suggestion would be something like /Users/myaccount/Documents/data/picaso_data/{category_download}. Please enter a path:')
            final_destination_dir=input()
            while not os.path.isdir(final_destination_dir):
                print("Whoops. I dont recognize this directory. Please type in an existing directory")
                final_destination_dir=input()

            
    
    temp_dir = os.path.join(os.getcwd(),f'temp_picaso_dir_{uuid.uuid4()}')
    os.mkdir(temp_dir)
    allzips = []
    for iurl, iname in zip(url_download,download_name): 
        print(f'Downloading target url: {iurl} to the temp directory called: {temp_dir}. Then we will unpack and move it. If something goes wrong you can find your file in this temp directory.')
        download_with_progress(iurl, os.path.join(temp_dir,iname))
        if (('zip' in iname) or ('tar' in iname)):
            print('Unpacking',iname)
            shutil.unpack_archive(os.path.join(temp_dir,iname), temp_dir)
            allzips += [os.path.join(temp_dir,iname)]
    
    contents = [i for i in glob.glob(os.path.join(temp_dir,'*')) if i not in allzips]
        
    if 'stellar' in category_download: 
        lets_move_this = [os.path.join(temp_dir,'grp','redcat','trds','grid',target_download)]
    elif len(contents)==1:
        lets_move_this = [contents[0]]
    elif len(contents)>1: 
        if not os.path.isdir(final_destination_dir):
            raise Exception(f'Cannot move many files {contents} to one destination file: {final_destination_dir}')
        lets_move_this = contents
            
        
    print(f"Moving '{str(lets_move_this)[0:200]}' to this desntination '{final_destination_dir}'. Press enter to proceed or anything else to quit.")
    proceed = input()
    
    if proceed!='':
        return 
    else: 
        checkexists =[i.split('/')[-1] for i in glob.glob(final_destination_dir+'*')+glob.glob(os.path.join(final_destination_dir,'*'))]
        for iLMT in lets_move_this:
            if iLMT.split('/')[-1] not in checkexists: 
                shutil.move(iLMT, final_destination_dir)  
            else: 
                raise Exception(f"I cannot move this file as it would overwrite an existing file. You should proceed manually, with intention to move these '{str(lets_move_this)[0:200]}' to this desntination '{final_destination_dir}'. ")

    remove_directory(temp_dir)
    print('Successfuly moved file and deleted temp dir.')
    return 

