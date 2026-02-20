import pooch
import os
import shutil
import glob
import json
import requests


from .opacity_factory import get_all_metadata

try:
    from IPython.display import display, HTML
except ImportError:
    def display(data):
        """A dummy display function for non-IPython environments."""
        print(data)
    def HTML(data):
        """A dummy HTML class for non-IPython environments."""
        return data

def download_github_folder_api(save_to_dir,user="natashabatalha", repo="picaso", folder_path="reference"):
    """
    Downloads a folder from a GitHub repository using the GitHub API.

    Args:
        user (str): The GitHub username or organization name.
        repo (str): The repository name.
        folder_path (str): The path to the folder within the repository.
        save_to_dir (str): The local directory where files will be saved.
    """
    api_url = f"https://api.github.com/repos/{user}/{repo}/contents/{folder_path}"

    try:
        response = requests.get(api_url)
        response.raise_for_status()
        contents = response.json()

        # Create the destination directory if it doesn't exist
        # destination_path = os.path.join(save_to_dir, os.path.basename(folder_path))
        #os.makedirs(destination_path, exist_ok=True)
        #print(f"Downloading files to: {os.path.abspath(destination_path)}")
        os.makedirs(save_to_dir, exist_ok=True)
        print(f"Downloading files to: {os.path.abspath(save_to_dir)}")
        for item in contents:
            if item['type'] == 'file':
                file_url = item['download_url']
                file_name = item['name']
                #local_file_path = os.path.join(destination_path, file_name)
                local_file_path = os.path.join(save_to_dir, file_name)

                print(f"  Downloading {file_name}...")
                file_response = requests.get(file_url)
                file_response.raise_for_status()

                with open(local_file_path, 'wb') as f:
                    f.write(file_response.content)
            elif item['type'] == 'dir':
                # Recursively download subdirectories
                #new_folder_path = f"{folder_path}/{item['name']}"
                #new_save_to_dir = os.path.join(save_to_dir, os.path.basename(folder_path))
                new_folder_path = item['path']
                new_save_to_dir = os.path.join(save_to_dir, item['name'])                
                download_github_folder_api(new_save_to_dir, user=user, repo=repo, folder_path=new_folder_path)
                
    except requests.exceptions.RequestException as e:
        print(f"Error: Could not access the GitHub API. {e}")
    except KeyError:
        print("Error: The provided URL might not be a valid folder path or the repository is private.")


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
    "reference":{ 'default':{
        'url':{'reference':'https://github.com/natashabatalha/picaso/reference'},
        'description':'The required refernece data needed to run PICASO',
        'default_destination':"cwd"}
    },
    "tutorials":{'default':{
        'url':{'notebooks':'https://github.com/natashabatalha/picaso/docs/notebooks'},
        'description':'Include the full set of jupyter-notebooks for all PICASO functionality',
        'default_destination':"cwd"}
    },
    "resampled_opacity":{
        'default':{
           'url':{'opacities_0.3_15_R15000.db.tar.gz':'https://zenodo.org/records/14861730/files/opacities_0.3_15_R15000.db.tar.gz'},
            'description':'7.34 GB file resampled at R=15,000 from 0.3-15um. This is sufficient for doing R=100 JWST calculations and serves as a good default opacity database for exploration.',
            'default_destination':os.path.join(__refdata__, inputs['opacities']['files']['resampled-default']) 
        },
        'R60000,0.6-6um':{
            'url':{'all_opacities_0.6_6_R60000.db.tar.gz':'https://zenodo.org/records/6928501/files/all_opacities_0.6_6_R60000.db.tar.gz'},
            'description':'38.3 GB file resampled at R=60,000 from 0.6-6um. This is sufficient for doing moderate resolution JWST calculations.',
            'default_destination':inputs['opacities']['files']['resampled']
            },
        'R20000,4.8-15um':{
            'url':{'all_opacities_4.8_15_R20000.db.tar.gz':'https://zenodo.org/records/6928501/files/all_opacities_4.8_15_R20000.db.tar.gz'},
            'description':'7.0 GB file resampled at R=20,000 from 4.8-15um. This is sufficient for doing low resolution JWST calculations.',
            'default_destination':inputs['opacities']['files']['resampled']
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
            'description':'Virga refractive index and Mie files on standard 196 grid. Spherical particles only.',
            'default_destination':os.path.join(__refdata__, 'virga')
            }, 
            'aggregates':{
                'url':{'VIRGA_2_mieff_files.zip':'https://zenodo.org/records/16581692/files/VIRGA_2_mieff_files.zip'},
                'description':'Virga refractive index and Mie files on a modified 189 grid for the v2 aggregates capabilities. Spheres + aggregates.',
                'default_destination':os.path.join(__refdata__, 'virga_aggregates')                
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

            }#,
        #'diamondback':{
        #    'url':'',
        #    'description':'',
        #    },
        },
    'ck_tables':{
        'by-molecule':{
            'url':{
                'H2S_1460.hdf5':'https://zenodo.org/records/18644980/files/H2S_1460.hdf5',
                'MgH_1460.hdf5':'https://zenodo.org/records/18644980/files/MgH_1460.hdf5',
                'O2_1460.hdf5':'https://zenodo.org/records/18644980/files/O2_1460.hdf5',
                'FeH_1460.hdf5':'https://zenodo.org/records/18644980/files/FeH_1460.hdf5',
                'TiO_1460.hdf5':'https://zenodo.org/records/18644980/files/TiO_1460.hdf5',
                'SO2_1460.hdf5':'https://zenodo.org/records/18644980/files/SO2_1460.hdf5',
                'Fe_1460.hdf5':'https://zenodo.org/records/18644980/files/Fe_1460.hdf5',
                'C2H4_1460.hdf5':'https://zenodo.org/records/18644980/files/C2H4_1460.hdf5',
                'OCS_1460.hdf5':'https://zenodo.org/records/18644980/files/OCS_1460.hdf5',
                'C2H6_1460.hdf5':'https://zenodo.org/records/18644980/files/C2H6_1460.hdf5',
                'SiO_1460.hdf5':'https://zenodo.org/records/18644980/files/SiO_1460.hdf5',
                'C2H2_1460.hdf5':'https://zenodo.org/records/18644980/files/C2H2_1460.hdf5',
                'LiCl_1460.hdf5':'https://zenodo.org/records/18644980/files/LiCl_1460.hdf5',
                'CO2_1460.hdf5':'https://zenodo.org/records/18644980/files/CO2_1460.hdf5',
                'CrH_1460.hdf5':'https://zenodo.org/records/18644980/files/CrH_1460.hdf5',
                'Na_1460.hdf5':'https://zenodo.org/records/18644980/files/Na_1460.hdf5',
                'Rb_1460.hdf5':'https://zenodo.org/records/18644980/files/Rb_1460.hdf5',
                'H3+_1460.hdf5':'https://zenodo.org/records/18644980/files/H3+_1460.hdf5',
                'O3_1460.hdf5':'https://zenodo.org/records/18644980/files/O3_1460.hdf5',
                'H2O_1460.hdf5':'https://zenodo.org/records/18644980/files/H2O_1460.hdf5',
                'H2_1460.hdf5':'https://zenodo.org/records/18644980/files/H2_1460.hdf5',
                'VO_1460.hdf5':'https://zenodo.org/records/18644980/files/VO_1460.hdf5',
                'CO_1460.hdf5':'https://zenodo.org/records/18644980/files/CO_1460.hdf5',
                'LiF_1460.hdf5':'https://zenodo.org/records/18644980/files/LiF_1460.hdf5',
                'N2_1460.hdf5':'https://zenodo.org/records/18644980/files/N2_1460.hdf5',
                'CaH_1460.hdf5':'https://zenodo.org/records/18644980/files/CaH_1460.hdf5',
                'LiH_1460.hdf5':'https://zenodo.org/records/18644980/files/LiH_1460.hdf5',
                'K_1460.hdf5':'https://zenodo.org/records/18644980/files/K_1460.hdf5',
                'CH4_1460.hdf5':'https://zenodo.org/records/18644980/files/CH4_1460.hdf5',
                'Li_1460.hdf5':'https://zenodo.org/records/18644980/files/Li_1460.hdf5',
                'HCN_1460.hdf5':'https://zenodo.org/records/18644980/files/HCN_1460.hdf5',
                'TiH_1460.hdf5':'https://zenodo.org/records/18644980/files/TiH_1460.hdf5',
                'Cs_1460.hdf5':'https://zenodo.org/records/18644980/files/Cs_1460.hdf5',
                'NH3_1460.hdf5':'https://zenodo.org/records/18644980/files/NH3_1460.hdf5',
                'PH3_1460.hdf5':'https://zenodo.org/records/18644980/files/PH3_1460.hdf5',
                'AlH_1460.hdf5':'https://zenodo.org/records/18644980/files/AlH_1460.hdf5'
        },
            'description':'By molecule CK-Table on the 1460 grid with 661 wavenumber points.',
            'default_destination':os.path.join(__refdata__, inputs['opacities']['files']['ktable_by_molecule']) ,
        },
            
        'pre-weighted':{
            # 'url':{"sonora_2020_feh+130_co_100_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_100_noTiOVO.data.196.tar.gz", "sonora_2020_feh-030_co_025_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_025_noTiOVO.data.196.tar.gz", "sonora_2020_feh+030_co_150.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_150.data.196.tar.gz", "sonora_2020_feh-030_co_050.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_050.data.196.tar.gz", "sonora_2020_feh+130_co_200.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_200.data.196.tar.gz", "sonora_2020_feh+070_co_200.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_200.data.196.tar.gz", "sonora_2020_feh+130_co_200_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_200_noTiOVO.data.196.tar.gz", "sonora_2020_feh-030_co_025.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_025.data.196.tar.gz", "sonora_2020_feh+150_co_025.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_025.data.196.tar.gz", "sonora_2020_feh+150_co_050_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_050_noTiOVO.data.196.tar.gz", "sonora_2020_feh-030_co_200_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_200_noTiOVO.data.196.tar.gz", "sonora_2020_feh+070_co_025_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_025_noTiOVO.data.196.tar.gz", "sonora_2020_feh-070_co_100_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_100_noTiOVO.data.196.tar.gz", "sonora_2020_feh-030_co_150_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_150_noTiOVO.data.196.tar.gz", "sonora_2020_feh-070_co_250.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_250.data.196.tar.gz", "sonora_2020_feh+050_co_100_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_100_noTiOVO.data.196.tar.gz", "sonora_2020_feh+100_co_250_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_250_noTiOVO.data.196.tar.gz", "sonora_2020_feh+000_co_025.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_025.data.196.tar.gz", "sonora_2020_feh+050_co_200_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_200_noTiOVO.data.196.tar.gz", "sonora_2020_feh-100_co_050.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_050.data.196.tar.gz", "sonora_2020_feh+130_co_250_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_250_noTiOVO.data.196.tar.gz", "sonora_2020_feh+000_co_200_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_200_noTiOVO.data.196.tar.gz", "sonora_2020_feh+050_co_050.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_050.data.196.tar.gz", "sonora_2020_feh-050_co_025_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_025_noTiOVO.data.196.tar.gz", "sonora_2020_feh-030_co_250_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_250_noTiOVO.data.196.tar.gz", "sonora_2020_feh-050_co_150.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_150.data.196.tar.gz", "sonora_2020_feh+100_co_100_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_100_noTiOVO.data.196.tar.gz", "sonora_2020_feh+050_co_200.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_200.data.196.tar.gz", "sonora_2020_feh+100_co_025.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_025.data.196.tar.gz", "sonora_2020_feh-050_co_200_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_200_noTiOVO.data.196.tar.gz", "sonora_2020_feh-070_co_050_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_050_noTiOVO.data.196.tar.gz", "sonora_2020_feh-100_co_250.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_250.data.196.tar.gz", "sonora_2020_feh+200_co_200.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_200.data.196.tar.gz", "sonora_2020_feh-030_co_100.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_100.data.196.tar.gz", "sonora_2020_feh+100_co_050_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_050_noTiOVO.data.196.tar.gz", "sonora_2020_feh+000_co_050_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_050_noTiOVO.data.196.tar.gz", "sonora_2020_feh+030_co_025.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_025.data.196.tar.gz", "sonora_2020_feh+170_co_025_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_025_noTiOVO.data.196.tar.gz", "sonora_2020_feh-070_co_250_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_250_noTiOVO.data.196.tar.gz", "sonora_2020_feh-070_co_025.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_025.data.196.tar.gz", "sonora_2020_feh-100_co_150.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_150.data.196.tar.gz", "sonora_2020_feh+030_co_200_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_200_noTiOVO.data.196.tar.gz", "sonora_2020_feh+070_co_050.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_050.data.196.tar.gz", "sonora_2020_feh+200_co_100_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_100_noTiOVO.data.196.tar.gz", "sonora_2020_feh+030_co_100_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_100_noTiOVO.data.196.tar.gz", "sonora_2020_feh+200_co_200_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_200_noTiOVO.data.196.tar.gz", "sonora_2020_feh-100_co_025.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_025.data.196.tar.gz", "sonora_2020_feh+200_co_025.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_025.data.196.tar.gz", "sonora_2020_feh+130_co_050_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_050_noTiOVO.data.196.tar.gz", "sonora_2020_feh+100_co_025_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_025_noTiOVO.data.196.tar.gz", "sonora_2020_feh-100_co_200.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_200.data.196.tar.gz", "sonora_2020_feh+070_co_250.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_250.data.196.tar.gz", "sonora_2020_feh+150_co_025_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_025_noTiOVO.data.196.tar.gz", "sonora_2020_feh+050_co_050_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_050_noTiOVO.data.196.tar.gz", "sonora_2020_feh+050_co_150.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_150.data.196.tar.gz", "sonora_2020_feh+130_co_025_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_025_noTiOVO.data.196.tar.gz", "sonora_2020_feh+070_co_250_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_250_noTiOVO.data.196.tar.gz", "sonora_2020_feh-100_co_250_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_250_noTiOVO.data.196.tar.gz", "sonora_2020_feh-100_co_100.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_100.data.196.tar.gz", "sonora_2020_feh+070_co_100.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_100.data.196.tar.gz", "sonora_2020_feh+150_co_150.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_150.data.196.tar.gz", "sonora_2020_feh+150_co_200.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_200.data.196.tar.gz", "sonora_2020_feh-050_co_250_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_250_noTiOVO.data.196.tar.gz", "sonora_2020_feh-070_co_200_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_200_noTiOVO.data.196.tar.gz", "sonora_2020_feh+100_co_050.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_050.data.196.tar.gz", "sonora_2020_feh+100_co_200.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_200.data.196.tar.gz", "sonora_2020_feh+130_co_150_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_150_noTiOVO.data.196.tar.gz", "sonora_2020_feh-070_co_100.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_100.data.196.tar.gz", "sonora_2020_feh+170_co_250.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_250.data.196.tar.gz", "sonora_2020_feh+070_co_150.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_150.data.196.tar.gz", "sonora_2020_feh+030_co_250_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_250_noTiOVO.data.196.tar.gz", "sonora_2020_feh+070_co_100_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_100_noTiOVO.data.196.tar.gz", "sonora_2020_feh+130_co_025.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_025.data.196.tar.gz", "sonora_2020_feh-030_co_200.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_200.data.196.tar.gz", "sonora_2020_feh+200_co_150_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_150_noTiOVO.data.196.tar.gz", "sonora_2020_feh+170_co_200.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_200.data.196.tar.gz", "sonora_2020_feh-050_co_050.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_050.data.196.tar.gz", "sonora_2020_feh-070_co_150.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_150.data.196.tar.gz", "sonora_2020_feh+170_co_100.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_100.data.196.tar.gz", "sonora_2020_feh+130_co_050.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_050.data.196.tar.gz", "sonora_2020_feh+150_co_100_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_100_noTiOVO.data.196.tar.gz", "sonora_2020_feh+000_co_100.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_100.data.196.tar.gz", "sonora_2020_feh+100_co_150_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_150_noTiOVO.data.196.tar.gz", "sonora_2020_feh-070_co_200.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_200.data.196.tar.gz", "sonora_2020_feh+030_co_025_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_025_noTiOVO.data.196.tar.gz", "sonora_2020_feh+000_co_250_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_250_noTiOVO.data.196.tar.gz", "sonora_2020_feh+150_co_150_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_150_noTiOVO.data.196.tar.gz", "sonora_2020_feh+200_co_150.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_150.data.196.tar.gz", "sonora_2020_feh-100_co_150_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_150_noTiOVO.data.196.tar.gz", "sonora_2020_feh+170_co_200_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_200_noTiOVO.data.196.tar.gz", "sonora_2020_feh-050_co_200.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_200.data.196.tar.gz", "sonora_2020_feh+200_co_250_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_250_noTiOVO.data.196.tar.gz", "sonora_2020_feh+000_co_150_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_150_noTiOVO.data.196.tar.gz", "sonora_2020_feh-030_co_050_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_050_noTiOVO.data.196.tar.gz", "sonora_2020_feh+050_co_150_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_150_noTiOVO.data.196.tar.gz", "sonora_2020_feh-030_co_100_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_100_noTiOVO.data.196.tar.gz", "sonora_2020_feh+100_co_100.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_100.data.196.tar.gz", "sonora_2020_feh-050_co_050_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_050_noTiOVO.data.196.tar.gz", "sonora_2020_feh+200_co_100.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_100.data.196.tar.gz", "sonora_2020_feh-030_co_150.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_150.data.196.tar.gz", "sonora_2020_feh+030_co_150_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_150_noTiOVO.data.196.tar.gz", "sonora_2020_feh-100_co_025_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_025_noTiOVO.data.196.tar.gz", "sonora_2020_feh-050_co_100.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_100.data.196.tar.gz", "sonora_2020_feh+150_co_050.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_050.data.196.tar.gz", "sonora_2020_feh+200_co_050_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_050_noTiOVO.data.196.tar.gz", "sonora_2020_feh+070_co_200_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_200_noTiOVO.data.196.tar.gz", "sonora_2020_feh+170_co_150.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_150.data.196.tar.gz", "sonora_2020_feh-100_co_200_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_200_noTiOVO.data.196.tar.gz", "sonora_2020_feh-070_co_150_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_150_noTiOVO.data.196.tar.gz", "sonora_2020_feh-070_co_050.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_050.data.196.tar.gz", "sonora_2020_feh+030_co_200.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_200.data.196.tar.gz", "sonora_2020_feh+030_co_250.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_250.data.196.tar.gz", "sonora_2020_feh+050_co_025.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_025.data.196.tar.gz", "sonora_2020_feh+050_co_250.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_250.data.196.tar.gz", "sonora_2020_feh-050_co_100_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_100_noTiOVO.data.196.tar.gz", "sonora_2020_feh+170_co_150_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_150_noTiOVO.data.196.tar.gz", "sonora_2020_feh+170_co_250_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_250_noTiOVO.data.196.tar.gz", "sonora_2020_feh+050_co_025_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_025_noTiOVO.data.196.tar.gz", "sonora_2020_feh+050_co_250_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_250_noTiOVO.data.196.tar.gz", "sonora_2020_feh-050_co_150_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_150_noTiOVO.data.196.tar.gz", "sonora_2020_feh+100_co_250.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_250.data.196.tar.gz", "sonora_2020_feh+000_co_100_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_100_noTiOVO.data.196.tar.gz", "sonora_2020_feh+170_co_025.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_025.data.196.tar.gz", "sonora_2020_feh+150_co_200_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_200_noTiOVO.data.196.tar.gz", "sonora_2020_feh+170_co_050.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_050.data.196.tar.gz", "sonora_2020_feh+070_co_050_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_050_noTiOVO.data.196.tar.gz", "sonora_2020_feh+170_co_100_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_100_noTiOVO.data.196.tar.gz", "sonora_2020_feh+150_co_250.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_250.data.196.tar.gz", "sonora_2020_feh+000_co_250.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_250.data.196.tar.gz", "sonora_2020_feh+200_co_050.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_050.data.196.tar.gz", "sonora_2020_feh+200_co_025_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_025_noTiOVO.data.196.tar.gz", "sonora_2020_feh+070_co_025.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_025.data.196.tar.gz", "sonora_2020_feh-070_co_025_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-070_co_025_noTiOVO.data.196.tar.gz", "sonora_2020_feh-100_co_050_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_050_noTiOVO.data.196.tar.gz", "sonora_2020_feh-100_co_100_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-100_co_100_noTiOVO.data.196.tar.gz", "sonora_2020_feh+150_co_250_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_250_noTiOVO.data.196.tar.gz", "sonora_2020_feh+170_co_050_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+170_co_050_noTiOVO.data.196.tar.gz", "sonora_2020_feh-050_co_025.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_025.data.196.tar.gz", "sonora_2020_feh+130_co_100.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_100.data.196.tar.gz", "sonora_2020_feh+000_co_200.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_200.data.196.tar.gz", "sonora_2020_feh-050_co_250.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-050_co_250.data.196.tar.gz", "sonora_2020_feh+050_co_100.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+050_co_100.data.196.tar.gz", "sonora_2020_feh+150_co_100.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+150_co_100.data.196.tar.gz", "sonora_2020_feh+000_co_150.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_150.data.196.tar.gz", "sonora_2020_feh+030_co_100.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_100.data.196.tar.gz", "sonora_2020_feh+100_co_200_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_200_noTiOVO.data.196.tar.gz", "sonora_2020_feh+030_co_050.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_050.data.196.tar.gz", "sonora_2020_feh+130_co_150.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_150.data.196.tar.gz", "sonora_2020_feh+000_co_050.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_050.data.196.tar.gz", "sonora_2020_feh-030_co_250.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh-030_co_250.data.196.tar.gz", "sonora_2020_feh+000_co_025_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+000_co_025_noTiOVO.data.196.tar.gz", "sonora_2020_feh+130_co_250.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+130_co_250.data.196.tar.gz", "sonora_2020_feh+200_co_250.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_250.data.196.tar.gz", "sonora_2020_feh+070_co_150_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+070_co_150_noTiOVO.data.196.tar.gz", "sonora_2020_feh+100_co_150.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+100_co_150.data.196.tar.gz", "sonora_2020_feh+030_co_050_noTiOVO.data.196.tar.gz": "https://zenodo.org/records/7542068/files/sonora_2020_feh+030_co_050_noTiOVO.data.196.tar.gz"},
            'url':{"sonora_2121grid_NoTiOVO.zip": "https://zenodo.org/records/18636725/files/sonora_2121grid.zip",
                   "kcoeff_picaso4_noTiOVO": "https://zenodo.org/records/18636725/files/sonora_2121grid_NoTiOVO.zip"},
            'description':'196 CK Tables in new H5 file structure.',
            'default_destination':os.path.join(__refdata__, inputs['opacities']['files']['preweighted']) ,

            }
    },
    'picaso-lite':{
        'tutorial_sagan23':{
            'url':{"picaso-lite-reference.tar.gz": "https://zenodo.org/records/18708158/files/picaso-lite-reference.tar.gz"},
                'description':'picaso-lite reference folder that includes the bare minimum to run the sagan 2023 tutorial set.',
            'default_destination':os.path.join(__refdata__)
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
                
                #now check to see if opacities.db exists and is readable 
                messages = check_default_opacity(picaso_refdata,messages)
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

    
    #Other files 
    # Check for sonora grids 
    other_checks = {'sonora_grids':'fitting brown dwarf data or initial guesses to climate solutions', 
                    'virga':'cloud modeling with the virga code or creating flex clouds with real optical constants',
                    'opacities/preweighted':'climate modeling on chemical equilibrium grids',
                    'opacities/resortrebin':'climate modeling with disequilibrium or flex chemistry',
                    'opacities/resampled':'other wavelength solutions for resampled forward models'}
    for icheck, info in other_checks.items():
        fullname = os.path.join(picaso_refdata,icheck)
        if os.path.exists(fullname):
            grids_inside = [os.path.basename(i) for i in glob.glob(os.path.join(fullname, '*')) if os.path.basename(i) != 'readme']
            if len(grids_inside)>0:
                messages.append(('info', f'{icheck} folders found in reference directory:'))
                messages.append(('list',grids_inside))
            else: 
                messages.append(('warning', f'{icheck} folder found but they are empty. This data product is used for {info}. You might have this data product outside the default path. That is okay.'))

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


def check_default_opacity(picaso_refdata,messages): 
    default_resampled = glob.glob(os.path.join(picaso_refdata,'opacities','opacities*.db'))
    if len(default_resampled)==1: 
        default_resampled = default_resampled[0]
        extra_text = ''
    elif len(default_resampled)>1:
        default_resampled = default_resampled[0]
        extra_text = ' Note, found multiple opacities*.db filenames. PICASO has just picked the alphabetically first one. '    
    else: 
        messages.append(('error', f'Resampled opacity file has not been set, which is usually required by the code. The file should live here: <code>{default_resampled}</code>. You can use get_data function to help you download or read the installation docs to do it manually.'))
        return messages
    
    if os.path.exists(default_resampled):
        try: 
            allmeta = get_all_metadata(default_resampled)
            reformat_meta = [f'{i}: {j}' for i,j in allmeta]
        except: 
            reformat_meta = 0
        if isinstance(reformat_meta,list):
            messages.append(('success','Resampled opacity default file has been set.'+extra_text))
            messages.append(('list',reformat_meta))
        else:
            messages.append(('error', f'Resampled opacity file has been set to <code>{default_resampled}</code> but I cannot read the metadata. Please redownload and/or check the file has not been corrupted.'+extra_text))
        
    else: 
        messages.append(('error', f'Resampled opacity file has not been set, which is usually required by the code. The file should live here: <code>{default_resampled}</code>. You can use get_data function to help you download or read the installation docs to do it manually.'))
    return messages
    
def get_reference(path_to_picaso_refdata):
    """
    Direct wrapper function on get data to just download the user reference data

    Users must already have their environment variable set to use this. 

    Inputs
    ------
    path_to_picaso_refdata : str 
        path to picaso ref data 
    """
    get_data(category_download='reference',target_download='default', final_destination_dir=path_to_picaso_refdata)


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
            if default=='cwd':default = os.getcwd()
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

        if 'github.com' in iurl: 
            print("Downloading from github")
            download_github_folder_api(final_destination_dir, folder_path= iurl.split('https://github.com/natashabatalha/picaso/')[-1])
        else: 
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

