import numpy as np
import os 
import picaso.justdoit as jdi
import picaso.analyze as lyz


#1) planet parameters from google sheet
sheet_id = '1R3DlcC25MyfP97DNcbfsy1dNehu20BWal8ePOol9Ftg'
sheet_name = '0'
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&gid={sheet_name}"

#2) define what opacities you want to run the grid run with
opacity = {
            '3':jdi.opannection( filename_db="/data2/picaso_dbs/R15000/all_opacities_0.3_15_R15000.db", wave_range=[0.3,15])
}

#3) point to grid location 
location = "tutorial_test_1/W39b_climate.nc"
grid_name = 'cldfree'
fitter = lyz.GridFitter(grid_name,location, verbose=True, to_fit='fpfs_emission', save_chem=True)
fitter.prep_gridtrieval(grid_name)

#4) edit setup data function
data_dir = 'tutorial_test_1/ZENODO/TRANSMISSION_SPECTRA_DATA/EUREKA_REDUCTION.txt'


# gets needed parameters
grid_parameters_unique = fitter.interp_params[grid_name]['grid_parameters_unique']
grid_points = fitter.interp_params[grid_name]['grid_parameters']


#planet parameters 
success = jdi.pd.read_csv(url)
log_g = success.loc[0,'logg']
metallicity =success.loc[0,'feh']
t_eff = success.loc[0,'st_teff'] #K
r_star = success.loc[0,'rstar'] #rsun
m_planet = success.loc[0,'pl_mass'] #mjup
r_planet =success.loc[0,'pl_rad']  #rjup


def get_data(nir_name, miri_name): 
    """
    Create a function to process your data in any way you see fit.
    You could, for example, want to read in the data and rebin.
    
    Make sure that this is ordered in ascending WAVENUMBER!

    Returns
    -------
    To run a spectral retrieval you will need wavenumber, flux, and error. The
    exact format will depend on how you create your likelihood function, which is up to you!
    """
    wlgrid_center,rprs_data2,wlgrid_width, e_rprs2=[],[],[],[]

    if 'Gressier' in nir_name: 
        #nrs1 = pd.read_csv(os.path.join(data_dir, 'nir_name','nrs1.txt'), delim_whitespace=True,
        #                 header=None, names=['wave','ppm','p'])
        raise Exception('not sure what the data is')
    elif "Wake" in nir_name: 
        g395h = jdi.xr.load_dataset(os.path.join(data_dir, 'NIRSpec_G395H_Eclipse',nir_name,
                                                 'emission-spectrum-h26-g395h-r100ch4-h1-2aug23_v1-x_y.nc'))
                                                 #'emission-spectrum-h26-g395h-10pix-h1-2aug23_v1-x_y.nc'))

        wlgrid_center+=    list(g395h['central_wavelength'].values) 
        rprs_data2+=    list(g395h['eclipse_depth'].values)
        wlgrid_width+=    list(g395h['bin_half_width'].values)
        e_rprs2+=    list(g395h['eclipse_depth_error'].values)

    if 'EUREKA' in miri_name:
        file = os.path.join(data_dir, 'MIRI_LRS_Eclipse',miri_name,'current_versions',
                            'dv_hp26_miri_eclipse_0.50_bin_v10.txt')
        miri = jdi.pd.read_csv(file, delim_whitespace=True)
        
        wlgrid_center+=    list(miri['wavelength'].values) 
        rprs_data2+=    list(miri['ecl_depth'].values)
        wlgrid_width+=    list(miri['bin_width'].values)
        #e_rprs2+=    list(miri['eclipse_depth_err'].values)
        
    elif 'Cornell' in miri_name: 
        file = os.path.join(data_dir, 'MIRI_LRS_Eclipse',miri_name,
                            'HATP26b_miri_lrs_eclipse_ch14.nc')
        miri = jdi.xr.load_dataset(file)        
        wlgrid_center+=    list(miri['central_wavelength'].values) 
        rprs_data2+=    list(miri['eclipse_depth'].values)
        wlgrid_width+=    list(miri['bin_half_width'].values)
        e_rprs2+=    list(miri['eclipse_depth_error'].values)
        
    
    final = jdi.pd.DataFrame(dict(wlgrid_center=wlgrid_center,
                rprs_data2=rprs_data2,
                wlgrid_width=wlgrid_width,
                e_rprs2=e_rprs2))
    
    final['wno'] = 1e4/final['wlgrid_center']
    final = final.sort_values(by='wno').reset_index(drop=True)

    returns = {nir_name+miri_name : [final['wno'].values, 
             final['rprs_data2'].values  ,final['e_rprs2'].values]   }
    #make sure this dictionary is always {'name':(wno, ydata, edata)}
    return returns
    
def setup_planet():
    """
    First need to setup initial parameters. Usually these are fixed (anything we wont be including
    in the retrieval would go here).
    
    Returns
    -------
    Must return the full picaso class
    """
    pl = {i:jdi.inputs() for i in opacity.keys()}
    for i in opacity.keys(): pl[i].star(opacity[i], t_eff,metallicity,log_g,
                                        radius=r_star,
                                        database = 'phoenix',
                                        radius_unit = jdi.u.Unit('R_sun') )
    #i am putting these here but in a lot of cases, this should go into the likelihood function
    #for example when adjusting radius through xrp factor
    #or if fitting for planet mass, radius, or gravity
    for i in opacity.keys(): pl[i].approx(p_reference=1)
    for i in opacity.keys(): pl[i].gravity(mass=m_planet, mass_unit=jdi.u.Unit('M_jup'),
              radius=r_planet, radius_unit=jdi.u.Unit('R_jup'))
    for i in opacity.keys(): pl[i].phase_angle(0)
    return pl

planet_og = setup_planet()
class param_set:
    """
    This is purely for book keeping what parameters you have run in each retrieval.
    It helps if you keep variables uniform.
    THINGS TO CHECK:
    1) Make sure that the variables here match how you are unpacking your cube in the model_set class and prior_set
    2) Make sure that the variable names here match the function names in model_set and prior_set
    """
    #line='m,b' #simplest test model = blackbody: blackbody='Tiff'
    cld_free = ','.join(grid_parameters_unique.keys())

class guesses_set: 
    """
    Optional! 
    This is only used if you want to test your initial functions before running 
    the full retrievals. See script test.py
    """
    #completely random guesses just to make sure it runs
    cld_free=[grid_parameters_unique[i][0]
             for i in grid_parameters_unique.keys()]# + [-.005,-.005]

class model_set:
    """100e-6=1e-4
    This is your full model set. It will include all the functions you want to test
    for a particular data set.
    """     
    def cld_free(cube): 
        final_goal = cube[0:len(grid_parameters_unique.keys())]
        spectra_interp = lyz.custom_interp(final_goal, fitter, grid_name)
        micron = fitter.wavelength[grid_name]
        wno = 1e4/fitter.wavelength[grid_name] 
        return wno, spectra_interp

class prior_set:
    """
    Store all your priors. You should have the same exact function names in here as
    you do in model_set and param_set

    Make sure the order of the unpacked cube follows the unpacking in your model 
    set and in your parameter set. 
    #pymultinest: http://johannesbuchner.github.io/pymultinest-tutorial/example1.html
    """   
    def cld_free(cube):#,ndim, nparams):
        params = cube.copy()
        for i,key in enumerate(grid_parameters_unique): 
            minn = np.min(grid_parameters_unique[key]) 
            maxx = np.max(grid_parameters_unique[key]) 
            params[i] = minn + (maxx-minn)*params[i]
        return params

