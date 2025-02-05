import numpy as np
import os 
import picaso.justdoit as jdi
import picaso.analyze as lyz
import xarray as xr

#1) planet parameters pulled from https://iopscience.iop.org/article/10.3847/2041-8213/acfc3b#apjlacfc3bt1
log_g = 4.149
metallicity = -0.25 #or 0.25
t_eff = 6550 #K
r_star = 0.8950 #rsun http://www.exoplanetkyoto.org/exohtml/WASP-39.html
m_planet = 0.477 #mjup
r_planet = 1.932 #rjup

#2) define what opacities you want to run the grid run with
opacity = {
            '3':jdi.opannection( filename_db="/data2/picaso_dbs/all_opacities_0.3_1_R20000.db", wave_range=[0.3,15])
}

#3) point to grid location 
location = "/data2/models/WASP-17b/spec/cldfree" #change to wasp 17b
grid_name = 'cldfree'
fitter = lyz.GridFitter(grid_name,location, verbose=True, to_fit='transit_depth', save_chem=True)
fitter.prep_gridtrieval(grid_name)

#4) edit setup data function
data_dir = 'gridtrieval_files/ExoTiC-MIRI/transmission_spectrum_vfinal_bin0.5_utc20230606_123544.nc'


# gets needed parameters
grid_parameters_unique = fitter.interp_params[grid_name]['grid_parameters_unique']
grid_points = fitter.interp_params[grid_name]['grid_parameters']





def get_data(): 
    """
    Create a function to process your data in any way you see fit.
    You could, for example, want to read in the data and rebin.
    
    Make sure that this is ordered in ascending WAVENUMBER!

    Returns
    -------
    To run a spectral retrieval you will need wavenumber, flux, and error. The
    exact format will depend on how you create your likelihood function, which is up to you!
    """
    wlgrid_center,fpfs,wlgrid_half_width,e_fpfs=[],[],[],[]
    miri = xr.open_dataset(data_dir)
    
    wlgrid_center+=    list(miri['central_wavelength'].values) 
    fpfs+=    list(miri['transit_depth'].values)
    wlgrid_half_width+=    list(miri['bin_half_width'].values)
    e_fpfs+=    list(miri['transit_depth_error'].values)
    
    final = jdi.pd.DataFrame(dict(wlgrid_center=wlgrid_center,
                fpfs=fpfs,
                wlgrid_half_width=wlgrid_half_width,
                e_fpfs=e_fpfs))
    
    final['wno'] = 1e4/final['wlgrid_center']
    final = final.sort_values(by='wno').reset_index(drop=True)

    returns = {'Eureka!' : [final['wno'].values, 
             final['fpfs'].values,final['e_fpfs'].values]}
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

