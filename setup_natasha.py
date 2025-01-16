import numpy as np
import os 
#os.environ['picaso_refdata'] = '/nobackupp17/nbatalh1/reference_data/picaso/reference'
#os.environ['PYSYN_CDBS'] ='/nobackupp17/nbatalh1/reference_data/grp/hst/cdbs'
import picaso.justdoit as jdi
import picaso.justplotit as jpi
import picaso.analyze as lyz
import virga.justdoit as vdi
pd=jdi.pd


filename_db = '/Users/nbatalh1/Documents/data/all_opacities/all_opacities_0.3_15_R15000.db'

#1) define what opacities you want to run the grid run with
opacity = {
           #'1':jdi.opannection( filename_db="/data2/picaso_dbs/R10000/all_opacities_0.3_5.3_R10000.db", wave_range=[0.4,2]),
           #'2':jdi.opannection( filename_db="/data2/picaso_dbs/R10000/all_opacities_5_14_R10000.db", wave_range=[5,14]),
            '3':jdi.opannection( filename_db=filename_db, wave_range=[4.5,15])
}

#2) point to grid location 
location = '/Users/nbatalh1/Documents/data/planets/HD-189733b/xarray/cld_free'
grid_name = 'cldfree'
fitter = lyz.GridFitter(grid_name,location, verbose=True, 
                        model_type='xarrays',
                        to_fit='temp_brightness',
                       save_chem=True)
fitter.prep_gridtrieval(grid_name, add_ptchem=True)
wavenumber = 1e4/fitter.wavelength[grid_name]

#5) define molecules you want to fit for. right now I am just grabbing everything from grid
mols = fitter.interp_params[grid_name]['square_chem_grid'].keys()


#4) get mie parameters for cloud you want 
mieff_dir = '/Users/nbatalh1/Documents/data/virga_0,3_15_R300/'
#mieff_dir = '/nobackupp17/nbatalh1/reference_data/virga_dbs/virga_0,3_15_R300'
qext, qscat, cos_qscat, nwave, radius, wave_in = vdi.get_mie('SiO2',directory=mieff_dir)



#5) get parameters from xarray file 
grid_parameters_unique = fitter.interp_params[grid_name]['grid_parameters_unique']
grid_points = fitter.interp_params[grid_name]['grid_parameters']

#planet/star parameters 
xr_usr = jdi.xr.load_dataset(fitter.list_of_files[grid_name][0])
log_g = eval(xr_usr.attrs['stellar_params'])['logg']
metallicity =eval(xr_usr.attrs['stellar_params'])['feh']
t_eff = eval(xr_usr.attrs['stellar_params'])['steff'] #K
r_star = eval(xr_usr.attrs['stellar_params'])['rs']['value'] #rsun
m_planet = eval(xr_usr.attrs['planet_params'])['mp']['value'] 
m_planet_unit = eval(xr_usr.attrs['planet_params'])['mp']['unit']
r_planet =eval(xr_usr.attrs['planet_params'])['rp']['value']  #rjup
r_planet_unit =eval(xr_usr.attrs['planet_params'])['rp']['unit']  #rjup

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
    for i in opacity.keys(): pl[i].approx(p_reference=1)
    for i in opacity.keys(): pl[i].gravity(mass=m_planet, mass_unit=jdi.u.Unit(m_planet_unit),
              radius=r_planet, radius_unit=jdi.u.Unit(r_planet_unit))
    for i in opacity.keys(): pl[i].phase_angle(0)
    return pl

planet_og = setup_planet()

"""
def get_data(data_file): 
    file = jdi.pd.read_csv(data_file,
                      delim_whitespace=True).sort_values(by='micron')
    micron =  file['micron']
    tbright = file['tbright']
    tbright_err = file['tbright_err']
    return micron, tbright, tbright_err
"""

def get_data(data_file):
    file = jdi.pd.read_csv(data_file,skiprows=1, header=None, names=['micron', 'width',
        'fp', 'fp_error','tbright','tbright_err'],
                      delim_whitespace=True).sort_values(by='micron')
    micron =  file['micron']
    tbright = file['tbright']
    tbright_err = file['tbright_err']
    return micron, tbright, tbright_err

class param_set:
    """
    This is purely for book keeping what parameters you have run in each retrieval.
    It helps if you keep variables uniform.
    THINGS TO CHECK:
    1) Make sure that the variables here match how you are unpacking your cube in the model_set class and prior_set
    2) Make sure that the variable names here match the function names in model_set and prior_set
    """
    cld_free = ','.join(grid_parameters_unique.keys())+',logf'
    #cld_free_virga = ','.join(grid_parameters_unique.keys())+',logcldbar,fsed,logndz,sigma,lograd'
    cld_free_virga = ','.join(grid_parameters_unique.keys())+',logf,logcldbar,fsed,logndz,sigma,lograd'

class guesses_set: 
    """
    Optional! 
    This is only used if you want to test your initial functions before running 
    the full retrievals. See script test.py
    """
    #completely random guesses just to make sure it runs
    cld_free=[grid_parameters_unique[i][0]
             for i in grid_parameters_unique.keys()]  + [-1]
    cld_free_virga=[grid_parameters_unique[i][0]
             for i in grid_parameters_unique.keys()] + [-1,-1,1,1,1,-5]

class model_set:
    """100e-6=1e-4
    This is your full model set. It will include all the functions you wnat to test
    for a particular data set.
    """
       
    def cld_free(cube): 
        final_goal = cube[0:len(grid_parameters_unique.keys())]
        logf = cube[-1]
        spectra_interp = lyz.custom_interp(final_goal, fitter, grid_name)
        return wavenumber, spectra_interp, logf

    def cld_free_virga(cube,planet=planet_og):
        final_goal = cube[0:len(grid_parameters_unique.keys())]
        logf = cube[-6]
        base_pressure = 10**cube[-5]
        fsed = 10**cube[-4]
        ndz = 10**cube[-3]
        sigma= cube[-2]
        lograd = cube[-1]
        
        
        pressure = fitter.pressure[grid_name][0]
        pressure_layer = np.sqrt(pressure[0:-1]*pressure[1:])
        nlayer = len(pressure_layer)
        
        #interpolate to get temp
        temp = lyz.custom_interp(final_goal,
              fitter,grid_name, to_interp='custom',
              array_to_interp=fitter.interp_params[grid_name]['square_temp_grid'])

        #interpolate to get chem
        df_chem = pd.DataFrame(dict(pressure=pressure,temperature=temp))
        for imol in mols: 
            df_chem[imol] = lyz.custom_interp(final_goal,
              fitter,grid_name, to_interp='custom',
              array_to_interp=fitter.interp_params[grid_name]['square_chem_grid'][imol])
        
        #slab cloud 
        scale_h = 10 #just arbitrary as this gets fit for via fsed and ndz 
        z = np.linspace(100,0,nlayer)
        
        logradius = np.log10(radius)
        dist = (1/(sigma * np.sqrt(2 * np.pi)) *
                       np.exp( - (logradius - lograd)**2 / (2 * sigma**2)))
        

        opd,w0,g0,wavenumber_grid=vdi.calc_optics_user_r_dist(wave_in, ndz ,radius, jdi.u.cm,
                                                              dist,
                                                              qext, qscat, cos_qscat)
        
        opd_h = pressure_layer*0+10
        opd_h[base_pressure<pressure_layer]=0
        opd_h[base_pressure>=pressure_layer]=opd_h[base_pressure>=pressure_layer]*np.exp(
                              -fsed*z[base_pressure>=pressure_layer]/scale_h)
        opd_h = opd_h/np.max(opd_h)
        
        df_cld = vdi.picaso_format_slab(base_pressure,opd, w0, g0, wavenumber_grid, pressure_layer, 
                                          p_decay=opd_h)

        for i in opacity.keys(): planet[i].atmosphere(df=df_chem)    
        for i in opacity.keys(): planet[i].clouds(df=df_cld.astype(float))
            
        
        x = []
        y = []
        
        for i in opacity.keys(): 
            out = planet[i].spectrum(opacity[i], calculation='thermal',full_output=True)
            temp_brightness = jpi.brightness_temperature(out, plot=False)
            x += list(out['wavenumber'])
            y += list(temp_brightness)
        
        combined = sorted(zip(x, y), key=lambda pair: pair[0])

        sorted_x = np.array([pair[0] for pair in combined])
        sorted_y = np.array([pair[1] for pair in combined])
        
        return sorted_x, sorted_y,logf

class prior_set:
    """
    Store all your priors. You should have the same exact function names in here as
    you do in model_set and param_set

    Make sure the order of the unpacked cube follows the unpacking in your model 
    set and in your parameter set. 
    #pymultinest: http://johannesbuchner.github.io/pymultinest-tutorial/example1.html
    """   
    def cld_free(cube):
        params = cube.copy()
        for i,key in enumerate(grid_parameters_unique): 
            minn = np.min(grid_parameters_unique[key]) 
            maxx = np.max(grid_parameters_unique[key]) 
            params[i] = minn + (maxx-minn)*params[i]
            
        #logf
        minn = -2.5
        maxx = 2.5
        i+=1;params[i] =  minn + (maxx-minn)*params[i]  
        
        return params

    def cld_free_virga(cube):
        params = cube.copy()
        for i,key in enumerate(grid_parameters_unique): 
            minn = np.min(grid_parameters_unique[key]) 
            maxx = np.max(grid_parameters_unique[key]) 
            params[i] = minn + (maxx-minn)*params[i]

        #logf
        minn = -2.5
        maxx = 2.5
        i+=1;params[i] =  minn + (maxx-minn)*params[i]        
        
        #log base_pressure
        minn = 1
        maxx = -4
        i+=1;params[i] = minn + (maxx-minn)*params[i]
        
        #log fsed
        minn = -1 
        maxx = 1
        i+=1;params[i] = minn + (maxx-minn)*params[i]
        
        #ndz
        minn = 1 
        maxx = 10
        i+=1;params[i] =  minn + (maxx-minn)*params[i]
        
        #sigma 
        minn = 0.5 
        maxx = 2.5
        i+=1;params[i] =  minn + (maxx-minn)*params[i]

        #loggradii 
        minn = -7
        maxx = -3
        i+=1;params[i] =  minn + (maxx-minn)*params[i]
        return params

