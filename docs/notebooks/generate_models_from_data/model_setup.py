import numpy as np
import os 
import picaso.justdoit as jdi
import picaso.analyze as lyz
import xarray as xr

#1) planet parameters pulled from https://docs.google.com/spreadsheets/d/1uxjt4ex24SyRrUFxP-kZ1-mQLk1saSEozcCS7fhd3Q4/gviz/tq?tqx=out:csv&gid=0
log_g = 4.149
metallicity = -0.25 #or 0.25
t_eff = 6550 #K
r_star = 1.583 #rsun
m_planet = 0.477 #mjup
r_planet = 1.932 #rjup

#2) define what opacities you want to run the grid run with
opacity = {
            '1':jdi.opannection( filename_db="/data2/picaso_dbs/R15000/all_opacities_0.3_15_R15000.db", wave_range=[0.3,15])
}

#3) point to grid location 
location = "/data2/models/WASP-17b/spec/cldfree/"
grid_name = 'cldfree'
fitter = lyz.GridFitter(grid_name,location, verbose=True, to_fit='transit_depth', save_chem=True)
fitter.prep_gridtrieval(grid_name)

#4) edit setup data function
data_dir = 'gridtrieval_files/ExoTiC-MIRI/'


# gets needed parameters
offset_pm = fitter.interp_params[grid_name]['offset_prior']
grid_parameters_unique = fitter.interp_params[grid_name]['grid_parameters_unique']
grid_points = fitter.interp_params[grid_name]['grid_parameters']
tree = lyz.cKDTree(grid_points)





def get_data(file, hst_file='', add_hst=False): 
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
    miri = xr.open_dataset(os.path.join(data_dir,file
                                               ))
    
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

    if add_hst:   
        hst = jdi.pd.read_csv(os.path.join(data_dir,hst_file), delim_whitespace=True)
        hst = hst.loc[((hst['wvl']>0.4) & (hst['wvl']<3) )]
        hst['wno'] = 1e4/hst['wvl']
        hst = hst.sort_values(by='wno').reset_index(drop=True)
        returns['hst']= ( hst['wno'].values, hst['depth'].values,  hst['depth_e'].values)
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
    cld_free = ','.join(grid_parameters_unique.keys())+',offset_hst,offset_jwst'
    cld_free_virga = ','.join(grid_parameters_unique.keys())+',offset_hst,offset_jwst,logkzz,fsed'
    #cld_free_virga_xrp = ','.join(grid_parameters_unique.keys())+',xrp,offset_hst,logkzz,fsed'

class guesses_set: 
    """
    Optional! 
    This is only used if you want to test your initial functions before running 
    the full retrievals. See script test.py
    """
    #completely random guesses just to make sure it runs
    cld_free=[grid_parameters_unique[i][0]
             for i in grid_parameters_unique.keys()] + [-.005,-.005]
    cld_free_virga=[grid_parameters_unique[i][0]
             for i in grid_parameters_unique.keys()] + [-.005,-.005,10,1]
    #cld_free_virga_xrp=[grid_parameters_unique[i][0]
             #for i in grid_parameters_unique.keys()] + [1,-.005,10,1]

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

    def cld_free_virga(cube,planet=planet_og):
        #1) get core parametrs
        final_goal = cube[0:len(grid_parameters_unique.keys())]
        off_hst = cube[-4]
        off_jwst = cube[-3]     
        kzz = 10**cube[-2]
        fsed = 10**cube[-1]  
        
        #2) from core parameters get PT profile and chemistry
        dd,indz = tree.query(final_goal,k=1)
        near = grid_points.iloc[indz]
        file = os.path.join(location,
            f"wasp-17btint{near['tint']}-rfacv{near['heat_redis']}-mh{near['mh']}-cto{near['cto']}.nc")
        
        ds = jdi.xr.load_dataset(file)
        df = {'pressure':ds.coords['pressure'].values}
        for i in [i for i in ds.data_vars.keys() if 'transit' not in i]:
            if i not in ['opd','ssa','asy']:
                #only get single coord pressure stuff
                if (len(ds.data_vars[i].values.shape)==1 &
                            ('pressure' in ds.data_vars[i].coords)):
                    df[i]=ds.data_vars[i].values        
        
        #mh = 10**float(near['mh'])
        
        for i in opacity.keys(): planet[i].atmosphere(df=jdi.pd.DataFrame(df))    
        
        #3) bonus parameters: add clouds 
        for i in opacity.keys(): planet[i].inputs['atmosphere']['profile']['kz'] = kzz
        for i in opacity.keys(): 
            planet[i].virga(['SiO2','Al2O3'], '/data/virga_dbs/virga_0,3_15_R300/', 
                 fsed=fsed, verbose=False,sig=1.2,
                                                mh=1)#, gas_mmr={'SiO2':1e-5}) full_output=True only exists for virga_3d
        
        x = []
        y = []
        
        #3) have to recompute the "spectrum" (not the climate!)
        for i in opacity.keys(): 
            out = planet[i].spectrum(opacity[i], calculation='transmission',full_output=True)
            x += list(out['wavenumber'])
            y += list(out['transit_depth'])
        
        combined = sorted(zip(x, y), key=lambda pair: pair[0])

        sorted_x = np.array([pair[0] for pair in combined])
        sorted_y = np.array([pair[1] for pair in combined])
        #adjust less than 2 micron for HST
        sorted_y[1e4/sorted_x<2] = sorted_y[1e4/sorted_x<2]+off_hst
        #adjust greater than 2 micron for JWST
        sorted_y[1e4/sorted_x>=2] = sorted_y[1e4/sorted_x>=2]+off_jwst                
        return sorted_x, sorted_y


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


    def cld_free_virga(cube):#,ndim, nparams):
        params = cube.copy()
        for i,key in enumerate(grid_parameters_unique): 
            minn = np.min(grid_parameters_unique[key]) 
            maxx = np.max(grid_parameters_unique[key]) 
            params[i] = minn + (maxx-minn)*params[i]
        
        #shift hst
        i+=1;params[i] = -0.01*params[i]
        #shift jwst
        i+=1;params[i] = -0.01*params[i]
        
        #logkzz
        i+=1;params[i] = 7 + 4*params[i]
        #log fsed 
        i+=1;params[i] = -1 + 2*params[i]
        return params