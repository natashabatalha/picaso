##!/usr/bin/env python
# coding: utf-8

# Author: Natasha E Batalha 

# PICASO Retrieval Script Template for fitting a grid model to a spectrum

# CTRL-F "CHANGEME" for areas you should consider changing
# IF CHANGEME is commented out then it is just a suggestion 
# IF CHANGEME is not commented you must change before proceeding 

# Use this as a starting point for create a retrieval script 

# HOW to run with OpenMPI

# Running with mpiexec command line 

# mpiexec -n numprocs python -m mpi4py pyfile
# for example: (-n specifies number of jobs)
# mpiexec -n 5 python -m mpi4py run.py



import numpy as np
import os 
import picaso.justdoit as jdi
pd = jdi.pd
import picaso.analyze as lyz
import picaso.retrieval as prt
import xarray as xr
import ultranest



# ## Step 1) Develop function to get data

def get_data(): 
    """
    Create a function to process your data in any way you see fit.
    Here we are using the ExoTiC-MIRI data 
    https://zenodo.org/records/8360121/files/ExoTiC-MIRI.zip?download=1
    But no need to download it.

    Checklist
    ---------
    - your function returns a spectrum that will be in the same units as your picaso model (e.g. rp/rs^2, erg/s/cm/cm or other) 
    - your function retuns a spectrum that is in ascending order of wavenumber 
    - your function returns a dictionary where the key specifies the instrument name (in the event there are multiple)

    Returns
    -------
    dict: 
        dictionary key: wavenumber (ascneding), flux or transit depth, and error.
        e.g. {'MIRI LRS':[wavenumber, transit depth, transit depth error], 'NIRSpec G395H':[wavenumber, transit depth, transit depth error]}
    """
    dat = xr.load_dataset(jdi.w17_data())
    #build nice dataframe so we can easily 
    final = jdi.pd.DataFrame(dict(wlgrid_center=dat.coords['central_wavelength'].values,
                transit_depth=dat.data_vars['transit_depth'].values,
                transit_depth_error=dat.data_vars['transit_depth_error'].values))

    #create a wavenumber grid 
    final['wavenumber'] = 1e4/final['wlgrid_center']

    #always ensure we are ordered correctly
    final = final.sort_values(by='wavenumber').reset_index(drop=True)

    #return a nice dictionary with the info we need 
    returns = {'MIRI_LRS': [final['wavenumber'].values, 
             final['transit_depth'].values  ,final['transit_depth_error'].values]   }
    return returns


# ## Step 2) Load Grid & Set other Need Directories

grid_location = grid_locationCHANGEME # CHANGEME should ultimately point to location of all .nc files
grid_name = grid_nameCHANGEME #CHANGEME for your own book-keeping
to_fit = to_fitCHANGEME # CHANGEME to the quantity you are fitting based on what is in the xarray (options are usually transit_depth, flux)

fitter = lyz.GridFitter(grid_name,grid_location, verbose=True, 
    to_fit=to_fit, save_chem=True)
fitter.prep_gridtrieval(grid_name)

# Opacity filename 

#let's point to what opacity database we want to use
#we will keep this in dictionary format for cases where we might need to use multiple 
#opacity databases 
opacity_filename_db=opacity_filename_dbCHANGEME
#other popular keywords to add to opannection
#wave_range = [5,10]#microns

#[[grid_virga 
# Cloud optical properties
virga_mieff_files = virga_mieff_dirCHANGEME
#]]grid_virga

#[[grid_flexcloud
#all species whose optical properties you want to preload
virga_mieff_files = virga_mieff_dirCHANGEME
cloud_species = cloud_speciesCHANGEME
param_tools = prt.Parameterize(load_cld_optical=cloud_species,
        mieff_dir=virga_mieff_files)
#]]grid_flexcloud

# ## Step 3) Setup `jdi.inputs` PICASO class

#bonus! because we are loading a planet grid here we can pull properties 
#direct from one of the xarray files 
ds = lyz.xr.load_dataset(fitter.list_of_files[grid_name][0])
planet_params = eval(ds.attrs['planet_params'])
stellar_params = eval(ds.attrs['stellar_params'])

#planet properties
mass = planet_params['mp']['value']
mass_unit =  planet_params['mp']['unit'] #you should always double check what units are stored. here i know the xarra
radius =  planet_params['rp']['value']
radius_unit =  planet_params['rp']['unit']

#stellar properties 
database = stellar_params['database']
t_eff = stellar_params['steff']
metallicity = stellar_params['feh']
log_g = stellar_params['logg']
r_star = stellar_params['rs']['value']
r_star_unit = stellar_params['rs']['unit']


opacity = {'db1':jdi.opannection(filename_db= opacity_filename_db)}
def setup_planet():
    """
    First need to setup initial parameters. Usually these are fixed (anything we wont be including
    in the retrieval would go here).
    
    Returns
    -------
    Must return the full picaso class
    """
    pl = {i:jdi.inputs() for i in opacity.keys()}
    #define stellar inputs 
    for i in opacity.keys(): pl[i].star(opacity[i], t_eff,metallicity,log_g,
                                        radius=r_star,
                                        database = database,
                                        radius_unit = jdi.u.Unit(r_star_unit) )
    #define reference pressure for transmission 
    for i in opacity.keys(): pl[i].approx(p_reference=1)
    #define planet gravity 
    for i in opacity.keys(): pl[i].gravity(mass=mass, mass_unit=jdi.u.Unit(mass_unit),
              radius=radius, radius_unit=jdi.u.Unit(radius_unit))
    return pl
planet = setup_planet()


# ## Step 4) Param Set

#we can get the grid parameters directly from the module load, this will help with setting up our free parameters
grid_parameters_unique = fitter.interp_params[grid_name]['grid_parameters_unique']

class param_set:
    """
    This is much easier now since it is the grid parameters plus an offset to account for unknown reference pressure
    """
    #[[grid_virga
    grid_virga = list(grid_parameters_unique.keys())+['xrp','logfsed','logkzz']
    #]]grid_virga
    #[[grid_addchem
    grid_addchem = list(grid_parameters_unique.keys())+['xrp','logoverwrite_moleculeCHANGEME']
    #]]grid_addchem
    #[[grid_flexcloud
    grid_flexcloud = list(grid_parameters_unique.keys())+['xrp','logcldbar','logfsed','logndz','sigma','lograd']
    #]]grid_flexcloud

# ## Step 5) Guesses Set
class guesses_set: 
    """
    For our guesses, we can verify things are working by just taking the first instance of each grid point
    """
    #completely random guesses just to make sure it runs
    #[[grid_virga
    grid_virga=[grid_parameters_unique[i][0]
             for i in grid_parameters_unique.keys()] + [1,-1,9]
    #]]grid_virga
    #[[grid_addchem
    grid_addchem=[grid_parameters_unique[i][0]
             for i in grid_parameters_unique.keys()] + [1,-6]
    #]]grid_addchem
    #[[grid_flexcloud    
    grid_flexcloud=[grid_parameters_unique[i][0]
             for i in grid_parameters_unique.keys()] + [1,1,1,1,1,-5]
    #]]grid_flexcloud

# ## Step 6) Model Set
class model_set:
    """
    There are several different ways to interpolate onto a grid. Here, we use picaso's fast custom interpolator that 
    uses a flex linear interpolator and can be used on any type of matrix grid. 
    """
    #[[grid_virga  
    def grid_virga(cube, return_ptchem=False): 
        """Here we want to interpolate on temperature and chemistry instead of the spectra as we did last time. 
        We can still use custom interp to do so! 

        Parameters
        ----------
        cube : list
            List of parameters sampled from Bayesian sampler 
        return_ptchem : bool 
            True/False; Default=False. This is new! 
            Returns the planet class from the function *without* running a spectrum. 
            This will enable you to use the kwarg `pressure_bands` in the function `picaso.retrieval.get_evaluations` 
            The formalism is to return the entire planet class in either dictionary or class format. e.g. 
            {'db1':picaso.inputs class} or just picaso.inputs class 
        """
        # 1. Grab parameters from your cube 
        final_goal = cube[0:len(grid_parameters_unique.keys())]
        ## using this formalism with index is a bit more "fool-proof" than relying on yourself to get the index number correct 
        xrp = cube[param_set.grid_virga.index('xrp')]
        ## note here I am removing the log in front of fsed and kzz 
        fsed = 10**cube[param_set.grid_virga.index('logfsed')]
        kzz = 10**cube[param_set.grid_virga.index('logkzz')]

        # 2. Reset the mass and radius based on the radius scaling factor
        for i in opacity.keys(): planet[i].gravity(mass=mass, mass_unit=jdi.u.Unit(mass_unit),
              radius=xrp*radius, radius_unit=jdi.u.Unit(radius_unit)) 
        
        #3. Interpolate to get temperature
        temp = lyz.custom_interp(final_goal,
              fitter,grid_name, to_interp='custom',
              array_to_interp=fitter.interp_params[grid_name]['square_temp_grid'])

        #4. Interpolate to get chemistry for each molecule 
        ## lets be intentional about what molecules we include in the retrieval
        mols = moleculesCHANGEME

        ## setup a dataframe that we will use to add our chemistry to 
        pressure = fitter.pressure[grid_name][0]
        df_chem = pd.DataFrame(dict(pressure=pressure,temperature=temp))
        for imol in mols: 
            #still using the same custom interp function just need to replace square_temp with square_chem
            df_chem[imol] = lyz.custom_interp(final_goal,
              fitter,grid_name, to_interp='custom',
              array_to_interp=fitter.interp_params[grid_name]['square_chem_grid'][imol])

        #5. Add chemistry and PT profile to PICASO atmosphere class 
        for i in opacity.keys(): planet[i].atmosphere(df=jdi.pd.DataFrame(df_chem),verbose=False)    
        ## add in kzz which we are setting as a free parameter 
        for i in opacity.keys(): planet[i].inputs['atmosphere']['profile']['kz'] = kzz

        #6. Now we can introduce the virga cloud code 
        cloud_species = cloud_speciesCHANGEME
        #in this example we are fixing the distribution width
        sig=1.2 #CHANGEME
        for i in opacity.keys(): planet[i].virga(cloud_species,virga_mieff_files, 
                 fsed=fsed, verbose=False,sig=sig)

        if return_ptchem: return planet #KEEP THIS ! This will give us a short cut to test our atmosphere and cloud setup without running the spectrum 

        #7. Create a spectrum -- note here I am just looping through all opacity dictionaries for cases where we have 
        #more than one opacity file in the dictionary. 
        x = []
        y = []
        for i in opacity.keys(): 
            out = planet[i].spectrum(opacity[i], calculation='transmission',full_output=True)
            x += list(out['wavenumber'])
            y += list(out['transit_depth'])

        #8. Sort by wavenumber so we know we always pass the correct thing to the likelihood function 
        combined = sorted(zip(x, y), key=lambda pair: pair[0])
        wno = np.array([pair[0] for pair in combined])
        spectra = np.array([pair[1] for pair in combined])
        
        offset={} #no offset needed for this calculation
        error_inf={} # let's not add error inf 
        return wno, spectra,offset,error_inf    
    #]]grid_virga 
    #[[grid_addchem
    def grid_addchem(cube, return_ptchem=False): 
        """Here we want to interpolate on temperature and chemistry instead of the spectra as we did last time. 
        We can still use custom interp to do so! 

        Parameters
        ----------
        cube : list
            List of parameters sampled from Bayesian sampler 
        return_ptchem : bool 
            True/False; Default=False. This is new! 
            Returns the planet class from the function *without* running a spectrum. 
            This will enable you to use the kwarg `pressure_bands` in the function `picaso.retrieval.get_evaluations` 
            The formalism is to return the entire planet class in either dictionary or class format. e.g. 
            {'db1':picaso.inputs class} or just picaso.inputs class 
        """
        # 1. Grab parameters from your cube 
        final_goal = cube[0:len(grid_parameters_unique.keys())]
        ## using this formalism with index is a bit more "fool-proof" than relying on yourself to get the index number correct 
        xrp = cube[param_set.grid_addchem.index('xrp')]
        ## note here I am removing the log in front of log abundance
        overwrite_moleculeCHANGEME = 10**cube[param_set.grid_addchem.index('logoverwrite_moleculeCHANGEME')]

        # 2. Reset the mass and radius based on the radius scaling factor
        for i in opacity.keys(): planet[i].gravity(mass=mass, mass_unit=jdi.u.Unit(mass_unit),
              radius=xrp*radius, radius_unit=jdi.u.Unit(radius_unit))         
        
        #3. Interpolate to get temperature
        temp = lyz.custom_interp(final_goal,
              fitter,grid_name, to_interp='custom',
              array_to_interp=fitter.interp_params[grid_name]['square_temp_grid'])

        #4. Interpolate to get chemistry for each molecule 
        ## lets be intentional about what molecules we include in the retrieval
        mols = moleculesCHANGEME

        ## setup a dataframe that we will use to add our chemistry to 
        pressure = fitter.pressure[grid_name][0]
        df_chem = pd.DataFrame(dict(pressure=pressure,temperature=temp))
        for imol in mols: 
            #still using the same custom interp function just need to replace square_temp with square_chem
            df_chem[imol] = lyz.custom_interp(final_goal,
              fitter,grid_name, to_interp='custom',
              array_to_interp=fitter.interp_params[grid_name]['square_chem_grid'][imol])
        
        #add in our bonus chemistry 
        df_chem['overwrite_moleculeCHANGEME'] = overwrite_moleculeCHANGEME

        #5. Add chemistry and PT profile to PICASO atmosphere class 
        for i in opacity.keys(): planet[i].atmosphere(df=jdi.pd.DataFrame(df_chem),verbose=False)    

        if return_ptchem: return planet #KEEP THIS ! This will give us a short cut to test our atmosphere and cloud setup without running the spectrum 

        #6. Create a spectrum -- note here I am just looping through all opacity dictionaries for cases where we have 
        #more than one opacity file in the dictionary. 
        x = []
        y = []
        for i in opacity.keys(): 
            out = planet[i].spectrum(opacity[i], calculation='transmission',full_output=True)
            x += list(out['wavenumber'])
            y += list(out['transit_depth'])

        #7. Sort by wavenumber so we know we always pass the correct thing to the likelihood function 
        combined = sorted(zip(x, y), key=lambda pair: pair[0])
        wno = np.array([pair[0] for pair in combined])
        spectra = np.array([pair[1] for pair in combined])
        
        offset={} #no offset needed for this calculation
        error_inf={} # let's not add error inf 
        return wno, spectra,offset,error_inf
    #]]grid_addchem
    #[[grid_flexcloud 
    def grid_flexcloud(cube, return_ptchem=False): 
        """Here we want to interpolate on temperature and chemistry instead of the spectra as we did last time. 
        We can still use custom interp to do so! 

        Parameters
        ----------
        cube : list
            List of parameters sampled from Bayesian sampler 
        return_ptchem : bool 
            True/False; Default=False. This is new! 
            Returns the planet class from the function *without* running a spectrum. 
            This will enable you to use the kwarg `pressure_bands` in the function `picaso.retrieval.get_evaluations` 
            The formalism is to return the entire planet class in either dictionary or class format. e.g. 
            {'db1':picaso.inputs class} or just picaso.inputs class 
        """
        # 1. Grab parameters from your cube 

        final_goal = cube[0:len(grid_parameters_unique.keys())]
        ## using this formalism with index is a bit more "fool-proof" than relying on yourself to get the index number correct 
        xrp = cube[param_set.grid_flexcloud.index('xrp')]
        ## note here I am removing the log in front of fsed and kzz 
        base_pressure  = 10**cube[param_set.grid_flexcloud.index('logcldbar')]
        fsed = 10**cube[param_set.grid_flexcloud.index('logfsed')]
        ndz = 10**cube[param_set.grid_flexcloud.index('logndz')]
        sigma= cube[param_set.grid_flexcloud.index('sigma')]
        eff_radius = 10**cube[param_set.grid_flexcloud.index('lograd')]

        # 2. Reset the mass and radius based on the radius scaling factor

        for i in opacity.keys(): planet[i].gravity(mass=mass, mass_unit=jdi.u.Unit(mass_unit),
              radius=xrp*radius, radius_unit=jdi.u.Unit(radius_unit)) 

        #3. Grab the pressure grid from the GridFitter class

        pressure = fitter.pressure[grid_name][0]
        pressure_layer = np.sqrt(pressure[0:-1]*pressure[1:])
        nlayer = len(pressure_layer)
        
        #4. Interpolate to get temperature

        temp = lyz.custom_interp(final_goal,
              fitter,grid_name, to_interp='custom',
              array_to_interp=fitter.interp_params[grid_name]['square_temp_grid'])

        #5. Interpolate to get chemistry for each molecule 
        ## lets be intentional about what molecules we include in the retrieval

        mols = moleculesCHANGEME

        ## setup a dataframe that we will use to add our chemistry to 
        df_chem = pd.DataFrame(dict(pressure=pressure,temperature=temp))
        for imol in mols: 
            #still using the same custom interp function just need to replace square_temp with square_chem
            df_chem[imol] = lyz.custom_interp(final_goal,
              fitter,grid_name, to_interp='custom',
              array_to_interp=fitter.interp_params[grid_name]['square_chem_grid'][imol])

        #6. Add chemistry and PT profile to PICASO atmosphere class 

        for i in opacity.keys(): planet[i].atmosphere(df=pd.DataFrame(df_chem),verbose=False)    

        #7. Now we can introduce the parameterized flex cloud 
        # add class will setup the pressure and pressure layer which we only need to do once
        
        param_tools.add_class(planet[i])
        lognorm_kwargs = {'sigma':sigma, 'lograd[cm]':eff_radius}
        df_cld = param_tools.flex_cloud(cloud_species,base_pressure, ndz, fsed, 'lognorm',
                lognorm_kwargs=lognorm_kwargs )
        for i in opacity.keys(): planet[i].clouds(df=df_cld.astype(float))

        if return_ptchem: return planet #KEEP THIS ! This will give us a short cut to test our atmosphere and cloud setup without running the spectrum 

        #8. Create a spectrum -- note here I am just looping through all opacity dictionaries for cases where we have 
        #more than one opacity file in the dictionary. 
        x = []
        y = []
        for i in opacity.keys(): 
            out = planet[i].spectrum(opacity[i], calculation='transmission',full_output=True)
            x += list(out['wavenumber'])
            y += list(out['transit_depth'])

        #9. Sort by wavenumber so we know we always pass the correct thing to the likelihood function 
        combined = sorted(zip(x, y), key=lambda pair: pair[0])
        wno = np.array([pair[0] for pair in combined])
        spectra = np.array([pair[1] for pair in combined])
        
        offset={} #no offset needed for this calculation
        error_inf={} # let's not add error inf 
        return wno, spectra,offset,error_inf
    #]]grid_flexcloud

# ## Step 6) Prior Set

class prior_set:
    """
    Store all your priors. You should have the same exact function names in here as
    you do in model_set and param_set

    Now we need to add priors for our new parameters, which include xrp, log fsed, and log kzz 

    Make sure the order of the unpacked cube follows the unpacking in your model 
    set and in your parameter set. 
    #pymultinest: http://johannesbuchner.github.io/pymultinest-tutorial/example1.html
    """
    #[[grid_virga     
    def grid_virga(cube):
        params = cube.copy()
        for i,key in enumerate(grid_parameters_unique): 
            minn = np.min(grid_parameters_unique[key]) 
            maxx = np.max(grid_parameters_unique[key]) 
            params[i] = minn + (maxx-minn)*params[i]
        
        #xrp 
        minn=0.7;maxx=1.3
        i = param_set.grid_virga.index('xrp')
        params[i] = minn + (maxx-minn)*params[i]
        
        #log fsed 
        i = param_set.grid_virga.index('logfsed')
        minn=-1; maxx=1
        params[i] = minn + (maxx-minn)*params[i]
        
        #logkzz
        minn=7; maxx=11
        i = param_set.grid_virga.index('logkzz')
        params[i] = minn + (maxx-minn)*params[i]
       
        return params
    #]]grid_virga 
    #[[grid_addchem
    def grid_addchem(cube):
        params = cube.copy()
        for i,key in enumerate(grid_parameters_unique): 
            minn = np.min(grid_parameters_unique[key]) 
            maxx = np.max(grid_parameters_unique[key]) 
            params[i] = minn + (maxx-minn)*params[i]
        
        #xrp 
        minn=0.7;maxx=1.3
        i = param_set.grid_addchem.index('xrp')
        params[i] = minn + (maxx-minn)*params[i]
        
        #log abundance 
        i = param_set.grid_addchem.index('logoverwrite_moleculeCHANGEME')
        minn=-12; maxx=-4
        params[i] = minn + (maxx-minn)*params[i]
    #]]grid_addchem
    #[[grid_flexcloud   
    def grid_flexcloud(cube):
        params = cube.copy()
        for i,key in enumerate(grid_parameters_unique): 
            minn = np.min(grid_parameters_unique[key]) 
            maxx = np.max(grid_parameters_unique[key]) 
            params[i] = minn + (maxx-minn)*params[i]
        
        #xrp 
        minn=0.7;maxx=1.3
        i = param_set.grid_flexcloud.index('xrp')
        params[i] = minn + (maxx-minn)*params[i]   


        #log base_pressure
        minn = 1
        maxx = -4
        i = param_set.grid_flexcloud.index('logcldbar')
        params[i] = minn + (maxx-minn)*params[i]
        
        #log fsed
        minn = -1 
        maxx = 1
        i = param_set.grid_flexcloud.index('logfsed')
        params[i] = minn + (maxx-minn)*params[i]
        
        #ndz
        minn = 1 
        maxx = 10
        i = param_set.grid_flexcloud.index('logndz')
        params[i] =  minn + (maxx-minn)*params[i]
        
        #sigma 
        minn = 0.5 
        maxx = 2.5
        i = param_set.grid_flexcloud.index('sigma')
        params[i] =  minn + (maxx-minn)*params[i]

        #loggradii 
        minn = -7
        maxx = -3
        i = param_set.grid_flexcloud.index('lograd')
        params[i] =  minn + (maxx-minn)*params[i]
        return params
    #]]grid_flexcloud

# ## Step 7) Loglikelihood

def loglikelihood(cube):
    """
    Log_likelihood function that ultimately is given to the sampler
    Note if you keep to our same formats you will not have to change this code move 

    Tips
    ----
    - Remember how we put our data dict, error inflation, and offsets all in dictionary format? Now we can utilize that 
    functionality if we properly named them all with the right keys! 

    Checklist
    --------- 
    - ensure that error inflation and offsets are incorporated in the way that suits your problem 
    - note there are many different ways to incorporate error inflation! this is just one example 
    """
    #compute model spectra
    resultx,resulty,offset_all,err_inf_all = MODEL(cube) # we will define MODEL below 

    #initiate the four terms we willn eed for the likelihood
    ydat_all=[];ymod_all=[];sigma_all=[];extra_term_all=[];

    #loop through data (if multiple instruments, add offsets if present, add err inflation if present)
    for ikey in DATA_DICT.keys(): #we will also define DATA_DICT below
        xdata,ydata,edata = DATA_DICT[ikey]
        xbin_model , y_model = jdi.mean_regrid(resultx, resulty, newx=xdata)#remember we put everything already sorted on wavenumber

        #add offsets if they exist to the data
        offset = offset_all.get(ikey,0) #if offset for that instrument doesnt exist, return 0
        ydata = ydata+offset

        #add error inflation if they exist
        err_inf = err_inf_all.get(ikey,0) #if err inf term for that instrument doesnt exist, return 0
        sigma = edata**2 + (err_inf)**2 #there are multiple ways to do this, here just adding in an extra noise term
        if err_inf !=0: 
            #see formalism here for example https://emcee.readthedocs.io/en/stable/tutorials/line/#maximum-likelihood-estimation
            extra_term = np.log(2*np.pi*sigma)
        else: 
            extra_term=sigma*0

        ydat_all.append(ydata);ymod_all.append(y_model);sigma_all.append(sigma);extra_term_all.append(extra_term); 

    ymod_all = np.concatenate(ymod_all)    
    ydat_all = np.concatenate(ydat_all)    
    sigma_all = np.concatenate(sigma_all)  
    extra_term_all = np.concatenate(extra_term_all)

    #compute likelihood
    loglike = -0.5*np.sum((ydat_all-ymod_all)**2/sigma_all + extra_term_all)
    return loglike


if __name__ == "__main__":

    # CHANGEME based on what model you want to run 
    MODEL_TO_RUN = rtypeCHANGEME#options for templates: grid_virga, grid_addchem, grid_flexcloud

    ultranest_output= sampler_output_pathCHANGEME  # point to your own directory

    #we can easity grab all the important pieces now that they are neatly stored in a class structure 
    DATA_DICT = get_data()
    PARAMS = getattr(param_set,MODEL_TO_RUN)
    MODEL = getattr(model_set,MODEL_TO_RUN)
    PRIOR = getattr(prior_set,MODEL_TO_RUN)
    GUESS = getattr(guesses_set,MODEL_TO_RUN)


    sampler = ultranest.ReactiveNestedSampler(PARAMS, loglikelihood, PRIOR,
        log_dir = ultranest_output,resume=True)
    result = sampler.run()


