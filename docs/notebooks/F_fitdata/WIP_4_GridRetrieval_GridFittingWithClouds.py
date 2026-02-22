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
# # Implemenation: Grid Fitting+Post-Processing using Bayesian Statistics w/ PICASO
#
# If this is your first crack at fitting parameters with PICASO, we strong encourage you to look at the first two retrieval tutorial. Here we will expand on that tutorial and you will learn:
#
# **What you will learn:**
#
# 1. How to load a grid and post-process clouds
#     - Instead of interpolating spectra like in the previous tutorial we are going to interpolate the temperature and chemistry
# 2. How to analyze grid fitting + clouds results
#
# **What you should know:**
#
# 1. `picaso`'s retrieval class structure (model_set, prior_set, param_set)
# 2. `picaso.retrieval.create_template` function to generate template scripts (covered in Retrieval Tutorial 1 & 2)
# 3. `picaso.analyze.prep_gridtrieval` function to simply generate and interpolate on a spectrum
# 4. Running `virga` cloud models `picaso` forward modeling (e.g. computing transmission spectra with `justdoit.inputs`)
#
# **What you will need:**
#
# We will build on the `grid` fitting tutorial 2, using Grant et al. 2024 data.
#
# 1. Download [WASP-17b grids here](www.doi.org/10.5281/zenodo.14681144) (and unpacked it is 436 Mb)
#
#

# %%
import numpy as np
import os
import picaso.justdoit as jdi
pd = jdi.pd
import picaso.analyze as lyz
import xarray as xr
import matplotlib.pyplot as plt


# %% [markdown]
# ## Step 1) Develop function to get data
#
# (same as tutorial 1 & 2)
#
# Let's create a function to pull our data where all we need to do is declare who the data is from and it pulls it for us automatically.
#
# Note: this format is only a recommendation and you can change any part of this to fit your needs

# %%
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


# %% [markdown]
# ## Step 2) Load Grid
#
# Same as tutorial 2!

# %% [markdown]
# Let's point towards the grid locations, create a grid fitter object for them, and prep them.
#
# If you are not familiar with `lyz.GridFitter` we encourate you to first become familiar with non-Bayesian grid fitting based on purely maximum chi-sq values. You can play around with this [Grid Search tutorial here](https://natashabatalha.github.io/picaso/notebooks/fitdata/GridSearch.html).
#
# The basic premise of `prep_gridtrieval`:
# - vets and transforms the grid to ensure it's square and interprelate-able
# - checks that there is a common pressure grid for the temperature

# %%
grid_location ='/data2/models/WASP-17b/spec/zenodo/v1'# should ultimately point to location of all .nc files
grid_name = 'cldfree' #for your own book-keeping
fitter = lyz.GridFitter(grid_name,grid_location, verbose=True, to_fit='transit_depth', save_chem=True)
fitter.prep_gridtrieval(grid_name)

# %% [markdown]
# ## Step 3) Setup `jdi.inputs` PICASO class
#
# This is going to get us ready to create a spectrum with PICASO. This also allows us to set anything that we plan to hold constant during the retrieval. For example, stellar parameters or planet parameters or phase angle might be things included as constant during a retrieval.

# %%
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

# %%
#let's point to what opacity database we want to use
#we will keep this in dictionary format for cases where we might need to use multiple
#opacity databases
opacity = {'db1':jdi.opannection(wave_range=[5,15],filename_db='/data2/picaso_dbs/R15000/all_opacities_0.3_15_R15000.db')}
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

# %% [markdown]
# ## Step 4) Param Set
#
# Let's add cloud parameters to our set. In this retrieval setup we are going to add a `virga` run to our model. If you are not familiar with `virga` cloud model we suggest you complete those forward modeling tutorials first. Let's fit for these two additional cloud parameters in addition to a radius factor to account for the unknown reference pressure.
#
# - xRp: Factor of the radius to account for the unknown reference pressure.
# - fsed : the sedimentation efficiency
# - logkzz : logarithm of the eddy diffusion (cm2/s)

# %%
#we can get the grid parameters directly from the module load, this will help with setting up our free parameters
grid_parameters_unique = fitter.interp_params[grid_name]['grid_parameters_unique']

class param_set:
    """
    This is much easier now since it is the grid parameters plus an offset to account for unknown reference pressure
    """
    grid_virga = list(grid_parameters_unique.keys())+['xrp','logfsed','logkzz']


# %% [markdown]
# ## Step 5) Guesses Set

# %% [markdown]
# In testing, it is very useful to check that it is grabbing the right parameters before doing a full analysis. This is available  for a sanity check if desired. Let's add cloud parameters to our guesses set.

# %%
class guesses_set:
    """
    For our guesses, we can verify things are working by just taking the first instance of each grid point
    """
    #completely random guesses just to make sure it runs
    grid_virga=[grid_parameters_unique[i][0]
             for i in grid_parameters_unique.keys()] + [1,-1,9]


# %% [markdown]
# ## Step 6) Model Set

# %% [markdown]
# Let's set up our grid interpolator using picaso's custom interp to get a pressure temperature profile and chemistry.

# %%
#since we are running virga we need to point to the Mieff files. If this is unfamiliar to you
#please see the virga tutorials first.

virga_mieff_files = '/data/virga_dbs/virga_0,3_15_R300/'
class model_set:
    """
    There are several different ways to interpolate onto a grid. Here, we use picaso's fast custom interpolator that
    uses a flex linear interpolator and can be used on any type of matrix grid.
    """
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
        mols = ['H2','He','CO2','CH4','CO','H2O']

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
        for i in opacity.keys(): planet[i].virga(['SiO2','Al2O3'],virga_mieff_files,
                 fsed=fsed, verbose=False,sig=1.2)

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


# %% [markdown]
# ## Step 6) Prior Set

# %% [markdown]
# Finally, we are storing all the priors for Ultranest to use.

# %%
class prior_set:
    """
    Store all your priors. You should have the same exact function names in here as
    you do in model_set and param_set

    Now we need to add priors for our new parameters, which include xrp, log fsed, and log kzz

    Make sure the order of the unpacked cube follows the unpacking in your model
    set and in your parameter set.
    #pymultinest: http://johannesbuchner.github.io/pymultinest-tutorial/example1.html
    """
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


# %% [markdown]
# ## Step 7) Loglikelihood
#
# No changes from our simple line example.

# %%
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


# %% [markdown]
# ## Step 7) Check models, likelihoods, priors!
#
# It looks like our priors are giving us an offset that is evenly distributed about the data. Looks good!

# %%
#we can easity grab all the important pieces now that they are neatly stored in a class structure
DATA_DICT = get_data()
PARAMS = getattr(param_set,'grid_virga')
MODEL = getattr(model_set,'grid_virga')
PRIOR = getattr(prior_set,'grid_virga')
GUESS = getattr(guesses_set,'grid_virga')

# %%
plt.figure()
#lets plot the data
for ikey in DATA_DICT.keys():
    plt.errorbar(x=DATA_DICT[ikey][0], y=DATA_DICT[ikey][1], yerr=DATA_DICT[ikey][2],color='black',
                 marker='o', ls=' ',label='Grant data')

ntests = 10 #lets do 10 random tests
for i in range(ntests):
    cube = np.random.uniform(size=len(PARAMS))
    params_evaluations = PRIOR(cube)
    loglike = loglikelihood(params_evaluations)
    x,y,off,err = MODEL(params_evaluations)
    plt.plot(x,y,label=str(i)+str(int(loglike)))

guessx,guessy,off,err = MODEL(GUESS)
guess_log = loglikelihood(GUESS)
plt.plot(guessx,guessy,color='black',label='guess '+ str(int(guess_log)))
plt.xlim([1e4/14,1e4/5])
plt.legend()

# %% [markdown]
# ## Step 8) Run the statistical sampler
#
# Now that we are running PICASO we recommend we move entirely to scripts. It is not recommended to run retrievals in notebook. The rest of the notebook assumes that the retrieval has been run and the output exists.

# %%
import ultranest

# %%
#sampler = ultranest.ReactiveNestedSampler(PARAMS, loglikelihood, PRIOR,resume=True,
#                                          log_dir='/data/test/ultranest/grid_virga')
#note if you wanted to turn thsi in the notebook and save the output you would add resume and log_dir above to save
#result = sampler.run()#adding a small number here just so we can test the results in the notebook

# %% [markdown]
# # Short cut to get grid fitting + `virga` retrieval template in script form
#
# Same as before but now we will add `grid_kwargs` to set some additional features in the `GridFitter`. Note this is only optional and you are always free to change the template manually.

# %%
import picaso.retrieval as pr

# %%
rtype='grid_virga' #first lets specify the retrieval type 'grid'
sript_name='run_test.py' #speciy a script name
sampler_output='/data/test/ultranest/grid_virga'

grid_location = '/data2/models/WASP-17b/spec/zenodo/v1' # should ultimately point to location of all .nc files
grid_name = 'cldfree' #for your own book-keeping
to_fit = 'transit_depth' #this is based on what you want to fit in the xarray files s

#opacity and chemistry
opacity_filename_db = '/data2/picaso_dbs/R15000/all_opacities_0.3_15_R15000.db'
molecules = ['H2','He','H2O','CO','CO2','CH4','H2O']

#cloud things
virga_mieff_files = '/data/virga_dbs/virga_0,3_15_R300/'
cloud_species = ['SiO2','Al2O3']

#new
grid_kwargs={'grid_location':grid_location,'grid_name':grid_name,'to_fit':to_fit,
            'opacity_filename_db':opacity_filename_db, 'molecules':molecules,
            'virga_mieff_dir':virga_mieff_files,'cloud_species':cloud_species}

pr.create_template(rtype,sript_name,sampler_output,grid_kwargs=grid_kwargs)

# %%
#need help remembering what keys are allowed?
pr.allowed_keys['grid_virga']

# %% [markdown]
# Open up `run_test.py` and modify what you need. We have marked key areas you might want to modify with "CHANGEME"
#
# Running with mpiexec with 5 cpu:
#
#     >> mpiexec -n 5 python -m mpi4py run_test.py

# %% [markdown]
# ## Some tricks and tips for once you have a built a script
#
# Now that you have a built script we can input it to grab our built models, priors, parameters set, etc.

# %%
import run_test as rt #YOUR OWN unique script

# %% [markdown]
# Now you can easily grab what you need directly and no need to have this in a notebook.

# %%
DATA_DICT = rt.get_data()
PARAMS = getattr(rt.param_set,'grid_virga')
MODEL = getattr(rt.model_set,'grid_virga')
PRIOR = getattr(rt.prior_set,'grid_virga')
GUESS = getattr(rt.guesses_set,'grid_virga')

# %% [markdown]
# We can also see use the fitter class to continue our analysis (which we will do in the following section)

# %% [markdown]
# # PICASO Analysis Tools from Saved Samples
#
# This part relies on having ran the above script for the run_test.py and having output in a directory defined by `sampler_output`
# ## Auto read ultranest results
#
# This function gives you back some of the most highly used output products:
#
# - `samples_equal`: equally weighted samples from the posterior
# - `max_logl` : the maximum loglikelihood (otherwise known as the Bayesian evidence)
# - `max_logl_point` : the set of parameters that is associated with the maximum loglikelihood
# - `med_intervals` : the 1 sigma median constraint intervals for each of the parameters of interest
# - `ultranest_out` : the raw ultranest output

# %%
results = pr.get_info(sampler_output, PARAMS)

# %%
results.keys()

# %% [markdown]
# ## Get spectra interval bands and evaluate max loglikelihood spectra
#
# - `bands_spectra`: 1,2,3 sigma bands for spectra (contains keys such as `1sig_lo` and `1sig_hi` for the spectra)
# - `max_logl_spectra`: the spectra evaluated at the max logl point
# - `max_logl_error_inflation` : if exists, the error inflation associated with the max logl point
# - `max_logl_offsets` : If exists, the offsets associated with the max logl point
# - `wavelength`: the wavelength grid in um

# %%
n_draws=200 #number of evaluations for which to
evaluations = pr.get_evaluations(results['samples_equal'], results['max_logl_point'], MODEL, n_draws,
                                 regrid=100.0,#spectral resolution to regrid to
                                 pressure_bands=['temperature','H2O','CO2','CO','CH4'])

# %%
evaluations.keys()

# %%
fig,axs = plt.subplots(1,1,figsize=(7,5))
color_scale =  pr.pals.Blues[3]
resolution=100
f=pr.plot_spectra_bands(evaluations,color_scale, ax=axs,R=resolution)
axs.set_title('Cld Free Model')#add other styles here
axs.set_xlim([5,14])

# %%
fig,axs = plt.subplots(1,2,figsize=(10,5))
color_scale =  pr.pals.Oranges[3]
f=pr.plot_pressure_bands(evaluations,color_scale, ax=axs)

# %% [markdown]
# ## Get Reduced Chi Squared Statistic of Max Logl spectrum Incl Offsets
#
# - `wavenumber`: new regridded wavenumber grid on data axis
# - `model` : model regridded on data axis
# - `datay` : data with offset included
# - `datae` : data error (no error inflation included )
# - `chisq_per_datapt` : chi squared per data point (DOF=len of data array)

# %%
chi_info = pr.get_chisq_max(evaluations, DATA_DICT)

# %%
chi_info.keys()

# %%
plt.plot(1e4/chi_info['wavenumber'], chi_info['model'],color='red',label='max logl model')
plt.errorbar(x=1e4/chi_info['wavenumber'], y=chi_info['datay'], yerr=chi_info['datae'],
             color='black',marker='o', ls=' ')


# %% [markdown]
# ## Get all bundled results in xarray format
#
# What does `data_output` do?
#
# 1. bundles all the median and sigma banded output data into an xarray
# 2. adds all the constraint intervals to your xarray in latex format
# 3. creates some default plots of banded spectra (banded chem if you have created it) and corner plots
# 4. creates a pickle of your equally weighted samples if you have requested it

# %%
filename = '/data/test/output/tagname'#name of file (without extension!!) where you want the output returned
return_samples=True # do you want a pickle file created of all the equally weighted samples?
spectrum_tag = 'transit_depth'#in the xarray what do you want the spectrum called?
spectrum_unit= 'cm**2/cm**2'#what are the units of your spectrum?
author = "NE Batalha"#who did the analysis? this is for the xarray
contact="natasha.e.batalha@nasa.gov"#how can you be contacted? this is for the xarray
model_description="cloud virga SiO2 & Al2O3 grid fit"#describe your model so people will know what it is
code = "PICASO,Virga,Ultranest" # what codes did you use for this analysis?

bxr=pr.data_output(evaluations, results, chi_info, filename,return_samples=True,
                     spectrum_tag=spectrum_tag,spectrum_unit=spectrum_unit,
                    author=author,contact=contact,
                    model_description=model_description,
                    code=code)#,
                    #round=[2, 1,1,5,3,2,2,1,2,2])

# %%
bxr

# %%
bxr.attrs['intervals_params']
#note if you do not like the round errors here you can adjust what these numbers are rounded to with
#round kwarg to data_output

# %% [markdown]
# ## Stylize Plots
#
# Our default plots were not great, let's beautify them a little

# %%
samples = results['samples_equal']

params =  results['param_names']

#create mapper for labels
pretty_labels={'cto':'C/O',
               'mh':'log M/H [xSolar]',
               'heat_redis':'heat redis.',
               'xrp':r'$x$R$_p$',
               'tint':r'T$_\mathrm{int}$',
               'logfsed':r'log f$_\mathrm{sed}$',
               'logkzz':r'log K$_\mathrm{zz}$'}

#create mapper for ranges ?
ranges={'cto':[0.25,1],
        'tint':[200,300],
        'xrp':[0.7,1.3],
        'heat_redis':[0.5,0.9],
        'mh':[1.3,1.7],
        'logfsed':[-1,1],
        'logkzz':[7,11]}

ints = eval(bxr.attrs['intervals_params'])#get pretty titles
intervals={i:ints[i] for i in results['param_names']}

f,a=pr.plot_pair(samples,params,pretty_labels=pretty_labels, ranges=ranges,figsize=(15,15),
              intervals=intervals)
#could make additional style change using a here
f.savefig('/data/output/plot_pair.png')

# %%
