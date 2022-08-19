import numpy as np
import pandas as pd
import xarray as xr
import json
import matplotlib.pyplot as plt
import os
from astropy.convolution import convolve, Box1DKernel, Gaussian1DKernel
from scipy import optimize
import glob

from matplotlib.ticker import StrMethodFormatter

import virga.justdoit as vj
from .justdoit import inputs, opannection, mean_regrid, u



class GridFitter(): 
    """
    Top level grid fitter

    Currently our GridFitter has these requirements for xarray models: 

    Required `coords`: 
    - wavelength
    - pressure 

    Required `data_vars`: 
    - transit_depth

    Parameters
    ----------
    grid_name : str 
        Grid name so that users can keep track of inputs 

    model_dir : str 
        Location of model grid. Should be a directory that points to 
        several files in the PICASO xarray format.
    """
    def __init__(self, grid_name, model_dir, grid_dimensions=False, verbose=True):
        self.verbose=verbose
        
        self.grids = []
        self.list_of_files = {}
        self.grid_params = {}
        self.overview = {}
        self.wavelength={}
        self.temperature={}
        self.pressure={}
        self.spectra={}
        
        #adds first grid
        self.add_grid(grid_name, model_dir)

    def find_grid(self, grid_name, model_dir):
        """
        Makes sure the grid exists with proper nc files. Then, adds the file directory to self.list_of_files

        """
        if not os.path.isdir(model_dir): 
            raise Exception(f'Path to models entered does not exist: {model_dir}')
        else: 
            self.list_of_files[grid_name] = glob.glob(os.path.join(model_dir,"*.nc"))
            nfiles = len(self.list_of_files[grid_name])
            if nfiles<=1:
                raise Exception("Oops! It looks like you only have 1 or less files with the extension '.nc'") 
            else: 
                if self.verbose: print(f'Total number of models in grid is {nfiles}')

    def add_grid(self,grid_name,model_dir):
        #loads in grid info
        self.grids += [grid_name]
        self.find_grid(grid_name, model_dir)
        self.load_grid_params(grid_name)

    def add_data(self, data_name, wlgrid_center,wlgrid_width,y_data,e_data): 
        """
        Adds data to class 

        Parameters
        ----------
        data_name : str 
            Create a distinguisher for the dataset to test 
        wlgrid_center : array 
            array of wavelength centers 
        wlgrid_width : array 
            array of wavelength bins 
        y_data : array 
            data to be comapred to spectrum. CautioN!! make sure that y_data and material pulled to spectra are the same units. 
        e_data : array 
            measurement error associated y_data 
        """        
        self.data =  getattr(self, 'data',{data_name: []})
        self.data[data_name] = {'wlgrid_center': wlgrid_center,
                                'wlgrid_width':wlgrid_width,
                                 'y_data':y_data,
                                 'e_data':e_data}

    def load_grid_params(self,grid_name):
        """
        This will read the grid parameters and set the array of parameters 

        Parameters 
        ----------
        grid_name : str 
            Name of grid for bookkeeping

        Returns
        -------
        None 
            Creates self.overview, and self.grid_params
        """
        possible_params = {'planet_params': ['rp','mp','tint', 'heat_redis','p_reference','logkzz','mh','cto','p_quench','rainout'],
                           'stellar_params' : ['rs','logg','steff','feh','ms'],
                           'cld_params': ['opd','ssa','asy','p_cloud','haze_eff','fsed']}

        #define possible grid parameters
        self.grid_params[grid_name] = {i:{j:np.array([]) for j in possible_params[i]} for i in possible_params.keys()}
        #define possible grid parameters
        self.overview[grid_name] = {i:{j:np.array([]) for j in possible_params[i]} for i in possible_params.keys()}


        #how many possible files 
        number_files = len(self.list_of_files[grid_name])

        #loop through grid files to get parameters
        for ct, filename in enumerate(self.list_of_files[grid_name]):

            ds = xr.open_dataset(filename)

            if ct == 0:
                nwave = len(ds['wavelength'].values)
                npress = len(ds['pressure'].values)
                
                spectra_grid = np.zeros(shape=(number_files,nwave))
                temperature_grid = np.zeros(shape=(number_files,npress))
                pressure_grid = np.zeros(shape=(number_files,npress))
                wavelength = ds['wavelength'].values
            
            #start filling out grid parameters
            #seems like we need to save these?????
            temperature_grid[ct,:] = ds['temperature'].values[:]
            pressure_grid[ct,:] = ds['pressure'].values[:]
            spectra_grid[ct,:] = ds['transit_depth'].values[:] 


            # Read all the paramaters in the Xarray so that User can gain insight into the 
            # grid parameters
            for iattr in possible_params.keys():#loops through e.g. planet_params, stellar_params,  
                if iattr in ds.attrs:
                    attr_dict = json.loads(ds.attrs[iattr])
                    for ikey in possible_params[iattr]:

                        self.grid_params[grid_name][iattr][ikey] = np.append(
                                                        self.grid_params[grid_name][iattr][ikey],
                                                        _get_xarray_attr(attr_dict, ikey))

        #now count how many are in each and pop 
        for iattr in possible_params.keys():#loops through e.g. planet_params, stellar_params,  
            if iattr in ds.attrs:
                for ikey in possible_params[iattr]:
                    uni_values = np.unique(self.grid_params[grid_name][iattr][ikey])
                    if (len(uni_values)==1): 
                        if ('not specified' in str(uni_values[0])):
                            self.overview[grid_name][iattr][ikey] = 'Not specified in attrs.'
                            self.grid_params[grid_name][iattr].pop(ikey)
                        elif ('None' in str(uni_values[0])):
                            self.overview[grid_name][iattr][ikey] = f'Not used in grid.'
                            self.grid_params[grid_name][iattr].pop(ikey)  
                        else: 
                            self.overview[grid_name][iattr][ikey] = uni_values[0]
                            self.grid_params[grid_name][iattr].pop(ikey)                            
                    else: 
                        self.overview[grid_name][iattr][ikey] = uni_values
    
            else:
                #e.g. if no stellar_params were included for a brown dwarf grid
                self.overview[grid_name].pop(iattr)

        for iattr in possible_params.keys():
            if len(self.grid_params[grid_name][iattr].keys())==0: 
                self.grid_params[grid_name].pop(iattr)

        cnt_params = 0
        for iattr in self.overview[grid_name].keys():#loops through e.g. planet_params, stellar_params,
            for ikey in   self.overview[grid_name][iattr]:
                if isinstance(self.overview[grid_name][iattr][ikey],np.ndarray):
                    cnt_params += 1
                    if self.verbose:
                        print(f'For {ikey} in {iattr} grid is: {self.overview[grid_name][iattr][ikey]}')

        self.overview[grid_name]['num_params'] = cnt_params

        #lastly save wavelength, temperature, spectra 
        self.wavelength[grid_name] = wavelength
        self.pressure[grid_name] = pressure_grid
        self.temperature[grid_name] = temperature_grid
        self.spectra[grid_name] = spectra_grid

    def fit_all(self):
        for i in self.grids:
            for j in self.data.keys() :
                self.fit_grid(i, j)

    def fit_grid(self,grid_name, data_name, offset=True):
        """
        Fits grids given model and data. Retrieves posteriors of fit parameters.

        Parameters
        ----------
        grid_name : str 
            grid name that was specified in GridFiter or add_grid
        data_name : str 
            data name that was specified before in add_data
        offset : bool 
            Fit for an offset (e.g. in transit spectra)

        To Dos
        ------
        - make general to fpfs_thermal  
        """
        #number of models 
        nmodels = len(self.list_of_files[grid_name])
        

        wlgrid_center = self.data[data_name]['wlgrid_center']
        y_data = self.data[data_name]['y_data']
        e_data = self.data[data_name]['e_data']

        #get chi_sqrs if it already exists 
        self.chi_sqs =  getattr(self, 'chi_sqs',{grid_name: {data_name:np.zeros(shape=(nmodels))}})
        #get best fit dicts if it already exists 
        self.best_fits =  getattr(self, 'best_fits',{grid_name:{data_name:np.zeros(shape=(nmodels,len(wlgrid_center)))}})
        #get rank order  
        self.rank =  getattr(self, 'rank',{grid_name:{data_name:np.zeros(shape=(nmodels))}})
        
        #get posetiors
        self.posteriors =  getattr(self, 'posteriors',{grid_name:{data_name:{}}})


        #make sure nothing exiting is overwritten 
        self.chi_sqs[grid_name] = self.chi_sqs.get(grid_name, {data_name:np.zeros(shape=(nmodels))})
        self.best_fits[grid_name] = self.best_fits.get(grid_name, {data_name:np.zeros(shape=(nmodels,len(wlgrid_center)))})
        self.rank[grid_name] = self.rank.get(grid_name, {data_name:np.zeros(shape=(nmodels))})
        self.posteriors[grid_name] = self.posteriors.get(grid_name, {data_name:{}})

        #make sure nothing existing is overwritten 
        self.chi_sqs[grid_name][data_name] = self.chi_sqs[grid_name].get(data_name, np.zeros(shape=(nmodels)))
        self.best_fits[grid_name][data_name]  = self.best_fits[grid_name].get(data_name,np.zeros(shape=(nmodels,len(wlgrid_center))))
        self.rank[grid_name][data_name]  = self.rank[grid_name].get(data_name,np.zeros(shape=(nmodels)))
        self.posteriors[grid_name][data_name]  = self.posteriors[grid_name].get(data_name,{})

        if offset: 

            self.offsets =  getattr(self, 'offsets',{grid_name:{data_name:np.zeros(nmodels) }})
            self.offsets[grid_name] = self.offsets.get(grid_name, {data_name:np.zeros(shape=(nmodels))})
            self.offsets[grid_name][data_name] = self.offsets.get(data_name,np.zeros(shape=(nmodels)))
            self.overview[grid_name]['num_params'] = self.overview[grid_name]['num_params'] + 1

        numparams = self.overview[grid_name]['num_params']

        def shift_spectrum(waves,shift):
            return flux_in_bin+shift

        #can be parallelized 
        for index in range(nmodels):
            xw , flux_in_bin = mean_regrid(self.wavelength[grid_name],self.spectra[grid_name][index,:],newx= wlgrid_center)

            if offset: 
                popt, pcov = optimize.curve_fit(shift_spectrum, wlgrid_center, y_data,p0=[-0.001])
                shift = popt[0]
            else: 
                shift = 0 

            self.chi_sqs[grid_name][data_name][index]= chi_squared(y_data,e_data,flux_in_bin+shift,numparams)

            self.best_fits[grid_name][data_name][index,:] = flux_in_bin+shift
            if offset: self.offsets[grid_name][data_name][index] = popt[0]

        self.rank[grid_name][data_name] = self.chi_sqs[grid_name][data_name].argsort()

        #finally compute the posteriors 
        for iattr in self.grid_params[grid_name].keys(): 
            for ikey in self.grid_params[grid_name][iattr].keys():
                self.posteriors[grid_name][data_name][ikey] = self.get_posteriors(grid_name, data_name, ikey)
    
    def print_best_fit(self, grid_name, data_name, verbose=True): 
        """
        Print out table of best fit parameters 

        Parameters
        ----------
        grid_name : str 
            grid name or string of single grid name to plot 
        data_name : str 
            data name or string of single 
        """
        best_fits = {}
        for iattr in self.grid_params[grid_name].keys(): 
            for ikey in self.grid_params[grid_name][iattr].keys():
                single_best_fit = self.grid_params[grid_name][iattr][ikey][self.rank[grid_name][data_name]][0]
                if verbose: print(f'{ikey}={single_best_fit}')
                best_fits[ikey] = single_best_fit
        return best_fits

    def plot_best_fit(self, grid_names, data_names): 
        """
        
        Parameters
        ----------
        grid_names : list, str 
            List of grid names or string of single grid name to plot 
        data_names : list, str 
            List of data names or string of single 
        """
        if isinstance(grid_names ,str):grid_names=[grid_names]
        if isinstance(data_names ,str):data_names=[data_names]

        x='''
        AA
        ..
        BB
        '''
        fig = plt.figure(figsize=(18,10))
        plt.style.use('seaborn-paper')
        plt.rcParams['figure.figsize'] = [7, 4]           # Figure dimensions
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['image.aspect'] = 1.2                       # Aspect ratio (the CCD is quite long!!!)
        plt.rcParams['lines.linewidth'] = 1
        plt.rcParams['lines.markersize'] = 3
        plt.rcParams['lines.markeredgewidth'] = 0
        
        cmap = plt.cm.magma
        #cmap.set_bad('k',1.)
        
        plt.rcParams['image.cmap'] = 'magma'                   # Colormap.
        plt.rcParams['image.interpolation'] = None
        plt.rcParams['image.origin'] = 'lower'
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = 'DejaVu Serif'
        plt.rcParams['mathtext.fontset'] = 'dejavuserif'
        plt.rcParams['axes.prop_cycle'] = \
        plt.cycler(color=["tomato", "dodgerblue", "gold", 'forestgreen', 'mediumorchid', 'lightblue'])
        plt.rcParams['figure.dpi'] = 300
        

        ax = fig.subplot_mosaic(x,gridspec_kw={
                # set the height ratios between the rows
                "height_ratios": [1,0.00001,0.2],
                # set the width ratios between the columns
                "width_ratios": [1,1]})


        all_data_waves = np.concatenate([self.data[i]['wlgrid_center'] for i in self.data.keys()])
        ax['A'].set_xlim(np.min(all_data_waves)-0.2,np.max(all_data_waves)+0.5)
        ax['B'].set_xlim(np.min(all_data_waves)-0.2,np.max(all_data_waves)+0.5)
        #ax['A'].set_ylim(np.min(rprs_data2)-0.01*np.min(rprs_data2),np.max(rprs_data2)+0.01*np.max(rprs_data2))

        colors = ['tomato', 'dodgerblue','forestgreen','green','orchid','slateblue']
        ii=0
        for igrid in grid_names:   
            for idata in data_names: 
                wlgrid_center = self.data[idata]['wlgrid_center']
                y_data = self.data[idata]['y_data']
                e_data = self.data[idata]['e_data']
                best_fit = self.best_fits[igrid][idata][self.rank[igrid][idata],:][0,:]
                chi1 = self.chi_sqs[igrid][idata][self.rank[igrid][idata]][0]

                ax['A'].errorbar(wlgrid_center,y_data,yerr=e_data,fmt="ko",label=idata+" Reduction",markersize=5)
                ax['A'].plot(wlgrid_center,best_fit,colors[ii],linewidth=2,label=r"Best Fit "+igrid+", ${\chi}_{\\nu}$$^2$= "+ str(np.round(chi1,2)))

                ax['B'].plot(wlgrid_center,(y_data-best_fit)/e_data,"o",color=colors[ii],markersize=5)
                if ii==0:ax['B'].plot(wlgrid_center,0*y_data,"k")

                ii+=1

        ax['B'].set_xlabel(r"Wavelength [$\mu$m]",fontsize=20)
        ax['A'].set_ylabel(r"(R$_{\rm p}$/R$_{*}$)$^2$",fontsize=20)

        ax['A'].minorticks_on()
        ax['A'].tick_params(axis='y',which='major',length =20, width=3,direction='in',labelsize=20)
        ax['A'].tick_params(axis='y',which='minor',length =10, width=2,direction='in',labelsize=20)
        ax['A'].tick_params(axis='x',which='major',length =20, width=3,direction='in',labelsize=20)
        ax['A'].tick_params(axis='x',which='minor',length =10, width=2,direction='in',labelsize=20)

        ax['B'].minorticks_on()
        ax['B'].tick_params(axis='y',which='major',length =20, width=3,direction='in',labelsize=20)
        ax['B'].tick_params(axis='y',which='minor',length =10, width=2,direction='in',labelsize=20)
        ax['B'].tick_params(axis='x',which='major',length =20, width=3,direction='in',labelsize=20)
        ax['B'].tick_params(axis='x',which='minor',length =10, width=2,direction='in',labelsize=20)

        
        ax['B'].set_ylabel("${\delta}/N$",fontsize=20)
        
            
        ax['A'].legend(fontsize=12)
        
        
        return fig,ax


    def get_posteriors(self, grid_name, data_name, parameter):
        """
        Get posteriors (x,y) given a grid name, data name and parameter specified in grid_params

        Parameters
        ----------
        grid_names : list, str 
            List of grid names or string of single grid name to plot 
        data_names : list, str 
            List of data names or string of single 
        parameter : str 
            Name of parameter to get the posterior of (e.g. mh or tint)
        """
        parameter_sort = _finditem(self.grid_params[grid_name], parameter)
        print(parameter)
        if isinstance(parameter_sort, type(None)): 
            raise Exception(f'Parameter {parameter} not found in grid {grid_name}')
        
        chi_sq = self.chi_sqs[grid_name][data_name]

        parameter_grid =np.unique(parameter_sort)
        
        prob_array = np.exp(-chi_sq/2.0)
        alpha = 1.0/np.sum(prob_array)
        prob_array = prob_array*alpha
        
        prob= np.array([])

        for m in parameter_grid:
            wh = np.where(parameter_sort == m)
            prob = np.append(prob,np.sum(prob_array[wh]))
            
        return parameter_grid,prob


    def plot_posteriors(self, grid_name, data_name,parameters, fig=None, ax=None,
                       x_label_style={}, x_axis_type={}):
        """
        Plots posteriors for a given parameter set 

        Parameters
        ----------
        grid_names : str 
            grid names or string of single grid name to plot 
        data_names : str 
            data names or string of single 
        parameters : list, str
            Name or list of parameter(s) to get the posterior of (e.g. mh or tint) 
        x_label_style : dict 
            dictionary with elements of parameters for stylized x axis labels 
        x_axis_type : dict 
            dictionry with 'linear' 'log' arguments for the x axis. 
        """
        if isinstance(parameters, str): parameters=[parameters]

        if fig == None:
            if ax == None:
                plt.style.use('seaborn-paper')
                plt.rcParams['figure.figsize'] = [7, 4]           # Figure dimensions
                plt.rcParams['figure.dpi'] = 300
                plt.rcParams['image.aspect'] = 1.2                       # Aspect ratio (the CCD is quite long!!!)
                plt.rcParams['lines.linewidth'] = 1
                plt.rcParams['lines.markersize'] = 3
                plt.rcParams['lines.markeredgewidth'] = 0

                cmap = plt.cm.magma
                #cmap.set_bad('k',1.)

                plt.rcParams['image.cmap'] = 'magma'                   # Colormap.
                plt.rcParams['image.interpolation'] = None
                plt.rcParams['image.origin'] = 'lower'
                plt.rcParams['font.family'] = 'serif'
                plt.rcParams['font.serif'] = 'DejaVu Serif'
                plt.rcParams['mathtext.fontset'] = 'dejavuserif'
                #plt.rcParams['axes.prop_cycle'] = \
                #color = plt.cycler()
                colors=["tomato", "dodgerblue", "gold", 'forestgreen', 'mediumorchid', 'lightblue']
                plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)
                plt.rcParams['figure.dpi'] = 300
                nrow = 2
                ncol = int(np.ceil(len(parameters)/nrow))
                fig,ax = plt.subplots(nrows=nrow,ncols=ncol,figsize=(30,20))
        
        nrow=ax.shape[0]
        ncol=ax.shape[1]
        #colors=["tomato", "dodgerblue", "gold", 'forestgreen', 'mediumorchid', 'lightblue']
        iparam = -1 
        for irow in range(nrow):
            for icol in range(ncol):
                iparam+=1
                if icol==0: ax[irow,icol].set_ylabel("Probability",fontsize=50)
                if iparam > len(parameters)-1: 
                    try:
                        fig.delaxes(ax[irow,icol])
                        continue
                    except: 
                        continue
                else: 
                    get_post = _finditem(self.posteriors[grid_name][data_name], parameters[iparam])
                    if isinstance(get_post, type(None)): 
                        xgrid,yprob = [0,0,0],[0,0,0]
                    else: 
                        xgrid,yprob = get_post

                    ax[irow,icol].set_ylim(1e-2,1)
                    
                    if x_axis_type.get(parameters[iparam],'linear') == 'log':
                        xgrid = np.log10(xgrid)
                    cycler = ax[irow,icol]._get_lines.prop_cycler
                    ax[irow,icol].bar(xgrid, yprob,
                        width=[np.mean(abs(np.diff(xgrid)))/2]*len(xgrid), 
                        color="None",edgecolor=next(cycler)['color'],
                        linewidth=5,label=grid_name)
                    ax[irow,icol].tick_params(axis='both',which='major',length =40, width=3,direction='in',labelsize=30)
                    ax[irow,icol].tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=30)
                    
                    label = x_label_style.get(parameters[iparam],parameters[iparam])
                    ax[irow,icol].set_xlabel(label,fontsize=50)
                    
                    ax[irow,icol].legend(fontsize=20)
        return fig, ax 


def chi_squared(data,data_err,model,numparams):
    """
    Compute reduced chi squared assuming DOF = ndata_pts - num parameters  
    """
    
    chi_squared = np.sum(((data-model)/(data_err))**2)/(len(data)-numparams)
    
    return chi_squared

def plot_atmosphere(location,bf_filename,gas_names=None,fig=None,ax=None,linestyle=None,color=None,label=None):

    f = os.path.join(location, bf_filename)

        
        
    if os.path.isfile(f):
            
        ds = xr.open_dataset(f)

        temp = ds['temperature'].values[:]
        pressure = ds['pressure'].values[:]
        gas_vmr = np.zeros(shape=(len(gas_names),len(pressure)))

        try: # see if clouds are here
            pressure_cld = ds['pressure_cld'].values[:]
            wno_cld = ds['wno_cld'].values[:]
            wno_cld = ds['wno_cld'].values[:]
            wno_cld = ds['wno_cld'].values[:]
            opd_cld = ds['opd'].values
            asy_cld = ds['asy'].values
            ssa_cld = ds['ssa'].values
        except:
            pressure_cld = 0
            wno_cld = 0
            wno_cld = 0
            wno_cld = 0
            opd_cld = 0
            asy_cld = 0
            ssa_cld = 0

        for igas,gases in zip(range(0,len(gas_names)),gas_names):

            try: # see if clouds are here
                gas_vmr[igas,:]= ds[gases].values[:]
                
            except:
                gas_vmr[igas,:]= 0.0
        

        if fig == None:
            if ax == None:
                plt.style.use('seaborn-paper')
                plt.rcParams['figure.figsize'] = [7, 4]           # Figure dimensions
                plt.rcParams['figure.dpi'] = 300
                plt.rcParams['image.aspect'] = 1.2                       # Aspect ratio (the CCD is quite long!!!)
                plt.rcParams['lines.linewidth'] = 1
                plt.rcParams['lines.markersize'] = 3
                plt.rcParams['lines.markeredgewidth'] = 0

                cmap = plt.cm.get_cmap('tab20b', len(gas_names))
                #cmap.set_bad('k',1.)

                plt.rcParams['image.cmap'] = 'magma'                   # Colormap.
                plt.rcParams['image.interpolation'] = None
                plt.rcParams['image.origin'] = 'lower'
                plt.rcParams['font.family'] = 'serif'
                plt.rcParams['font.serif'] = 'DejaVu Serif'
                plt.rcParams['mathtext.fontset'] = 'dejavuserif'
                plt.rcParams['axes.prop_cycle'] = \
                plt.cycler(color=["tomato", "dodgerblue", "gold", 'forestgreen', 'mediumorchid', 'lightblue'])
                plt.rcParams['figure.dpi'] = 300
                fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(30,10))
        
        cmap = plt.cm.get_cmap('tab20b', len(gas_names))


        ax[0].set_ylim(1000,1e-6)
        ax[0].set_xlim(500,3500)

        ax[0].semilogy(temp,pressure,linewidth=3,linestyle=linestyle,color=color,label=label)
        ax[0].minorticks_on()
        ax[0].tick_params(axis='both',which='major',length =40, width=3,direction='in',labelsize=30)
        ax[0].tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=30)
        ax[0].legend(fontsize=15)
        ax[0].set_xlabel(r"Temperature [K]",fontsize=30)
        ax[0].set_ylabel(r"Pressure [Bars]",fontsize=30)

        ax[1].set_ylim(1000,1e-6)
        ax[1].set_xlim(1e-8,1)
        
        
        #ax[1].set_ylabel(r"Pressure [Bars]",fontsize=50)


        for igas,gases in zip(range(0,len(gas_names)),gas_names):
            
            ax[1].loglog(gas_vmr[igas,:],pressure,linewidth=3,linestyle=linestyle,color=cmap(igas),label=gases+" "+label)
            
        
        ax[1].minorticks_on()
        ax[1].tick_params(axis='both',which='major',length =40, width=3,direction='in',labelsize=30)
        ax[1].tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=30)
        ax[1].set_xlabel(r"VMR",fontsize=30)
        
        
        ax[1].legend(fontsize=12)


        ax[2].set_ylim(1000,1e-6)
        ax[2].set_xlim(1e-5,500)

        
        #ax[2].set_ylabel(r"Pressure [Bars]",fontsize=50)
        if np.sum(opd_cld) != 0:
            ax[2].loglog(opd_cld[:,150],pressure_cld,linewidth=2,linestyle=linestyle,color=color,label=label)
        
        ax[2].minorticks_on()
        ax[2].tick_params(axis='both',which='major',length =40, width=3,direction='in',labelsize=30)
        ax[2].tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=30)
        ax[2].set_xlabel(r"Cloud OPD (1 ${\mu}$m)",fontsize=30)
        ax[2].legend(fontsize=15)

    else:
        print("Filename or directory is not correct.")
        fig,ax=0


    
    
    # function to show best fit atmospheric abundances
    
    return fig,ax



def plot_contribution(mass,radius,T_st,met_st,logg_st,radius_st,opa,location,bf_filename,offset_mp,wlgrid_center,rprs_data2,e_rprs2,gas_contribution=None,fig=None,ax=None):
    # function to show best fit spectrum
    # and contribution of each gas to the best fit spectrum

    gas_names = [ 'e-','H2','He',  'H', 'H+', 'H-',
       'H2-', 'H2+', 'H3+', 'H2O', 'CH4', 'CO', 'NH3', 'N2', 'PH3',
       'H2S', 'TiO', 'VO', 'Fe', 'FeH', 'CrH', 'Na', 'K', 'Rb', 'Cs', 'CO2',
       'HCN', 'C2H2', 'C2H4', 'C2H6', 'SiO', 'MgH', 'OCS', 'Li', 'LiOH', 'LiH',
       'LiCl'] 

    f = os.path.join(location, bf_filename)

        
        
    if os.path.isfile(f):
            
        ds = xr.open_dataset(f)

        temp = ds['temperature'].values[:]
        pressure = ds['pressure'].values[:]
        gas_vmr = np.zeros(shape=(len(gas_names),len(pressure)))
        spectra_mp = ds['transit_depth'].values[:]
        wvl_mp = ds['wavelength'].values[:]
        
        try: # see if clouds are here
            pressure_cld = ds['pressure_cld'].values[:]
            wno_cld = ds['wno_cld'].values[:]
            wno_cld = ds['wno_cld'].values[:]
            wno_cld = ds['wno_cld'].values[:]
            opd_cld = ds['opd'].values
            asy_cld = ds['asy'].values
            ssa_cld = ds['ssa'].values
        except:
            pressure_cld = 0
            wno_cld = 0
            wno_cld = 0
            wno_cld = 0
            opd_cld = 0
            asy_cld = 0
            ssa_cld = 0

        for igas,gases in zip(range(0,len(gas_names)),gas_names):

            try: # see if clouds are here
                gas_vmr[igas,:]= ds[gases].values[:]
                
            except:
                gas_vmr[igas,:]= 0.0
        

        if fig == None:
            if ax == None:
                plt.style.use('seaborn-paper')
                plt.rcParams['figure.figsize'] = [7, 4]           # Figure dimensions
                plt.rcParams['figure.dpi'] = 300
                plt.rcParams['image.aspect'] = 1.2                       # Aspect ratio (the CCD is quite long!!!)
                plt.rcParams['lines.linewidth'] = 1
                plt.rcParams['lines.markersize'] = 3
                plt.rcParams['lines.markeredgewidth'] = 0

                cmap = plt.cm.get_cmap('tab20b', len(gas_names))
                #cmap.set_bad('k',1.)

                plt.rcParams['image.cmap'] = 'magma'                   # Colormap.
                plt.rcParams['image.interpolation'] = None
                plt.rcParams['image.origin'] = 'lower'
                plt.rcParams['font.family'] = 'serif'
                plt.rcParams['font.serif'] = 'DejaVu Serif'
                plt.rcParams['mathtext.fontset'] = 'dejavuserif'
                plt.rcParams['axes.prop_cycle'] = \
                plt.cycler(color=["tomato", "dodgerblue", "gold", 'forestgreen', 'mediumorchid', 'lightblue'])
                plt.rcParams['figure.dpi'] = 300
                fig,ax = plt.subplots(nrows=1,figsize=(18,10))
    
    
    
    

    
    ax.set_xlim(np.min(wlgrid_center)-0.2,np.max(wlgrid_center)+0.5)
    ax.errorbar(wlgrid_center,rprs_data2,yerr=e_rprs2,fmt="ko",label="Data",markersize=5)

    
    one_by_gas = np.zeros(shape=(len(gas_contribution),len(wlgrid_center)))
    
    count=0
    
    data_atm = {'pressure': pressure,
        'temperature': temp}
  
    # Convert the dictionary into DataFrame
    df_atm = pd.DataFrame(data_atm)
    
    def shift_spectrum(waves,shift):
            return y+shift
    for igas,gases in zip(range(0,len(gas_names)),gas_names):
    
        df_atm[gases] = gas_vmr[igas,:]
       
    case1 = jdi.inputs()
    case1.approx(p_reference=10)
    case1.phase_angle(0)
    case1.gravity(mass=mass, mass_unit=jdi.u.Unit('M_jup'),
                  radius=radius, radius_unit=jdi.u.Unit('R_jup'))

    T_st,met_st,logg_st,radius_st = T_st,met_st,logg_st,radius_st

    case1.star(opa, T_st,met_st,logg_st,radius=radius_st, radius_unit = jdi.u.Unit('R_sun') )

    case1.atmosphere( df =df_atm)

    if np.sum(opd_cld) != 0:
        df_cld= vj.picaso_format(opd_cld, ssa_cld, asy_cld)
    
        case1.clouds(df=df_cld)
        #case1.inputs['clouds']['wavenumber'] = wno_cld 

    df_one_gas= case1.spectrum(opa, full_output=True,calculation=['transmission']) #note the new last key

    x,y = jdi.mean_regrid(1e4/df_one_gas['wavenumber'],df_one_gas['transit_depth'],newx=wlgrid_center)
    
    mmw_old = np.copy(df_one_gas['full_output']['layer']['mmw'])
    

    ax.plot(x,y+offset_mp,linewidth=5,label="Best Fit Complete")
    
    
    count=0
    for gases in gas_contribution:
        
        opa = jdi.opannection(wave_range=[0.49,5.7])
        
        def shift_spectrum(waves,shift):
            return y+shift
        
        case1 = jdi.inputs()
        case1.approx(p_reference=10)
        case1.phase_angle(0)
        case1.gravity(mass=mass, mass_unit=jdi.u.Unit('M_jup'),
                      radius=radius, radius_unit=jdi.u.Unit('R_jup'))
        
        T_st,met_st,logg_st,radius_st = T_st,met_st,logg_st,radius_st
        
        case1.star(opa, T_st,met_st,logg_st,radius=radius_st, radius_unit = jdi.u.Unit('R_sun') )
        
        #case1.atmosphere( df =df_atm)
        case1.atmosphere( df = df_atm,exclude_mol=[gases])
        df_one_gas= case1.spectrum(opa, full_output=True,calculation=['transmission']) #note the new last key
        
        mmw_new = df_one_gas['full_output']['layer']['mmw']
        
        df_temp = df_atm.copy()
        df_temp['temperature'] = df_atm['temperature']*(np.mean(mmw_new)/np.mean(mmw_old))
        
        case1.atmosphere( df = df_temp,exclude_mol=[gases])
        df_one_gas= case1.spectrum(opa, full_output=True,calculation=['transmission']) #note the new last key

        x,y = jdi.mean_regrid(1e4/df_one_gas['wavenumber'],df_one_gas['transit_depth'],newx=wlgrid_center)
        
        
        #popt, pcov = optimize.curve_fit(shift_spectrum, wlgrid_center, rprs_data2)
        
        ax.plot(x,y+offset_mp,label="No "+gases,linewidth=2)
        
        count+=1

    if np.sum(opd_cld) != 0:

        case1 = jdi.inputs()
        case1.approx(p_reference=10)
        case1.phase_angle(0)
        case1.gravity(mass=mass, mass_unit=jdi.u.Unit('M_jup'),
                    radius=radius, radius_unit=jdi.u.Unit('R_jup'))

        T_st,met_st,logg_st,radius_st = T_st,met_st,logg_st,radius_st

        case1.star(opa, T_st,met_st,logg_st,radius=radius_st, radius_unit = jdi.u.Unit('R_sun') )

        case1.atmosphere( df =df_atm)

        df_one_gas= case1.spectrum(opa, full_output=True,calculation=['transmission']) #note the new last key

        x,y = jdi.mean_regrid(1e4/df_one_gas['wavenumber'],df_one_gas['transit_depth'],newx=wlgrid_center)
        
        mmw_old = np.copy(df_one_gas['full_output']['layer']['mmw'])
        

        ax.plot(x,y+offset_mp,linewidth=4,label="No Clouds",linestyle="--",color="k")

        
    
    ax.legend(fontsize=20,ncol=3)
    
    
    
    
    
    
    
    #ax.legend(fontsize=20)
    ax.set_xlabel(r"Wavelength [$\mu$m]",fontsize=20)
    ax.set_ylabel(r"(R$_{\rm p}$/R$_{*}$)$^2$",fontsize=20)


    ax.minorticks_on()
    ax.tick_params(axis='y',which='major',length =20, width=3,direction='in',labelsize=20)
    ax.tick_params(axis='y',which='minor',length =10, width=2,direction='in',labelsize=20)
    ax.tick_params(axis='x',which='major',length =20, width=3,direction='in',labelsize=20)
    ax.tick_params(axis='x',which='minor',length =10, width=2,direction='in',labelsize=20)

    plt.show()

    return fig,ax

def _finditem(obj, key):
    if key in obj: return obj[key]
    for k, v in obj.items():
        if isinstance(v,dict):
            item = _finditem(v, key)
            if item is not None:
                return item

def _get_xarray_attr(attr_dict, parameter):
    not_found_msg = "{parameter} not specified"
    #we assume clear if no fsed parameter specified
    if parameter =='fsed':
        not_found_msg='clear'
    param = attr_dict.get(parameter,not_found_msg)
    if isinstance(param, dict):
        param = param.get('value',param)
    return param
