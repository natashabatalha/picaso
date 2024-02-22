import numpy as np
import pandas as pd
import xarray as xr
import scipy as sp
from scipy.interpolate import griddata
cKDTree = sp.spatial.cKDTree
optimize = sp.optimize

import json
import matplotlib.pyplot as plt
import os
from astropy.convolution import convolve, Box1DKernel, Gaussian1DKernel
import astropy.units as u
import glob
import itertools

from matplotlib.ticker import StrMethodFormatter

import virga.justdoit as vj
from .justdoit import inputs, opannection, mean_regrid, u, input_xarray, copy

from bokeh.palettes import Cividis
from multiprocessing import Pool



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
        Only keep as None if user chooses 'init' as type
    model_type : str , ('xarrays','user')
        Default = xarray which points to a directly full of xarrays in picaso format 
        Otherwise can input "user" which initializes the grid fitter without 
        reading anything in. "user" gives users the flexibility to add  
        in their own input
    to_fit : str 
        parameter to fit, default is transit_depth. other common is flux
    """
    def __init__(self, grid_name, model_dir=None,
        to_fit='transit_depth', model_type='xarrays', save_chem=False, 
        verbose=True):
        self.verbose=verbose
        
        self.grids = []
        self.list_of_files = {}
        self.grid_params = {}
        self.overview = {}
        self.wavelength={}
        self.temperature={}
        if save_chem: self.chemistry={}
        self.pressure={}
        self.spectra={}
        self.interp_params={}
        
        #adds first grid
        if model_type=='xarrays': 
            self.add_grid(grid_name, model_dir, to_fit=to_fit, save_chem=save_chem)
        elif model_type == 'user': 
            self.grids += [grid_name]
            
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

    def add_grid(self,grid_name,model_dir,to_fit='transit_depth',
        save_chem=False):
        #loads in grid info
        self.grids += [grid_name]
        self.find_grid(grid_name, model_dir)
        self.load_grid_params(grid_name,to_fit=to_fit,save_chem=save_chem)

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
    def as_dict(self):
        """
        get class in dictionary form to easily search
        """
        return {
        'list_of_files':self.list_of_files, 
        'spectra_w_offset':self.best_fits,
        'rank_order':self.rank,
        'grid_params':self.grid_params, 
        'offsets': getattr(self, 'offsets',0), #,
        'chi_sqs': self.chi_sqs,
        'posteriors': self.posteriors
        }


    def load_grid_params(self,grid_name,to_fit='transit_depth',
        save_chem=False):
        """
        This will read the grid parameters and set the array of parameters 

        Parameters 
        ----------
        grid_name : str 
            Name of grid for bookkeeping
        to_fit : str 
            Default is transit_depth but also could be flux or any other xarray parameter 
            you are interested in fitting. 

        Returns
        -------
        None 
            Creates self.overview, and self.grid_params
        """
        possible_params = {'planet_params': ['rp','mp','tint', 'heat_redis','p_reference','logkzz','mh','cto','p_quench','rainout','teff','logg','m_length'],
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
            if save_chem:
                #grab everything on pressure grid thats not temperature
                mols = [i for i in ds.data_vars.keys() if 
                        'pressure' in ds.data_vars[i].coords 
                       and i != 'temperature']

            if ct == 0:
                nwave = len(ds['wavelength'].values)
                npress = len(ds['pressure'].values)
                
                spectra_grid = np.zeros(shape=(number_files,nwave))
                temperature_grid = np.zeros(shape=(number_files,npress))
                pressure_grid = np.zeros(shape=(number_files,npress))
                wavelength = ds['wavelength'].values
                if save_chem:
                    molecule_dict = {}
                    for imol in mols: 
                        molecule_dict[imol]= np.zeros(
                        shape=(number_files,npress))
            
            #start filling out grid parameters
            #seems like we need to save these?????
            temperature_grid[ct,:] = ds['temperature'].values[:]
            pressure_grid[ct,:] = ds['pressure'].values[:]
            spectra_grid[ct,:] = ds[to_fit].values[:] 
            if save_chem:
                for imol in mols: 
                    molecule_dict[imol][ct,:] = ds[imol].values[:]

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
                self.grid_params[grid_name].pop(iattr)

        #for iattr in possible_params.keys():
        #    if len(self.grid_params[grid_name][iattr].keys())==0: 
        #        self.grid_params[grid_name].pop(iattr)

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
        if save_chem: 
            self.chemistry[grid_name] = molecule_dict
    def fit_all(self):
        for i in self.grids:
            for j in self.data.keys() :
                self.fit_grid(i, j)

    def fit_grid(self,grid_name, data_name,dof='ndata', offset=True):
        """
        Fits grids given model and data. Retrieves posteriors of fit parameters.

        Parameters
        ----------
        grid_name : str 
            grid name that was specified in GridFiter or add_grid
        data_name : str 
            data name that was specified before in add_data
        dof : str 
            used for chi square. if dof=='ndata' then chi square is computed as chi2/len(data). 
            otherwise it computes as chi2/(len(data) - numparameters)
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
            #self.overview[grid_name]['num_params'] = self.overview[grid_name]['num_params'] + 1

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
            if dof == 'ndata': numparams=0
            self.chi_sqs[grid_name][data_name][index]= chi_squared(y_data,e_data,flux_in_bin+shift,numparams)

            self.best_fits[grid_name][data_name][index,:] = flux_in_bin+shift
            if offset: self.offsets[grid_name][data_name][index] = shift

        self.rank[grid_name][data_name] = self.chi_sqs[grid_name][data_name].argsort()

        #finally compute the posteriors 
        for iattr in self.grid_params[grid_name].keys(): 
            for ikey in self.grid_params[grid_name][iattr].keys():
                self.posteriors[grid_name][data_name][ikey] = self.get_chi_posteriors(grid_name, data_name, ikey)
    
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

    def plot_best_fit(self, grid_names, data_names, plot_kwargs={}): 
        """
        
        Parameters
        ----------
        grid_names : list, str 
            List of grid names or string of single grid name to plot 
        data_names : list, str 
            List of data names or string of single 
        plot_kwargs : dict 
            key word arguments for matplotlib plt
        """
        if isinstance(grid_names ,str):grid_names=[grid_names]
        if isinstance(data_names ,str):data_names=[data_names]

        x='''
        AA
        ..
        BB
        '''
        fig = plt.figure(figsize=(18,10))
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams['figure.figsize'] = [7, 4]           # Figure dimensions
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['image.aspect'] = 1.2                       # Aspect ratio (the CCD is quite long!!!)
        plt.rcParams['lines.linewidth'] = 1
        plt.rcParams['lines.markersize'] = 3
        plt.rcParams['lines.markeredgewidth'] = 0
        
        cmap = plt.cm.magma
        #cmap.set_bad('k',1.)
        
        plt.rcParams['image.cmap'] = 'magma'                   # Colormap.
        plt.rcParams['image.interpolation'] = 'None'
        plt.rcParams['image.origin'] = 'lower'
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.serif'] = 'DejaVu Sans'
        plt.rcParams['mathtext.fontset'] = 'stixsans'
        #plt.rcParams['axes.prop_cycle'] = \
        #plt.cycler(color=["tomato", "dodgerblue", "gold", 'forestgreen', 'mediumorchid', 'lightblue'])
        plt.rcParams['figure.dpi'] = 300
        colors=["xkcd:salmon", "dodgerblue", "sandybrown", 'cadetblue', 'orchid', 'lightblue']
        plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)        

        ax = fig.subplot_mosaic(x,gridspec_kw={
                # set the height ratios between the rows
                "height_ratios": [1,0.00001,0.2],
                # set the width ratios between the columns
                "width_ratios": [1,1]})


        all_data_waves = np.concatenate([self.data[i]['wlgrid_center'] for i in self.data.keys()])
        ax['A'].set_xlim(np.min(all_data_waves)-0.1,np.max(all_data_waves)+0.1)
        ax['B'].set_xlim(np.min(all_data_waves)-0.1,np.max(all_data_waves)+0.1)
        #ax['A'].set_ylim(np.min(rprs_data2)-0.01*np.min(rprs_data2),np.max(rprs_data2)+0.01*np.max(rprs_data2))

        #colors = ['tomato', 'dodgerblue','forestgreen','green','orchid','slateblue']
        ii=0
        for igrid in grid_names:   
            for idata in data_names: 
                color = ax['A']._get_lines.get_next_color()

                wlgrid_center = self.data[idata]['wlgrid_center']
                y_data = 100*self.data[idata]['y_data']
                e_data = 100*self.data[idata]['e_data']
                best_fit = 100*self.best_fits[igrid][idata][self.rank[igrid][idata],:][0,:]
                chi1 = self.chi_sqs[igrid][idata][self.rank[igrid][idata]][0]

                ax['A'].plot(wlgrid_center,best_fit,color,linewidth=2,label=r"Best Fit "+igrid+"+"+idata+", ${\chi}_{\\nu}$$^2$= "+ str(np.round(chi1,2)))

                ax['B'].plot(wlgrid_center,(y_data-best_fit)/e_data,"o",color=color,markersize=5)
                if ii==0:ax['B'].plot(wlgrid_center,0*y_data,"k")

                ii+=1

        for i,idata in enumerate(data_names):
            wlgrid_center = self.data[idata]['wlgrid_center']
            y_data = 100*self.data[idata]['y_data']
            e_data = 100*self.data[idata]['e_data']
            ax['A'].errorbar(wlgrid_center,y_data,yerr=e_data,fmt="o",color=Cividis[7][i],label=idata+" Reduction",markersize=5)
        
        ax['B'].set_xlabel(plot_kwargs.get('xlabel',r"wavelength [$\mu$m]"),fontsize=20)
        ax['A'].set_ylabel(plot_kwargs.get('ylabel',r"transit depth [%]"),fontsize=20)

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
        
            
        ax['A'].legend(fontsize=16)
        
        
        return fig,ax


    def get_chi_posteriors(self, grid_name, data_name, parameter):
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


    def plot_chi_posteriors(self, grid_name, data_name,parameters, fig=None, ax=None,
                       x_label_style={}, x_axis_type={}, label=''):
        """
        Plots posteriors for a given parameter set 

        Parameters
        ----------
        grid_names : str 
            string of single grid name to plot 
        data_names : str 
            data names or string of single 
        parameters : list, str
            Name or list of parameter(s) to get the posterior of (e.g. mh or tint) 
        x_label_style : dict 
            dictionary with elements of parameters for stylized x axis labels 
        x_axis_type : dict 
            dictionry with 'linear' 'log' arguments for the x axis. 
        labels : list 
            how to label the data 
        """
        if isinstance(parameters, str): parameters=[parameters]

        if label == '':
            legend_label = grid_name + ' ' + data_name
        else: 
            legend_label = label

        if fig == None:
            if ax == None:
                plt.style.use('seaborn-v0_8-colorblind')
                plt.rcParams['figure.figsize'] = [7, 4]           # Figure dimensions
                plt.rcParams['figure.dpi'] = 300
                plt.rcParams['image.aspect'] = 1.2                       # Aspect ratio (the CCD is quite long!!!)
                plt.rcParams['lines.linewidth'] = 1
                plt.rcParams['lines.markersize'] = 3
                plt.rcParams['lines.markeredgewidth'] = 0

                cmap = plt.cm.magma
                #cmap.set_bad('k',1.)

                plt.rcParams['image.cmap'] = 'magma'                   # Colormap.
                plt.rcParams['image.interpolation'] = ''
                plt.rcParams['image.origin'] = 'lower'
                plt.rcParams['font.family'] = 'sans-serif'
                plt.rcParams['font.serif'] = 'DejaVu Sans'
                plt.rcParams['mathtext.fontset'] = 'stixsans'
                #plt.rcParams['axes.prop_cycle'] = \
                #color = plt.cycler()
                colors=["xkcd:salmon", "dodgerblue", "sandybrown", 'cadetblue', 'orchid', 'lightblue']
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
                    #cycler = ax[irow,icol]._get_lines.prop_cycler
                    #col = next(cycler)['color']
                    ax[irow,icol].bar(xgrid, yprob,
                        width=[np.mean(abs(np.diff(xgrid)))/2]*len(xgrid), 
                        #color=col,edgecolor=col,
                        linewidth=5,label=legend_label,alpha=0.2 )
                    ax[irow,icol].tick_params(axis='both',which='major',length =40, width=3,direction='in',labelsize=30)
                    ax[irow,icol].tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=30)
                    
                    label = x_label_style.get(parameters[iparam],parameters[iparam])
                    ax[irow,icol].set_xlabel(label,fontsize=50)
                    
                    ax[irow,icol].legend(fontsize=20)
        return fig, ax 


    def prep_gridtrieval(self, grid_name,add_ptchem=False):
        """
        Preps the loaded grid for interpolated gridtrieval 
        """
        if add_ptchem:
            sqr_spec , sqr_pt, sqr_chem, uniq, offset_prior,df_grid_params = self.transform_4_interp(grid_name, add_ptchem=add_ptchem)
        else: 
            sqr_spec , uniq, offset_prior,df_grid_params = self.transform_4_interp(grid_name, add_ptchem=add_ptchem)
        
        self.interp_params = {}
        self.interp_params[grid_name] = {}
        self.interp_params[grid_name]['offset_prior'] = offset_prior
        self.interp_params[grid_name]['grid_parameters'] = df_grid_params
        self.interp_params[grid_name]['grid_parameters_unique'] = uniq
        self.interp_params[grid_name]['square_spectra_grid'] = sqr_spec

        if add_ptchem:
            self.interp_params[grid_name]['square_temp_grid'] = sqr_pt
            self.interp_params[grid_name]['square_chem_grid'] = sqr_chem

    def transform_4_interp(fitter, grid_name,add_ptchem=False):
        """
        Transforms the fitter output into a square grid with np.nan in place 
        of spectra that are missing 
        This allows us to use the custom grid interpolator for cases where spectra are 
        available. for everything else it will use the non square grid 
        
        Parameters
        ----------
        fitter : analyze.GridFitter
            class that has all the spectra information loaded 
        grid_name : str 
            Name of grid that you want to interpolate on
        """
        all_spectra = fitter.spectra[grid_name]
        
        if add_ptchem: 
            all_pt = fitter.temperature[grid_name]
            all_chem = fitter.chemistry[grid_name]
            mols = all_chem.keys()
            all_pressures = fitter.pressure[grid_name]
            unique_ps = np.unique(all_pressures, axis=0)
            if len(unique_ps)>1: 
                raise Exception("""Detected non-uniform pressure grid.
                Grid needs to be on a uniform pressure grid if you want to use the interpolate feature 
                You can use the function analyze.interp_pressure_grid to adjust the grid data""")

        df_grid_params = pd.DataFrame(index = range(len(fitter.list_of_files[grid_name])))
        grid_params=[]
        for i in fitter.grid_params[grid_name].keys():
            for j in fitter.grid_params[grid_name][i].keys():
                grid_params+=[j]
                df_grid_params[j] = fitter.grid_params[grid_name][i][j]
        grid_params_unique = {}

        offset_pm = abs(2*(np.min(fitter.spectra[grid_name]) - 
                           np.max(fitter.spectra[grid_name])))

        for i in grid_params:
            grid_params_unique[i]=np.array([float(i) for i in 
                                        sorted(df_grid_params[i].unique())])
        
        #if the grid were square what size would it be based on unique params
        square_size = np.prod([len(i) for i in grid_params_unique.values()])
        #if square_size != all_spectra.shape[0]:
        #grid is not square let's fix that for interpolation 
        spectra_square = []
        #and add pt/chem if we want
        if add_ptchem: 
            pt_square = []
            chem_square = {imol:[] for imol in mols}

        full_df_grid = pd.DataFrame(columns = grid_params_unique.keys(), 
                                   index = range(square_size))
        for i,icombo in enumerate(itertools.product(*grid_params_unique.values())): 

            matches = df_grid_params.astype(float).eq(icombo)
            matches_all_rows = matches.all(axis=1)
            matches = df_grid_params.loc[matches_all_rows]

            if len(matches.index)==0: 
                #if there are no matches then let's add a nan to that location
                full_df_grid.iloc[i,:] = icombo
                spectra_square += [[np.nan]*all_spectra.shape[1]]
                if add_ptchem: 
                    pt_square += [[np.nan]*all_pt.shape[1]]
                    for imol in mols:
                        chem_square[imol] += [[np.nan]*all_chem[imol].shape[1]]

            else: 
                #if there are matches then let's add in the corresponding value
                ind = matches.index[0]
                full_df_grid.iloc[i,:] = icombo
                spectra_square += [all_spectra[ind]]
                if add_ptchem: 
                    pt_square += [all_pt[ind]]
                    for imol in mols: 
                        chem_square[imol] += [all_chem[imol][ind]]

        #now we can properly reshape everything
        spectra_square = np.reshape(spectra_square, 
                                    [len(i) for i in grid_params_unique.values()]
                                    +[all_spectra.shape[1]])

        if add_ptchem: 
            pt_square = np.reshape(pt_square, [len(i) for i in grid_params_unique.values()]
                                    +[all_pt.shape[1]])
            for imol in mols: 
                chem_square[imol] = np.reshape(chem_square[imol], [len(i) for i in grid_params_unique.values()]
                                    +[all_chem[imol].shape[1]])
        """else: 
                                    #reshape all_spectra to be on npar1 x npar2 x npar3 etc 
                                    spectra_square = np.reshape(all_spectra, 
                                                                [len(i) for i in grid_params_unique.values()]
                                                                +[all_spectra.shape[1]])
                                    if add_ptchem: 
                                        #reshape all_spectra to be on npar1 x npar2 x npar3 etc 
                                        pt_square = np.reshape(all_pt, 
                                                                [len(i) for i in grid_params_unique.values()]
                                                                +[all_pt.shape[1]])
                                        for imol in mols: 
                                            chem_square = {imol: np.reshape(all_chem[imol], 
                                                                [len(i) for i in grid_params_unique.values()]
                                                                +[all_chem[imol].shape[1]])
                                                            for imol in mols}"""
        

        #lastly replace nans in grid with real values 
        def replace_nans(data):
            #will first interpolate nearest neighbors 
            #then it will replace nans with nearest neighbors
            for imethod in ['linear','nearest']:
                nan_coords = np.argwhere(np.isnan(data))

                # Find the coordinates and values of non-NaN values
                non_nan_coords = np.argwhere(~np.isnan(data))
                non_nan_values = data[~np.isnan(data)]

                # Perform interpolation to fill NaN values
                filled_data = griddata(non_nan_coords, non_nan_values, nan_coords, method=imethod)

                # Replace NaN values with interpolated values
                data[np.isnan(data)] = filled_data
            return data
        #import time
        #removing this because grid data takes way too long with 
        #high res spectra 

        #if np.argwhere(np.isnan(spectra_square)).shape[0]!=0: 
        #    spectra_square = replace_nans(spectra_square)
        #    print('finish spec', time.time()-start);start = time.time()

        if add_ptchem:
            if np.argwhere(np.isnan(pt_square)).shape[0]!=0: 
                pt_square = replace_nans(pt_square)
                for imol in chem_square.keys():
                    chem_square[imol] = replace_nans(chem_square[imol])

        if add_ptchem: 
            return (spectra_square, pt_square, chem_square, 
                                grid_params_unique, offset_pm,df_grid_params)
        else :
            return spectra_square, grid_params_unique, offset_pm,df_grid_params
    
    def interp_pressure_grid(self, new_press_grid ,grid_name):
        """
        This function will help you reinterpolate your grid to a new 
        common pressure grid. 

        Parameters
        ----------
        new_press_grid : ndarray    
            new pressure grid in bars, ascending order 
        grid_name : str 
            name of grid you would like to reinterpolate
        """
        new_press_grid = np.sort(new_press_grid)
        nlevels=len(new_press_grid)

        #old stuff
        all_pt = self.temperature[grid_name]
        all_chem = self.chemistry[grid_name]
        all_pressures = self.pressure[grid_name]

        #double check we can actually interpolate with a ordered pressure grid
        unique_ps = np.unique(all_pressures, axis=0)
        for ips in unique_ps: 
            if ips[0] != np.min(ips): 
                raise Exception('Uh oh! Youve read in a grid that is not in ascending order. Please reorder before proceeding')

        #define new stuff
        new_all_pt = np.zeros((all_pt.shape[0], nlevels))
        new_all_chem = {imol:np.zeros((all_chem[imol].shape[0], nlevels)) for imol in all_chem.keys()}
        new_all_pressures = np.zeros((all_pressures.shape[0], nlevels))

        #loop through and interpolate everything
        new_logp = np.log10(new_press_grid)
        for i in range(all_pt.shape[0]): 
            new_all_pressures[i,:] = new_press_grid 
            
            old_logp = np.log10(all_pressures[i,:])
            
            new_all_pt[i,:] = np.interp(new_logp,old_logp,all_pt[i,:])
            
            for imol in new_all_chem.keys(): 
                new_all_chem[imol][i,:] = 10**np.interp(new_logp,old_logp,np.log10(all_chem[imol][i,:]))
        
        self.temperature[grid_name] = new_all_pt
        self.chemistry[grid_name] = new_all_chem
        self.pressure[grid_name] = new_all_pressures  
        return 

    
def custom_interp(final_goal,fitter,grid_name, to_interp='spectra',array_to_interp=None ): 
    """
    Custom interpolation routine that interpolates based on the nearest two neighbors
    of each parameter. e.g. if interpolating M/H and C/O it will find the upper 
    and lower M/H, C/O and interpolate between those four grid points. 
    Currently, this does not handle cases that go off the grid. 
    
    Parameters
    ----------
    final_goal : array
        Values on which to interpolate on in the order of grid_pars 
        e.g. [mh_interp, co_interp]
    fitter : analyze.GridFitter
        See tutorial, provided loaded grid fitter tool
    grid_name : str
        name of grid provided to analyze.GridFitter
    interp : str 
        Default = 'spectra', this dictates the entity you want to interpolate 
        Other option is to specify "custom" in this case you will have to 
        add in a array of something else (e.g. temperature, chemistry). 
    array_to_interp : array
        Default = None, in this case it assumes you are fitting for the spectrum. 
        Otherwise you have to input the array of what you want to input via this 
        variable.
        This array should be on an identical array as the spectra, except the last dimension
        which might not be of wavelength (could be of pressure for instance)
    
    Returns
    -------
    ndarray
        Final spectra interpolated onto final_goal values requested 
    """

    grid_points = fitter.interp_params[grid_name]['grid_parameters']
    grid_pars = fitter.interp_params[grid_name]['grid_parameters_unique']

    if 'spectra' in to_interp:
        spectra = fitter.interp_params[grid_name]['square_spectra_grid']
    else: 
        spectra = array_to_interp

    
    #transform to list of unique values 
    grid_pars = list(grid_pars.values())
    hypercube = np.array(list(itertools.product([0,1],repeat=len(grid_pars))))
    hilos = np.array([find_bounding_values(arr, val)[0] for arr, val in 
                 zip(grid_pars, final_goal)])
    hilos_inds = np.array([find_bounding_values(arr, val)[1] for arr, val in 
                 zip(grid_pars, final_goal)])

    #@jit(nopython=True)
    def weight_interp(grid_pars, final_goal, spectra,hypercube,hilos, hilos_inds): 
        weights = []
        for i in range(len(final_goal)): 
            weights += [(final_goal[i] - hilos[i,0])/(hilos[i,1] - hilos[i,0])]

        weights = np.array(weights)#length of number of params
        inv_weights = 1-weights
        all_weights = np.array([inv_weights, weights]).T#same as inds

        interp = 0 
        for irow in hypercube: 
            weight_multip = [all_weights[i][j] for i, j in enumerate(irow)]
            inds = [hilos_inds[i][j] for i, j in enumerate(irow)]+[-1]
            spec_interp = get_last_dimension(spectra, inds)
            interp+= np.product(weight_multip)*spec_interp
        return interp
    interp = weight_interp(grid_pars, final_goal, spectra,hypercube, hilos,hilos_inds)
    
    if 'spectra' in to_interp:
        #only fill this gap if you are fitting spectra which takes too long with the griddata method
        if np.any(np.isnan(interp)):
            tree = cKDTree(grid_points)
            dd, inds = tree.query(final_goal, k=3)
            weights = 1.0 / dd**2
            interp = np.dot(weights , fitter.spectra[grid_name][inds]) / np.sum(weights)  
        
    return interp

def get_last_dimension(arr, indices):
    """
    Given an array of any dimenion, returns the last dimension with the other 
    dimensions specified by "indices". 
    
    for example, if arr = np.random.randn([4,4]) and indices = [1,-1], it would
    return arr[1,:]. this is slightly confusing because -1 usually indicates last element, 
    which is different from the use of ":". However, jit nopython cannot compare string 
    elements. Therefore this code uses -1 in place of :. 
    """
    if len(indices) == 1 and indices[0] == -1:
        return arr
    elif len(indices) == 1:
        return arr[indices[0]]
    
    return get_last_dimension(arr[indices[0]], indices[1:])

def find_bounding_values(arr, value):
    """
    Given an array and a value this finds the values on each side of the array. 
    If this locates a value on the edge, it returns the edge and the parameter to the 
    right or left. If it is out of bounds it returns just the edge value 
    
    Parameters
    ----------
    arr : array
        sorted array of values 
    value : float 
        float located within the bounds of the array specified 
    """
    # Find the indices of the elements that are less than or equal to the value
    indices = np.where(arr <= value)[0]

    # Check if the value is smaller than the smallest element in the array
    if len(indices) == 0:
        return [arr[0]]

    # Check if the value is larger than the largest element in the array
    if indices[-1] == len(arr) - 1:
        lower = arr[-2]
        upper = arr[-1]
        ind_lower = indices[-1] - 1
        ind_upper = indices[-1]
    else:
        # Get the elements at the indices and the next index
        lower = arr[indices[-1]]
        upper = arr[indices[-1] + 1]
        ind_lower = indices[-1]
        ind_upper =indices[-1] + 1 

    return [lower, upper], [ind_lower,ind_upper ]

def detection_test(fitter, molecule, min_wavelength, max_wavelength,
                   grid_name, data_name, 
                   filename, molecule_baseline=None,baseline_wavelength=[],
                   model_full=None, 
                   opa_kwargs={},plot=True):
    """
    Computes the detection significance of a molecule given a grid name, data name, 
    filename
    """
    try: 
        import dynesty 
        from dynesty import utils as dyfunc 
    except ModuleNotFoundError: 
        raise Exception('You are running a PICASO that requires the additional package `dynesty`. Please install with `pip install dynesty` and rerun this function')

    wlgrid_center = fitter.data[data_name]['wlgrid_center']
    y_data = fitter.data[data_name]['y_data']
    e_data = fitter.data[data_name]['e_data']
    
    index = fitter.list_of_files[grid_name].index(filename)
    
    shift = fitter.offsets[grid_name][data_name][index]
    
    xr_data = xr.load_dataset(filename)
        
    opa = opannection(**opa_kwargs)
    case = input_xarray(xr_data, opa)        
    og_atmo = copy.deepcopy(case.inputs['atmosphere']['profile'])
    
    if isinstance(model_full,type(None)):
        model_full = xr_data.data_vars['transit_depth']
        wavelength = xr_data.coords['wavelength']
        wavelength, model_full = mean_regrid(wavelength,model_full,newx=wlgrid_center)
        model_full = model_full + shift
    
    
    #
    double_gauss =False
    if isinstance(molecule_baseline,str):
        molecule = [molecule, molecule_baseline]
        double_gauss = True
        if len(baseline_wavelength)==2: 
            min_wavelength_add = sorted(baseline_wavelength)[0]
            max_wavelength_add = sorted(baseline_wavelength)[1]
        else: 
            min_wavelength_add = min_wavelength
            max_wavelength_add = max_wavelength

    case.atmosphere(df = og_atmo,exclude_mol=molecule)
    df= case.spectrum(opa, full_output=True,calculation='transmission') #note the new last key 
    wno, model_exclude  = df['wavenumber'] , df['transit_depth']
    wno, model_exclude = mean_regrid(wno,model_exclude,newx=np.sort(1e4/wlgrid_center))
    model_exclude = model_exclude + shift 
    out = pd.DataFrame({
        'wno':wno, 
        'wavelength':1e4/wno,
        'model_exclude':model_exclude})
    out = out.sort_values(by='wavelength')
    model_exclude = out['model_exclude']
    wavelength = out['wavelength']
    
    if plot: 
        fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(15,10))
        ax[0].plot(wlgrid_center, model_full,color='blue',label='Full Model')
        ax[0].plot(wlgrid_center, model_exclude,color='red',label=f'Without {molecule}')
        ax[0].errorbar(wlgrid_center, y_data, yerr=e_data,fmt='ok')
        ax[0].set_xlabel('wavelength [microns]')
        ax[0].set_ylabel('transit depth') 
        ax[0].legend(fontsize=12)
    
    residual_model = model_full-model_exclude
    residual_data = y_data-model_exclude
    if plot: 
        ax[1].plot(wlgrid_center, residual_model, color='blue',label='Residual Model')
        ax[1].errorbar(wlgrid_center, residual_data, yerr=e_data,fmt='ok',label='Residual in data')
        ax[1].set_xlabel('wavelength [microns]')
        ax[1].set_ylabel('delta transit depth')     
        ax[1].legend(fontsize=12)

    #defining gaussian model-params are centeral wavlenegth, width, amplitude, and a constant "DC" offset
    def model_gauss(wlgrid, lam0, sig, Amp,cst):
        return (Amp*np.exp(-(wlgrid-lam0)**2/sig**2)+cst)/1e6

    def model_double_gauss(wlgrid, lam01, sig1, Amp1,cst1,lam02, sig2, Amp2,cst2):
        return ((Amp1*np.exp(-(wlgrid-lam01)**2/sig1**2)+cst1)/1e6 + 
                (Amp2*np.exp(-(wlgrid-lam02)**2/sig2**2)+cst2)/1e6    )
    #TK ADD IN DOUBLE LOGLIKE AND DOUBLE PRIOR
    #likelihood function
    def loglike_gauss(theta):
        logAmp, lam0,logsig,cst=theta #fitting for the "log" amplitude and witdths b/c why not...could try linear to see if it affects answer
        mod=model_gauss(wlgrid_center, lam0, 10**logsig, 10**logAmp,cst) #evaluating model
        lnl=-0.5*np.sum((residual_data-mod)**2/e_data**2) #the equation for -1/2 chi-square....
        return lnl
    def loglike_double_gauss(theta):
        logAmp1, lam01,logsig1,cst1,logAmp2, lam02,logsig2,cst2=theta #fitting for the "log" amplitude and witdths b/c why not...could try linear to see if it affects answer
        mod=model_double_gauss(wlgrid_center, lam01, 10**logsig1, 10**logAmp1, cst1,
                                       lam02, 10**logsig2, 10**logAmp2, cst2) #evaluating model
        lnl=-0.5*np.sum((residual_data-mod)**2/e_data**2) #the equation for -1/2 chi-square....
        return lnl

    #prior transform
    def prior_transform_gauss(theta):
        logAmp, lam0,logsig,cst=theta
        logAmp=-1+(4.5+1)*logAmp
        lam0=min_wavelength+(max_wavelength-min_wavelength)*lam0 
        logsig=-2+(1+2)*logsig
        cst=-200+(400)*cst
        return logAmp, lam0,logsig,cst
    def prior_transform_double_gauss(theta):
        logAmp1, lam01,logsig1,cst1,logAmp2, lam02,logsig2,cst2=theta
        logAmp1=-1+(4.5+1)*logAmp1
        lam01=min_wavelength+(max_wavelength-min_wavelength)*lam01 
        logsig1=-2+(1+2)*logsig1
        cst1=-200+(400)*cst1
        logAmp2=-1+(4.5+1)*logAmp2
        lam02=min_wavelength_add+(max_wavelength_add-min_wavelength_add)*lam02
        logsig2=-2+(1+2)*logsig2
        cst2=-200+(400)*cst2
        return logAmp1, lam01,logsig1,cst1,logAmp2, lam02,logsig2,cst2 
    
    Nproc=4  #number of processors for multi processing--best if you can run on a 12 core+ node or something
    Nlive=500 #number of nested sampling live points

    #setting up multi-threading and sampler     
    #pool = Pool(processes=Nproc)
    results = {}
    models = []
    if double_gauss:
        models += ['double']
        Nparam=8  #number of parameters--make sure it is the same as what is in prior and loglike
        results['double'] = dynesty.NestedSampler(loglike_double_gauss, prior_transform_double_gauss, ndim=Nparam,
                                            bound='multi', sample='auto', nlive=Nlive)#,
                                            #pool=pool, queue_size=Nproc)
    #run single for comparison 
    Nparam = 4
    models += ['single']
    results['single'] = dynesty.NestedSampler(loglike_gauss, prior_transform_gauss, ndim=Nparam,
                                        bound='multi', sample='auto', nlive=Nlive)#,
                                        #pool=pool, queue_size=Nproc)
    keys = list(results.keys())
    for dsampler in keys:
        results[dsampler].run_nested()
        #GAUSS RESULTS
        results[f'dres_{dsampler}'] = results[dsampler].results #results
        ##grabbing the final evidence--will be used for bayes factor (see Dynesty documnetation)
        results[f'logZ_{dsampler}'] = results[f'dres_{dsampler}'].logz[-1] 
        samples, weights = results[f'dres_{dsampler}'].samples, np.exp(results[f'dres_{dsampler}'].logwt - results[f'dres_{dsampler}'].logz[-1])
        results[f'samp_{dsampler}'] = dyfunc.resample_equal(samples, weights)
    
    #flat line test
    def model_line(wlgrid,cst):
        #flat line slope = 0 
        return (cst+wlgrid*0. )/1e6

    #loglike with 
    def loglike_line(theta):
        cst=theta
        mod=model_line(wlgrid_center, cst)
        lnl=-0.5*np.sum((residual_data-mod)**2/e_data**2)
        return lnl

    #prior cube 
    def prior_transform_line(theta):
        cst=theta
        cst=-200+(2000)*cst
        return cst 
    
    Nparam=1
    results['line'] = dynesty.NestedSampler(loglike_line, prior_transform_line, ndim=Nparam,
                                        bound='multi', sample='auto', nlive=Nlive#,
                                        #pool=pool, queue_size=Nproc
                                    )
    
    results['line'].run_nested()
    results['dres_line'] = results['line'].results
    results['logZ_line'] = results['dres_line'].logz[-1] 
    samples, weights = results['dres_line'].samples, np.exp(results['dres_line'].logwt - results['dres_line'].logz[-1])
    results['samp_line'] = dyfunc.resample_equal(samples, weights)
    
    
    if plot:
        ax[2].errorbar(wlgrid_center, residual_data,yerr=e_data,fmt='ob',ms=3,label='Residual Data')
        
        samp_gauss = results['samp_single']
        for i in range(samp_gauss.shape[0]):
            logAmp, lam0,logsig,cst=samp_gauss[i,:]
            mod=model_gauss(wlgrid_center, lam0, 10**logsig, 10**logAmp,cst)
            ax[2].plot(wlgrid_center, mod,alpha=0.01,color='purple')

        ax[2].plot(wlgrid_center, mod,alpha=0.5,color='purple')

        if double_gauss:
            samp_gauss = results['samp_double']
            for i in range(samp_gauss.shape[0]):
                logAmp1, lam01,logsig1,cst1,logAmp2, lam02,logsig2,cst2=samp_gauss[i,:]
                mod=model_double_gauss(wlgrid_center, lam01, 10**logsig1, 10**logAmp1, cst1,
                                   lam02, 10**logsig2, 10**logAmp2, cst2)
                ax[2].plot(wlgrid_center, mod,alpha=0.01,color='orange')

            ax[2].plot(wlgrid_center, mod,alpha=0.5,color='orange')

        samp_line = results['samp_line']
        for i in range(samp_line.shape[0]):
            cst=samp_line[i,:]
            mod=model_line(wlgrid_center, cst)
            ax[2].plot(wlgrid_center, mod,alpha=0.01,color='grey')
       
        ax[2].plot(wlgrid_center, mod,alpha=0.5,color='grey',label='Constant Fit Ensemble')
        
        ax[2].set_xlabel('wavelength [microns]')
        ax[2].set_ylabel('Delta Transit Depth') 
        ax[2].legend(fontsize=12) 

    results['sigma_single_v_line'],results['lnB_single_v_line']= sigma(
                                        results['logZ_single'], results['logZ_line'])

    if double_gauss: 
        results['sigma_double_v_single'],results['lnB_double_v_single']= sigma(
                                    results['logZ_double'], results['logZ_single'])
    return results
        


def chi_squared(data,data_err,model,numparams):
    """
    Compute reduced chi squared assuming DOF = ndata_pts - num parameters  
    """
    
    chi_squared = np.sum(((data-model)/(data_err))**2)/(len(data)-(numparams))
    
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
                plt.rcParams['font.serif'] = 'DejaVu Sans'
                plt.rcParams['mathtext.fontset'] = 'stixsans'
                plt.rcParams['axes.prop_cycle'] = \
                plt.cycler(color=["xkcd:salmon", "dodgerblue", "sandybrown", 'cadetblue', 'orchid', 'lightblue'])
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



#equations in Trotta 2008
def sigma(lnz1,lnz2):
    """ 
    Author: Mike Line (mrline@asu.edu)
    
    Computes equatiosn from Trotta 2008
    https://ui.adsabs.harvard.edu/abs/2008ConPh..49...71T/abstract
    
    Tests model preference from model 1 vs model 2
    Returns Bayes Factor (Eqn. 21) and sigma significance (Table 2)
    
    
    Parameters
    ----------
    lnz1 : float 
        evidence model 1
    lnz2 : float 
        evidence model 2
    
    Returns 
    -------
    sigma, bayes factor
    """

    # This is the python version of sigma.pro

    lnB = lnz1 - lnz2
    logp = np.arange(-300.00,0.00,.1) #reverse order
    logp = logp[::-1] # original order
    P = 10.0**logp
    Barr = -1./(np.exp(1)*P*np.log(P))

    sigma = np.arange(0.1,100.10,.01)
    p_p = sp.special.erfc(sigma/np.sqrt(2.0))
    B = np.exp(lnB)
    pvalue = 10.0**np.interp(np.log10(B),np.log10(Barr),np.log10(P))
    sig = np.interp(pvalue,p_p[::-1],sigma[::-1])
    
    return sig , lnB

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
        param_flt = param.get('value',param)
        #get unit
        #if isinstance(param.get('unit',np.nan),str): 
        #    try: 
        #        param_unit = u.Unit(param.get('unit'))
        #        param = param_flt*param_unit
        #    except ValueError: 
        #        param = param_flt
        #        pass 
        #else: 
        #    param = param_flt
        param = param_flt

    if isinstance(param, str):
        if len(param.split(' ')) > 1: 
            #float value
            try: 
                param_flt = float(param.split(' ')[0])
            except ValueError: 
                param_flt = np.nan
                pass
            param = param_flt
            #unit value
            #if not np.isnan(param_flt):
            #    try: 
            #        param_unit = u.Unit(''.join(param.split(' ')[1:]))
            #        param = param_flt*param_unit
            #    except ValueError: 
            #        param = param_flt
            #        pass            
    return param
