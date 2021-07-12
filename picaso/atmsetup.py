from .elements import ELEMENTS as ele 
import json 
import os
from .io_utils import read_json
import astropy.units as u
import astropy.constants as c
import pandas as pd
import warnings 
import numpy as np
from .wavelength import get_cld_input_grid, regrid
from numba import jit
import pysynphot as psyn
import math 

__refdata__ = os.environ.get('picaso_refdata')

class ATMSETUP():
    """
    Reads in default source configuration from JSON and creates a full atmosphere class. 

    - Gets PT profile 
    - Computes mean molecular weight, density, column density, mixing ratios 
    - Gets cloud profile 
    - Gets stellar profile 

    """
    def __init__(self, config):
        if __refdata__ is None:
            warnings.warn("Reference data has not been initialized, some functionality will be crippled", UserWarning)
        else: 
            self.ref_dir = os.path.join(__refdata__)
        self.input = config
        self.warnings = []
        self.planet = type('planet', (object,),{})
        self.layer = {}
        self.level = {}
        self.longitude , self.latitude = self.input['disco']['longitude'], self.input['disco']['latitude']
        self.get_constants()

    def get_constants(self):
        """
        This function gets all conversion factors based on input units and all 
        constants. Goal here is to get everything to cgs and make sure we are only 
        call lengths of things once (e.g. number of wavelength points, number of layers etc)

        Some stuff will be added to this in other routines (e.g. number of layers)
        """
        self.c = type('c', (object,),{})
        #conversion units
        self.c.pconv = 1e6 #convert bars to cgs 

        #physical constants
        self.c.k_b = (c.k_B.to(u.erg / u.K).value)      
        self.c.G = c.G.to(u.cm*u.cm*u.cm/u.g/u.s/u.s).value 
        self.c.amu = c.u.to(u.g).value #grams
        self.c.rgas = c.R.value
        self.c.pi = np.pi

        #code constants
        self.c.ngangle = self.input['disco']['num_gangle']
        self.c.ntangle = self.input['disco']['num_tangle']

        return

    def get_profile_3d(self):
        """
        Get profile from file or from user input file or direct pandas dataframe. If PT profile 
        is not given, use parameterization 

        Currently only needs inputs from config file

        Parameters
        ----------
        lat : array
            Array of latitudes (radians)
        lon : array 
            Array of longitudes (radians)

        Todo
        ----
        - Add regridding to this by having users be able to set a different nlevel than the input cloud code is
        """
        #get chemistry input from configuration
        #SET DIMENSIONALITY
        self.dimension = '3d'
        latitude, longitude = self.latitude*180/np.pi, self.longitude*180/np.pi
        read_3d = self.input['atmosphere']['profile'] #huge dictionary with [lat][lon][bundle]

        self.c.nlevel = read_3d[int(latitude[0])][int(longitude[0])].shape[0]
        self.c.nlayer = self.c.nlevel - 1  
        ng , nt = self.c.ngangle, self.c.ntangle

        self.level['temperature'] = np.zeros((self.c.nlevel, ng, nt))
        self.level['pressure'] = np.zeros((self.c.nlevel, ng, nt))
        self.level['electrons'] = np.zeros((self.c.nlevel, ng, nt))

        self.layer['temperature'] = np.zeros((self.c.nlayer, ng, nt))
        self.layer['pressure'] = np.zeros((self.c.nlayer, ng, nt))
        self.layer['electrons'] = np.zeros((self.c.nlayer, ng, nt))

        #COMPUTE THE MOLECULAT WEIGHTS OF THE MOLECULES
        weights = pd.DataFrame({})

        #Cycle through each column
        self.molecules = np.array([],dtype=str)

        # loop over gangles and tangles 
        first = True
        electrons = False
        for g in range(ng):
            for t in range(nt):
                read = read_3d[int(latitude[t])][int(longitude[g])].sort_values('pressure').reset_index(drop=True)

                #on the first pass look through all the molecules, parse out the electrons and 
                #add warnings for molecules that aren't recognized
                if first:
                    for i in read.keys():
                        if i in ['pressure', 'temperature']: continue
                        try:
                            weights[i] = pd.Series([self.get_weights([i])[i]])
                            self.molecules = np.concatenate((self.molecules ,np.array([i])))
                        except:
                            if i == 'e-':
                                electrons = True
                            else:                   #don't raise exception, instead add user warning that a column has been automatically skipped
                                self.add_warnings("Ignoring %s in input file, not recognized molecule" % i)
                                warnings.warn("Ignoring %s in input file, not a recognized molecule" % i, UserWarning)
                    
                    first = False
                    self.weights = weights 
                    num_mol = len(weights.keys())
                    self.layer['mixingratios'] = np.zeros((self.c.nlayer, num_mol, ng, nt))
                    self.level['mixingratios'] = np.zeros((self.c.nlevel,num_mol,  ng, nt))


                if electrons:
                    self.level['electrons'][:,g,t] = read['e-'].values
                    self.layer['electrons'][:,g,t] = 0.5*(self.level['electrons'][1:,g,t] + self.level['electrons'][:-1,g,t])
                #DEFINE MIXING RATIOS
                self.level['mixingratios'][:,:,g,t] = read[list(weights.keys())]

                self.layer['mixingratios'][:,:,g,t] = 0.5*(self.level['mixingratios'][1:,:,g,t] + 
                                    self.level['mixingratios'][0:-1,:,g,t])

                self.level['temperature'][:,g,t] = read['temperature'].values
                self.level['pressure'][:,g,t] = read['pressure'].values*self.c.pconv #CONVERTING BARS TO DYN/CM2
                self.layer['temperature'][:,g,t] = 0.5*(self.level['temperature'][1:,g,t] + self.level['temperature'][:-1,g,t])
                self.layer['pressure'][:,g,t] = np.sqrt(self.level['pressure'][1:,g,t] * self.level['pressure'][:-1,g,t])

    def get_profile(self):
        """
        Get profile from file or from user input file or direct pandas dataframe. If PT profile 
        is not given, use parameterization 

        Currently only needs inputs from config file

        Todo
        ----
        - Add regridding to this by having users be able to set a different nlevel than the input cloud code is
        """
        #get chemistry input from configuration
        #SET DIMENSIONALITY
        self.dimension = '1d'

        read = self.input['atmosphere']['profile']

        
        #COMPUTE THE MOLECULAT WEIGHTS OF THE MOLECULES
        weights = pd.DataFrame({})

        #Cycle through each column
        self.molecules = np.array([],dtype=str)

        for i in read.keys():
            if i in ['pressure', 'temperature']: continue
            try:
                weights[i] = pd.Series([self.get_weights([i])[i]])
                self.molecules = np.concatenate((self.molecules ,np.array([i])))
            except:
                if i == 'e-':
                    self.level['electrons'] = read['e-'].values
                    self.layer['electrons'] = 0.5*(self.level['electrons'][1:] + self.level['electrons'][:-1])
                else:                   #don't raise exception, instead add user warning that a column has been automatically skipped
                    self.add_warnings("Ignoring %s in input file, not recognized molecule" % i)
                    warnings.warn("Ignoring %s in input file, not a recognized molecule" % i, UserWarning)
        
        self.weights = weights 

        #DEFINE MIXING RATIOS
        self.level['mixingratios'] = read[list(weights.keys())]
        self.layer['mixingratios'] = 0.5*(self.level['mixingratios'][1:].reset_index(drop=True) + 
                            self.level['mixingratios'][0:-1].reset_index(drop=True))

        self.level['temperature'] = read['temperature'].values
        self.level['pressure'] = read['pressure'].values*self.c.pconv #CONVERTING BARS TO DYN/CM2
        self.layer['temperature'] = 0.5*(self.level['temperature'][1:] + self.level['temperature'][:-1])
        self.layer['pressure'] = np.sqrt(self.level['pressure'][1:] * self.level['pressure'][:-1])


        #Define nlevel and nlayers after profile has been built
        self.c.nlevel = self.level['mixingratios'].shape[0]

        self.c.nlayer = self.c.nlevel - 1


    def calc_PT(self,logPc, T, logKir, logg1):
        """
        Calculates parameterized PT profile from Guillot. This isntance is here 
        primary for the retrieval scheme, so this can be updated

        Parameters
        ----------

        T : float 
        logKir : float
        logg1 : float 
        logPc : float 
        """
        return np.zeros(len(logPc)) + T #return isothermal for now

    def get_needed_continuum(self,available_ray_mol):
        """
        This will define which molecules are needed for the continuum opacities. THis is based on 
        temperature and molecules. Eventually CIA's will expand but we may not necessarily 
        want all of them. 
        'wno','h2h2','h2he','h2h','h2ch4','h2n2']

        Todo
        ----
        - Add in temperature dependent to negate h- and things when not necessary
        """
        self.rayleigh_molecules = []
        self.continuum_molecules = []
        if "H2" in self.molecules:
            self.continuum_molecules += [['H2','H2']]
        if ("H2" in self.molecules) and ("He" in self.molecules):
            self.continuum_molecules += [['H2','He']]
        if ("H2" in self.molecules) and ("N2" in self.molecules):
            self.continuum_molecules += [['H2','N2']]   
        if  ("H2" in self.molecules) and ("H" in self.molecules):
            self.continuum_molecules += [['H2','H']]
        if  ("H2" in self.molecules) and ("CH4" in self.molecules):
            self.continuum_molecules += [['H2','CH4']]
        if ("H-" in self.molecules):
            self.continuum_molecules += [['H-','bf']]
        if ("H" in self.molecules) and ("electrons" in self.level.keys()):
            self.continuum_molecules += [['H-','ff']]
        if ("H2" in self.molecules) and ("electrons" in self.level.keys()):
            self.continuum_molecules += [['H2-','']]
        #now we can remove continuum molecules from self.molecules to keep them separate 
        if 'H+' in ['H','H2-','H2','H-','He','N2']: self.add_warnings('No H+ continuum opacity included')

        #and rayleigh opacity
        for i in self.molecules: 
            if i in available_ray_mol : self.rayleigh_molecules += [i]

        #self.molecules = np.array([ x for x in self.molecules if x not in ['H','H2-','H2','H-','He','N2', 'H+'] ])

    def get_weights(self, molecule):
        """
        Automatically gets mean molecular weights of any molecule. Requires that 
        user inputs case sensitive molecules i.e. TiO instead of TIO. 

        Parameters
        ----------
        molecule : str or list
            any molecule string e.g. "H2O", "CH4" etc "TiO" or ["H2O", 'CH4']

        Returns
        -------
        dict 
            molecule as keys with molecular weights as value
        """
        weights = {}
        if not isinstance(molecule,list):
            molecule = [molecule]

        for i in molecule:
            molecule_list = []
            for j in range(0,len(i)):
                try:
                    molecule_list += [float(i[j])]
                except: 
                    if i[j].isupper(): molecule_list += [i[j]] 
                    elif i[j].islower(): molecule_list[-1] =  molecule_list[-1] + i[j]
            totmass=0
            for j in range(0,len(molecule_list)): 
                
                if isinstance(molecule_list[j],str):
                    elem = ele[molecule_list[j]]
                    try:
                        num = float(molecule_list[j+1])
                    except: 
                        num = 1 
                    totmass += elem.mass * num
            weights[i] = totmass
        return weights


    def get_mmw(self):
        """
        Returns the mean molecular weight of the atmosphere 
        """
        if self.dimension=='1d':
            weighted_matrix = self.level['mixingratios'].values @ self.weights.values[0]
        elif self.dimension=='3d':
            weighted_matrix=np.zeros((self.c.nlevel, self.c.ngangle, self.c.ntangle))
            for g in range(self.c.ngangle):
                for t in range(self.c.ntangle):
                    weighted_matrix[:,g,t] = self.level['mixingratios'][:,:,g,t] @ self.weights.values[0]

        #levels are the edges
        self.level['mmw'] = weighted_matrix
        #layer is the midpoint
        self.layer['mmw'] = 0.5*(weighted_matrix[:-1]+weighted_matrix[1:])

        return 

    def get_density(self):
        """
        Calculates density of atmospheres used on TP profile: LEVEL
        """
        self.level['den'] = self.level['pressure'] / (self.c.k_b * self.level['temperature']) 
        return

    def get_altitude(self, p_reference=1,constant_gravity=False):
        """
        Calculates z and gravity  

        Parameters
        ----------
        p_reference : float
            Reference pressure for radius. (bars)
        constant_gravity : bool 
            creates zero altitude dependence in the height
        """
        #convert to dyn/cm2
        p_reference = p_reference*self.c.pconv

        if np.isnan(self.planet.radius):
            constant_gravity = True

        #set zero arays of things we want out 
        nlevel = self.c.nlevel

        mmw = self.level['mmw'] * self.c.amu #make sure mmw in grams
        tlevel = self.level['temperature']
        plevel = self.level['pressure']

        z = np.zeros(np.shape(tlevel)) + self.planet.radius
        dz = np.zeros(np.shape(tlevel)) 
        gravity = np.zeros(np.shape(tlevel))  

        for i in np.where(plevel>p_reference)[0]-1:
            if constant_gravity:
                gravity[i] = self.planet.gravity
            else:
                gravity[i] = self.c.G * self.planet.mass / ( z[i] )**2

            scale_h = self.c.k_b * tlevel[i] / (mmw[i] * gravity[i])
            dz[i] = scale_h * (np.log(plevel[i+1] / plevel[i])) #from eddysed
            z[i+1] = z[i] - dz[i]

        for i in np.where(plevel<=p_reference)[0][::-1][:-1]:
            if constant_gravity:
                gravity[i] = self.planet.gravity
            else:
                gravity[i] = self.c.G * self.planet.mass / ( z[i] )**2  

            scale_h = self.c.k_b * tlevel[i] / (mmw[i] * gravity[i])
            dz[i] = scale_h*(np.log(plevel[i+1]/ plevel[i]))
            z[i-1] = z[i] + dz[i]

        self.level['z'] = z
        self.level['dz'] = dz
        #for get_column_density calculation below we want gravity at layers
        self.layer['gravity'] = 0.5*(gravity[0:-1] + gravity[1:])
        self.layer['gravity'][0] = self.layer['gravity'][1]
        self.layer['gravity'][-1] = self.layer['gravity'][-2]
        
    def get_column_density(self):
        """
        Calculates the column desntiy based on TP profile: LAYER
        """
        self.layer['colden'] = (self.level['pressure'][1:] - self.level['pressure'][:-1] ) / self.layer['gravity'] 
        return


    def get_clouds(self, wno):
        """
        Get cloud properties from .cld input returned from eddysed. The eddysed file should have the following specifications 

        1) Have the following column names (any order)  opd g0 w0 

        2) Be white space delimeted 

        3) Has to have values for pressure levels (N) and wwavenumbers (M). The row order should go:

        level wavenumber opd w0 g0
        1.   1.   ... . .
        1.   2.   ... . .
        1.   3.   ... . .
        .     . ... . .
        .     . ... . .
        1.   M.   ... . .
        2.   1.   ... . .
        .     . ... . .
        N.   .  ... . .

        Warning
        -------
        The order of the rows is very important because each column will be transformed 
        into a matrix that has the size: [nlayer,nwave]. 

        Parameters
        ----------
        wno : array
            Array in ascending order of wavenumbers. This is used to regrid the cloud output

        Todo 
        ----
        - Take regridding out of here and add it to `justdoit`
        """
        self.input_wno = self.input['clouds']['wavenumber']

        self.c.output_npts_wave = np.size(wno)
        
        #if a cloud profile exists 
        if ((self.dimension=='1d') & (not isinstance(self.input_wno, type(None)))) :
            self.c.input_npts_wave = len(self.input_wno)

            #read in the file that was supplied         
            cld_input = self.input['clouds']['profile'] 
            
            #then reshape and regrid inputs to be a nice matrix that is nlayer by nwave
            #total extinction optical depth 
            opd = np.reshape(cld_input['opd'].values, (self.c.nlayer,self.c.input_npts_wave))
            opd = regrid(opd, self.input_wno, wno)
            self.layer['cloud'] = {'opd': opd}
            #cloud assymetry parameter
            g0 = np.reshape(cld_input['g0'].values, (self.c.nlayer,self.c.input_npts_wave))
            g0 = regrid(g0, self.input_wno, wno)
            self.layer['cloud']['g0'] = g0
            #cloud single scattering albedo 
            w0 = np.reshape(cld_input['w0'].values, (self.c.nlayer,self.c.input_npts_wave))
            w0 = regrid(w0, self.input_wno, wno)
            self.layer['cloud']['w0'] = w0  

        #if no filepath was given and nothing was given for g0/w0, then assume the run is cloud free and give zeros for all thi stuff         
        elif (isinstance(self.input['clouds']['profile'] , type(None)) and (self.dimension=='1d')):

            zeros = np.zeros((self.c.nlayer,self.c.output_npts_wave))
            self.layer['cloud'] = {'w0': zeros}
            self.layer['cloud']['g0'] = zeros
            self.layer['cloud']['opd'] = zeros
        # 3D without clouds           
        elif (isinstance(self.input['clouds']['profile'] , type(None)) and (self.dimension=='3d')):
            self.layer['cloud'] = {'w0': np.zeros((self.c.nlayer,self.c.output_npts_wave,self.c.ngangle,self.c.ntangle))}
            self.layer['cloud']['g0'] = np.zeros((self.c.nlayer,self.c.output_npts_wave,self.c.ngangle,self.c.ntangle))
            self.layer['cloud']['opd'] = np.zeros((self.c.nlayer,self.c.output_npts_wave,self.c.ngangle,self.c.ntangle))
        #3D with clouds
        elif ((self.dimension=='3d') & (not isinstance(self.input_wno, type(None)))):
            self.c.input_npts_wave = len(self.input_wno)
            latitude, longitude = self.latitude*180/np.pi, self.longitude*180/np.pi
            cld_input = self.input['clouds']['profile'] 

            opd = np.zeros((self.c.nlayer,self.c.output_npts_wave,self.c.ngangle,self.c.ntangle))
            g0 = np.zeros((self.c.nlayer,self.c.output_npts_wave,self.c.ngangle,self.c.ntangle)) 
            w0 = np.zeros((self.c.nlayer,self.c.output_npts_wave,self.c.ngangle,self.c.ntangle))

            #stick in clouds that are gangle and tangle dependent 
            for g in range(self.c.ngangle):
                for t in range(self.c.ntangle):

                    data = cld_input[int(latitude[t])][int(longitude[g])]

                    #make sure cloud input has the correct number of waves and PT points
                    assert data.shape[0] == self.c.nlayer*self.c.input_npts_wave, "Cloud input file is not on the same grid as the input PT/Angles profile:"

                    #Then, reshape and regrid inputs to be a nice matrix that is nlayer by nwave
                    #total extinction optical depth 
                    opd_lowres = np.reshape(data['opd'].values, (self.c.nlayer,self.c.input_npts_wave))
                    opd[:,:,g,t] = regrid(opd_lowres, self.input_wno, wno)

                    #cloud assymetry parameter
                    g0_lowres = np.reshape(data['g0'].values, (self.c.nlayer,self.c.input_npts_wave))
                    g0[:,:,g,t] = regrid(g0_lowres, self.input_wno, wno)
                    

                    #cloud single scattering albedo 
                    w0_lowres = np.reshape(data['w0'].values, (self.c.nlayer,self.c.input_npts_wave))
                    w0[:,:,g,t] = regrid(w0_lowres, self.input_wno, wno)
                    
            self.layer['cloud'] = {'opd': opd}
            self.layer['cloud']['g0'] = g0
            self.layer['cloud']['w0'] = w0  
        else:

            raise Exception("CLD input not recognized. Either input a filepath, or input None")

        return



    def add_warnings(self, warn):
        """
        Accumulate warnings from ATMSETUP
        """
        self.warnings += [warn]
        return


    def get_surf_reflect(self,nwno):
        """
        Gets the surface reflectivity from input

        """
        self.surf_reflect = np.zeros(nwno)
        return

    def disect(self,g,t):
        """
        This disects the 3d input to a 1d input which is a function of a single gangle and tangle. 
        This makes it possible for us to get the opacities at each facet before we go into the 
        flux calculation

        Parameters
        ----------

        g : int 
            Gauss angle index 
        t : int 
            Tchebyshev angle index  
        """
        self.level['temperature'] = self.level['temperature'][:,g,t]
        self.level['pressure'] = self.level['pressure'][:,g,t]
        self.layer['temperature'] = self.layer['temperature'][:,g,t]
        self.layer['pressure'] = self.layer['pressure'][:,g,t]

        self.layer['mmw'] = self.layer['mmw'][:,g,t]
        self.layer['mixingratios'] = pd.DataFrame(self.layer['mixingratios'][:,:,g,t],columns=self.weights.keys())
        self.layer['colden'] = self.layer['colden'][:,g,t]
        self.layer['electrons'] = self.layer['electrons'][:,g,t]

        self.layer['cloud']['opd'] = self.layer['cloud']['opd'][:,:,g,t]
        self.layer['cloud']['g0']= self.layer['cloud']['g0'][:,:,g,t]
        self.layer['cloud']['w0'] = self.layer['cloud']['w0'][:,:,g,t]

    def as_dict(self):
        """
        Get output into picklable dict format 
        """
        df = {} 
        df['weights'] = self.weights
        df['layer'] = {}
        df['layer']['pressure_unit'] = 'bars'
        df['layer']['mixingratio_unit'] = 'volume/volume'
        df['layer']['temperature_unit'] = 'K'
        df['layer']['pressure'] = self.layer['pressure']/ self.c.pconv #bars
        df['layer']['mixingratios'] = self.layer['mixingratios']
        df['layer']['temperature'] = self.layer['temperature']
        df['layer']['column_density'] = self.layer['colden']
        df['layer']['mmw'] = self.layer['mmw']
        df['wavenumber'] = self.wavenumber
        df['wavenumber_unit'] = 'cm-1'

        df['layer']['cloud'] = {}
        df['layer']['cloud']['w0'] = self.layer['cloud']['w0']
        df['layer']['cloud']['g0'] = self.layer['cloud']['g0']
        df['layer']['cloud']['opd'] = self.layer['cloud']['opd']

        df['taugas'] = self.taugas
        df['tauray'] = self.tauray
        df['taucld'] = self.taucld

        df['level'] = {}
        df['level']['pressure'] = self.level['pressure']/ self.c.pconv #bars
        df['level']['temperature'] = self.level['temperature']

        df['latitude'] = self.latitude
        df['longitude'] = self.longitude

        df['star'] = {}
        df['star']['flux_unit'] = 'erg/cm2/s/cm'

        try: 
            x =  self.xint_at_top
            df['albedo_3d'] = x
            df['reflected_unit'] = 'albedo'
        except:
            pass 
    

        try: 
            x = self.flux_at_top
            df['thermal_unit'] = 'erg/cm2/s/cm'
            df['thermal_3d'] = x
        except:
            pass

        return df