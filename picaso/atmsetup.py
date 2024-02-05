from .elements import ELEMENTS as ele 
import json 
import os
import re
from .io_utils import read_json
import astropy.units as u
import astropy.constants as c
import pandas as pd
import warnings 
import numpy as np
from .wavelength import get_cld_input_grid
from .wavelength import regrid as regrid_cld
from numba import jit
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

        self.c.nlevel = self.input['atmosphere']['profile'].dims['pressure']
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
                ilat = list(read_3d.coords['lat'].values.astype(np.float32)).index(np.float32(latitude[t]))
                ilon = list(read_3d.coords['lon'].values.astype(np.float32)).index(np.float32(longitude[g]))
                #read = read_3d[int(latitude[t])][int(longitude[g])].sort_values('pressure').reset_index(drop=True)
                read = read_3d.isel(lon=ilon,lat=ilat).to_pandas().reset_index().drop(['lat','lon'],axis=1).sort_values('pressure')
                if 'phase' in read.keys():
                    read=read.drop('phase',axis=1)
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
                            else: #don't raise exception, instead add user warning that a column has been automatically skipped
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

    def get_needed_continuum(self,available_ray_mol,available_continuum):
        """
        This will define which molecules are needed for the continuum opacities. THis is based on 
        temperature and molecules. This is terrible code I

        Parameters
        ----------
        available_ray_mol : list of str
            list of available rayleigh molecules 
        available_continuum : list of str
            list of available continuum molecules 
        """
        self.rayleigh_molecules = []
        self.continuum_molecules = []

        simple_names = [convert_to_simple(i) for i in self.molecules]
        
        if "H2" in simple_names:
            self.continuum_molecules += [['H2','H2']]
        if ("H2" in simple_names) and ("He" in simple_names):
            self.continuum_molecules += [['H2','He']]
        if ("H2" in simple_names) and ("N2" in simple_names):
            self.continuum_molecules += [['H2','N2']]   
        if  ("H2" in simple_names) and ("H" in simple_names):
            self.continuum_molecules += [['H2','H']]
        if  ("H2" in simple_names) and ("CH4" in simple_names):
            self.continuum_molecules += [['H2','CH4']]
        if  ("N2" in simple_names):
            self.continuum_molecules += [['N2','N2']]
        if  ("CO2" in simple_names):
            self.continuum_molecules += [['CO2','CO2']]      
        if  ("O2" in simple_names):
            self.continuum_molecules += [['O2','O2']]    

        if ("H-" in simple_names):
            self.continuum_molecules += [['H-','bf']]
        if ("H" in simple_names) and ("electrons" in self.level.keys()):
            self.continuum_molecules += [['H-','ff']]
        if ("H2" in simple_names) and ("electrons" in self.level.keys()):
            self.continuum_molecules += [['H2-','']]
        #now we can remove continuum molecules from self.molecules to keep them separate 
        if 'H+' in ['H','H2-','H2','H-','He','N2']: self.add_warnings('No H+ continuum opacity included')

        #remove anything not in opacity file
        cm =[]
        for i in self.continuum_molecules: 
            if ''.join(i) in available_continuum: cm += [i]
        self.continuum_molecules = cm

        #and rayleigh opacity
        for i in simple_names: 
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
            Isotopologues should be specified with a dash separating the 
            elemtents. E.g. 12C-16O2 or 13C-16O2. If you only want to specify one 
            then you would have 12C-O2. 

        Returns
        -------
        dict 
            molecule as keys with molecular weights as value
        """    
        weights = {}
        separator='_' #e.g. C_16O2 NOT C-16O2
        if not isinstance(molecule,list):
                molecule = [molecule]
                
        for i in molecule:
            totmass = 0
            #this indicates the user has specified the isotope
            if separator in i: 
                elements = [separate_molecule_name(j) for j in i.split(separator)]
            else:    
                elements = separate_molecule_name(i)

            for iele in elements:
                if isinstance(iele,list):
                    #this is the user's specified isotope
                    if len(iele)==1:
                        iele = iele[0]
                        iso_num = 'main'
                    else:
                        iso_num = int(iele[0])
                        iele = iele[1]
                else: 
                    iso_num = 'main'

                sep = separate_string_number(iele)
                if len(sep)==1:
                    el, num = sep[0], 1
                else: 
                    el, num = sep
                #default isotope
                if iso_num=='main':
                    iso_num = list(ele[el].isotopes.keys())[0] 
                totmass += ele[el].isotopes[iso_num].mass*float(num)

            weights[i]=totmass
        
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
        units of cm-3
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
        
        if p_reference >= np.max(plevel):
            p_reference = np.max(plevel)
        else: 
            #choose a reference pressure that is on the 
            #user specified pressure grid
            #if you dont do this your model becomes 
            #highly sensitive to # of layers used 
            p_reference = plevel[plevel>=p_reference][0]

        z = np.zeros(np.shape(tlevel)) + self.planet.radius
        dz = np.zeros(np.shape(tlevel)) 
        gravity = np.zeros(np.shape(tlevel))  
        #unique avoids duplicates for 3d grids where pressure is repeated for ngangle,ntangle
        #would break for nonuniform pressure grids 
        indx = np.unique(np.where(plevel>p_reference)[0]) 
        #if there are any pressures less than the reference pressure
        if len(indx)>0:
            for i in indx-1:
                
                if constant_gravity:
                    gravity[i] = self.planet.gravity
                else:
                    gravity[i] = self.c.G * self.planet.mass / ( z[i] )**2

                scale_h = self.c.k_b * tlevel[i] / (mmw[i] * gravity[i])
                dz[i] = scale_h * (np.log(plevel[i+1] / plevel[i])) #from eddysed
                z[i+1] = z[i] - dz[i]

        for i in np.unique(np.where(plevel<=p_reference)[0])[::-1][:-1]:#unique to avoid 3d bug
            
            if constant_gravity:
                gravity[i] = self.planet.gravity
            else:
                gravity[i] = self.c.G * self.planet.mass / ( z[i] )**2  

            scale_h = self.c.k_b * tlevel[i] / (mmw[i] * gravity[i])
            dz[i] = scale_h*(np.log(plevel[i]/ plevel[i-1]))#plevel[i+1]/ plevel[i]))
            z[i-1] = z[i] + dz[i]

        self.level['z'] = z
        self.level['dz'] = dz

        #for get_column_density calculation below we want gravity at layers
        self.layer['gravity'] = 0.5*(gravity[0:-1] + gravity[1:])
        
    def get_column_density(self):
        """
        Calculates the column desntiy based on TP profile: LAYER
        unit = g/cm2 = pressure/gravity =dyne/cm2/gravity = (g*cm/s2)/cm2/(cm/s2)=g/cm2
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
        #check to see if regridding is necessary
        regrid = True 
        if np.array_equal(self.input_wno, wno): regrid = False

        self.c.output_npts_wave = np.size(wno)
        
        #if a cloud profile exists 
        if ((self.dimension=='1d') & (not isinstance(self.input_wno, type(None)))) :
            self.c.input_npts_wave = len(self.input_wno)

            #read in the file that was supplied         
            cld_input = self.input['clouds']['profile'] 
            
            #then reshape and regrid inputs to be a nice matrix that is nlayer by nwave
            #total extinction optical depth 
            opd = np.reshape(cld_input['opd'].values, (self.c.nlayer,self.c.input_npts_wave))
            if regrid: opd = regrid_cld(opd, self.input_wno, wno)
            self.layer['cloud'] = {'opd': opd}
            #cloud assymetry parameter
            g0 = np.reshape(cld_input['g0'].values, (self.c.nlayer,self.c.input_npts_wave))
            if regrid: g0 = regrid_cld(g0, self.input_wno, wno)
            self.layer['cloud']['g0'] = g0
            #cloud single scattering albedo 
            w0 = np.reshape(cld_input['w0'].values, (self.c.nlayer,self.c.input_npts_wave))
            if regrid: w0 = regrid_cld(w0, self.input_wno, wno)
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
            cld_input = cld_input.sortby('wno').sortby('pressure')
            if regrid: cld_input = cld_input.interp(wno = wno)
            if [i for i in cld_input.dims] != ["pressure","wno","lon", "lat"]:
                opd = cld_input['opd'].transpose("pressure","wno","lon", "lat").values
                g0 = cld_input['g0'].transpose("pressure","wno","lon", "lat").values
                w0 = cld_input['w0'].transpose("pressure","wno","lon", "lat").values
            else: 
                opd = cld_input['opd'].values
                g0 = cld_input['g0'].values
                w0 = cld_input['w0'].values                
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
        
        #return the level fluxes if the user requests that particular output
        if self.get_lvl_flux:
            if not isinstance(getattr(self,'lvl_output_thermal',None), type(None)):
                df['level']['thermal_fluxes'] = self.lvl_output_thermal
            if not isinstance(getattr(self,'lvl_output_reflected',None), type(None)):    
                df['level']['reflected_fluxes'] = self.lvl_output_reflected

        df['latitude'] = self.latitude
        df['longitude'] = self.longitude

        df['star'] = {}
        df['star']['flux_unit'] = 'erg/cm2/s/cm'
        
        try: 
            df['level']['dz'] = self.level['dz']
            df['level']['z'] = self.level['z']
        except: 
            pass

        try: 
            x =  self.xint_at_top
            df['albedo_3d'] = x
            df['reflected_unit'] = 'albedo'
        except:
            pass 

        try: 
            x =  self.flux
            df['flux'] = x
            df['reflected_unit'] = 'albedo'
        except:
            pass 
    

        try: 
            x = self.flux_at_top
            df['thermal_unit'] = 'erg/cm2/s/cm'
            df['thermal_3d'] = x
        except:
            pass

        try: 
            x = self.flux_layers
            df['layer']['flux_minus'] = self.flux_layers[0]
            df['layer']['flux_plus'] = self.flux_layers[1]
            df['layer']['flux_minus_mdpt'] = self.flux_layers[2]
            df['layer']['flux_plus_mdpt'] = self.flux_layers[3]
        except:
            pass

        return df

def convert_to_simple(iso_name):
    """
    Converts iso name (e.g. 13C-16O2 to CO2)
    Returns same name if not (e.g. CO2 gives CO2)
    """
    separator = '_'
    if separator not in iso_name: return iso_name
    separate = [separate_molecule_name(j) for j in iso_name.split('-')]
    mol=''
    for i in separate: 
        if len(i)>1:
            mol+=i[1]
        else: 
            mol += i[0]
    return mol

def separate_molecule_name(molecule_name):
    """used for separating molecules
    For example, CO2 becomes "C" "O2"
    """
    elements = re.findall('[A-Z][a-z]?\d*|\d+', molecule_name)
    return elements
def separate_string_number(string):
    """used for separating numbers from molecules
    For example, CO2 becomes "C" "O2" in `separate_molecule_name` 
    then this function turns it into [['C'],['O','2']]
    """
    elements = re.findall('[A-Za-z]+|\d+', string)
    return elements
"""
## not using this for now.
def hunt(xx , n , x, jlow):
    flag1 = 0
    if ((jlow <= 0) or (jlow > n-1)): # for py
        jlow = 0
        jhigh= n+1 -1 # for py

        flag1 = 1
    inc = 1
    flag2 = 0
    
    if flag1 == 0 :

        if (x >= xx[jlow]) and (xx[n-1] >= xx[0]): #for py
            while flag2 == 0:
                jhigh = jlow + inc

                if jhigh > n-1: # for py
                    jhigh = n+1 -1
                    flag2 = 1
                elif (x >= xx[jhigh]) and (xx[n-1] >= xx[0]):
                    jlow = jhigh
                    inc += inc
                    flag2 = 0
        
        else :
            jhigh = jlow
            while flag3  == 0:
                jlow = jhigh - inc
                if jlow < 0 :
                    jlow  = 0
                    flag3 = 1
                elif (x >= xx[jlow]) and (xx[n-1] >= xx[0]):
                    jhigh = jlow
                    inc += inc
                    flag3  = 0
    flag4 = 0
    while flag4 == 0:
        if jhigh - jlow == 1:
            if x == xx[n-1]:
                jlow = n-1 -1 # for py
            if x == xx[0]:
                jlow = 0
            
            return jlow, jhigh
        jmid= (jhigh+jlow)/2

        if (x >= xx[jmid]) and (xx[n-1] >= xx[0]):
            jlow = jmid
        else :
            jhigh = jmid
    
    return jlow, jhigh

"""
