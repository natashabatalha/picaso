import numpy as np
import pandas as pd
import os
from scipy import interpolate
from astropy.convolution import convolve, Gaussian1DKernel


from .justdoit import vj,u,get_cld_input_grid,special


## Parameterizations 
class Parameterize():
    def __init__(self, load_cld_optical = None, mieff_dir = None):
        """
        picaso_inputs_class : class picaso.justdoit.inputs 
            PICASO inputs class 
        load_cld_optical : str
            Load optical constants for a certain cloud species 
            see virga.available to see available species you can load 
        mieff_dir : str 
            If a condensate species is supplied to load_cld_optical, then you must also supplie a mieff directory
            
        """
        self.mieff_dir=mieff_dir
        if isinstance(load_cld_optical, (str,list)):
            if isinstance(load_cld_optical,str): load_cld_optical=[load_cld_optical]
            if isinstance(mieff_dir, str):
                if os.path.exists(mieff_dir):
                    self.qext, self.qscat, self.cos_qscat, self.nwave, self.radius, self.wave_in = {},{},{},{},{},{}
                    for isp in load_cld_optical:
                        self.qext[isp], self.qscat[isp], self.cos_qscat[isp], self.nwave[isp], self.radius[isp], self.wave_in[isp]=vj.get_mie(
                                    isp,directory=mieff_dir)
                else: 
                    raise Exception("path supplied through mieff_dir does not exist")
            else: 
                raise Exception("mieff_dir was not supplied as a str but needs to be if a condensate species was supplied through load_cld_optical")

        return

    def add_class(self,picaso_inputs_class):
        """Add a picaso class that loads in the pressure grid (at the very least)

        Example
        -------
        start = jdi.inputs()
        start.add_pt(P=np.logspace(-6,3,91))
        param = Parameterize(load_cld_optical=['SiO2','Al2O3'],mieff_dir='/data/virga')
        param.add_class(start)
        """
        self.picaso = picaso_inputs_class
        self.pressure_level = picaso_inputs_class.inputs['atmosphere']['profile']['pressure'].values
        self.temperature_level  = picaso_inputs_class.inputs['atmosphere']['profile'].get('temperature',pd.DataFrame()).values
        self.pressure_layer = np.sqrt(self.pressure_level [0:-1]*self.pressure_level [1:])
        self.nlevel = len(self.pressure_level )
        self.nlayer = self.nlevel -1 
        self.gravity_cgs = picaso_inputs_class.inputs['planet'].get('gravity',np.nan)

          
    def get_particle_dist(self,species,distribution,
                  lognorm_kwargs = {'sigma':np.nan, 'lograd':np.nan}, 
                  hansen_kwargs={'b':np.nan,'lograd':np.nan}):
        logradius = np.log10(self.radius[species])
        
        if 'lognorm' in distribution:
            sigma=lognorm_kwargs['sigma']
            lograd=lognorm_kwargs['lograd']
            
            if np.isnan(sigma):
                raise Exception('lognorm_kwargs have not been defined')
            
            dist = (1/(sigma * np.sqrt(2 * np.pi)) *
                       np.exp( - (logradius - lograd)**2 / (2 * sigma**2)))
        elif 'hansen' in distribution: 
            a = 10**hansen_kwargs['lograd']
            b = hansen_kwargs['b']
            dist = (10**self.radius[species])**((1-3*b)/b)*np.exp(-self.radius[species]/(a*b))
        else: 
            raise Exception("Only lognormal and hansen distributions available")        
        
        return dist 

    def cloud_virga(self,**virga_kwargs):
        """
        A function that runs picaso virga from justdoit.inputs class. modifies the toml inputs to run 
        virga direct 
        """
        virga_kwargs['directory']=self.mieff_dir
        self.picaso.inputs['atmosphere']['profile']['kz']=virga_kwargs['kzz']
        virga_kwargs.pop('kzz')
        self.picaso.virga(**virga_kwargs)
        df_cld = self.picaso.inputs['clouds']['profile']
        return df_cld 

    def cloud_flex_fsed(self, condensate, base_pressure, ndz, fsed, distribution, 
                  lognorm_kwargs = {'sigma':np.nan, 'lograd':np.nan}, 
                  hansen_kwargs={'b':np.nan,'lograd':np.nan}): 
        """
        Given a base_pressure and fsed to set the exponential drop of the cloud integrate a particle 
        radius distribution via gaussian or hansen distributions to get optical properties in picaso 
        format. 

        Parameters
        ----------
        species : str 
            Name of species. Should already have been preloaded via Parameterize options in load_cld_optical
        base_pressure : float 
            base of the cloud deck in bars 
        ndz : float 
            number density of the cloud deck cgs 
        fsed : float 
            sedimentation efficiency 
        distribution : str 
            either lognormal or hansen 
        lognorm_kwargs : dict 
            diectionary with the format: {'sigma':np.nan, 'lograd':np.nan}
            lograd median particle radius in cm 
            sigma width of the distribtuion must be >1 
        hansen_kwargs : dict 
            dictionary with the format: {'b':np.nan,'lograd':np.nan}
            lograd and b from Hansen+1971: https://web.gps.caltech.edu/~vijay/Papers/Polarisation/hansen-71b.pdf
            lograd = a = effective particle radius 
            b = varience of the particle radius 

        Returns 
        -------
        pandas.DataFrame 
            PICASO formatted cld input dataframe 
        """
        scale_h = 10 #just arbitrary as this gets fit for via fsed and ndz 
        z = np.linspace(100,0,self.nlayer)
        
        dist = self.get_particle_dist(condensate,distribution,lognorm_kwargs,hansen_kwargs)
            
        opd,w0,g0,wavenumber_grid=vj.calc_optics_user_r_dist(self.wave_in[condensate], ndz ,self.radius[condensate], u.cm,
                                                              dist, self.qext[condensate], self.qscat[condensate], self.cos_qscat[condensate])
        
        opd_h = self.pressure_layer*0+10
        opd_h[base_pressure<self.pressure_layer]=0
        opd_h[base_pressure>=self.pressure_layer]=opd_h[base_pressure>=self.pressure_layer]*np.exp(
                              -fsed*z[base_pressure>=self.pressure_layer]/scale_h)
        opd_h = opd_h/np.max(opd_h)
        
        df_cld = picaso_format(opd, w0, g0, wavenumber_grid, self.pressure_layer, 
                                          p_bottom=base_pressure,p_decay=opd_h)

        return df_cld 
    flex_cloud =  cloud_flex_fsed  
    def cloud_brewster_mie(self, condensate, distribution, decay_type,
                  lognorm_kwargs = {'sigma':np.nan, 'lograd':np.nan}, 
                  hansen_kwargs={'b':np.nan,'lograd':np.nan},
                  slab_kwargs={'ptop':np.nan,'dp':np.nan, 'reference_tau':np.nan},
                  deck_kwargs={'ptop':np.nan,'dp':np.nan}): 
        """
        Given a base_pressure and fsed to set the exponential drop of the cloud integrate a particle 
        radius distribution via gaussian or hansen distributions to get optical properties in picaso 
        format. 

        Parameters
        ----------
        species : str 
            Name of species. Should already have been preloaded via Parameterize options in load_cld_optical
        base_pressure : float 
            base of the cloud deck in bars 
        ndz : float 
            number density of the cloud deck cgs 
        fsed : float 
            sedimentation efficiency 
        distribution : str 
            either lognormal or hansen 
        lognorm_kwargs : dict 
            diectionary with the format: {'sigma':np.nan, 'lograd':np.nan}
            lograd median particle radius in cm 
            sigma width of the distribtuion must be >1 
        hansen_kwargs : dict 
            dictionary with the format: {'b':np.nan,'lograd':np.nan}
            lograd and b from Hansen+1971: https://web.gps.caltech.edu/~vijay/Papers/Polarisation/hansen-71b.pdf
            lograd = a = effective particle radius 
            b = varience of the particle radius 

        Returns 
        -------
        pandas.DataFrame 
            PICASO formatted cld input dataframe 
        """
        
        dist = self.get_particle_dist(condensate,distribution,lognorm_kwargs,hansen_kwargs)
            
        opd,w0,g0,wavenumber_grid=vj.calc_optics_user_r_dist(self.wave_in[condensate], 1 ,self.radius[condensate], u.cm,
                                                              dist, self.qext[condensate], self.qscat[condensate], self.cos_qscat[condensate])
        
        if decay_type == 'slab':
            opd_profile = self.slab_decay(**slab_kwargs)
        elif decay_type == 'deck':
            opd_profile = self.deck_decay(**deck_kwargs)
        
        df = picaso_format(opd, w0, g0, wavenumber_grid, self.pressure_layer, opd_profile=opd_profile)

        return df 
    
    def cloud_brewster_grey(self, decay_type, alpha, ssa, reference_wave=1,
                  slab_kwargs={'ptop':np.nan,'dp':np.nan, 'reference_tau':np.nan},
                  deck_kwargs={'ptop':np.nan,'dp':np.nan}): 
        """
        Creates grey cloud with either slab or deck decay and an alpha wavelength scaling 

        Parameters
        ----------
        decay_type: str
            One of 'deck' or 'slab'
        ssa: float
            Single Scattering Albedo: can have values from 0 to 1
        alpha: float
            set to 0 for grey cloud
 
        Returns 
        -------
        pandas.DataFrame 
            PICASO formatted cld input dataframe 
        """
        
        wavenumber_grid =get_cld_input_grid()
        wavelength= 1e4/wavenumber_grid

        if decay_type == 'slab':
            opd_profile = self.slab_decay(**slab_kwargs)
        elif decay_type == 'deck':
            opd_profile = self.deck_decay(**deck_kwargs)

        wave_dependent_opd =  np.concatenate([opd_profile[i]*(wavelength/reference_wave)**(-alpha) for i in range(self.nlayer)])
        wvnos =  np.concatenate([wavenumber_grid for i in range(self.nlayer)])
        pressures =  np.concatenate([[self.pressure_layer[i]]*len(wavelength) for i in range(self.nlayer)])
        w0=wave_dependent_opd*0+ssa
        g0=wave_dependent_opd*0
        df=pd.DataFrame({
                'opd':wave_dependent_opd,
                'g0':g0,
                'w0':w0,
                'wavenumber':wvnos,
                'pressure':pressures
            })

        return df 

    def cloud_hard_grey(self,g0, w0, opd,p, dp): 
        if isinstance(g0,int):g0=[g0]
        if isinstance(w0,int):w0=[w0]
        if isinstance(opd,int):opd=[opd]
        if isinstance(p,int):p=[p]
        if isinstance(dp,int):dp=[dp]

        self.picaso.clouds(g0=g0, w0=w0, opd=opd,p=p,dp=dp)
        df_cld = self.picaso.inputs['clouds']['profile']
        return df_cld

    def deck_decay(self,ptop, dp=0.005): 
        """
        Emualates brewster opacity decay for the deck model 
        
        Parameters 
        ----------
        ptop : float 
            ptop is log pressure (bar) at which tau of cloud ~1 
        dp : float 
            dtau / dP = const * exp((P-P0) / pressure_scale)
        """
        pressure_layer=self.pressure_layer
        nlayer = len(self.pressure_layer)
        opd_by_layer = np.zeros(nlayer)

        pressure_top = 10**ptop

        pressure_scale = ((pressure_top * 10.**dp) - pressure_top)  / 10.**dp
        const = 1. / (1 - np.exp(-pressure_top / pressure_scale))

        for i in range (0,nlayer):
            p_grid_top, p_grid_bot = atlev(i,pressure_layer)
            # now get dtau for each layer, where tau = 1 at pressure_top
            term1 = (p_grid_bot - pressure_top) / pressure_scale
            term2 = (p_grid_top - pressure_top) / pressure_scale
            if (term1 > 10 or term2 > 10):
                #sets large optical depths to 100 
                opd_by_layer[i] = 100.00
            else:
                opd_by_layer[i] = const * (np.exp(term1) - np.exp(term2))
        
        return opd_by_layer

    def slab_decay(self, ptop, dp=0.005, reference_tau=1): 
        """
        Modeled after brewster slabs see Eqn 13 and 14 Whiteford et al. 

        Parameters 
        ----------
        ptop : float 
            pressure top in log bars 
        dp : float 
            pressure thickness in dex bars, default - 0.005
        reference_tau : float 
            reference tau for 1 micron 

        Returns 
        -------
        optical depth per layer as a function of layer  
        """
        pressure = self.pressure_layer #levels 
        nlayer = len(pressure)

        opd_by_layer = np.zeros(nlayer)

        pressure_top = 10**ptop #p1 brewster e.g., 1e-3
        pressure_bottom = pressure_top * 10.**dp #p2 brewster e.g. 1e-3*10^2 = 1e-1

        #find index of layer for pressure top and pressure bottom 
        index_top = np.argmin(abs(np.log(pressure) - np.log(pressure_top)))
        index_bottom = np.argmin(abs(np.log(pressure) - np.log(pressure_bottom)))
        if index_top == index_bottom: 
            raise Exception('dp entered was not large enough to create a cloud given the pressure grid spacing')

        #compute tau scaling 
        tau_scaling = reference_tau / (pressure_bottom**2 - pressure_top**2)

        _ , p_grid_bot = atlev(index_top,pressure)
        opd_by_layer[index_top] = tau_scaling * (p_grid_bot**2 - pressure_top**2)  

        p_grid_top , _ = atlev(index_bottom,pressure)
        opd_by_layer[index_bottom] = tau_scaling * (pressure_bottom**2 - p_grid_top**2)        

        for i in range (index_top+1,index_bottom):
            p_grid_top,p_grid_bot = atlev(i,pressure)
            opd_by_layer[i] = tau_scaling * (p_grid_bot**2 - p_grid_top**2)

        return opd_by_layer

    def chem_free(self, **species):
        ''''
        Abundance profile for free chemistry

        Parameters
        ----------
        species: dict
            Dictionary containing the species and their abundances. Should 
            also contain background gases and their ratios. 
            Example: species=dict(H2O=dict(value=1e-4, unit='v/v'), background=dict(gases=['H2', 'He'], ratios=[0.85, 0.15]))

        Return
        ------
        Data frame with chemical abundances per level
        '''
        #free = chem_config['free']
        pressure_grid = self.pressure_level
        temp_grid = self.temperature_level
        total_sum_of_gases = 0*pressure_grid
        assert len(temp_grid)==len(pressure_grid), 'Len of t grid does not match len of p grid. likely t grid has not been set yet '
        mixingratio_df = pd.DataFrame(dict(pressure=pressure_grid, temperature=temp_grid))
        for i in species.keys(): 
            #make sure its not the background
            if i !='background':
                value = species[i].get('value',None)
                #easy case where there is just one well-mixed value 
                if value is not None: #abundance of the chemistry input per molecule
                    mixingratio_df[i] = value
                    
                else: #each molecule input manually
                    values=[]
                    for j,key in enumerate(species[i].keys()):
                        if key.startswith('value'):
                            values.append(species[i][key].get('value')) 
                    pressures = species[i]['p_switch'].get('value') 
                    pressure_unit= species[i]['p_switch'].get('unit') 
                    pressure_bar = (np.array(pressures)*u.Unit(pressure_unit)).to(u.bar).value 
                    
                    #make sure its in ascending pressure order 
                    # first = pressure_bar[0] 
                    # last = pressure_bar[-1] 

                    # #flip if the ordering has been input incorrectly
                    # if first > last : 
                    #     pressure_bar=pressure_bar[::-1]
                    #     values=values[::-1]

                    vmr = values[0] + 0*pressure_grid
                    # need to finish the ability to input free chem here 
                    for ii,ivmr in enumerate(values[1:]):
                        vmr[pressure_grid<=pressure_bar] = ivmr 
                    
                    #add to dataframe 
                    mixingratio_df[i]=vmr

                total_sum_of_gases += mixingratio_df[i].values
        #add background gas if it is requested
        if 'background' in species.keys():
            total_sum_of_background = 1-total_sum_of_gases
            if len(species['background']['gases'])==2: #2 background gasses
                gas1_name = species['background']['gases'][0]
                gas2_name = species['background']['gases'][1]
                fraction = species['background']['fraction']
                gas2_absolute_value = total_sum_of_background / (fraction + 1)
                gas1_absolute_value = fraction * gas2_absolute_value
                mixingratio_df[gas1_name] = gas1_absolute_value
                mixingratio_df[gas2_name] = gas2_absolute_value
            if len(species['background']['gases'])==1: #1 background gas
                mixingratio_df[species['background']['gases'][0]] = total_sum_of_background
        return mixingratio_df

    def chem_visscher(self,cto_absolute, log_mh): 
        self.picaso.chemeq_visscher_2121(cto_absolute, log_mh)
        return self.picaso.inputs['atmosphere']['profile']

    def pt_madhu_seager_09_noinversion(self, alpha_1, alpha_2, P_1, P_3, T_3, beta=0.5):
        """"
        Implements the temperature structure parameterization from Madhusudhan & Seager (2009)

        Parameters
        -----------

        Returns
        -------
        Temperature per layer
        """

        pressure = self.pressure_level
        P0 = pressure[0]
        nlevel = len(pressure)

        temp_by_level = np.zeros(nlevel)

        # Set T1 from T3
        T1 = T_3 - (np.log(P_3/P_1) / alpha_2)**(1/beta)
        # Set T0 from T1
        T0 = T1 - (np.log(P_1/P0) / alpha_1)**(1/beta)

        

        # Set pressure ranges
        layer_1=(pressure<P_1)
        layer_2=(pressure>=P_1)*(pressure<P_3)
        layer_3=(pressure>=P_3)

        # Define temperature at each pressure range
        temp_by_level[layer_1] = T0 + ((1/alpha_1)*np.log(pressure[layer_1]/P0))**(1/beta)
        temp_by_level[layer_2] = T1 + ((1/alpha_2)*np.log(pressure[layer_2]/P_1))**(1/beta)
        temp_by_level[layer_3] = T_3

        temp_by_level = convolve(temp_by_level,Gaussian1DKernel(5),boundary='extend')

        return pd.DataFrame(dict(pressure=pressure, temperature=temp_by_level))
    
    def pt_madhu_seager_09_inversion(self, alpha_1, alpha_2, P_1, P_2, P_3, T_3, beta=0.5):
        """"
        Implements the temperature structure parameterization from Madhusudhan & Seager (2009)
          allowing for inversions

        Parameters
        -----------

        Returns
        -------
        Temperature per layer
        """

        pressure = self.pressure_level
        nlevel = len(pressure)

        temp_by_level = np.zeros(nlevel)

        P0 = pressure[0]

        # Set pressure ranges
        layer_1=(pressure<P_1)
        layer_2=(pressure<P_3)*(pressure>=P_1)
        layer_3=(pressure>=P_3)

        # Define temperatures at boundaries to ensure continuity
        T2 = T_3 - (np.log(P_3/P_2) / alpha_2)**(1/beta)
        T1 = T2 + (np.log(P_1/P_2) / alpha_2)**(1/beta)
        T0 = T1 - (np.log(P_1/P0) / alpha_1)**(1/beta)

        # Define temperature at each pressure range
        temp_by_level[layer_1] = T0 + (np.log(pressure[layer_1]/P0)/alpha_1)**(1/beta)
        temp_by_level[layer_2] = T2 + (np.log(pressure[layer_2]/P_2)/alpha_2)**(1/beta)
        temp_by_level[layer_3] = T_3

        temp_by_level = convolve(temp_by_level,Gaussian1DKernel(5),boundary='extend')

        return pd.DataFrame(dict(pressure=pressure, temperature=temp_by_level))
    
    def pt_knots(self,  P_knots, T_knots, interpolation='brewster',scipy_interpolate_kwargs={}):
        """"
        Knot-based temperature profile. Implements different types of interpolation.

        Parameters
        -----------

        Returns
        -------
        Temperature per layer
        """
        if isinstance(P_knots, dict):
            P_knots = list(P_knots.values())
        
        if isinstance(T_knots, dict):
            T_knots = [T_knots[k]["value"] for k in sorted(T_knots.keys())]

        pressure = self.pressure_level
        nlevel = len(pressure)

        temp_by_level = np.zeros(nlevel)

        # Interpolation requires pressures to be sorted from lowest to highest
        order = np.argsort(P_knots)
        P_knots=np.array(P_knots)[order]
        T_knots=np.array(T_knots)[order]
        
        # Perform the interpolation
        if interpolation=='brewster':
            interpolator = interpolate.splrep(np.log10(P_knots), T_knots, s=0)
            temp_by_level=np.abs(interpolate.splev(np.log10(pressure),interpolator,der=0))
        elif interpolation=='linear':
            interpolator = interpolate.interp1d(np.log10(P_knots), T_knots, kind='linear', bounds_error=False, fill_value='extrapolate')
            temp_by_level = interpolator(np.log10(pressure))
        elif interpolation=='quadratic_spline':
            assert len(P_knots)>=3, 'Quadratic splines require at least 3 knots'
            interpolator = interpolate.interp1d(np.log10(P_knots), T_knots, kind='quadratic', bounds_error=False, fill_value='extrapolate')
            temp_by_level = interpolator(np.log10(pressure))
        elif interpolation=='cubic_spline':
            assert len(P_knots)>=4, 'Cubic splines require at least 4 knots'
            interpolator = interpolate.interp1d(np.log10(P_knots), T_knots, kind='cubic', bounds_error=False, fill_value='extrapolate')
            temp_by_level = interpolator(np.log10(pressure))
        elif getattr(interpolate, interpolation, np.nan)!=np.nan:
            interpolator = getattr(interpolate, interpolation)
            interpolator(np.log10(P_knots), T_knots, *scipy_interpolate_kwargs)
        else:
            raise Exception(f'Unknown interpolation method \'{interpolation}\'')

        #check that T is strictly positive everywhere

        return pd.DataFrame(dict(pressure=pressure, temperature=temp_by_level))

    def pt_guillot(self, Teq, T_int, logg1, logKir, alpha):
        """
        Creates temperature pressure profile given parameterization in Guillot 2010 TP profile
        called in fx()
        Parameters
        ----------
        Teq : float 
            equilibrium temperature 
        T_int : float 
            Internal temperature, if low (100) currently set to 100 for everything  
        kv1 : float 
            see parameterization Guillot 2010 (10.**(logg1+logKir))
        kv2 : float
            see parameterization Guillot 2010 (10.**(logg1+logKir))
        kth : float
            see parameterization Guillot 2010 (10.**logKir)
        alpha : float , optional
            set to 0.5
            
        Returns
        -------
        T : numpy.array 
            Temperature grid 
        P : numpy.array
            Pressure grid
                
        """
        kv1, kv2 =10.**(logg1+logKir),10.**(logg1+logKir)
        kth=10.**logKir

        Teff = T_int
        f = 1.0  # solar re-radiation factor
        A = 0.0  # planetary albedo
        g0 = self.gravity_cgs/100.0 #cm/s2 to m/s2
        assert not np.isnan(g0),'Graivty was not supplied but is being requested for guillot p-t profile parameterization'

        # Compute equilibrium temperature and set up gamma's
        T0 = Teq
        gamma1 = kv1/kth #Eqn. 25
        gamma2 = kv2/kth

        # Initialize arrays
        logtau =np.arange(-10,20,.1)
        tau =10**logtau

        #computing temperature
        T4ir = 0.75*(Teff**(4.))*(tau+(2.0/3.0))
        f1 = 2.0/3.0 + 2.0/(3.0*gamma1)*(1.+(gamma1*tau/2.0-1.0)*np.exp(-gamma1*tau))+2.0*gamma1/3.0*(1.0-tau**2.0/2.0)*special.expn(2.0,gamma1*tau)
        f2 = 2.0/3.0 + 2.0/(3.0*gamma2)*(1.+(gamma2*tau/2.0-1.0)*np.exp(-gamma2*tau))+2.0*gamma2/3.0*(1.0-tau**2.0/2.0)*special.expn(2.0,gamma2*tau)
        T4v1=f*0.75*T0**4.0*(1.0-alpha)*f1
        T4v2=f*0.75*T0**4.0*alpha*f2
        T=(T4ir+T4v1+T4v2)**(0.25)
        P=tau*g0/(kth*0.1)/1.E5

        logP = np.log10(self.pressure_level)#np.linspace(p_top,p_bottom,nlevel)

        T = np.interp(logP,np.log10(P),T)

        # Return TP profile
        return pd.DataFrame({'temperature': T, 'pressure':self.pressure_level})
    
    def pt_isothermal(self, T):
        return pd.DataFrame({'temperature': T, 'pressure':self.pressure_level})

def atlev(l0,pressure_layer):
    nlayers = pressure_layer.size
    if (l0 <= nlayers-2):
        pressure_top = np.exp(((1.5)*np.log(pressure_layer[l0])) - ((0.5)*np.log(pressure_layer[l0+1])))
        pressure_bottom = np.exp((0.5)*(np.log(pressure_layer[l0] * pressure_layer[l0+1])))
    else:
        pressure_top = np.exp((0.5 * np.log(pressure_layer[l0-1] * pressure_layer[l0])))
        pressure_bottom = pressure_layer[l0]**2 / pressure_top

    return pressure_top, pressure_bottom

def picaso_format(opd, w0, g0, wavenumber_grid, pressure_grid ,
                       p_bottom=None,p_top=None,p_decay=None,opd_profile=None):
    """
    Sets up a PICASO-readable dataframe that inserts a wavelength dependent aerosol layer at the user's 
    given pressure bounds, i.e., a wavelength-dependent slab of clouds or haze.
    
    Parameters
    ----------
    p_bottom : float 
        the cloud/haze base pressure
        the upper bound of pressure (i.e., lower altitude bound) to set the aerosol layer. (Bars)
    opd : ndarray
        wavelength-dependent optical depth of the aerosol
    w0 : ndarray
        wavelength-dependent single scattering albedo of the aerosol
    g0 : ndarray
        asymmetry parameter = Q_scat wtd avg of <cos theta>
    wavenumber_grid : ndarray
        wavenumber grid in (cm^-1) 
    pressure_grid : ndarray
        bars, user-defined pressure grid for the model atmosphere
    p_top : float
         bars, the cloud/haze-top pressure
         This cuts off the upper cloud region as a step function. 
         You must specify either p_top or p_decay. 
    p_decay : ndarray
        noramlized to 1, unitless
        array the same size as pressure_grid which specifies a 
        height dependent optical depth. The usual format of p_decay is 
        a fsed like exponential decay ~np.exp(-fsed*z/H)


    Returns
    -------
    Dataframe of aerosol layer with pressure (in levels - non-physical units!), wavenumber, opd, w0, and g0 to be read by PICASO
    """
    if isinstance(p_bottom, type(None)): 
        p_bottom = np.max(pressure_grid)+10#arbitrarily big to make sure float comparison includes clouds
        
    if (isinstance(p_top, type(None)) & isinstance(p_decay, type(None)) & isinstance(opd_profile, type(None))): 
        raise Exception("Must specify cloud top pressure via p_top, or the vertical pressure decay via p_decay, or an opd profile via opd_profile")
    
    if (isinstance(p_top, type(None))): 
        p_top = 1e-10#arbitarily small pressure to make sure float comparison doest break


    df = pd.DataFrame(index=[ i for i in range(pressure_grid.shape[0]*opd.shape[0])], columns=['pressure','wavenumber','opd','w0','g0'])
    i = 0 
    LVL = []
    WV,OPD,WW0,GG0 =[],[],[],[]
    
    # this loops the opd, w0, and g0 between p and dp bounds and put zeroes for them everywhere else
    for j in range(pressure_grid.shape[0]):
           for w in range(opd.shape[0]):
                #stick in pressure bounds for the aerosol layer:
                if p_top <= pressure_grid[j] <= p_bottom:
                    LVL+=[pressure_grid[j]]
                    WV+=[wavenumber_grid[w]]
                    if (isinstance(p_decay,type(None)) & isinstance(opd_profile,type(None))):
                        OPD+=[opd[w]]
                    elif not (isinstance(p_decay,type(None))): 
                        OPD+=[p_decay[j]/np.max(p_decay)*opd[w]]
                    elif not (isinstance(opd_profile,type(None))): 
                        OPD+=[opd_profile[j]*(opd[w]/np.max(opd))]
                    WW0+=[w0[w]]
                    GG0+=[g0[w]]
                else:
                    LVL+=[pressure_grid[j]]
                    WV+=[wavenumber_grid[w]]
                    OPD+=[opd[w]*0]
                    WW0+=[w0[w]*0]
                    GG0+=[g0[w]*0]       
                    
    df.iloc[:,0 ] = LVL
    df.iloc[:,1 ] = WV
    df.iloc[:,2 ] = OPD
    df.iloc[:,3 ] = WW0
    df.iloc[:,4 ] = GG0
    return df


def cloud_averaging(dfs):
    """
    This is a function that takes a list of pre-formated picaso cloud dataframes and averages them. 
    This is used for the case where a user might be computing several types of cloud decks and wants 
    the final averaged cloud profile. 

    Parameters
    ----------
    dfs : list, pd.DataFrame
        This is a list of `picaso_format` dataframes for the cloud profiles 
        Expected that these all have the same dimensionality
    
    Returns
    -------
    df 
        single picaso format cloud dictionary
    """
    opd = 0 * dfs[0]['opd'].values
    g0 = 0 * dfs[0]['opd'].values
    w0 = 0 * dfs[0]['opd'].values
    for idf in dfs:  
        opdnext =  idf['opd'].values
        assert len(opd)==len(opdnext), 'cloud dataframes were made on different grids'
        opd += opdnext
    
    for idf in dfs: 
        g0 += idf['opd'].values*idf['g0'].values
        w0 += idf['opd'].values*idf['w0'].values
    
    opd[opd==0]=1e-33 #filler to avoid divide by zeros

    g0 = g0/opd
    w0 =w0/opd 

    #return single df
    df_cld = dfs[0]
    df_cld['opd']=opd
    df_cld['g0']=g0
    df_cld['w0']=w0
    return df_cld