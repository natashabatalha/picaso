import numpy as np 
import scipy.constants as sc   

class Rayleigh():
    """Contains functionality to compute Rayleigh cross sections. 

    This module is adapted from code 
    provided by Ryan MacDonald (rmacdonald@astro.cornell.edu). It contains 
    several hard coded values that were taken from the references shown in each method.
    
    Attributes 
    ----------
    wno : array 
        Wavenumber in inverse cm 
    wavelength : array 
        Wavelength array in microns 
    n_ref : float 
        Number density at reference conditions of refractive index measurements
        Number density (cm^-3) at 0 C and 1 atm (1.01325 bar)
        http://refractiveindex.info
    polarisabilities : dict 
        Dictionary of polarisabilities for each molecule (CGS units, cm^3). 
        Mostly taken from CRC handbook. If a molecule doesn't exist in this 
        dictionary we are not including it in as a Rayleigh scatterer. 
    king_correction_no_wave : dict 
        King correction factor for depolarisation. Usually this is a wavelength 
        dependent quantity. When we don't have the full wavelength dependent 
        information, we use a mean value. 
    rayleigh_molecules : list 
        Essentially just `polarisabilities.keys()`. All species where we are 
        specifically computing cross sections for. Everything else will 
        return an artifficially small value for the cross section ~ 1e-40
    
    Methods
    -------
    compute_sigma 
        Returns cross sections in CGS (cm2/molecule)
    CH4
        Returns polarisability and King correction factor 
    CO2
        Returns polarisability and King correction factor         
    H2
        Returns polarisability and King correction factor 
    H2O 
        Returns polarisability and King correction factor 
    He
        Returns polarisability and King correction factor 
    N2
        Returns polarisability and King correction factor 
    N2O
        Returns polarisability and King correction factor 
    NH3
        Returns polarisability and King correction factor 
    O2
        Returns polarisability and King correction factor 
    get_polarizability
    get_anisotropy
    get_Lorentz_Lorenz
    get_king_correction
    generic
    """
    def __init__(self, wavenumber):
        self.wno = wavenumber
        self.wavelength = 1e4/wavenumber
        self.n_ref = (101325.0/(sc.k * 273.15)) * 1.0e-6 
        # Polarisabilities (cgs units, cm^3) used for computing refractive index and Rayleigh scattering - Mostly from CRC handbook
        self.polarisabilities= {'H2':  0.80e-24,  'He':  0.21e-24, 'N2':   1.74e-24,  'O2':  1.58e-24, 
                    'O3':  3.21e-24,  'H2O': 1.45e-24, 'CH4':  2.59e-24,  'CO':  1.95e-24,
                    'CO2': 2.91e-24,  'NH3': 2.26e-24, 'HCN':  2.59e-24,  'PH3': 4.84e-24, 
                    'SO2': 3.72e-24,  'SO3': 4.84e-24, 'C2H2': 3.33e-24,  'H2S': 3.78e-24,
                    'NO':  1.70e-24,  'NO2': 3.02e-24, 'H3+':  0.385e-24, 'OH':  6.965e-24,   # H3+from Kawaoka & Borkman, 1971
                    'Na':  24.11e-24, 'K':   42.9e-24, 'Li':   24.33e-24, 'Rb':  47.39e-24,     
                    'Cs':  59.42e-24, 'TiO': 16.9e-24, 'VO':   14.4e-24,  'AlO': 8.22e-24,    # Without tabulated values for metal
                    'SiO': 5.53e-24,  'CaO': 23.8e-24, 'TiH':  16.9e-24,  'MgH': 10.5e-24,    # oxides and hydrides, these are taken
                    'NaH': 24.11e-24, 'AlH': 8.22e-24, 'CrH':  11.6e-24,  'FeH': 9.47e-24,    # to be metal atom polarisabilities
                    'CaH': 23.8e-24,  'BeH': 5.60e-24, 'ScH':  21.2e-24}
        #O3 comes from Brasseur & De Rudder, 1986
        #all else comes from mean value from Bogaard+, 1978            
        self.king_correction_no_wave = {"O3":1.060000,"CO":1.016995,"C2H2":1.064385,
                                        "C2H6":1.006063, "OCS":1.138786 ,"CH3Cl":1.026042,
                                        "H2S":1.001880 , "SO2":1.062638}
        self.rayleigh_molecules = list(self.polarisabilities.keys())
        
    def compute_sigma(self, species):
        """Returns cross section
        
        Computes cross section in CGS units (cm2/g)

        Parameters
        ----------
        species : str 
            Species name (case-sensitive). TiH instead of TIH, H2O instead of h2o. 

        Returns
        -------
        array 
            Rayleigh cross section in cm2/g units. 
        """
        nu  = self.wno   
        n_ref = self.n_ref
        get_ray = getattr(self, species, None)

        if get_ray is None:
            eta, F = self.generic(species)
        else: 
            eta, F = get_ray()

        sigma_ray = (((24.0 * np.pi**3 * nu**4)/(n_ref**2)) * (((eta**2 - 1.0)/(eta**2 + 2.0))**2) * F)

        return sigma_ray * 6.02214086e+23 

    def CH4(self):
        """Returns polarisability and King correction factor using various sources 
        
        Notes 
        -----
        .. [1] Sneep, M., & Ubachs, W. 2005, JQSRT, 92, 293
        """
        nu = self.wno  
        wl = self.wavelength
        #http://refractiveindex.info (Polyanskiy, 2016)
        eta = 1.0 + (46662.0e-8 + (4.02e-14 * nu**2))   # Sneep & Ubachs, 2005 (sec 5.2) / Hohm, 1993
        eta[wl < 0.325] = 1.000504679
        eta[wl > 0.633] = 1.000476653
        #scale to 0 C and 1 atm (1.01325 bar), for refractive indices defined at 15 C and 1013 hPa - Sneep & Ubachs, 2005 
        eta = ((eta-1.0) * (288.15/273.15)) + 1.0 
        #King correction 
        F = 1.000000 * nu**0    # Negligable difference from unity - Sneep & Ubachs, (sec 5.2)  
        return eta , F
    def CO2(self):
        """Returns polarisability and King correction factur using formula from Hohm, 1993
        
        Notes 
        -----
        .. [1] A. Bideau-Mehu, Y. Guern, R. Abjean and A. Johannin-Gilles. Interferometric determination of the refractive index of carbon dioxide in the ultraviolet region, Opt. Commun. 9, 432-434 (1973)
        """
        f_par = 6.00332
        w_par_sq = 0.22525399 
        f_perp = 8.54433 
        w_perp_sq = 0.66083749
        alpha = self.get_polarizability(f_par, w_par_sq,f_perp,w_perp_sq)
        #convert to cm^3 
        eta = self.get_Lorentz_Lorenz(alpha*0.148184e-24 )    
        gamma = self.get_anisotropy(f_par, w_par_sq,f_perp,w_perp_sq)
        F = self.get_king_correction(alpha, gamma)
        return     eta, F
    def H2(self):
        """Returns polarisability and King correction factur using formula from Hohm, 1993

        Notes 
        -----
        .. [1] Peck, E. R., & Huang, S. 1977, JOSA, 67, 1550
        """
        f_par = 1.62632           
        w_par_sq = 0.23940245 
        f_perp = 1.40105 
        w_perp_sq = 0.29486069     
        alpha = self.get_polarizability(f_par, w_par_sq,f_perp,w_perp_sq)
        #convert to cm^3 
        eta = self.get_Lorentz_Lorenz(alpha*0.148184e-24 )    
        gamma = self.get_anisotropy(f_par, w_par_sq,f_perp,w_perp_sq)
        F = self.get_king_correction(alpha, gamma)
        return      eta, F
    def H2O(self):
        """Returns polarisability and King correction factur using various sources 
        
        Notes 
        -----
        .. [1] Hill, R. J., & Lawrence, R. S. 1986, InfPh, 26, 371
        .. [2] Polyanskiy, M. N. 2016, Refractive index database, http://refractiveindex.info
        """
        wl = self.wavelength
        #http://refractiveindex.info (Polyanskiy, 2016)
        eta = 1.0 + ((3.011e-2/(124.40 - 1.0/(wl**2))) +   # Hill & Lawrence, 1986
                     (7.46e-3 * (0.203 - 1.0/wl))/(1.03 - 1.98e3/(wl**2) + 8.1e4/(wl**4) - 1.7e8/(wl**8)))   
        eta[wl < 0.360] = 1.000258047
        eta[wl > 17.60] = 1.000000000   # Technically formula goes to 19um, but can't have n<1.0  
        F = 1.001005 * wl**0    # Derived from values in Hinchliffe , 2007
        return eta, F
    def He(self):
        """Returns polarisability and King correction factur using various sources 
        
        Notes 
        -----
        .. [1] C. Cuthbertson and M. Cuthbertson. The refraction and dispersion of neon and helium. Proc. R. Soc. London A 135, 40-47 (1936)
        .. [2] Polyanskiy, M. N. 2016, Refractive index database, http://refractiveindex.info
        .. [3] Mansfield, C. R., & Peck, E. R. 1969, JOSA, 59, 199
        """
        #http://refractiveindex.info (Polyanskiy, 2016)
        wl = self.wavelength
        eta = 1.0 + ((0.014755297/(426.29740 - 1.0/(wl**2)))*1.0018141444038913)   # Cuthbertson & Cuthbertson, 1936 (multiplicative factor for continuity)
        eta[wl < 0.2753] = 1.00003578
        eta[wl > 0.4801] = 1.0 + (0.01470091/(423.98 - 1.0/(wl[wl > 0.4801]**2)))   # Mansfield & Peck, 1969
        eta[wl > 2.0586] = 1.00003469
        F = 1.000000 * wl**0   # Spherical atom, so King correction factor = 1
        return eta , F
    def N2(self):
        """Returns polarisability and King correction factur using various sources 
        
        Notes 
        -----
        .. [1] Sneep, M., & Ubachs, W. 2005, JQSRT, 92, 293
        .. [2] E. R. Peck and B. N. Khanna. Dispersion of nitrogen, J. Opt. Soc. Am. 56, 1059-1063 (1966)
        """
        nu = self.wno  
        wl = self.wavelength
        eta = 1.0 + ((5677.465e-8 + (318.81874e4/(14.4e9 - nu**2)))*1.0001468057477378)   # Sneep & Ubachs, 2005 (sec 4.2) / Bates, 1984  (fmultiplicative actor for continuity)
        eta[wl < 0.2540] = 1.00030493
        eta[wl > 0.46816] = 1.0 + (6498.2e-8 + (307.43305e4/(14.4e9 - nu[wl > 0.46816]**2)))   # Sneep & Ubachs, 2005 (sec 4.2) / Peck & Khanna, 1966
        eta[wl > 2.0576] = 1.00027883
        #scale to 0 C and 1 atm (1.01325 bar), for refractive indices defined at 15 C and 1013 hPa - Sneep & Ubachs, 2005 
        eta = ((eta-1.0) * (288.15/273.15)) + 1.0      
        F = 1.034 + 3.17e-12 * nu**2   # Sneep & Ubachs, 2005 (sec 4.2) / Bates, 1984
        return eta   , F

    def N2O(self):
        """Returns polarisability and King correction factur using various sources 
        
        Notes 
        -----
        .. [1] C. Cuthbertson and M. Cuthbertson. On the refraction and dispersion of the halogens, halogen acids, ozone, steam, oxides of nitrogen and ammonia, Phil. Trans. R. Soc. Lond. A 213, 1-26 (1914)
        """ 
        f_par = 5.65126
        w_par_sq = 0.17424213
        f_perp = 9.72095
        w_perp_sq = 0.72904985 
        alpha = self.get_polarizability(f_par, w_par_sq,f_perp,w_perp_sq)
        #convert to cm^3 
        eta = self.get_Lorentz_Lorenz(alpha*0.148184e-24  )    
        gamma = self.get_anisotropy(f_par, w_par_sq,f_perp,w_perp_sq)
        F = self.get_king_correction(alpha,gamma)
        return     eta   , F
    def NH3(self):  
        """Returns polarisability and King correction factur using various sources 
        
        Notes 
        -----
        .. [1] C. Cuthbertson and M. Cuthbertson. On the refraction and dispersion of the halogens, halogen acids, ozone, steam, oxides of nitrogen and ammonia, Phil. Trans. R. Soc. Lond. A 213, 1-26 (1914)
        .. [2] Hohm, U. 1993. Mol. Phys., 78: 929
        """            
        f_par = 1.28964
        w_par_sq = 0.08454599 
        f_perp = 10.84943 
        w_perp_sq = 0.76338846 
        alpha = self.get_polarizability(f_par, w_par_sq,f_perp,w_perp_sq)
        #convert to cm^3 
        eta = self.get_Lorentz_Lorenz(alpha*0.148184e-24 )      
        gamma = self.get_anisotropy(f_par, w_par_sq,f_perp,w_perp_sq)
        F = self.get_king_correction(alpha,gamma)
        return     eta   , F    
    def O2(self): 
        """Returns polarisability and King correction factur using various sources 
        
        Notes 
        -----
        .. [1] P. L. Smith, M. C. E. Huber, W. H. Parkinson. Refractivities of H2, He, O2, CO, and Kr for 168≤λ≤288 nm Phys Rev. A 13, 199-203 (1976)
        .. [2] Hohm, U. 1993. Mol. Phys., 78: 929
        """         
        f_par = 2.74876
        w_par_sq = 0.18095751 
        f_perp = 4.86007 
        w_perp_sq = 0.58545449 
        alpha = self.get_polarizability(f_par, w_par_sq,f_perp,w_perp_sq)
        #convert to cm^3 
        eta = self.get_Lorentz_Lorenz(alpha*0.148184e-24 )    
        gamma = self.get_anisotropy(f_par, w_par_sq,f_perp,w_perp_sq)
        F = self.get_king_correction(alpha,gamma)
        return     eta   , F        

    def get_polarizability(self,f_par, w_par_sq,f_perp,w_perp_sq):
        """Calculate polarisability from Hohm equation 

        Notes
        -----
        .. [1] Hohm, U. 1993. Mol. Phys., 78: 929
        """
        nu = self.wno
        # Now calculate polarisability using formula from Hohm, 1993
        # Polarisability - Hohm, 1993
        alpha = ((1.0/3.0)*((f_par/(w_par_sq - (nu/219474.6305)**2)) +                 
                       2.0*(f_perp/(w_perp_sq - (nu/219474.6305)**2))))     
        return alpha

    def get_anisotropy(self,f_par, w_par_sq,f_perp,w_perp_sq):
        """get polarisavility anisotropy from Hohm 1993"""
        nu = self.wno  
        gamma = ((f_par/(w_par_sq - (nu/219474.6305)**2)) - 
                 (f_perp/(w_perp_sq - (nu/219474.6305)**2)))    
        return gamma
         

    def get_Lorentz_Lorenz(self, alpha):
        """Lorentz-Lorenz relation""" 
        return np.sqrt((1.0 + (8.0*np.pi*self.n_ref*alpha/3.0))/(1.0 - (4.0*np.pi*self.n_ref*alpha/3.0)))  

    def get_king_correction(self, alpha,gamma):
        return 1.0 + 2.0 * (gamma/(3.0*alpha))**2 

    def generic(self,species):
        nu = self.wno   
        polarisabilities = self.polarisabilities
        king_correction_no_wave = self.king_correction_no_wave
        if species in polarisabilities.keys():
            alpha = polarisabilities[species] * nu **0 
            eta = self.get_Lorentz_Lorenz(alpha)
        else:
            eta = 0*nu
        
        if species in king_correction_no_wave.keys():
            F = king_correction_no_wave[species] * nu**0
        else: 
            F = 1*nu**0

        return eta, F

   