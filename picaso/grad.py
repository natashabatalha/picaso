import numpy as np
from numpy import log10
from numba import jit, float64
from numba.experimental import jitclass
from .numerics import locate

@jit(nopython=True, cache=True)
def moist_grad(t, p, AdiabatBundle, Atmosphere, ind):
    """
    Parameters
    ----------
    t : float
        Temperature  value
    p : float 
        Pressure value
    AdiabatBundle : namedtuple 
        includes:
        - t_table : array 
            array of Temperature values with 53 entries
        - p_table : array 
            array of Pressure value with 26 entries
        - grad : array 
            array of gradients of dimension 53*26
        - cp : array 
            array of cp of dimension 53*26
    Atmosphere : namedtuple 
        Atmosphere namedtuple which is created in picaso.climate.calculate_atm and includes info about the condensates, PT profile, and atmosphere properties
    ind: int
        index of current layer of the t and p to retrieve the right abundance at this layer
    
    Returns
    -------
    float 
        grad_x
    
    """
    # Python version of moistgrad function in convec.f in EGP
    #t_table, p_table, grad, cp = AdiabatBundle.t_table, AdiabatBundle.p_table, AdiabatBundle.grad, AdiabatBundle.cp
    #gas MMW organized into one vector (g/mol)
    MoistGradInfo = MoistGradClass()

    Rgas = 8.314e7 #erg/K/mol

    #indexes of species that are allowed to condense
    #icond = [9,10,12,18] #h2o, ch4, nh3, fe

    condensables = Atmosphere.condensables
    ncond = len(condensables) #Only 4 molecules are considered for now (H2O, CH4, NH3, Fe) 
    output_abunds = Atmosphere.condensable_abundances
    mmw = Atmosphere.condensable_weights

    #Tcrit = [647.,   191.,   406.,  4000.]
    #Tfr   = [273.,    90.,   195.,  1150.]
    #hfus  = [6.00e10, 9.46e9, 5.65e10, 1.4e11] #(erg/mol)
    Tcrit, Tfr, hfus  = np.zeros(ncond),np.zeros(ncond),np.zeros(ncond)
    
    for i,imol in enumerate(condensables): 
        info = MoistGradInfo.returns(imol)
        Tcrit[i] = info[0]
        Tfr[i] = info[1]
        hfus[i] = info[2]

    #set heat of vaporization + fusion (when applicable)
    dH = np.zeros(ncond)
    
    for i,imol in enumerate(condensables): 
        hvap = HVapClass(t, mmw[i])
        if(t < Tcrit[i]):
            dH[i] = dH[i] + hvap.returns(imol)#hvapfunc(icond[i],t, mmw)
        if(t < Tfr[i]):
            dH[i] = dH[i] + hfus[i]

    # find condensible partial pressures and H/R/T for condensibles.  
    # also find background pressure, which makes up difference between partial pressures and total pressure
    pb = p
    pc = np.zeros(ncond)
    a  = np.zeros(ncond)

    for i in range(ncond):
        #icond[i]+1 since output_abunds has t, p as first two columns so index is shifted by 1
        #ind is the index of the current layer
        pc[i] = output_abunds[i][ind]*p
        a[i]  = dH[i]/Rgas/t
        pb    -= pc[i]

    # summed heat capacity for ideal gas case. note that this cp is in erg/K/mol
    cpI = 0.0
    f = 0.0
    for i,imol in enumerate(condensables):
        cpfoo = CPClass(t,mmw[i])
        f  += output_abunds[i][ind]
        cpI += output_abunds[i][ind]*cpfoo.returns(imol)*mmw[i]

    # ideal gas adiaibatic gradient
    gradI = Rgas/cpI*f

    #non-ideal gas from Didier
    gradNI, cp_x = did_grad_cp(t,p, AdiabatBundle)
    cp_NI = Rgas/gradNI

    #weighted combination of non-ideal and ideal components
    gradb = 1.0/((1.0-f)*cp_NI/Rgas + f*cpI/Rgas)

    #moist adiabatic gradient from note by T. Robinson.
    numer = 1.0
    denom = 1.0/gradb

    for i in range(ncond):
        numer += a[i]*pc[i]/p
        denom += a[i]**2*pc[i]/p

    grad_x = numer/denom

    return grad_x, cp_x 


MoistGradTypes = [(i, float64[:]) for i in ['H2O','CH4','NH3','Fe']]
@jitclass(MoistGradTypes)
class MoistGradClass(object):
    def __init__(self):
        #arrays are Tcrit, tfr, hfus in erg/mol
        self.H2O = np.array([647.0, 273., 6.00e10])
        self.CH4 = np.array([191.0, 90.,  9.46e9])
        self.NH3 = np.array([406.0, 195., 5.65e10])
        self.Fe = np.array([4000.0, 1150., 1.4e11])
    def returns(self,mol):
        """
        This is the ONLY way to get around numba not being able to run getattr function 
        """
        if mol == 'H2O': 
            a = self.H2O 
        elif mol == 'CH4':
            a = self.CH4 
        elif mol == 'NH3':
            a = self.NH3
        elif mol == 'Fe':
            a = self.Fe
        else: 
            raise Exception("Only H2O, CH4, NH3, and Fe have been added to the moist adiabat function")
        return a

@jit(nopython=True, cache=True)
def did_grad_cp( t, p, AdiabatBundle):
    """
    Parameters
    ----------
    t : float
        Temperature  value
    p : float 
        Pressure value
    AdiabatBundle : tuple
        tuple containing the adiabat table, pressure table, gradient, and cp
    
    Returns
    -------
    float 
        grad_x,cp_x
    
    """
    # Python version of DIDGRAD function in convec.f in EGP
    # This has been benchmarked with the fortran version
    t_table, p_table, grad, cp=AdiabatBundle.t_table, AdiabatBundle.p_table, AdiabatBundle.grad, AdiabatBundle.cp
       
    temp_log= log10(t)
    pres_log= log10(p)
    
    pos_t = locate(t_table, temp_log)
    pos_p = locate(p_table, pres_log)

    ipflag=0
    if pos_p ==0: ## lowest pressure point
        factkp= 0.0
        ipflag=1
    elif pos_p ==25 : ## highest pressure point
        factkp= 1.0
        pos_p=24  ## use highest point
        ipflag=1

    itflag=0
    if pos_t ==0: ## lowest pressure point
        factkt= 0.0
        itflag=1
    elif pos_t == 52 : ## highest temp point
        factkt= 1.0
        pos_t=51 ## use highest point
        itflag=1
    
    if (pos_p > 0) and (pos_p < 26) and (ipflag == 0):
        factkp= (-p_table[pos_p]+pres_log)/(p_table[pos_p+1]-p_table[pos_p])
    
    if (pos_t > 0) and (pos_t < 53) and (itflag == 0):
        factkt= (-t_table[pos_t]+temp_log)/(t_table[pos_t+1]-t_table[pos_t])

    
    gp1 = grad[pos_t,pos_p]
    gp2 = grad[pos_t+1,pos_p]
    gp3 = grad[pos_t+1,pos_p+1]
    gp4 = grad[pos_t,pos_p+1]

    cp1 = cp[pos_t,pos_p]
    cp2 = cp[pos_t+1,pos_p]
    cp3 = cp[pos_t+1,pos_p+1]
    cp4 = cp[pos_t,pos_p+1]


    

    grad_x = (1.0-factkt)*(1.0-factkp)*gp1 + factkt*(1.0-factkp)*gp2 + factkt*factkp*gp3 + (1.0-factkt)*factkp*gp4
    cp_x= (1.0-factkt)*(1.0-factkp)*cp1 + factkt*(1.0-factkp)*cp2 + factkt*factkp*cp3 + (1.0-factkt)*factkp*cp4
    cp_x= 10**cp_x
    
    
    return grad_x,cp_x

HVapTypes = [(i, float64) for i in ['temperature','mmw']]
@jitclass(HVapTypes)
class HVapClass(object):
    def __init__(self,temperature,mmw):
        self.temperature = temperature 
        self.mmw = mmw
        return 
        
    def H2O(self):
        t = self.temperature/647.
        if( self.temperature < 647. ):
            hvap = 51.67*np.exp(0.199*t)*(1 - t)**0.410
        else:
            hvap = 0. 
        return  hvap*1.e10#convert from kJ/mol to erg/mol

    def CH4(self): 
        t = self.temperature/191
        if( self.temperature < 191 ):
            hvap = 10.11*np.exp(0.22*t)*(1 - t)**0.388
        else:
            hvap = 0. 
        return hvap*1.e10#convert from kJ/mol to erg/mol

    def  NH3(self):
        t = self.temperature - 273.
        if( self.temperature < 406. ):
            hvap = (137.91*(133. - t)**0.5 - 2.466*(133. - t))/1.e3*self.mmw
        else:
            hvap = 0.
        return hvap*1.e10 #convert from kJ/mol to erg/mol
    
    def Fe(self):
        hvap = 3.50e2 
        return hvap*1.e10 #convert from kJ/mol to erg/mol
    
    
    def returns(self,mol):
        """
        This is the ONLY way to get around numba not being able to run getattr function 
        """
        if mol == 'H2O': 
            a = self.H2O() 
        elif mol == 'CH4':
            a = self.CH4() 
        elif mol == 'NH3':
            a = self.NH3()
        elif mol == 'Fe':
            a = self.Fe()
        else: 
            raise Exception("Only H2O, CH4, NH3, and Fe have been added to the moist adiabat function")
        return a 

CPTypes = [(i, float64) for i in ['temperature','mmw']]
@jitclass(CPTypes)
class CPClass(object):
    """
    Parameters
    ----------
    gas: int 
        gas index
    temp : float
        Temperature  value
    mmw: list
        list of mmw of all gases (g/mol)

    Returns
    -------
    float 
        cp
    """
    def __init__(self,temperature,mmw):
        self.temperature = temperature 
        self.mmw = mmw
        return 
        
    def H2O(self): 
        #coefficients NIST in polynomial fit
        A = [      33.7476,      22.1440,      43.2009]
        B = [     -6.85376,      24.6949,      7.91703]
        C = [      24.6006,     -6.23914,     -1.35732]
        D = [     -10.2578,     0.576813,    0.0883558]
        E = [  0.000170650,   -0.0143783,     -12.3810]
        G = [      230.708,      210.968,      219.916]
        default_cp = 33.299
        return A, B, C, D, E, G, default_cp
    def CH4(self):
        A = [      30.1333,      33.3642,      107.517]
        B = [     -10.7805,      62.9633,    -0.420051]
        C = [      116.987,     -20.9146,     0.158105]
        D = [     -64.8550,      2.54256,   -0.0135050]
        E = [    0.0315890,     -6.26634,     -53.2270]
        G = [      221.436,      191.066,      225.284]
        default_cp = 33.258
        return A, B, C, D, E, G, default_cp
    def CO(self):
        A = [      30.7036,      34.2259,      35.3293]
        B = [     -11.7368,      1.51655,      1.14525]
        C = [      25.8658,    0.0492481,    -0.170423]
        D = [     -11.6476,   -0.0690167,    0.0111323]
        E = [  -0.00675277,     -2.61424,     -2.85798]
        G = [      237.225,      231.715,      231.882]
        default_cp = 29.104
        return A, B, C, D, E, G, default_cp
    def NH3(self):
        A = [      28.6905,      48.0925,      89.3168]
        B = [      14.9648,      16.6892,   -0.0283260]
        C = [      32.2849,    -0.765783,    -0.403009]
        D = [     -19.5766,    -0.465621,    0.0366428]
        E = [    0.0281968,     -7.37491,     -68.5295]
        G = [      221.899,      226.660,      222.041]
        default_cp = 33.284
        return A, B, C, D, E, G, default_cp
    def N2(self):
        A = [      30.7036,      34.2259,      35.3293]
        B = [     -11.7368,      1.51655,      1.14525]
        C = [      25.8658,    0.0492481,    -0.170423]
        D = [     -11.6476,   -0.0690167,    0.0111323]
        E = [  -0.00675277,     -2.61424,     -2.85798]
        G = [      237.225,      231.715,      231.882]
        default_cp = 29.104
        return A, B, C, D, E, G, default_cp
    def PH3(self):
        A = [      24.1623,      75.4246,      82.3854]
        B = [      35.7131,    -0.467915,     0.229399]
        C = [      28.4716,      2.70503,   -0.0280155]
        D = [     -24.2205,    -0.650872,   0.00135605]
        E = [    0.0530053,     -13.0455,     -24.2573]
        G = [      228.047,      262.751,      258.876]
        default_cp = 33.259
        return A, B, C, D, E, G, default_cp
    def H2S(self):
        A = [      32.3729,      45.0479,      59.8489]
        B = [     -1.43579,      7.28547,    -0.380368]
        C = [      29.0118,    -0.645552,     0.218138]
        D = [     -14.1925,    -0.109566,   -0.0148742]
        E = [   0.00759539,     -6.02580,     -21.7958]
        G = [      244.187,      242.650,      243.798]
        default_cp = 33.259
        return A, B, C, D, E, G, default_cp
    def TiO(self): #elif (igas == 16): # tio
        A = [      24.6205,      42.5795,      25.6986]
        B = [      30.8607,     -3.86291,      2.45240]
        C = [     -23.2493,      1.15148,     0.770717]
        D = [      5.39026,   -0.0315822,   -0.0946717]
        E = [    0.0642488,     -2.14344,      26.1268]
        G = [      255.386,      278.646,      282.105]
        default_cp = 33.880
        return A, B, C, D, E, G, default_cp
    def VO(self): #elif (igas == 17): # vo
        A = [      23.6324,      40.2277,      31.0958]
        B = [      28.8676,     -2.68241,    0.0444865]
        C = [     -21.5825,     0.855477,      1.06932]
        D = [      5.35779,  -0.00729363,    -0.106395]
        E = [    0.0281114,     -2.10348,      13.7865]
        G = [      251.949,      273.020,      275.689]
        default_cp = 29.106
        return A, B, C, D, E, G, default_cp
    def Fe(self): #elif (igas == 18): # fe
        A = [      22.5120,      29.3785,      31.0353]
        B = [      23.6042,     -12.7912,     -3.09778]
        C = [     -49.5765,      6.80824,     0.766662]
        D = [      26.1116,    -0.979241,   0.00158800]
        E = [   -0.0305055,    0.0621550,     -22.0154]
        G = [      202.527,      219.780,      206.035]
        default_cp = 21.387
        return A, B, C, D, E, G, default_cp
    def FeH(self): # feh
        A = [      17.0970,      43.7692,      80.0135]
        B = [      52.0678,     0.968978,     -18.2832]
        C = [     -34.3367,     0.818403,     3.55466]
        D = [      7.96189,    -0.356898,    -0.288758]
        E = [     0.455643,     -1.88073,     -41.0125]
        G = [      285.000,      285.000,      285.000]
        default_cp = 34.906
        return A, B, C, D, E, G, default_cp
    def CrH(self): #elif (igas == 20): # crh
        A = [      24.6453,      40.9948,      100.083]
        B = [      12.9392,     -3.29251,     -36.2074]
        C = [    0.0477315,      1.40327,      7.79945]
        D = [     -2.45803,   -0.0468814,    -0.458881]
        E = [    0.0859445,     -3.87926,     -68.1415]
        G = [      260.000,      280.000,      280.000]
        default_cp = 29.417
        return A, B, C, D, E, G, default_cp
    def Na(self): #elif (igas == 21): # na
        A = [      20.8154,      21.0812,      38.7681]
        B = [    -0.162936,   -0.0211313,     -9.69137]
        C = [     0.281035,    -0.188686,      1.61045]
        D = [    -0.149202,    0.0703542,   -0.0183163]
        E = [ -0.000166252,    -0.169969,     -21.5246]
        G = [      178.894,      178.829,      179.923]
        default_cp = 20.786
        return A, B, C, D, E, G, default_cp
    def K(self): #elif (igas == 22): # k
        A = [      20.8154,      20.1077,      80.8587]
        B = [    -0.162936,      1.72326,     -38.6316]
        C = [     0.281035,     -1.42054,      8.80886]
        D = [    -0.149202,     0.388577,    -0.553605]
        E = [ -0.000166252,   -0.0178336,     -57.1459]
        G = [      185.566,      184.342,      197.881]
        default_cp = 20.786
        return A, B, C, D, E, G, default_cp
    def Rb(self): #elif (igas == 23): # rb
        A = [      20.8110,      21.8305,      67.6946]
        B = [    -0.139382,    -0.120618,     -36.4056]
        C = [     0.241553,    -0.759797,      9.45407]
        D = [    -0.129505,     0.324361,    -0.654225]
        E = [ -0.000134562,    -0.519578,     -22.9711]
        G = [      195.310,      195.381,      215.367]
        default_cp = 20.786
        return A, B, C, D, E, G, default_cp
    def Cs(self):#elif (igas == 24): # cs
        A = [      20.8111,      19.3844,     -99.0597]
        B = [    -0.139259,      3.51623,      42.3576]
        C = [     0.238592,     -3.00169,     -2.76224]
        D = [    -0.126005,     0.867065,   -0.0552789]
        E = [ -0.000147773,    0.0177750,      218.172]
        G = [      200.816,      198.458,      231.228]
        default_cp = 20.786
        return A, B, C, D, E, G, default_cp
    def CO2(self):#elif (igas == 25): # co2
        A = [      17.1622,      59.7854,      65.7964]
        B = [      84.3617,    -0.472970,     -1.17414]
        C = [     -71.5668,      1.36583,     0.232788]
        D = [      24.3579,    -0.300212,  -0.00788867]
        E = [    0.0429191,     -6.20314,     -17.2749]
        G = [      212.619,      266.092,      263.469]
        default_cp = 20.786
        return A, B, C, D, E, G, default_cp
    
    #polynomial function for cp
    def polyAE(self,A, B, C, D, E,t,it):
        cp = A[it] + B[it]*t + C[it]*t**2 + D[it]*t**3 + E[it]/t**2
        return cp
        
    def returns(self,mol):
        if mol == 'H2O': 
            A, B, C, D, E, G, default_cp=self.H2O() 
        elif mol == 'CH4':
            A, B, C, D, E, G, default_cp=self.CH4()
        elif mol == 'NH3':
            A, B, C, D, E, G, default_cp=self.NH3()
        elif mol == 'Fe':
            A, B, C, D, E, G, default_cp=self.Fe()
        else: 
            raise Exception("Only H2O, CH4, NH3, and Fe have been added to the moist adiabat function")
        
        m = self.mmw
        temp = self.temperature
        t = temp/1000.
        
        if ( temp > 2500.):
            it = 2
            cp = self.polyAE(A, B, C, D, E,t,it)
        elif ( temp > 1000. and temp <= 2500.):
            it = 1
            cp = self.polyAE(A, B, C, D, E,t,it)
        elif ( temp > 100. and temp <= 1000.):
            it = 0
            cp = self.polyAE(A, B, C, D, E,t,it)
        else:
            cp = default_cp
        
        # convert from J/K/mol to erg/g/K
        cp = cp/m*1.e7
        return cp