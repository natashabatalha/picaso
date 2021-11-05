from numba import jit, vectorize, njit, objmode
from numpy import exp, zeros, where, sqrt, cumsum , pi, outer, sinh, cosh, min, dot, array,log, stack, ones, floor, array_equal
import numpy as np
#import pentapy as pp
import time
import pickle as pk
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve_banded
from numpy.linalg import solve
from numpy.linalg import inv as npinv
from scipy.linalg import inv as spinv

#@jit(nopython=True, cache=True)
def get_flux_toon(nlevel, wno, nwno, tau, dtau, w0, cosbar, surf_reflect, ubar0, F0PI):
    """
    Warning
    -------
    Discontinued function. See `get_flux_geom_1d` and `get_flux_geom_3d`.

    Parameters
    ----------
    nlevel : int 
        Number of levels in the model 
    wno : array of float 
        Wave number grid in cm -1 
    nwno : int 
        Number of wave points
    dtau : ndarray of float
        This is the opacity contained within each individual layer (defined at midpoints of "levels")
        Dimensions=# layer by # wave
    w0 : ndarray of float 
        This is the single scattering albedo, from scattering, clouds, raman, etc 
        Dimensions=# layer by # wave
    cosbar : ndarray of float 
        This is the asymmetry factor 
        Dimensions=# layer by # wave
    surf_reflect : float 
        Surface reflectivity 
    ubar0 : float 
        Cosine of the incident angle 
    F0PI : array 
        Downward incident solar radiation


    Returns
    -------
    flux_up and flux_down through each layer as a function of wavelength 

    Todo
    ----
    - Replace detla-function adjustment with better approximation (e.g. Cuzzi)
    - F0PI Solar flux shouldn't always be 1.. Follow up to make sure that this isn't a bad 
          hardwiring to solar 
    
    Examples
    --------
    
    >>> flux_plus, flux_minus  = fluxes.get_flux_toon(atm.c.nlevel, wno,nwno,
                                                    tau_dedd,dtau_dedd, w0_dedd, cosb_dedd, surf_reflect, ubar0, F0PI)


    """
    nlayer = nlevel - 1 

    #initially used for the downard flux calc
    #ubar0 = 0.5 
    #F0PI = np.zeros(wno) + 1.0 

    #First thing to do is to use the delta function to icorporate the forward 
    #peak contribution of scattering by adjusting optical properties such that 
    #the fraction of scattered energy in the forward direction is removed from 
    #the scattering parameters 

    #Joseph, J.H., W. J. Wiscombe, and J. A. Weinman, 
    #The Delta-Eddington approximation for radiative flux transfer, J. Atmos. Sci. 33, 2452-2459, 1976.

    #also see these lecture notes are pretty good
    #http://irina.eas.gatech.edu/EAS8803_SPRING2012/Lec20.pdf
    
    #w0=wbar_WDEL*(1.-cosb_CDEL**2)/(1.0-wbar_WDEL*cosb_CDEL**2)
    #cosbar=cosb_CDEL/(1.+cosb_CDEL)
    #dtau=dtau_DTDEL*(1.-wbar_WDEL*cosb_CDEL**2) 
    
    #sum up taus starting at the top, going to depth
    #tau = zeros((nlevel, nwno))
    #tau[1:,:]=numba_cumsum(dtau)

    #now define terms of Toon et al 1989 quadrature Table 1 
    #https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/JD094iD13p16287
    #see table of terms 
    sq3 = sqrt(3.)
    g1  = (sq3*0.5)*(2. - w0*(1.+cosbar))   #table 1
    g2  = (sq3*w0*0.5)*(1.-cosbar)         #table 1
    g3  = 0.5*(1.-sq3*cosbar*ubar0)       #table 1
    lamda = sqrt(g1**2 - g2**2)                  #eqn 21
    gama  = (g1-lamda)/g2                             #eqn 22
    
    # now calculate c_plus and c_minus (equation 23 and 24)
    g4 = 1.0 - g3
    denominator = lamda**2 - 1.0/ubar0**2.0

    #assert 0 in denomiator , ('Zero in C+, C- Toon+1989. Add catch.')

    #everything but the exponential 
    a_minus = F0PI*w0* (g4*(g1 + 1.0/ubar0) +g2*g3 ) / denominator
    a_plus  = F0PI*w0*(g3*(g1-1.0/ubar0) +g2*g4) / denominator

    #add in exponntial to get full eqn
    #_up is the terms evaluated at lower optical depths (higher altitudes)
    #_down is terms evaluated at higher optical depths (lower altitudes)
    x = exp(-tau[:-1,:]/ubar0)
    c_minus_up = a_minus*x #CMM1
    c_plus_up  = a_plus*x #CPM1
    x = exp(-tau[1:,:]/ubar0)
    c_minus_down = a_minus*x #CM
    c_plus_down  = a_plus*x #CP

    #calculate exponential terms needed for the tridiagonal rotated layered method
    exptrm = lamda*dtau
    #save from overflow 
    exptrm = slice_gt (exptrm, 35.0)

    exptrm_positive = exp(exptrm) #EP
    exptrm_minus = 1.0/exptrm_positive#exp(-exptrm) #EM


    #boundary conditions 
    b_top = 0.0                                       
    b_surface = 0. + surf_reflect*ubar0*F0PI*exp(-tau[-1, :]/ubar0)

    #Now we need the terms for the tridiagonal rotated layered method
    if tridiagonal==0:
        A, B, C, D = setup_tri_diag(nlayer,nwno,  c_plus_up, c_minus_up, 
                            c_plus_down, c_minus_down, b_top, b_surface, surf_reflect,
                             gama, dtau, 
                            exptrm_positive,  exptrm_minus) 
    else:
        A_, B_, C_, D_, E_, F_ = setup_pent_diag(nlayer,nwno,  c_plus_up, c_minus_up, 
                            c_plus_down, c_minus_down, b_top, b_surface, surf_reflect,
                             gama, dtau, 
                            exptrm_positive,  exptrm_minus, g1,g2,exptrm,lamda) 

    positive = zeros((nlayer, nwno))
    negative = zeros((nlayer, nwno))
    #========================= Start loop over wavelength =========================
    L = 2*nlayer
    for w in range(nwno):
        #coefficient of posive and negative exponential terms 
        if tridiagonal==0:
            X = tri_diag_solve(L, A[:,w], B[:,w], C[:,w], D[:,w])
            #unmix the coefficients
            positive[:,w] = X[::2] + X[1::2] 
            negative[:,w] = X[::2] - X[1::2]
        else:
            X = pent_diag_solve(L, A_[:,w], B_[:,w], C_[:,w], D_[:,w], E_[:,w], F_[:,w])
            #positive[:,w] = X[::2] 
            #negative[:,w] = X[1::2]
            positive[:,w] = exptrm_minus[:,w] * (X[::2] + X[1::2])
            negative[:,w] = X[::2] - X[1::2]
    #========================= End loop over wavelength =========================

    #might have to add this in to avoid numerical problems later. 
    #if len(np.where(negative[:,w]/X[::2] < 1e-30)) >0 , print(negative[:,w],X[::2],negative[:,w]/X[::2])

    #evaluate the fluxes through the layers 
    #use the top optical depth expression to evaluate fp and fm 
    #at the top of each layer 
    flux_plus  = zeros((nlevel, nwno))
    flux_minus = zeros((nlevel, nwno))
    flux_plus[:-1,:]  = positive + gama*negative + c_plus_up #everything but the last row (botton of atmosphere)
    flux_minus[:-1,:] = positive*gama + negative + c_minus_up #everything but the last row (botton of atmosphere

    #use expression for bottom flux to get the flux_plus and flux_minus at last
    #bottom layer
    flux_plus[-1,:]  = positive[-1,:]*exptrm_positive[-1,:] + gama[-1,:]*negative[-1,:]*exptrm_minus[-1,:] + c_plus_down[-1,:]
    flux_minus[-1,:] = positive[-1,:]*exptrm_positive[-1,:]*gama[-1,:] + negative[-1,:]*exptrm_minus[-1,:] + c_minus_down[-1,:]

    #we have solved for fluxes directly and no further integration is needed 
    #ubar is absorbed into the definition of g1-g4
    #ubar=0.5: hemispheric constant 
    #ubar=sqrt(3): gauss quadrature 
    #other cases in meador & weaver JAS, 37, 630-643, 1980

    #now add direct flux term to the downwelling radiation, Liou 1982
    flux_minus = flux_minus + ubar0*F0PI*exp(-1.0*tau/ubar0)

    #now calculate the fluxes at the midpoints of the layers 
    #exptrm_positive_mdpt = exp(0.5*exptrm) #EP_mdpt
    #exptrm_minus_mdpt = exp(-0.5*exptrm) #EM_mdpt

    #tau_mdpt = tau[:-1] + 0.5*dtau #start from bottom up to define midpoints 
    #c_plus_mdpt = a_plus*exp(-tau_mdpt/ubar0)
    #c_minus_mdpt = a_minus*exp(-tau_mdpt/ubar0)
    
    #flux_plus_mdpt = positive*exptrm_positive_mdpt + gama*negative*exptrm_minus_mdpt + c_plus_mdpt
    #flux_minus_mdpt = positive*exptrm_positive_mdpt*gama + negative*exptrm_minus_mdpt + c_minus_mdpt

    #add direct flux to downwelling term 
    #flux_minus_mdpt = flux_minus_mdpt + ubar0*F0PI*exp(-1.0*tau_mdpt/ubar0)

    return flux_plus, flux_minus

#@jit(nopython=True, cache=True)
def slice_eq(array, lim, value):
    """Funciton to replace values with upper or lower limit
    """
    for i in range(array.shape[0]):
        new = array[i,:] 
        new[where(new==lim)] = value
        array[i,:] = new     
    return array

#@jit(nopython=True, cache=True)
def slice_lt(array, lim):
    """Funciton to replace values with upper or lower limit
    """
    for i in range(array.shape[0]):
        new = array[i,:] 
        new[where(new<lim)] = lim
        array[i,:] = new     
    return array

#@jit(nopython=True, cache=True)
def slice_gt(array, lim):
    """Funciton to replace values with upper or lower limit
    """
    for i in range(array.shape[0]):
        new = array[i,:] 
        new[where(new>lim)] = lim
        new[where(new<-lim)] = -lim
        array[i,:] = new     
    return array

#@jit(nopython=True, cache=True)
def numba_cumsum(mat):
    """Function to compute cumsum along axis=0 to bypass numba not allowing kwargs in 
    cumsum 
    """
    new_mat = zeros(mat.shape)
    for i in range(mat.shape[1]):
        new_mat[:,i] = cumsum(mat[:,i])
    return new_mat

#@jit(nopython=True, cache=True)
def setup_tri_diag(nlayer,nwno ,c_plus_up, c_minus_up, 
    c_plus_down, c_minus_down, b_top, b_surface, surf_reflect,
    gama, dtau, exptrm_positive,  exptrm_minus):
    """
    Before we can solve the tridiagonal matrix (See Toon+1989) section
    "SOLUTION OF THE TwO-STREAM EQUATIONS FOR MULTIPLE LAYERS", we 
    need to set up the coefficients. 

    Parameters
    ----------
    nlayer : int 
        number of layers in the model 
    nwno : int 
        number of wavelength points
    c_plus_up : array 
        c-plus evaluated at the top of the atmosphere 
    c_minus_up : array 
        c_minus evaluated at the top of the atmosphere 
    c_plus_down : array 
        c_plus evaluated at the bottom of the atmosphere 
    c_minus_down : array 
        c_minus evaluated at the bottom of the atmosphere 
    b_top : array 
        The diffuse radiation into the model at the top of the atmosphere
    b_surface : array
        The diffuse radiation into the model at the bottom. Includes emission, reflection 
        of the unattenuated portion of the direct beam  
    surf_reflect : array 
        Surface reflectivity 
    g1 : array 
        table 1 toon et al 1989
    g2 : array 
        table 1 toon et al 1989
    g3 : array 
        table 1 toon et al 1989
    lamba : array 
        Eqn 21 toon et al 1989 
    gama : array 
        Eqn 22 toon et al 1989
    dtau : array 
        Opacity per layer
    exptrm_positive : array 
        Eqn 44, expoential terms needed for tridiagonal rotated layered, clipped at 35 
    exptrm_minus : array 
        Eqn 44, expoential terms needed for tridiagonal rotated layered, clipped at 35 


    Returns
    -------
    array 
        coefficient of the positive exponential term 
    
    """
    L = 2 * nlayer

    #EQN 44 

    e1 = exptrm_positive + gama*exptrm_minus
    e2 = exptrm_positive - gama*exptrm_minus
    e3 = gama*exptrm_positive + exptrm_minus
    e4 = gama*exptrm_positive - exptrm_minus


    #now build terms 
    A = zeros((L,nwno)) 
    B = zeros((L,nwno )) 
    C = zeros((L,nwno )) 
    D = zeros((L,nwno )) 

    A[0,:] = 0.0
    B[0,:] = gama[0,:] + 1.0
    C[0,:] = gama[0,:] - 1.0
    D[0,:] = b_top - c_minus_up[0,:]

    #even terms, not including the last !CMM1 = UP
    A[1::2,:][:-1] = (e1[:-1,:]+e3[:-1,:]) * (gama[1:,:]-1.0) #always good
    B[1::2,:][:-1] = (e2[:-1,:]+e4[:-1,:]) * (gama[1:,:]-1.0)
    C[1::2,:][:-1] = 2.0 * (1.0-gama[1:,:]**2)          #always good 
    D[1::2,:][:-1] =((gama[1:,:]-1.0)*(c_plus_up[1:,:] - c_plus_down[:-1,:]) + 
                            (1.0-gama[1:,:])*(c_minus_down[:-1,:] - c_minus_up[1:,:]))
    #import pickle as pk
    #pk.dump({'GAMA_1':(gama[1:,:]-1.0), 'CPM1':c_plus_up[1:,:] , 'CP':c_plus_down[:-1,:], '1_GAMA':(1.0-gama[1:,:]), 
    #   'CM':c_minus_down[:-1,:],'CMM1':c_minus_up[1:,:],'Deven':D[1::2,:][:-1]}, open('../testing_notebooks/GFLUX_even_D_terms.pk','wb'))
    
    #odd terms, not including the first 
    A[::2,:][1:] = 2.0*(1.0-gama[:-1,:]**2)
    B[::2,:][1:] = (e1[:-1,:]-e3[:-1,:]) * (gama[1:,:]+1.0)
    C[::2,:][1:] = (e1[:-1,:]+e3[:-1,:]) * (gama[1:,:]-1.0)
    D[::2,:][1:] = (e3[:-1,:]*(c_plus_up[1:,:] - c_plus_down[:-1,:]) + 
                            e1[:-1,:]*(c_minus_down[:-1,:] - c_minus_up[1:,:]))

    #last term [L-1]
    A[-1,:] = e1[-1,:]-surf_reflect*e3[-1,:]
    B[-1,:] = e2[-1,:]-surf_reflect*e4[-1,:]
    C[-1,:] = 0.0
    D[-1,:] = b_surface-c_plus_down[-1,:] + surf_reflect*c_minus_down[-1,:]

    return A, B, C, D

#@jit(nopython=True, cache=True)
def setup_pent_diag(nlayer,nwno ,c_plus_up, c_minus_up, 
    c_plus_down, c_minus_down, b_top, b_surface, surf_reflect,
    gama, dtau, exptrm_positive,  exptrm_minus, g1, g2, exptrm, lamda):
    """
    Parameters
    ----------
    nlayer : int 
        number of layers in the model 
    nwno : int 
        number of wavelength points
    c_plus_up : array 
        c-plus evaluated at the top of the atmosphere 
    c_minus_up : array 
        c_minus evaluated at the top of the atmosphere 
    c_plus_down : array 
        c_plus evaluated at the bottom of the atmosphere 
    c_minus_down : array 
        c_minus evaluated at the bottom of the atmosphere 
    b_top : array 
        The diffuse radiation into the model at the top of the atmosphere
    b_surface : array
        The diffuse radiation into the model at the bottom. Includes emission, reflection 
        of the unattenuated portion of the direct beam  
    surf_reflect : array 
        Surface reflectivity 
    g1 : array 
        table 1 toon et al 1989
    g2 : array 
        table 1 toon et al 1989
    g3 : array 
        table 1 toon et al 1989
    lamba : array 
        Eqn 21 toon et al 1989 
    gama : array 
        Eqn 22 toon et al 1989
    dtau : array 
        Opacity per layer
    exptrm_positive : array 
        Eqn 44, expoential terms needed for tridiagonal rotated layered, clipped at 35 
    exptrm_minus : array 
        Eqn 44, expoential terms needed for tridiagonal rotated layered, clipped at 35 


    Returns
    -------
    array 
        coefficient of the positive exponential term 
    
    """
    L = 2 * nlayer

    #EQN 44 

    e1 = 1. + gama*exptrm_minus
    e2 = 1. - gama*exptrm_minus
    e3 = gama + exptrm_minus
    e4 = gama - exptrm_minus

    #f1 = g2 * sinh(exptrm) / lamda
    #f2 = cosh(exptrm) + g1 * sinh(exptrm) / lamda
    #f3 = cosh(exptrm) - g1 * sinh(exptrm) / lamda

    #now build terms 
    A = zeros((L,nwno)) 
    B = zeros((L,nwno )) 
    C = zeros((L,nwno )) 
    D = zeros((L,nwno )) 
    E = zeros((L,nwno )) 
    F = zeros((L,nwno )) 

    A[0,:] = 0.0
    B[0,:] = 0.0
    C[0,:] = e1[0,:] #1.#
    D[0,:] = -e2[0,:] #0.#
    E[0,:] = 0.0
    F[0,:] = b_top - c_minus_up[0,:]

    #even terms, not including the last !CMM1 = UP
    # A values are zero so don't do anything
    B[1::2,:][:-1] = e1[:-1,:] #f1[:-1,:]#
    C[1::2,:][:-1] = e2[:-1,:] #f2[:-1,:]#
    D[1::2,:][:-1] = -e3[1:,:] #D[1::2,:][:-1]
    E[1::2,:][:-1] = e4[1:,:] #E[1::2,:][:-1] - 1.#
    F[1::2,:][:-1] = c_plus_up[1:,:] - c_plus_down[:-1,:]
    
    #odd terms, not including the first 
    A[::2,:][1:] = e3[:-1,:] #f3[:-1,:]#
    B[::2,:][1:] = e4[:-1,:] #f1[:-1,:]# 
    C[::2,:][1:] = -e1[1:,:] #C[::2,:][1:] - 1.#
    D[::2,:][1:] = e2[1:,:] #D[1::2,:][:-1]
    # E values are zero so don't do anything
    F[::2,:][1:] = c_minus_up[1:,:] - c_minus_down[:-1,:]

    #last term [L-1]
    A[-1,:] = 0.
    B[-1,:] = e1[-1,:]-surf_reflect*e3[-1,:] #f1[-1,:] - surf_reflect * f3[-1,:]#
    C[-1,:] = e2[-1,:]-surf_reflect*e4[-1,:] #f2[-1,:] - surf_reflect * f1[-1,:]#
    D[-1,:] = 0. 
    E[-1,:] = 0.
    F[-1,:] = b_surface - c_plus_down[-1,:] + surf_reflect*c_minus_down[-1,:]

    return A, B, C, D, E, F


#@jit(nopython=True, cache=True)
def tri_diag_solve(l, a, b, c, d):
    """
    Tridiagonal Matrix Algorithm solver, a b c d can be NumPy array type or Python list type.
    refer to this wiki_ and to this explanation_. 
    
    .. _wiki: http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    .. _explanation: http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    
    A, B, C and D refer to: 

    .. math:: A(I)*X(I-1) + B(I)*X(I) + C(I)*X(I+1) = D(I)

    This solver returns X. 

    Parameters
    ----------
    A : array or list 
    B : array or list 
    C : array or list 
    C : array or list 

    Returns
    -------
    array 
        Solution, x 
    """
    AS, DS, CS, DS,XK = zeros(l), zeros(l), zeros(l), zeros(l), zeros(l) # copy arrays

    AS[-1] = a[-1]/b[-1]
    DS[-1] = d[-1]/b[-1]

    for i in range(l-2, -1, -1):
        x = 1.0 / (b[i] - c[i] * AS[i+1])
        AS[i] = a[i] * x
        DS[i] = (d[i]-c[i] * DS[i+1]) * x
    XK[0] = DS[0]
    for i in range(1,l):
        XK[i] = DS[i] - AS[i] * XK[i-1]
    return XK

#@jit(nopython=True, cache=True)
def pent_diag_solve(l, A, B, C, D, E, F):
    """
    Pentadiagonal Matrix solver

    Parameters
    ----------
    A : array or list 
    B : array or list 
    C : array or list 
    D : array or list 
    E : array or list 
    F : array or list 

    Returns
    -------
    array 
        Solution, x 
    """

    Mrow = zeros((5, len(A)))
    Mrow[0,:] = E
    Mrow[1,:] = D
    Mrow[2,:] = C
    Mrow[3,:] = B
    Mrow[4,:] = A

    X = pp.solve(Mrow, F, is_flat=True)

    return X

#@jit(nopython=True, cache=True)
def get_reflected_3d(nlevel, wno,nwno, numg,numt, dtau_3d, tau_3d, w0_3d, cosb_3d,gcos2_3d, ftau_cld_3d,ftau_ray_3d,
    dtau_og_3d, tau_og_3d, w0_og_3d, cosb_og_3d, 
    surf_reflect,ubar0, ubar1,cos_theta, F0PI,single_phase, multi_phase,
    frac_a, frac_b, frac_c, constant_back, constant_forward,tridiagonal):
    """
    Computes toon fluxes given tau and everything is 3 dimensional. This is the exact same function 
    as `get_flux_geom_1d` but is kept separately so we don't have to do unecessary indexing for 
    retrievals. 

    Parameters
    ----------
    nlevel : int 
        Number of levels in the model 
    wno : array of float 
        Wave number grid in cm -1 
    nwno : int 
        Number of wave points
    numg : int 
        Number of Gauss angles 
    numt : int 
        Number of Chebyshev angles 
    dtau_3d : ndarray of float
        This is the opacity contained within each individual layer (defined at midpoints of "levels")
        WITHOUT D-Eddington Correction
        Dimensions=# layer by # wave
    tau_3d : ndarray of float
        This is the cumulative summed opacity 
        WITHOUT D-Eddington Correction
        Dimensions=# level by # wave        
    w0_3d : ndarray of float 
        This is the single scattering albedo, from scattering, clouds, raman, etc 
        WITHOUT D-Eddington Correction
        Dimensions=# layer by # wave
    cosb_3d : ndarray of float 
        This is the asymmetry factor 
        WITHOUT D-Eddington Correction
        Dimensions=# layer by # wave
    gcos2_3d : ndarray of float 
        Parameter that allows us to directly include Rayleigh scattering 
        = 0.5*tau_rayleigh/(tau_rayleigh + tau_cloud)
    ftau_cld_3d : ndarray of float 
        Fraction of cloud extinction to total 
        = tau_cloud/(tau_rayleigh + tau_cloud)
    ftau_ray_3d : ndarray of float 
        Fraction of rayleigh extinction to total 
        = tau_rayleigh/(tau_rayleigh + tau_cloud)
    dtau_og_3d : ndarray of float 
        This is the opacity contained within each individual layer (defined at midpoints of "levels")
        WITHOUT the delta eddington correction, if it was specified by user
        Dimensions=# layer by # wave
    tau_og_3d : ndarray of float
        This is the cumulative summed opacity 
        WITHOUT the delta eddington correction, if it was specified by user
        Dimensions=# level by # wave    
    w0_og_3d : ndarray of float 
        Same as w0 but WITHOUT the delta eddington correction, if it was specified by user  
    cosb_og_3d : ndarray of float 
        Same as cosbar buth WITHOUT the delta eddington correction, if it was specified by user
    surf_reflect : float 
        Surface reflectivity 
    ubar0 : ndarray of float 
        matrix of cosine of the incident angle from geometric.json
    ubar1 : ndarray of float 
        matrix of cosine of the observer angles
    cos_theta : float 
        Cosine of the phase angle of the planet 
    F0PI : array 
        Downward incident solar radiation
    single_phase : str 
        Single scattering phase function, default is the two-term henyey-greenstein phase function
    multi_phase : str 
        Multiple scattering phase function, defulat is N=2 Legendre polynomial
    frac_a : float 
        (Optional), If using the TTHG phase function. Must specify the functional form for fraction 
        of forward to back scattering (A + B * gcosb^C)
    frac_b : float 
        (Optional), If using the TTHG phase function. Must specify the functional form for fraction 
        of forward to back scattering (A + B * gcosb^C)
    frac_c : float 
        (Optional), If using the TTHG phase function. Must specify the functional form for fraction 
        of forward to back scattering (A + B * gcosb^C), Default is : 1 - gcosb^2
    constant_back : float 
        (Optional), If using the TTHG phase function. Must specify the assymetry of back scatterer. 
        Remember, the output of A & M code does not separate back and forward scattering.
    constant_forward : float 
        (Optional), If using the TTHG phase function. Must specify the assymetry of forward scatterer. 
        Remember, the output of A & M code does not separate back and forward scattering.
    tridiagonal : int 
        0 for tridiagonal, 1 for pentadiagonal 

    Returns
    -------
    intensity at the top of the atmosphere for all the different ubar1 and ubar2 

    To Do
    -----
    - F0PI Solar flux shouldn't always be 1.. Follow up to make sure that this isn't a bad 
          hardwiring to solar, despite "relative albedo"
    - take delta eddington option out of fluxes and move it all over to optics

    """
    #what we want : intensity at the top as a function of all the different angles

    xint_at_top = zeros((numg, numt, nwno))

    nlayer = nlevel - 1 

    #now define terms of Toon et al 1989 quadrature Table 1 
    #https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/JD094iD13p16287
    #see table of terms 

    #terms not dependent on incident angle
    sq3 = sqrt(3.)

    #================ START CRAZE LOOP OVER ANGLE #================
    for ng in range(numg):
        for nt in range(numt):

            #get needed chunk for 3d inputs
            #should consider thinking of a better method for when doing 1d only
            cosb = cosb_3d[:,:,ng,nt]
            dtau = dtau_3d[:,:,ng,nt]
            tau = tau_3d[:,:,ng,nt]
            w0 = w0_3d[:,:,ng,nt]
            gcos2 = gcos2_3d[:,:,ng,nt]
            ftau_cld = ftau_cld_3d[:,:,ng,nt]
            ftau_ray = ftau_ray_3d[:,:,ng,nt]

            #uncorrected original values (in case user specified D-Eddington)
            #If they did not, this is the same thing as what is defined above 
            #These are used because HG single scattering phase function does get 
            #the forward and back scattering pretty accurately so delta-eddington
            #is only applied to the multiple scattering terms
            cosb_og = cosb_og_3d[:,:,ng,nt]
            dtau_og = dtau_og_3d[:,:,ng,nt]
            tau_og = tau_og_3d[:,:,ng,nt]
            w0_og = w0_og_3d[:,:,ng,nt]         

            g1  = (sq3*0.5)*(2. - w0*(1.+cosb)) #table 1
            g2  = (sq3*w0*0.5)*(1.-cosb)           #table 1
            lamda = sqrt(g1**2 - g2**2)           #eqn 21
            gama  = (g1-lamda)/g2                   #eqn 22
            g3  = 0.5*(1.-sq3*cosb*ubar0[ng, nt])   #table 1 #ubar is now 100x 10 matrix.. 
    
            # now calculate c_plus and c_minus (equation 23 and 24)
            g4 = 1.0 - g3
            denominator = lamda**2 - 1.0/ubar0[ng, nt]**2.0

            #everything but the exponential 
            a_minus = F0PI*w0* (g4*(g1 + 1.0/ubar0[ng, nt]) +g2*g3 ) / denominator
            a_plus  = F0PI*w0*(g3*(g1-1.0/ubar0[ng, nt]) +g2*g4) / denominator

            #add in exponential to get full eqn
            #_up is the terms evaluated at lower optical depths (higher altitudes)
            #_down is terms evaluated at higher optical depths (lower altitudes)
            x = exp(-tau[:-1,:]/ubar0[ng, nt])
            c_minus_up = a_minus*x #CMM1
            c_plus_up  = a_plus*x #CPM1
            x = exp(-tau[1:,:]/ubar0[ng, nt])
            c_minus_down = a_minus*x #CM
            c_plus_down  = a_plus*x #CP

            #calculate exponential terms needed for the tridiagonal rotated layered method
            exptrm = lamda*dtau
            #save from overflow 
            exptrm = slice_gt (exptrm, 40.0) 

            exptrm_positive = exp(exptrm) #EP
            exptrm_minus = 1.0/exptrm_positive#exp(-exptrm) #EM


            #boundary conditions 
            b_top = 0.0                                       
            b_surface = 0. + surf_reflect*ubar0[ng, nt]*F0PI*exp(-tau[-1, :]/ubar0[ng, nt])

            #Now we need the terms for the tridiagonal rotated layered method
            if tridiagonal==0:
                A, B, C, D = setup_tri_diag(nlayer,nwno,  c_plus_up, c_minus_up, 
                                    c_plus_down, c_minus_down, b_top, b_surface, surf_reflect,
                                     gama, dtau, 
                                    exptrm_positive,  exptrm_minus) 
            #else:
            #   A_, B_, C_, D_, E_, F_ = setup_pent_diag(nlayer,nwno,  c_plus_up, c_minus_up, 
            #                       c_plus_down, c_minus_down, b_top, b_surface, surf_reflect,
            #                        gama, dtau, 
            #                       exptrm_positive,  exptrm_minus, g1,g2,exptrm,lamda) 

            positive = zeros((nlayer, nwno))
            negative = zeros((nlayer, nwno))
            #========================= Start loop over wavelength =========================
            L = 2*nlayer
            for w in range(nwno):
                #coefficient of posive and negative exponential terms 
                if tridiagonal==0:
                    X = tri_diag_solve(L, A[:,w], B[:,w], C[:,w], D[:,w])
                    #unmix the coefficients
                    positive[:,w] = X[::2] + X[1::2] 
                    negative[:,w] = X[::2] - X[1::2]
                #else:
                #   X = pent_diag_solve(L, A_[:,w], B_[:,w], C_[:,w], D_[:,w], E_[:,w], F_[:,w])
                #   #unmix the coefficients
                #   positive[:,w] = exptrm_minus[:,w] * (X[::2] + X[1::2])
                #   negative[:,w] = X[::2] - X[1::2]

            #========================= End loop over wavelength =========================

            #use expression for bottom flux to get the flux_plus and flux_minus at last
            #bottom layer
            flux_zero  = positive[-1,:]*exptrm_positive[-1,:] + gama[-1,:]*negative[-1,:]*exptrm_minus[-1,:] + c_plus_down[-1,:]
            
            xint = zeros((nlevel,nwno))
            xint[-1,:] = flux_zero/pi

            ################################ BEGIN OPTIONS FOR MULTIPLE SCATTERING####################

            #Legendre polynomials for the Phase function due to multiple scatterers 
            if multi_phase ==0:#'N=2':
                #ubar2 is defined to deal with the integration over the second moment of the 
                #intensity. It is FIT TO PURE RAYLEIGH LIMIT, ~(1/sqrt(3))^(1/2)
                #this is a decent assumption because our second order legendre polynomial 
                #is forced to be equal to the rayleigh phase function
                ubar2 = 0.767  # 
                multi_plus = (1.0+1.5*cosb*ubar1[ng,nt] #!was 3
                                + gcos2*(3.0*ubar2*ubar2*ubar1[ng,nt]*ubar1[ng,nt] - 1.0)/2.0)
                multi_minus = (1.-1.5*cosb*ubar1[ng,nt] 
                                + gcos2*(3.0*ubar2*ubar2*ubar1[ng,nt]*ubar1[ng,nt] - 1.0)/2.0)
            elif multi_phase ==1:#'N=1':
                multi_plus = 1.0+1.5*cosb*ubar1[ng,nt]  
                multi_minus = 1.-1.5*cosb*ubar1[ng,nt]
            ################################ END OPTIONS FOR MULTIPLE SCATTERING####################

            G=w0*positive*(multi_plus+gama*multi_minus)
            H=w0*negative*(gama*multi_plus+multi_minus)
            A=w0*(multi_plus*c_plus_up+multi_minus*c_minus_up)

            G=G*0.5/pi
            H=H*0.5/pi
            A=A*0.5/pi

            ################################ BEGIN OPTIONS FOR DIRECT SCATTERING####################
            #define f (fraction of forward to back scattering), 
            #g_forward (forward asymmetry), g_back (backward asym)
            #needed for everything except the OTHG
            if single_phase!=1: 
                g_forward = constant_forward*cosb_og
                g_back = constant_back*cosb_og#-
                f = frac_a + frac_b*g_back**frac_c

            # NOTE ABOUT HG function: we are translating to the frame of the downward propagating beam
            # Therefore our HG phase function becomes:
            # p_single=(1-cosb_og**2)/sqrt((1+cosb_og**2+2*cosb_og*cos_theta)**3) 
            # as opposed to the traditional:
            # p_single=(1-cosb_og**2)/sqrt((1+cosb_og**2-2*cosb_og*cos_theta)**3) (NOTICE NEGATIVE)

            if single_phase==0:#'cahoy':
                #Phase function for single scattering albedo frum Solar beam
                #uses the Two term Henyey-Greenstein function with the additiona rayleigh component 
                      #first term of TTHG: forward scattering
                p_single=(f * (1-g_forward**2)
                                /sqrt((1+cosb_og**2+2*cosb_og*cos_theta)**3) 
                                #second term of TTHG: backward scattering
                                +(1-f)*(1-g_back**2)
                                /sqrt((1+(-cosb_og/2.)**2+2*(-cosb_og/2.)*cos_theta)**3)+
                                #rayleigh phase function
                                (gcos2))
            elif single_phase==1:#'OTHG':
                p_single=(1-cosb_og**2)/sqrt((1+cosb_og**2+2*cosb_og*cos_theta)**3) 
            elif single_phase==2:#'TTHG':
                #Phase function for single scattering albedo frum Solar beam
                #uses the Two term Henyey-Greenstein function with the additiona rayleigh component 
                      #first term of TTHG: forward scattering
                p_single=(f * (1-g_forward**2)
                                /sqrt((1+g_forward**2+2*g_forward*cos_theta)**3) 
                                #second term of TTHG: backward scattering
                                +(1-f)*(1-g_back**2)
                                /sqrt((1+g_back**2+2*g_back*cos_theta)**3))
            elif single_phase==3:#'TTHG_ray':
                #Phase function for single scattering albedo frum Solar beam
                #uses the Two term Henyey-Greenstein function with the additiona rayleigh component 
                            #first term of TTHG: forward scattering
                p_single=(ftau_cld*(f * (1-g_forward**2)
                                                /sqrt((1+g_forward**2+2*g_forward*cos_theta)**3) 
                                                #second term of TTHG: backward scattering
                                                +(1-f)*(1-g_back**2)
                                                /sqrt((1+g_back**2+2*g_back*cos_theta)**3))+            
                                #rayleigh phase function
                                ftau_ray*(0.75*(1+cos_theta**2.0)))

            ################################ END OPTIONS FOR DIRECT SCATTERING####################

            for i in range(nlayer-1,-1,-1):
                #direct beam
                #note when delta-eddington=off, then tau_single=tau, cosb_single=cosb, w0_single=w0, etc
                xint[i,:] =( xint[i+1,:]*exp(-dtau[i,:]/ubar1[ng,nt])
                        #single scattering albedo from sun beam (from ubar0 to ubar1)
                        +(w0_og[i,:]*F0PI/(4.*pi))*
                        (p_single[i,:])*exp(-tau_og[i,:]/ubar0[ng,nt])*
                        (1. - exp(-dtau_og[i,:]*(ubar0[ng,nt]+ubar1[ng,nt])/(ubar0[ng,nt]*ubar1[ng,nt])))*
                        (ubar0[ng,nt]/(ubar0[ng,nt]+ubar1[ng,nt]))
                        #three multiple scattering terms 
                        +A[i,:]* (1. - exp(-dtau[i,:] *(ubar0[ng,nt]+1*ubar1[ng,nt])/(ubar0[ng,nt]*ubar1[ng,nt])))*
                        (ubar0[ng,nt]/(ubar0[ng,nt]+1*ubar1[ng,nt]))
                        +G[i,:]*(exp(exptrm[i,:]*1-dtau[i,:]/ubar1[ng,nt]) - 1.0)/(lamda[i,:]*1*ubar1[ng,nt] - 1.0)
                        +H[i,:]*(1. - exp(-exptrm[i,:]*1-dtau[i,:]/ubar1[ng,nt]))/(lamda[i,:]*1*ubar1[ng,nt] + 1.0))
                        #thermal

            xint_at_top[ng,nt,:] = xint[0,:]    
    return xint_at_top

#@jit(nopython=True, cache=True)
def get_reflected_1d(nlevel, wno,nwno, numg,numt, dtau, tau, w0, cosb,gcos2, ftau_cld, ftau_ray,
    dtau_og, tau_og, w0_og, cosb_og, 
    surf_reflect,ubar0, ubar1,cos_theta, F0PI,single_phase, multi_phase,
    frac_a, frac_b, frac_c, constant_back, constant_forward,
    approximation=0, tridiagonal=0, b_top=0):
    """
    Computes toon fluxes given tau and everything is 1 dimensional. This is the exact same function 
    as `get_flux_geom_3d` but is kept separately so we don't have to do unecessary indexing for fast
    retrievals. 

    Parameters
    ----------
    nlevel : int 
        Number of levels in the model 
    wno : array of float 
        Wave number grid in cm -1 
    nwno : int 
        Number of wave points
    numg : int 
        Number of Gauss angles 
    numt : int 
        Number of Chebyshev angles 
    DTAU : ndarray of float
        This is the opacity contained within each individual layer (defined at midpoints of "levels")
        WITHOUT D-Eddington Correction
        Dimensions=# layer by # wave
    TAU : ndarray of float
        This is the cumulative summed opacity 
        WITHOUT D-Eddington Correction
        Dimensions=# level by # wave        
    W0 : ndarray of float 
        This is the single scattering albedo, from scattering, clouds, raman, etc 
        WITHOUT D-Eddington Correction
        Dimensions=# layer by # wave
    COSB : ndarray of float 
        This is the asymmetry factor 
        WITHOUT D-Eddington Correction
        Dimensions=# layer by # wave
    GCOS2 : ndarray of float 
        Parameter that allows us to directly include Rayleigh scattering 
        = 0.5*tau_rayleigh/(tau_rayleigh + tau_cloud)
    ftau_cld : ndarray of float 
        Fraction of cloud extinction to total 
        = tau_cloud/(tau_rayleigh + tau_cloud)
    ftau_ray : ndarray of float 
        Fraction of rayleigh extinction to total 
        = tau_rayleigh/(tau_rayleigh + tau_cloud)
    dtau_og : ndarray of float 
        This is the opacity contained within each individual layer (defined at midpoints of "levels")
        WITHOUT the delta eddington correction, if it was specified by user
        Dimensions=# layer by # wave
    tau_og : ndarray of float
        This is the cumulative summed opacity 
        WITHOUT the delta eddington correction, if it was specified by user
        Dimensions=# level by # wave    
    w0_og : ndarray of float 
        Same as w0 but WITHOUT the delta eddington correction, if it was specified by user  
    cosb_og : ndarray of float 
        Same as cosbar buth WITHOUT the delta eddington correction, if it was specified by user
    surf_reflect : float 
        Surface reflectivity 
    ubar0 : ndarray of float 
        matrix of cosine of the incident angle from geometric.json
    ubar1 : ndarray of float 
        matrix of cosine of the observer angles
    cos_theta : float 
        Cosine of the phase angle of the planet 
    F0PI : array 
        Downward incident solar radiation
    single_phase : str 
        Single scattering phase function, default is the two-term henyey-greenstein phase function
    multi_phase : str 
        Multiple scattering phase function, defulat is N=2 Legendre polynomial 
    frac_a : float 
        (Optional), If using the TTHG phase function. Must specify the functional form for fraction 
        of forward to back scattering (A + B * gcosb^C)
    frac_b : float 
        (Optional), If using the TTHG phase function. Must specify the functional form for fraction 
        of forward to back scattering (A + B * gcosb^C)
    frac_c : float 
        (Optional), If using the TTHG phase function. Must specify the functional form for fraction 
        of forward to back scattering (A + B * gcosb^C), Default is : 1 - gcosb^2
    constant_back : float 
        (Optional), If using the TTHG phase function. Must specify the assymetry of back scatterer. 
        Remember, the output of A & M code does not separate back and forward scattering.
    constant_forward : float 
        (Optional), If using the TTHG phase function. Must specify the assymetry of forward scatterer. 
        Remember, the output of A & M code does not separate back and forward scattering.
    tridiagonal : int 
        0 for tridiagonal, 1 for pentadiagonal 

    Returns
    -------
    intensity at the top of the atmosphere for all the different ubar1 and ubar2 

    To Do
    -----
    - F0PI Solar flux shouldn't always be 1.. Follow up to make sure that this isn't a bad 
          hardwiring to solar, despite "relative albedo"
    """
    #what we want : intensity at the top as a function of all the different angles

    xint_at_top = zeros((numg, numt, nwno))
    intensity = zeros((numg, numt, nlevel, nwno))
    direct_flux = zeros((numg, numt, nlevel, nwno))
    single_scat = zeros((numg, numt, nlevel, nwno))
    multi_scat = zeros((numg, numt, nlevel, nwno))

    nlayer = nlevel - 1 
    flux_out = zeros((numg, numt, 2*nlayer, nwno))

    #now define terms of Toon et al 1989 quadrature Table 1 
    #https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/JD094iD13p16287
    #see table of terms 

    #terms not dependent on incident angle
    sq3 = sqrt(3.)
    if approximation == 1:#eddington
        g1  = (7-w0*(4+3*cosb))/4 #(sq3*0.5)*(2. - w0*(1.+cosb)) #table 1 # 
        g2  = -(1-w0*(4-3*cosb))/4 #(sq3*w0*0.5)*(1.-cosb)        #table 1 # 
    elif approximation == 0:#quadrature
        g1  = (sq3*0.5)*(2. - w0*(1.+cosb)) #table 1 # 
        g2  = (sq3*w0*0.5)*(1.-cosb)        #table 1 # 
    lamda = sqrt(g1**2 - g2**2)         #eqn 21
    gama  = (g1-lamda)/g2               #eqn 22

    #================ START CRAZE LOOP OVER ANGLE #================
    for ng in range(numg):
        for nt in range(numt):
            u1 = ubar1[ng,nt]
            u0 = ubar0[ng,nt]
            if approximation == 1 : #eddington
                g3  = (2-3*cosb*u0)/4#0.5*(1.-sq3*cosb*ubar0[ng, nt]) #  #table 1 #ubar has dimensions [gauss angles by tchebyshev angles ]
            elif approximation == 0 :#quadrature
                g3  = 0.5*(1.-sq3*cosb*u0) #  #table 1 #ubar has dimensions [gauss angles by tchebyshev angles ]
    
            # now calculate c_plus and c_minus (equation 23 and 24 toon)
            g4 = 1.0 - g3
            denominator = lamda**2 - 1.0/u0**2.0

            #everything but the exponential 
            a_minus = F0PI*w0* (g4*(g1 + 1.0/u0) +g2*g3 ) / denominator
            a_plus  = F0PI*w0*(g3*(g1-1.0/u0) +g2*g4) / denominator

            #add in exponential to get full eqn
            #_up is the terms evaluated at lower optical depths (higher altitudes)
            #_down is terms evaluated at higher optical depths (lower altitudes)
            x = exp(-tau[:-1,:]/u0)
            c_minus_up = a_minus*x #CMM1
            c_plus_up  = a_plus*x #CPM1
            x = exp(-tau[1:,:]/u0)
            c_minus_down = a_minus*x #CM
            c_plus_down  = a_plus*x #CP

            #calculate exponential terms needed for the tridiagonal rotated layered method
            exptrm = lamda*dtau
            #save from overflow 
            exptrm = slice_gt (exptrm, 35.0) 

            exptrm_positive = exp(exptrm) #EP
            exptrm_minus = 1.0/exptrm_positive#exp(-exptrm) #EM


            #boundary conditions 
            #b_top = 0.0                                       

            b_surface = 0. + surf_reflect*u0*F0PI*exp(-tau[-1, :]/u0)

            #Now we need the terms for the tridiagonal rotated layered method
            if tridiagonal==0:
                A, B, C, D = setup_tri_diag(nlayer,nwno,  c_plus_up, c_minus_up, 
                                    c_plus_down, c_minus_down, b_top, b_surface, surf_reflect,
                                     gama, dtau, 
                                    exptrm_positive,  exptrm_minus) 

            #else:
            #   A_, B_, C_, D_, E_, F_ = setup_pent_diag(nlayer,nwno,  c_plus_up, c_minus_up, 
            #                       c_plus_down, c_minus_down, b_top, b_surface, surf_reflect,
            #                        gama, dtau, 
            #                       exptrm_positive,  exptrm_minus, g1,g2,exptrm,lamda) 

            positive = zeros((nlayer, nwno))
            negative = zeros((nlayer, nwno))
            #========================= Start loop over wavelength =========================
            L = 2*nlayer
            for w in range(nwno):
                #coefficient of posive and negative exponential terms 
                if tridiagonal==0:
                    X = tri_diag_solve(L, A[:,w], B[:,w], C[:,w], D[:,w])
                    #unmix the coefficients
                    positive[:,w] = X[::2] + X[1::2] 
                    negative[:,w] = X[::2] - X[1::2]

                #else: 
                #   X = pent_diag_solve(L, A_[:,w], B_[:,w], C_[:,w], D_[:,w], E_[:,w], F_[:,w])
                    #unmix the coefficients
                #   positive[:,w] = exptrm_minus[:,w] * (X[::2] + X[1::2])
                #   negative[:,w] = X[::2] - X[1::2]

            #========================= End loop over wavelength =========================

            #use expression for bottom flux to get the flux_plus and flux_minus at last
            #bottom layer
            flux_zero  = positive[-1,:]*exptrm_positive[-1,:] + gama[-1,:]*negative[-1,:]*exptrm_minus[-1,:] + c_plus_down[-1,:]
            flux_minus  = gama*positive*exptrm_positive + negative*exptrm_minus + c_minus_down
            flux_plus  = positive*exptrm_positive + gama*negative*exptrm_minus + c_plus_down
            flux = zeros((2*nlayer, nwno))
            flux[::2, :] = flux_minus
            flux[1::2, :] = flux_plus
            flux_out[ng,nt,:,:] = flux

            xint = zeros((nlevel,nwno))
            term1 = zeros((nlevel,nwno))
            term2 = zeros((nlevel,nwno))
            term3 = zeros((nlevel,nwno))
            xint[-1,:] = flux_zero/pi

            ################################ BEGIN OPTIONS FOR MULTIPLE SCATTERING####################

            #Legendre polynomials for the Phase function due to multiple scatterers 
            if multi_phase ==0:#'N=2':
                #ubar2 is defined to deal with the integration over the second moment of the 
                #intensity. It is FIT TO PURE RAYLEIGH LIMIT, ~(1/sqrt(3))^(1/2)
                #this is a decent assumption because our second order legendre polynomial 
                #is forced to be equal to the rayleigh phase function
                ubar2 = 0.767  # 
                multi_plus = (1.0+1.5*cosb*u1 #!was 3
                                + gcos2*(3.0*ubar2*ubar2*u1*u1 - 1.0)/2.0)
                multi_minus = (1.-1.5*cosb*u1 
                                + gcos2*(3.0*ubar2*ubar2*u1*u1 - 1.0)/2.0)
            elif multi_phase ==1:#'N=1':
                multi_plus = 1.0+1.5*cosb*u1  
                multi_minus = 1.-1.5*cosb*u1
            ################################ END OPTIONS FOR MULTIPLE SCATTERING####################

            G=positive*(multi_plus+gama*multi_minus)    *w0
            H=negative*(gama*multi_plus+multi_minus)    *w0
            A=(multi_plus*c_plus_up+multi_minus*c_minus_up) *w0

            G=G*0.5/pi
            H=H*0.5/pi
            A=A*0.5/pi

            ################################ BEGIN OPTIONS FOR DIRECT SCATTERING####################
            #define f (fraction of forward to back scattering), 
            #g_forward (forward asymmetry), g_back (backward asym)
            #needed for everything except the OTHG
            if single_phase!=1: 
                g_forward = constant_forward*cosb_og
                g_back = constant_back*cosb_og#-
                f = frac_a + frac_b*g_back**frac_c

            # NOTE ABOUT HG function: we are translating to the frame of the downward propagating beam
            # Therefore our HG phase function becomes:
            # p_single=(1-cosb_og**2)/sqrt((1+cosb_og**2+2*cosb_og*cos_theta)**3) 
            # as opposed to the traditional:
            # p_single=(1-cosb_og**2)/sqrt((1+cosb_og**2-2*cosb_og*cos_theta)**3) (NOTICE NEGATIVE)

            if single_phase==0:#'cahoy':
                #Phase function for single scattering albedo frum Solar beam
                #uses the Two term Henyey-Greenstein function with the additiona rayleigh component 
                      #first term of TTHG: forward scattering
                p_single=(f * (1-g_forward**2)
                                /sqrt((1+cosb_og**2+2*cosb_og*cos_theta)**3) 
                                #second term of TTHG: backward scattering
                                +(1-f)*(1-g_back**2)
                                /sqrt((1+(-cosb_og/2.)**2+2*(-cosb_og/2.)*cos_theta)**3)+
                                #rayleigh phase function
                                (gcos2))
            elif single_phase==1:#'OTHG':
                p_single=(1-cosb_og**2)/sqrt((1+cosb_og**2+2*cosb_og*cos_theta)**3) 
                #def P(mu): # Legendre polynomials
                #    return [1, mu, (3*mu**2 - 1)/2, (5*mu**3 - 3*mu)/2,
                #            (35*mu**4 - 30*mu**2 + 3)/8, 
                #            (63*mu**5 - 70*mu**3 + 15*mu)/8, 
                #            (231*mu**6 - 315*mu**4 + 105*mu**2 - 5)/16 ]
                #maxterm = 2
                #for l in range(maxterm):
                #    #from scipy.special import legendre
                #    ff = cosb_og**maxterm
                #    w_temp = (2*l+1) * (cosb_og**l -  ff) / (1-ff)
                #    #Pn = legendre(l)
                #    #p_single = p_single + w_temp * Pn(-u0)*Pn(u1)
                #    p_single = p_single + w_temp * P(-u0)[l]*P(u1)[l]
            elif single_phase==2:#'TTHG':
                #Phase function for single scattering albedo frum Solar beam
                #uses the Two term Henyey-Greenstein function with the additiona rayleigh component 
                      #first term of TTHG: forward scattering
                p_single=(f * (1-g_forward**2)
                                /sqrt((1+g_forward**2+2*g_forward*cos_theta)**3) 
                                #second term of TTHG: backward scattering
                                +(1-f)*(1-g_back**2)
                                /sqrt((1+g_back**2+2*g_back*cos_theta)**3))
            elif single_phase==3:#'TTHG_ray':
                #Phase function for single scattering albedo frum Solar beam
                #uses the Two term Henyey-Greenstein function with the additiona rayleigh component 
                            #first term of TTHG: forward scattering
                p_single=(ftau_cld*(f * (1-g_forward**2)
                                                /sqrt((1+g_forward**2+2*g_forward*cos_theta)**3) 
                                                #second term of TTHG: backward scattering
                                                +(1-f)*(1-g_back**2)
                                                /sqrt((1+g_back**2+2*g_back*cos_theta)**3))+            
                                #rayleigh phase function
                                ftau_ray*(0.75*(1+cos_theta**2.0)))

            ################################ END OPTIONS FOR DIRECT SCATTERING####################

            for i in range(nlayer-1,-1,-1):
                #direct beam
                #single scattering albedo from sun beam (from ubar0 to ubar1)
                single_scat[ng,nt,i,:] = single_scat[ng,nt,i+1,:]*exp(-dtau[i,:]/u1) + (
                        w0_og[i,:]*F0PI/(4.*pi) * p_single[i,:]
                        * exp(-tau_og[i,:]/u0)
                        *(1. - exp(-dtau_og[i,:]*(u0+u1)/(u0*u1)))
                        *(u0/(u0+u1)))
                multi_scat[ng,nt,i,:] = multi_scat[ng,nt,i+1,:]*exp(-dtau[i,:]/u1) + (
                        +A[i,:]*(1. - exp(-dtau[i,:] *(u0+1*u1)/(u0*u1)))*
                        (u0/(u0+1*u1))
                        +G[i,:]*(exp(exptrm[i,:]*1-dtau[i,:]/u1) - 1.0)/(lamda[i,:]*1*u1 - 1.0)
                        +H[i,:]*(1. - exp(-exptrm[i,:]*1-dtau[i,:]/u1))/(lamda[i,:]*1*u1 + 1.0))

                direct_flux[ng,nt,i,:] = (w0_og[i,:]*F0PI/(4.*pi)
                        *(p_single[i,:])
                        *exp(-tau_og[i,:]/u0)
                        *(1. - exp(-dtau_og[i,:]*(u0+u1)/(u0*u1)))
                        *(u0/(u0+u1))
                        )
                xint[i,:] =( xint[i+1,:]*exp(-dtau[i,:]/u1) 
                        + direct_flux[ng,nt,i,:]
                        #multiple scattering terms p_single
                        +A[i,:]*(1. - exp(-dtau[i,:] *(u0+1*u1)/(u0*u1)))*
                        (u0/(u0+1*u1))
                        +G[i,:]*(exp(exptrm[i,:]*1-dtau[i,:]/u1) - 1.0)/(lamda[i,:]*1*u1 - 1.0)
                        +H[i,:]*(1. - exp(-exptrm[i,:]*1-dtau[i,:]/u1))/(lamda[i,:]*1*u1 + 1.0)
                        )

            xint_at_top[ng,nt,:] = xint[0,:]
            intensity[ng,nt,:,:] = xint

    return xint_at_top, flux_out, intensity

#@jit(nopython=True, cache=True)
def blackbody(t,w):
    """
    Blackbody flux in cgs units in per unit wavelength

    Parameters
    ----------
    t : array,float
        Temperature (K)
    w : array, float
        Wavelength (cm)
    
    Returns
    -------
    ndarray with shape ntemp x numwave
    """
    h = 6.62607004e-27 # erg s 
    c = 2.99792458e+10 # cm/s
    k = 1.38064852e-16 #erg / K

    return ((2.0*h*c**2.0)/(w**5.0))*(1.0/(exp((h*c)/outer(t, w*k)) - 1.0))

#@jit(nopython=True, cache=True)
def get_thermal_1d(nlevel, wno,nwno, numg,numt,tlevel, dtau, w0,cosb,plevel, ubar1,surf_reflect, tridiagonal):
    """
    This function uses the source function method, which is outlined here : 
    https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/JD094iD13p16287
    
    The result of this routine is the top of the atmosphere thermal flux as 
    a function of gauss and chebychev points accross the disk. 

    Everything here is in CGS units:

    Fluxes - erg/s/cm^3
    Temperature - K 
    Wave grid - cm-1
    Pressure ; dyne/cm2

    Reminder: Flux = pi * Intensity, so if you are trying to compare the result of this with 
    a black body you will need to compare with pi * BB !

    Parameters
    ----------
    nlevel : int 
        Number of levels which occur at the grid points (not to be confused with layers which are
        mid points)
    wno : numpy.ndarray
        Wavenumber grid in inverse cm 
    nwno : int 
        Number of wavenumber points 
    numg : int 
        Number of gauss points (think longitude points)
    numt : int 
        Number of chebychev points (think latitude points)
    tlevel : numpy.ndarray
        Temperature as a function of level (not layer)
    dtau : numpy.ndarray
        This is a matrix of nlayer by nwave. This describes the per layer optical depth. 
    w0 : numpy.ndarray
        This is a matrix of nlayer by nwave. This describes the single scattering albedo of 
        the atmosphere. Note this is free of any Raman scattering or any d-eddington correction 
        that is sometimes included in reflected light calculations.
    cosb : numpy.ndarray
        This is a matrix of nlayer by nwave. This describes the asymmetry of the 
        atmosphere. Note this is free of any Raman scattering or any d-eddington correction 
        that is sometimes included in reflected light calculations.
    plevel : numpy.ndarray
        Pressure for each level (not layer, which is midpoints). CGS units (dyne/cm2)
    ubar1 : numpy.ndarray
        This is a matrix of ng by nt. This describes the outgoing incident angles and is generally
        computed in `picaso.disco`
    surf_reflect : numpy.ndarray    
        Surface reflectivity as a function of wavenumber. 
    tridiagonal : int 
        0 for tridiagonal, 1 for pentadiagonal

    Returns
    -------
    numpy.ndarray
        Thermal flux in CGS units (erg/cm3/s) in a matrix that is 
        numg x numt x nwno
    """
    nlayer = nlevel - 1 #nlayers 

    mu1 = 0.5#0.88#0.5 #from Table 1 Toon  
    twopi = pi#+pi #NEB REMOVING A PI FROM HERE BECAUSE WE ASSUME NO SYMMETRY! 

    #get matrix of blackbodies 
    all_b = blackbody(tlevel, 1/wno) #returns nlevel by nwave   
    b0 = 0*all_b[0:-1,:]
    b1 = 0*(all_b[1:,:] - b0) / dtau # eqn 26 toon 89
    #b0 = zeros(b0.shape)
    #b1 = zeros(b1.shape)

    #hemispheric mean parameters from Tabel 1 toon 
    #**originally written in terms of alpha which isn't in the table. 
    #**changed to more closely resemble Toon (no change in actual values)
    alpha = sqrt( (1.-w0) / (1.-w0*cosb) )
    g1 = 2 - w0*(1 + cosb) # (7-w0*(4+3*cosb))/4  # 
    g2 = w0*(1 - cosb)     # -(1-w0*(4-3*cosb))/4 # 
    lamda = alpha*(1.-w0*cosb)/mu1 #(g1**2 - g2**2)**0.5 #eqn 21 toon
    gama = (1.-alpha)/(1.+alpha) #g2 / (g1 + lamda) #eqn 22 toon
    lamda = sqrt(g1**2 - g2**2)         #eqn 21
    gama  = (g1-lamda)/g2               #eqn 22
    g1_plus_g2 = 1/(g1 + g2) #mu1/(1.-w0*cosb) #effectively 1/(gamma1 + gamma2) .. second half of eqn.27

    #same as with reflected light, compute c_plus and c_minus 
    #these are eqns 27a & b in Toon89
    #_ups are evaluated at lower optical depth, TOA
    #_dows are evaluated at higher optical depth, bottom of atmosphere
    c_plus_up = (b0 + b1* g1_plus_g2)*pi # introduced pi here and removed from expressions for G,H,J,K
    c_minus_up = (b0 - b1* g1_plus_g2)*pi

    c_plus_down = (b0 + b1 * dtau + b1 * g1_plus_g2)*pi
    c_minus_down = (b0 + b1 * dtau - b1 * g1_plus_g2)*pi
    # note there should be a factor of 2mu1 in c expressions, need to include that if mu1 not 0.5

    #calculate exponential terms needed for the tridiagonal rotated layered method
    exptrm = lamda*dtau
    #save from overflow 
    exptrm = slice_gt (exptrm, 35.0) 

    exptrm_positive = exp(exptrm) 
    exptrm_minus = 1.0/exptrm_positive

    tau_top = dtau[0,:]*plevel[0]/(plevel[1]-plevel[0]) #tried this.. no luck*exp(-1)# #tautop=dtau[0]*np.exp(-1)
    b_top = (1.0 - exp(-tau_top / mu1 )) * all_b[0,:]  # Btop=(1.-np.exp(-tautop/ubari))*B[0]
    b_surface = all_b[-1,:] + b1[-1,:]*mu1 #Bsurf=B[-1] #    bottom=Bsurf+B1[-1]*ubari

    #Now we need the terms for the tridiagonal rotated layered method
    if tridiagonal==0:
        A, B, C, D = setup_tri_diag(nlayer,nwno,  c_plus_up, c_minus_up, 
                            c_plus_down, c_minus_down, b_top, b_surface, surf_reflect,
                             gama, dtau, 
                            exptrm_positive,  exptrm_minus) 
    positive = zeros((nlayer, nwno))
    negative = zeros((nlayer, nwno))
    #========================= Start loop over wavelength =========================
    L = nlayer+nlayer
    for w in range(nwno):
        #coefficient of posive and negative exponential terms 
        if tridiagonal==0:
            X = tri_diag_solve(L, A[:,w], B[:,w], C[:,w], D[:,w])
            #unmix the coefficients
            positive[:,w] = X[::2] + X[1::2] 
            negative[:,w] = X[::2] - X[1::2]

    #if you stop here this is regular ole 2 stream
    f_up = 0*pi*(positive * exptrm_positive + gama * negative * exptrm_minus + c_plus_up)


    #calculate everyting from Table 3 toon
    #alphax = ((1.0-w0)/(1.0-w0*cosb))**0.5
    #G = twopi*w0*positive*(1.0+cosb*alphax)/(1.0+alphax)#
    #H = twopi*w0*negative*(1.0-cosb*alphax)/(1.0+alphax)#
    #J = twopi*w0*positive*(1.0-cosb*alphax)/(1.0+alphax)#
    #K = twopi*w0*negative*(1.0+cosb*alphax)/(1.0+alphax)#
    #alpha1 = twopi*(b0+ b1*(mu1*w0*cosb/(1.0-w0*cosb)))
    #alpha2 = twopi*b1
    #sigma1 = twopi*(b0- b1*(mu1*w0*cosb/(1.0-w0*cosb)))
    #sigma2 = twopi*b1
    G = positive*(1.0/mu1 - lamda)
    H = negative*gama*(1.0/mu1 + lamda)
    J = positive*gama*(1.0/mu1 + lamda)
    K = negative*(1.0/mu1 - lamda)
    alpha1 = twopi*(b0+ b1*w0*(g1_plus_g2 - mu1))
    alpha2 = twopi*b1
    sigma1 = twopi*(b0- b1*(g1_plus_g2 - mu1))
    sigma2 = twopi*b1

    flux_minus = zeros((nlevel,nwno))
    flux_plus = zeros((nlevel,nwno))
    flux_minus_mdpt = zeros((nlevel,nwno))
    flux_plus_mdpt = zeros((nlevel,nwno))

    exptrm_positive_mdpt = exp(0.5*exptrm) 
    exptrm_minus_mdpt = 1/exptrm_positive_mdpt 

    #================ START CRAZE LOOP OVER ANGLE #================
    flux_at_top = zeros((numg, numt, nwno))
    flux_down = zeros((numg, numt, nwno))

    #work through building eqn 55 in toon (tons of bookeeping exponentials)
    for ng in range(numg):
        for nt in range(numt): 

            iubar = ubar1[ng,nt]

            flux_plus[-1,:] = twopi * (b_surface + b1[-1,:] * iubar)
            #flux_plus[-1,:] = zeros(flux_plus[-1,:].shape)
            flux_minus[0,:] = twopi * (1 - exp(-tau_top / iubar)) * all_b[0,:]
            
            exptrm_angle = exp( - dtau / iubar)
            exptrm_angle_mdpt = exp( -0.5 * dtau / iubar) 

            for itop in range(nlayer):

                #disbanning this for now because we dont need it in the thermal emission code
                flux_minus[itop+1,:]=(flux_minus[itop,:]*exptrm_angle[itop,:]+
                                     (J[itop,:]/(lamda[itop,:]*iubar+1.0))*(exptrm_positive[itop,:]-exptrm_angle[itop,:])+
                                     (K[itop,:]/(lamda[itop,:]*iubar-1.0))*(exptrm_angle[itop,:]-exptrm_minus[itop,:])+
                                     sigma1[itop,:]*(1.-exptrm_angle[itop,:])+
                                     sigma2[itop,:]*(iubar*exptrm_angle[itop,:]+dtau[itop,:]-iubar) )

                flux_minus_mdpt[itop,:]=(flux_minus[itop,:]*exptrm_angle_mdpt[itop,:]+
                                        (J[itop,:]/(lamda[itop,:]*iubar+1.0))*(exptrm_positive_mdpt[itop,:]-exptrm_angle_mdpt[itop,:])+
                                        (K[itop,:]/(-lamda[itop,:]*iubar+1.0))*(exptrm_minus_mdpt[itop,:]-exptrm_angle_mdpt[itop,:])+
                                        sigma1[itop,:]*(1.-exptrm_angle_mdpt[itop,:])+
                                        sigma2[itop,:]*(iubar*exptrm_angle_mdpt[itop,:]+0.5*dtau[itop,:]-iubar))

                ibot=nlayer-1-itop

                flux_plus[ibot,:]=(flux_plus[ibot+1,:]*exptrm_angle[ibot,:]+
                                  (G[ibot,:]/(lamda[ibot,:]*iubar-1.0))*(exptrm_positive[ibot,:]*exptrm_angle[ibot,:]-1.0)+
                                  (H[ibot,:]/(lamda[ibot,:]*iubar+1.0))*(1.0-exptrm_minus[ibot,:] * exptrm_angle[ibot,:])+
                                  alpha1[ibot,:]*(1.-exptrm_angle[ibot,:])+
                                  alpha2[ibot,:]*(iubar-(dtau[ibot,:]+iubar)*exptrm_angle[ibot,:]) )

                flux_plus_mdpt[ibot,:]=(flux_plus[ibot+1,:]*exptrm_angle_mdpt[ibot,:]+
                                       (G[ibot,:]/(lamda[ibot,:]*iubar-1.0))*(exptrm_positive[ibot,:]*exptrm_angle_mdpt[ibot,:]-exptrm_positive_mdpt[ibot,:])-
                                       (H[ibot,:]/(lamda[ibot,:]*iubar+1.0))*(exptrm_minus[ibot,:]*exptrm_angle_mdpt[ibot,:]-exptrm_minus_mdpt[ibot,:])+
                                       alpha1[ibot,:]*(1.-exptrm_angle_mdpt[ibot,:])+
                                       alpha2[ibot,:]*(iubar+0.5*dtau[ibot,:]-(dtau[ibot,:]+iubar)*exptrm_angle_mdpt[ibot,:])  )

            flux_at_top[ng,nt,:] = flux_plus[0,:]#flux_plus_mdpt[0,:] #nlevel by nwno #
            #flux_down[ng,nt,:] = flux_minus_mdpt[0,:] #nlevel by nwno, Dont really need to compute this for now

    import IPython; IPython.embed()
    import sys; sys.exit()
    return flux_at_top #, flux_down# numg x numt x nwno



#@jit(nopython=True, cache=True)
def get_thermal_3d(nlevel, wno,nwno, numg,numt,tlevel_3d, dtau_3d, w0_3d,cosb_3d,plevel_3d, ubar1, tridiagonal):
    """
    This function uses the source function method, which is outlined here : 
    https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/JD094iD13p16287
    
    The result of this routine is the top of the atmosphere thermal flux as 
    a function of gauss and chebychev points accross the disk. 

    Everything here is in CGS units:

    Fluxes - erg/s/cm^3
    Temperature - K 
    Wave grid - cm-1
    Pressure ; dyne/cm2

    Reminder: Flux = pi * Intensity, so if you are trying to compare the result of this with 
    a black body you will need to compare with pi * BB !

    Parameters
    ----------
    nlevel : int 
        Number of levels which occur at the grid points (not to be confused with layers which are
        mid points)
    wno : numpy.ndarray
        Wavenumber grid in inverse cm 
    nwno : int 
        Number of wavenumber points 
    numg : int 
        Number of gauss points (think longitude points)
    numt : int 
        Number of chebychev points (think latitude points)
    tlevel_3d : numpy.ndarray
        Temperature as a function of level (not layer). This 3d matrix has dimensions [nlevel,ngangle,ntangle].
    dtau_3d : numpy.ndarray
        This is a matrix of nlayer by nwave. This describes the per layer optical depth. 
        This 4d matrix has dimensions [nlevel, nwave,ngangle,ntangle].
    w0_3d : numpy.ndarray
        This is a matrix of nlayer by nwave. This describes the single scattering albedo of 
        the atmosphere. Note this is free of any Raman scattering or any d-eddington correction 
        that is sometimes included in reflected light calculations.
        This 4d matrix has dimensions [nlevel, nwave,ngangle,ntangle].
    cosb_3d : numpy.ndarray
        This is a matrix of nlayer by nwave. This describes the asymmetry of the 
        atmosphere. Note this is free of any Raman scattering or any d-eddington correction 
        that is sometimes included in reflected light calculations.
        This 4d matrix has dimensions [nlevel, nwave,ngangle,ntangle].
    plevel : numpy.ndarray
        Pressure for each level (not layer, which is midpoints). CGS units (dyne/cm2)
    ubar1 : numpy.ndarray
        This is a matrix of ng by nt. This describes the outgoing incident angles and is generally
        computed in `picaso.disco`
    tridiagonal : int 
        Zero for tridiagonal solver. 1 for pentadiagonal (not yet implemented)

    Returns
    -------
    numpy.ndarray
        Thermal flux in CGS units (erg/cm3/s) in a matrix that is 
        numg x numt x nwno
    """ 

    nlayer = nlevel - 1 #nlayers 
    mu1 = 0.5 #from Table 1 Toon 

    #twopi is just pi because we are assuming no lat/lon symmetry
    #the 2 comes back in when we do the gauss/tchebychev angle integration
    twopi = pi#+pi 

    #eventual output
    flux_at_top = zeros((numg, numt, nwno))
    flux_down = zeros((numg, numt, nwno))
    
    #in 3D we have to immediately loop through ng and nt 
    #so that we can use different TP profiles at each point
    for ng in range(numg):
        for nt in range(numt): 

            cosb = cosb_3d[:,:,ng,nt]
            dtau = dtau_3d[:,:,ng,nt]
            w0 = w0_3d[:,:,ng,nt]
            tlevel = tlevel_3d[:, ng,nt]
            plevel = plevel_3d[:, ng,nt]

            #get matrix of blackbodies 
            all_b = blackbody(tlevel, 1/wno) #returns nlevel by nwave   
            b0 = all_b[0:-1,:]
            b1 = (all_b[1:,:] - b0) / dtau # eqn 26 toon 89

            #hemispheric mean parameters from Tabe 1 toon 
            alpha = sqrt( (1.-w0) / (1.-w0*cosb) )
            lamda = alpha*(1.-w0*cosb)/mu1 #eqn 21 toon
            gama = (1.-alpha)/(1.+alpha) #eqn 22 toon
            g1_plus_g2 = mu1/(1.-w0*cosb) #effectively 1/(gamma1 + gamma2) .. second half of eqn.27

            #same as with reflected light, compute c_plus and c_minus 
            #these are eqns 27a & b in Toon89
            #_ups are evaluated at lower optical depth, TOA
            #_dows are evaluated at higher optical depth, bottom of atmosphere
            c_plus_up = b0 + b1* g1_plus_g2
            c_minus_up = b0 - b1* g1_plus_g2

            c_plus_down = b0 + b1 * dtau + b1 * g1_plus_g2 
            c_minus_down = b0 + b1 * dtau - b1 * g1_plus_g2

            #calculate exponential terms needed for the tridiagonal rotated layered method
            exptrm = lamda*dtau
            #save from overflow 
            exptrm = slice_gt (exptrm, 35.0) 

            exptrm_positive = exp(exptrm) 
            exptrm_minus = 1.0/exptrm_positive#exp(-exptrm) 

            tau_top = dtau[0,:]*plevel[0]/(plevel[1]-plevel[0])
            b_top = (1.0 - exp(-tau_top / mu1 )) * all_b[0,:] 
            b_surface = all_b[-1,:] + b1[-1,:]*mu1
            surf_reflect = 0.

            #Now we need the terms for the tridiagonal rotated layered method
            if tridiagonal==0:
                A, B, C, D = setup_tri_diag(nlayer,nwno,  c_plus_up, c_minus_up, 
                                    c_plus_down, c_minus_down, b_top, b_surface, surf_reflect,
                                     gama, dtau, 
                                    exptrm_positive,  exptrm_minus) 
            #else:
            #   A_, B_, C_, D_, E_, F_ = setup_pent_diag(nlayer,nwno,  c_plus_up, c_minus_up, 
            #                       c_plus_down, c_minus_down, b_top, b_surface, surf_reflect,
            #                        gama, dtau, 
            #                       exptrm_positive,  exptrm_minus, g1,g2,exptrm,lamda) 

            positive = zeros((nlayer, nwno))
            negative = zeros((nlayer, nwno))
            #========================= Start loop over wavelength =========================
            L = nlayer+nlayer
            for w in range(nwno):
                #coefficient of posive and negative exponential terms 
                if tridiagonal==0:
                    X = tri_diag_solve(L, A[:,w], B[:,w], C[:,w], D[:,w])
                    #unmix the coefficients
                    positive[:,w] = X[::2] + X[1::2] 
                    negative[:,w] = X[::2] - X[1::2]
                #else:
                #   X = pent_diag_solve(L, A_[:,w], B_[:,w], C_[:,w], D_[:,w], E_[:,w], F_[:,w])
                #   positive[:,w] = exptrm_minus[:,w] * (X[::2] + X[1::2])
                #   negative[:,w] = X[::2] - X[1::2]

            f_up = pi*(positive * exptrm_positive + gama * negative * exptrm_minus + c_plus_up)

            #calculate everyting from Table 3 toon
            alphax = ((1.0-w0)/(1.0-w0*cosb))**0.5
            G = twopi*w0*positive*(1.0+cosb*alphax)/(1.0+alphax)#
            H = twopi*w0*negative*(1.0-cosb*alphax)/(1.0+alphax)#
            J = twopi*w0*positive*(1.0-cosb*alphax)/(1.0+alphax)#
            K = twopi*w0*negative*(1.0+cosb*alphax)/(1.0+alphax)#
            alpha1 = twopi*(b0+ b1*(mu1*w0*cosb/(1.0-w0*cosb)))
            alpha2 = twopi*b1
            sigma1 = twopi*(b0- b1*(mu1*w0*cosb/(1.0-w0*cosb)))
            sigma2 = twopi*b1

            flux_minus = zeros((nlevel,nwno))
            flux_plus = zeros((nlevel,nwno))
            flux_minus_mdpt = zeros((nlevel,nwno))
            flux_plus_mdpt = zeros((nlevel,nwno))

            exptrm_positive_mdpt = exp(0.5*exptrm) 
            exptrm_minus_mdpt = 1/exptrm_positive_mdpt 

            #================ START CRAZE LOOP OVER ANGLE #================

            #work through building eqn 55 in toon (tons of bookeeping exponentials)
            flux_plus[-1,:] = twopi * (b_surface + b1[-1,:] * ubar1[ng,nt])
            flux_minus[0,:] = twopi * (1 - exp(-tau_top / ubar1[ng,nt])) * all_b[0,:]
            
            exptrm_angle = exp( - dtau / ubar1[ng,nt])
            exptrm_angle_mdpt = exp( -0.5 * dtau / ubar1[ng,nt]) 

            for itop in range(nlayer):

                #disbanning this for now because we dont need it in the thermal emission code
                flux_minus[itop+1,:]=(flux_minus[itop,:]*exptrm_angle[itop,:]+
                                     (J[itop,:]/(lamda[itop,:]*ubar1[ng,nt]+1.0))*(exptrm_positive[itop,:]-exptrm_angle[itop,:])+
                                     (K[itop,:]/(lamda[itop,:]*ubar1[ng,nt]-1.0))*(exptrm_angle[itop,:]-exptrm_minus[itop,:])+
                                     sigma1[itop,:]*(1.-exptrm_angle[itop,:])+
                                     sigma2[itop,:]*(ubar1[ng,nt]*exptrm_angle[itop,:]+dtau[itop,:]-ubar1[ng,nt]) )

                flux_minus_mdpt[itop,:]=(flux_minus[itop,:]*exptrm_angle_mdpt[itop,:]+
                                        (J[itop,:]/(lamda[itop,:]*ubar1[ng,nt]+1.0))*(exptrm_positive_mdpt[itop,:]-exptrm_angle_mdpt[itop,:])+
                                        (K[itop,:]/(-lamda[itop,:]*ubar1[ng,nt]+1.0))*(exptrm_minus_mdpt[itop,:]-exptrm_angle_mdpt[itop,:])+
                                        sigma1[itop,:]*(1.-exptrm_angle_mdpt[itop,:])+
                                        sigma2[itop,:]*(ubar1[ng,nt]*exptrm_angle_mdpt[itop,:]+0.5*dtau[itop,:]-ubar1[ng,nt]))

                ibot=nlayer-1-itop

                flux_plus[ibot,:]=(flux_plus[ibot+1,:]*exptrm_angle[ibot,:]+
                                  (G[ibot,:]/(lamda[ibot,:]*ubar1[ng,nt]-1.0))*(exptrm_positive[ibot,:]*exptrm_angle[ibot,:]-1.0)+
                                  (H[ibot,:]/(lamda[ibot,:]*ubar1[ng,nt]+1.0))*(1.0-exptrm_minus[ibot,:] * exptrm_angle[ibot,:])+
                                  alpha1[ibot,:]*(1.-exptrm_angle[ibot,:])+
                                  alpha2[ibot,:]*(ubar1[ng,nt]-(dtau[ibot,:]+ubar1[ng,nt])*exptrm_angle[ibot,:]) )

                flux_plus_mdpt[ibot,:]=(flux_plus[ibot+1,:]*exptrm_angle_mdpt[ibot,:]+
                                       (G[ibot,:]/(lamda[ibot,:]*ubar1[ng,nt]-1.0))*(exptrm_positive[ibot,:]*exptrm_angle_mdpt[ibot,:]-exptrm_positive_mdpt[ibot,:])-
                                       (H[ibot,:]/(lamda[ibot,:]*ubar1[ng,nt]+1.0))*(exptrm_minus[ibot,:]*exptrm_angle_mdpt[ibot,:]-exptrm_minus_mdpt[ibot,:])+
                                       alpha1[ibot,:]*(1.-exptrm_angle_mdpt[ibot,:])+
                                       alpha2[ibot,:]*(ubar1[ng,nt]+0.5*dtau[ibot,:]-(dtau[ibot,:]+ubar1[ng,nt])*exptrm_angle_mdpt[ibot,:])  )

            flux_at_top[ng,nt,:] = flux_plus_mdpt[0,:] #nlevel by nwno
            #flux_down[ng,nt,:] = flux_minus_mdpt[0,:] #nlevel by nwno, Dont really need to compute this for now

    return flux_at_top #, flux_down# numg x numt x nwno

def get_transit_1d(z, dz,nlevel, nwno, rstar, mmw, k_b,amu,
                    player, tlayer, colden, DTAU):
    """
    Routine to get the transmission spectrum 

    Parameters
    ----------
    z : float, array
        Altitude in decreasing order (cm)
    dz : float, array 
        length of each atmospheric layer 
    nlevel : int 
        Number of levels 
    nwno : int
        Number of wavelength points 
    rstar : float 
        Radius of star (cm)
    mmw : float, array
        Mean molecular weight 
    k_b : float 
        Boltzman constant cgs 
    amu : float 
        Atomic mass units cgs 
    player : float, array
        Pressure at layers (dyn/cm2)
    tlayer : float, array 
        Temperature at layers (K)
    colden : float, array
        Column density conputed in atmsetup.get_column_density()
    DTAU : float, array
        Matrix of summed tau opacities from optics. This is 
        TAUGAS + TAURAY + TAUCLD

    Returns
    -------
    array 
        Rp**2 /Rs**2 as a function of wavelength 
    """
    mmw = mmw * amu #make sure mmw in grams

    delta_length=zeros((nlevel,nlevel))

    for i in range(nlevel):
        for j in range(i):
            reference_shell = z[i]
            inner_shell = z[i-j]
            outer_shell = z[i-j-1]
            #this is the path length between two layers 
            #essentially tangent from the inner_shell and toward 
            #line of sight to the outer shell
            integrate_segment=((outer_shell**2-reference_shell**2)**0.5-
                    (inner_shell**2-reference_shell**2)**0.5)
            #make sure to use the pressure and temperature  
            #between inner and outer shell
            #this is the same index as outer shell because ind = 0 is the outer-
            #most layer 
            delta_length[i,j]=integrate_segment*player[i-j-1]/tlayer[i-j-1]/k_b
            
    #remove column density and mmw from DTAU which was calculated in 
    #optics because line of site integration is diff for transit
    TAU = array([DTAU[:,i]  / colden * mmw  for i in range(nwno)])

    transmitted=zeros((nwno, nlevel))+1.0
    for i in range(nlevel):
        TAUALL=0.
        for j in range(i):
            #two because symmetry of sphere
            TAUALL += 2*TAU[:,i-j-1]*delta_length[i,j]
        transmitted[:,i]=exp(-TAUALL)

    F=(((min(z))/(rstar))**2 + 
        2./(rstar)**2.*dot((1.-transmitted),z*dz))

    return F


def get_transit_3d(nlevel, nwno, radius, gravity,rstar, mass, mmw, k_b, G,amu,
                   p_reference, plevel, tlevel, player, tlayer, colden, DTAU):
    """
    Routine to get the 3D transmission spectrum 
    """
    return 

#@jit(nopython=True, cache=True, debug=True)
def get_reflected_new(nlevel, wno, nwno, numg, numt, dtau, tau, w0, cosb, gcos2, ftau_cld, ftau_ray,
    dtau_og, tau_og, w0_og, cosb_og, 
    surf_reflect, ubar0, ubar1, cos_theta, F0PI, single_phase, multi_phase, 
    frac_a, frac_b, frac_c, constant_back, constant_forward, dim, stream, b_top=0, flx=0):
    """
    Computes rooney fluxes given tau and everything is 3 dimensional. This is the exact same function 
    as `get_flux_geom_1d` but is kept separately so we don't have to do unecessary indexing for 
    retrievals. 
    
    Parameters
    ----------
    nlevel : int 
        Number of levels in the model 
    nwno : int 
        Number of wave points
    numg : int 
        Number of Gauss angles 
    numt : int 
        Number of Chebyshev angles 
    dtau_3d : ndarray of float
        This is the opacity contained within each individual layer (defined at midpoints of "levels")
        WITHOUT D-Eddington Correction
        Dimensions=# layer by # wave
    tau_3d : ndarray of float
        This is the cumulative summed opacity 
        WITHOUT D-Eddington Correction
        Dimensions=# level by # wave        
    w0_3d : ndarray of float 
        This is the single scattering albedo, from scattering, clouds, raman, etc 
        WITHOUT D-Eddington Correction
        Dimensions=# layer by # wave
    cosb_3d : ndarray of float 
        This is the asymmetry factor 
        WITHOUT D-Eddington Correction
        Dimensions=# layer by # wave
    gcos2_3d : ndarray of float 
        Parameter that allows us to directly include Rayleigh scattering 
        = 0.5*tau_rayleigh/(tau_rayleigh + tau_cloud)
    dtau_og_3d : ndarray of float 
        This is the opacity contained within each individual layer (defined at midpoints of "levels")
        WITHOUT the delta eddington correction, if it was specified by user
        Dimensions=# layer by # wave
    tau_og_3d : ndarray of float
        This is the cumulative summed opacity 
        WITHOUT the delta eddington correction, if it was specified by user
        Dimensions=# level by # wave    
    w0_og_3d : ndarray of float 
        Same as w0 but WITHOUT the delta eddington correction, if it was specified by user  
    cosb_og_3d : ndarray of float 
        Same as cosbar buth WITHOUT the delta eddington correction, if it was specified by user
    surf_reflect : float 
        Surface reflectivity 
    ubar0 : ndarray of float 
        matrix of cosine of the incident angle from geometric.json
    ubar1 : ndarray of float 
        matrix of cosine of the observer angles
    F0PI : array 
        Downward incident solar radiation
    Returns
    -------
    intensity at the top of the atmosphere for all the different ubar1 and ubar2 
    
    To Do
    -----
    - F0PI Solar flux shouldn't always be 1.. Follow up to make sure that this isn't a bad 
          hardwiring to solar, despite "relative albedo"
    - take delta eddington option out of fluxes and move it all over to optics
    
    """
    #what we want : intensity at the top as a function of all the different angles
    
    nlayer = nlevel - 1 
    xint_at_top = zeros((numg, numt, nwno))
    xint_out = zeros((numg, numt, nlevel, nwno))
    flux = zeros((numg, numt, stream*nlevel, nwno))
    def P(mu): # Legendre polynomials
        return [1, mu, (3*mu**2 - 1)/2, (5*mu**3 - 3*mu)/2,
            (35*mu**4 - 30*mu**2 + 3)/8, 
            (63*mu**5 - 70*mu**3 + 15*mu)/8, 
            (231*mu**6 - 315*mu**4 + 105*mu**2 - 5)/16 ]
    
    #================ START CRAZE LOOP OVER ANGLE #================
    for ng in range(numg):
        for nt in range(numt):
            u1 = ubar1[ng,nt]
            u0 = ubar0[ng,nt]

            a = zeros((stream, nlayer, nwno))
            b = zeros((stream, nlayer, nwno))
            w_single = ones((stream, nlayer, nwno))
            w_multi = ones(((stream, nlayer, nwno)))
            if array_equal(cosb,cosb_og):
                ff = 0.*cosb_og
            else:
                ff = cosb_og**stream
            p_single = ones(cosb_og.shape)

            if single_phase!=1: 
                g_forward = constant_forward*cosb_og
                g_back = constant_back*cosb_og
                f = frac_a + frac_b*g_back**frac_c
                if array_equal(cosb,cosb_og):
                    ff1 = 0.*cosb_og
                    ff2 = 0.*cosb_og
                else:
                    ff1 = (constant_forward*cosb_og)**stream
                    ff2 = (constant_back*cosb_og)**stream

            if single_phase==1:#'OTHG':
                for l in range(1,stream):
                    w_multi[l,:,:] = (2*l+1) * (cosb_og**l - ff) / (1 - ff)
                    w_single[l,:,:] = (2*l+1) * (cosb_og**l -  ff) / (1-ff)

                    #cos_theta = -u0 * u1 + sqrt(1-u0**2) * sqrt(1-u1**2)
                    #p_single=(1-cosb_og**2)/(sqrt(1+cosb_og**2+2*cosb_og*cos_theta)**3) 
                    #p_single = p_single + w_single[l,:,:] * P(-u0)[l]*P(u1)[l]
                p_single=(1-cosb_og**2)/(sqrt(1+cosb_og**2+2*cosb_og*cos_theta)**3) 

            elif single_phase==2:#'TTHG':
                #Phase function for single scattering albedo frum Solar beam
                #uses the Two term Henyey-Greenstein function
                for l in range(1,stream):
                    #w_multi[l,:,:] = (2*l+1) * (f*(g_forward**l - ff1) / (1 - ff) 
                    #                    + (1-f)*(g_back**l - ff2) / (1 - ff))
                    #w_single[l,:,:] = (2*l+1) * (f*(g_forward**l - ff1) / (1 - ff) 
                    #                    + (1-f)*(g_back**l - ff2) / (1 - ff))
                    w_multi[l,:,:] = (2*l+1) * (cosb_og**l -  ff) / (1-ff)
                    w_single[l,:,:] = (2*l+1) * (cosb_og**l -  ff) / (1-ff)

                    #cos_theta = -u0 * u1 + sqrt(1-u0**2) * sqrt(1-u1**2)
                    #p_single=(1-cosb_og**2)/(sqrt(1+cosb_og**2+2*cosb_og*cos_theta)**3) 
                    #p_single = p_single + w_single[l,:,:] * P(-u0)[l]*P(u1)[l]

                p_single=(f * (1-g_forward**2) /sqrt((1+g_forward**2+2*g_forward*cos_theta)**3) 
                                #second term of TTHG: backward scattering
                                +(1-f)*(1-g_back**2)
                                /sqrt((1+g_back**2+2*g_back*cos_theta)**3))

            elif single_phase==3:#'TTHG_ray':
                # not happy with Rayleigh, not getting the same inclusion in multiscattering as picaso
                for l in range(1,stream):
                    #w_multi[l,:,:] = (2*l+1) * (f*(g_forward**l - ff1) / (1 - ff) 
                    #                    + (1-f)*(g_back**l - ff2) / (1 - ff))
                    #w_single[l,:,:] = (2*l+1) * (f*(g_forward**l - ff1) / (1 - ff) 
                    #                    + (1-f)*(g_back**l - ff2) / (1 - ff))
                    w_multi[l,:,:] = (2*l+1) * (cosb_og**l -  ff) / (1-ff)
                    w_single[l,:,:] = (2*l+1) * (cosb_og**l -  ff) / (1-ff)

                    if l==2:
                        w_single[2] = w_single[2] + 0.5*ftau_ray
                        w_multi[2] = w_multi[2] + 0.5*ftau_ray

                    #cos_theta = -u0 * u1 + sqrt(1-u0**2) * sqrt(1-u1**2)
                    #p_single=(1-cosb_og**2)/(sqrt(1+cosb_og**2+2*cosb_og*cos_theta)**3) 
                    #p_single = p_single + w_single[l,:,:] * P(-u0)[l]*P(u1)[l]

                p_single=(ftau_cld*(f * (1-g_forward**2)
                                                /sqrt((1+g_forward**2+2*g_forward*cos_theta)**3) 
                                                #second term of TTHG: backward scattering
                                                +(1-f)*(1-g_back**2)
                                                /sqrt((1+g_back**2+2*g_back*cos_theta)**3))+            
                                #rayleigh phase function
                                ftau_ray*(0.75*(1+cos_theta**2.0)))


            for l in range(stream):
                a[l,:,:] = (2*l + 1) -  w0 * w_multi[l,:,:]
                b[l,:,:] = ( F0PI * (w0 * w_single[l,:,:])) * P(-u0)[l] / (4*pi)

            #boundary conditions 
            b_surface = 0. + surf_reflect*u0*F0PI*exp(-tau[-1, :]/u0)

            if stream==2:
                M, B, A_int, N_int, F_bot, G_bot, F, G, Q1, Q2 = setup_2_stream_banded(nlayer, wno, nwno, w0, b_top, b_surface, 
                surf_reflect, F0PI, u0, dtau, tau, a, b, u1, P, fluxes=flx) 

            if stream==4:
                M, B, A_int, N_int, F_bot, G_bot, F, G = setup_4_stream_banded(nlayer, wno, nwno, w0, b_top, b_surface, 
                surf_reflect, F0PI, u0, dtau, tau, a, b, u1, P, fluxes=flx) 
                # F and G will be nonzero if fluxes=1

            flux_bot = zeros(nwno)
            intgrl_new = zeros((stream*nlayer, nwno))
            flux_temp = zeros((stream*nlevel, nwno))
            intgrl_per_layer = zeros((nlayer, nwno))
            xint_temp = zeros((nlevel, nwno))
            multi_scat = zeros((nlayer, nwno))

            #========================= Start loop over wavelength =========================
            for W in range(nwno):
                (intgrl_new[:,W], flux_bot[W], X) = solve_4_stream_banded(M[:,:,W], B[:,W],  
                A_int[:,:,W], N_int[:,W], F_bot[:,W], G_bot[W], stream, nlayer)
                if flx==1:
                    flux_temp[:,W] = calculate_flux(F[:,:,W], G[:,W], X)

            mus = (u1 + u0) / (u1 * u0)
            expo_mus = mus * dtau_og 
            expo_mus = slice_gt(expo_mus, 35.0)    
            exptrm_mus = exp(-expo_mus)

            for i in range(nlayer):
                for l in range(stream):
                    multi_scat[i,:] = multi_scat[i,:] + w_multi[l,i,:] * P(u1)[l] * intgrl_new[stream*i+l,:]

            intgrl_per_layer = (w0 *  multi_scat 
                        + w0_og * F0PI / (4*np.pi) * p_single 
                        * (1 - exptrm_mus) * exp(-tau_og[:-1,:]/u0)
                        / mus
                        )

            xint_temp[-1,:] = flux_bot/pi
            for i in range(nlayer-1,-1,-1):
                xint_temp[i, :] = (xint_temp[i+1, :] * np.exp(-dtau[i,:]/u1)
                            + intgrl_per_layer[i,:] / u1) 

            xint_at_top[ng,nt,:] = xint_temp[0, :]
            xint_out[ng,nt,:,:] = xint_temp
            flux[ng,nt,:,:] = flux_temp
    
    return xint_at_top, flux, xint_out

def get_thermal_new(nlevel, wno, nwno, numg, numt, tlevel, dtau, tau, w0, cosb, 
            dtau_og, tau_og, w0_og, w0_no_raman, cosb_og, plevel, ubar1,
            constant_forward, constant_back, frac_a, frac_b, frac_c,
            surf_reflect, single_phase, dimension, stream, flx=0, calculation=1):
    nlayer = nlevel - 1 #nlayers 

    mu1 = 0.5#0.88#0.5 #from Table 1 Toon  

    def P(mu): # Legendre polynomials
        return [1, mu, (3*mu**2 - 1)/2, (5*mu**3 - 3*mu)/2,
            (35*mu**4 - 30*mu**2 + 3)/8, 
            (63*mu**5 - 70*mu**3 + 15*mu)/8, 
            (231*mu**6 - 315*mu**4 + 105*mu**2 - 5)/16 ]
    
    #get matrix of blackbodies 
    all_b = blackbody(tlevel, 1/wno) #returns nlevel by nwave   
    b0 = 0*all_b[0:-1,:]
    if calculation == 1: # linear thermal
        b1 = 0*(all_b[1:,:] - b0) / dtau # eqn 26 toon 89
        f0 = 0.
    elif calculation == 2: # exponential thermal
        b1 = all_b[1:,:] 
        f0 = -1/dtau * log(b1/b0)

    
    tau_top = dtau[0,:]*plevel[0]/(plevel[1]-plevel[0]) #tried this.. no luck*exp(-1)# #tautop=dtau[0]*np.exp(-1)
    b_top = (1.0 - exp(-tau_top / mu1 )) * all_b[0,:]  # Btop=(1.-np.exp(-tautop/ubari))*B[0]
    #b_surface = all_b[-1,:] + b1[-1,:]*mu1 #Bsurf=B[-1] #    bottom=Bsurf+B1[-1]*ubari
    b_surface = all_b[-1,:] + 0*(all_b[1:,:]-b0)[-1,:]*mu1 #Bsurf=B[-1] #    bottom=Bsurf+B1[-1]*ubari
    
    #if single_phase==1:#'OTHG':
    if np.array_equal(cosb,cosb_og):
        ff = 0.
    else:
        ff = cosb_og**stream

    w_single = zeros((stream, nlayer, nwno))
    w_multi = zeros(((stream, nlayer, nwno)))
    a = zeros(((stream, nlayer, nwno)))
    b = zeros(((stream, nlayer, nwno)))
    for l in range(stream):
        w_multi[l,:,:] = (2*l+1) * (cosb_og**l - ff) / (1 - ff)
        a[l,:,:] = (2*l + 1) -  w0 * w_multi[l,:,:]
    b[0] = (1-w0) * b0

    xint_at_top = zeros((numg, numt, nwno))
    for ng in range(numg):
        for nt in range(numt):
            if stream==2:
                M, B, A_int, N_int, F_bot, G_bot, F, G, Q1, Q2 = setup_2_stream_banded(nlayer, wno, nwno, w0, b_top, b_surface, 
                surf_reflect, 0, ubar1[ng,nt], dtau, tau, a, b, ubar1[ng,nt], P, b0, b1, f0, fluxes=flx, calculation=calculation)

            elif stream==4:
                M, B, A_int, N_int, F_bot, G_bot, F, G = setup_4_stream_banded(nlayer, wno, nwno, w0, b_top, b_surface, 
                surf_reflect, 0, ubar1[ng,nt], dtau, tau, a, b, ubar1[ng,nt], P, b0, b1, f0, fluxes=flx, calculation=calculation) 
                # F and G will be nonzero if fluxes=1

            flux_bot = zeros(nwno)
            intgrl_new = zeros((stream*nlayer, nwno))
            X = zeros((stream*nlayer, nwno))
            intgrl_per_layer = zeros((nlayer, nwno))
            multi_scat = zeros((nlayer, nwno))
            xint_temp = zeros((nlevel, nwno))
            #========================= Start loop over wavelength =========================
            for W in range(nwno):
                (intgrl_new[:,W], flux_bot[W], X) = solve_4_stream_banded(M[:,:,W], B[:,W],  
                A_int[:,:,W], N_int[:,W], F_bot[:,W], G_bot[W], stream, nlayer)
                if flx==1:
                    flux_temp[:,W] = calculate_flux(F[:,:,W], G[:,W], X)

            for i in range(nlayer):
                for l in range(stream):
                    multi_scat[i,:] = multi_scat[i,:] + w_multi[l,i,:] * P(ubar1[ng,nt])[l] * intgrl_new[stream*i+l,:]

            if calculation==1:
                expo = dtau_og / ubar1[ng,nt] 
                expo_mus = slice_gt(expo, 35.0)    
                expdtau = exp(-expo)

                intgrl_per_layer = (w0 *  multi_scat 
                            + pi * ((1-w0_og) * ubar1[ng,nt] *
                            (b0 * (1 - expdtau)
                            + b1 * (ubar1[ng,nt] - (dtau_og + ubar1[ng,nt]) * expdtau))))

            elif calculation==2:
                expo = dtau * (f0 + 1/ubar1[ng,nt])
                expo = slice_gt(expo, 35.0)    
                expdtau = exp(-expo)

                intgrl_per_layer = (w0 *  multi_scat 
                            + (1-w0) 
                            * b0 * (1 - expdtau)
                            / (f0 + 1/ubar1[ng,nt]) )


            xint_temp[-1,:] = pi * (b_surface + b1[-1,:] * ubar1[ng,nt])#zeros(flux_bot.shape)#
            for i in range(nlayer-1,-1,-1):
                xint_temp[i, :] = (xint_temp[i+1, :] * np.exp(-dtau[i,:]/ubar1[ng,nt]) 
                            + intgrl_per_layer[i,:] / ubar1[ng,nt]) 

            xint_temp = xint_temp #* pi/2 
            xint_at_top[ng,nt,:] = xint_temp[0, :]
    
    import IPython; IPython.embed()
    import sys; sys.exit()
    return xint_at_top 

#@jit(nopython=True, cache=True)
def setup_2_stream_banded(nlayer, wno, nwno, w0, b_top, b_surface, surf_reflect, F0PI, ubar0, dtau,tau, 
        a, b, ubar1, P, B0=0., B1=0., f0=0., fluxes=0, calculation=0):#'reflected'):

    if calculation==0:
        Del = ((1 / ubar0)**2 - a[0]*a[1])
        eta = [(b[1] /ubar0 - a[1]*b[0]) / Del,
            (b[0] /ubar0 - a[0]*b[1]) / Del]
    elif calculation==2:
        Del = f0**2 - a[0]*a[1]
        eta = [(-a[1] * b[0]) / Del,
                (f0 * b[0]) / Del]

    lam = sqrt(a[0]*a[1])
    expo = lam*dtau
    expo = slice_gt(expo, 35.0) 
    exptrm = exp(-expo)

    #   parameters in matrices
    q = lam/a[1]
    Q1 = 2*pi*(0.5 + q)
    Q2 = 2*pi*(0.5 - q)

    Q1mn = Q1*exptrm;  Q2mn = Q2*exptrm
    Q1pl = Q1/exptrm;  Q2pl = Q2/exptrm

    if calculation != 1:
        zmn = 2*pi*(0.5*eta[0] - eta[1]) 
        zpl = 2*pi*(0.5*eta[0] + eta[1])
        if calculation == 0:
            expon = exp(-tau/ubar0)
            zmn_up = zmn * expon[1:,:] 
            zpl_up = zpl * expon[1:,:] 
            zmn_down = zmn * expon[:-1,:] 
            zpl_down = zpl * expon[:-1,:] 
        elif calculation == 2:
            expon = exp(-slice_gt(dtau*f0, 35.0))
            zmn_up = zmn * expon
            zpl_up = zpl * expon
            zmn_down = zmn 
            zpl_down = zpl 
    elif calculation == 1: # linear thermal
        zmn_down = 2*pi**2 * (1-w0)/a[0] * (B0/2 - B1/a[1])#+ B1*dtau/2)
        zmn_up = 2*pi**2 * (1-w0)/a[0] * (B0/2 - B1/a[1] + B1*dtau/2)
        zpl_down = 2*pi**2 * (1-w0)/a[0] * (B0/2 + B1/a[1])# + B1*dtau/2)
        zpl_up = 2*pi**2 * (1-w0)/a[0] * (B0/2 + B1/a[1] + B1*dtau/2)

    alpha = 1/ubar1 + lam
    beta = 1/ubar1 - lam
    expo_alp = slice_gt(alpha * dtau, 35.0)
    expo_bet = slice_gt(beta * dtau, 35.0) 
    exptrm_alp = (1 - exp(-expo_alp)) / alpha 
    exptrm_bet = (1 - exp(-expo_bet)) / beta

    if calculation == 0:
        mus = (ubar1 + ubar0) / (ubar1 * ubar0)
        expo_mus = slice_gt(mus * dtau, 35.0)    
        exptrm_mus = (1 - exp(-expo_mus)) / mus
        tau_mu = tau[:-1,:] * 1/ubar0
        tau_mu = slice_gt(tau_mu, 35.0)
        exptau_mu = exp(-tau_mu)
        expon1 = exptrm_mus * exptau_mu
    elif calculation == 2:
        f0_ubar = f0 + 1/ubar1
        expo_f0 = slice_gt(f0_ubar * dtau, 35.0)
        exptrm_f0 = (1 - exp(-expo_f0)) / f0_ubar
        expon1 = exptrm_f0

    #   construct matrices
    Mb = zeros((5, 2*nlayer, nwno))
    B = zeros((2*nlayer, nwno))
    A_int = zeros((2*nlayer, 2*nlayer, nwno))
    N_int = zeros((2*nlayer, nwno))
    nlevel = nlayer+1
    F = zeros((2*nlevel, 2*nlayer, nwno))
    G = zeros((2*nlevel, nwno))

    #   first row: BC 1
    Mb[2,0,:] = Q1[0,:]
    Mb[1,1,:] = Q2[0,:]
    B[0,:] = b_top - zmn_down[0,:]

    Mb[0,3::2,:] = -Q2[1:,:]
    Mb[1,2::2,:] = -Q1[1:,:]
    Mb[1,3::2,:] = -Q1[1:,:]
    Mb[2,1:-1:2,:] = Q2pl[:-1,:]
    Mb[2,2::2,:] = -Q2[1:,:]
    Mb[3,:-2:2,:] = Q1mn[:-1,:]
    Mb[3,1:-1:2,:] = Q1pl[:-1,:]
    Mb[4,:-2:2,:] = Q2mn[:-1,:]
    B[1:-1:2,:] = zmn_down[1:,:] - zmn_up[:-1,:]
    B[2::2,:] = zpl_down[1:,:] - zpl_up[:-1,:]

    nn = np.arange(2*nlayer)
    indcs = nn[::2]
    k = 0
    for i in indcs:
        A_int[i,i,:] = exptrm_alp[k,:]
        A_int[i,i+1,:] = exptrm_bet[k,:]
        A_int[i+1,i,:] = (-q * exptrm_alp)[k,:]
        A_int[i+1,i+1,:] = (q * exptrm_bet)[k,:]
        k = k+1

    if calculation == 0 or calculation == 2: # reflected or exponential thermal
        N_int[::2,:] = eta[0] * expon1
        N_int[1::2,:] = eta[1] * expon1
    elif calculation == 1: # linear thermal
        expdtau = exp(-dtau/ubar1)
        N_int[::2,:] = pi * (1-w0) * ubar1 / a[0] * ((B0 + B1*tau[:-1,:])*(1-expdtau) + B1*(ubar1 - (dtau+ubar1)*expdtau))
        N_int[1::2,:] = pi * (1-w0) * ubar1 / a[0] * ( B1*(1-expdtau) / a[1])

    #   last row: BC 4
    n = nlayer-1
    Mb[3, 2*nlayer-2,:] = Q2mn[n,:] - surf_reflect*Q1mn[n,:]
    Mb[2, 2*nlayer-1,:] = Q1pl[n,:] - surf_reflect*Q2pl[n,:]
    B[2*nlayer-1,:] = b_surface - zpl_up[n,:] + surf_reflect * zmn_up[n,:]

    F_bot = zeros((2*nlayer, nwno))
    G_bot = zeros(nwno)
    F_bot[-2,:] = Q2mn[-1,:]
    F_bot[-1,:] = Q1pl[-1,:]
    G_bot = zpl_up[-1,:]


    if fluxes == 1: # fluxes per layer
        F[0,0,:] = Q1[0,:]
        F[0,1,:] = Q2[0,:]
        F[1,0,:] = Q2[0,:]
        F[1,1,:] = Q1[0,:]

        nn = np.arange(2*nlayer)
        indcs = nn[::2]
        k = 0
        for i in indcs:
            F[i+2,i,:] = Q1mn[k,:]
            F[i+2,i+1,:] = Q2pl[k,:]
            F[i+3,i,:] = Q2mn[k,:]
            F[i+3,i+1,:] = Q1pl[k,:]
            k = k+1

        G[0,:] = zmn[0,:]
        G[1,:] = zpl[0,:]

        G[2::2,:] = zmn_up
        G[3::2,:] = zpl_up

    return Mb, B, A_int, N_int, F_bot, G_bot, F, G, Q1, Q2

#@jit(nopython=True, cache=True, debug=True)
def setup_4_stream_banded(nlayer, wno, nwno, w0, b_top, b_surface, surf_reflect, F0PI, ubar0, dtau,tau, 
        a, b, ubar1, P, B0=0., B1=0., f0=0., fluxes=0, calculation=0):#'reflected'):

    nlevel = nlayer+1
    beta = a[0]*a[1] + 4*a[0]*a[3]/9 + a[2]*a[3]/9
    gama = a[0]*a[1]*a[2]*a[3]/9
    lam1 = sqrt((beta + sqrt(beta**2 - 4*gama)) / 2)
    lam2 = sqrt((beta - sqrt(beta**2 - 4*gama)) / 2)

    def f(x):
        return x**4 - beta*x**2 + gama
    
    if calculation != 1:
        Dels = zeros((4, nlayer, nwno))
        if calculation==0:
            Del = 9 * f(1/ubar0)
            Dels[0,:,:] = ((a[1]*b[0] - b[1]/ubar0) * (a[2]*a[3] - 9/ubar0**2) 
                + 2*(a[3]*b[2] - 2*a[3]*b[0] - 3*b[3]/ubar0)/ubar0**2)
            Dels[1,:,:] = ((a[0]*b[1] - b[0]/ubar0) * (a[2]*a[3] - 9/ubar0**2) 
                - 2*a[0]*(a[3]*b[2] - 3*b[3]/ubar0)/ubar0)
            Dels[2,:,:] = ((a[3]*b[2] - 3*b[3]/ubar0) * (a[0]*a[1] - 1/ubar0**2) 
                - 2*a[3]*(a[0]*b[1] - b[0]/ubar0)/ubar0)
            Dels[3,:,:] = ((a[2]*b[3] - 3*b[2]/ubar0) * (a[0]*a[1] - 1/ubar0**2) 
                + 2*(3*a[0]*b[1] - 2*a[0]*b[3] - 3*b[0]/ubar0)/ubar0**2)
        elif calculation==2:
            b0 = (1-w0) * B0
            Del = 9 * f(f0)
            Dels[0,:,:] = a[1]*b0 * (a[2]*a[3] - 9*f0**2) - 4*a[3]*b0*f0**2
            Dels[1,:,:] = b0*f0 * (9*f0**2 - a[2]*a[3])
            Dels[2,:,:] = 2*a[3]*b0*f0**2
            Dels[3,:,:] = -6*b0*f0**3


        #eta = []
        eta = zeros((4, nlayer, nwno))
        for l in range(4):
            #eta.append(Dels[l]/Del)
            eta[l,:,:] = (Dels[l]/Del)

        z1pl = 2*pi*(eta[0]/2 + eta[1] + 5*eta[2]/8); 
        z1mn = 2*pi*(eta[0]/2 - eta[1] + 5*eta[2]/8);
        z2pl = 2*pi*(-eta[0]/8 + 5*eta[2]/8 + eta[3]); 
        z2mn = 2*pi*(-eta[0]/8 + 5*eta[2]/8 - eta[3]);
    
    expo1 = slice_gt(lam1*dtau, 35.0) 
    expo2 = slice_gt(lam2*dtau, 35.0) 
    exptrm1 = exp(-expo1)
    exptrm2 = exp(-expo2)

    R1 = -a[0]/lam1; R2 = -a[0]/lam2
    Q1 = 1/2 * (a[0]*a[1]/(lam1**2) - 1); Q2 = 1/2 * (a[0]*a[1]/(lam2**2) - 1)
    S1 = -3/(2*a[3]) * (a[0]*a[1]/lam1 - lam1); S2 = -3/(2*a[3]) * (a[0]*a[1]/lam2 - lam2)
    
    p1pl = 2*pi*(1/2 + R1 + 5*Q1/8); p1mn = 2*pi*(1/2 - R1 + 5*Q1/8);
    p2pl = 2*pi*(1/2 + R2 + 5*Q2/8); p2mn = 2*pi*(1/2 - R2 + 5*Q2/8);
    q1pl = 2*pi*(-1/8 + 5*Q1/8 + S1); q1mn = 2*pi*(-1/8 + 5*Q1/8 - S1)
    q2pl = 2*pi*(-1/8 + 5*Q2/8 + S2); q2mn = 2*pi*(-1/8 + 5*Q2/8 - S2)

    f00 = p1mn*exptrm1; f01 = p1pl/exptrm1; f02 = p2mn*exptrm2; f03 = p2pl/exptrm2
    f10 = q1mn*exptrm1; f11 = q1pl/exptrm1; f12 = q2mn*exptrm2; f13 = q2pl/exptrm2
    f20 = p1pl*exptrm1; f21 = p1mn/exptrm1; f22 = p2pl*exptrm2; f23 = p2mn/exptrm2
    f30 = q1pl*exptrm1; f31 = q1mn/exptrm1; f32 = q2pl*exptrm2; f33 = q2mn/exptrm2

    if calculation == 0:# 'reflected':
        expon = exp(-slice_gt(tau/ubar0, 35.0))
        z1mn_up = z1mn * expon[1:,:]
        z2mn_up = z2mn * expon[1:,:]
        z1pl_up = z1pl * expon[1:,:]
        z2pl_up = z2pl * expon[1:,:]
        z1mn_down = z1mn * expon[:-1,:]
        z2mn_down = z2mn * expon[:-1,:]
        z1pl_down = z1pl * expon[:-1,:]
        z2pl_down = z2pl * expon[:-1,:]
    elif calculation == 2: # exponential thermal
        expon = exp(-slice_gt(dtau * f0, 35.0))
        z1mn_up = z1mn * expon
        z2mn_up = z2mn * expon
        z1pl_up = z1pl * expon
        z2pl_up = z2pl * expon
        z1mn_down = z1mn 
        z2mn_down = z2mn 
        z1pl_down = z1pl 
        z2pl_down = z2pl 
    elif calculation == 1: # linear thermal
        z1mn_up = 2*pi * (1-w0)/a[0] * (B0/2 - B1/a[1] + B1*tau[1:,:]/2) 
        z2mn_up = -pi * (1-w0) / (4*a[0]) * (B0 + B1*tau[1:,:]) 
        z1pl_up = 2*pi * (1-w0)/a[0] * (B0/2 + B1/a[1] + B1*tau[1:,:]/2) 
        z2pl_up = -pi * (1-w0) / (4*a[0]) * (B0 + B1*tau[1:,:]) 
        z1mn_down = 2*pi * (1-w0)/a[0] * (B0/2 - B1/a[1] + B1*tau[:-1,:]/2) 
        z2mn_down = -pi * (1-w0) / (4*a[0]) * (B0 + B1*tau[:-1,:]) 
        z1pl_down = 2*pi * (1-w0)/a[0] * (B0/2 + B1/a[1] + B1*tau[:-1,:]/2) 
        z2pl_down = -pi * (1-w0) / (4*a[0]) * (B0 + B1*tau[:-1,:]) 

    alpha1 = 1/ubar1 + lam1
    alpha2 = 1/ubar1 + lam2
    beta1 = 1/ubar1 - lam1
    beta2 = 1/ubar1 - lam2
    expo_alp1 = alpha1 * dtau
    expo_alp2 = alpha2 * dtau
    expo_bet1 = beta1 * dtau
    expo_bet2 = beta2 * dtau
    expo_alp1 = slice_gt(expo_alp1, 35.0)
    expo_alp2 = slice_gt(expo_alp2, 35.0)
    expo_bet1 = slice_gt(expo_bet1, 35.0)
    expo_bet2 = slice_gt(expo_bet2, 35.0)
    exptrm_alp1 = exp(-expo_alp1)
    exptrm_alp2 = exp(-expo_alp2)
    exptrm_bet1 = exp(-expo_bet1)
    exptrm_bet2 = exp(-expo_bet2)

    A00 = (1-exptrm_alp1)/alpha1; A01 = (1-exptrm_bet1)/beta1; 
    A02 = (1-exptrm_alp2)/alpha2; A03 = (1-exptrm_bet2)/beta2
    A10 = R1 * A00; A11 = -R1 * A01; A12 = R2 * A02; A13 = -R2 * A03; 
    A20 = Q1 * A00; A21 =  Q1 * A01; A22 = Q2 * A02; A23 =  Q2 * A03; 
    A30 = S1 * A00; A31 = -S1 * A01; A32 = S2 * A02; A33 = -S2 * A03; 
    
    if calculation != 1:
        if calculation == 0:#is 'reflected':
            mus = (ubar1 + ubar0) / (ubar1 * ubar0)
            expo_mus = mus * dtau 
            expo_mus = slice_gt(expo_mus, 35.0)    
            exptrm_mus = exp(-expo_mus)
            tau_mu = tau[:-1,:] * (1/ubar0)
            tau_mu = slice_gt(tau_mu, 35.0)
            exptau_mu = exp(-tau_mu)
            expon1 = (1 - exptrm_mus) * exptau_mu / mus
        elif calculation == 2: # exponential thermal
            expo_thermal = dtau * (f0 + 1/ubar1)
            expon1 = (1 - exp(-slice_gt(expo_thermal, 35.0))) / (f0 + 1/ubar1)
        N0 = eta[0] * expon1;   
        N1 = eta[1] * expon1;   
        N2 = eta[2] * expon1;   
        N3 = eta[3] * expon1;   
    elif calculation == 1:
        #expdtau = exp(-tau[:-1,:]/ubar1)
        expdtau = exp(-dtau/ubar1)
        N0 = (1-w0) * ubar1 / a[0] * ( (B0+B1*tau[:-1,:])*(1-expdtau) + B1*(ubar1 - (dtau+ubar1)*expdtau))
        #N0 = (1-w0) * ubar1 / a[0] * ( B0*(1-expdtau) + B1*(ubar1 - (dtau+ubar1)*expdtau))
        N1 = (1-w0) * ubar1 / a[0] * ( B1*(1-expdtau) / a[1])
        N2 = zeros(w0.shape)
        N3 = zeros(w0.shape)

    Mb = zeros((11, 4*nlayer, nwno))
    B = zeros((4*nlayer, nwno))
    A_int = zeros((4*nlayer, 4*nlayer, nwno))
    N_int = zeros((4*nlayer, nwno))
    F_bot = zeros((4*nlayer, nwno))
    G_bot = zeros(nwno)
    F = zeros((4*nlevel, 4*nlayer, nwno))
    G = zeros((4*nlevel, nwno))

    Mb[5,0,:] = p1mn[0,:]
    Mb[5,1,:] = q1pl[0,:]
    Mb[4,1,:] = p1pl[0,:]
    Mb[4,2,:] = q2mn[0,:]
    Mb[3,2,:] = p2mn[0,:]
    Mb[3,3,:] = q2pl[0,:]
    Mb[2,3,:] = p2pl[0,:]
    Mb[6,0,:] = q1mn[0,:]

    B[0,:] = b_top - z1mn_down[0,:]
    B[1,:] = -b_top/4 - z2mn_down[0,:]

    Mb[5,2:-4:4,:] = f02[:-1,:]
    Mb[5,3:-4:4,:] = f13[:-1,:]
    Mb[5,4::4,:] = -p1pl[1:,:]
    Mb[5,5::4,:] = -q1mn[1:,:]
      
    Mb[4,3:-4:4,:] = f03[:-1,:]
    Mb[4,4::4,:] = -q1mn[1:,:]
    Mb[4,5::4,:] = -p1mn[1:,:]
    Mb[4,6::4,:] = -q2pl[1:,:]
      
    Mb[3,4::4,:] = -p1mn[1:,:]
    Mb[3,5::4,:] = -q1pl[1:,:]
    Mb[3,6::4,:] = -p2pl[1:,:]
    Mb[3,7::4,:] = -q2mn[1:,:]
      
    Mb[2,5::4,:] = -p1pl[1:,:]
    Mb[2,6::4,:] = -q2mn[1:,:]
    Mb[2,7::4,:] = -p2mn[1:,:]
      
    Mb[1,6::4,:] = -p2mn[1:,:]
    Mb[1,7::4,:] = -q2pl[1:,:]
      
    Mb[0,7::4,:] = -p2pl[1:,:]
      
    Mb[6,1:-4:4,:] = f01[:-1,:]
    Mb[6,2:-4:4,:] = f12[:-1,:]
    Mb[6,3:-4:4,:] = f23[:-1,:]
    Mb[6,4::4,:] = -q1pl[1:,:]
      
    Mb[7,0:-4:4,:] = f00[:-1,:]
    Mb[7,1:-4:4,:] = f11[:-1,:]
    Mb[7,2:-4:4,:] = f22[:-1,:]
    Mb[7,3:-4:4,:] = f33[:-1,:]
      
    Mb[8,0:-4:4,:] = f10[:-1,:]
    Mb[8,1:-4:4,:] = f21[:-1,:]
    Mb[8,2:-4:4,:] = f32[:-1,:]
      
    Mb[9,0:-4:4,:] = f20[:-1,:]
    Mb[9,1:-4:4,:] = f31[:-1,:]
      
    Mb[10,0:-4:4,:] = f30[:-1,:]

    B[2:-4:4,:] = z1mn_down[1:,:] - z1mn_up[:-1,:]
    B[3:-4:4,:] = z2mn_down[1:,:] - z2mn_up[:-1,:]
    B[4::4,:] = z1pl_down[1:,:] - z1pl_up[:-1,:]
    B[5::4,:] = z2pl_down[1:,:] - z2pl_up[:-1,:]

    NN = np.arange(4*nlayer)
    indcs = NN[::4]
    k = 0
    for i in indcs:
        A_int[i,i,:] = A00[k,:]
        A_int[i,i+1,:] = A01[k,:]
        A_int[i,i+2,:] = A02[k,:]
        A_int[i,i+3,:] = A03[k,:]
        A_int[i+1,i,:] = A10[k,:]
        A_int[i+1,i+1,:] = A11[k,:]
        A_int[i+1,i+2,:] = A12[k,:]
        A_int[i+1,i+3,:] = A13[k,:]
        A_int[i+2,i,:] = A20[k,:]
        A_int[i+2,i+1,:] = A21[k,:]
        A_int[i+2,i+2,:] = A22[k,:]
        A_int[i+2,i+3,:] = A23[k,:]
        A_int[i+3,i,:] = A30[k,:]
        A_int[i+3,i+1,:] = A31[k,:]
        A_int[i+3,i+2,:] = A32[k,:]
        A_int[i+3,i+3,:] = A33[k,:]
        k = k+1

    N_int[::4,:] = N0
    N_int[1::4,:] = N1
    N_int[2::4,:] = N2
    N_int[3::4,:] = N3

    n = nlayer-1
    Mb[5,4*nlayer-2,:] = f22[n,:] - surf_reflect*f02[n,:]
    Mb[5,4*nlayer-1,:] = f33[n,:] - surf_reflect*f13[n,:]
    Mb[4,4*nlayer-1,:] = f23[n,:] - surf_reflect*f03[n,:]
    Mb[6,4*nlayer-3,:] = f21[n,:] - surf_reflect*f01[n,:]
    Mb[6,4*nlayer-2,:] = f32[n,:] - surf_reflect*f12[n,:]
    Mb[7,4*nlayer-4,:] = f20[n,:] - surf_reflect*f00[n,:]
    Mb[7,4*nlayer-3,:] = f31[n,:] - surf_reflect*f11[n,:]
    Mb[8,4*nlayer-4,:] = f30[n,:] - surf_reflect*f10[n,:]

    B[4*nlayer-2,:] = b_surface - z1pl_up[n,:] + surf_reflect*z1mn_up[n,:]
    #B[4*nlayer-1,:] = b_surface - z2pl_up[n,:] + surf_reflect*z2mn_up[n,:]
    B[4*nlayer-1,:] = - z2pl_up[n,:] + surf_reflect*z2mn_up[n,:]

    F_bot[-4,:] = f20[-1,:]
    F_bot[-3,:] = f21[-1,:]
    F_bot[-2,:] = f22[-1,:]
    F_bot[-1,:] = f23[-1,:]
    G_bot = z1pl_up[-1,:]

    if fluxes == 1: # fluxes per layer
        F[0,0,:] = p1mn[0,:]
        F[0,1,:] = p1pl[0,:]
        F[0,2,:] = p2mn[0,:]
        F[0,3,:] = p2pl[0,:]
        F[1,0,:] = q1mn[0,:]
        F[1,1,:] = q1pl[0,:]
        F[1,2,:] = q2mn[0,:]
        F[1,3,:] = q2pl[0,:]
        F[2,0,:] = p1pl[0,:]
        F[2,1,:] = p1mn[0,:]
        F[2,2,:] = p2pl[0,:]
        F[2,3,:] = p2mn[0,:]
        F[3,0,:] = q1pl[0,:]
        F[3,1,:] = q1mn[0,:]
        F[3,2,:] = q2pl[0,:]
        F[3,3,:] = q2mn[0,:]

        nn = np.arange(4*nlayer)
        indcs = nn[::4]
        k = 0
        for i in indcs:
            F[i+4,i,:] = f00[k,:]
            F[i+4,i+1,:] = f01[k,:]
            F[i+4,i+2,:] = f02[k,:]
            F[i+4,i+3,:] = f03[k,:]
            F[i+5,i,:] = f10[k,:]
            F[i+5,i+1,:] = f11[k,:]
            F[i+5,i+2,:] = f12[k,:]
            F[i+5,i+3,:] = f13[k,:]
            F[i+6,i,:] = f20[k,:]
            F[i+6,i+1,:] = f21[k,:]
            F[i+6,i+2,:] = f22[k,:]
            F[i+6,i+3,:] = f23[k,:]
            F[i+7,i,:] = f30[k,:]
            F[i+7,i+1,:] = f31[k,:]
            F[i+7,i+2,:] = f32[k,:]
            F[i+7,i+3,:] = f33[k,:]
            k = k+1
                             
        G[0,:] = z1mn[0,:]
        G[1,:] = z2mn[0,:]
        G[2,:] = z1pl[0,:]
        G[3,:] = z2pl[0,:]
        G[4::4,:] = z1mn_up
        G[5::4,:] = z2mn_up
        G[6::4,:] = z1pl_up
        G[7::4,:] = z2pl_up

    return Mb, B, A_int, N_int, F_bot, G_bot, F, G

#@jit(nopython=True, cache=True)
def solve_4_stream_banded(M, B, A_int, N_int, F, G, stream, nlayer):
    #   find constants
    diag = int(3*stream/2 - 1)
    with objmode(X='float64[:]'):
        X = solve_banded((diag,diag), M, B)
    #   integral of Iexp(-tau/ubar1) at each level 
    intgrl_new =  A_int.dot(X) + N_int 
    #   flux at bottom
    flux = F.dot(X) + G
    return (intgrl_new, flux, X)

#@jit(nopython=True, cache=True)
def calculate_flux(F, G, X):
    return F.dot(X) + G
