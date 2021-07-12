from numba import jit, vectorize
from numpy import exp, zeros, where, sqrt, cumsum , pi, outer, sinh, cosh, min, dot, array,log
import numpy as np
#import pentapy as pp
import time
import pickle as pk
from scipy.sparse.linalg import spsolve

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

@jit(nopython=True, cache=True)
def slice_eq(array, lim, value):
    """Funciton to replace values with upper or lower limit
    """
    for i in range(array.shape[0]):
        new = array[i,:] 
        new[where(new==lim)] = value
        array[i,:] = new     
    return array

@jit(nopython=True, cache=True)
def slice_lt(array, lim):
    """Funciton to replace values with upper or lower limit
    """
    for i in range(array.shape[0]):
        new = array[i,:] 
        new[where(new<lim)] = lim
        array[i,:] = new     
    return array

@jit(nopython=True, cache=True)
def slice_gt(array, lim):
    """Funciton to replace values with upper or lower limit
    """
    for i in range(array.shape[0]):
        new = array[i,:] 
        new[where(new>lim)] = lim
        array[i,:] = new     
    return array

@jit(nopython=True, cache=True)
def numba_cumsum(mat):
    """Function to compute cumsum along axis=0 to bypass numba not allowing kwargs in 
    cumsum 
    """
    new_mat = zeros(mat.shape)
    for i in range(mat.shape[1]):
        new_mat[:,i] = cumsum(mat[:,i])
    return new_mat

@jit(nopython=True, cache=True)
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


@jit(nopython=True, cache=True)
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

@jit(nopython=True, cache=True)
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

@jit(nopython=True, cache=True)
def get_reflected_1d(nlevel, wno,nwno, numg,numt, dtau, tau, w0, cosb,gcos2, ftau_cld, ftau_ray,
    dtau_og, tau_og, w0_og, cosb_og, 
    surf_reflect,ubar0, ubar1,cos_theta, F0PI,single_phase, multi_phase,
    frac_a, frac_b, frac_c, constant_back, constant_forward, tridiagonal):
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

    nlayer = nlevel - 1 

    #now define terms of Toon et al 1989 quadrature Table 1 
    #https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/JD094iD13p16287
    #see table of terms 

    #terms not dependent on incident angle
    sq3 = sqrt(3.)
    g1  = (sq3*0.5)*(2. - w0*(1.+cosb)) #table 1 # (7-w0*(4+3*cosb))/4 #
    g2  = (sq3*w0*0.5)*(1.-cosb)        #table 1 # -(1-w0*(4-3*cosb))/4 #
    lamda = sqrt(g1**2 - g2**2)         #eqn 21
    gama  = (g1-lamda)/g2               #eqn 22

    #================ START CRAZE LOOP OVER ANGLE #================
    for ng in range(numg):
        for nt in range(numt):

            g3  = 0.5*(1.-sq3*cosb*ubar0[ng, nt]) #(2-3*cosb*ubar0[ng,nt])/4#  #table 1 #ubar has dimensions [gauss angles by tchebyshev angles ]
    
            # now calculate c_plus and c_minus (equation 23 and 24 toon)
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
            exptrm = slice_gt (exptrm, 35.0) 

            exptrm_positive = exp(exptrm) #EP
            exptrm_minus = 1.0/exptrm_positive#EM


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
                    #unmix the coefficients
                #   positive[:,w] = exptrm_minus[:,w] * (X[::2] + X[1::2])
                #   negative[:,w] = X[::2] - X[1::2]

            #========================= End loop over wavelength =========================

            #use expression for bottom flux to get the flux_plus and flux_minus at last
            #bottom layer
            flux_zero  = positive[-1,:]*exptrm_positive[-1,:] + gama[-1,:]*negative[-1,:]*exptrm_minus[-1,:] + c_plus_down[-1,:]
            #flux_minus  = gama*positive*exptrm_positive + negative*exptrm_minus + c_minus_down
            #flux_plus  = positive*exptrm_positive + gama*negative*exptrm_minus + c_plus_down
            #flux = zeros((2*nlayer, nwno))
            #flux[::2, :] = flux_minus
            #flux[1::2, :] = flux_plus

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
                xint[i,:] =( xint[i+1,:]*exp(-dtau[i,:]/ubar1[ng,nt]) 
                        #single scattering albedo from sun beam (from ubar0 to ubar1)
                        +(w0_og[i,:]*F0PI/(4.*pi))
                        *(p_single[i,:])*exp(-tau_og[i,:]/ubar0[ng,nt])
                        *(1. - exp(-dtau_og[i,:]*(ubar0[ng,nt]+ubar1[ng,nt])
                        /(ubar0[ng,nt]*ubar1[ng,nt])))*
                        (ubar0[ng,nt]/(ubar0[ng,nt]+ubar1[ng,nt]))
                        #multiple scattering terms p_single
                        +A[i,:]*(1. - exp(-dtau[i,:] *(ubar0[ng,nt]+1*ubar1[ng,nt])/(ubar0[ng,nt]*ubar1[ng,nt])))*
                        (ubar0[ng,nt]/(ubar0[ng,nt]+1*ubar1[ng,nt]))
                        +G[i,:]*(exp(exptrm[i,:]*1-dtau[i,:]/ubar1[ng,nt]) - 1.0)/(lamda[i,:]*1*ubar1[ng,nt] - 1.0)
                        +H[i,:]*(1. - exp(-exptrm[i,:]*1-dtau[i,:]/ubar1[ng,nt]))/(lamda[i,:]*1*ubar1[ng,nt] + 1.0)
                        )

            xint_at_top[ng,nt,:] = xint[0,:]

    return xint_at_top

@jit(nopython=True, cache=True)
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

@jit(nopython=True, cache=True)
def get_thermal_1d(nlevel, wno,nwno, numg,numt,tlevel, dtau, w0,cosb,plevel, ubar1,
    surf_reflect, hard_surface, tridiagonal):
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
    hard_surface : int
        0 for no hard surface (e.g. Jupiter/Neptune), 1 for hard surface (terrestrial)
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
    exptrm_minus = 1.0/exptrm_positive

    tau_top = dtau[0,:]*plevel[0]/(plevel[1]-plevel[0]) #tried this.. no luck*exp(-1)# #tautop=dtau[0]*np.exp(-1)
    b_top = (1.0 - exp(-tau_top / mu1 )) * all_b[0,:]  # Btop=(1.-np.exp(-tautop/ubari))*B[0]
    if hard_surface:
        b_surface = all_b[-1,:] #for terrestrial, hard surface  
    else: 
        b_surface=all_b[-1,:] + b1[-1,:]*mu1 #(for non terrestrial)

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

    #if you stop here this is regular ole 2 stream
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
    flux_at_top = zeros((numg, numt, nwno))
    flux_down = zeros((numg, numt, nwno))

    #work through building eqn 55 in toon (tons of bookeeping exponentials)
    for ng in range(numg):
        for nt in range(numt): 

            iubar = ubar1[ng,nt]

            if hard_surface:
                flux_plus[-1,:] = twopi * (b_surface ) # terrestrial
            else:
                flux_plus[-1,:] = twopi*( all_b[-1,:] + b1[-1,:] * iubar) #no hard surface
                
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

            flux_at_top[ng,nt,:] = flux_plus_mdpt[0,:] #nlevel by nwno 

            #to get the convective heat flux 
            #flux_minus_mdpt_disco[ng,nt,:,:] = flux_minus_mdpt #nlevel by nwno
            #flux_plus_mdpt_disco[ng,nt,:,:] = flux_plus_mdpt #nlevel by nwno

    return flux_at_top #, flux_down# numg x numt x nwno



@jit(nopython=True, cache=True)
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

@jit(nopython=True, cache=True)
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
    #TAU = array([DTAU[:,i]  / colden * mmw  for i in range(nwno)])
    TAU = zeros((nwno, nlevel-1))
    for i in range(nwno):
        TAU[i,:] = DTAU[:,i]  / colden * mmw 

    transmitted=zeros((nwno, nlevel))+1.0
    for i in range(nlevel):
        TAUALL=zeros(nwno)#0.
        for j in range(i):
            #two because symmetry of sphere
            TAUALL = TAUALL + 2*TAU[:,i-j-1]*delta_length[i,j]
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


def setup_4_stream_scaled(nlayer, nwno, W0, b_top, b_surface, surf_reflect, F0PI, ubar0, dtau,tau, w_single, w_multi, ubar1, P):
	"""
	Parameters
	----------
	nlayer : int 
		number of layers in the model 
	nwno : int 
		number of wavelength points ## need to include this
	W0: int
		single scattering albedo 
	b_top : array 
		The diffuse radiation into the model at the top of the atmosphere
	b_surface : array
		The diffuse radiation into the model at the bottom. Includes emission, reflection 
		of the unattenuated portion of the direct beam  
	surf_reflect : array 
		Surface reflectivity 
	F0PI : int  
		solar radiation
	ubar0: array
		cosine of solar incident angle
	dtau : array 
		Opacity per layer
	g : array 
		asymmetry parameters
	P : array
		Legendre polynomials
	"""
	w0 = W0.T
	dtau = dtau.T
	tau = tau.T

	a = []; b = []
	for l in range(4):
		a.append((2*l + 1) - w0 * w_multi[l])
		b.append((F0PI * (w0 * w_single[l]).T).T * P(-ubar0)[l] / (4 * np.pi))

	beta = a[0]*a[1] + 4*a[0]*a[3]/9 + a[2]*a[3]/9
	gama = a[0]*a[1]*a[2]*a[3]/9
	lam2 = np.sqrt((beta + np.sqrt(beta**2 - 4*gama)) / 2)
	lam1 = np.sqrt((beta - np.sqrt(beta**2 - 4*gama)) / 2)

	def f(x):
		return x**4 - beta*x**2 + gama
	
	Del = 9 * f(1/ubar0)
	Dels = []
	Dels.append((a[1]*b[0] - b[1]/ubar0) * (a[2]*a[3] - 9/ubar0**2) 
		+ 2*(a[3]*b[2] - 2*a[3]*b[0] - 3*b[3]/ubar0)/ubar0**2)
	Dels.append((a[0]*b[1] - b[0]/ubar0) * (a[2]*a[3] - 9/ubar0**2) 
		- 2*a[0]*(a[3]*b[2] - 3*b[3]/ubar0)/ubar0)
	Dels.append((a[3]*b[2] - 3*b[3]/ubar0) * (a[0]*a[1] - 1/ubar0**2) 
		- 2*a[3]*(a[0]*b[1] - b[0]/ubar0)/ubar0)
	Dels.append((a[2]*b[3] - 3*b[2]/ubar0) * (a[0]*a[1] - 1/ubar0**2) 
		+ 2*(3*a[0]*b[1] - 2*a[0]*b[3] - 3*b[0]/ubar0)/ubar0**2)
	
	eta = []
	for l in range(4):
		eta.append(Dels[l]/Del)

	expo1 = slice_gt(lam1*dtau, 35.0) 
	expo2 = slice_gt(lam2*dtau, 35.0) 
	exptrm1 = np.exp(-expo1)
	exptrm2 = np.exp(-expo2)

	R1 = -a[0]/lam1; R2 = -a[0]/lam2
	Q1 = 1/2 * (a[0]*a[1]/(lam1**2) - 1); Q2 = 1/2 * (a[0]*a[1]/(lam2**2) - 1)
	S1 = -3/(2*a[3]) * (a[0]*a[1]/lam1 - lam1); S2 = -3/(2*a[3]) * (a[0]*a[1]/lam2 - lam2)
	
	p1pl = 2*np.pi*(1/2 + R1 + 5*Q1/8); p1mn = 2*np.pi*(1/2 - R1 + 5*Q1/8);
	p2pl = 2*np.pi*(1/2 + R2 + 5*Q2/8); p2mn = 2*np.pi*(1/2 - R2 + 5*Q2/8);
	q1pl = 2*np.pi*(-1/8 + 5*Q1/8 + S1); q1mn = 2*np.pi*(-1/8 + 5*Q1/8 - S1)
	q2pl = 2*np.pi*(-1/8 + 5*Q2/8 + S2); q2mn = 2*np.pi*(-1/8 + 5*Q2/8 - S2)
	z1pl = 2*np.pi*(eta[0]/2 + eta[1] + 5*eta[2]/8); z1mn = 2*np.pi*(eta[0]/2 - eta[1] + 5*eta[2]/8);
	z2pl = 2*np.pi*(-eta[0]/8 + 5*eta[2]/8 + eta[3]); z2mn = 2*np.pi*(-eta[0]/8 + 5*eta[2]/8 - eta[3]);
	
	zero = np.zeros(nwno)

	f00 = p1mn*exptrm1; f01 = p1pl/exptrm1;	f02 = p2mn*exptrm2; f03 = p2pl/exptrm2
	f10 = q1mn*exptrm1; f11 = q1pl/exptrm1;	f12 = q2mn*exptrm2; f13 = q2pl/exptrm2
	f20 = p1pl*exptrm1; f21 = p1mn/exptrm1;	f22 = p2pl*exptrm2; f23 = p2mn/exptrm2
	f30 = q1pl*exptrm1; f31 = q1mn/exptrm1;	f32 = q2pl*exptrm2; f33 = q2mn/exptrm2
	def F_block(n, lev):
		if lev == "up":
			return (np.array([[f00[:,n], f10[:,n], f20[:,n], f30[:,n]],
					    [f01[:,n], f11[:,n], f21[:,n], f31[:,n]],
			    		    [f02[:,n], f12[:,n], f22[:,n], f32[:,n]],
			    		    [f03[:,n], f13[:,n], f23[:,n], f33[:,n]]])).T
		elif lev == "down":
			return (np.array([[p1mn[:,n], q1mn[:,n], p1pl[:,n], q1pl[:,n]],
					    [p1pl[:,n], q1pl[:,n], p1mn[:,n], q1mn[:,n]],
					    [p2mn[:,n], q2mn[:,n], p2pl[:,n], q2pl[:,n]],
					    [p2pl[:,n], q2pl[:,n], p2mn[:,n], q2mn[:,n]]])).T

		#block = np.array([[p1mn[:,n]*e1mn, q1mn[:,n]*e1mn, p1pl[:,n]*e1mn, q1pl[:,n]*e1mn],
		#			[p1pl[:,n]*e1pl, q1pl[:,n]*e1pl, p1mn[:,n]*e1pl, q1mn[:,n]*e1pl],
		#			[p2mn[:,n]*e2mn, q2mn[:,n]*e2mn, p2pl[:,n]*e2mn, q2pl[:,n]*e2mn],
		#			[p2pl[:,n]*e2pl, q2pl[:,n]*e2pl, p2mn[:,n]*e2pl, q2mn[:,n]*e2pl]])
		#return block.T
	
	exptau_u0 = np.exp(-tau/ubar0)
	z1mn_up = z1mn * exptau_u0[:,1:]
	z2mn_up = z2mn * exptau_u0[:,1:]
	z1pl_up = z1pl * exptau_u0[:,1:]
	z2pl_up = z2pl * exptau_u0[:,1:]
	z1mn_down = z1mn * exptau_u0[:,:-1]
	z2mn_down = z2mn * exptau_u0[:,:-1]
	z1pl_down = z1pl * exptau_u0[:,:-1]
	z2pl_down = z2pl * exptau_u0[:,:-1]

	def Z_block(n, lev):
		if lev == "up":
			return (np.array([z1mn_up[:,n], z2mn_up[:,n], z1pl_up[:,n], z2pl_up[:,n]])).T
		elif lev == "down":
			return (np.array([z1mn_down[:,n], z2mn_down[:,n], z1pl_down[:,n], z2pl_down[:,n]])).T

	
	a00 = exptrm1;	    a01 = 1/exptrm1;	a02 = exptrm2;	    a03 = 1/exptrm2
	a10 = R1*exptrm1;   a11 = -R1/exptrm1;	a12 = R2*exptrm2;   a13 = -R2/exptrm2
	a20 = Q1*exptrm1;   a21 = Q1/exptrm1;	a22 = Q1*exptrm2;   a23 = Q1/exptrm2
	a30 = S1*exptrm1;   a31 = -S1/exptrm1;	a32 = S1*exptrm2;   a33 = -S1/exptrm2
	def A_block(n, lev):
		if lev == "up":
			return (np.array([[a00[:,n], a10[:,n], a20[:,n], a30[:,n]],
						[a01[:,n], a11[:,n], a21[:,n], a31[:,n]],
						[a02[:,n], a12[:,n], a22[:,n], a32[:,n]],
						[a03[:,n], a13[:,n], a23[:,n], a33[:,n]]])).T
		elif lev == "down":
			return (np.array([[np.ones(nwno), R1[:,n], Q1[:,n], S1[:,n]],
					    [np.ones(nwno), -R1[:,n], Q1[:,n], -S1[:,n]],
					    [np.ones(nwno),  R2[:,n], Q2[:,n],  S2[:,n]],
					    [np.ones(nwno), -R2[:,n], Q2[:,n], -S2[:,n]]])).T

		#block = np.array([[e1mn, R1[:,n]*e1mn, Q1[:,n]*e1mn, S1[:,n]*e1mn],
		#	[e1pl, -R1[:,n]*e1pl, Q1[:,n]*e1pl, -S1[:,n]*e1pl],
		#	[e2mn,  R2[:,n]*e2mn, Q2[:,n]*e2mn,  S2[:,n]*e2mn],
		#	[e2pl, -R2[:,n]*e2pl, Q2[:,n]*e2pl, -S2[:,n]*e2pl]])

		#return block.T
	
	n0_up = eta[0]*exptau_u0[:,1:]; n0_down = eta[0]*exptau_u0[:,:-1]
	n1_up = eta[1]*exptau_u0[:,1:]; n1_down = eta[1]*exptau_u0[:,:-1]
	n2_up = eta[2]*exptau_u0[:,1:]; n2_down = eta[2]*exptau_u0[:,:-1]
	n3_up = eta[3]*exptau_u0[:,1:]; n3_down = eta[3]*exptau_u0[:,:-1]
	def N_block(n, lev):
		if lev == "up":
			return (np.array([n0_up[:,n], n1_up[:,n], n2_up[:,n], n3_up[:,n]])).T
		elif lev == "down":
			return (np.array([n0_down[:,n], n1_down[:,n], n2_down[:,n], n3_down[:,n]])).T


	alpha1 = 1/ubar1 + lam1;		beta1 = 1/ubar1 - lam1;			mus = (ubar1 + ubar0) / (ubar1 * ubar0)
	alpha2 = 1/ubar1 + lam2;		beta2 = 1/ubar1 - lam2;
	expo_alp1 = alpha1 * dtau;		expo_bet1 = beta1 * dtau;		expo_mus = mus * dtau 
	expo_alp2 = alpha2 * dtau;		expo_bet2 = beta2 * dtau;		
	expo_alp1 = slice_gt(expo_alp1, 35.0);	expo_bet1 = slice_gt(expo_bet1, 35.0);	expo_mus = slice_gt(expo_mus, 35.0)    
	expo_alp2 = slice_gt(expo_alp2, 35.0);	expo_bet2 = slice_gt(expo_bet2, 35.0);	
	exptrm_alp1 = np.exp(-expo_alp1);	exptrm_bet1 = np.exp(-expo_bet1);	exptrm_mus = np.exp(-expo_mus)
	exptrm_alp2 = np.exp(-expo_alp2);	exptrm_bet2 = np.exp(-expo_bet2);

	A00 = (1-exptrm_alp1)/alpha1;	A01 = (1-exptrm_bet1)/beta1;	A02 = (1-exptrm_alp2)/alpha2;	A03 = (1-exptrm_bet2)/beta2
	A10 = R1 * A00;			A11 = -R1 * A01;		A12 = R2 * A02;			A13 = -R2 * A03; 
	A20 = Q1 * A00;			A21 =  Q1 * A01;		A22 = Q2 * A02;			A23 =  Q2 * A03; 
	A30 = S1 * A00;			A31 = -S1 * A01;		A32 = S2 * A02;			A33 = -S2 * A03; 
	
	def A_int_block(n):

		return (np.array([[A00[:,n], A10[:,n], A20[:,n], A30[:,n]],
					    [A01[:,n], A11[:,n], A21[:,n], A31[:,n]],
			    		    [A02[:,n], A12[:,n], A22[:,n], A32[:,n]],
			    		    [A03[:,n], A13[:,n], A23[:,n], A33[:,n]]])).T

		#block = np.array([[exp_alp1[:,n], (R1*exp_alp1)[:,n], (Q1*exp_alp1)[:,n], (S1*exp_alp1)[:,n]],
		#    [exp_bet1[:,n], -(R1*exp_bet1)[:,n], (Q1*exp_bet1)[:,n], -(S1*exp_bet1)[:,n]],
		#    [exp_alp2[:,n],  (R2*exp_alp2)[:,n], (Q2*exp_alp2)[:,n],  (S2*exp_alp2)[:,n]],
		#    [exp_bet2[:,n], -(R2*exp_bet2)[:,n], (Q2*exp_bet2)[:,n], -(S2*exp_bet2)[:,n]]])

		#return block.T
	
	tau_mu = tau[:,:-1] * (mus - 1/ubar1)
	tau_mu = slice_gt(tau_mu, 35.0)
	exptau_mu = np.exp(-tau_mu)
	exp_mu = (1 - exptrm_mus) * exptau_mu / mus
	N0 = eta[0] * exp_mu;	N1 = eta[1] * exp_mu;	N2 = eta[2] * exp_mu;	N3 = eta[3] * exp_mu;	
	def N_int_block(n):

	    return (np.array([N0[:,n], N1[:,n], N2[:,n], N3[:,n]])).T


	M = np.zeros((nwno, 4*nlayer, 4*nlayer))
	B = np.zeros((nwno, 4*nlayer))

	nlevel = nlayer+1
	F = np.zeros((nwno, 4*nlevel, 4*nlayer))
	G = np.zeros((nwno, 4*nlevel))
	A = np.zeros((nwno, 4*nlevel, 4*nlayer))
	N = np.zeros((nwno, 4*nlevel))
	A_int = np.zeros((nwno, 4*nlayer, 4*nlayer))
	N_int = np.zeros((nwno, 4*nlayer))
	
	#   first two rows: BC 1
	M[:,0:2,0:4] = F_block(0,"down")[:,0:2,]
	B[:,0:2] = b_top - Z_block(0,"down")[:,0:2]        

	#F[:,0:4,0:4] = F_block(0,zero)
	#G[:,0:4] = Z_block(0,zero)

	A[:,0:4,0:4] = A_block(0,"down")
	N[:,0:4] = N_block(0,"down")

	#   rows 3 through 4nlayer-2: BCs 2 and 3
	for n in range(0, nlayer-1):
		im = 4*n+2; iM = (4*n+5)+1
		jm = 4*n; j_ = (4*n+3)+1; jM = (4*n+7)+1
		M[:,im:iM,jm:j_] = F_block(n,"up")
		M[:,im:iM,j_:jM] = -F_block(n+1,"down")
		B[:,im:iM] = Z_block(n+1,"down") - Z_block(n,"up")
		
		im = 4*n+4; iM = (4*n+7)+1
		jm = 4*n; jM = (4*n+3)+1
		#F[:,im:iM,jm:jM] = F_block(n,dtau[:,n])
		#G[:,im:iM] = Z_block(n,dtau[:,n])

		A[:,im:iM,jm:jM] = A_block(n,"up")
		N[:,im:iM] = N_block(n,"up")

		im = 4*n; iM = (4*n+3)+1
		A_int[:,im:iM,jm:jM] = A_int_block(n)
		N_int[:,im:iM] = N_int_block(n)

	#   last two rows: BC 4
	im = 4*nlayer-2; iM = 4*nlayer
	jm = 4*nlayer-4; jM = 4*nlayer
	n = nlayer-1
	M[:,im:iM,jm:jM] = F_block(n,"up")[:,[2,3],] - surf_reflect*F_block(n,"up")[:,[0,1],]
	B[:,im] = b_surface - Z_block(n,"up")[:,2] + surf_reflect * Z_block(n,"up")[:,0]  
	B[:,iM-1] = b_surface - Z_block(n,"up")[:,3] + surf_reflect * Z_block(n,"up")[:,1]  
	## should have b_surface in here but it's zero
	
	im = 4*nlevel-4; iM = 4*nlevel
	jm = 4*nlayer-4; jM = 4*nlayer
	#F[:,im:iM,jm:jM] = F_block(n,dtau[:,n])
	#G[:,im:iM] = Z_block(n,dtau[:,n])

	A[:,im:iM,jm:jM] = A_block(n,"up")
	N[:,im:iM] = N_block(n,"up")

	im = 4*nlayer-4; iM = 4*nlayer
	jm = 4*nlayer-4; jM = 4*nlayer
	A_int[:,im:iM,jm:jM] = A_int_block(n)
	N_int[:,im:iM] = N_int_block(n)

	return M, B, A, N, F, G, A_int, N_int

def setup_4_stream(nlayer, nwno, W0, b_top, b_surface, surf_reflect, F0PI, ubar0, dtau,tau, w_single, w_multi, ubar1, P):
	"""
	Parameters
	----------
	nlayer : int 
		number of layers in the model 
	nwno : int 
		number of wavelength points ## need to include this
	W0: int
		single scattering albedo 
	b_top : array 
		The diffuse radiation into the model at the top of the atmosphere
	b_surface : array
		The diffuse radiation into the model at the bottom. Includes emission, reflection 
		of the unattenuated portion of the direct beam  
	surf_reflect : array 
		Surface reflectivity 
	F0PI : int  
		solar radiation
	ubar0: array
		cosine of solar incident angle
	dtau : array 
		Opacity per layer
	g : array 
		asymmetry parameters
	P : array
		Legendre polynomials
	"""
	w0 = W0.T
	dtau = dtau.T
	tau = tau.T
	tauN = tau[0,-1]
	tau = (tau.T/tauN).T
	dtau = (dtau.T/tauN).T
	
	a = []; b = []
	for l in range(4):
		a.append((tauN * ((2*l + 1) - w0 * w_multi[l]).T).T)
		b.append((tauN * F0PI * (w0 * w_single[l]).T).T * P(-ubar0)[l] / (4 * np.pi))

	beta = a[0]*a[1] + 4*a[0]*a[3]/9 + a[2]*a[3]/9
	gama = a[0]*a[1]*a[2]*a[3]/9
	lam1 = np.sqrt((beta + np.sqrt(beta**2 - 4*gama)) / 2)
	lam2 = np.sqrt((beta - np.sqrt(beta**2 - 4*gama)) / 2)

	ubar0 = ubar0/tauN

	def f(x):
		return x**4 - beta*x**2 + gama
	
	Del = 9 * f(1/ubar0)
	Dels = []
	Dels.append((a[1]*b[0] - b[1]/ubar0) * (a[2]*a[3] - 9/ubar0**2) 
		+ 2*(a[3]*b[2] - 2*a[3]*b[0] - 3*b[3]/ubar0)/ubar0**2)
	Dels.append((a[0]*b[1] - b[0]/ubar0) * (a[2]*a[3] - 9/ubar0**2) 
		- 2*a[0]*(a[3]*b[2] - 3*b[3]/ubar0)/ubar0)
	Dels.append((a[3]*b[2] - 3*b[3]/ubar0) * (a[0]*a[1] - 1/ubar0**2) 
		- 2*a[3]*(a[0]*b[1] - b[0]/ubar0)/ubar0)
	Dels.append((a[2]*b[3] - 3*b[2]/ubar0) * (a[0]*a[1] - 1/ubar0**2) 
		+ 2*(3*a[0]*b[1] - 2*a[0]*b[3] - 3*b[0]/ubar0)/ubar0**2)
	
	eta = []
	for l in range(4):
		eta.append(Dels[l]/Del)


	expo1_up = slice_gt(lam1*tau[:,1:], 35.0) 
	expo1_down = slice_gt(lam1*tau[:,:-1], 35.0) 
	expo2_up = slice_gt(lam2*tau[:,1:], 35.0) 
	expo2_down = slice_gt(lam2*tau[:,:-1], 35.0) 
	exptrm1_up = np.exp(-expo1_up)
	exptrm1_down = np.exp(-expo1_down)
	exptrm2_up = np.exp(-expo2_up)
	exptrm2_down = np.exp(-expo2_down)

	exptau_up = np.exp(-tau[:,1:] / ubar0)
	exptau_down = np.exp(-tau[:,:-1] / ubar0)

	R1 = -a[0]/lam1; R2 = -a[0]/lam2
	Q1 = 1/2 * (a[0]*a[1]/(lam1**2) - 1); Q2 = 1/2 * (a[0]*a[1]/(lam2**2) - 1)
	S1 = -3/(2*a[3]) * (a[0]*a[1]/lam1 - lam1); S2 = -3/(2*a[3]) * (a[0]*a[1]/lam2 - lam2)
	
	p1pl = 2*np.pi*(1/2 + R1 + 5*Q1/8); p1mn = 2*np.pi*(1/2 - R1 + 5*Q1/8);
	p2pl = 2*np.pi*(1/2 + R2 + 5*Q2/8); p2mn = 2*np.pi*(1/2 - R2 + 5*Q2/8);
	q1pl = 2*np.pi*(-1/8 + 5*Q1/8 + S1); q1mn = 2*np.pi*(-1/8 + 5*Q1/8 - S1)
	q2pl = 2*np.pi*(-1/8 + 5*Q2/8 + S2); q2mn = 2*np.pi*(-1/8 + 5*Q2/8 - S2)
	z1pl = 2*np.pi*(eta[0]/2 + eta[1] + 5*eta[2]/8); z1mn = 2*np.pi*(eta[0]/2 - eta[1] + 5*eta[2]/8);
	z2pl = 2*np.pi*(-eta[0]/8 + 5*eta[2]/8 + eta[3]); z2mn = 2*np.pi*(-eta[0]/8 + 5*eta[2]/8 - eta[3]);
	
	zero = np.zeros(nwno)

	def F_block(n, lay):
		if lay=="up":
			e1mn = exptrm1_up[:,n]; e1pl = 1/e1mn
			e2mn = exptrm2_up[:,n]; e2pl = 1/e2mn
		elif lay=="down":
			e1mn = exptrm1_down[:,n]; e1pl = 1/e1mn
			e2mn = exptrm2_down[:,n]; e2pl = 1/e2mn
		

		block = np.array([[p1mn[:,n]*e1mn, q1mn[:,n]*e1mn, p1pl[:,n]*e1mn, q1pl[:,n]*e1mn],
					[p1pl[:,n]*e1pl, q1pl[:,n]*e1pl, p1mn[:,n]*e1pl, q1mn[:,n]*e1pl],
					[p2mn[:,n]*e2mn, q2mn[:,n]*e2mn, p2pl[:,n]*e2mn, q2pl[:,n]*e2mn],
					[p2pl[:,n]*e2pl, q2pl[:,n]*e2pl, p2mn[:,n]*e2pl, q2mn[:,n]*e2pl]])
		return block.T

	def Z_block(n, lay):
		if lay=="up":
			exptau = exptau_up[:,n]
		elif lay=="down":
			exptau = exptau_down[:,n]

		return (np.array([z1mn[:,n], z2mn[:,n], z1pl[:,n], z2pl[:,n]]) * exptau).T
	
	def A_block(n, lay):
		if lay=="up":
			e1mn = exptrm1_up[:,n]; e1pl = 1/e1mn
			e2mn = exptrm2_up[:,n]; e2pl = 1/e2mn
		elif lay=="down":
			e1mn = exptrm1_down[:,n]; e1pl = 1/e1mn
			e2mn = exptrm2_down[:,n]; e2pl = 1/e2mn

		block = np.array([[e1mn, R1[:,n]*e1mn, Q1[:,n]*e1mn, S1[:,n]*e1mn],
			[e1pl, -R1[:,n]*e1pl, Q1[:,n]*e1pl, -S1[:,n]*e1pl],
			[e2mn,  R2[:,n]*e2mn, Q2[:,n]*e2mn,  S2[:,n]*e2mn],
			[e2pl, -R2[:,n]*e2pl, Q2[:,n]*e2pl, -S2[:,n]*e2pl]])

		return block.T
	
	def N_block(n, lay):
		if lay=="up":
			exptau = exptau_up[:,n]
		elif lay=="down":
			exptau = exptau_down[:,n]
		
		return (np.array([eta[0][:,n], eta[1][:,n], eta[2][:,n], eta[3][:,n]]) * exptau).T

	ubar0 = ubar0*tauN
	alpha1 = (tauN/ubar1 + lam1);		beta1 = (tauN/ubar1 - lam1);		mus = tauN * (ubar1 + ubar0) / (ubar1 * ubar0)
	alpha2 = (tauN/ubar1 + lam2);		beta2 = (tauN/ubar1 - lam2);
	expo_alp1 = alpha1 * dtau;		expo_bet1 = beta1 * dtau;		expo_mus = mus * dtau 
	expo_alp2 = alpha2 * dtau;		expo_bet2 = beta2 * dtau;		
	expo_alp1 = slice_gt(expo_alp1, 35.0);	expo_bet1 = slice_gt(expo_bet1, 35.0);	expo_mus = slice_gt(expo_mus, 35.0)    
	expo_alp2 = slice_gt(expo_alp2, 35.0);	expo_bet2 = slice_gt(expo_bet2, 35.0);	
	exptrm_alp1 = np.exp(-expo_alp1);	exptrm_bet1 = np.exp(-expo_bet1);	exptrm_mus = np.exp(-expo_mus)
	exptrm_alp2 = np.exp(-expo_alp2);	exptrm_bet2 = np.exp(-expo_bet2);

	def A_int_block(n):
		exptau_alp1 = np.exp(-tau[:,n] * alpha1[:,n])
		exptau_alp2 = np.exp(-tau[:,n] * alpha2[:,n])
		exptau_bet1 = np.exp(-tau[:,n] * beta1[:,n])
		exptau_bet2 = np.exp(-tau[:,n] * beta2[:,n])
		e1mn = (1 - exptrm_alp1[:,n]) * exptau_alp1 / alpha1[:,n]
		e1pl = (1 - exptrm_bet1[:,n]) * exptau_bet1 / beta1[:,n]
		e2mn = (1 - exptrm_alp2[:,n]) * exptau_alp2 / alpha2[:,n]
		e2pl = (1 - exptrm_bet2[:,n]) * exptau_bet2 / beta2[:,n]

		block = np.array([[e1mn, R1[:,n]*e1mn, Q1[:,n]*e1mn, S1[:,n]*e1mn],
			[e1pl, -R1[:,n]*e1pl, Q1[:,n]*e1pl, -S1[:,n]*e1pl],
			[e2mn,  R2[:,n]*e2mn, Q2[:,n]*e2mn,  S2[:,n]*e2mn],
			[e2pl, -R2[:,n]*e2pl, Q2[:,n]*e2pl, -S2[:,n]*e2pl]])

		return block.T
	
	def N_int_block(n):
		exptau = np.exp(-tau[:,n] * mus)
		e1 = (1 - exptrm_mus[:,n]) * exptau / mus

		return (np.array([eta[0][:,n], eta[1][:,n], eta[2][:,n], eta[3][:,n]]) * e1).T

	M = np.zeros((nwno, 4*nlayer, 4*nlayer))
	B = np.zeros((nwno, 4*nlayer))

	nlevel = nlayer+1
	F = np.zeros((nwno, 4*nlevel, 4*nlayer))
	G = np.zeros((nwno, 4*nlevel))
	A = np.zeros((nwno, 4*nlevel, 4*nlayer))
	N = np.zeros((nwno, 4*nlevel))
	A_int = np.zeros((nwno, 4*nlayer, 4*nlayer))
	N_int = np.zeros((nwno, 4*nlayer))
	
	#   first two rows: BC 1
	M[:,0:2,0:4] = F_block(0,"down")[:,0:2,]
	B[:,0:2] = b_top - Z_block(0,"down")[:,0:2]        

	F[:,0:4,0:4] = F_block(0,"down")
	G[:,0:4] = Z_block(0,"down")

	A[:,0:4,0:4] = A_block(0,"down")
	N[:,0:4] = N_block(0,"down")

	#   rows 3 through 4nlayer-2: BCs 2 and 3
	for n in range(0, nlayer-1):
		im = 4*n+2; iM = (4*n+5)+1
		jm = 4*n; j_ = (4*n+3)+1; jM = (4*n+7)+1
		M[:,im:iM,jm:j_] = F_block(n,"up")
		M[:,im:iM,j_:jM] = -F_block(n+1,"down")
		B[:,im:iM] = Z_block(n+1,"down") - Z_block(n,"up")
		
		im = 4*n+4; iM = (4*n+7)+1
		jm = 4*n; jM = (4*n+3)+1
		F[:,im:iM,jm:jM] = F_block(n,"up")
		G[:,im:iM] = Z_block(n,"up")

		A[:,im:iM,jm:jM] = A_block(n,"up")
		N[:,im:iM] = N_block(n,"up")

		im = 4*n; iM = (4*n+3)+1
		A_int[:,im:iM,jm:jM] = A_int_block(n)
		N_int[:,im:iM] = N_int_block(n)


	#   last two rows: BC 4
	im = 4*nlayer-2; iM = 4*nlayer
	jm = 4*nlayer-4; jM = 4*nlayer
	n = nlayer-1
	M[:,im:iM,jm:jM] = F_block(n,"up")[:,[2,3],] - surf_reflect*F_block(n,"up")[:,[0,1],]
	B[:,im] = b_surface - Z_block(n,"up")[:,2] + surf_reflect * Z_block(n,"up")[:,0]  
	B[:,iM-1] = b_surface - Z_block(n,"up")[:,3] + surf_reflect * Z_block(n,"up")[:,1]  
	## should have b_surface in here but it's zero
	
	im = 4*nlevel-4; iM = 4*nlevel
	jm = 4*nlayer-4; jM = 4*nlayer
	F[:,im:iM,jm:jM] = F_block(n,"up")
	G[:,im:iM] = Z_block(n,"up")

	A[:,im:iM,jm:jM] = A_block(n,"up")
	N[:,im:iM] = N_block(n,"up")

	im = 4*nlayer-4; iM = 4*nlayer
	jm = 4*nlayer-4; jM = 4*nlayer
	A_int[:,im:iM,jm:jM] = A_int_block(n)

	M_inv = np.linalg.inv(M[0])
	X = M_inv.dot(B[0])
	
	flux = F.dot(X) + G
	
	I = A[0].dot(X) + N[0]

	intgrl = A_int[0].dot(X) + N_int[0]

	return M, B, A, N, F, G, A_int, N_int


def solve_4_stream(M, B, A, N, F, G, A_int, N_int, stream):
	#M_inv = np.linalg.inv(M)
	#X = M_inv.dot(B)
	X = spsolve(M, B)
	
	#flux = F.dot(X) + G
	
	I = A.dot(X) + N
	intgrl_new = A_int.dot(X) + N_int

	return (I, intgrl_new)


#@jit(nopython=True, cache=True)
def get_reflected_new(nlevel, nwno, numg, numt, dtau, tau, w0, cosb, gcos2, ftau_cld, ftau_ray,
	dtau_og, tau_og, w0_og, cosb_og, 
	surf_reflect, ubar0, ubar1, cos_theta, F0PI, single_phase, multi_phase, 
	frac_a, frac_b, frac_c, constant_back, constant_forward, dim, stream, print_time):
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
	
	xint_at_top = zeros((numg, numt, nwno))
	xint_at_top_new = zeros((numg, numt, nwno))
	
	nlayer = nlevel - 1 

	
	#================ START CRAZE LOOP OVER ANGLE #================
	for ng in range(numg):
		for nt in range(numt):
			t1 = time.time()
			if dim == '3d':
				#get needed chunk for 3d inputs
				#should consider thinking of a better method for when doing 1d only
				cosb = cosb[:,:,ng,nt]
				dtau = dtau[:,:,ng,nt]
				tau = tau[:,:,ng,nt]
				w0 = w0[:,:,ng,nt]
				gcos2 = gcos2[:,:,ng,nt]
			
				#uncorrected original values (in case user specified D-Eddington)
				#If they did not, this is the same thing as what is defined above 
				#These are used because HG single scattering phase function does get 
				#the forward and back scattering pretty accurately so delta-eddington
				#is only applied to the multiple scattering terms
				cosb_og = cosb_og[:,:,ng,nt]
				dtau_og = dtau_og[:,:,ng,nt]
				tau_og = tau_og[:,:,ng,nt]
				w0_og = w0_og[:,:,ng,nt]
			

			if single_phase!=1: 
				g_forward = constant_forward*cosb_og
				g_back = constant_back*cosb_og
				f = frac_a + frac_b*g_back**frac_c

			if single_phase==0:#'cahoy':
				import IPython; IPython.embed()
				print('Do not know what cahoy is meant to do')
				import sys; sys.exit()
				g0_s = (ftau_cld + ftau_ray).T
				#g0_s = np.zeros((nwno, nlayer)) #+ 2.
				#g1_s = (3*cosb).T 
				g1_s = (3*ftau_cld*cosb).T 
				g2_s = (5*ftau_cld*(cosb**2)).T + 0.5*ftau_ray.T
				#g3_s = (7*(cosb**3)).T
				g3_s = (7*ftau_cld*(cosb**3)).T

				g0_m = g0_s
				g1_m = g1_s
				g2_m = g2_s
				g3_m = g3_s

			elif single_phase==1:#'OTHG':
				g0_s = np.ones((nwno, nlayer))
				g1_s = 3*cosb.T
				#g2 = gcos2.T
				g2_s = 5*(cosb**2).T
				#g3 = gcos2.T/10 # moments of phase function ** need to define 4th moment
				g3_s = 7*(cosb**3).T

				g0_m = g0_s
				g1_m = g1_s
				g2_m = g2_s
				g3_m = g3_s

				p_single=(1-cosb_og**2)/sqrt((1+cosb_og**2+2*cosb_og*cos_theta)**3) 

			elif single_phase==2:#'TTHG':
				#Phase function for single scattering albedo frum Solar beam
				#uses the Two term Henyey-Greenstein function with the additiona rayleigh component 
					  #first term of TTHG: forward scattering
				g0_s = np.ones((nwno, nlayer))
				g1_s = 3*(f*g_forward + (1-f)*g_back).T
				g2_s = 5*(f*g_forward**2 + (1-f)*g_back**2).T
				g3_s = 7*(f*g_forward**3 + (1-f)*g_back**3).T

				g0_m = np.ones((nwno, nlayer))
				g1_m = 3*cosb.T
				g2_m = 5*(cosb**2).T
				g3_m = 7*(cosb**3).T


				p_single=(f * (1-g_forward**2) /sqrt((1+g_forward**2+2*g_forward*cos_theta)**3) 
								#second term of TTHG: backward scattering
								+(1-f)*(1-g_back**2)
								/sqrt((1+g_back**2+2*g_back*cos_theta)**3))

			elif single_phase==3:#'TTHG_ray':
				#Phase function for single scattering albedo frum Solar beam
				#uses the Two term Henyey-Greenstein function with the additiona rayleigh component 
					  		#first term of TTHG: forward scattering

				g0_s = (ftau_cld + ftau_ray).T
				g1_s = (3*ftau_cld*(f*g_forward + (1-f)*g_back)).T 
				g2_s = (5*ftau_cld*(f*g_forward**2 + (1-f)*g_back**2)).T + 0.5*ftau_ray.T 
				g3_s = (7*ftau_cld*(f*g_forward**3 + (1-f)*g_back**3)).T

				g0_m = np.ones((nwno, nlayer))
				g1_m = 3*(ftau_cld*cosb).T
				g2_m = 5*(ftau_cld*cosb**2).T + 0.5*ftau_ray.T
				g3_m = 7*(ftau_cld*cosb**3).T

				p_single=(ftau_cld*(f * (1-g_forward**2)
							/sqrt((1+g_forward**2+2*g_forward*cos_theta)**3) 
							#second term of TTHG: backward scattering
							+(1-f)*(1-g_back**2)
							/sqrt((1+g_back**2+2*g_back*cos_theta)**3))+
								#rayleigh phase function
								ftau_ray*(0.75*(1+cos_theta**2.0)))
			
			w_single = [g0_s, g1_s, g2_s, g3_s]
			w_multi = [g0_m, g1_m, g2_m, g3_m]
			
			def P(mu): # Legendre polynomials
				return [1, mu, 1/2 * (3*mu**2 - 1), 1/2 * (5*mu**3 - 3*mu)]

			#boundary conditions 
			b_top = 0.0
			b_surface = 0. + surf_reflect*ubar0[ng, nt]*F0PI*exp(-tau[-1, :]/ubar0[ng, nt])

			if stream==2:
				M, B, A, N, F, G, A_int, N_int = setup_2_stream_scaled(nlayer, nwno, w0, b_top, b_surface, surf_reflect, F0PI, 
						ubar0[ng, nt], dtau, tau, w_single, w_multi, ubar1[ng,nt], P)
			else:
				M, B, A, N, F, G, A_int, N_int = setup_4_stream_scaled(nlayer, nwno, w0, b_top, b_surface, surf_reflect, F0PI, 
						ubar0[ng, nt], dtau, tau, w_single, w_multi, ubar1[ng,nt], P)

			t2 = time.time()
			t_setup = t2-t1
			if print_time:
				print("time to set up matrices = ", t_setup)

			t3 = time.time()

			X = zeros((stream*nlevel, nwno))
			intgrl_new = np.zeros((stream*nlayer, nwno))
			intgrl_per_layer = np.zeros((nlayer, nwno))
			full_intgrl = np.zeros((nlayer, nwno))
			sing_scat = np.zeros((nlayer, nwno))
			xint_temp = zeros((nlevel, nwno))
			xint_new = zeros(nwno)
			#========================= Start loop over wavelength =========================
			for W in range(nwno):
			    
				(X[:,W], intgrl_new[:,W]) = solve_4_stream(M[W], B[W], A[W], N[W], F[W], G[W], A_int[W], N_int[W], stream)

			t4 = time.time()
			t_invert = t4-t3
			if print_time:
				print("time to invert matrices = ", t_invert)
			t5 = time.time()

			mus = (ubar1[ng,nt] + ubar0[ng,nt]) / (ubar1[ng,nt] * ubar0[ng,nt])
			expo_mus = mus * dtau 
			expo_mus = slice_gt(expo_mus, 35.0)    
			exptrm_mus = np.exp(-expo_mus)

			for i in range(nlayer):
				for l in range(stream):
					sing_scat[i,:] = sing_scat[i,:] + w_multi[l][:,i] * P(ubar1[ng,nt])[l] * intgrl_new[stream*i+l,:]

			intgrl_per_layer = w0 * ( sing_scat 
						+ F0PI / (4*np.pi) * p_single 
						* np.exp(-tau[:-1,:]*(mus-1/ubar1[ng,nt])) * (1 - exptrm_mus) / mus)

			for i in range(nlayer):
				j = nlayer-i
				full_intgrl[i] = np.sum(intgrl_per_layer[-j:,:])

			for l in range(stream):
				xint_temp[-1, :] = xint_temp[-1, :] + (2*l+1) * X[stream*(nlevel-1)+l, :] * P(ubar1[ng,nt])[l]
			for i in range(nlayer-1,-1,-1):
				xint_temp[i, :] = (xint_temp[i+1, :] * np.exp(-dtau[i,:]/ubar1[ng,nt]) 
							+ intgrl_per_layer[i,:] / ubar1[ng,nt]) 

			xint_new = xint_temp[0, :]
			xint_at_top[ng,nt,:] = xint_new
			t6 = time.time()
			t_sum = t6-t5
			if print_time:
				print("time to sum over layers = ", t_sum)
	filename = 'xint_at_top_%d.pk' % stream
	pk.dump(xint_at_top, open(filename,'wb'))
	#import IPython; IPython.embed()
	#import sys; sys.exit()

	return xint_at_top

def setup_2_stream(nlayer, nwno, W0, b_top, b_surface, surf_reflect, F0PI, ubar0, dtau, tau, w_single, w_multi, ubar1, P):
	"""
	Parameters
	----------
	nlayer : int 
		number of layers in the model 
	nwno : int 
		number of wavelength points ## need to include this
	W0: int
		single scattering albedo 
	b_top : array 
		The diffuse radiation into the model at the top of the atmosphere
	b_surface : arra
		The diffuse radiation into the model at the bottom. Includes emission, reflection 
		of the unattenuated portion of the direct beam  
	surf_reflect : array 
		Surface reflectivity 
	F0PI : int  
		solar radiation
	ubar0: array
		cosine of solar incident angle
	dtau : array 
		Opacity per layer
	g : array 
		asymmety parameters
	P : array
		Legendre polynomials
	"""

	w0 = W0.T
	dtau = dtau.T
	tau = tau.T
	tauN = tau[0,-1]
	tau = (tau.T/tauN).T
	dtau = (dtau.T/tauN).T
	
	a = []; b = []
	for l in range(2):
		a.append((tauN * ((2*l + 1) - w0 * w_multi[l]).T).T)
		b.append((tauN * F0PI * (w0 * w_single[l]).T).T * P(-ubar0)[l] / (4 * np.pi))

	lam = np.sqrt(a[0]*a[1])

	Del = ((tauN / ubar0)**2 - a[0]*a[1])
	Dels = []
	Dels.append(b[1] * tauN /ubar0 - a[1]*b[0])
	Dels.append(b[0] * tauN /ubar0 - a[0]*b[1])
	
	eta = []
	for l in range(2):
		eta.append(Dels[l]/Del)

	#expo = lam*dtau
	expo_up = lam*tau[:,1:]
	expo_down = lam*tau[:,:-1]
	#save from overflow 
	expo_up = slice_gt(expo_up, 35.0) 
	expo_down = slice_gt(expo_down, 35.0) 
	exptrm_up = np.exp(-expo_up)
	exptrm_down = np.exp(-expo_down)

	expt_up = tau[:,1:] * tauN / ubar0
	expt_down = tau[:,:-1] * tauN / ubar0
	expt_up = slice_gt(expt_up,35.0)
	expt_down = slice_gt(expt_down,35.0)
	exptau_up = np.exp(-expt_up)
	exptau_down = np.exp(-expt_down)

	q = lam/a[1]
	Q1 = 0.5 + q
	Q2 = 0.5 - q
	#e1 = np.exp(exptrm)

	def F_block(n, e1):
		#if np.any(t!=0):
		#	e1 = exptrm[:,n]; e2 = 1/e1
		#else:
		#	e1 = np.ones(nwno); e2 = e1
		e2 = 1/e1

		block = 2*np.pi * np.array([[Q1[:,n]*e1, Q2[:,n]*e1], [Q2[:,n]*e2, Q1[:,n]*e2]])
		return block.T

	def Z_block(n, exptau):
		#if np.any(t!=0):
		#	exptau = np.exp(-tau[:,n+1] * tauN /ubar0)
		#else:
		#	exptau = np.exp(-tau[:,n] * tauN /ubar0)
		
		return 2*np.pi * (np.array([(0.5 * eta[0] - eta[1])[:,n], 
			(0.5 * eta[0] + eta[1])[:,n]]) * exptau).T
	
	def A_block(n, e1):
		#if np.any(t!=0):
		#	e1 = exptrm[:,n]; e2 = 1/e1
		#else:
		#	e1 = np.ones(nwno); e2 = e1
		e2 = 1/e1

		block =  np.array([[e1, -q[:,n]*e1], [e2, q[:,n] * e2]])
		return block.T
	
	def N_block(n, exptau):
		#if np.any(t!=0):
		#	exptau = np.exp(-tau[:,n+1] * tauN /ubar0)
		#else:
		#	exptau = np.exp(-tau[:,n] * tauN /ubar0)

		return (np.array([eta[0][:,n], eta[1][:,n]]) * exptau).T

	alpha = (tauN/ubar1 + lam);		beta = (tauN/ubar1 - lam);		mus = tauN * (ubar1 + ubar0) / (ubar1 * ubar0)
	expo_alp = alpha * dtau;		expo_bet = beta * dtau;			expo_mus = mus * dtau 
	expo_alp = slice_gt(expo_alp, 35.0);	expo_bet = slice_gt(expo_bet, 35.0);	expo_mus = slice_gt(expo_mus, 35.0)    
	exptrm_alp = np.exp(-expo_alp);		exptrm_bet = np.exp(-expo_bet);		exptrm_mus = np.exp(-expo_mus)

	def A_int_block(n):
		exptau_alp = np.exp(-slice_gt(tau[:,:-1] * alpha, 35.0))
		exptau_bet = np.exp(-slice_gt(tau[:,:-1] * beta, 35.0))
		e1 = (1 - exptrm_alp[:,n]) * exptau_alp[:,n] / alpha[:,n]
		e2 = (1 - exptrm_bet[:,n]) * exptau_bet[:,n] / beta[:,n]

		block =  np.array([[e1, -q[:,n] * e1], 
				    [e2, q[:,n] * e2]])
		return block.T

	def N_int_block(n):
		exptau = np.exp(-slice_gt(tau * mus, 35.0))
		e1 = (1 - exptrm_mus[:,n]) * exptau[:,n] / mus

		return (np.array([eta[0][:,n], eta[1][:,n]]) * e1 ).T

	
	M = np.zeros((nwno, 2*nlayer, 2*nlayer))
	B = np.zeros((nwno, 2*nlayer))

	nlevel = nlayer+1
	F = np.zeros((nwno, 2*nlevel, 2*nlayer))
	G = np.zeros((nwno, 2*nlevel))
	A = np.zeros((nwno, 2*nlevel, 2*nlayer))
	N = np.zeros((nwno, 2*nlevel))

	A_int = np.zeros((nwno, 2*nlayer, 2*nlayer))
	N_int = np.zeros((nwno, 2*nlayer))

	zero = np.zeros(nwno)
	
	#   first row: BC 1
	M[:,0,0:2] = F_block(0,exptrm_down[:,0])[:,0,]
	B[:,0] = b_top - Z_block(0,exptau_down[:,0])[:,0]        

	F[:,0:2,0:2] = F_block(0,exptrm_down[:,0])
	G[:,0:2] = Z_block(0,exptau_down[:,0])

	A[:,0:2,0:2] = A_block(0,exptrm_down[:,0])
	N[:,0:2] = N_block(0,exptau_down[:,0])

	#   rows 1 through 2*nlayer-1: BCs 2 and 3
	for n in range(0, nlayer-1):
		im = 2*n+1; iM = (2*n+2)+1
		jm = 2*n; j_ = (2*n+1)+1; jM = (2*n+3)+1
		M[:,im:iM,jm:j_] = F_block(n,exptrm_up[:,n])
		M[:,im:iM,j_:jM] = -F_block(n+1,exptrm_down[:,n+1])
		B[:,im:iM] = Z_block(n+1,exptau_down[:,n+1]) - Z_block(n,exptau_up[:,n])
		
		im = 2*n+2; iM = (2*n+3)+1
		jm = 2*n; jM = (2*n+1)+1
		F[:,im:iM,jm:jM] = F_block(n,exptrm_up[:,n])
		G[:,im:iM] = Z_block(n,exptau_up[:,n])

		#im = 2*n; iM = (2*n+1)+1
		A[:,im:iM,jm:jM] = A_block(n,exptrm_up[:,n])
		N[:,im:iM] = N_block(n,exptau_up[:,n])

		im = 2*n; iM = (2*n+1)+1
		A_int[:,im:iM,jm:jM] = A_int_block(n)
		N_int[:,im:iM] = N_int_block(n)

	#   last row: BC 4
	im = 2*nlayer-1; 
	jm = 2*nlayer-2; jM = 2*nlayer
	n = nlayer-1
	M[:,im,jm:jM] = F_block(n,exptrm_up[:,n])[:,1,] - surf_reflect*F_block(n,exptrm_up[:,n])[:,0,]
	B[:,im] = b_surface - Z_block(n,exptau_up[:,n])[:,1] + surf_reflect * Z_block(n,exptau_up[:,n])[:,0]
	

	n = nlayer-1
	im = 2*nlevel-2; iM = 2*nlevel
	jm = 2*nlayer-2; jM = 2*nlayer
	F[:,im:iM,jm:jM] = F_block(n,exptrm_up[:,n])
	G[:,im:iM] = Z_block(n,exptau_up[:,n])

	A[:,im:iM,jm:jM] = A_block(n,exptrm_up[:,n])
	N[:,im:iM] = N_block(n,exptau_up[:,n])

	im = 2*nlayer-2; iM = 2*nlayer
	jm = 2*nlayer-2; jM = 2*nlayer
	A_int[:,im:iM,jm:jM] = A_int_block(n)
	N_int[:,im:iM] = N_int_block(n)

	A0 = A_block(0,zero)
	N0 = N_block(0,zero)

	M_inv = np.linalg.inv(M[0])
	X = M_inv.dot(B[0])
	
	flux = F.dot(X) + G
	
	I = A.dot(X) + N

	return M, B, A, N, F, G, A_int, N_int

def setup_2_stream_scaled(nlayer, nwno, W0, b_top, b_surface, surf_reflect, F0PI, ubar0, dtau, tau, w_single, w_multi, ubar1, P):
	"""
	Parameters
	----------
	nlayer : int 
		number of layers in the model 
	nwno : int 
		number of wavelength points ## need to include this
	W0: int
		single scattering albedo 
	b_top : array 
		The diffuse radiation into the model at the top of the atmosphere
	b_surface : arra
		The diffuse radiation into the model at the bottom. Includes emission, reflection 
		of the unattenuated portion of the direct beam  
	surf_reflect : array 
		Surface reflectivity 
	F0PI : int  
		solar radiation
	ubar0: array
		cosine of solar incident angle
	dtau : array 
		Opacity per layer
	g : array 
		asymmety parameters
	P : array
		Legendre polynomials
	"""

	w0 = W0.T
	dtau = dtau.T
	tau = tau.T
	
	a = []; b = []
	for l in range(2):
		a.append((((2*l + 1) - w0 * w_multi[l]).T).T)
		b.append((F0PI * (w0 * w_single[l]).T).T * P(-ubar0)[l] / (4 * np.pi))

	lam = np.sqrt(a[0]*a[1])

	Del = ((1 / ubar0)**2 - a[0]*a[1])
	Dels = []
	Dels.append(b[1] /ubar0 - a[1]*b[0])
	Dels.append(b[0] /ubar0 - a[0]*b[1])
	
	eta = []
	for l in range(2):
		eta.append(Dels[l]/Del)

	expo = lam*dtau
	#save from overflow 
	expo = slice_gt(expo, 35.0) 
	exptrm = np.exp(-expo)

	q = lam/a[1]
	Q1 = 0.5 + q
	Q2 = 0.5 - q

	Q1mn = Q1*exptrm;  Q2mn = Q2*exptrm
	Q1pl = Q1/exptrm;  Q2pl = Q2/exptrm
	def F_block(n, lev):
		if lev == "up":
		    return 2*np.pi * np.array([[Q1mn[:,n], Q2mn[:,n]], [Q2pl[:,n], Q1pl[:,n]]]).T
		elif lev == "down":
		    return 2*np.pi * np.array([[Q1[:,n], Q2[:,n]], [Q2[:,n], Q1[:,n]]]).T

	exptau_u0 = np.exp(-tau/ubar0)
	zmn = 0.5*eta[0] - eta[1] 
	zpl = 0.5*eta[0] + eta[1]
	zmn_up = zmn * exptau_u0[:,1:] 
	zpl_up = zpl * exptau_u0[:,1:] 
	zmn_down = zmn * exptau_u0[:,:-1] 
	zpl_down = zpl * exptau_u0[:,:-1] 
	def Z_block(n, lev):
		if lev == "up":
			return 2*np.pi * (np.array([zmn_up[:,n], zpl_up[:,n]])).T
		elif lev == "down":
			return 2*np.pi * (np.array([zmn_down[:,n], zpl_down[:,n]])).T
	
	exptrmpl = 1/exptrm
	qmn = -q*exptrm; qpl = q/exptrm
	def A_block(n):
		return np.array([[exptrm[:,n], qmn[:,n]], [exptrmpl[:,n], qpl[:,n]]]).T
	
	n0_up = eta[0] * exptau_u0[:,1:]
	n1_up = eta[1] * exptau_u0[:,1:]
	def N_block(n):
		return (np.array([n0_up[:,n], n1_up[:,n]])).T

	alpha = 1/ubar1 + lam;			beta = 1/ubar1 - lam;			mus = (ubar1 + ubar0) / (ubar1 * ubar0)
	expo_alp = alpha * dtau;		expo_bet = beta * dtau;			expo_mus = mus * dtau 
	expo_alp = slice_gt(expo_alp, 35.0);	expo_bet = slice_gt(expo_bet, 35.0);	expo_mus = slice_gt(expo_mus, 35.0)    
	exptrm_alp = np.exp(-expo_alp);		exptrm_bet = np.exp(-expo_bet);		exptrm_mus = np.exp(-expo_mus)

	exp_alp = (1 - exptrm_alp) / alpha 
	exp_bet = (1 - exptrm_bet) / beta  
	qexp_alp = -q * exp_alp	
	qexp_bet = q * exp_bet
	def A_int_block(n):

		block =  np.array([[exp_alp[:,n], qexp_alp[:,n]], 
				    [exp_bet[:,n], qexp_bet[:,n]]])
		return block.T

	tau_mu = tau[:,:-1] * (mus - 1/ubar1)
	tau_mu = slice_gt(tau_mu, 35.0)
	exptau_mu = np.exp(-tau_mu)
	exp_mu = (1 - exptrm_mus) * exptau_mu / mus
	nint0 = eta[0] * exp_mu
	nint1 = eta[1] * exp_mu
	def N_int_block(n):

		return (np.array([nint0[:,n], nint1[:,n]])).T
	
	M = np.zeros((nwno, 2*nlayer, 2*nlayer))
	B = np.zeros((nwno, 2*nlayer))

	nlevel = nlayer+1
	F = np.zeros((nwno, 2*nlevel, 2*nlayer))
	G = np.zeros((nwno, 2*nlevel))
	A = np.zeros((nwno, 2*nlevel, 2*nlayer))
	N = np.zeros((nwno, 2*nlevel))
	A_int = np.zeros((nwno, 2*nlayer, 2*nlayer))
	N_int = np.zeros((nwno, 2*nlayer))

	zero = np.zeros(nwno)
	
	#   first row: BC 1
	M[:,0,0:2] = F_block(0,"down")[:,0,]
	B[:,0] = b_top - Z_block(0,"down")[:,0]        

	#F[:,0:2,0:2] = F_block(0,zero)
	#G[:,0:2] = Z_block(0,zero)

	#   rows 1 through 2*nlayer-1: BCs 2 and 3
	#   need to get rid of loop to speed things up
	for n in range(0, nlayer-1):
		im = 2*n+1; iM = (2*n+2)+1
		jm = 2*n; j_ = (2*n+1)+1; jM = (2*n+3)+1
		M[:,im:iM,jm:j_] = F_block(n,"up")
		M[:,im:iM,j_:jM] = -F_block(n+1,"down")
		B[:,im:iM] = Z_block(n+1,"down") - Z_block(n,"up")
		
		im = 2*n+2; iM = (2*n+3)+1
		jm = 2*n; jM = (2*n+1)+1
		#F[:,im:iM,jm:jM] = F_block(n,dtau[:,n])
		#G[:,im:iM] = Z_block(n,dtau[:,n])

		A[:,im:iM,jm:jM] = A_block(n)
		N[:,im:iM] = N_block(n)

		im = 2*n; iM = (2*n+1)+1
		A_int[:,im:iM,jm:jM] = A_int_block(n)
		N_int[:,im:iM] = N_int_block(n)

	#   last row: BC 4
	im = 2*nlayer-1; 
	jm = 2*nlayer-2; jM = 2*nlayer
	n = nlayer-1
	M[:,im,jm:jM] = F_block(n,"up")[:,1,] - surf_reflect*F_block(n,"up")[:,0,]
	B[:,im] = b_surface - Z_block(n,"up")[:,1] + surf_reflect * Z_block(n,"up")[:,0]
	

	n = nlayer-1
	im = 2*nlevel-2; iM = 2*nlevel
	jm = 2*nlayer-2; jM = 2*nlayer
	#F[:,im:iM,jm:jM] = F_block(n,dtau[:,n])
	#G[:,im:iM] = Z_block(n,dtau[:,n])

	A[:,im:iM,jm:jM] = A_block(n)
	N[:,im:iM] = N_block(n)

	im = 2*nlayer-2; iM = 2*nlayer
	jm = 2*nlayer-2; jM = 2*nlayer
	A_int[:,im:iM,jm:jM] = A_int_block(n)
	N_int[:,im:iM] = N_int_block(n)

	return M, B, A, N, F, G, A_int, N_int
