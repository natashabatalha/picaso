from numba import jit, objmode
from numpy import exp, zeros, where, sqrt, cumsum , pi, outer, sinh, cosh, min, dot, array,log, log10,ones, array_equal
import numpy as np
import time
import pickle as pk
from scipy.linalg import solve_banded
#from numpy.linalg import solve

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
def slice_lt_cond(array, cond_array, cond, newval):
    """Funciton to replace values with upper or lower limit
    """
    for i in range(array.shape[0]):
        new = array[i,:] 
        new_cond = cond_array[i,:]
        new[where(new_cond<cond)] = newval
        array[i,:] = new     
    return array


@jit(nopython=True, cache=True)
def slice_lt_cond_arr(array, cond_array, cond, newarray):
    """Funciton to replace values with upper or lower limit
    """
    shape = cond_array.shape#e.g. dtau

    cond_array=cond_array.ravel()
    new = array.ravel() #e.g. b0 
    newarray1 = newarray[0:-1,:].ravel()
    newarray2 = newarray[1:,:].ravel()

    #for i in range(array.shape[0]):
    replace1 = newarray1[where(cond_array<cond)]
    replace2 = newarray2[where(cond_array<cond)]
    new[where(cond_array<cond)] = 0.5*(replace1+replace2)
    array = new.reshape(shape)    
    return array

@jit(nopython=True, cache=True)
def slice_rav(array, lim):
    """Funciton to replace values with upper or lower limit
    """
    shape = array.shape
    new = array.ravel()
    new[where(new>lim)] = lim
    new[where(new<-lim)] = -lim
    return new.reshape(shape)

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

@jit(nopython=True, cache=True)
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

@jit(nopython=True, cache=True)
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
    frac_a, frac_b, frac_c, constant_back, constant_forward):
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

            g1  = (sq3*0.5)*(2. - w0*(1.+ftau_cld*cosb)) #table 1
            g2  = (sq3*w0*0.5)*(1.-ftau_cld*cosb)           #table 1
            lamda = sqrt(g1**2 - g2**2)           #eqn 21
            gama  = (g1-lamda)/g2                   #eqn 22
            g3  = 0.5*(1.-sq3*ftau_cld*cosb*ubar0[ng, nt])   #table 1 #ubar is now 100x 10 matrix.. 
    
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
            #if tridiagonal==0:
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
                #if tridiagonal==0:
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
                multi_plus = (1.0+1.5*ftau_cld*cosb*ubar1[ng,nt] #!was 3
                                + gcos2*(3.0*ubar2*ubar2*ubar1[ng,nt]*ubar1[ng,nt] - 1.0)/2.0)
                multi_minus = (1.-1.5*ftau_cld*cosb*ubar1[ng,nt] 
                                + gcos2*(3.0*ubar2*ubar2*ubar1[ng,nt]*ubar1[ng,nt] - 1.0)/2.0)
            elif multi_phase ==1:#'N=1':
                multi_plus = 1.0+1.5*ftau_cld*cosb*ubar1[ng,nt]  
                multi_minus = 1.-1.5*ftau_cld*cosb*ubar1[ng,nt]
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
def get_reflected_1d_deprecate(nlevel, wno,nwno, numg,numt, dtau, tau, w0, cosb,gcos2, ftau_cld, ftau_ray,
    dtau_og, tau_og, w0_og, cosb_og, 
    surf_reflect,ubar0, ubar1,cos_theta, F0PI,single_phase, multi_phase,
    frac_a, frac_b, frac_c, constant_back, constant_forward,
    toon_coefficients=0,b_top=0):
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
    toon_coefficients : int     
        0 for quadrature (default) 1 for eddington

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
    #intensity = zeros((numg, numt, nlevel, nwno))

    nlayer = nlevel - 1 
    flux_out = zeros((numg, numt, 2*nlevel, nwno))

    #now define terms of Toon et al 1989 quadrature Table 1 
    #https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/JD094iD13p16287
    #see table of terms 

    #terms not dependent on incident angle
    sq3 = sqrt(3.)
    if toon_coefficients == 1:#eddington
        g1  = (7-w0*(4+3*ftau_cld*cosb))/4 #(sq3*0.5)*(2. - w0*(1.+cosb)) #table 1 # 
        g2  = -(1-w0*(4-3*ftau_cld*cosb))/4 #(sq3*w0*0.5)*(1.-cosb)        #table 1 # 
    elif toon_coefficients == 0:#quadrature
        g1  = (sq3*0.5)*(2. - w0*(1.+ftau_cld*cosb)) #table 1 # 
        g2  = (sq3*w0*0.5)*(1.-ftau_cld*cosb)        #table 1 # 

    lamda = sqrt(g1**2 - g2**2)         #eqn 21
    gama  = (g1-lamda)/g2               #eqn 22

    #================ START CRAZE LOOP OVER ANGLE #================
    for ng in range(numg):
        for nt in range(numt):
            u1 = ubar1[ng,nt]
            u0 = ubar0[ng,nt]
            if toon_coefficients == 1 : #eddington
                g3  = (2-3*ftau_cld*cosb*u0)/4#0.5*(1.-sq3*cosb*ubar0[ng, nt]) #  #table 1 #ubar has dimensions [gauss angles by tchebyshev angles ]
            elif toon_coefficients == 0 :#quadrature
                g3  = 0.5*(1.-sq3*ftau_cld*cosb*u0) #  #table 1 #ubar has dimensions [gauss angles by tchebyshev angles ]
    
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
            exptrm_minus = 1.0/exptrm_positive#EM


            #boundary conditions 
            #b_top = 0.0                                       

            b_surface = 0. + surf_reflect*u0*F0PI*exp(-tau[-1, :]/u0)

            #Now we need the terms for the tridiagonal rotated layered method
            #if tridiagonal==0:
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
                #if tridiagonal==0:
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
            flux = zeros((2*nlevel, nwno))
            flux[0,:] = (gama*positive + negative + a_minus)[0,:]
            flux[1,:] = (positive + gama*negative + a_plus)[0,:]
            flux[2::2, :] = flux_minus
            flux[3::2, :] = flux_plus
            flux_out[ng,nt,:,:] = flux

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
                multi_plus = (1.0+1.5*ftau_cld*cosb*u1 #!was 3
                                + gcos2*(3.0*ubar2*ubar2*u1*u1 - 1.0)/2.0)
                multi_minus = (1.-1.5*ftau_cld*cosb*u1 
                                + gcos2*(3.0*ubar2*ubar2*u1*u1 - 1.0)/2.0)
            elif multi_phase ==1:#'N=1':
                multi_plus = 1.0+1.5*ftau_cld*cosb*u1  
                multi_minus = 1.-1.5*ftau_cld*cosb*u1


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
            # p_single=(1-cosb_og**2)/sqrt((1+cosb_og**2-2*cosb_og*cos_theta)**3) (NOTICE NEGATIVE FROM COS_THETA)

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
            
            #removing single form option from code 
            #single_form : int 
            #    form of the phase function can either be written as an 'explicit' (0) henyey greinstein 
            #    or it can be written as a 'legendre' (1) expansion. Default is 'explicit'=0

            #if single_form==1:
            #    TAU = tau; DTAU = dtau; W0 = w0
            #    p_single = 0*p_single
            #    Pu0 = legP(-u0) # legendre polynomials for -u0
            #    Pu1 = legP(u1) # legendre polynomials for -u0
            #    maxterm = 2
            #    for l in range(maxterm):
            #        w = (2*l+1) * cosb_og**l
            #        w_single = (w - (2*l+1)*cosb_og**maxterm) / (1 - cosb_og**maxterm) 
            #        p_single = p_single + w_single * Pu0[l]*Pu1[l]
            #else:
            #    TAU = tau_og; DTAU = dtau_og; W0 = w0_og

            ################################ END OPTIONS FOR DIRECT SCATTERING####################

            single_scat = zeros((nlevel,nwno))
            multi_scat = zeros((nlevel,nwno))
            for i in range(nlayer-1,-1,-1):
                single_scat[i,:] = ((w0_og[i,:]*F0PI/(4.*pi))
                        *(p_single[i,:])*exp(-tau_og[i,:]/u0)
                        *(1. - exp(-dtau_og[i,:]*(u0+u1)
                        /(u0*u1)))*
                        (u0/(u0+u1)))

                multi_scat[i,:] = (A[i,:]*(1. - exp(-dtau[i,:] *(u0+1*u1)/(u0*u1)))*
                        (u0/(u0+1*u1))
                        +G[i,:]*(exp(exptrm[i,:]*1-dtau[i,:]/u1) - 1.0)/(lamda[i,:]*1*u1 - 1.0)
                        +H[i,:]*(1. - exp(-exptrm[i,:]*1-dtau[i,:]/u1))/(lamda[i,:]*1*u1 + 1.0)
                        )

                #direct beam
                xint[i,:] =( xint[i+1,:]*exp(-dtau[i,:]/u1) 
                        #single scattering albedo from sun beam (from ubar0 to ubar1)
                        +(w0_og[i,:]*F0PI/(4.*pi))
                        *(p_single[i,:])*exp(-tau_og[i,:]/u0)
                        *(1. - exp(-dtau_og[i,:]*(u0+u1)
                        /(u0*u1)))*
                        (u0/(u0+u1))
                        #multiple scattering terms p_single
                        +A[i,:]*(1. - exp(-dtau[i,:] *(u0+1*u1)/(u0*u1)))*
                        (u0/(u0+1*u1))
                        +G[i,:]*(exp(exptrm[i,:]*1-dtau[i,:]/u1) - 1.0)/(lamda[i,:]*1*u1 - 1.0)
                        +H[i,:]*(1. - exp(-exptrm[i,:]*1-dtau[i,:]/u1))/(lamda[i,:]*1*u1 + 1.0)
                        )

            xint_at_top[ng,nt,:] = xint[0,:]
            #intensity[ng,nt,:,:] = xint

    return xint_at_top 


@jit(nopython=True, cache=True)
def get_reflected_1d(nlevel, wno,nwno, numg,numt, dtau, tau, w0, cosb,gcos2, ftau_cld, ftau_ray,
    dtau_og, tau_og, w0_og, cosb_og, 
    surf_reflect,ubar0, ubar1,cos_theta, F0PI,single_phase, multi_phase,
    frac_a, frac_b, frac_c, constant_back, constant_forward, 
    get_toa_intensity=1,get_lvl_flux=0,
    toon_coefficients=0,b_top=0):
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
    get_toa_intensity : int 
        (Optional) Default=1 is to only return the TOA intensity you would need for a 1D spectrum (1)
        otherwise it will return zeros for TOA intensity 
    get_lvl_flux : int 
        (Optional) Default=0 is to only compute TOA intensity and NOT return the lvl fluxes so this needs 
        to be flipped on for the climate calculations
    toon_coefficients : int     
        (Optional) 0 for quadrature (default) 1 for eddington

    Returns
    -------
    intensity at the top of the atmosphere for all the different ubar1 and ubar2 
    """
    #these are only filled in if get_toa_intensity=1
    #outgoing intensity as a function of all the different angles
    xint_at_top = zeros(shape=(numg, numt, nwno))

    #these are only filled in if get_lvl_flux=1
    #fluxes at the boundaries 
    flux_minus_all = zeros(shape=(numg, numt,nlevel, nwno)) ## level downwelling fluxes
    flux_plus_all = zeros(shape=(numg, numt, nlevel, nwno)) ## level upwelling fluxes
    #fluxes at the midpoints
    flux_minus_midpt_all = zeros(shape=(numg, numt, nlevel, nwno)) ##  layer downwelling fluxes
    flux_plus_midpt_all = zeros(shape=(numg, numt, nlevel, nwno))  ## layer upwelling fluxes



    nlayer = nlevel - 1 

    #now define terms of Toon et al 1989 quadrature Table 1 
    #https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/JD094iD13p16287
    #see table of terms 

    #terms not dependent on incident angle
    sq3 = sqrt(3.)
    if toon_coefficients == 1:#eddington
        g1  = (7-w0*(4+3*ftau_cld*cosb))/4 #(sq3*0.5)*(2. - w0*(1.+cosb)) #table 1 # 
        g2  = -(1-w0*(4-3*ftau_cld*cosb))/4 #(sq3*w0*0.5)*(1.-cosb)        #table 1 # 
    elif toon_coefficients == 0:#quadrature
        g1  = (sq3*0.5)*(2. - w0*(1.+ftau_cld*cosb)) #table 1 # 
        g2  = (sq3*w0*0.5)*(1.-ftau_cld*cosb)        #table 1 # 

    lamda = sqrt(g1**2 - g2**2)         #eqn 21
    gama  = (g1-lamda)/g2               #eqn 22

    #================ START CRAZE LOOP OVER ANGLE #================
    for ng in range(numg):
        for nt in range(numt):
            u1 = ubar1[ng,nt]
            u0 = ubar0[ng,nt]
            if toon_coefficients == 1 : #eddington
                g3  = (2-3*ftau_cld*cosb*u0)/4#0.5*(1.-sq3*cosb*ubar0[ng, nt]) #  #table 1 #ubar has dimensions [gauss angles by tchebyshev angles ]
            elif toon_coefficients == 0 :#quadrature
                g3  = 0.5*(1.-sq3*ftau_cld*cosb*u0) #  #table 1 #ubar has dimensions [gauss angles by tchebyshev angles ]
    
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
            exptrm_minus = 1.0/exptrm_positive#EM


            #boundary conditions 
            #b_top = 0.0                                       

            b_surface = 0. + surf_reflect*u0*F0PI*exp(-tau[-1, :]/u0)

            #Now we need the terms for the tridiagonal rotated layered method
            #if tridiagonal==0:
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
                #if tridiagonal==0:
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

            #========================= Get fluxes if needed for climate =========================
            if get_lvl_flux: 
                flux_minus=np.zeros(shape=(nlevel,nwno))
                flux_plus=np.zeros(shape=(nlevel,nwno))
                
                flux_minus_midpt = np.zeros(shape=(nlevel,nwno))
                flux_plus_midpt = np.zeros(shape=(nlevel,nwno))
                #use expression for bottom flux to get the flux_plus and flux_minus at last
                #bottom layer
                flux_minus[:-1, :]  = positive*gama + negative + c_minus_up
                flux_plus[:-1, :]  = positive + gama*negative + c_plus_up
                
                flux_zero_minus  = gama[-1,:]*positive[-1,:]*exptrm_positive[-1,:] + negative[-1,:]*exptrm_minus[-1,:] + c_minus_down[-1,:]
                flux_zero_plus  = positive[-1,:]*exptrm_positive[-1,:] + gama[-1,:]*negative[-1,:]*exptrm_minus[-1,:] + c_plus_down[-1,:]
                
                flux_minus[-1, :], flux_plus[-1, :] = flux_zero_minus, flux_zero_plus 
                
                #add in direct flux term to the downwelling radiation, liou 182
                flux_minus = flux_minus + u0*F0PI*exp(-tau/u0)

                #now get midpoint values 
                exptrm_positive_midpt = exp(0.5*exptrm) #EP
                exptrm_minus_midpt = 1.0/exptrm_positive_midpt#EM
                
                #taus at the midpoint
                taumid=tau[:-1]+0.5*dtau
                x = exp(-taumid/ubar0[ng, nt])
                c_plus_mid= a_plus*x
                c_minus_mid=a_minus*x
                #fluxes at the midpoints 
                flux_minus_midpt[:-1,:]= gama*positive*exptrm_positive_midpt + negative*exptrm_minus_midpt + c_minus_mid
                flux_plus_midpt[:-1,:]= positive*exptrm_positive_midpt + gama*negative*exptrm_minus_midpt + c_plus_mid
                #add in midpoint downwelling radiation
                flux_minus_midpt[:-1,:] = flux_minus_midpt[:-1,:] + ubar0[ng, nt]*F0PI*exp(-taumid/ubar0[ng, nt])

                #ARRAYS TO RETURN with all NG and NTs
                flux_minus_all[ng, nt, :, :]=flux_minus
                flux_plus_all[ng, nt, :, :]=flux_plus
                flux_minus_midpt_all[ng, nt, :, :]=flux_minus_midpt
                flux_plus_midpt_all[ng, nt, :, :]=flux_plus_midpt
            #========================= End get fluxes if needed for climate =========================


            #========================= Get intensities if needed for spectrum =========================
            if get_toa_intensity:
                ################################ BEGIN OPTIONS FOR MULTIPLE SCATTERING####################
                #use expression for bottom flux to get the flux_plus and flux_minus at last
                #bottom layer
                flux_zero  = positive[-1,:]*exptrm_positive[-1,:] + gama[-1,:]*negative[-1,:]*exptrm_minus[-1,:] + c_plus_down[-1,:]


                xint = zeros((nlevel,nwno))
                xint[-1,:] = flux_zero/pi

                ################################ begin options for multiple scattering ####################

                #Legendre polynomials for the Phase function due to multiple scatterers 
                if multi_phase ==0:#'N=2':
                    #ubar2 is defined to deal with the integration over the second moment of the 
                    #intensity. It is FIT TO PURE RAYLEIGH LIMIT, ~(1/sqrt(3))^(1/2)
                    #this is a decent assumption because our second order legendre polynomial 
                    #is forced to be equal to the rayleigh phase function
                    ubar2 = 0.767  # 
                    multi_plus = (1.0+1.5*ftau_cld*cosb*u1 #!was 3
                                    + gcos2*(3.0*ubar2*ubar2*u1*u1 - 1.0)/2.0)
                    multi_minus = (1.-1.5*ftau_cld*cosb*u1 
                                    + gcos2*(3.0*ubar2*ubar2*u1*u1 - 1.0)/2.0)
                elif multi_phase ==1:#'N=1':
                    multi_plus = 1.0+1.5*ftau_cld*cosb*u1  
                    multi_minus = 1.-1.5*ftau_cld*cosb*u1
                ################################ end options for multiple scatteirng ####################

                G=positive*(multi_plus+gama*multi_minus)    *w0
                H=negative*(gama*multi_plus+multi_minus)    *w0
                A=(multi_plus*c_plus_up+multi_minus*c_minus_up) *w0

                G=G*0.5/pi
                H=H*0.5/pi
                A=A*0.5/pi

                ################################ being options for direct scattering ####################
                #define f (fraction of forward to back scattering), 
                #g_forward (forward asymmetry), g_back (backward asym)
                #needed for everything except the OTHG
                #the default values in conjig.json are from Cahoy+2010
                if single_phase!=1: 
                    g_forward = constant_forward*cosb_og
                    g_back = constant_back*cosb_og#-
                    f = frac_a + frac_b*g_back**frac_c

                # NOTE ABOUT HG function: we are translating to the frame of the downward propagating beam
                # Therefore our HG phase function becomes:
                # p_single=(1-cosb_og**2)/sqrt((1+cosb_og**2+2*cosb_og*cos_theta)**3) 
                # as opposed to the traditional:
                # p_single=(1-cosb_og**2)/sqrt((1+cosb_og**2-2*cosb_og*cos_theta)**3) (NOTICE NEGATIVE)

                #original phase function of Cahoy+2010, which apporximates the rayleigh with gcos2
                if single_phase==0:#'cahoy':
                    HG_forward = (1-g_forward**2) / sqrt((1+g_forward**2 + 2*g_forward*cos_theta)**3) 
                    HG_backward =(1-g_back**2) / sqrt((1+g_back**2 + 2*g_back*cos_theta)**3)

                    p_single=(f * HG_forward #first term of TTHG: forward scattering
                            +(1-f)*HG_backward #second term of TTHG: backward scattering
                            + (gcos2)) #rayleigh phase function

                #single term HG phase function. Does not separate forward and back scattering. Does not have Rayleigh direct.  
                elif single_phase==1:#'OTHG':
                    p_single=(1-cosb_og**2)/sqrt((1+cosb_og**2+2*cosb_og*cos_theta)**3) 

                #Two HG phase function. Separate forward and back scattering based on parameters above
                #Does not have Rayleigh direct. 
                elif single_phase==2:#'TTHG':
                    HG_forward = (1-g_forward**2) / sqrt((1+g_forward**2 + 2*g_forward*cos_theta)**3) 
                    HG_backward = (1-g_back**2) / sqrt((1+g_back**2 + 2*g_back*cos_theta)**3)
                    p_single=(f * HG_forward #first term of TTHG: forward scattering
                             +(1-f)* HG_backward) #second term of TTHG: backward scattering
                
                #Two HG phase function. Separate forward and back scattering based on parameters above
                #Same as above except now is weighted by the fractional contribution of both 
                #rayleigh vs. cloud scattering
                elif single_phase==3:#'TTHG_ray':
                    #Phase function for single scattering albedo frum Solar beam
                    #uses the Two term Henyey-Greenstein function with the additiona rayleigh component 
                                
                    HG_forward =  (1-g_forward**2) /sqrt((1+g_forward**2+2*g_forward*cos_theta)**3)    
                    HG_back = (1-g_back**2)/sqrt((1+g_back**2+2*g_back*cos_theta)**3)
                    
                    p_single=(
                            ftau_cld * (          #opacity of cloud / total opacity
                                f * HG_forward  + #first term of TTHG: forward scattering
                                (1-f) * HG_back  #second term of TTHG: backward scattering  
                                )+  
                            ftau_ray * (
                                0.75*(1+cos_theta**2.0) #rayleigh phase function
                                )
                            )
                #exploring.... 
                #elif single_phase==4:#'P(HG) exact w/ approx costheta'
                #    deltaphi=0
                #    cos_theta_approx = (-ubar0[ng,nt])*ubar1[ng,nt] + sqrt(1-ubar1[ng,nt]**2)*sqrt(1-ubar0[ng,nt]**2)*cos(deltaphi)
                #    
                #    HG_forward =  (1-g_forward**2) /sqrt((1+g_forward**2+2*g_forward*cos_theta_approx)**3)    
                #    HG_back = (1-g_back**2)/sqrt((1+g_back**2+2*g_back*cos_theta_approx)**3)

                #    p_single=(
                #            ftau_cld * (          #opacity of cloud / total opacity
                #                f * HG_forward  + #first term of TTHG: forward scattering
                #                (1-f) * HG_back   #second term of TTHG: backward scattering  
                #                )+  
                #            ftau_ray * (
                #                0.75*(1+cos_theta_approx**2.0) #rayleigh phase function
                #                )
                #            )

                
                ################################ end options for direct scattering ####################

                #in codes like DISORT the single and multiple scattering beams are reported separately
                #in Rooney et al. 2023a we needed to separate these to 
                #single_scat = zeros((nlevel,nwno))
                #multi_scat = zeros((nlevel,nwno))
                for i in range(nlayer-1,-1,-1):
                    #single_scat[i,:] = ((w0_og[i,:]*F0PI/(4.*pi))
                    #        *(p_single[i,:])*exp(-tau_og[i,:]/u0)
                    #        *(1. - exp(-dtau_og[i,:]*(u0+u1)
                    #        /(u0*u1)))*
                    #        (u0/(u0+u1)))

                    #multi_scat[i,:] = (A[i,:]*(1. - exp(-dtau[i,:] *(u0+1*u1)/(u0*u1)))*
                    #        (u0/(u0+1*u1))
                    #        +G[i,:]*(exp(exptrm[i,:]*1-dtau[i,:]/u1) - 1.0)/(lamda[i,:]*1*u1 - 1.0)
                    #        +H[i,:]*(1. - exp(-exptrm[i,:]*1-dtau[i,:]/u1))/(lamda[i,:]*1*u1 + 1.0)
                    #        )

                    #direct beam
                    xint[i,:] =( xint[i+1,:]*exp(-dtau[i,:]/u1) 
                            #single scattering albedo from sun beam (from ubar0 to ubar1)
                            +(w0_og[i,:]*F0PI/(4.*pi))
                            *(p_single[i,:])*exp(-tau_og[i,:]/u0)
                            *(1. - exp(-dtau_og[i,:]*(u0+u1)
                            /(u0*u1)))*
                            (u0/(u0+u1))
                            #multiple scattering terms p_single
                            +A[i,:]*(1. - exp(-dtau[i,:] *(u0+1*u1)/(u0*u1)))*
                            (u0/(u0+1*u1))
                            +G[i,:]*(exp(exptrm[i,:]*1-dtau[i,:]/u1) - 1.0)/(lamda[i,:]*1*u1 - 1.0)
                            +H[i,:]*(1. - exp(-exptrm[i,:]*1-dtau[i,:]/u1))/(lamda[i,:]*1*u1 + 1.0)
                            )


                xint_at_top[ng,nt,:] = xint[0,:]
            #========================= End get intensities if needed for spectrum =========================
    
    return xint_at_top, (flux_minus_all, flux_plus_all, flux_minus_midpt_all, flux_plus_midpt_all )
    
@jit(nopython=True, cache=True)
def get_reflected_1d_gfluxv_deprecate(nlevel, wno,nwno, numg,numt, dtau, tau, w0, cosb,
    surf_reflect,b_top,b_surface,ubar0, F0PI,tridiagonal, delta_approx):
    """
    Computes upwelling and downwelling layer and level toon fluxes given tau and everything is 1 dimensional. This is the exact same function 
    as `GFLUXV.f'.
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
    surf_reflect : float 
        Surface reflectivity
    b_top : float 
        Top Boundary Conditions
    b_surface : float 
        Surface Boundary Conditions
    ubar0 : ndarray of float 
        matrix of cosine of the incident angle from geometric.json
    F0PI : array 
        Downward incident solar radiation
    delta_approx : int 
        0 for no Delta Approx, 1 for Delta Approx
    Returns
    -------
    intensity at the top of the atmosphere for all the different ubar1 and ubar2 
    To Do
    -----
    - F0PI Solar flux shouldn't always be 1.. Follow up to make sure that this isn't a bad 
          hardwiring to solar, despite "relative albedo"
    """

    
    

    nlayer = nlevel - 1 
    
    
    
    
    #### --SM-- formulas from https://arxiv.org/pdf/1904.09355.pdf
    if delta_approx == 1 :
        dtau=dtau*(1.-w0*cosb**2)
        tau[0]=tau[0]*(1.-w0[0]*cosb[0]**2)
        for i in range(nlayer):
            tau[i+1]=tau[i]+dtau[i]
        
    ##### --SM-- need to correct the tau arrays first and the w0 and cosb arrays later
        w0=w0*((1.-cosb**2)/(1.-w0*(cosb**2)))
        cosb=cosb/(1.+cosb)
        
    
    ## --SM-- creating the four outputs
    flux_minus_all = zeros(shape=(numg, numt,nlevel, nwno)) ## --SM-- level downwelling fluxes
    flux_plus_all = zeros(shape=(numg, numt, nlevel, nwno)) ## --SM-- level upwelling fluxes
    flux_minus_midpt_all = zeros(shape=(numg, numt, nlevel, nwno)) ## --SM-- layer downwelling fluxes
    flux_plus_midpt_all = zeros(shape=(numg, numt, nlevel, nwno))  ## --SM-- layer upwelling fluxes
    
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
                
            A, B, C, D = setup_tri_diag(nlayer,nwno,  c_plus_up, c_minus_up, 
                                c_plus_down, c_minus_down, b_top, b_surface, surf_reflect,
                                    gama, dtau, 
                                exptrm_positive,  exptrm_minus) 

            #else:
            #   A_, B_, C_, D_, E_, F_ = setup_pent_diag(nlayer,nwno,  c_plus_up, c_minus_up, 
            #                       c_plus_down, c_minus_down, b_top, b_surface, surf_reflect,
            #                        gama, dtau, 
            #                       exptrm_positive,  exptrm_minus, g1,g2,exptrm,lamda) 

            positive = zeros(shape=(nlayer, nwno))
            negative = zeros(shape=(nlayer, nwno))
            #========================= Start loop over wavelength =========================
            L = 2*nlayer
            for w in range(nwno):
        
                if tridiagonal==0:
                    X = tri_diag_solve(L, A[:,w], B[:,w], C[:,w], D[:,w])
                    #unmix the coefficients
                    positive[:,w] = X[::2] + X[1::2] 
                    negative[:,w] = X[::2] - X[1::2]
    
            flux_minus=np.zeros(shape=(nlevel,nwno))
            flux_plus=np.zeros(shape=(nlevel,nwno))
            flux_minus_midpt = np.zeros(shape=(nlevel,nwno))
            flux_plus_midpt = np.zeros(shape=(nlevel,nwno))
            #========================= End loop over wavelength =========================
    
            #use expression for bottom flux to get the flux_plus and flux_minus at last
            #bottom layer
            flux_minus[:-1, :]  = positive*gama + negative + c_minus_up
            flux_plus[:-1, :]  = positive + gama*negative + c_plus_up
            
            flux_zero_minus  = gama[-1,:]*positive[-1,:]*exptrm_positive[-1,:] + negative[-1,:]*exptrm_minus[-1,:] + c_minus_down[-1,:]
            flux_zero_plus  = positive[-1,:]*exptrm_positive[-1,:] + gama[-1,:]*negative[-1,:]*exptrm_minus[-1,:] + c_plus_down[-1,:]
            
            flux_minus[-1, :], flux_plus[-1, :] = flux_zero_minus, flux_zero_plus 
            
            flux_minus = flux_minus + ubar0[ng, nt]*F0PI*exp(-tau/ubar0[ng, nt])

            
            exptrm_positive_midpt = exp(0.5*exptrm) #EP
            exptrm_minus_midpt = 1.0/exptrm_positive_midpt#EM
            
            taumid=tau[:-1]+0.5*dtau
            x = exp(-taumid/ubar0[ng, nt])
            c_plus_mid= a_plus*x
            c_minus_mid=a_minus*x

            flux_minus_midpt[:-1,:]= gama*positive*exptrm_positive_midpt + negative*exptrm_minus_midpt + c_minus_mid
            flux_plus_midpt[:-1,:]= positive*exptrm_positive_midpt + gama*negative*exptrm_minus_midpt + c_plus_mid
            
            flux_minus_midpt[:-1,:] = flux_minus_midpt[:-1,:] + ubar0[ng, nt]*F0PI*exp(-taumid/ubar0[ng, nt])

            
            flux_minus_all[ng, nt, :, :]=flux_minus
            flux_plus_all[ng, nt, :, :]=flux_plus
            

            flux_minus_midpt_all[ng, nt, :, :]=flux_minus_midpt
            flux_plus_midpt_all[ng, nt, :, :]=flux_plus_midpt
    
    return flux_minus_all, flux_plus_all, flux_minus_midpt_all, flux_plus_midpt_all 

@jit(nopython=True, cache=True)
def blackbody_integrated(T, wave, dwave):
    """
    This computes the total energey per wavenumber bin needed for the climate calculation 
    Note that this is different than the raw flux at an isolated wavenumber. Therefore this function is 
    different than the blackbody function in `picaso.fluxes` which computes blackbody in raw 
    cgs units. 
    
    Parameters 
    ----------
    T : float, array 
        temperature in Kelvin 
    wave : float, array 
        wavenumber in cm-1 
    dwave : float, array 
        Wavenumber bins in cm-1 
    
    Returns 
    -------
    array 
        num temperatures by num wavenumbers 
        units of ergs/cm*2/s/cm-1 for *integrated* bins ()
    """

    h = 6.62607004e-27 # erg s 
    c = 2.99792458e+10 # cm/s
    k = 1.38064852e-16 #erg / K
    c1 = 2*h*c**2
    c2 = h*c/k
    
    #this number was tested for accuracy against the original number of bins (4)
    #nbb 1 create three wavenumber bins (one on either side of center)
    #It achieves <1% integration accuracy up to black bodies ~50 K for the 
    #legacy 196 and 661 (for 661 max error is only 1e-3%) wavenumber grids. 
    nbb = 1 

    num_wave = len(wave)
    num_T = len(T)

    planck_sum = zeros((num_T, num_wave))

    for i in range(num_wave):
        for j in range(num_T):
            for k in range(-nbb, nbb + 1, 1):
                wavenum = wave[i] + k * dwave[i] / (2.0 * nbb)
                #erg/s/cm2/(cm-1)
                planck_sum[j, i] += c1 * (wavenum**3) / (exp(c2 * wavenum / T[j])-1)
                
    planck_sum /= (2 * nbb + 1.0)

    return planck_sum

@jit(nopython=True, cache=True)
def blackbody(t,w):
    """
    Blackbody flux in cgs units in per unit wavelength (cm)

    Parameters
    ----------
    t : array,float
        Temperature (K)
    w : array, float
        Wavelength (cm)
    
    Returns
    -------
    ndarray with shape ntemp x numwave in units of erg/cm/s2/cm
    """
    h = 6.62607004e-27 # erg s 
    c = 2.99792458e+10 # cm/s
    k = 1.38064852e-16 #erg / K

    return ((2.0*h*c**2.0)/(w**5.0))*(1.0/(exp((h*c)/outer(t, w*k)) - 1.0)) #* (w*w)

@jit(nopython=True, cache=True)
def get_thermal_1d(nlevel, wno,nwno, numg,numt,tlevel, dtau, w0,cosb,plevel, ubar1,
    surf_reflect, hard_surface, dwno, calc_type):
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
    dwno : int 
        delta wno needed for climate
    calc_type : int 
        0 for spectrum model, 1 for climate solver
    Returns
    -------
    numpy.ndarray
        Thermal flux in CGS units (erg/cm3/s) in a matrix that is 
        numg x numt x nwno
    """
    nlayer = nlevel - 1 #nlayers 

    mu1 = 0.5#0.88#0.5 #from Table 1 Toon  

    #get matrix of blackbodies 
    if calc_type == 0: 
        all_b = blackbody(tlevel, 1/wno) #returns nlevel by nwave   
    elif calc_type==1:
        all_b = blackbody_integrated(tlevel, wno, dwno)

    b0 = all_b[0:-1,:]
    b1 = (all_b[1:,:] - b0) / dtau # eqn 26 toon 89

    #hemispheric mean parameters from Tabe 1 toon 
    g1 = 2.0 - w0*(1+cosb); g2 = w0*(1-cosb)

    alpha = sqrt( (1.-w0) / (1.-w0*cosb) )
    lamda = sqrt(g1**2 - g2**2) #eqn 21 toon 
    gama = (g1-lamda)/g2 # #eqn 22 toon
    
    g1_plus_g2 = 1.0/(g1+g2) #second half of eqn.27

    #same as with reflected light, compute c_plus and c_minus 
    #these are eqns 27a & b in Toon89
    #_ups are evaluated at lower optical depth, TOA
    #_dows are evaluated at higher optical depth, bottom of atmosphere
    c_plus_up = 2*pi*mu1*(b0 + b1* g1_plus_g2) 
    c_minus_up = 2*pi*mu1*(b0 - b1* g1_plus_g2)
    #NOTE: to keep consistent with Toon, we keep these 2pis here. However, 
    #in 3d cases where we no long assume azimuthal symmetry, we divide out 
    #by 2pi when we multiply out the weights as seen in disco.compress_thermal 

    c_plus_down = 2*pi*mu1*(b0 + b1 * dtau + b1 * g1_plus_g2) 
    c_minus_down = 2*pi*mu1*(b0 + b1 * dtau - b1 * g1_plus_g2)



    #calculate exponential terms needed for the tridiagonal rotated layered method
    exptrm = lamda*dtau
    #save from overflow 
    exptrm = slice_gt (exptrm, 35.0) 

    exptrm_positive = exp(exptrm) 
    exptrm_minus = 1.0/exptrm_positive

    #for flux heating calculations, the energy balance solver 
    #does not like a fixed zero at the TOA. 
    #to avoid a discontinuous kink at the last atmospher
    #layer we create this "fake" boundary condition
    #we imagine that the atmosphere continus up at an isothermal T and that 
    #there is optical depth from above the top to infinity 
    tau_top = dtau[0,:]*plevel[0]/(plevel[1]-plevel[0]) #tried this.. no luck*exp(-1)# #tautop=dtau[0]*np.exp(-1)
    #print(list(tau_top))
    #tau_top = 26.75*plevel[0]/(plevel[1]-plevel[0]) 
    b_top = (1.0 - exp(-tau_top / mu1 )) * all_b[0,:] * pi #  Btop=(1.-np.exp(-tautop/ubari))*B[0]
    
    if hard_surface:
        b_surface = all_b[-1,:]*pi #for terrestrial, hard surface  
    else: 
        b_surface= (all_b[-1,:] + b1[-1,:]*mu1)*pi #(for non terrestrial)

    #Now we need the terms for the tridiagonal rotated layered method
    #pentadiagonal solver is left here because it may be useful someday 
    #however, curret scipy implementation is too slow to use currently 
    #if tridiagonal==0:
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
        X = tri_diag_solve(L, A[:,w], B[:,w], C[:,w], D[:,w])
        #unmix the coefficients
        positive[:,w] = X[::2] + X[1::2] 
        negative[:,w] = X[::2] - X[1::2]
        #else:
        #   X = pent_diag_solve(L, A_[:,w], B_[:,w], C_[:,w], D_[:,w], E_[:,w], F_[:,w])
        #   positive[:,w] = exptrm_minus[:,w] * (X[::2] + X[1::2])
        #   negative[:,w] = X[::2] - X[1::2]

    #if you stop here this is regular ole 2 stream
    f_up = (positive * exptrm_positive + gama * negative * exptrm_minus + c_plus_up)

    #calculate everyting from Table 3 toon
    #from here forward is source function technique in toon
    G = (1/mu1 - lamda)*positive     
    H = gama*(lamda + 1/mu1)*negative 
    J = gama*(lamda + 1/mu1)*positive 
    K = (1/mu1 - lamda)*negative     
    alpha1 = 2*pi*(b0+b1*(g1_plus_g2 - mu1)) 
    alpha2 = 2*pi*b1 
    sigma1 = 2*pi*(b0-b1*(g1_plus_g2 - mu1)) 
    sigma2 = 2*pi*b1 

    flux_minus = zeros((numg, numt,nlevel,nwno))
    flux_plus = zeros((numg, numt,nlevel,nwno))
    flux_minus_mdpt = zeros((numg, numt,nlevel,nwno))
    flux_plus_mdpt = zeros((numg, numt,nlevel,nwno))

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
                flux_plus[ng,nt,-1,:] = all_b[-1,:] *2*pi  # terrestrial flux /pi = intensity
            else:
                flux_plus[ng,nt,-1,:] = ( all_b[-1,:] + b1[-1,:] * iubar)*2*pi #no hard surface   
                
            flux_minus[ng,nt,0,:] = (1 - exp(-tau_top / iubar)) * all_b[0,:] *2*pi
            
            exptrm_angle = exp( - dtau / iubar)
            exptrm_angle_mdpt = exp( -0.5 * dtau / iubar) 

            for itop in range(nlayer):

                #disbanning this for now because we dont need it in the thermal emission code
                flux_minus[ng,nt,itop+1,:]=(flux_minus[ng,nt,itop,:]*exptrm_angle[itop,:]+
                                     (J[itop,:]/(lamda[itop,:]*iubar+1.0))*(exptrm_positive[itop,:]-exptrm_angle[itop,:])+
                                     (K[itop,:]/(lamda[itop,:]*iubar-1.0))*(exptrm_angle[itop,:]-exptrm_minus[itop,:])+
                                     sigma1[itop,:]*(1.-exptrm_angle[itop,:])+
                                     sigma2[itop,:]*(iubar*exptrm_angle[itop,:]+dtau[itop,:]-iubar) )

                flux_minus_mdpt[ng,nt,itop,:]=(flux_minus[ng,nt,itop,:]*exptrm_angle_mdpt[itop,:]+
                                        (J[itop,:]/(lamda[itop,:]*iubar+1.0))*(exptrm_positive_mdpt[itop,:]-exptrm_angle_mdpt[itop,:])+
                                        (K[itop,:]/(-lamda[itop,:]*iubar+1.0))*(exptrm_minus_mdpt[itop,:]-exptrm_angle_mdpt[itop,:])+
                                        sigma1[itop,:]*(1.-exptrm_angle_mdpt[itop,:])+
                                        sigma2[itop,:]*(iubar*exptrm_angle_mdpt[itop,:]+0.5*dtau[itop,:]-iubar))

                ibot=nlayer-1-itop

                flux_plus[ng,nt,ibot,:]=(flux_plus[ng,nt,ibot+1,:]*exptrm_angle[ibot,:]+
                                  (G[ibot,:]/(lamda[ibot,:]*iubar-1.0))*(exptrm_positive[ibot,:]*exptrm_angle[ibot,:]-1.0)+
                                  (H[ibot,:]/(lamda[ibot,:]*iubar+1.0))*(1.0-exptrm_minus[ibot,:] * exptrm_angle[ibot,:])+
                                  alpha1[ibot,:]*(1.-exptrm_angle[ibot,:])+
                                  alpha2[ibot,:]*(iubar-(dtau[ibot,:]+iubar)*exptrm_angle[ibot,:]) )

                flux_plus_mdpt[ng,nt,ibot,:]=(flux_plus[ng,nt,ibot+1,:]*exptrm_angle_mdpt[ibot,:]+
                                       (G[ibot,:]/(lamda[ibot,:]*iubar-1.0))*(exptrm_positive[ibot,:]*exptrm_angle_mdpt[ibot,:]-exptrm_positive_mdpt[ibot,:])-
                                       (H[ibot,:]/(lamda[ibot,:]*iubar+1.0))*(exptrm_minus[ibot,:]*exptrm_angle_mdpt[ibot,:]-exptrm_minus_mdpt[ibot,:])+
                                       alpha1[ibot,:]*(1.-exptrm_angle_mdpt[ibot,:])+
                                       alpha2[ibot,:]*(iubar+0.5*dtau[ibot,:]-(dtau[ibot,:]+iubar)*exptrm_angle_mdpt[ibot,:])  )


            flux_at_top[ng,nt,:] = flux_plus_mdpt[ng,nt,0,:] #nlevel by nwno 

    return flux_at_top , (flux_minus, flux_plus, flux_minus_mdpt, flux_plus_mdpt)

@jit(nopython=True, cache=True)
def get_thermal_1d_deprecate(nlevel, wno,nwno, numg,numt,tlevel, dtau, w0,cosb,plevel, ubar1,
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
    #flux_out = zeros((numg, numt, 2*nlevel, nwno))

    mu1 = 0.5#0.88#0.5 #from Table 1 Toon  

    #get matrix of blackbodies 
    all_b = blackbody(tlevel, 1/wno) #returns nlevel by nwave   
    b0 = all_b[0:-1,:]
    b1 = (all_b[1:,:] - b0) / dtau # eqn 26 toon 89

    #hemispheric mean parameters from Tabe 1 toon 
    g1 = 2.0 - w0*(1+cosb); g2 = w0*(1-cosb)

    alpha = sqrt( (1.-w0) / (1.-w0*cosb) )
    lamda = sqrt(g1**2 - g2**2) #eqn 21 toon 
    gama = (g1-lamda)/g2 # #eqn 22 toon
    
    g1_plus_g2 = 1.0/(g1+g2) #second half of eqn.27

    #same as with reflected light, compute c_plus and c_minus 
    #these are eqns 27a & b in Toon89
    #_ups are evaluated at lower optical depth, TOA
    #_dows are evaluated at higher optical depth, bottom of atmosphere
    c_plus_up = 2*pi*mu1*(b0 + b1* g1_plus_g2) 
    c_minus_up = 2*pi*mu1*(b0 - b1* g1_plus_g2)
    #NOTE: to keep consistent with Toon, we keep these 2pis here. However, 
    #in 3d cases where we no long assume azimuthal symmetry, we divide out 
    #by 2pi when we multiply out the weights as seen in disco.compress_thermal 

    c_plus_down = 2*pi*mu1*(b0 + b1 * dtau + b1 * g1_plus_g2) 
    c_minus_down = 2*pi*mu1*(b0 + b1 * dtau - b1 * g1_plus_g2)



    #calculate exponential terms needed for the tridiagonal rotated layered method
    exptrm = lamda*dtau
    #save from overflow 
    exptrm = slice_gt (exptrm, 35.0) 

    exptrm_positive = exp(exptrm) 
    exptrm_minus = 1.0/exptrm_positive

    #for flux heating calculations, the energy balance solver 
    #does not like a fixed zero at the TOA. 
    #to avoid a discontinuous kink at the last atmospher
    #layer we create this "fake" boundary condition
    #we imagine that the atmosphere continus up at an isothermal T and that 
    #there is optical depth from above the top to infinity 
    tau_top = dtau[0,:]*plevel[0]/(plevel[1]-plevel[0]) #tried this.. no luck*exp(-1)# #tautop=dtau[0]*np.exp(-1)
    b_top = (1.0 - exp(-tau_top / mu1 )) * all_b[0,:] * pi #  Btop=(1.-np.exp(-tautop/ubari))*B[0]
    #print('hard_surface=',hard_surface)
    if hard_surface:
        b_surface = all_b[-1,:]*pi #for terrestrial, hard surface  
    else: 
        b_surface= (all_b[-1,:] + b1[-1,:]*mu1)*pi #(for non terrestrial)

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
            positive[:,w] = X[::2] + X[1::2] #Y1+Y2 in toon (table 3)
            negative[:,w] = X[::2] - X[1::2] #Y1-Y2 in toon (table 3)
        #else:
        #   X = pent_diag_solve(L, A_[:,w], B_[:,w], C_[:,w], D_[:,w], E_[:,w], F_[:,w])
        #   positive[:,w] = exptrm_minus[:,w] * (X[::2] + X[1::2])
        #   negative[:,w] = X[::2] - X[1::2]

    #if you stop here this is regular ole 2 stream 
    f_up = (positive * exptrm_positive + gama * negative * exptrm_minus + c_plus_up)
    #flux_minus  = gama*positive*exptrm_positive + negative*exptrm_minus + c_minus_down
    #flux_plus  = positive*exptrm_positive + gama*negative*exptrm_minus + c_plus_down
    #flux = zeros((2*nlevel, nwno))
    #flux[0,:] = (gama*positive + negative + c_minus_down)[0,:]
    #flux[1,:] = (positive + gama*negative + c_plus_down)[0,:]
    #flux[2::2, :] = flux_minus
    #flux[3::2, :] = flux_plus


    #calculate everyting from Table 3 toon
    #from here forward is source function technique in toon
    G = (1/mu1 - lamda)*positive     
    H = gama*(lamda + 1/mu1)*negative 
    J = gama*(lamda + 1/mu1)*positive 
    K = (1/mu1 - lamda)*negative     
    alpha1 = 2*pi*(b0+b1*(g1_plus_g2 - mu1)) 
    alpha2 = 2*pi*b1 
    sigma1 = 2*pi*(b0-b1*(g1_plus_g2 - mu1)) 
    sigma2 = 2*pi*b1 

    int_minus = zeros((nlevel,nwno))
    int_plus = zeros((nlevel,nwno))
    int_minus_mdpt = zeros((nlevel,nwno))
    int_plus_mdpt = zeros((nlevel,nwno))
    #intensity = zeros((numg, numt, nlevel, nwno))

    exptrm_positive_mdpt = exp(0.5*exptrm) 
    exptrm_minus_mdpt = 1/exptrm_positive_mdpt 

    #================ START CRAZE LOOP OVER ANGLE #================
    int_at_top = zeros((numg, numt, nwno)) #get intensity 
    int_down = zeros((numg, numt, nwno))

    #work through building eqn 55 in toon (tons of bookeeping exponentials)
    for ng in range(numg):
        for nt in range(numt): 
            #flux_out[ng,nt,:,:] = flux

            iubar = ubar1[ng,nt]

            #intensity boundary conditions
            if hard_surface:
                int_plus[-1,:] = all_b[-1,:] *2*pi  # terrestrial flux /pi = intensity
            else:
                int_plus[-1,:] = ( all_b[-1,:] + b1[-1,:] * iubar)*2*pi #no hard surface   

            int_minus[0,:] =  (1 - exp(-tau_top / iubar)) * all_b[0,:] *2*pi
            
            exptrm_angle = exp( - dtau / iubar)
            exptrm_angle_mdpt = exp( -0.5 * dtau / iubar) 

            for itop in range(nlayer):

                #disbanning this for now because we dont need it in the thermal emission code
                #EQN 56,toon
                int_minus[itop+1,:]=(int_minus[itop,:]*exptrm_angle[itop,:]+
                                     (J[itop,:]/(lamda[itop,:]*iubar+1.0))*(exptrm_positive[itop,:]-exptrm_angle[itop,:])+
                                     (K[itop,:]/(lamda[itop,:]*iubar-1.0))*(exptrm_angle[itop,:]-exptrm_minus[itop,:])+
                                     sigma1[itop,:]*(1.-exptrm_angle[itop,:])+
                                     sigma2[itop,:]*(iubar*exptrm_angle[itop,:]+dtau[itop,:]-iubar) )

                int_minus_mdpt[itop,:]=(int_minus[itop,:]*exptrm_angle_mdpt[itop,:]+
                                        (J[itop,:]/(lamda[itop,:]*iubar+1.0))*(exptrm_positive_mdpt[itop,:]-exptrm_angle_mdpt[itop,:])+
                                        (K[itop,:]/(-lamda[itop,:]*iubar+1.0))*(exptrm_minus_mdpt[itop,:]-exptrm_angle_mdpt[itop,:])+
                                        sigma1[itop,:]*(1.-exptrm_angle_mdpt[itop,:])+
                                        sigma2[itop,:]*(iubar*exptrm_angle_mdpt[itop,:]+0.5*dtau[itop,:]-iubar))

                ibot=nlayer-1-itop
                #EQN 55,toon
                int_plus[ibot,:]=(int_plus[ibot+1,:]*exptrm_angle[ibot,:]+
                                  (G[ibot,:]/(lamda[ibot,:]*iubar-1.0))*(exptrm_positive[ibot,:]*exptrm_angle[ibot,:]-1.0)+
                                  (H[ibot,:]/(lamda[ibot,:]*iubar+1.0))*(1.0-exptrm_minus[ibot,:] * exptrm_angle[ibot,:])+
                                  alpha1[ibot,:]*(1.-exptrm_angle[ibot,:])+
                                  alpha2[ibot,:]*(iubar-(dtau[ibot,:]+iubar)*exptrm_angle[ibot,:]) )

                int_plus_mdpt[ibot,:]=(int_plus[ibot+1,:]*exptrm_angle_mdpt[ibot,:]+
                                       (G[ibot,:]/(lamda[ibot,:]*iubar-1.0))*(exptrm_positive[ibot,:]*exptrm_angle_mdpt[ibot,:]-exptrm_positive_mdpt[ibot,:])-
                                       (H[ibot,:]/(lamda[ibot,:]*iubar+1.0))*(exptrm_minus[ibot,:]*exptrm_angle_mdpt[ibot,:]-exptrm_minus_mdpt[ibot,:])+
                                       alpha1[ibot,:]*(1.-exptrm_angle_mdpt[ibot,:])+
                                       alpha2[ibot,:]*(iubar+0.5*dtau[ibot,:]-(dtau[ibot,:]+iubar)*exptrm_angle_mdpt[ibot,:])  )

            int_at_top[ng,nt,:] = int_plus_mdpt[0,:] #nlevel by nwno 
            #intensity[ng,nt,:,:] = int_plus

            #to get the convective heat flux 
            #flux_minus_mdpt_disco[ng,nt,:,:] = flux_minus_mdpt #nlevel by nwno
            #flux_plus_mdpt_disco[ng,nt,:,:] = int_plus_mdpt #nlevel by nwno

    return int_at_top #, intensity, flux_out #, int_down# numg x numt x nwno

@jit(nopython=True, cache=True)
def get_thermal_3d(nlevel, wno,nwno, numg,numt,tlevel_3d, dtau_3d, w0_3d,cosb_3d,plevel_3d, ubar1,
    surf_reflect, hard_surface):
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

    Returns
    -------
    numpy.ndarray
        Thermal flux in CGS units (erg/cm3/s) in a matrix that is 
        numg x numt x nwno
    """ 

    nlayer = nlevel - 1 #nlayers 
    mu1 = 0.5 #from Table 1 Toon 

    #eventual output
    int_at_top = zeros((numg, numt, nwno))
    int_down = zeros((numg, numt, nwno))
    
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
            g1 = 2.0 - w0*(1+cosb); g2 = w0*(1-cosb)
            alpha = sqrt( (1.-w0) / (1.-w0*cosb) )
            lamda = sqrt(g1**2 - g2**2) #eqn 21 toon 
            gama = (g1-lamda)/g2 # #eqn 22 toon
            g1_plus_g2 = 1.0/(g1+g2) #second half of eqn.27

            #same as with reflected light, compute c_plus and c_minus 
            #these are eqns 27a & b in Toon89
            #_ups are evaluated at lower optical depth, TOA
            #_dows are evaluated at higher optical depth, bottom of atmosphere
            c_plus_up = 2*pi*mu1*(b0 + b1* g1_plus_g2)
            c_minus_up = 2*pi*mu1*(b0 - b1* g1_plus_g2)

            c_plus_down = 2*pi*mu1*(b0 + b1 * dtau + b1 * g1_plus_g2 )
            c_minus_down = 2*pi*mu1*(b0 + b1 * dtau - b1 * g1_plus_g2)

            #calculate exponential terms needed for the tridiagonal rotated layered method
            exptrm = lamda*dtau
            #save from overflow 
            exptrm = slice_gt (exptrm, 35.0) 

            exptrm_positive = exp(exptrm) 
            exptrm_minus = 1.0/exptrm_positive#exp(-exptrm) 

            tau_top = dtau[0,:]*plevel[0]/(plevel[1]-plevel[0])
            b_top = pi*(1.0 - exp(-tau_top / mu1 )) * all_b[0,:] 
            
            if hard_surface:
                b_surface = pi*all_b[-1,:] #for terrestrial, hard surface  
            else: 
                b_surface= pi*(all_b[-1,:] + b1[-1,:]*mu1) #(for non terrestrial)
            #Now we need the terms for the tridiagonal rotated layered method
            #if tridiagonal==0:
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
                #if tridiagonal==0:
                X = tri_diag_solve(L, A[:,w], B[:,w], C[:,w], D[:,w])
                #unmix the coefficients
                positive[:,w] = X[::2] + X[1::2] #Y1+Y2 in toon (table 3)
                negative[:,w] = X[::2] - X[1::2] #Y1-Y2 in toon (table 3)
                #else:
                #   X = pent_diag_solve(L, A_[:,w], B_[:,w], C_[:,w], D_[:,w], E_[:,w], F_[:,w])
                #   positive[:,w] = exptrm_minus[:,w] * (X[::2] + X[1::2])
                #   negative[:,w] = X[::2] - X[1::2]

            f_up = (positive * exptrm_positive + gama * negative * exptrm_minus + c_plus_up)

            #calculate everyting from Table 3 toon
            #alphax = ((1.0-w0)/(1.0-w0*cosb))**0.5
            G = (1/mu1 - lamda)*positive     #G = twopi*w0*positive*(1.0+cosb*alphax)/(1.0+alphax)#
            H = gama*(lamda + 1/mu1)*negative #H = twopi*w0*negative*(1.0-cosb*alphax)/(1.0+alphax)#
            J = gama*(lamda + 1/mu1)*positive #J = twopi*w0*positive*(1.0-cosb*alphax)/(1.0+alphax)#
            K = (1/mu1 - lamda)*negative     #K = twopi*w0*negative*(1.0+cosb*alphax)/(1.0+alphax)#
            alpha1 = 2*pi*(b0+b1*(g1_plus_g2 - mu1)) #alpha1 = twopi*(b0+ b1*(mu1*w0*cosb/(1.0-w0*cosb)))
            alpha2 = 2*pi*b1 #alpha2 = twopi*b1
            sigma1 = 2*pi*(b0-b1*(g1_plus_g2 - mu1)) #sigma1 = twopi*(b0- b1*(mu1*w0*cosb/(1.0-w0*cosb)))
            sigma2 = 2*pi*b1 #sigma2 = twopi*b1

            int_minus = zeros((nlevel,nwno))
            int_plus = zeros((nlevel,nwno))
            int_minus_mdpt = zeros((nlevel,nwno))
            int_plus_mdpt = zeros((nlevel,nwno))

            exptrm_positive_mdpt = exp(0.5*exptrm) 
            exptrm_minus_mdpt = 1/exptrm_positive_mdpt 

            #================ START CRAZE LOOP OVER ANGLE #================
            if hard_surface:
                int_plus[-1,:] = pi * (b_surface ) # terrestrial
            else:
                int_plus[-1,:] = pi * ( all_b[-1,:] + b1[-1,:] * ubar1[ng,nt]) #no hard surface
            
            #work through building eqn 55 in toon (tons of bookeeping exponentials)
            int_minus[0,:] = pi * (1 - exp(-tau_top / ubar1[ng,nt])) * all_b[0,:]
            
            exptrm_angle = exp( - dtau / ubar1[ng,nt])
            exptrm_angle_mdpt = exp( -0.5 * dtau / ubar1[ng,nt]) 

            for itop in range(nlayer):

                #disbanning this for now because we dont need it in the thermal emission code
                int_minus[itop+1,:]=(int_minus[itop,:]*exptrm_angle[itop,:]+
                                     (J[itop,:]/(lamda[itop,:]*ubar1[ng,nt]+1.0))*(exptrm_positive[itop,:]-exptrm_angle[itop,:])+
                                     (K[itop,:]/(lamda[itop,:]*ubar1[ng,nt]-1.0))*(exptrm_angle[itop,:]-exptrm_minus[itop,:])+
                                     sigma1[itop,:]*(1.-exptrm_angle[itop,:])+
                                     sigma2[itop,:]*(ubar1[ng,nt]*exptrm_angle[itop,:]+dtau[itop,:]-ubar1[ng,nt]) )

                int_minus_mdpt[itop,:]=(int_minus[itop,:]*exptrm_angle_mdpt[itop,:]+
                                        (J[itop,:]/(lamda[itop,:]*ubar1[ng,nt]+1.0))*(exptrm_positive_mdpt[itop,:]-exptrm_angle_mdpt[itop,:])+
                                        (K[itop,:]/(-lamda[itop,:]*ubar1[ng,nt]+1.0))*(exptrm_minus_mdpt[itop,:]-exptrm_angle_mdpt[itop,:])+
                                        sigma1[itop,:]*(1.-exptrm_angle_mdpt[itop,:])+
                                        sigma2[itop,:]*(ubar1[ng,nt]*exptrm_angle_mdpt[itop,:]+0.5*dtau[itop,:]-ubar1[ng,nt]))

                ibot=nlayer-1-itop

                int_plus[ibot,:]=(int_plus[ibot+1,:]*exptrm_angle[ibot,:]+
                                  (G[ibot,:]/(lamda[ibot,:]*ubar1[ng,nt]-1.0))*(exptrm_positive[ibot,:]*exptrm_angle[ibot,:]-1.0)+
                                  (H[ibot,:]/(lamda[ibot,:]*ubar1[ng,nt]+1.0))*(1.0-exptrm_minus[ibot,:] * exptrm_angle[ibot,:])+
                                  alpha1[ibot,:]*(1.-exptrm_angle[ibot,:])+
                                  alpha2[ibot,:]*(ubar1[ng,nt]-(dtau[ibot,:]+ubar1[ng,nt])*exptrm_angle[ibot,:]) )

                int_plus_mdpt[ibot,:]=(int_plus[ibot+1,:]*exptrm_angle_mdpt[ibot,:]+
                                       (G[ibot,:]/(lamda[ibot,:]*ubar1[ng,nt]-1.0))*(exptrm_positive[ibot,:]*exptrm_angle_mdpt[ibot,:]-exptrm_positive_mdpt[ibot,:])-
                                       (H[ibot,:]/(lamda[ibot,:]*ubar1[ng,nt]+1.0))*(exptrm_minus[ibot,:]*exptrm_angle_mdpt[ibot,:]-exptrm_minus_mdpt[ibot,:])+
                                       alpha1[ibot,:]*(1.-exptrm_angle_mdpt[ibot,:])+
                                       alpha2[ibot,:]*(ubar1[ng,nt]+0.5*dtau[ibot,:]-(dtau[ibot,:]+ubar1[ng,nt])*exptrm_angle_mdpt[ibot,:])  )

            int_at_top[ng,nt,:] = int_plus_mdpt[0,:] #nlevel by nwno
            #int_down[ng,nt,:] = int_minus_mdpt[0,:] #nlevel by nwno, Dont really need to compute this for now

    return int_at_top #, int_down# numg x numt x nwno

@jit(nopython=True, cache=True)
def get_thermal_1d_gfluxi_deprecate(nlevel, wno,nwno, numg,numt,tlevel, dtau, w0,cosb,plevel, ubar1,surf_reflect,ugauss_angles,ugauss_weights, tridiagonal, calc_type ,dwno): 
    #bb , y2, tp, tmin, tmax):
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
    calc_type : int 
        0 for outgoing flux at top level output, 1 for upward and downward layer and level flux outputs
    Returns
    -------
    numpy.ndarray
        Thermal flux in CGS units (erg/cm3/s) in a matrix that is 
        numg x numt x nwno if calc_type=0
    
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Thermal flux in CGS units (erg/cm3/s) in four matrices if calc_type=0: that is 
        level downward flux and level upward flux numg x numt x nlevel x nwno and then
        layer downward flux and layer upward flux numg x numt x nlayer x nwno  
    """
    nlayer = nlevel - 1 #nlayers 
    

    # Initialising Output Arrays
    #flux_at_top = zeros((numg, numt, nwno)) # output when calc_type=0
    # outputs when calc_type =1
    flux_minus_all = zeros((nlevel, nwno)) ## level downwelling fluxes
    flux_plus_all = zeros(( nlevel, nwno)) ## level upwelling fluxes
    flux_minus_midpt_all = zeros(( nlevel, nwno)) ##  layer downwelling fluxes
    flux_plus_midpt_all = zeros(( nlevel, nwno))  ## layer upwelling fluxes

    mu1 = 0.5#0.88#0.5 #from Table 1 Toon  
    twopi = 2*pi#+pi #NEB REMOVING A PI FROM HERE BECAUSE WE ASSUME NO SYMMETRY!  ############

    #get matrix of blackbodies 
    #all_b = blackbody_climate_deprecate(wno, tlevel, bb, y2, tp, tmin, tmax) #returns nlevel by nwave 
    all_b = blackbody_integrated(tlevel, wno, dwno)

    b0 = all_b[0:-1,:]
    b1 = (all_b[1:,:] - b0) / dtau # eqn 26 toon 89

    #if dtau is less than 1e-6 set b1 to zero 
    #neb-was in fortran but doesnt look needed, keep for now
    #b1 = slice_lt_cond(b1, dtau, 1e-6, 0.0)
    #b0 = slice_lt_cond_arr(b0, dtau, 1e-6, all_b)

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

    #*
    tau_top = dtau[0,:]*plevel[0]/(plevel[1]-plevel[0]) #tried this.. no luck*exp(-1)# #tautop=dtau[0]*np.exp(-1)
    b_top = (1.0 - exp(-tau_top / mu1 )) * all_b[0,:]  # Btop=(1.-np.exp(-tautop/ubari))*B[0]
    #b_surface = all_b[-1,:] #for terrestrial, hard surface  
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
    
    

    #work through building eqn 55 in toon (tons of bookeeping exponentials)
    #for ng in range(numg):
    #    for nt in range(numt): 
    for iubar, weight in zip(ugauss_angles,ugauss_weights):
            #iubar = ubar1[ng,nt]

            #flux_plus[-1,:] = twopi * (b_surface )# terrestrial
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

            #flux_at_top[ng,nt,:] = flux_plus_mdpt[0,:] #nlevel by nwno 
            
        flux_minus_all[:,:]+=flux_minus[:,:]*weight
        flux_plus_all[:,:]+=flux_plus[:,:]*weight

        flux_minus_midpt_all[:,:]+=flux_minus_mdpt[:,:]*weight
        flux_plus_midpt_all[:,:]+=flux_plus_mdpt[:,:]*weight
    

    
    return flux_minus_all, flux_plus_all, flux_minus_midpt_all, flux_plus_midpt_all

@jit(nopython=True, cache=True,fastmath=True)
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

    Notes
    -----
    .. [1] Brown, Timothy M. "Transmission spectra as diagnostics of extrasolar giant planet atmospheres." The Astrophysical Journal 553.2 (2001): 1006.
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
    #equation 11 from Brown, T (2001)
    #https://ui.adsabs.harvard.edu/abs/2001ApJ...553.1006B/abstract 
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
def get_reflected_SH(nlevel, nwno, numg, numt, dtau, tau, w0, cosb, ftau_cld, ftau_ray, f_deltaM,
    dtau_og, tau_og, w0_og, cosb_og, 
    surf_reflect, ubar0, ubar1, cos_theta, F0PI, 
    w_single_form, w_multi_form, psingle_form, w_single_rayleigh, w_multi_rayleigh, psingle_rayleigh,
    frac_a, frac_b, frac_c, constant_back, constant_forward, stream, b_top=0, flx=0, single_form=0):
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
    dtau : ndarray of float
        This is the opacity contained within each individual layer (defined at midpoints of "levels")
        WITHOUT D-Eddington Correction
        Dimensions=# layer by # wave
    tau : ndarray of float
        This is the cumulative summed opacity 
        WITHOUT D-Eddington Correction
        Dimensions=# level by # wave        
    w0 : ndarray of float 
        This is the single scattering albedo, from scattering, clouds, raman, etc 
        WITHOUT D-Eddington Correction
        Dimensions=# layer by # wave
    cosb : ndarray of float 
        This is the asymmetry factor 
        WITHOUT D-Eddington Correction
        Dimensions=# layer by # wave
    ftau_cld : ndarray of float 
        Fraction of cloud extinction to total 
        = tau_cloud/(tau_rayleigh + tau_cloud)
    ftau_ray : ndarray of float 
        Fraction of rayleigh extinction to total 
        = tau_rayleigh/(tau_rayleigh + tau_cloud)
    f_deltaM : ndarray of float 
        Fractional scattering coefficient for delta-M calculation
        f_deltaM = 0 if delta_eddington=False
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
    w_single_form : str 
        Single scattering phase function approximation for SH
    w_multi_form : str 
        Multiple scattering phase function approximation for SH
    psingle_form : str 
        Scattering phase function approximation for psingle in SH
    w_single_rayleigh : str 
        Toggle rayleigh scattering on/off for single scattering in SH
    w_multi_rayleigh : str 
        Toggle rayleigh scattering on/off for multi scattering in SH
    psingle_rayleigh : str 
        Toggle rayleigh scattering on/off for psingle in SH
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
    stream : int 
        Order of expansion of Legendre polynomials (2 or 4)
    b_top : float 
        Upper boundary condition for incoming intensity
    flx : int 
        Toggle calculation of layerwise fluxes (0 = do not calculate, 1 = calculate)
    single_form : int 
        Toggle which version of p_single to use (0 = explicit, 1 = legendre)
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
    
    #================ START CRAZE LOOP OVER ANGLE #================
    for ng in range(numg):
        for nt in range(numt):
            u1 = ubar1[ng,nt]
            u0 = ubar0[ng,nt]
            Pu0 = legP(-u0) # legendre polynomials for -u0
            Pu1 = legP(u1) # legendre polynomials for -u0

            a = zeros((stream, nlayer, nwno))
            b = zeros((stream, nlayer, nwno))
            w_single = ones((stream, nlayer, nwno))
            w_multi = ones(((stream, nlayer, nwno)))
            p_single = zeros(cosb_og.shape)



            if (w_single_form==1 or w_multi_form==1): # OTHG:
                for l in range(1,stream):
                    w = (2*l+1) * cosb_og**l
                    if w_single_form==1:
                        w_single[l,:,:] = (w - (2*l+1)*f_deltaM) / (1 - f_deltaM)
                    if w_multi_form==1:
                        w_multi[l,:,:] = (w - (2*l+1)*f_deltaM) / (1 - f_deltaM)

            if ((w_single_form==0) or (w_multi_form==0)): # TTHG
                g_forward = constant_forward*cosb_og
                g_back = constant_back*cosb_og
                f = frac_a + frac_b*g_back**frac_c
                f_deltaM_ = f_deltaM
                f_deltaM_ *= (f*constant_forward**stream + (1-f)*constant_back**stream)

                for l in range(1,stream):
                    w = (2*l+1) * (f*g_forward**l + (1-f)*g_back**l)
                    if w_single_form==0:
                        w_single[l,:,:] = (w - (2*l+1)*f_deltaM_) / (1 - f_deltaM_)
                    if w_multi_form==0:
                        w_multi[l,:,:] = (w - (2*l+1)*f_deltaM_) / (1 - f_deltaM_)

            if w_single_rayleigh==1:
                w_single[1:] *= ftau_cld
                if stream==4:
                    w_single[2] += 0.5*ftau_ray 
            if w_multi_rayleigh==1: 
                w_multi[1:] *= ftau_cld
                if stream==4:
                    w_multi[2] += 0.5*ftau_ray 

            #single-scattering options
            if single_form==0: # explicit single form
                if psingle_form==1: #OTHG
                    p_single=(1-cosb_og**2)/(sqrt(1+cosb_og**2+2*cosb_og*cos_theta)**3) 
                elif psingle_form==0: #'TTHG':
                    g_forward = constant_forward*cosb_og
                    g_back = constant_back*cosb_og
                    f = frac_a + frac_b*g_back**frac_c
                    p_single=(f * (1-g_forward**2) /sqrt((1+g_forward**2+2*g_forward*cos_theta)**3) 
                                #second term of TTHG: backward scattering
                                +(1-f)*(1-g_back**2) /sqrt((1+g_back**2+2*g_back*cos_theta)**3))

                if psingle_rayleigh==1: 
                    p_single = ftau_cld*p_single + ftau_ray*(0.75*(1+cos_theta**2.0))


            for l in range(stream):
                a[l,:,:] = (2*l + 1) -  w0 * w_multi[l,:,:]
                b[l,:,:] = ( F0PI * (w0 * w_single[l,:,:])) * Pu0[l] / (4*pi)

            #boundary conditions 
            b_surface = (0. + surf_reflect*u0*F0PI*exp(-tau[-1, :]/u0))#/(2*pi)
            b_surface_SH4 = -(0. + surf_reflect*u0*F0PI*exp(-tau[-1, :]/u0))/4#/(2*pi)# need to double check BCs)
            b_top_ = b_top#/(2*pi)

            if stream==2:
                M, B, F_bot, G_bot, F, G, Q1, Q2, lam, q, eta  = setup_2_stream_fluxes(nlayer, nwno, w0, b_top_, b_surface, 
                surf_reflect, u0, dtau, tau, a, b, fluxes=flx, calculation=0) 

            if stream==4:
                M, B, F_bot, G_bot, F, G, lam1, lam2, A, eta = setup_4_stream_fluxes(nlayer, nwno, w0, b_top_, b_surface, b_surface_SH4, 
                    surf_reflect, u0, dtau, tau, a, b, fluxes=flx, calculation=0) 

                # F and G will be nonzero if fluxes=1

            flux_bot = zeros(nwno)
            intgrl_new = zeros((stream*nlayer, nwno))
            flux_temp = zeros((stream*nlevel, nwno))
            intgrl_per_layer = zeros((nlayer, nwno))
            xint_temp = zeros((nlevel, nwno))
            multi_scat = zeros((nlayer, nwno))

            #========================= Start loop over wavelength =========================
            X = zeros((stream*nlayer, nwno))
            flux_temp = zeros((stream*nlevel, nwno))
            for W in range(nwno):
                X[:,W] = solve_4_stream_banded(M[:,:,W], B[:,W], stream)
                if flx==1:
                    flux_temp[:,W] = calculate_flux(F[:,:,W], G[:,W], X[:,W])
            flux_bot = np.sum(F_bot*X, axis=0) + G_bot

            intgrl_new = zeros((stream*nlayer, nwno))
            intgrl_per_layer = zeros((nlayer, nwno))
            xint_temp = zeros((nlevel, nwno))
            multi_scat = zeros((nlayer, nwno))

            Pubar1 = legP(u1) 

            mus = (u1 + u0) / (u1 * u0)
            expo_mus = slice_rav(mus * dtau, 35.0)    
            exptrm_mus = (1 - exp(-expo_mus)) / mus
            tau_mu = slice_rav(tau[:-1,:] * 1/u0, 35.0)
            exptau_mu = exp(-tau_mu)
            expon1 = exptrm_mus * exptau_mu

            if stream==2:
                alpha = 1/u1 + lam
                beta = 1/u1 - lam
                expo_alp = slice_rav(alpha * dtau, 35.0)
                expo_bet = slice_rav(beta * dtau, 35.0) 
                exptrm_alp = (1 - exp(-expo_alp)) / alpha 
                exptrm_bet = (1 - exp(-expo_bet)) / beta

                # fill integrated matrices needed for source-function technique
                Aint0 = X[::2,:]  * (w_multi[0]-w_multi[1]*Pubar1[1]*q) * exptrm_alp
                Aint1 = X[1::2,:] * (w_multi[0]+w_multi[1]*Pubar1[1]*q) * exptrm_bet

                Nint0 = w_multi[0]*(eta[0] * expon1)
                Nint1 = w_multi[1]*Pubar1[1]*(eta[1] * expon1)

                multi_scat = Aint0 + Nint0 + Aint1 + Nint1

            elif stream==4:
                alpha1 = 1/u1 + lam1
                alpha2 = 1/u1 + lam2
                beta1 =  1/u1 - lam1
                beta2 =  1/u1 - lam2
                expo_alp1 = slice_rav(alpha1 * dtau,35.0)
                expo_alp2 = slice_rav(alpha2 * dtau,35.0)
                expo_bet1 = slice_rav(beta1 * dtau ,35.0)
                expo_bet2 = slice_rav(beta2 * dtau ,35.0)
                exptrm = np.zeros((4,nlayer,nwno))
                exptrm[0] = (1 - exp(-expo_alp1)) / alpha1 * X[::4]
                exptrm[1] = (1 - exp(-expo_bet1)) / beta1  * X[1::4]
                exptrm[2] = (1 - exp(-expo_alp2)) / alpha2 * X[2::4]
                exptrm[3] = (1 - exp(-expo_bet2)) / beta2  * X[3::4]

                # fill integrated matrices needed for source-function technique
                Aint=np.zeros((4,nlayer,nwno))
                for j in range(4):
                    Aint = Aint + w_multi[j]*Pubar1[j] * A[j]
                Aint = Aint * exptrm

                Nint0 = w_multi[0]*Pubar1[0] * eta[0] * expon1
                Nint1 = w_multi[1]*Pubar1[1] * eta[1] * expon1
                Nint2 = w_multi[2]*Pubar1[2] * eta[2] * expon1
                Nint3 = w_multi[3]*Pubar1[3] * eta[3] * expon1

                # this could be tidier but going for transparency with paper for now
                multi_scat = (Aint[0] + Nint0 + Aint[1] + Nint1
                                + Aint[2] + Nint2 + Aint[3] + Nint3)

            if single_form==1:
                maxterm = stream
                for l in range(maxterm):
                    p_single = p_single + w_single[l] * Pu0[l]*Pu1[l]

            expo_mus1 = slice_rav(mus * dtau_og, 35.0)    
            exptrm_mus1 = exp(-expo_mus1)
            intgrl_per_layer = (w0 *  multi_scat 
                        + w0_og * F0PI / (4*np.pi) * p_single 
                        * (1 - exptrm_mus1) * exp(-tau_og[:-1,:]/u0)
                        / mus
                        )

            xint_temp[-1,:] = flux_bot/pi
            for i in range(nlayer-1,-1,-1):
                xint_temp[i, :] = (xint_temp[i+1, :] * np.exp(-dtau[i,:]/u1)
                            + intgrl_per_layer[i,:] / u1) 

            xint_at_top[ng,nt,:] = xint_temp[0, :]
            #xint_out[ng,nt,:,:] = xint_temp
            flux[ng,nt,:,:] = flux_temp
    
    return xint_at_top, flux#, xint_out

#@jit(nopython=True, cache=True, debug=True)
def get_thermal_SH(nlevel, wno, nwno, numg, numt, tlevel, dtau, tau, w0, cosb, 
            dtau_og, tau_og, w0_og, w0_no_raman, cosb_og, plevel, ubar1,
            surf_reflect, stream, hard_surface, flx=0):
    """
    The result of this routine is the top of the atmosphere thermal intensity as 
    a function of gauss and chebychev points accross the disk. 

    Everything here is in CGS units:

    Fluxes - erg/s/cm^3
    Temperature - K 
    Wave grid - cm-1
    Pressure ; dyne/cm2

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
    tau : numpy.ndarray
        This is a matrix of nlevel by nwave. This describes the cumulative optical depth. 
    w0 : numpy.ndarray
        This is a matrix of nlayer by nwave. This describes the single scattering albedo of 
        the atmosphere. Note this is free of any Raman scattering or any d-eddington correction 
        that is sometimes included in reflected light calculations.
    cosb : numpy.ndarray
        This is a matrix of nlayer by nwave. This describes the asymmetry of the 
        atmosphere. Note this is free of any Raman scattering or any d-eddington correction 
        that is sometimes included in reflected light calculations.
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
    plevel : numpy.ndarray
        Pressure for each level (not layer, which is midpoints). CGS units (dyne/cm2)
    ubar1 : numpy.ndarray
        This is a matrix of ng by nt. This describes the outgoing incident angles and is generally
        computed in `picaso.disco`
    surf_reflect : numpy.ndarray    
        Surface reflectivity as a function of wavenumber. 
    stream : int 
        Order of expansion of Legendre polynomials (2 or 4)
    hard_surface : int
        0 for no hard surface (e.g. Jupiter/Neptune), 1 for hard surface (terrestrial)
    flx : int 
        Toggle calculation of layerwise fluxes (0 = do not calculate, 1 = calculate)

    Returns
    -------
    numpy.ndarray
        Thermal flux in CGS units (erg/cm3/s) in a matrix that is 
        numg x numt x nwno
    """
    nlayer = nlevel - 1 #nlayers 

    mu1 = 0.5#0.88#0.5 #from Table 1 Toon  
    w0_og = w0_no_raman

    #get matrix of blackbodies 
    all_b = blackbody(tlevel, 1/wno) #returns nlevel by nwave   
    b0 = all_b[0:-1,:]
    b1 = (all_b[1:,:] - b0) / dtau # eqn 26 toon 89
    
    tau_top = dtau[0,:]*plevel[0]/(plevel[1]-plevel[0]) #tried this.. no luck*exp(-1)# #tautop=dtau[0]*np.exp(-1)
    b_top = pi*(1.0 - exp(-tau_top / mu1 )) * all_b[0,:]  # Btop=(1.-np.exp(-tautop/ubari))*B[0]

    if hard_surface:
        b_surface = pi*all_b[-1,:] #for terrestrial, hard surface
    else:
        b_surface = pi*(all_b[-1,:] + b1[-1,:]*mu1)  #(for non terrestrial)

    b_surface_SH4 = (-pi*all_b[-1,:]/4 )

    if np.array_equal(cosb,cosb_og):
        ff = 0.*cosb_og
    else:
        ff = cosb_og**stream

    w_single = zeros((stream, nlayer, nwno))
    w_multi = zeros(((stream, nlayer, nwno)))
    a = zeros(((stream, nlayer, nwno)))
    b = zeros(((stream, nlayer, nwno)))
    for l in range(stream):
        w_multi[l,:,:] = (2*l+1) * (cosb_og**l - ff) / (1 - ff)
        a[l,:,:] = (2*l + 1) -  w0 * w_multi[l,:,:]

    xint_at_top = zeros((numg, numt, nwno))
    intensity = zeros((numg, numt, nlevel, nwno))
    flux = zeros((numg, numt, stream*nlevel, nwno))

    if stream==2:
        M, B, F_bot, G_bot, F, G, Q1, Q2, lam, q, eta =  setup_2_stream_fluxes(nlayer, nwno, w0, b_top, b_surface, 
                surf_reflect, 0, dtau, tau, a, b, B0=b0, B1=b1, fluxes=flx, calculation=1)
    elif stream==4:
        M, B, F_bot, G_bot, F, G, lam1, lam2, A, eta = setup_4_stream_fluxes(nlayer, nwno, w0, 
                b_top, b_surface, b_surface_SH4, surf_reflect, 0, dtau, tau, a, b, B0=b0, B1=b1,  
                fluxes=flx, calculation=1)

    #========================= Start loop over wavelength =========================
    X = zeros((stream*nlayer, nwno))
    for W in range(nwno):
        X[:,W] = solve_4_stream_banded(M[:,:,W], B[:,W], stream)
        if flx==1:
            flux_temp[:,W] = calculate_flux(F[:,:,W], G[:,W], X)
    flux_bot = np.sum(F_bot*X, axis=0) + G_bot

    for ng in range(numg):
        for nt in range(numt):

            intgrl_new = zeros((stream*nlayer, nwno))
            intgrl_per_layer = zeros((nlayer, nwno))
            multi_scat = zeros((nlayer, nwno))
            xint_temp = zeros((nlevel, nwno))
            flux_temp = zeros((stream*nlevel, nwno))

            Pubar1 = legP(ubar1[ng,nt]) 

            if stream==2:
                alpha = 1/ubar1[ng,nt] + lam
                beta = 1/ubar1[ng,nt] - lam
                expo_alp = slice_rav(alpha * dtau, 35.0)
                expo_bet = slice_rav(beta * dtau, 35.0) 
                exptrm_alp = (1 - exp(-expo_alp)) / alpha 
                exptrm_bet = (1 - exp(-expo_bet)) / beta

                Aint0 = X[::2,:]  * (w_multi[0]-w_multi[1]*Pubar1[1]*q) * exptrm_alp
                Aint1 = X[1::2,:] * (w_multi[0]+w_multi[1]*Pubar1[1]*q) * exptrm_bet

                expdtau = exp(-dtau/ubar1[ng,nt])
                Nint0 = w_multi[0]*((1-w0) * ubar1[ng,nt] / a[0] * (b0 *(1-expdtau) + b1*(ubar1[ng,nt] - (dtau+ubar1[ng,nt])*expdtau))) 
                Nint1 = w_multi[1]*Pubar1[1]*((1-w0) * ubar1[ng,nt] / a[0] * ( b1*(1-expdtau) / a[1])) #* 2*pi

                multi_scat = Aint0 + Nint0 + Aint1 + Nint1

            elif stream==4:
                u1 = ubar1[ng,nt]
                alpha1 = 1/u1 + lam1
                alpha2 = 1/u1 + lam2
                beta1 =  1/u1 - lam1
                beta2 =  1/u1 - lam2
                expo_alp1 = slice_rav(alpha1 * dtau,35.0)
                expo_alp2 = slice_rav(alpha2 * dtau,35.0)
                expo_bet1 = slice_rav(beta1 * dtau ,35.0)
                expo_bet2 = slice_rav(beta2 * dtau ,35.0)
                exptrm = np.zeros((4,nlayer,nwno))
                exptrm[0] = (1 - exp(-expo_alp1)) / alpha1 * X[::4]
                exptrm[1] = (1 - exp(-expo_bet1)) / beta1  * X[1::4]
                exptrm[2] = (1 - exp(-expo_alp2)) / alpha2 * X[2::4]
                exptrm[3] = (1 - exp(-expo_bet2)) / beta2  * X[3::4]

                Aint=np.zeros((4,nlayer,nwno))
                for j in range(4):
                    Aint = Aint + w_multi[j]*Pubar1[j] * A[j]
                Aint = Aint * exptrm

                expdtau = exp(-slice_rav(dtau/u1,35.0))
                Nint0 = w_multi[0] * ((1-w0) * u1 / a[0] * ( b0*(1-expdtau) + b1*(u1 - (dtau+u1)*expdtau)))
                Nint1 = w_multi[1]*u1* ((1-w0) * u1 / a[0] * ( b1*(1-expdtau) / a[1]))
                Nint2 = zeros(w0.shape)
                Nint3 = zeros(w0.shape)

                multi_scat = Aint[0] + Aint[1] + Aint[2] + Aint[3] + Nint0 + Nint1 + Nint2 + Nint3


            expo = dtau / ubar1[ng,nt] 
            expo_mus = slice_rav(expo, 35.0)    
            expdtau = exp(-expo)

            intgrl_per_layer = (w0 *  multi_scat *2*pi
                        + 2*pi*(1-w0) * ubar1[ng,nt] *
                        (b0 * (1 - expdtau)
                        + b1 * (ubar1[ng,nt] - (dtau + ubar1[ng,nt]) * expdtau)))


            if hard_surface:
                xint_temp[-1,:] = all_b[-1,:] *2*pi  # terrestrial flux /pi = intensity
            else:
                xint_temp[-1,:] = ( all_b[-1,:] + b1[-1,:] * ubar1[ng,nt])*2*pi #no hard surface   

            for i in range(nlayer-1,-1,-1):
                xint_temp[i, :] = (xint_temp[i+1, :] * np.exp(-dtau[i,:]/ubar1[ng,nt]) 
                            + intgrl_per_layer[i,:] / ubar1[ng,nt]) 

            xint_at_top[ng,nt,:] = xint_temp[0, :]
            #intensity[ng,nt,:,:] = xint_temp
            flux[ng,nt,:,:] = flux_temp
    
    return xint_at_top, flux 

#@jit(nopython=True, cache=True)
def setup_2_stream_fluxes(nlayer, nwno, w0, b_top, b_surface, surf_reflect, ubar0, 
        dtau, tau, a, b, B0=0., B1=0., fluxes=0, calculation=0):#'reflected'):
    """
    Setup up matrices to solve flux problem for spherical harmonics method.

    Parameters
    ----------
    nlayer : int 
        Number of layers
    nwno : int 
        Number of wavenumber points 
    w0 : numpy.ndarray
        This is a matrix of nlayer by nwave. This describes the single scattering albedo of 
        the atmosphere. Note this is free of any Raman scattering or any d-eddington correction 
        that is sometimes included in reflected light calculations.
    b_top : array 
        The diffuse radiation into the model at the top of the atmosphere
    b_surface : array
        The diffuse radiation into the model at the bottom. Includes emission, reflection 
        of the unattenuated portion of the direct beam  
    b_surface_SH4 : array
        Second bottom BC for SH4 method.
    surf_reflect : numpy.ndarray    
        Surface reflectivity as a function of wavenumber. 
    ubar0 : ndarray of float 
        matrix of cosine of the incident angle from geometric.json
    dtau : numpy.ndarray
        This is a matrix of nlayer by nwave. This describes the per layer optical depth. 
    tau : numpy.ndarray
        This is a matrix of nlevel by nwave. This describes the cumulative optical depth. 
    a: numpy.ndarray
        Coefficients of matrix capturing legendre expansion of phase function for multiple scattering
    b: numpy.ndarray
        Coefficients of source vector capturing legendre expansion of phase function for single 
        scattering
    B0 : numpy.ndarray
        Matrix of blackbodies
    B1 : numpy.ndarray
        Eqn (26) Toon 89
    fluxes : int 
        Toggle calculation of layerwise fluxes (0 = do not calculate, 1 = calculate)
    calculation : int 
        Toggle calculation method (1 = linear, 2 = exponential)

    Returns
    -------
    numpy.ndarrays
       Matrices and vectors used to calculate fluxes and intensities at each level 
    """

    eta = zeros((2, nlayer, nwno)) # will remain zero for thermal
    if calculation==0: #reflected light
        Del = ((1 / ubar0)**2 - a[0]*a[1])
        eta = [(b[1] /ubar0 - a[1]*b[0]) / Del,
            (b[0] /ubar0 - a[0]*b[1]) / Del]

    lam = sqrt(a[0]*a[1])
    expo = lam*dtau
    expo = slice_rav(expo, 35.0) 
    exptrm = exp(-expo)

    #   parameters in matrices
    q = lam/a[1]
    Q1 = (0.5 + q)*2*pi
    Q2 = (0.5 - q)*2*pi

    Q1mn = Q1*exptrm;  Q2mn = Q2*exptrm
    Q1pl = Q1/exptrm;  Q2pl = Q2/exptrm

    if calculation == 0: #reflected light
        zmn = (0.5*eta[0] - eta[1])*2*pi
        zpl = (0.5*eta[0] + eta[1])*2*pi
        expon = exp(-tau/ubar0)
        zmn_up = zmn * expon[1:,:] 
        zpl_up = zpl * expon[1:,:] 
        zmn_down = zmn * expon[:-1,:] 
        zpl_down = zpl * expon[:-1,:] 
    elif calculation == 1: # linear thermal
        zmn_down = ((1-w0)/a[0] * (B0/2 - B1/a[1])) *2*pi           #* 2*pi
        zmn_up = ((1-w0)/a[0] * (B0/2 - B1/a[1] + B1*dtau/2)) *2*pi #* 2*pi
        zpl_down = ((1-w0)/a[0] * (B0/2 + B1/a[1])) *2*pi           #* 2*pi
        zpl_up = ((1-w0)/a[0] * (B0/2 + B1/a[1] + B1*dtau/2)) *2*pi #* 2*pi


    #   construct matrices
    Mb = zeros((5, 2*nlayer, nwno))
    B = zeros((2*nlayer, nwno))
    nlevel = nlayer+1
    F = zeros((2*nlevel, 2*nlayer, nwno))
    G = zeros((2*nlevel, nwno))

    #   first row: BC 1
    Mb[2,0,:] = Q1[0,:]
    Mb[1,1,:] = Q2[0,:]
    B[0,:] = b_top - zmn_down[0,:]

    #   last row: BC 4
    n = nlayer-1
    Mb[3, 2*nlayer-2,:] = Q2mn[n,:] - surf_reflect*Q1mn[n,:]
    Mb[2, 2*nlayer-1,:] = Q1pl[n,:] - surf_reflect*Q2pl[n,:]
    B[2*nlayer-1,:] = b_surface - zpl_up[n,:] + surf_reflect * zmn_up[n,:]

    # remaining rows
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


    # flux at bottom of atmosphere
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

        G[0,:] = zmn_down[0,:]
        G[1,:] = zpl_down[0,:]

        G[2::2,:] = zmn_up
        G[3::2,:] = zpl_up

    return Mb, B, F_bot, G_bot, F, G, Q1, Q2, lam, q, eta

#@jit(nopython=True, cache=True, debug=True)
def setup_4_stream_fluxes(nlayer, nwno, w0, b_top, b_surface, b_surface_SH4, surf_reflect, ubar0, 
        dtau, tau, a, b, B0=0., B1=0., fluxes=0, calculation=0):#'reflected'):

    """
    Setup up matrices to solve flux problem for spherical harmonics method.

    Parameters
    ----------
    nlayer : int 
        Number of layers
    nwno : int 
        Number of wavenumber points 
    w0 : numpy.ndarray
        This is a matrix of nlayer by nwave. This describes the single scattering albedo of 
        the atmosphere. Note this is free of any Raman scattering or any d-eddington correction 
        that is sometimes included in reflected light calculations.
    b_top : array 
        The diffuse radiation into the model at the top of the atmosphere
    b_surface : array
        The diffuse radiation into the model at the bottom. Includes emission, reflection 
        of the unattenuated portion of the direct beam  
    b_surface_SH4 : array
        Second bottom BC for SH4 method.
    surf_reflect : numpy.ndarray    
        Surface reflectivity as a function of wavenumber. 
    ubar0 : ndarray of float 
        matrix of cosine of the incident angle from geometric.json
    dtau : numpy.ndarray
        This is a matrix of nlayer by nwave. This describes the per layer optical depth. 
    tau : numpy.ndarray
        This is a matrix of nlevel by nwave. This describes the cumulative optical depth. 
    a: numpy.ndarray
        Coefficients of matrix capturing legendre expansion of phase function for multiple scattering
    b: numpy.ndarray
        Coefficients of source vector capturing legendre expansion of phase function for single 
        scattering
    B0 : numpy.ndarray
        Matrix of blackbodies
    B1 : numpy.ndarray
        Eqn (26) Toon 89
    fluxes : int 
        Toggle calculation of layerwise fluxes (0 = do not calculate, 1 = calculate)
    calculation : int 
        Toggle calculation method (0 = reflected, 1 = thermal)

    Returns
    -------
    numpy.ndarrays
       Matrices and vectors used to calculate fluxes and intensities at each level 
    """

    nlevel = nlayer+1
    beta = a[0]*a[1] + 4*a[0]*a[3]/9 + a[2]*a[3]/9
    gama = a[0]*a[1]*a[2]*a[3]/9
    lam1 = sqrt((beta + sqrt(beta**2 - 4*gama)) / 2)
    lam2 = sqrt((beta - sqrt(beta**2 - 4*gama)) / 2)

    def f(x):
        return x**4 - beta*x**2 + gama
    
    eta = zeros((4, nlayer, nwno)) # will remain zero for thermal
    if calculation == 0: #reflected light
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

        for l in range(4):
            eta[l,:,:] = (Dels[l]/Del)

        z1pl = (eta[0]/2 + eta[1] + 5*eta[2]/8) *2*pi
        z1mn = (eta[0]/2 - eta[1] + 5*eta[2]/8) *2*pi
        z2pl = (-eta[0]/8 + 5*eta[2]/8 + eta[3])*2*pi 
        z2mn = (-eta[0]/8 + 5*eta[2]/8 - eta[3])*2*pi
    
    expo1 = slice_rav(lam1*dtau, 35.0) 
    expo2 = slice_rav(lam2*dtau, 35.0) 
    exptrm1 = exp(-expo1)
    exptrm2 = exp(-expo2)

    R1 = -a[0]/lam1; R2 = -a[0]/lam2
    Q1 = 1/2 * (a[0]*a[1]/(lam1**2) - 1); Q2 = 1/2 * (a[0]*a[1]/(lam2**2) - 1)
    S1 = -3/(2*a[3]) * (a[0]*a[1]/lam1 - lam1); S2 = -3/(2*a[3]) * (a[0]*a[1]/lam2 - lam2)
    
    p1pl = (1/2 + R1 + 5*Q1/8)  *2*pi;  
    p2pl = (1/2 + R2 + 5*Q2/8)  *2*pi;  
    q1pl = (-1/8 + 5*Q1/8 + S1) *2*pi;  
    q2pl = (-1/8 + 5*Q2/8 + S2) *2*pi;  
    p1mn = (1/2 - R1 + 5*Q1/8)  *2*pi
    p2mn = (1/2 - R2 + 5*Q2/8)  *2*pi
    q1mn = (-1/8 + 5*Q1/8 - S1) *2*pi
    q2mn = (-1/8 + 5*Q2/8 - S2) *2*pi

    f00 = p1mn*exptrm1; f01 = p1pl/exptrm1; f02 = p2mn*exptrm2; f03 = p2pl/exptrm2
    f10 = q1mn*exptrm1; f11 = q1pl/exptrm1; f12 = q2mn*exptrm2; f13 = q2pl/exptrm2
    f20 = p1pl*exptrm1; f21 = p1mn/exptrm1; f22 = p2pl*exptrm2; f23 = p2mn/exptrm2
    f30 = q1pl*exptrm1; f31 = q1mn/exptrm1; f32 = q2pl*exptrm2; f33 = q2mn/exptrm2

    if calculation == 0:# 'reflected':
        expon = exp(-slice_rav(tau/ubar0, 35.0))
        z1mn_up = z1mn * expon[1:,:]
        z2mn_up = z2mn * expon[1:,:]
        z1pl_up = z1pl * expon[1:,:]
        z2pl_up = z2pl * expon[1:,:]
        z1mn_down = z1mn * expon[:-1,:]
        z2mn_down = z2mn * expon[:-1,:]
        z1pl_down = z1pl * expon[:-1,:]
        z2pl_down = z2pl * expon[:-1,:]
    elif calculation == 1: # linear thermal
        z1mn_up = (1-w0)/a[0] * (B0/2 - B1/a[1] + B1*dtau/2) *2*pi #* 2*pi
        z2mn_up = -0.5 * (1-w0) / (4*a[0]) * (B0 + B1*dtau) *2*pi  #* 2*pi
        z1pl_up = (1-w0)/a[0] * (B0/2 + B1/a[1] + B1*dtau/2) *2*pi #* 2*pi
        z2pl_up = -0.5 * (1-w0) / (4*a[0]) * (B0 + B1*dtau) *2*pi  #* 2*pi
        z1mn_down = (1-w0)/a[0] * (B0/2 - B1/a[1]) *2*pi           #* 2*pi
        z2mn_down = -0.5 * (1-w0) / (4*a[0]) * (B0) *2*pi          #* 2*pi
        z1pl_down = (1-w0)/a[0] * (B0/2 + B1/a[1]) *2*pi           #* 2*pi
        z2pl_down = -0.5 * (1-w0) / (4*a[0]) * (B0) *2*pi          #* 2*pi


    Mb = zeros((11, 4*nlayer, nwno))
    B = zeros((4*nlayer, nwno))
    F_bot = zeros((4*nlayer, nwno))
    G_bot = zeros(nwno)
    F = zeros((4*nlevel, 4*nlayer, nwno))
    G = zeros((4*nlevel, nwno))

    # top boundary conditions
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

    # bottom boundary conditions
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
    B[4*nlayer-1,:] = b_surface_SH4 - z2pl_up[n,:] + surf_reflect*z2mn_up[n,:]

    # fill remaining rows of matrix
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

    # flux at bottom of atmosphere
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
                             
        G[0,:] = z1mn_down[0,:]
        G[1,:] = z2mn_down[0,:]
        G[2,:] = z1pl_down[0,:]
        G[3,:] = z2pl_down[0,:]
        G[4::4,:] = z1mn_up
        G[5::4,:] = z2mn_up
        G[6::4,:] = z1pl_up
        G[7::4,:] = z2pl_up

    A = np.ones((4,4,nlayer,nwno))
    #A[0,0] =  1;  A[0,1] =  1;  A[0,2] =  1; A[0,3] =   1
    A[1,0] = R1;  A[1,1] = -R1; A[1,2] = R2; A[1,3] = -R2
    A[2,0] = Q1;  A[2,1] =  Q1; A[2,2] = Q2; A[2,3] =  Q2
    A[3,0] = S1;  A[3,1] = -S1; A[3,2] = S2; A[3,3] = -S2

    return Mb, B, F_bot, G_bot, F, G, lam1, lam2, A, eta 

#@jit(nopython=True, cache=True)
def solve_4_stream_banded(M, B, stream):
    """
    Solve the Spherical Harmonics Problem

    Returns
    -------
    intgrl_new : numpy.ndarray
       Integrated source function for source function technique
    flux : numpy.ndarray
        Upwards lux at bottom of atmosphere
    X : numpy.ndarray
        Coefficients of flux/intensity matrix problem
    """
    #   find constants
    diag = int(3*stream/2 - 1)
    #with objmode(X='float64[:]'):
    X = solve_banded((diag,diag), M, B)

    return X

#@jit(nopython=True, cache=True)
def calculate_flux(F, G, X):
    """
    Calculate fluxes
    """
    return F.dot(X) + G
    #return mat_dot(F,X) + G

#@jit(nopython=True, cache=True)
def legP(mu): # Legendre polynomials
    """
    Generate array of Legendre polynomials
    """
    return np.array([1, mu, (3*mu**2 - 1)/2, (5*mu**3 - 3*mu)/2,
        (35*mu**4 - 30*mu**2 + 3)/8, 
        (63*mu**5 - 70*mu**3 + 15*mu)/8, 
        (231*mu**6 - 315*mu**4 + 105*mu**2 - 5)/16 ])

#@jit(nopython=True, cache=True)
def mat_dot(A,B):
    """
    Matrix-vector dot product
    """
    C = zeros(B.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            C[i,:] += A[i,j,:]*B[j,:]
    #C = np.dot(A,B)
    return C

#@jit(nopython=True, cache=True)
def vec_dot(A,B):
    """
    Vector-vector dot product
    """
    C = 0
    for i in range(A.shape[0]):
        C += A[i]*B[i]
    return C


def tidal_flux(T_e, wave_in,nlevel, pressure, pm, hratio, col_den):
    """
    Computes Tidal Fluxes in all levels. Py of TIDALWAVE subroutine. 
	
    Parameters
	----------
	T_e : float 
		Temperature (internal?)
	wave_in : float
		what is this?
	nlevel : int 
		# of levels
	pressure : array 
		pressure array 
	pm : float
		Some pressure (?)
    hratio : float
		Ratio of Scale Height over Chapman Scale Height
    col_den : array
        Column density array
    Returns
	-------
	Tidal Fluxes and DE/DM in ergs/g sec
    
    """

    sigma_sb = 0.56687e-4 # stefan-boltzmann constant

    tide = -sigma_sb* (T_e**4)

    T_tot= 0.0 #TTOT

    tidal=np.zeros(shape=(nlevel))

    dedm=np.zeros(shape=(nlevel-1))

    for j in range(nlevel):
        if j > 1 :
            tidal[j] = tidal[j-1] - chapman(pressure[j],pm,hratio)*col_den[j-1]
            T_tot += tidal[j] -tidal[j-1]
    
    tidal = (tidal*wave_in/T_tot) + tide - (tidal[-1]*wave_in/T_tot)
    
    #for j in range(nlevel-1):
        # dE/dM (ergs/g sec)
     #   dedm[j]= eff_g0*(tidal[j+1]-tidal[j])/(1e6*(pressure[j+1]-pressure[j]))
    
    return tidal

def chapman(pressure, pm, hratio):
    """
    Computes Chapman function for use in tidal routine
    
    Parameters
	----------
    pressure : float 
		pressure 
    pm : float
        Some pressure (?)
    hratio : float
        Ratio of Scale Height over Chapman Scale Height
    
    Returns
	-------
	Chapman function
    
    """

    chapman_func = exp(1.0+ hratio*log(pressure/pm)- (pressure/pm)**hratio) 
    return chapman_func

def set_bb_deprecate(wno,delta_wno,nwno,ntmps,dt,tmin,tmax):
    """
    Function to compute a grid of black bodies before the code runs. 
    This allows us to interpolate on a blackbody instead of computing the planck 
    function repetitively. This was done because historically computing the 
    planck function was a bottleneck in speed. 

    Parameters
    ----------
    wno : array, float 
        Wavenumber array cm-1 
    delta_wno : array, float 
        Wavenumber bins cm-1
    nwno : int 
        Number of wavenumbers (len(wno))
    ntmps : int 
        Number of temperature points to compute. Default number is set in config.json
    dt : float    
        Spacing in temperature to compute. Default number is set in config.json
    tmin : float 
        Minimum temperature to compute the grid 
    tmax : float 
        Maximum temperature to compute the grid 

    Returns 
    -------
    array
        black body grid (CGS), number of temperatures x number of wavenumbers
    array 
        spline values for interpolation, number of temperatures x number of wavenumbers
    array 
        temperature grid
    """

    bb=np.zeros(shape=(ntmps,nwno))
    tp= np.zeros(shape=(ntmps))
    y2=np.zeros(shape=(ntmps,nwno))
    for it in range(ntmps):
        temp_bb = tmin +(it)*dt
        tp[it]= temp_bb
    #GET RID OF PLACK CGS     
        for ik in range(nwno):
            x= planck_cgs_deprecate(wno[ik],temp_bb,delta_wno[ik])
            if x > 0.0 :
                bb[it,ik] = log(x)
            else:
                bb[it,ik] = -700.0
    
    dts = 0.02
    for ik in range(nwno):
        yp_n= (-bb[ntmps-1,ik]+log(planck_cgs_deprecate(wno[ik],tmax+dts,delta_wno[ik])))/dts
        yp_0 = (-bb[0,ik]+log(planck_cgs_deprecate(wno[ik],tmin+dts,delta_wno[ik])))/dts
        
        pass0=bb[:,ik]

        y2x = spline_deprecate(tp,pass0,ntmps,yp_0,yp_n)
        
        
        y2[:,ik] = y2x
    
    return bb , y2 , tp

def spline_deprecate(x , y, n, yp0, ypn):
    
    u=np.zeros(shape=(n))
    y2 = np.zeros(shape=(n))

    if yp0 > 0.99 :
        y2[0] = 0.0
        u[0] =0.0
    else:
        y2[0]=-0.5
        u[0] = (3.0/(x[1]-x[0]))*((y[1]-y[0])/(x[1]-x[0])-yp0)

    for i in range(1,n-1):
        sig=(x[i]-x[i-1])/(x[i+1]-x[i-1])
        p=sig*y2[i-1]+2.
        y2[i]=(sig-1.)/p
        u[i]=(6.0*((y[i+1]-y[i])/(x[i+1]-x[i])-(y[i]-y[i-1])/(x[i]-x[i-1]))/(x[i+1]-x[i-1])-sig*u[i-1])/p

    if ypn > 0.99 :
        qn = 0.0
        un = 0.0
    else:
        qn =0.5
        un = (3.0/(x[n-1]-x[n-2]))*(ypn-(y[n-1]-y[n-2])/(x[n-1]-x[n-2]))
    
    y2[n-1] = (un - qn*u[n-2])/(qn*y2[n-2]+1.0)

    for k in range(n-2, -1, -1):
        y2[k] = y2[k] * y2[k+1] +u[k]
    
    return y2

def planck_cgs_deprecate(wave, T , dwave):
    # PLANCK FUNCTION RETURNS B IN CGS UNITS, ERGS CM-2 WAVENUMBER-1
    # wave IS WAVENUMBER IN CM-1
    # T IS IN KELVIN
    nbb = 4

    planck_sum = 0.0

    for i in range(-nbb, nbb+1, 1):
        wavenum = wave + i*dwave/(2.0*nbb)
        planck_sum += 1.191e-5*(wavenum**3)*exp(-1.438769* wavenum/T ) / ( 1.0 - exp(-1.4387690 * wavenum/T ) )  
    planck_sum = planck_sum/(2*nbb +1.0)
    if planck_sum <= 1e-300:
        planck_sum = 1e-300
    
    return planck_sum


@jit(nopython=True, cache=True)
def planck_rad_deprecate(iw, T, dT ,  tmin, tmax, bb , y2, tp):

    if T < tmin :
       # itchx = 1
        T= tmax
    elif T > tmax :
       # itchx = 1
        T=tmax
    
    k_low = int((T-tmin)/dT)
    k_high = k_low+1
    h= dT
    a= (tp[k_high]-T)/h
    b= (T-tp[k_low])/h
    
    planck_rad = a*bb[k_low,iw]+b*bb[k_high,iw]+((a**3-a)*y2[k_low,iw]+(b**3-b)* y2[k_high,iw])*(h**2)/6.0
    
    planck_rad = exp(planck_rad)

    return planck_rad


@jit(nopython=True, cache=True)
def blackbody_climate_deprecate(wave,temp, bb, y2, tp, tmin, tmax):

    blackbody_array = np.zeros(shape=(len(temp),len(wave)))
    dT= 2.5

    for itemp in range(len(temp)):
        for iwave in range(len(wave)):
            blackbody_array[itemp, iwave] = planck_rad_deprecate(iwave, temp[itemp], dT ,  tmin, tmax, bb , y2, tp)

    return blackbody_array

# still not developed fully. virga has a function already maybe just use that
@jit(nopython=True, cache=True)
def get_kzz(pressure, temp,grav,mmw,tidal,flux_net_ir_layer, flux_plus_ir_attop,t_table, p_table, grad, cp, calc_type,nstr):

    grav_cgs = grav*1e2
    p_cgs = pressure *1e6
    
    nlevel = len(temp)
    
    
    if len(mmw) == len(temp)-1:
        r_atmos = 8.3143e7/mmw
    else:
        r_atmos = 8.3143e7/mmw[:-1]

    
    nz= nlevel -1
    p = np.zeros_like(p_cgs)
    t = np.zeros_like(p_cgs)

    for iz in range(nz-1, -1,-1):
        itop = iz
        ibot = iz+1

        dlnp = np.log(p_cgs[ibot]/p_cgs[itop])
        p[iz] = 0.5*(p_cgs[itop]+p_cgs[ibot])

        dtdlnp = (temp[itop]-temp[ibot])/dlnp

        t[iz] = temp[ibot] +np.log(p_cgs[ibot]/p[iz])*dtdlnp
        #scale_h =  r_atmos[iz]*t[iz]/grav_cgs
    
    
    # flux_plux_ir is already summed up with dwni in climate routine
    # so just add to get f_sum

    f_sum = np.sum(flux_plus_ir_attop)

    sigmab =  0.56687e-4 #cgs

    teff_now = (f_sum/sigmab)**0.25
    target_teff = (abs(tidal[0])/sigmab)**0.25
    flx_min = sigmab*((target_teff*0.05)**4)
    
    #print("Teff now ", teff_now, "Target Teff ", target_teff)

    #     we explictly assume that the bottom layer is 100%
    #     convective energy transport.  This helps with
    #     the correction logic below and should always be true
    #     in a well formed model.

    chf = np.zeros_like(tidal)

    chf[nz-1] = f_sum
    
    for iz in range(nz-1-1,-1,-1):
        chf[iz] = f_sum - flux_net_ir_layer[iz]
        ratio_min = (1./3.)*p[iz]/p[iz+1]
        
#     set the minimum allowed heat flux in a layer by assuming some overshoot
#     the 1/3 is arbitrary, allowing convective flux to fall faster than
#     pressure scale height
        
        if chf[iz] < ratio_min*chf[iz+1]:
            chf[iz]= ratio_min*chf[iz+1]
        
#     Now we adjust so that the convective flux is equal to the3
#     target convective flux to see if this helps with the
#     convergence.
    f_target = abs(tidal[0])
    f_actual = chf[nz-1]
    
    ratio = f_target/f_actual
    
    for iz in range(nz-1,-1,-1):
        
        chf[iz] = max(chf[iz]*ratio,flx_min) 
        
    
    player, tlayer = np.zeros(len(pressure)-1), np.zeros(len(pressure)-1)
    lapse_ratio = np.zeros_like(player)
    for j in range(len(pressure)-1):
        tlayer[j]=0.5*(temp[j]+temp[j+1])
        player[j]=np.sqrt(p_cgs[j]*p_cgs[j+1]) # cgs

        dtdp = (np.log(temp[j])-np.log(temp[j+1]))/(np.log(p_cgs[j+1]/p_cgs[j]))
        tbar = 0.5*(temp[j]+temp[j+1])
        pbar = 0.5*(p_cgs[j] +p_cgs[j+1])
        # weirdly layer routine of eddysed uses did_grad with pressures in cgs
        # supposed to be used with pressure in bars
        grad_x,cp_x = did_grad_cp(tbar, pbar, t_table, p_table, grad, cp, calc_type)
        lapse_ratio[j] = min(np.array([1.0, -dtdp/grad_x]))
        
    
    
    rho_atmos = player/ (r_atmos * tlayer)
    
    c_p = (7./2.)*r_atmos
    scale_h = r_atmos * tlayer / (grav_cgs)
    
    #0.1 just to explore was not here 
    #mixl = scale_h #lapse_ratio*scale_h*1e-1
    mixl = np.zeros_like(lapse_ratio)
    for jj in range(len(pressure)-1):
        mixl[jj] = max(0.1,lapse_ratio[jj])*scale_h[jj]
    
    scalef_kz = 1./3.
    
    kz = scalef_kz * scale_h * (mixl/scale_h)**(4./3.) *( ( r_atmos*chf[:-1] ) / ( rho_atmos*c_p ) )**(1./3.)
    
    
    kz = np.append(kz,kz[-1])
    
    
    #### julien moses 2021
    
    logp = np.log10(pressure)
    wh = np.where(np.absolute(logp-(-3)) == np.min(np.absolute(logp-(-3))))
    
    kzrad1 = (5e8/np.sqrt(pressure[nstr[0]:nstr[1]]))*(scale_h[wh]/(620*1e5))*((target_teff/1450)**4)
    kzrad2 = (5e8/np.sqrt(pressure[nstr[3]:nstr[4]]))*(scale_h[wh]/(620*1e5))*((target_teff/1450)**4)
    #
    if nstr[3] != 0:
        kz[nstr[0]:nstr[1]] = kzrad1#/100 #*10#kz[nstr[0]:nstr[1]]/1.0
        kz[nstr[3]:nstr[4]] = kzrad2#/100 #*10 #kz[nstr[3]:nstr[4]]/1.0
    else:
        kz[nstr[0]:nstr[1]] = kzrad1#/100
    
    return kz


@jit(nopython=True, cache=True)
def did_grad_cp( t, p, t_table, p_table, grad, cp, calc_type):
    """
    Parameters
    ----------
    t : float
        Temperature  value
    p : float 
        Pressure value
    t_table : array 
        array of Temperature values with 53 entries
    p_table : array 
        array of Pressure value with 26 entries
    grad : array 
        array of gradients of dimension 53*26
    cp : array 
        array of cp of dimension 53*26
    calc_type : int 
        not used to make compatible with nopython. 
    
    Returns
    -------
    float 
        grad_x,cp_x
    
    """
    # Python version of DIDGRAD function in convec.f in EGP
    # This has been benchmarked with the fortran version
    
       
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

@jit(nopython=True, cache=True)
def locate(array,value):
    """
    Parameters
    ----------
    array : array
        Array to be searched.
    value : float 
        Value to be searched for.
    
    
    Returns
    -------
    int 
        location of nearest point by bisection method 
    
    """
    # this is from numerical recipes
    
    n = len(array)
    
    
    jl = 0
    ju = n
    while (ju-jl > 1):
        jm=int(0.5*(ju+jl)) 
        if (value >= array[jm]):
            jl=jm
        else:
            ju=jm
    
    if (value <= array[0]): # if value lower than first point
        jl=0
    elif (value >= array[-1]): # if value higher than first point
        jl= n-1
    
    return jl
