import numpy as np
from numpy import exp, zeros, where, sqrt, cumsum , pi, outer, sinh, cosh, min, dot, array,log

def get_reflected_1d_gfluxv(nlevel, wno,nwno, numg,numt, dtau, tau, w0, cosb,
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
    #what we want : intensity at the top as a function of all the different angles

    
    #################### --SM-- SAME AS GFLUXV

    nlayer = nlevel - 1 
    
    ################################################
    #################### --SM-- DELTA eddington options are not here-- adding them
    ################################################

    
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
    flux_minus_all = zeros((numg, numt,nlevel, nwno)) ## --SM-- level downwelling fluxes
    flux_plus_all = zeros((numg, numt, nlevel, nwno)) ## --SM-- level upwelling fluxes
    flux_minus_midpt_all = zeros((numg, numt, nlayer, nwno)) ## --SM-- layer downwelling fluxes
    flux_plus_midpt_all = zeros((numg, numt, nlayer, nwno))  ## --SM-- layer upwelling fluxes

    #now define terms of Toon et al 1989 quadrature Table 1 
    #https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/JD094iD13p16287
    #see table of terms 
    
    
    
    
    #terms not dependent on incident angle
    ################################################
    #################### --SM-- ALPHA term not defined or used here
    ################################################
    alpha= sqrt((1.-w0)/(1.-w0*cosb))    ################## --SM-- I have added here
    sq3 = sqrt(3.)
    g1  = (sq3*0.5)*(2. - w0*(1.+cosb)) #table 1 # (7-w0*(4+3*cosb))/4 #
    g2  = (sq3*w0*0.5)*(1.-cosb)        #table 1 # -(1-w0*(4-3*cosb))/4 #
    lamda = sqrt(g1**2 - g2**2)         #eqn 21
    gama  = (g1-lamda)/g2               #eqn 22

    #================ START CRAZE LOOP OVER ANGLE #================
    for ng in range(numg):
        for nt in range(numt):
    ################################################        
    ##################### --SM-- ubar0 looks like just a number in fortran here it is gauss and chebyshev dependant variable       
    ################################################
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
    ################################################        
    ##################### --SM-- b_top, b_surface and surf_reflect are taken as inputs into the routine
    ##################### --SM-- changing these just commenting out and taking as inputs to the routine) 
    ################################################

            #boundary conditions 
         ##   b_top = 0.                                      

         ##   b_surface = 0. + surf_reflect*ubar0[ng, nt]*F0PI*exp(-tau[-1, :]/ubar0[ng, nt])
    ################################################        
    ##################### --SM-- Next we go to the DSOLVER part-- go to setup_tri_diag part
    ################################################
            #Now we need the terms for the tridiagonal rotated layered method
            
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
    ################################################        
    ##################### --SM-- Next we go to the DTRIDGL = tri_diag_solve
    ################################################
                #coefficient of posive and negative exponential terms 
    
                if tridiagonal==0:
                    X = tri_diag_solve(L, A[:,w], B[:,w], C[:,w], D[:,w])
                    #unmix the coefficients
                    positive[:,w] = X[::2] + X[1::2] 
                    negative[:,w] = X[::2] - X[1::2]
    ################################################        
    ##################### --SM-- UPTO this point done-- looks like things change after this
    ################################################

                #else: 
                #   X = pent_diag_solve(L, A_[:,w], B_[:,w], C_[:,w], D_[:,w], E_[:,w], F_[:,w])
                    #unmix the coefficients
                #   positive[:,w] = exptrm_minus[:,w] * (X[::2] + X[1::2])
                #   negative[:,w] = X[::2] - X[1::2]
            flux_minus=np.zeros((nlevel,nwno))
            flux_plus=np.zeros((nlevel,nwno))
            #========================= End loop over wavelength =========================
    ################################################        
    ##################### --SM-- code lacks up and downstream treatment adding that in
    ################################################
            #use expression for bottom flux to get the flux_plus and flux_minus at last
            #bottom layer
            flux_minus[:-1, :]  = positive*gama + negative + c_minus_up
            flux_plus[:-1, :]  = positive + gama*negative + c_plus_up
            
            flux_zero_minus  = gama[-1,:]*positive[-1,:]*exptrm_positive[-1,:] + negative[-1,:]*exptrm_minus[-1,:] + c_minus_down[-1,:]
            flux_zero_plus  = positive[-1,:]*exptrm_positive[-1,:] + gama[-1,:]*negative[-1,:]*exptrm_minus[-1,:] + c_plus_down[-1,:]
            
            flux_minus[-1, :], flux_plus[-1, :] = flux_zero_minus, flux_zero_plus 
            
            flux_minus = flux_minus + ubar0[ng, nt]*F0PI*exp(-tau/ubar0[ng, nt])

    ################################################        
    ##################### --SM-- now the midpoint calculations
    ################################################        
            exptrm_positive_midpt = exp(0.5*exptrm) #EP
            exptrm_minus_midpt = 1.0/exptrm_positive_midpt#EM
            
            taumid=tau[:-1]+0.5*dtau
            x = exp(-taumid/ubar0[ng, nt])
            c_plus_mid= a_plus*x
            c_minus_mid=a_minus*x

            flux_minus_midpt= gama*positive*exptrm_positive_midpt + negative*exptrm_minus_midpt + c_minus_mid
            flux_plus_midpt= positive*exptrm_positive_midpt + gama*negative*exptrm_minus_midpt + c_plus_mid

            flux_minus_midpt = flux_minus_midpt + ubar0[ng, nt]*F0PI*exp(-taumid/ubar0[ng, nt])

            flux_minus_all[ng, nt, :, :]=flux_minus
            flux_plus_all[ng, nt, :, :]=flux_plus
            

            flux_minus_midpt_all[ng, nt, :, :]=flux_minus_midpt
            flux_plus_midpt_all[ng, nt, :, :]=flux_plus_midpt
    return flux_minus_all, flux_plus_all, flux_minus_midpt_all, flux_plus_midpt_all 


######## Done till here

           
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

    ################################################        
    ##################### --SM-- From here follow dsolver routine
    ################################################
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
    ################################################        
    ##################### --SM-- Exactly same
    ################################################
    return A, B, C, D

def slice_gt(array, lim):
    """Funciton to replace values with upper or lower limit
    """
    for i in range(array.shape[0]):
        new = array[i,:] 
        new[where(new>lim)] = lim
        array[i,:] = new     
    return array


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
    