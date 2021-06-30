import numpy as np
from numpy import exp, zeros, where, sqrt, cumsum , pi, outer, sinh, cosh, min, dot, array,log


def flux_get_reflected_1d(nlevel, wno,nwno, numg,numt, dtau, tau, w0, cosb,gcos2, ftau_cld, ftau_ray,
    dtau_og, tau_og, w0_og, cosb_og, 
    surf_reflect,ubar0, ubar1,cos_theta, F0PI,single_phase, multi_phase,
    frac_a, frac_b, frac_c, constant_back, constant_forward, tridiagonal, calc_type):
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
    calc_type : int
        0 for Outgoing Intensity on Top, 1 for layer and level upwelling and downwelling fluxes
        
    Returns
    -------
    intensity at the top of the atmosphere for all the different ubar1 and ubar2 if calc_type is 0 and 
          layer and level upwelling and downwelling fluxes if calc_type is 1
    To Do
    -----
    - F0PI Solar flux shouldn't always be 1.. Follow up to make sure that this isn't a bad 
          hardwiring to solar, despite "relative albedo"
    """
    #what we want : intensity at the top as a function of all the different angles

    xint_at_top = zeros((numg, numt, nwno))

    nlayer = nlevel - 1 
    
    # creating the output arrays
    if calc_type == 0:
        xint_at_top = zeros((numg, numt, nlevel, nwno))
    
    elif calc_type == 1:

        flux_minus_all = zeros((numg, numt,nlevel, nwno)) ## level downwelling fluxes
        flux_plus_all = zeros((numg, numt, nlevel, nwno)) ## level upwelling fluxes
        flux_minus_midpt_all = zeros((numg, numt, nlayer, nwno)) ##  layer downwelling fluxes
        flux_plus_midpt_all = zeros((numg, numt, nlayer, nwno))  ## layer upwelling fluxes

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
            # diverging the two options here because calculating layer+level fluxes stuff can add time  
            # because this function is often used in retrievals

            if calc_type == 0 : # will just give outgoing flux at top
                
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
            
            
            elif calc_type == 1 :   # start flux calculation for climate part

      
                #  BC for level calculations
                #  downward BC at top
                flux_0_minus  = positive[0,:]*gama[0,:] + negative[0,:] + c_minus_up[0,:] # upper BC for downwelling

                # upward BC at bottom
                flux_n_plus  = positive[-1,:]*exptrm_positive[-1,:] + gama[-1,:]*negative[-1,:]*exptrm_minus[-1,:] + c_plus_down[-1,:]

                # add in flux due to the Downward incident solar radiation
                flux_0_minus+=ubar0[ng, nt]*F0PI*exp(-tau[0,:]/ubar0[ng, nt])
                



                # now BCs for the midpoint calculations

                exptrm_positive_midpt = exp(0.5*exptrm) #EP
                exptrm_minus_midpt = 1.0/exptrm_positive_midpt#EM

                taumid=tau[:-1]+0.5*dtau
                taumid_og=tau[:-1]+0.5*dtau_og

                x = exp(-taumid/ubar0[ng, nt])
                c_plus_mid= a_plus*x
                c_minus_mid=a_minus*x
                
                # midpt downward BC at top
                flux_0_minus_midpt= gama[0,:]*positive[0,:]*exptrm_positive_midpt[0,:] + negative[0,:]*exptrm_minus_midpt[0,:] + c_minus_mid[0,:]
                
                # midpt upward BC at bottom
                flux_n_plus_midpt= positive[-1,:]*exptrm_positive_midpt[-1,:] + gama[-1,:]*negative[-1,:]*exptrm_minus_midpt[-1,:] + c_plus_mid[-1,:]
                
                # add in flux due to the Downward incident solar radiation
                flux_0_minus_midpt += ubar0[ng, nt]*F0PI*exp(-taumid[0,:]/ubar0[ng, nt])


                # going to level and layer intensities

                # starting layer and level intensity arrays for upwelling part
                xint_up = zeros((nlevel,nwno))
                xint_up_layer=zeros((nlayer,nwno))
                
                #convert fluxe BCs to intensities
                xint_up[-1,:] = flux_n_plus/pi  ## BC
                xint_up_layer[-1,:] = flux_n_plus_midpt/pi ##BC



                # starting layer and level intensity arrays for downwelling part
                xint_down=zeros((nlevel,nwno))
                xint_down_layer=zeros((nlayer,nwno))

                xint_down[0,:] = flux_0_minus/pi  ## BC
                xint_down_layer[0,:] = flux_0_minus_midpt/pi #BC



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
            # SM -- a bit confused since I thought this function was initially for the upward propagating beam till the top
            # SM -- will the phase function be different for the downward and upward post-processing?

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

                for i in range(nlayer):
                #direct beam
                    ibottom=nlayer-i-1
                    itop=i
                    xint_up[ibottom,:] =( xint_up[ibottom+1,:]*exp(-dtau[ibottom,:]/ubar1[ng,nt]) 
                        #single scattering albedo from sun beam (from ubar0 to ubar1)
                            +(w0_og[ibottom,:]*F0PI/(4.*pi))
                            *(p_single[ibottom,:])*exp(-tau_og[ibottom,:]/ubar0[ng,nt])
                            *(1. - exp(-dtau_og[ibottom,:]*(ubar0[ng,nt]+ubar1[ng,nt])
                            /(ubar0[ng,nt]*ubar1[ng,nt])))*
                            (ubar0[ng,nt]/(ubar0[ng,nt]+ubar1[ng,nt]))
                        #multiple scattering terms p_single
                            +A[ibottom,:]*(1. - exp(-dtau[ibottom,:] *(ubar0[ng,nt]+1*ubar1[ng,nt])/(ubar0[ng,nt]*ubar1[ng,nt])))*
                            (ubar0[ng,nt]/(ubar0[ng,nt]+1*ubar1[ng,nt]))
                            +G[ibottom,:]*(exp(exptrm[ibottom,:]*1-dtau[ibottom,:]/ubar1[ng,nt]) - 1.0)/(lamda[ibottom,:]*1*ubar1[ng,nt] - 1.0)
                            +H[ibottom,:]*(1. - exp(-exptrm[ibottom,:]*1-dtau[ibottom,:]/ubar1[ng,nt]))/(lamda[ibottom,:]*1*ubar1[ng,nt] + 1.0)
                            )
                    
                    xint_up_layer[ibottom,:] =( xint_up[ibottom+1,:]*exp(-0.5*dtau[ibottom,:]/ubar1[ng,nt]) 
                    #single scattering albedo from sun beam (from ubar0 to ubar1)
                            +(w0_og[ibottom,:]*F0PI/(4.*pi))
                            *(p_single[ibottom,:])*exp(-taumid_og[ibottom,:]/ubar0[ng,nt])
                            *(1. - exp(-0.5*dtau_og[ibottom,:]*(ubar0[ng,nt]+ubar1[ng,nt])
                            /(ubar0[ng,nt]*ubar1[ng,nt])))*
                            (ubar0[ng,nt]/(ubar0[ng,nt]+ubar1[ng,nt]))
                        #multiple scattering terms p_single
                            +A[ibottom,:]*(1. - exp(-0.5*dtau[ibottom,:] *(ubar0[ng,nt]+1*ubar1[ng,nt])/(ubar0[ng,nt]*ubar1[ng,nt])))*
                            (ubar0[ng,nt]/(ubar0[ng,nt]+1*ubar1[ng,nt]))
                            +G[ibottom,:]*(exp(0.5*exptrm[ibottom,:]*1-0.5*dtau[ibottom,:]/ubar1[ng,nt]) - 1.0)/(lamda[ibottom,:]*1*ubar1[ng,nt] - 1.0)
                            +H[ibottom,:]*(1. - exp(-0.5*exptrm[ibottom,:]*1-0.5*dtau[ibottom,:]/ubar1[ng,nt]))/(lamda[ibottom,:]*1*ubar1[ng,nt] + 1.0)
                            )


                  
                    xint_down[itop+1,:] =( xint_down[itop,:]*exp(-dtau[itop,:]/ubar1[ng,nt]) 
                        #single scattering albedo from sun beam (from ubar0 to ubar1)
                            +(w0_og[itop,:]*F0PI/(4.*pi))
                            *(p_single[itop,:])*exp(-tau_og[itop,:]/ubar0[ng,nt])
                            *(1. - exp(-dtau_og[itop,:]*(ubar0[ng,nt]+ubar1[ng,nt])
                            /(ubar0[ng,nt]*ubar1[ng,nt])))*
                            (ubar0[ng,nt]/(ubar0[ng,nt]+ubar1[ng,nt]))
                            #multiple scattering terms p_single
                            +A[itop,:]*(1. - exp(-dtau[itop,:] *(ubar0[ng,nt]+1*ubar1[ng,nt])/(ubar0[ng,nt]*ubar1[ng,nt])))*
                            (ubar0[ng,nt]/(ubar0[ng,nt]+1*ubar1[ng,nt]))
                            +G[itop,:]*(exp(exptrm[itop,:]*1-dtau[itop,:]/ubar1[ng,nt]) - 1.0)/(lamda[itop,:]*1*ubar1[ng,nt] - 1.0)
                            +H[itop,:]*(1. - exp(-exptrm[itop,:]*1-dtau[itop,:]/ubar1[ng,nt]))/(lamda[itop,:]*1*ubar1[ng,nt] + 1.0)
                            )
                    #print(itop+1)
                    xint_down_layer[itop,:] =( xint_down[itop,:]*exp(-0.5*dtau[itop,:]/ubar1[ng,nt]) 
                        #single scattering albedo from sun beam (from ubar0 to ubar1)
                            +(w0_og[itop,:]*F0PI/(4.*pi))
                            *(p_single[itop,:])*exp(-taumid_og[itop,:]/ubar0[ng,nt])
                            *(1. - exp(-0.5*dtau_og[itop,:]*(ubar0[ng,nt]+ubar1[ng,nt])
                            /(ubar0[ng,nt]*ubar1[ng,nt])))*
                            (ubar0[ng,nt]/(ubar0[ng,nt]+ubar1[ng,nt]))
                            #multiple scattering terms p_single
                            +A[itop,:]*(1. - exp(-0.5*dtau[itop,:] *(ubar0[ng,nt]+1*ubar1[ng,nt])/(ubar0[ng,nt]*ubar1[ng,nt])))*
                            (ubar0[ng,nt]/(ubar0[ng,nt]+1*ubar1[ng,nt]))
                            +G[itop,:]*(exp(0.5*exptrm[itop,:]*1-0.5*dtau[itop,:]/ubar1[ng,nt]) - 1.0)/(lamda[itop,:]*1*ubar1[ng,nt] - 1.0)
                            +H[itop,:]*(1. - exp(-0.5*exptrm[itop,:]*1-0.5*dtau[itop,:]/ubar1[ng,nt]))/(lamda[itop,:]*1*ubar1[ng,nt] + 1.0)
                            )



                

                flux_minus_all[ng,nt,:,:]=xint_down[:,:]*pi
                flux_plus_all[ng,nt,:,:]=xint_up[:,:]*pi

                flux_minus_midpt_all[ng,nt,:,:]=xint_down_layer[:,:]*pi
                flux_plus_midpt_all[ng,nt,:,:]=xint_up_layer[:,:]*pi

            
    if calc_type == 0:        
        return xint_at_top
    elif calc_type == 1:
        return flux_minus_all, flux_plus_all, flux_minus_midpt_all, flux_plus_midpt_all   



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

def get_reflected_1d_original(nlevel, wno,nwno, numg,numt, dtau, tau, w0, cosb,gcos2, ftau_cld, ftau_ray,
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

    xint_at_top = zeros((numg, numt, nlevel, nwno))

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

            xint_at_top[ng,nt,:,:] = xint[:,:]*pi

    return xint_at_top