import cupy as cp 
#import nvtx
import math
import numba.cuda
from numba import jit, vectorize
from numpy import exp, zeros, where, sqrt, cumsum , pi, outer, sinh, cosh, min, dot, array,log, stack, ones, floor, array_equal
import numpy as np


def get_transit_1d_cupy(z, dz,nlevel, nwno, rstar, mmw, k_b,amu,
                    player, tlayer, colden, DTAU):
    """
    Routine to get the transmission spectrum using cupy

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
    #rng = nvtx.start_range(message="transfer memory", color="blue")
    #transfer data to numpy
    z = cp.asarray(z)
    dz = cp.asarray(dz)
    mmw = cp.asarray(mmw)
    player = cp.asarray(player)
    tlayer = cp.asarray(tlayer)
    colden= cp.asarray(colden)
    DTAU= cp.asarray(DTAU)
    #nvtx.end_range(rng)
    
    mmw = mmw * amu #make sure mmw in grams
    
    #rng = nvtx.start_range(message="delta_length loop", color="orange")
    #delta_length=cp.zeros((nlevel,nlevel))
    #for i in range(nlevel):
    #    for j in range(i):
    #        reference_shell = z[i]
    #        inner_shell = z[i-j]
    #        outer_shell = z[i-j-1]
    #        #this is the path length between two layers 
    #        #essentially tangent from the inner_shell and toward 
    #        #line of sight to the outer shell
    #        integrate_segment=(cp.power(cp.power(outer_shell,2)-cp.power(reference_shell,2), 0.5)-
    #                cp.power(cp.power(inner_shell,2)-cp.power(reference_shell,2), 0.5))
    #        #make sure to use the pressure and temperature  
    #        #between inner and outer shell
    #        #this is the same index as outer shell because ind = 0 is the outer-
    #        #most layer 
    #        delta_length[i,j]=integrate_segment*player[i-j-1]/tlayer[i-j-1]/k_b
    ii, jj = cp.mgrid[0:nlevel, 0:nlevel]
    integrate_segment = (
        cp.sqrt(cp.tri(nlevel, k=-1)*(z[ii-jj-1]**2 - z[ii]**2)) - 
        cp.sqrt(cp.tri(nlevel, k=-1)*(z[ii-jj]**2 - z[ii]**2))
    )
    delta_length  = integrate_segment*player[ii-jj-1]/tlayer[ii-jj-1]/k_b

    #nvtx.end_range(rng)
    
    #rng = nvtx.start_range(message="Sum TUAS", color="blue")
    #remove column density and mmw from DTAU which was calculated in 
    #optics because line of site integration is diff for transit
    #TAU = cp.zeros((nwno, nlevel-1)) #change 1: remove this loop 
    #for i in range(nwno):
    #    TAU[i,:] = DTAU[:,i]  / colden * mmw 
    TAU =  mmw * DTAU.T / colden 
    #transmitted=cp.zeros((nwno, nlevel))+1.0
    #for i in range(nlevel):
    #    TAUALL=cp.zeros(nwno)#0.
    #    for j in range(i):
    #        #two because symmetry of sphere
    #        TAUALL = TAUALL + 2*TAU[:,i-j-1]*delta_length[i,j]
    #    transmitted[:,i]=cp.exp(-TAUALL)
    transmitted = cp.exp(-cp.sum(2*TAU[:,ii-jj-1]*delta_length,axis=2))
    #nvtx.end_range(rng)
    
    #rng = nvtx.start_range(message="Dot Product", color="orange")
    F=(((min(z))/(rstar))**2 + 
        2./(rstar)**2.*cp.dot((1.-transmitted),z*dz))
    #nvtx.end_range(rng)
    return F.get()

@numba.cuda.jit()
def _get_transit_1d_cuda(z,TAU,player,tlayer,k_b,TAUALL,transmitted):
    k = numba.cuda.grid(1)
    if k < TAU.shape[0]:
        for i in range(len(z)):
            tmp = 0
            for j in range(i):
                #two because symmetry of sphere
                reference_shell = z[i]
                inner_shell =z[i-j]
                outer_shell = z[i-j-1]
                integrate_segment = (math.pow(math.pow(outer_shell,2)-math.pow(reference_shell,2),0.5)-
                                    math.pow(math.pow(inner_shell,2)-math.pow(reference_shell,2),0.5))
                delta_length = integrate_segment*player[i-j-1]/tlayer[i-j-1]/k_b
                tmp += 2.0*TAU[k,i-j-1]*delta_length
            transmitted[k, i] = math.exp(-tmp)
            
def get_transit_1d_cuda(z, dz,nlevel, nwno, rstar, mmw, k_b,amu,
                    player, tlayer, colden, DTAU): 
    """
    Routine to get the transmission spectrum using numba.cuda.jit

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
    z = cp.asarray(z)
    dz = cp.asarray(dz)
    mmw = cp.asarray(mmw)
    player = cp.asarray(player)
    tlayer = cp.asarray(tlayer)
    colden= cp.asarray(colden)
    DTAU= cp.asarray(DTAU)
    mmw = mmw * amu #make sure mmw in grams
    TAU = mmw*DTAU.T/colden
    
    transmitted=cp.zeros((nwno, nlevel))+1.0
    TAUALL =cp.zeros(nwno)
    blocksize = 32
    numblocks = (nwno + blocksize - 1) // blocksize
    _get_transit_1d_cuda[numblocks, blocksize](z,TAU,player,tlayer,k_b,TAUALL,transmitted)   #
    F=(((min(z))/(rstar))**2 + 
            2./(rstar)**2.*cp.dot((1.-transmitted),z*dz))
    return F.get()

def get_thermal_1d_cuda(nlevel, wno,nwno, numg,numt,tlevel, dtau, w0,cosb,plevel, ubar1,
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
    
    wno, tlevel, dtau, w0, cosb, plevel, ubar1, surf_reflect = (cp.asarray(wno),
                                                               cp.asarray(tlevel),
                                                               cp.asarray(dtau), 
                                                               cp.asarray(w0),
                                                               cp.asarray(cosb),
                                                               cp.asarray(plevel),
                                                               cp.asarray(ubar1),
                                                               cp.asarray(surf_reflect))
    
    nlayer = nlevel - 1 #nlayers 
    #flux_out = zeros((numg, numt, 2*nlevel, nwno))

    mu1 = 0.5#0.88#0.5 #from Table 1 Toon  

    #get matrix of blackbodies 
    all_b = blackbody_cupy(tlevel, 1/wno) #returns nlevel by nwave   
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
    exptrm = slice_gt_cupy (exptrm, 35.0) 

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
    
    A, B, C, D = setup_tri_diag_cuda(nlayer,nwno,  c_plus_up, c_minus_up, 
                        c_plus_down, c_minus_down, b_top, b_surface, surf_reflect,
                         gama, dtau, 
                        exptrm_positive,  exptrm_minus)
    
    positive, negative = tri_diag_solve_cuda(A, B, C, D,nlayer,nwno)
    
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

    int_minus = cp.zeros((nlevel,nwno))
    int_plus = cp.zeros((nlevel,nwno))
    int_minus_mdpt = cp.zeros((nlevel,nwno))
    int_plus_mdpt = cp.zeros((nlevel,nwno))
    #intensity = zeros((numg, numt, nlevel, nwno))

    exptrm_positive_mdpt = cp.exp(0.5*exptrm) 
    exptrm_minus_mdpt = 1/exptrm_positive_mdpt 

    #================ START CRAZE LOOP OVER ANGLE #================
    int_at_top = cp.zeros((numg, numt, nwno)) #get intensity 
    int_down = cp.zeros((numg, numt, nwno))

    blocksize=32
    numblocks =(nwno+blocksize-1)//blocksize

    _post_get_thermal_1d_cuda[numblocks,blocksize](nwno,numg,numt,nlayer,G,H,J,K,alpha1,alpha2,sigma1,sigma2,exptrm_positive,exptrm_minus,exptrm_positive_mdpt,exptrm_minus_mdpt,lamda,ubar1,all_b,b1,tau_top,hard_surface,dtau,int_plus,int_minus,int_plus_mdpt,int_minus_mdpt,int_at_top)

    return int_at_top.get()    

@numba.cuda.jit()
def _tri_diag_solve_cuda( a, b, c, d,AS,DS,XK):
    k=numba.cuda.grid(1)
    l = a.shape[0]
    if k < b.shape[1]:#loop over wavelenghts
    #for k in range(b.shape[1]):#range(2):#
        for i in range(l-2, -1, -1):
            #print(k,i,b[i,k] , c[i,k])#wave,layer
            x = 1.0 / (b[i,k] - c[i,k] * AS[i+1,k])
            AS[i,k] = a[i,k] * x
            DS[i,k] = (d[i,k]-c[i,k] * DS[i+1,k]) * x
        XK[0,k] = DS[0,k]
        for i in range(1,l):
            XK[i,k] = DS[i,k] - AS[i,k] * XK[i-1,k]

def tri_diag_solve_cuda(a,b,c,d,l,nwno):
    L=2*l
    AS,DS,XK = cp.zeros((L,nwno)),cp.zeros((L,nwno)),cp.zeros((L,nwno))
    a,b,c,d =cp.asarray(a),cp.asarray(b),cp.asarray(c),cp.asarray(d)
    AS[-1,:] = a[-1,:]/b[-1,:]
    DS[-1,:] = d[-1,:]/b[-1,:]
    
    blocksize=32
    numblocks =(nwno+blocksize-1)//blocksize
    _tri_diag_solve_cuda[numblocks,blocksize](a,b,c,d,AS,DS,XK)#
    positive = XK[::2,:] + XK[1::2,:] #Y1+Y2 in toon (table 3)
    negative = XK[::2,:] - XK[1::2,:] #Y1-Y2 in toon (table 3)    
    return positive,negative

def slice_gt(array,lim):
    """Funciton to replace values with upper or lower limit
    Identical without decorator
    """
    shape = array.shape
    new = array.ravel()
    new[np.where(new>lim)] = lim 
    return new.reshape(shape)

def blackbody(t,w):
    """
    Blackbody flux in cgs units in per unit wavelength
    Identical to fluxes.py CPU without decorator

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

    return ((2.0*h*c**2.0)/(w**5.0))*(1.0/(np.exp((h*c)/np.outer(t, w*k)) - 1.0))

def setup_tri_diag_cuda(nlayer,nwno ,c_plus_up, c_minus_up, 
    c_plus_down, c_minus_down, b_top, b_surface, surf_reflect,
    gama, dtau, exptrm_positive,  exptrm_minus):

    L = 2 * nlayer
    #c_plus_up, c_minus_up, c_plus_down, c_minus_down = cp.asarray(c_plus_up),cp.asarray(c_minus_up),cp.asarray(c_plus_down), cp.asarray(c_minus_down)

    #b_top, b_surface, surf_reflect,gama, dtau = cp.asarray(b_top), cp.asarray(b_surface), cp.asarray(surf_reflect),  cp.asarray(gama), cp.asarray(dtau)
    
    #exptrm_positive,  exptrm_minus = cp.asarray(exptrm_positive), cp.asarray(exptrm_minus)

    A = cp.zeros((L,nwno)) 
    B = cp.zeros((L,nwno )) 
    C = cp.zeros((L,nwno )) 
    D = cp.zeros((L,nwno )) 
    e1,e2,e3,e4 = cp.zeros((nlayer,nwno )) , cp.zeros((nlayer,nwno )) , cp.zeros((nlayer,nwno )) , cp.zeros((nlayer,nwno )) 

    blocksize=32
    numblocks =(nwno+blocksize-1)//blocksize

    _setup_tri_diag_cuda[numblocks,blocksize](L,nlayer,nwno ,c_plus_up, c_minus_up, 
    c_plus_down, c_minus_down, b_top, b_surface, surf_reflect,
    gama, dtau, exptrm_positive,  exptrm_minus,e1,e2,e3,e4,A,B,C,D)#
    
    return A, B, C, D

@numba.cuda.jit()
def _setup_tri_diag_cuda( L,nlayer,nwno ,c_plus_up, c_minus_up, 
    c_plus_down, c_minus_down, b_top, b_surface, surf_reflect,
    gama, dtau, exptrm_positive,  exptrm_minus,e1,e2,e3,e4,A,B,C,D):
    
    k=numba.cuda.grid(1)
    #loop over wavelenghts
    
    
    if k < nwno:
        #EQN 44 
        #now build terms 
        for n in range(nlayer): 
            e1[n,k] = exptrm_positive[n,k] + gama[n,k]*exptrm_minus[n,k]
            e2[n,k] = exptrm_positive[n,k] - gama[n,k]*exptrm_minus[n,k]
            e3[n,k] = gama[n,k]*exptrm_positive[n,k] + exptrm_minus[n,k]
            e4[n,k] = gama[n,k]*exptrm_positive[n,k] - exptrm_minus[n,k]
        A[0,k] = 0.0
        B[0,k] = gama[0,k] + 1.0
        C[0,k] = gama[0,k] - 1.0
        D[0,k] = b_top[k] - c_minus_up[0,k]

        #even terms, not including the last !CMM1 = UP
        ilayer = 0 
        for n in range(1,L-1,2): #equiv of A[1::2,k][:-1]
            A[n,k] = (e1[ilayer,k]+e3[ilayer,k]) * (gama[ilayer+1,k]-1.0) #always good
            B[n,k] = (e2[ilayer,k]+e4[ilayer,k]) * (gama[ilayer+1,k]-1.0)
            C[n,k] = 2.0 * (1.0-gama[ilayer+1,k]**2) #always good 
            D[n,k] =((gama[ilayer+1,k]-1.0)*(c_plus_up[ilayer+1,k] - c_plus_down[ilayer,k]) + 
                                (1.0-gama[ilayer+1,k])*(c_minus_down[ilayer,k] - c_minus_up[ilayer+1,k]))
            ilayer += 1
            
        #odd terms, not including the first 
        ilayer = 0 
        for n in range(2,L,2):#equiv of A[::2,k][1:]
            A[n,k] = 2.0*(1.0-gama[ilayer,k]**2)
            B[n,k] = (e1[ilayer,k]-e3[ilayer,k]) * (gama[ilayer+1,k]+1.0)
            C[n,k] = (e1[ilayer,k]+e3[ilayer,k]) * (gama[ilayer+1,k]-1.0)
            D[n,k] = (e3[ilayer,k]*(c_plus_up[ilayer+1,k] - c_plus_down[ilayer,k]) + 
                                    e1[ilayer,k]*(c_minus_down[ilayer,k] - c_minus_up[ilayer+1,k]))
            ilayer += 1
            
        #last term [L-1]
        A[-1,k] = e1[-1,k]-surf_reflect[k]*e3[-1,k]
        B[-1,k] = e2[-1,k]-surf_reflect[k]*e4[-1,k]
        C[-1,k] = 0.0
        D[-1,k] = b_surface[k]-c_plus_down[-1,k] + surf_reflect[k]*c_minus_down[-1,k]

@numba.cuda.jit()
def _post_get_thermal_1d_cuda(nwno,numg,numt,nlayer,G,H,J,K,alpha1,alpha2,sigma1,sigma2,exptrm_positive,exptrm_minus,exptrm_positive_mdpt,exptrm_minus_mdpt,lamda,ubar1,all_b,b1,tau_top,hard_surface,dtau,int_plus,int_minus,int_plus_mdpt,int_minus_mdpt,int_at_top):
    k=numba.cuda.grid(1)
    if k < nwno:
        for ng in range(numg):
            for nt in range(numt): 
                #flux_out[ng,nt,:,:] = flux

                iubar = ubar1[ng,nt]

                #intensity boundary conditions
                if hard_surface:
                    int_plus[-1,k] = all_b[-1,k] *2*pi  # terrestrial flux /pi = intensity
                else:
                    int_plus[-1,k] = ( all_b[-1,k] + b1[-1,k] * iubar)*2*pi #no hard surface   

                int_minus[0,k] =  (1 - math.exp(-tau_top[k] / iubar)) * all_b[0,k] *2*pi
                


                for itop in range(nlayer):
                    exptrm_angle_itop = math.exp( - dtau[itop,k] / iubar)
                    exptrm_angle_mdpt_itop = math.exp( -0.5 * dtau[itop,k] / iubar) 
                    
                    #disbanning this for now because we dont need it in the thermal emission code
                    #EQN 56,toon
                    int_minus[itop+1,k]=(int_minus[itop,k]*exptrm_angle_itop+
                                            (J[itop,k]/(lamda[itop,k]*iubar+1.0))*(exptrm_positive[itop,k]-exptrm_angle_itop)+
                                            (K[itop,k]/(lamda[itop,k]*iubar-1.0))*(exptrm_angle_itop-exptrm_minus[itop,k])+
                                            sigma1[itop,k]*(1.-exptrm_angle_itop)+
                                            sigma2[itop,k]*(iubar*exptrm_angle_itop+dtau[itop,k]-iubar) )

                    int_minus_mdpt[itop,k]=(int_minus[itop,k]*exptrm_angle_mdpt_itop+
                                            (J[itop,k]/(lamda[itop,k]*iubar+1.0))*(exptrm_positive_mdpt[itop,k]-exptrm_angle_mdpt_itop)+
                                            (K[itop,k]/(-lamda[itop,k]*iubar+1.0))*(exptrm_minus_mdpt[itop,k]-exptrm_angle_mdpt_itop)+
                                            sigma1[itop,k]*(1.-exptrm_angle_mdpt_itop)+
                                            sigma2[itop,k]*(iubar*exptrm_angle_mdpt_itop+0.5*dtau[itop,k]-iubar))

                    ibot=nlayer-1-itop
                    #EQN 55,toon
                    exptrm_angle_ibot = math.exp( - dtau[ibot,k] / iubar)
                    exptrm_angle_mdpt_ibot = math.exp( -0.5 * dtau[ibot,k] / iubar)
                    int_plus[ibot,k]=(int_plus[ibot+1,k]*exptrm_angle_ibot+
                                        (G[ibot,k]/(lamda[ibot,k]*iubar-1.0))*(exptrm_positive[ibot,k]*exptrm_angle_ibot-1.0)+
                                        (H[ibot,k]/(lamda[ibot,k]*iubar+1.0))*(1.0-exptrm_minus[ibot,k] * exptrm_angle_ibot)+
                                        alpha1[ibot,k]*(1.-exptrm_angle_ibot)+
                                        alpha2[ibot,k]*(iubar-(dtau[ibot,k]+iubar)*exptrm_angle_ibot) )

                    int_plus_mdpt[ibot,k]=(int_plus[ibot+1,k]*exptrm_angle_mdpt_ibot+
                                            (G[ibot,k]/(lamda[ibot,k]*iubar-1.0))*(exptrm_positive[ibot,k]*exptrm_angle_mdpt_ibot-exptrm_positive_mdpt[ibot,k])-
                                            (H[ibot,k]/(lamda[ibot,k]*iubar+1.0))*(exptrm_minus[ibot,k]*exptrm_angle_mdpt_ibot-exptrm_minus_mdpt[ibot,k])+
                                            alpha1[ibot,k]*(1.-exptrm_angle_mdpt_ibot)+
                                            alpha2[ibot,k]*(iubar+0.5*dtau[ibot,k]-(dtau[ibot,k]+iubar)*exptrm_angle_mdpt_ibot)  )

                int_at_top[ng,nt,k] = int_plus_mdpt[0,k] #nlevel by nwno 

@jit(nopython=True, cache=True)
def testing_pre_get_thermal_1d(nlevel, wno,nwno, numg,numt,tlevel, dtau, w0,cosb,plevel, ubar1,
    surf_reflect, hard_surface, tridiagonal):
    nlayer = nlevel - 1 #nlayers 

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

    A, B, C, D = setup_tri_diag(nlayer,nwno,  c_plus_up, c_minus_up, 
                        c_plus_down, c_minus_down, b_top, b_surface, surf_reflect,
                         gama, dtau, 
                        exptrm_positive,  exptrm_minus) 

    return (A, B, C, D,nlayer, nwno),(exptrm_positive , exptrm_minus, gama, c_plus_up,
    mu1, lamda, g1_plus_g2, b0, b1,nlevel,nwno,exptrm,numg, numt,
    all_b, ubar1,dtau,hard_surface,tau_top,nlayer)

@jit(nopython=True, cache=True)
def testing_tri_get_thermal_1d(A, B, C, D,nlayer, nwno):

    positive = zeros((nlayer, nwno))
    negative = zeros((nlayer, nwno))

    #========================= Start loop over wavelength =========================
    L = nlayer+nlayer
    for w in range(nwno):
        #coefficient of posive and negative exponential terms 
        X = tri_diag_solve(L, A[:,w], B[:,w], C[:,w], D[:,w])
        #unmix the coefficients
        positive[:,w] = X[::2] + X[1::2] #Y1+Y2 in toon (table 3)
        negative[:,w] = X[::2] - X[1::2] #Y1-Y2 in toon (table 3)

    return positive, negative

@jit(nopython=True, cache=True)
def testing_post_get_thermal_1d(pre_out,tri_out):

    (exptrm_positive , exptrm_minus, gama, c_plus_up,
        mu1, lamda, g1_plus_g2, b0, b1,nlevel,nwno,exptrm,numg, numt,
        all_b, ubar1,dtau,hard_surface,tau_top,nlayer) = pre_out
    positive,negative = tri_out

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

def deprecate_post_get_thermal_1d_cuda(pre_out,tri_out):

    (exptrm_positive , exptrm_minus, gama, c_plus_up,
        mu1, lamda, g1_plus_g2, b0, b1,nlevel,nwno,exptrm,numg, numt,
        all_b, ubar1,dtau,hard_surface,tau_top,nlayer) = pre_out
    positive,negative = tri_out

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

    int_minus = cp.zeros((nlevel,nwno))
    int_plus = cp.zeros((nlevel,nwno))
    int_minus_mdpt = cp.zeros((nlevel,nwno))
    int_plus_mdpt = cp.zeros((nlevel,nwno))
    #intensity = zeros((numg, numt, nlevel, nwno))

    exptrm_positive_mdpt = cp.exp(0.5*exptrm) 
    exptrm_minus_mdpt = 1/exptrm_positive_mdpt 

    #================ START CRAZE LOOP OVER ANGLE #================
    int_at_top = cp.zeros((numg, numt, nwno)) #get intensity 
    int_down = cp.zeros((numg, numt, nwno))

    blocksize=32
    numblocks =(nwno+blocksize-1)//blocksize

    _post_get_thermal_1d_cuda[numblocks,blocksize](nwno,numg,numt,nlayer,G,H,J,K,alpha1,alpha2,sigma1,sigma2,exptrm_positive,exptrm_minus,exptrm_positive_mdpt,exptrm_minus_mdpt,lamda,ubar1,all_b,b1,tau_top,hard_surface,dtau,int_plus,int_minus,int_plus_mdpt,int_minus_mdpt,int_at_top)

    return int_at_top.get() #, intensity, flux_out #, int_down# numg x numt x nwno

def deprecate_pre_get_thermal_1d_cuda(nlevel, wno,nwno, numg,numt,tlevel, dtau, w0,cosb,plevel, ubar1,
    surf_reflect, hard_surface, tridiagonal):
    
    wno, tlevel, dtau, w0, cosb, plevel, ubar1, surf_reflect = (cp.asarray(wno),
                                                               cp.asarray(tlevel),
                                                               cp.asarray(dtau), 
                                                               cp.asarray(w0),
                                                               cp.asarray(cosb),
                                                               cp.asarray(plevel),
                                                               cp.asarray(ubar1),
                                                               cp.asarray(surf_reflect))
    
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
    
    A, B, C, D = setup_tri_diag_cuda(nlayer,nwno,  c_plus_up, c_minus_up, 
                        c_plus_down, c_minus_down, b_top, b_surface, surf_reflect,
                         gama, dtau, 
                        exptrm_positive,  exptrm_minus)
    
    #positive, negative = tri_diag_solve_cuda(A, B, C, D,nlayer,nwno)
    return (A, B, C, D,nlayer, nwno),(exptrm_positive , exptrm_minus, gama, c_plus_up,
    mu1, lamda, g1_plus_g2, b0, b1,nlevel,nwno,exptrm,numg, numt,
    all_b, ubar1,dtau,hard_surface,tau_top,nlayer)



