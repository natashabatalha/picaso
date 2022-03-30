from numba import jit
from numpy import pi, zeros, cos, arcsin, sin, arccos,outer,array,sum,zeros, linspace
from numpy import polynomial
import json 
import os 

@jit(nopython=True, cache=True)
def compute_disco(ng, nt, gangle, tangle, phase_angle):
    """
    Computes ubar0, the incident angle, and ubar1, the outgoing angle from the 
    chebyshev angles in geometry.json 

    Parameters 
    ----------
    ng : int 
        Number of gauss angles 
    nt : int 
        Number of tchebyshev angles 
    gangle : float  
        Gaussian Angles 
    tangle : float 
        Chevychev Angles  
    phase_angle : float 
        Planetary phase angle 

    Returns 
    -------
    ubar0
        the incident angles
    ubar1 
        the outgoing angles 
    cos_theta 
        Cosine of the phase angle 
    """
    #this theta is defined from the frame of the downward propagating beam
    cos_theta = cos(phase_angle)
    longitude = arcsin((gangle-(cos_theta-1.0)/(cos_theta+1.0))/(2.0/(cos_theta+1)))
    colatitude = arccos(tangle)#colatitude = 90-latitude 
    latitude = pi/2-colatitude
    f = sin(colatitude) #define to eliminate repitition
    ubar0 = outer(cos(longitude-phase_angle) , f) #ng by nt 
    ubar1 = outer(cos(longitude) , f) 

    return ubar0, ubar1, cos_theta ,latitude,longitude

def get_angles_1d(ngauss):
    """Computes integration over half sphere. ngauss=5 is 5 angles over half a 
    sphere. 
    
    These are taken from : 
    Abramowitz and Stegun Table 25.8
    https://books.google.com/books?id=32HJ1orVXDEC&pg=PA876&lpg=PA876&dq=Abramowitz+and+Stegun+Table+25.8&source=bl&ots=_eKu6k5SG8&sig=ACfU3U0_piSAi8aDHOVd0itf_QSAFxcFLQ&hl=en&sa=X&ved=2ahUKEwj5neLR2fDoAhXemHIEHUSVCjEQ6AEwD3oECAwQKQ#v=onepage&q=Abramowitz%20and%20Stegun%20Table%2025.8&f=false

    Parameters
    ----------
    ngauss: int 
        Can only be 5, 6, 7, 8

    """
    #n=5 
    if ngauss ==5:
        gangle = array([0.0985350858, 0.3045357266, 0.5620251898, 0.8019865821,0.9601901429])
        gweight = array([0.0157479145, 0.0739088701, 0.1463869871,  0.1671746381,0.0967815902])

    #n=6 
    elif ngauss ==6:
        gangle = array([0.0730543287, 0.2307661380, 0.4413284812, 0.6630153097, 0.8519214003,0.9706835728])
        gweight = array([0.0087383018, 0.0439551656, 0.0986611509, 0.1407925538, 0.1355424972, 0.0723103307])

    #n=7
    elif ngauss ==7:
        gangle = array([0.0562625605, 0.1802406917, 0.3526247171, 0.5471536263, 0.7342101772, 0.8853209468, 0.9775206136])
        gweight = array([0.0052143622, 0.0274083567, 0.0663846965, 0.1071250657, 0.1273908973, 0.1105092582, 0.0559673634])

    #n=8
    elif ngauss ==8:
        gangle = array([0.0446339553, 0.1443662570, 0.2868247571, 0.4548133152, 0.6280678354, 0.7856915206, 0.9086763921, 0.9822200849])
        gweight = array([0.0032951914, 0.0178429027, 0.0454393195, 0.0791995995, 0.1060473594, 0.1125057995, 0.0911190236, 0.0445508044])
    else: 
        raise Exception("Please enter ngauss=5,6,7 or 8.")
    tangle,tweight = array([0]), array([2*pi])

    return gangle, gweight, tangle,tweight


def get_angles_3d(num_gangle, num_tangle):
    """Computes angles for full disco ball 
    
    Parameters
    ----------
    num_gangles : int 
        Number of Gauss angles 
    num_tangles : int 
        Number of Tchebychev angles desired

    Returns
    -------
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
        Gauss Angles,Gauss Weights,Tchebyshev Angles,Tchebyshev weights
    """
    #compute tangles tweights 
    i = linspace(1,num_tangle,num_tangle)
    tangle = cos(i*pi/(num_tangle + 1))
    tweight = pi/(num_tangle + 1) * sin(i*pi/(num_tangle + 1))**2.0

    #gangles and gweights 
    gangle, gweight=polynomial.legendre.leggauss(num_gangle)

    return gangle,gweight,tangle,tweight

@jit(nopython=True, cache=True)
def compress_disco( nwno, cos_theta, xint_at_top, gweight, tweight,F0PI): 
    """
    Last step in albedo code. Integrates over phase angle based on the 
    Gaussian-Chebychev weights in geometry.json 
    
    Parameters
    ----------
    nwno : int 
        Number of wavenumbers 
    cos_theta : float 
        Cosine of phase angle 
    xint_at_top : ndarray of floats 
        Planetary intensity at the top of the atmosphere with dimensions (ng, nt, nwno)
    gweight : ndarray of floats 
        Gaussian weights for integration 
    tweight : ndarray of floats 
        Chebychev weights for integration
    F0PI : ndarray of floats 
        Stellar flux 
    """
    ng = len(gweight)
    nt = len(tweight)
    albedo=zeros(nwno)
    #for w in range(nwno):
    #   albedo[w] = 0.5 * sum((xint_at_top[:,:,w]*tweight).T*gweight)
    for ig in range(ng): 
        for it in range(nt): 
            albedo = albedo + xint_at_top[ig,it,:] * gweight[ig] * tweight[it]
    albedo = 0.5 * albedo /F0PI * (cos_theta + 1.0)
    return albedo

@jit(nopython=True, cache=True)
def compress_thermal(nwno, ubar1, flux_at_top, gweight, tweight): 
    """
    Last step in albedo code. Integrates over phase angle based on the 
    Gaussian-Chebychev weights in geometry.json 
    
    Parameters
    ----------
    nwno : int 
        Number of wavenumbers 
    ubar1 : ndarray of floats 
        Outgoing angles 
    flux_at_top : ndarray of floats 
        Thermal Flux at the top of the atmosphere with dimensions (ng, nt, nwno)
    gweight : ndarray of floats 
        Gaussian weights for integration 
    tweight : ndarray of floats 
        Chebychev weights for integration
    """
    ng = len(gweight)
    nt = len(tweight)
    flux=zeros(nwno)

    fac = 1/pi 

    for ig in range(ng):
        for it in range(nt):
            flux = flux + flux_at_top[ig,it,:] * gweight[ig] * tweight[it]    

    return  fac*flux
