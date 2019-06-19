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
	cos_theta = cos(phase_angle)
	longitude = arcsin((gangle-(cos_theta-1.0)/(cos_theta+1.0))/(2.0/(cos_theta+1)))
	colatitude = arccos(tangle)#colatitude = 90-latitude 
	latitude = pi/2-colatitude
	f = sin(colatitude) #define to eliminate repitition
	ubar0 = outer(cos(longitude-phase_angle) , f) #ng by nt 
	ubar1 = outer(cos(longitude) , f) 

	return ubar0, ubar1, cos_theta ,latitude,longitude



def get_angles(num_gangle, num_tangle):
	"""Computes angles for disco ball 
	
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
	#a=json.load(open(os.path.join(refdata,'geometry.json')))
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
	albedo=zeros(nwno)
	for w in range(nwno):
		albedo[w] = sum((xint_at_top[:,:,w]*tweight).T*gweight)
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
	flux=zeros(nwno)
	for w in range(nwno):
		flux[w] = 0.5 * sum((ubar1*flux_at_top[:,:,w]*tweight).T*gweight)
	return flux
