from numba import jit
from numpy import zeros, cos, arcsin, sin, arccos,outer,array,sum,zeros
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
		Number of angles 
	nt : int 
		Number of angles 
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
	numtx = arcsin((gangle-(cos_theta-1.0)/(cos_theta+1.0))/(2.0/(cos_theta+1)))

	f = sin(arccos(tangle)) #define to eliminate repitition
	ubar0 = outer(cos(numtx-phase_angle) , f) #ng by nt (100,10)
	ubar1 = outer(cos(numtx) , f) 

	return ubar0, ubar1, cos_theta 



def get_angles(refdata):
	"""Grabs the angles from the geometry.json file
	"""
	a=json.load(open(os.path.join(refdata,'geometry.json')))
	return array(a['gangle']),array(a['gweight']),array(a['tangle']),array(a['tweight'])

@jit(nopython=True, cache=True)
def compress_disco(numg, numt, nwno, cos_theta, xint_at_top, gweight, tweight,F0PI): 
	"""
	Last step in albedo code. Integrates over phase angle based on the 
	Gaussian-Chebychev weights in geometry.json 
	
	Parameters
	----------
	numg : int 
		Number of Gauss angles 
	numt : int 
		Number of Chebychev Angles 
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

