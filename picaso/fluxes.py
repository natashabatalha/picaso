from numba import jit
import numpy as np
import copy

#@jit(nopython=True, cache=True)
def get_flux_toon(nlevel, wno, nwno, dtau_DTDEL, wbar_WDEL, cosb_CDEL, surf_reflect, ubar0, F0PI):
	"""

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
	wbar : ndarray of float 
		This is the single scattering albedo, from scattering, clouds, raman, etc 
		Dimensions=# layer by # wave
	cosb : ndarray of float 
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

	To Do
	-----
		- Replace detla-function adjustment with better approximation (e.g. Cuzzi)
		- F0PI Solar flux shouldn't always be 1.. Follow up to make sure that this isn't a bad 
		  hardwiring to solar 


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
	w0=wbar_WDEL*(1.-cosb_CDEL**2)/(1.0-wbar_WDEL*cosb_CDEL**2)
	cosbar=cosb_CDEL/(1.+cosb_CDEL)
	dtau=dtau_DTDEL*(1.-wbar_WDEL*cosb_CDEL**2) 
	#import pickle as pk
	#pk.dump([w0, cosbar,dtau] ,open('../testing_notebooks/optc_postcorrect.pk','wb'))

	#sum up taus starting at the top, going to depth
	tau = np.zeros((nlevel, nwno))
	tau[1:,:]=numba_cumsum(copy.deepcopy(dtau))
	#now define terms of Toon et al 1989 quadrature Table 1 
	#https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/JD094iD13p16287
	#see table of terms 
	g1    = (np.sqrt(3.)*0.5)*(2. - w0*(1.+cosbar))    #table 1
	g2    = (np.sqrt(3.)*w0*0.5)*(1.-cosbar)           #table 1
	g3    = 0.5*(1.-np.sqrt(3.)*cosbar*ubar0)          #table 1
	lamda = np.sqrt(g1**2 - g2**2)                     #eqn 21
	gama  = (g1-lamda)/g2                              #eqn 22
	
	#pk.dump([g1, g2, g3, lamda, gama] ,open('../testing_notebooks/g1g2g3lamdagama.pk','wb'))

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
	c_minus_up = a_minus*np.exp(-tau[:-1,:]/ubar0) #CMM1
	c_plus_up  = a_plus*np.exp(-tau[:-1,:]/ubar0) #CPM1
	c_minus_down = a_minus*np.exp(-tau[1:,:]/ubar0) #CM
	c_plus_down  = a_plus*np.exp(-tau[1:,:]/ubar0) #CP

	#calculate exponential terms needed for the tridiagonal rotated layered method
	exptrm = lamda*dtau
	#save from overflow 
	exptrm = slice_gt (exptrm, 35.0)

	exptrm_positive = np.exp(exptrm) #EP
	exptrm_minus = np.exp(-exptrm) #EM
	import pickle as pk
	pk.dump({'AM':a_minus, 'AP':a_plus,
		'CPM1':c_plus_up, 'CMM1':c_minus_up, 'CP':c_plus_down,'CM':c_minus_down,
		'EXPTRM':exptrm,'DTAU':dtau}, open('../testing_notebooks/gflux_a_c_pm.pk','wb'))
	
	#========================= Start loop over wavelength =========================
	positive = np.zeros((nlayer, nwno))
	negative = np.zeros((nlayer, nwno))
	#import pandas as pd
	#import pickle as pk
	#testingA = pd.DataFrame(columns=wno, index=range(2*nlayer) )
	#testingB = pd.DataFrame(columns=wno, index=range(2*nlayer) )
	#testingC = pd.DataFrame(columns=wno, index=range(2*nlayer) )
	#testingD = pd.DataFrame(columns=wno, index=range(2*nlayer) )

	for w in range(nwno):
		#boundary conditions 
		b_top = 0.0                                          
		b_surface = 0. + surf_reflect*ubar0*F0PI[w]*np.exp(-tau[-1, w]/ubar0)

		#Now we need the terms for the tridiagonal rotated layered method
		A, B, C, D = setup_tri_diag(nlayer, c_plus_up[:,w], c_minus_up[:,w], 
									c_plus_down[:,w], c_minus_down[:,w], b_top, b_surface, surf_reflect,
									g1[:,w], g2[:,w], g3[:,w], lamda[:,w], gama[:,w], dtau[:,w], 
									exptrm_positive[:,w],  exptrm_minus[:,w]) 
		#testingA[wno[w]] = A
		#testingB[wno[w]] = B
		#testingC[wno[w]] = C
		#testingD[wno[w]] = D
		
		#coefficient of posive and negative exponential terms 
		X = tri_diag_solve(2*nlayer, A, B, C, D)
	
		#unmix the coefficients
		positive[:,w] = X[::2] + X[1::2] 
		negative[:,w] = X[::2] - X[1::2]
	#pk.dump([testingA, testingB, testingC, testingD], open('../testing_notebooks/GFLUX_ABCD.pk','wb'))
	#========================= End loop over wavelength =========================

	#might have to add this in to avoid numerical problems later. 
	#if len(np.where(negative[:,w]/X[::2] < 1e-30)) >0 , print(negative[:,w],X[::2],negative[:,w]/X[::2])

	#evaluate the fluxes through the layers 
	#use the top optical depth expression to evaluate fp and fm 
	#at the top of each layer 
	flux_plus  = np.zeros((nlevel, nwno))
	flux_minus = np.zeros((nlevel, nwno))
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
	flux_minus = flux_minus + ubar0*F0PI*np.exp(-1.0*tau/ubar0)

	#now calculate the fluxes at the midpoints of the layers 
	exptrm_positive_mdpt = np.exp(0.5*exptrm) #EP_mdpt
	exptrm_minus_mdpt = np.exp(-0.5*exptrm) #EM_mdpt

	tau_mdpt = tau[:-1] + 0.5*dtau #start from bottom up to define midpoints 
	c_plus_mdpt = a_plus*np.exp(-tau_mdpt/ubar0)
	c_minus_mdpt = a_minus*np.exp(-tau_mdpt/ubar0)
	
	flux_plus_mdpt = positive*exptrm_positive_mdpt + gama*negative*exptrm_minus_mdpt + c_plus_mdpt
	flux_minus_mdpt = positive*exptrm_positive_mdpt*gama + negative*exptrm_minus_mdpt + c_minus_mdpt

	#add direct flux to downwelling term 
	flux_minus_mdpt = flux_minus_mdpt + ubar0*F0PI*np.exp(-1.0*tau_mdpt/ubar0)

	#finally, get cumulative fluxes 
	flux_plus_net  = numba_cumsum(flux_plus)
	flux_minus_net = numba_cumsum(flux_minus)

	return flux_plus_net, flux_minus_net 

@jit(nopython=True, cache=True)
def slice_eq(array, lim, value):
	"""Funciton to replace values with upper or lower limit
	"""
	for i in range(array.shape[0]):
		new = array[i,:] 
		new[np.where(new==lim)] = value
		array[i,:] = new     
	return array

@jit(nopython=True, cache=True)
def slice_lt(array, lim):
	"""Funciton to replace values with upper or lower limit
	"""
	for i in range(array.shape[0]):
		new = array[i,:] 
		new[np.where(new<lim)] = lim
		array[i,:] = new     
	return array

@jit(nopython=True, cache=True)
def slice_gt(array, lim):
	"""Funciton to replace values with upper or lower limit
	"""
	for i in range(array.shape[0]):
		new = array[i,:] 
		new[np.where(new>lim)] = lim
		array[i,:] = new     
	return array

@jit(nopython=True, cache=True)
def numba_cumsum(mat):
	"""Function to compute cumsum along axis=0 to bypass numba not allowing kwargs in 
	cumsum 
	"""
	for i in range(mat.shape[1]):
		mat[:,i] = np.cumsum(mat[:,i])
	return mat


@jit(nopython=True, cache=True)
def setup_tri_diag(nlayer, c_plus_up, c_minus_up, 
	c_plus_down, c_minus_down, b_top, b_surface, surf_reflect,
	g1, g2, g3, lamda, gama, dtau, exptrm_positive,  exptrm_minus):
	"""
	Before we can solve the tridiagonal matrix (See Toon+1989) section
	"SOLUTION OF THE TwO-STREAM EQUATIONS FOR MULTIPLE LAYERS", we 
	need to set up the coefficients. 

	Parameters
	----------
	nlayer : int 
		number of layers in the model 
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
	LM2 = L - 2 
	LM1 = L - 1 

	#EQN 44 

	e1 = exptrm_positive + gama*exptrm_minus
	e2 = exptrm_positive - gama*exptrm_minus
	e3 = gama*exptrm_positive + exptrm_minus
	e4 = gama*exptrm_positive - exptrm_minus

	#now build terms 
	A = np.zeros(L) 
	B = np.zeros(L) 
	C = np.zeros(L)
	D = np.zeros(L)

	A[0] = 0.0
	B[0] = gama[0] + 1.0
	C[0] = gama[0] - 1.0
	D[0] = b_top - c_minus_up[0]

	#even terms, not including the last !CMM1 = UP
	A[1::2][:-1] = (e1[:-1]+e3[:-1]) * (gama[1:]-1.0) #always good
	B[1::2][:-1] = (e2[:-1]+e4[:-1]) * (gama[1:]-1.0)
	C[1::2][:-1] = 2.0 * (1.0-gama[1:]**2)            #always good 
	D[1::2][:-1] = ((gama[1:]-1.0) * (c_plus_up[:1] - c_plus_down[:-1])
							+ (1.-gama[1:]) * (c_minus_down[:-1] - c_minus_up[1:]))

	#odd terms, not including the first 
	A[::2][1:] = 2.0*(1.0-gama[:-1]**2)
	B[::2][1:] = (e1[:-1]-e3[:-1]) * (gama[1:]+1.0)
	C[::2][1:] = (e1[:-1]+e3[:-1]) * (gama[1:]-1.0)
	D[::2][1:] = (e3[:-1]*(c_plus_up[1:] - c_plus_down[:-1]) + 
							e1[:-1]*(c_minus_down[:-1] - c_minus_up[1:]))

	#last term [L-1]
	A[-1] = e1[-1]-surf_reflect*e3[-1]
	B[-1] = e2[-1]-surf_reflect*e4[-1]
	C[-1] = 0.0
	D[-1] = b_surface-c_plus_down[-1] + surf_reflect*c_minus_down[-1]

	return A, B, C, D

@jit(nopython=True, cache=True)
def tri_diag_solve(l, a, b, c, d):
    """
    Tridiagonal Matrix Algorithm solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    Modified from : https://gist.github.com/cbellei/8ab3ab8551b8dfc8b081c518ccd9ada9
	
	A, B, C and D refer to: A(I)*X(I-1) + B(I)*X(I) + C(I)*X(I+1) = D(I)

	Solver returns X. 

	Parameters
	----------
    A : array or list 
    B : array or list 
    C : array or list 
    C : array or list 

    Returns
    --------
    array 
    	Solution, x 
    """
    nf = l # number of equations
    for it in range(1, nf):
        mc = a[it-1]/b[it-1]
        b[it] = b[it] - mc*c[it-1] 
        d[it] = d[it] - mc*d[it-1]
        	    
    xc = b
    xc[-1] = d[-1]/b[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (d[il]-c[il]*xc[il+1])/b[il]

    return xc