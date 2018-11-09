from numba import jit, vectorize
from numpy import exp, zeros, where, sqrt, cumsum , pi

@jit(nopython=True, cache=True)
def get_flux_toon(nlevel, wno, nwno, tau, dtau, w0, cosbar, surf_reflect, ubar0, F0PI):
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

	To Do
	-----
		- Replace detla-function adjustment with better approximation (e.g. Cuzzi)
		- F0PI Solar flux shouldn't always be 1.. Follow up to make sure that this isn't a bad 
		  hardwiring to solar 
	
	Examples
	--------
	flux_plus, flux_minus  = fluxes.get_flux_toon(atm.c.nlevel, wno,nwno,
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
	g1    = (sq3*0.5)*(2. - w0*(1.+cosbar))    #table 1
	g2    = (sq3*w0*0.5)*(1.-cosbar)           #table 1
	g3    = 0.5*(1.-sq3*cosbar*ubar0)          #table 1
	lamda = sqrt(g1**2 - g2**2)                     #eqn 21
	gama  = (g1-lamda)/g2                              #eqn 22
	
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
	A, B, C, D = setup_tri_diag(nlayer,nwno,  c_plus_up, c_minus_up, 
									c_plus_down, c_minus_down, b_top, b_surface, surf_reflect,
									g1, g2, g3, lamda, gama, dtau, 
									exptrm_positive,  exptrm_minus) 

	positive = zeros((nlayer, nwno))
	negative = zeros((nlayer, nwno))
	#========================= Start loop over wavelength =========================
	L = 2*nlayer
	for w in range(nwno):
		#coefficient of posive and negative exponential terms 
		X = tri_diag_solve(L, A[:,w], B[:,w], C[:,w], D[:,w])

		#unmix the coefficients
		positive[:,w] = X[::2] + X[1::2] 
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
	g1, g2, g3, lamda, gama, dtau, exptrm_positive,  exptrm_minus):
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
	C[1::2,:][:-1] = 2.0 * (1.0-gama[1:,:]**2)            #always good 
	D[1::2,:][:-1] =((gama[1:,:]-1.0)*(c_plus_up[1:,:] - c_plus_down[:-1,:]) + 
							(1.0-gama[1:,:])*(c_minus_down[:-1,:] - c_minus_up[1:,:]))
	#import pickle as pk
	#pk.dump({'GAMA_1':(gama[1:,:]-1.0), 'CPM1':c_plus_up[1:,:] , 'CP':c_plus_down[:-1,:], '1_GAMA':(1.0-gama[1:,:]), 
	#	'CM':c_minus_down[:-1,:],'CMM1':c_minus_up[1:,:],'Deven':D[1::2,:][:-1]}, open('../testing_notebooks/GFLUX_even_D_terms.pk','wb'))
	
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
def tri_diag_solve(l, a, b, c, d):
    """
    Tridiagonal Matrix Algorithm solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
	
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
def get_flux_geom_3d(nlevel, wno,nwno, numg,numt, dtau_3d, tau_3d, w0_3d, cosb_3d,gcos2_3d, 
	surf_reflect,ubar0, ubar1,cos_theta, F0PI):
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
	dtau_dedd : ndarray of float 
		This is the opacity contained within each individual layer (defined at midpoints of "levels")
		WITH D-Eddington Correction
		Dimensions=# layer by # wave
	tau_dedd : ndarray of float
		This is the cumulative summed opacity 
		WITH D-Eddington Correction
		Dimensions=# level by # wave	
	w0_dedd : ndarray of float 
		Same as w0 but with the delta eddington correction		
	cosb_dedd : ndarray of float 
		Same as cosbar buth with the delta eddington correction 
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

			g1    = (sq3*0.5)*(2. - w0*(1.+cosb))    #table 1
			g2    = (sq3*w0*0.5)*(1.-cosb)           #table 1
			lamda = sqrt(g1**2 - g2**2)              #eqn 21
			gama  = (g1-lamda)/g2                    #eqn 22
			g3    = 0.5*(1.-sq3*cosb*ubar0[ng, nt])   #table 1 #ubar is now 100x 10 matrix.. 
	
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
			A, B, C, D = setup_tri_diag(nlayer,nwno,  c_plus_up, c_minus_up, 
									c_plus_down, c_minus_down, b_top, b_surface, surf_reflect,
									g1, g2, g3, lamda, gama, dtau, 
									exptrm_positive,  exptrm_minus) 

			positive = zeros((nlayer, nwno))
			negative = zeros((nlayer, nwno))
			#========================= Start loop over wavelength =========================
			L = 2*nlayer
			for w in range(nwno):
				#coefficient of posive and negative exponential terms 
				X = tri_diag_solve(L, A[:,w], B[:,w], C[:,w], D[:,w])

				#unmix the coefficients
				positive[:,w] = X[::2] + X[1::2] 
				negative[:,w] = X[::2] - X[1::2]
			#========================= End loop over wavelength =========================

			#use expression for bottom flux to get the flux_plus and flux_minus at last
			#bottom layer
			flux_zero  = positive[-1,:]*exptrm_positive[-1,:] + gama[-1,:]*negative[-1,:]*exptrm_minus[-1,:] + c_plus_down[-1,:]
			
			xint = zeros((nlevel,nwno))
			xint[-1,:] = flux_zero/pi

			#Legendre polynomials for the Phase function due to multiple scatterers 

			ubar2 = 0.767  # FIT TO PURE RAYLEIGH LIMIT, ~(1/sqrt(3))^(1/2)
			F1 = (1.0+1.5*cosb*ubar1[ng,nt] #!was 3
							+ gcos2*(3.0*ubar2*ubar2*ubar1[ng,nt]*ubar1[ng,nt] - 1.0)/2.0)
			F2 = (1.-1.5*cosb*ubar1[ng,nt] 
							+ gcos2*(3.0*ubar2*ubar2*ubar1[ng,nt]*ubar1[ng,nt] - 1.0)/2.0)

			G=w0*positive*(F1+gama*F2)
			H=w0*negative*(gama*F1+F2)
			A=w0*(F1*c_plus_up+F2*c_minus_up)

			G=G*0.5/pi
			H=H*0.5/pi
			A=A*0.5/pi

			#Phase function for single scattering albedo frum Solar beam
			#uses the Two term Henyey-Greenstein function with the additiona rayleigh component 
                  #first term of TTHG: forward scattering
			puu0=((1-(cosb/2)**2) * (1-cosb**2)
							/sqrt((1+cosb**2+2*cosb*cos_theta)**3) 
							#second term of TTHG: backward scattering
							+((cosb/2)**2)*(1-(-cosb/2.)**2)
							/sqrt((1+(-cosb/2.)**2+2*(-cosb/2.)*cos_theta)**3)+
                            #rayleigh phase function
							(gcos2))

			for i in range(nlayer-1,-1,-1):
                        #direct beam
				xint[i,:] =( xint[i+1,:]*exp(-dtau[i,:]/ubar1[ng,nt]) 
					    #single scattering albedo from sun beam (from ubar0 to ubar1)
						+(w0[i,:]*F0PI/(4.*pi))*
						(puu0[i,:])*exp(-tau[i,:]/ubar0[ng,nt])*
						(1. - exp(-dtau[i,:]*(ubar0[ng,nt]+ubar1[ng,nt])/(ubar0[ng,nt]*ubar1[ng,nt])))*
						(ubar0[ng,nt]/(ubar0[ng,nt]+ubar1[ng,nt]))
						#multiple scattering terms 
						+A[i,:]*(1. - exp(-dtau[i,:] *(ubar0[ng,nt]+1*ubar1[ng,nt])/(ubar0[ng,nt]*ubar1[ng,nt])))*
						(ubar0[ng,nt]/(ubar0[ng,nt]+1*ubar1[ng,nt]))
						+G[i,:]*(exp(exptrm[i,:]*1-dtau[i,:]/ubar1[ng,nt]) - 1.0)/(lamda[i,:]*1*ubar1[ng,nt] - 1.0)
						+H[i,:]*(1. - exp(-exptrm[i,:]*1-dtau[i,:]/ubar1[ng,nt]))/(lamda[i,:]*1*ubar1[ng,nt] + 1.0))
			xint_at_top[ng,nt,:] = xint[0,:]	
	return xint_at_top

@jit(nopython=True, cache=True)
def get_flux_geom_1d(nlevel, wno,nwno, numg,numt, dtau, tau, w0, cosb,gcos2, 
	surf_reflect,ubar0, ubar1,cos_theta, F0PI):
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
	dtau_dedd : ndarray of float 
		This is the opacity contained within each individual layer (defined at midpoints of "levels")
		WITH D-Eddington Correction
		Dimensions=# layer by # wave
	tau_dedd : ndarray of float
		This is the cumulative summed opacity 
		WITH D-Eddington Correction
		Dimensions=# level by # wave	
	w0_dedd : ndarray of float 
		Same as w0 but with the delta eddington correction		
	cosb_dedd : ndarray of float 
		Same as cosbar buth with the delta eddington correction 
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
	g1    = (sq3*0.5)*(2. - w0*(1.+cosb))    #table 1
	g2    = (sq3*w0*0.5)*(1.-cosb)           #table 1
	lamda = sqrt(g1**2 - g2**2)              #eqn 21
	gama  = (g1-lamda)/g2                    #eqn 22

	#================ START CRAZE LOOP OVER ANGLE #================
	for ng in range(numg):
		for nt in range(numt):

			g3    = 0.5*(1.-sq3*cosb*ubar0[ng, nt])   #table 1 #ubar has dimensions [gauss angles by tchebyshev angles ]
	
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
			A, B, C, D = setup_tri_diag(nlayer,nwno,  c_plus_up, c_minus_up, 
									c_plus_down, c_minus_down, b_top, b_surface, surf_reflect,
									g1, g2, g3, lamda, gama, dtau, 
									exptrm_positive,  exptrm_minus) 

			positive = zeros((nlayer, nwno))
			negative = zeros((nlayer, nwno))
			#========================= Start loop over wavelength =========================
			L = 2*nlayer
			for w in range(nwno):
				#coefficient of posive and negative exponential terms 
				X = tri_diag_solve(L, A[:,w], B[:,w], C[:,w], D[:,w])

				#unmix the coefficients
				positive[:,w] = X[::2] + X[1::2] 
				negative[:,w] = X[::2] - X[1::2]
			#========================= End loop over wavelength =========================

			#use expression for bottom flux to get the flux_plus and flux_minus at last
			#bottom layer
			flux_zero  = positive[-1,:]*exptrm_positive[-1,:] + gama[-1,:]*negative[-1,:]*exptrm_minus[-1,:] + c_plus_down[-1,:]
			
			xint = zeros((nlevel,nwno))
			xint[-1,:] = flux_zero/pi

			#AM = c_minus_up
			#AP = c_plus_up

			#Legendre polynomials for the Phase function due to multiple scatterers 

			ubar2 = 0.767  # FIT TO PURE RAYLEIGH LIMIT, ~(1/sqrt(3))^(1/2)
			F1 = (1.0+1.5*cosb*ubar1[ng,nt] #!was 3
							+ gcos2*(3.0*ubar2*ubar2*ubar1[ng,nt]*ubar1[ng,nt] - 1.0)/2.0)
			F2 = (1.-1.5*cosb*ubar1[ng,nt] 
							+ gcos2*(3.0*ubar2*ubar2*ubar1[ng,nt]*ubar1[ng,nt] - 1.0)/2.0)

			G=w0*positive*(F1+gama*F2)
			H=w0*negative*(gama*F1+F2)
			A=w0*(F1*c_plus_up+F2*c_minus_up)

			G=G*0.5/pi
			H=H*0.5/pi
			A=A*0.5/pi

			#Phase function for single scattering albedo frum Solar beam
			#uses the Two term Henyey-Greenstein function with the additiona rayleigh component 
                  #first term of TTHG: forward scattering
			puu0=((1-(cosb/2)**2) * (1-cosb**2)
							/sqrt((1+cosb**2+2*cosb*cos_theta)**3) 
							#second term of TTHG: backward scattering
							+((cosb/2)**2)*(1-(-cosb/2.)**2)
							/sqrt((1+(-cosb/2.)**2+2*(-cosb/2.)*cos_theta)**3)+
                            #rayleigh phase function
							(gcos2))

			for i in range(nlayer-1,-1,-1):
                        #direct beam
				xint[i,:] =( xint[i+1,:]*exp(-dtau[i,:]/ubar1[ng,nt]) 
					    #single scattering albedo from sun beam (from ubar0 to ubar1)
						+(w0[i,:]*F0PI/(4.*pi))*
						(puu0[i,:])*exp(-tau[i,:]/ubar0[ng,nt])*
						(1. - exp(-dtau[i,:]*(ubar0[ng,nt]+ubar1[ng,nt])/(ubar0[ng,nt]*ubar1[ng,nt])))*
						(ubar0[ng,nt]/(ubar0[ng,nt]+ubar1[ng,nt]))
						#multiple scattering terms 
						+A[i,:]*(1. - exp(-dtau[i,:] *(ubar0[ng,nt]+1*ubar1[ng,nt])/(ubar0[ng,nt]*ubar1[ng,nt])))*
						(ubar0[ng,nt]/(ubar0[ng,nt]+1*ubar1[ng,nt]))
						+G[i,:]*(exp(exptrm[i,:]*1-dtau[i,:]/ubar1[ng,nt]) - 1.0)/(lamda[i,:]*1*ubar1[ng,nt] - 1.0)
						+H[i,:]*(1. - exp(-exptrm[i,:]*1-dtau[i,:]/ubar1[ng,nt]))/(lamda[i,:]*1*ubar1[ng,nt] + 1.0))
			xint_at_top[ng,nt,:] = xint[0,:]	
	return xint_at_top
