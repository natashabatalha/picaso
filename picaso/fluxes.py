from numba import jit, vectorize
from numpy import exp, zeros, where, sqrt, cumsum , pi, outer

@jit(nopython=True, cache=True)
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
	g1	= (sq3*0.5)*(2. - w0*(1.+cosbar))	#table 1
	g2	= (sq3*w0*0.5)*(1.-cosbar)		   #table 1
	g3	= 0.5*(1.-sq3*cosbar*ubar0)		  #table 1
	lamda = sqrt(g1**2 - g2**2)					 #eqn 21
	gama  = (g1-lamda)/g2							  #eqn 22
	
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
	C[1::2,:][:-1] = 2.0 * (1.0-gama[1:,:]**2)			#always good 
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

			g1	= (sq3*0.5)*(2. - w0*(1.+cosb))	#table 1
			g2	= (sq3*w0*0.5)*(1.-cosb)		   #table 1
			lamda = sqrt(g1**2 - g2**2)			  #eqn 21
			gama  = (g1-lamda)/g2					#eqn 22
			g3	= 0.5*(1.-sq3*cosb*ubar0[ng, nt])   #table 1 #ubar is now 100x 10 matrix.. 
	
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
									 gama, dtau, 
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
				g_back = -constant_back*cosb_og
				f = frac_a + frac_b*g_back**frac_c


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
								/sqrt((1+cosb_og**2+2*cosb_og*cos_theta)**3) 
								#second term of TTHG: backward scattering
								+(1-f)*(1-g_back**2)
								/sqrt((1+(-cosb_og/2.)**2+2*(-cosb_og/2.)*cos_theta)**3))
			elif single_phase==3:#'TTHG_ray':
				#Phase function for single scattering albedo frum Solar beam
				#uses the Two term Henyey-Greenstein function with the additiona rayleigh component 
					  		#first term of TTHG: forward scattering
				p_single=(ftau_cld*(f * (1-g_forward**2)
												/sqrt((1+cosb_og**2+2*cosb_og*cos_theta)**3) 
												#second term of TTHG: backward scattering
												+(1-f)*(1-g_back**2)
												/sqrt((1+(-cosb_og/2.)**2+2*(-cosb_og/2.)*cos_theta)**3))+			
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
	frac_a, frac_b, frac_c, constant_back, constant_forward):
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
	g1	= (sq3*0.5)*(2. - w0*(1.+cosb))	#table 1
	g2	= (sq3*w0*0.5)*(1.-cosb)		#table 1
	lamda = sqrt(g1**2 - g2**2)			#eqn 21
	gama  = (g1-lamda)/g2				#eqn 22

	#================ START CRAZE LOOP OVER ANGLE #================
	for ng in range(numg):
		for nt in range(numt):

			g3	= 0.5*(1.-sq3*cosb*ubar0[ng, nt])   #table 1 #ubar has dimensions [gauss angles by tchebyshev angles ]
	
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
			A, B, C, D = setup_tri_diag(nlayer,nwno,  c_plus_up, c_minus_up, 
									c_plus_down, c_minus_down, b_top, b_surface, surf_reflect,
									gama, dtau, 
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
				g_back = -constant_back*cosb_og
				f = frac_a + frac_b*g_back**frac_c


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
								/sqrt((1+cosb_og**2+2*cosb_og*cos_theta)**3) 
								#second term of TTHG: backward scattering
								+(1-f)*(1-g_back**2)
								/sqrt((1+(-cosb_og/2.)**2+2*(-cosb_og/2.)*cos_theta)**3))
			elif single_phase==3:#'TTHG_ray':
				#Phase function for single scattering albedo frum Solar beam
				#uses the Two term Henyey-Greenstein function with the additiona rayleigh component 
					  		#first term of TTHG: forward scattering
				p_single=(ftau_cld*(f * (1-g_forward**2)
												/sqrt((1+cosb_og**2+2*cosb_og*cos_theta)**3) 
												#second term of TTHG: backward scattering
												+(1-f)*(1-g_back**2)
												/sqrt((1+(-cosb_og/2.)**2+2*(-cosb_og/2.)*cos_theta)**3))+			
								#rayleigh phase function
								ftau_ray*(0.75*(1+cos_theta**2.0)))

			################################ END OPTIONS FOR DIRECT SCATTERING####################

			for i in range(nlayer-1,-1,-1):
						#direct beam
				xint[i,:] =( xint[i+1,:]*exp(-dtau[i,:]/ubar1[ng,nt]) 
						#single scattering albedo from sun beam (from ubar0 to ubar1)
						+(w0_og[i,:]*F0PI/(4.*pi))*
						(p_single[i,:])*exp(-tau_og[i,:]/ubar0[ng,nt])*
						(1. - exp(-dtau_og[i,:]*(ubar0[ng,nt]+ubar1[ng,nt])/(ubar0[ng,nt]*ubar1[ng,nt])))*
						(ubar0[ng,nt]/(ubar0[ng,nt]+ubar1[ng,nt]))
						#multiple scattering terms p_single
						+A[i,:]*(1. - exp(-dtau[i,:] *(ubar0[ng,nt]+1*ubar1[ng,nt])/(ubar0[ng,nt]*ubar1[ng,nt])))*
						(ubar0[ng,nt]/(ubar0[ng,nt]+1*ubar1[ng,nt]))
						+G[i,:]*(exp(exptrm[i,:]*1-dtau[i,:]/ubar1[ng,nt]) - 1.0)/(lamda[i,:]*1*ubar1[ng,nt] - 1.0)
						+H[i,:]*(1. - exp(-exptrm[i,:]*1-dtau[i,:]/ubar1[ng,nt]))/(lamda[i,:]*1*ubar1[ng,nt] + 1.0))
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
		Wavelength (cgs)
	
	Returns
	-------
	ndarray with shape ntemp x numwave
	"""
	h = 6.62607004e-27 # erg s 
	c = 2.99792458e+10 # cm/s
	k = 1.38064852e-16 #erg / K

	return ((2.0*h*c**2.0)/(w**5.0))*(1.0/(exp((h*c)/outer(t, w*k)) - 1.0))

@jit(nopython=True, cache=True)
def get_thermal_1d(nlevel, wno,nwno, numg,numt,tlevel, dtau, w0,cosb,plevel, ubar1):
	
	nlayer = nlevel -1 #nlayers 

	mu1 = 0.5 #from Table 1 Toon  
	twopi = pi+pi

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
	A, B, C, D = setup_tri_diag(nlayer,nwno,  c_plus_up, c_minus_up, 
							c_plus_down, c_minus_down, b_top, b_surface, surf_reflect,
							gama, dtau, 
							exptrm_positive,  exptrm_minus) 
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

	f_up = pi*(positive * exptrm_positive + gama * negative * exptrm_minus + c_plus_up)

	#calculate everyting from Table 3 toon
	alphax = ((1.0-w0)/(1.0-w0*cosb))**0.5
	G = twopi*w0*positive*(1.0+cosb*alphax)/(1.0+alphax)
	H = twopi*w0*negative*(1.0-cosb*alphax)/(1.0+alphax)
	J = twopi*w0*positive*(1.0-cosb*alphax)/(1.0+alphax)
	K = twopi*w0*negative*(1.0+cosb*alphax)/(1.0+alphax)
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
			flux_plus[-1,:] = twopi * (b_surface + b1[-1,:] * ubar1[ng,nt])
			flux_minus[0,:] = twopi * (1 - exp(-tau_top / ubar1[ng,nt])) * all_b[0,:]
			
			exptrm_angle = exp( - dtau / ubar1[ng,nt])
			exptrm_angle_mdpt = exp( -0.5 * dtau / ubar1[ng,nt]) 

			for itop in range(nlayer):

				#disbanning this for now because we dont need it in the thermal emission code
				#flux_minus[itop+1,:]=(flux_minus[itop,:]*exptrm_angle[itop,:]+
				#	                 (J[itop,:]/(lamda[itop,:]*ubar1[ng,nt]+1.0))*(exptrm_positive[itop,:]-exptrm_angle[itop,:])+
				#	                 (K[itop,:]/(lamda[itop,:]*ubar1[ng,nt]-1.0))*(exptrm_angle[itop,:]-exptrm_minus[itop,:])+
				#	                 sigma1[itop,:]*(1.-exptrm_angle[itop,:])+
				#	                 sigma2[itop,:]*(ubar1[ng,nt]*exptrm_angle[itop,:]+dtau[itop,:]-ubar1[ng,nt]) )

				#flux_minus_mdpt[itop,:]=(flux_minus[itop,:]*exptrm_angle_mdpt[itop,:]+
				#                        (J[itop,:]/(lamda[itop,:]*ubar1[ng,nt]+1.0))*(exptrm_positive_mdpt[itop,:]-exptrm_angle_mdpt[itop,:])+
				#                        (K[itop,:]/(-lamda[itop,:]*ubar1[ng,nt]+1.0))*(exptrm_minus_mdpt[itop,:]-exptrm_angle_mdpt[itop,:])+
				#                        sigma1[itop,:]*(1.-exptrm_angle_mdpt[itop,:])+
				#                        sigma2[itop,:]*(ubar1[ng,nt]*exptrm_angle_mdpt[itop,:]+0.5*dtau[itop,:]-ubar1[ng,nt]))

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
		
		#for testing purposes
		#import pickle as pk
		#pk.dump([flux_at_top, flux_down, flux_minus, flux_minus_mdpt, flux_plus, flux_plus_mdpt], open('/Users/natashabatalha/Documents/PICASO/test/thermal/debugthermal.pk','wb'))

	return flux_at_top #, flux_down# numg x numt x nwno
