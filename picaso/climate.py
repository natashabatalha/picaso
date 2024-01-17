import numpy as np 
import warnings
from numba import jit, vectorize
from numpy import exp, zeros, where, sqrt, cumsum , pi, outer, sinh, cosh, min, dot, array,log,log10
from .fluxes import get_reflected_1d,get_thermal_1d,get_reflected_1d_gfluxv
#from .fluxes import get_thermal_1d_newclima, get_thermal_1d_gfluxi #deprecated
from .atmsetup import ATMSETUP
from .optics import compute_opacity
from .disco import compress_thermal

#testing error tracker
# from loguru import logger 

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
def convec(temp,pressure, t_table, p_table, grad, cp):
    """
    Calculates Grad arrays from profiles
    
    Parameters 
    ----------
    temp : array 
        level temperature array
    pressure : array
        level pressure array
    t_table : array
        array of Temperature values with 53 entries
    p_table : array 
        array of Pressure value with 26 entries
    grad : array 
        array of gradients of dimension 53*26
    cp : array 
        array of cp of dimension 53*26
    Return
    ------
    grad_x, cp_x
    """
    # layer profile arrays
    tbar= np.zeros(shape=(len(temp)-1))
    pbar= np.zeros(shape=(len(temp)-1))
    
    grad_x, cp_x = np.zeros(shape=(len(temp)-1)), np.zeros(shape=(len(temp)-1))

    for j in range(len(temp)-1):
        tbar[j] = 0.5*(temp[j]+temp[j+1])
        pbar[j] = sqrt(pressure[j]*pressure[j+1])
        calc_type = 0
        grad_x[j], cp_x[j] =  did_grad_cp( tbar[j], pbar[j], t_table, p_table, grad, cp, calc_type)

    return grad_x, cp_x

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





@jit(nopython=True, cache=True)
def mat_sol(a, nlevel, nstrat, dflux):
    """
    Parameters
    ----------
    A : array
        Matrix to be decomposed dimension nlevel*nlevel
    nlevel : int 
        # of levels (not layers)
    nstrat : int 
        tropopause level
    dflux : array 
        dimension is nlevel
    
    
    Returns
    -------
    array 
        anew (nlevel*nlevel) and bnew(nstrat)
    
    """
    #      Numerical Recipes Matrix inversion solution.
    #  Utilizes LU decomposition and iterative improvement.
    # This is a py version of the MATSOL routine of the fortran version

    anew , indx = lu_decomp(a , nstrat, nlevel)

    bnew = lu_backsubs(anew, nstrat, nlevel, indx, dflux) 

    return anew, bnew       

@jit(nopython=True, cache=True)
def lu_decomp(a, n, ntot):
    """
    Parameters
    ----------
    A : array
        Matrix to be decomposed dimension np*np
    n : int 
        n*n subset of A matrix is used
    ntot : int 
        dimension of A is ntot*ntot
     
    Returns
    -------
    array 
        A array and indx array
    
    """

    # Numerical Recipe routine of LU decomposition
    TINY= 1e-20
    NMAX=100
    
    d=1.
    vv=np.zeros(shape=(NMAX))
    indx=np.zeros(shape=(n),dtype=np.int8)

    for i in range(n):
        aamax=0.0
        for j in range(n):
            if abs(a[i,j]) > aamax:
                aamax=abs(a[i,j])
        if aamax == 0.0:
        	raise ValueError("Array is singular, cannot be decomposed in n:" + str(n))
        vv[i]=1.0/aamax  

    for j in range(n):
        for i in range(j):
            sum= a[i,j]
            for k in range(i):
                sum=sum-a[i,k]*a[k,j]
            a[i,j]=sum

        aamax=0.0
        for i in range(j,n):
            sum=a[i,j]
            for k in range(j):
                sum=sum-a[i,k]*a[k,j]
            a[i,j]=sum
            dum=vv[i]*abs(sum)
            
            if dum >= aamax:
                imax=i
                aamax=dum
        
        if j != imax:
            for k in range(n):
                dum=a[imax,k]
                a[imax,k]=a[j,k]
                a[j,k]=dum
            d=-d
            vv[imax]=vv[j]
        
        indx[j]=imax

        if a[j,j] == 0:
            a[j,j]= TINY
        if j != n-1 : # python vs. fortran array referencing difference
            dum=1.0/a[j,j]
            for i in range(j+1,n):
                a[i,j]=a[i,j]*dum
        
    return a , indx

@jit(nopython=True, cache=True)
def lu_backsubs(a, n, ntot, indx, b):
    """
    Parameters
    ----------
    A : array
        Matrix to be decomposed dimension np*np
    n : int 
        n*n subset of A matrix is used
    ntot : int 
        dimension of A is ntot*ntot
    indx: array
        Index array of dimension n, output from lu_decomp
    b: array
        Input array for calculation
        
    Returns
    -------
    array 
        B array of dimension n*n

    """

    # Numerical Recipe routine of back substitution

    ii = -1

    for i in range(n):
        ll=indx[i]
        sum=b[ll]
        b[ll]=b[i]
        
        if ii != -1 :
            for j in range(ii,i):
                sum=sum-a[i,j]*b[j]
    
        elif sum != 0.0:
            ii=i 
        b[i]=sum
        
    for i in range(n-1,-1,-1):
        sum=b[i]
        for j in range(i+1,n):
            sum=sum-a[i,j]*b[j]
        
        b[i]=sum/a[i,i]
        
    
    return b

# @logger.catch # Add this to track errors
#@jit(nopython=True, cache=True)
def t_start(nofczns,nstr,it_max,conv,x_max_mult, 
            rfaci, rfacv, nlevel, temp, pressure, p_table, t_table, 
            grad, cp, tidal, tmin,tmax, dwni , bb , y2, tp, DTAU, TAU, W0, COSB, 
            ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman , surf_reflect, ubar0,ubar1,
            cos_theta, FOPI, single_phase,multi_phase,frac_a,frac_b,frac_c,
            constant_back,constant_forward, tridiagonal , wno,nwno,ng,nt,gweight,tweight, ngauss, gauss_wts, save_profile, all_profiles):
    """
    Module to iterate on the level TP profile to make the Net Flux as close to 0.
    Opacities/chemistry are not updated while iterating in this module.
    Parameters
    ----------
    nofczns : int
        # of convective zones 
    nstr : array
        dimension of 20
        NSTR vector describes state of the atmosphere:
        0   is top layer
        1   is top layer of top convective region
        2   is bottom layer of top convective region
        3   is top layer of lower radiative region
        4   is top layer of lower convective region
        5   is bottom layer of lower convective region
    it_max : int 
        # of maximum iterations allowed
    conv: 
        
    x_max_mult: 
    rfaci : float 
        IR flux addition fraction 
    rfacv : float
        Visible flux addition fraction
    nlevel : int
        # of levels
    temp : array
        Guess Temperature array, dimension is nlevel
    pressure : array
        Pressure array
    p_table : array
        Tabulated pressure array for convection calculations
    t_table : array
        Tabulated Temperature array for convection calculations
    grad : array
        Tabulated grad array for convection calculations
    
    cp : array
        Tabulated cp array for convection calculations
    tidal : array
        Tidal Fluxes dimension = nlevel
    tmin : float
        Minimum allwed Temp in the profile
    tmax : float
        Maximum allowed Temp in the profile
    dwni : array
        Spectral interval corrections (dimension= nwvno)   
    bb : array
        Array of BB fluxes used in RT
    y2 : array
        Output of set_bb function in fluxes.py
    tp : array
        Output of set_bb function in fluxes.py
    
    Returns
    -------
    array 
        Temperature array and lapse ratio array if converged
        else Temperature array twice
    """
    #     Routine to iteratively solve for T(P) profile.  Uses a Newton-
    #     Raphson iteration to zero out the net flux in the radiative
    #     zone above NSTRAT.  Repeats until average temperature change
    #     is less than CONV or until ITMAX is reached.

    # -- SM -- needs a lot of documentation

    #Climate default is to run both reflected and thermal. Though sometimes, in most cases we only want thermal.
    eps=1e-4

    n_top_r=nstr[0]-1

    # here are other  convergence and tolerance criterias

    step_max = 0.01e0 # scaled maximum step size in line searches
    alf = 1.e-4    # ? 
    alam2 = 0.0   # ? 
    tolmin=1.e-5   # ?
    tolf = 5e-3    # tolerance in fractional Flux we are aiming for
    tolx = 5e-3    # tolerance in fractional T change we are aiming for

    #both reflected and thermal
    #neb this double true in the first call to reflected light needs to be changed
    if rfacv==0:compute_reflected=False
    else:compute_reflected=True
    flux_net_v_layer_full, flux_net_v_full, flux_plus_v_full, flux_minus_v_full , flux_net_ir_layer_full, flux_net_ir_full, flux_plus_ir_full, flux_minus_ir_full = get_fluxes(pressure, temp, dwni, bb , y2, tp, tmin, tmax, DTAU, TAU, W0, 
            COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman , surf_reflect, 
            ubar0,ubar1,cos_theta, FOPI, single_phase,multi_phase,frac_a,frac_b,frac_c,constant_back,constant_forward, tridiagonal , 
            wno,nwno,ng,nt, gweight,tweight, nlevel, ngauss, gauss_wts,compute_reflected, True)#True for reflected, True for thermal

    # extract visible fluxes
    flux_net_v_layer = flux_net_v_layer_full[0,0,:]  #fmnetv
    flux_net_v = flux_net_v_full[0,0,:]#fnetv
    flux_plus_v =  flux_plus_v_full[0,0,:,:]
    flux_minus_v = flux_minus_v_full[0,0,:,:]

    # extract ir fluxes
    flux_net_ir_layer = flux_net_ir_layer_full[:] #fmneti
    flux_net_ir = flux_net_ir_full[:]     #fneti
    flux_plus_ir = flux_plus_ir_full[:,:]  
    flux_minus_ir = flux_minus_ir_full[:,:]
    
   
    
    # arrays for total net fluxes = optical+ir + tidal
    flux_net=np.zeros(shape=(nlevel))
    flux_net_midpt=np.zeros(shape=(nlevel))
    dflux=np.zeros(shape=(nlevel))
    f_vec=np.zeros(shape=(nlevel)) #fvec
    p=np.zeros(shape=(nlevel)) #p
    g=np.zeros(shape=(nlevel))
    
    # jacobian of zeros
    A= np.zeros(shape=(nlevel,nlevel)) 
    

    
    for its in range(it_max):
        
        # the total net flux = optical + ir + tidal component
        
        flux_net = rfaci* flux_net_ir + rfacv* flux_net_v +tidal #fnet
        flux_net_midpt = rfaci* flux_net_ir_layer + rfacv* flux_net_v_layer +tidal #fmnet
        
        #print('flux_net_midpt',flux_net_midpt)

        #raise Exception ('stop in tstart')

        beta= temp.copy() # beta vector
        
       
        # store old fluxes and temp before iteration
        # do not store the ir+vis flux because we are going to perturb only thermal structure

        
        temp_old= temp.copy() 
        flux_net_old = flux_net_ir.copy() #fnetip
        flux_net_midpt_old= flux_net_ir_layer.copy()  #fmip

        nao = n_top_r
        n_total = 0#0 #ntotl

        sum = 0.0
        sum_1 = 0.0 # sum1
        test = 0.0
        
        flag_nao = 0
        if nao < 0 :
            nao_temporary = nao
            nao = 0 # to avoid negative of negative index leading to wrong indexing, fort vs. py
            flag_nao = 1 # so that it can be reversed back to previous value after loop

        for nca in range(0, 3*nofczns, 3): 
            
            # first fill in the dflux  vector
            # contains flux ant top and midpt fluxes for nstrat -1 layers in stratosphere

            n_top_a= nstr[nca] #ntopa -- top of atmosphere or top of other rad zones

            n_strt_a = nstr[nca+1] #nstrta -- top of top conv zone
            
            
            # n_top_a to n_strt_a is a radiative zone
            # n_bot_a is the top of the next rad zone than this
            
            n_bot_a= nstr[nca+2] +1 #nbota -- top of lower rad zone

            if n_top_a == n_top_r+1 : # if the rad zone is also top of atmos
                dflux[0] = flux_net[n_top_r+1]
                f_vec[0] = dflux[0]

                sum += f_vec[0]**2
                sum_1 += temp[0]**2

                if abs(f_vec[0]) > test :
                    test = abs(dflux[0])
                n_total += 1
            
            # +1 to include last element
            for j in range(n_top_a+1, n_strt_a+1):
                
                dflux[j-nao] = flux_net_midpt[j-1] 

                f_vec[j-nao] = dflux[j-nao] 

                sum += f_vec[j-nao]**2 

                sum_1 += temp[j-nao]**2 

                if abs(f_vec[j-nao]) > test : 
                    test = abs(dflux[j-nao]) 
                n_total += 1
            
            if flag_nao == 1 :
                nao= nao_temporary
                flag_nao = 0

            
            
            nao += n_bot_a - n_strt_a

        
        f = 0.5*sum # used in linesearch, defined in NR function fmin

        # test if we are already at a root
        if (test/abs(tidal[0])) < 0.01*tolf :
            print(" We are already at a root, tolf , test = ",0.01*tolf,", ",test/abs(tidal[0]))
            flag_converge = 2
            dtdp=np.zeros(shape=(nlevel-1))
            for j in range(nlevel -1):
                dtdp[j] = (log( temp[j]) - log( temp[j+1]))/(log(pressure[j]) - log(pressure[j+1]))
            
            return   temp,  dtdp, flag_converge, flux_net_ir, flux_plus_ir[0,:], all_profiles
            
        
        # define maximum T step size
        step_max *= max(sqrt(sum_1),n_total*1.0)#step_max_tolerance*
        print('maximum scaled step size',step_max, n_total, sum_1)
        no =n_top_r
        
        i_count= 1 #icount
        
        flag_no = 0
        if no < 0 :
            no_temporary = no
            no = 0 # to avoid negative of negative index leading to wrong indexing, fort vs. py
            flag_no = 1 # so that it can be reversed back to previous value after loop
       
        for nz in range(0, 3*nofczns, 3):

            n_top = nstr[nz] +1 #ntop
            n_strt = nstr[nz+1] #nstrt
            n_conv_top = n_strt + 1 #ncnvtop
            n_conv_bot= nstr[nz+2] +1 #ncnvbot

            if nz == 0 :
                n_top -= 1
            
            
            
        # begin jacobian calculation here
        # +1 to include last element
            for jm in range(n_top, n_strt+1):

                # chose perturbation for each level

                i_count += 1

                #eps is just a tolerance value currently fixed at 1e-4
                del_t = max(eps * temp_old[jm], 3.0) # perturbation

                beta[jm] += del_t # perturb

                
                # now reconstruct Temp profile

                for nb in range(0, 3*nofczns, 3):

                    n_top_b = nstr[nb] + 1 # ntopb
                    
                    if nb == 0:
                        n_top_b -= 1 #ntopb
                    
                    n_strt_b = nstr[nb+1] # nstrtb
                    
                    n_conv_top_b = n_strt_b + 1 #nctopb

                    n_bot_b = nstr[nb+2] +1 #nbotb

                    
                    # +1 to include last element   
                    for j1 in range(n_top_b,n_strt_b+1):
                        temp[j1] = beta[j1]
                    
                    # +1 to include last element
                    for j1 in range(n_conv_top_b, n_bot_b+1): 
                        
                        press = sqrt(pressure[j1-1]*pressure[j1])
                        calc_type =  0 
                        grad_x, cp_x = did_grad_cp( beta[j1-1], press, t_table, p_table, grad, cp, calc_type)
                        
                        temp[j1]= exp(log(temp[j1-1]) + grad_x*(log(pressure[j1]) - log(pressure[j1-1])))
                
                

                # temperature has been perturbed
                # now recalculate the IR fluxes, so call picaso with only thermal

                flux_net_v_layer_full, flux_net_v_full, flux_plus_v_full, flux_minus_v_full , flux_net_ir_layer_full, flux_net_ir_full, flux_plus_ir_full, flux_minus_ir_full = get_fluxes(pressure, temp, dwni, bb , y2, tp, tmin, tmax, DTAU, TAU, W0, 
            COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman , surf_reflect, 
            ubar0,ubar1,cos_theta, FOPI, single_phase,multi_phase,frac_a,frac_b,frac_c,constant_back,constant_forward, tridiagonal , 
            wno,nwno,ng,nt,gweight,tweight, nlevel, ngauss, gauss_wts, False, True) #false for reflected, True for thermal


                # extract ir fluxes

                flux_net_ir_layer = flux_net_ir_layer_full[:] #fmneti
                flux_net_ir = flux_net_ir_full[:]     #fneti
                flux_plus_ir = flux_plus_ir_full[:,:]  
                flux_minus_ir = flux_minus_ir_full[:,:]

     
                
                # now calculate jacobian terms in the same way as dflux
                nco = n_top_r
                
                # -ve nco and no will mess indexing
                # so we want to set them to 0 temporarily if that occurs
                flag_nco = 0 
                if nco < 0 :
                    nco_temporary = nco
                    nco = 0 # to avoid negative of negative index leading to wrong indexing, fort vs. py
                    flag_nco = 1 # so that it can be reversed back to previous value after loop

                for nc in range(0,3*nofczns, 3):
                    
                    n_top_c = nstr[nc] +1 # ntopc

                    if nc ==0:
                        n_top_c -= 1
                    n_strt_c = nstr[nc+1]
                    n_bot_c = nstr[nc+2] +1
                    
                    
                        
                    
                   
                    if n_top_c == n_top_r+1 :
                        
                        A[n_top_c-nco,jm-no] = (flux_net_ir[n_top_c]-flux_net_old[n_top_c])/del_t
                        
                    else:
                        
                        A[n_top_c-nco,jm-no] = (flux_net_ir_layer[n_top_c-1]-flux_net_midpt_old[n_top_c-1])/del_t
                        
                    
                    
                    # omitted -1 to include last element 
                    
                    for im in range(n_top_c,n_strt_c):
                        #print(im+1-nco,jm-no, "3rd",jm,no)
                        A[im+1-nco,jm-no] = (flux_net_ir_layer[im]-flux_net_midpt_old[im])/del_t
                        
                    # changing them back to what they were
                    
                    if flag_nco == 1 :
                        nco= nco_temporary
                        flag_nco = 0
                    
                    
                       

                    nco+= n_bot_c-n_strt_c
                

                # undo beta vector perturbation
                beta[jm] = beta[jm] - del_t
            
            if flag_no == 1 :
                no= no_temporary
                flag_no = 0
            
            no += n_conv_bot-n_strt
        
        # a long print statement here in original. dont know if needed

        
        for i in range(n_total):
            sum=0.0
            for j in range(n_total):
                sum += A[j,i]*f_vec[j]
            
            g[i] = sum

            p[i] = -f_vec[i]
        
        f_old = f #fold
        
        #print('f_vec[0],f_vec[-1],min,max:f_vec', 
        #    f_vec[0],f_vec[-1],min(f_vec),max(f_vec))
        #print(f_vec)
        #raise Exception ("stop")

        A, p = mat_sol(A, nlevel, n_total, p)
        
        #print(p)
        
        

        check = False

        sum = 0.0
        # Now we are in the "line search" routine
        # we ignore the first two points since they are flaky
        # start from 2 (3rd array pos in fortran), so changing loop initial
        
        for i in range(2,n_total):
            sum += p[i]**2
        sum = sqrt(sum)
        
        
        # scale if attempted step is too big
        if sum > step_max:
            for i in range(n_total):
                p[i] *= step_max/sum

                dflux[i] = -p[i]
        
        slope = 0.0

        for i in range(n_total):
            slope += g[i]*p[i]
        # SM -- next two lines is problematic ? 
        #if slope >= 0.0 :
        #    raise ValueError("roundoff problem in linen search")
        
        ## checked till here -- SM
        test = 0.0
        
        for i in range(n_total):
            tmp = abs(p[i])/temp_old[i]
            if tmp > test :
                test= tmp 

        alamin = tolx/test
        alam = 1.0
        
        f2= f #################### to avoid call before assignment and run using numba
        #     Convergence test:  Find magnitude of correction by comparing
        #        temperature steps to a appropriate scale SCALT.  If average
        #        correction ERR is large, use only a fraction of the step.
        #        When ERR is less than CONV, routine has converged.
        
        flag_converge = 0
        # instead of the goto statement here
        #ct_num = 0
        while flag_converge == 0 :
            #ct_num+=1
            err = 0.0
            dmx = 0.0
            scalt = 1.0
            slow =8.0/scalt

            for j in range(n_total):
                dzx= abs(p[j])

                if dzx > dmx :
                    dmx = dzx
                    jmx = j+ n_top_r
                err += dzx
            
            err= err/(float(n_total)*scalt)

            if jmx > nstr[1] :
                jmx+= nstr[2]-nstr[1]
            
            ndo = n_top_r
            flag_ndo = 0 
            if ndo < 0 :
                ndo_temporary = ndo
                ndo = 0 # to avoid negative of negative index leading to wrong indexing, fort vs. py
                flag_ndo = 1 # so that it can be reversed back to previous value after loop
            
            for nd in range(0,3*nofczns, 3):
                n_top_d = nstr[nd] +1
                if nd == 0:
                    n_top_d -= 1

                n_strt_d = nstr[nd+1]

                n_bot_d= nstr[nd+2] +1
                
                
                   
                #+1 for fort to py
                
                for j in range(n_top_d,n_strt_d+1):
                    temp[j]= beta[j]+ alam*p[j-ndo]
                    #print(p[j-ndo],beta[j])
                #+1 for fort to py
                for j1 in range(n_strt_d+1, n_bot_d+1):

                    press = sqrt(pressure[j1-1]*pressure[j1])
                    calc_type =  0 # only need grad_x in return
                    grad_x, cp_x = did_grad_cp( temp[j1-1], press, t_table, p_table, grad, cp, calc_type)
                            
                    temp[j1]= exp(log(temp[j1-1]) + grad_x*(log(pressure[j1]) - log(pressure[j1-1])))
                
                if flag_ndo == 1 :
                        ndo= ndo_temporary
                        flag_ndo = 0

                ndo += n_bot_d - n_strt_d
            
            # artificial damper

            for j1 in range(n_top_r+1, nlevel):
                if temp[j1] < tmin:
                    temp[j1] = tmin+ 0.1
                elif temp[j1] > tmax:
                    temp[j1] = tmax- 0.1
            
            # re calculate thermal flux
            flux_net_v_layer_full, flux_net_v_full, flux_plus_v_full, flux_minus_v_full , flux_net_ir_layer_full, flux_net_ir_full, flux_plus_ir_full, flux_minus_ir_full = get_fluxes(pressure, temp, dwni, bb , y2, tp, tmin, tmax, DTAU, TAU, W0, 
            COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman , surf_reflect, 
            ubar0,ubar1,cos_theta, FOPI, single_phase,multi_phase,frac_a,frac_b,frac_c,constant_back,constant_forward, tridiagonal , 
            wno,nwno,ng,nt,gweight,tweight, nlevel, ngauss, gauss_wts, False, True) #false reflected, True thermal


           

            # extract ir fluxes

            flux_net_ir_layer = flux_net_ir_layer_full[:] #fmneti
            flux_net_ir = flux_net_ir_full[:]     #fneti
            flux_plus_ir = flux_plus_ir_full[:,:]  
            flux_minus_ir = flux_minus_ir_full[:,:]
    
            # re calculate net fluxes
            flux_net = rfaci* flux_net_ir + rfacv* flux_net_v +tidal #fnet
            flux_net_midpt = rfaci* flux_net_ir_layer + rfacv* flux_net_v_layer +tidal #fmnet
            
            sum = 0.0
            nao = n_top_r
            flag_nao = 0
            if nao < 0 :
                nao_temporary = nao
                nao = 0 # to avoid negative of negative index leading to wrong indexing, fort vs. py
                flag_nao = 1 # so that it can be reversed back to previous value after loop

            for nca in range(0,3*nofczns,3):
                n_top_a = nstr[nca] + 1
                if nca ==0 :
                    n_top_a -= 1
                
                n_strt_a=nstr[nca+1]
                n_bot_a = nstr[nca+2] + 1
                
                
                   

                if n_top_a == n_top_r +1 :
                    
                    f_vec[0]= flux_net[n_top_r +1]
                    sum += f_vec[0]**2
                else:
                    f_vec[n_top_a-nao] = flux_net_midpt[n_top_a -1]
                    sum += f_vec[n_top_a - nao]**2
                
                for j in range(n_top_a+1,n_strt_a+1):
                    #print(j-1)
                    f_vec[j-nao] = flux_net_midpt[j -1]
                    sum += f_vec[j-nao]**2
                
                if flag_nao == 1 :
                        nao= nao_temporary
                        flag_nao = 0
                
                nao+= n_bot_a - n_strt_a
                        
            f= 0.5*sum
            #print('cond1:alam.lt.alamin',alam, alamin)
            #print('cond2:f.le.CCC',f,f_old + alf*alam*slope)
            #print('f,fold,alf,alam,slope',f,f_old,alf,alam,slope)
            #First check: Is T too small to continue? 
            if alam < alamin :
                #print(alam, alamin)
                check = True
                print(' CONVERGED ON SMALL T STEP alam, alamin', alam, alamin)
                #print("1st if")
                #print(alam, alamin)
                flag_converge, check = check_convergence(f_vec, n_total, tolf, check, f, dflux, tolmin, temp, temp_old, g , tolx)
 
            #Second check: Has the net flux decreased enough that we are happy in the line search
            #If so you can proceed
            elif f <= f_old + alf*alam*slope :
                #print("2nd if")
                
                
                print ('Exit with decreased f')
                flag_converge, check = check_convergence(f_vec, n_total, tolf, check, f, dflux, tolmin, temp, temp_old, g , tolx)

            #Else: Let's back track     
            else:
                
                # we backtrack
                #print("3rd if")
                print(' Now backtracking, f, fold, alf, alam, slope', f, f_old, alf, alam, slope)
                if alam == 1.0:
                    
                    tmplam= -slope/ (2*(f-f_old-slope))
                else:
                    
                    rhs_1 = f- f_old - alam*slope
                    rhs_2 = f2 - f_old - alam2*slope
                    anr= ((rhs_1/alam**2)-(rhs_2/alam2**2))/(alam-alam2)
                    b= (-alam2*rhs_1/alam**2+alam*rhs_2/alam2**2)/(alam-alam2)
                    

                    if anr == 0 :
                        tmplam= -slope/(2.0*b)
                        
                        
                    else:
                        disc= b*b - 3.0*anr*slope
                        
                        if disc < 0.0 :
                            tmplam= 0.5*alam
                           
                        elif b <= 0.0:
                            tmplam=(-b + sqrt(disc))/(3.0*anr)
                            

                        else:
                            tmplam= -slope/(b+sqrt(disc))
                            
                    if tmplam > 0.5*alam:
                        
                        tmplam= 0.5*alam
            if ((flag_converge != 2) & (flag_converge != 1)):
                alam2=alam
                f2=f
                
                alam = max(tmplam,0.1*alam)

            if np.isnan(np.sum(temp)) == True:
                
                flag_converge = 1 # to avoid getting stuck here unnecesarily.
                temp = temp_old.copy() +0.5
                print("Got stuck with temp NaN -- so escaping the while loop in tstart")
        

        print("Iteration number ", its,", min , max temp ", min(temp),max(temp), ", flux balance ", flux_net[0]/abs(tidal[0]))
        #print(f, f_old, tolf, np.max((temp-temp_old)/temp_old), tolx)
        if save_profile == 1:
            all_profiles = np.append(all_profiles,temp_old)
        if flag_converge == 2 : # converged
            # calculate  lapse rate
            dtdp=np.zeros(shape=(nlevel-1))
            for j in range(nlevel -1):
                dtdp[j] = (log( temp[j]) - log( temp[j+1]))/(log(pressure[j]) - log(pressure[j+1]))
            
            print("In t_start: Converged Solution in iterations ",its)
            
           
           
            return   temp,  dtdp, flag_converge , flux_net_ir, flux_plus_ir[0,:] , all_profiles
        
    print("Iterations exceeded it_max ! sorry ")#,np.max(dflux/tidal), tolf, np.max((temp-temp_old)/temp_old), tolx)
    dtdp=np.zeros(shape=(nlevel-1))
    for j in range(nlevel -1):
        dtdp[j] = (log( temp[j]) - log( temp[j+1]))/(log(pressure[j]) - log(pressure[j+1]))


    return temp, dtdp, flag_converge  , flux_net_ir_layer, flux_plus_ir[0,:], all_profiles

@jit(nopython=True, cache=True)
def check_convergence(f_vec, n_total, tolf, check, f, dflux, tolmin, temp, temp_old, g , tolx):
    """
    
    Module for checking convergence. Used in t_start module.

    """
    test = 0.0
    for i in range(n_total):
        if abs(f_vec[i]) > test:
            test=abs(f_vec[i])
    
    if test < tolf :
        check = False
        
        flag_converge = 2
        return flag_converge , check

    if check == True :
        test = 0.0
        den1 = max(f,0.5*(n_total))
        
        for i in range(n_total):
            tmp= abs(g[i])*dflux[i]/den1
            if tmp > test:
                test= tmp
        
        if test < tolmin :
            check= True
        else :
            check= False
        

        flag_converge = 2
        return flag_converge, check
    
    test = 0.0
    
    for i in range(n_total):
        tmp = (abs(temp[i]-temp_old[i]))/temp_old[i]
        if tmp > test:
            test=tmp
    if test < tolx :
        
        
        flag_converge = 2

        return flag_converge, check
    

    flag_converge = 1
    return flag_converge , check

@jit(nopython=True, cache=True)
def growup(nlv, nstr, ngrow) :
    """
    
    Module for growing conv zone. Used in find_strat module.
    
    """
    n = 2+3*(nlv-1) -1 # -1 for the py referencing
    nstr[n]= nstr[n]-1*ngrow

    return nstr

@jit(nopython=True, cache=True)
def growdown(nlv,nstr, ngrow) :
    """
    
    Module for growing down conv zone. Used in find_strat module.
    
    """

    n = 3+3*(nlv-1) -1 # -1 for the py referencing
    nstr[n] = nstr[n] + 1*ngrow
    nstr[n+1] = nstr[n+1] + 1*ngrow

    return nstr

@jit(nopython=True, cache=True)
def get_fluxes( pressure, temperature, dwni,  bb , y2, tp, tmin, tmax ,DTAU, TAU, W0, 
            COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman , surf_reflect, 
            ubar0,ubar1,cos_theta, FOPI, single_phase,multi_phase,frac_a,frac_b,frac_c,constant_back,constant_forward, tridiagonal , 
            wno,nwno,ng,nt,gweight,tweight, nlevel, ngauss, gauss_wts,reflected, thermal):
    """
    Program to run RT for climate calculations. Runs the thermal and reflected module.
    And combines the results with wavenumber widths.

    Parameters 
    ----------
    pressure : array 
        Level Pressure  Array
    temperature : array
        Opacity class from `justdoit.opannection`
    dwni : array 
        IR wavenumber intervals.
    bb : array 
        BB flux array. output from set_bb
    y2 : array
        output from set_bb
    tp : array
        output from set_bb
    tmin : float
        Minimum temp upto which interpolation has been done.
    tmax : float
        Maximum temp upto which interpolation has been done.
    

    reflected : bool 
        Run reflected light
    thermal : bool 
        Run thermal emission

        
    Return
    ------
    array
        Visible and IR -- net (layer and level), upward (level) and downward (level)  fluxes
    """
    #print('enter climate')

    # for visible
    flux_net_v = np.zeros(shape=(ng,nt,nlevel)) #net level visible fluxes
    flux_net_v_layer=np.zeros(shape=(ng,nt,nlevel)) #net layer visible fluxes

    flux_plus_v= np.zeros(shape=(ng,nt,nlevel,nwno)) # level plus visible fluxes
    flux_minus_v= np.zeros(shape=(ng,nt,nlevel,nwno)) # level minus visible fluxes
    
    #"""<<<<<<< NEWCLIMA
    # for thermal
    flux_plus_midpt = np.zeros(shape=(ng,nt,nlevel,nwno))
    flux_minus_midpt = np.zeros(shape=(ng,nt,nlevel,nwno))

    flux_plus = np.zeros(shape=(ng,nt,nlevel,nwno))
    flux_minus = np.zeros(shape=(ng,nt,nlevel,nwno))
    #"""

    """<<<<<<< OG
    # for thermal
    flux_plus_midpt = np.zeros(shape=(nlevel,nwno))
    flux_minus_midpt = np.zeros(shape=(nlevel,nwno))

    flux_plus = np.zeros(shape=(nlevel,nwno))
    flux_minus = np.zeros(shape=(nlevel,nwno))
    """

    # outputs needed for climate
    flux_net_ir = np.zeros(shape=(nlevel)) #net level visible fluxes
    flux_net_ir_layer=np.zeros(shape=(nlevel)) #net layer visible fluxes

    flux_plus_ir= np.zeros(shape=(nlevel,nwno)) # level plus visible fluxes
    flux_minus_ir= np.zeros(shape=(nlevel,nwno)) # level minus visible fluxes

    
    #ugauss_angles= np.array([0.0985350858,0.3045357266,0.5620251898,0.8019865821,0.9601901429])    
    #ugauss_weights = np.array([0.0157479145,0.0739088701,0.1463869871,0.1671746381,0.0967815902])
    #ugauss_angles = np.array([0.66666])
    #ugauss_weights = np.array([0.5])

    if reflected:
        #use toon method (and tridiagonal matrix solver) to get net cumulative fluxes 
        b_top = 0.0
        for ig in range(ngauss): # correlated - loop (which is different from gauss-tchevychev angle)
            """
            <<<<<<< NEWCLIMA
            #here only the fluxes are returned since we dont care about the outgoing intensity at the 
            #top, which is only used for albedo/ref light spectra
            ng_clima,nt_clima=1,1
            ubar0_clima = ubar0*0+0.5
            ubar1_clima = ubar1*0+0.5

            _, out_ref_fluxes = get_reflected_1d_newclima(nlevel, wno,nwno,ng_clima,nt_clima,
                                    DTAU[:,:,ig], TAU[:,:,ig], W0[:,:,ig], COSB[:,:,ig],
                                    GCOS2[:,:,ig],ftau_cld[:,:,ig],ftau_ray[:,:,ig],
                                    DTAU_OG[:,:,ig], TAU_OG[:,:,ig], W0_OG[:,:,ig], COSB_OG[:,:,ig],
                                    surf_reflect, ubar0_clima,ubar1_clima,cos_theta, F0PI,
                                    single_phase,multi_phase,
                                    frac_a,frac_b,frac_c,constant_back,constant_forward, tridiagonal,
                                    get_toa_intensity=0, get_lvl_flux=1)

            flux_minus_all_v, flux_plus_all_v, flux_minus_midpt_all_v, flux_plus_midpt_all_v = out_ref_fluxes

            flux_net_v_layer += (np.sum(flux_plus_midpt_all_v,axis=3)-np.sum(flux_minus_midpt_all_v,axis=3))*gauss_wts[ig]
            flux_net_v += (np.sum(flux_plus_all_v,axis=3)-np.sum(flux_minus_all_v,axis=3))*gauss_wts[ig]

            ======="""
            #nlevel = atm.c.nlevel

            ng_clima,nt_clima=1,1
            ubar0_clima = ubar0*0+0.5
            ubar1_clima = ubar1*0+0.5

            RSFV = 0.01 # from tgmdat.f of EGP
            
            b_surface = 0.0 +RSFV*ubar0[0]*FOPI*np.exp(-TAU[-1,:,ig]/ubar0[0])
            
            delta_approx = 0 # assuming delta approx is already applied on opds 
                        
            flux_minus_all_v, flux_plus_all_v, flux_minus_midpt_all_v, flux_plus_midpt_all_v = get_reflected_1d_gfluxv(nlevel, wno,nwno, ng_clima,nt_clima, DTAU[:,:,ig], TAU[:,:,ig], W0[:,:,ig], COSB[:,:,ig],
                                                                                       surf_reflect,b_top,b_surface,ubar0_clima, FOPI,tridiagonal, delta_approx)
            

            flux_net_v_layer += (np.sum(flux_plus_midpt_all_v,axis=3)-np.sum(flux_minus_midpt_all_v,axis=3))*gauss_wts[ig]
            flux_net_v += (np.sum(flux_plus_all_v,axis=3)-np.sum(flux_minus_all_v,axis=3))*gauss_wts[ig]

            flux_plus_v += flux_plus_all_v*gauss_wts[ig]
            flux_minus_v += flux_minus_all_v*gauss_wts[ig]

        #if full output is requested add in xint at top for 3d plots


    if thermal:

        #use toon method (and tridiagonal matrix solver) to get net cumulative fluxes 
        
        for ig in range(ngauss): # correlated - loop (which is different from gauss-tchevychev angle)
            
            #remember all OG values (e.g. no delta eddington correction) go into thermal as well as 
            #the uncorrected raman single scattering 
            
            #"""<<<<<<< NEWCLIMA
            hard_surface = 0 
            _,out_therm_fluxes = get_thermal_1d(nlevel, wno,nwno,ng,nt,temperature,
                                            DTAU_OG[:,:,ig], W0_no_raman[:,:,ig], COSB_OG[:,:,ig], 
                                            pressure,ubar1,
                                            surf_reflect, hard_surface, dwni, calc_type=1)

            flux_minus_all_i, flux_plus_all_i, flux_minus_midpt_all_i, flux_plus_midpt_all_i = out_therm_fluxes

            flux_plus += flux_plus_all_i*gauss_wts[ig]
            flux_minus += flux_minus_all_i*gauss_wts[ig]
            flux_plus_midpt += flux_plus_midpt_all_i*gauss_wts[ig]#*weights
            flux_minus_midpt += flux_minus_midpt_all_i*gauss_wts[ig]#*weights
            #"""
            
            """<<<<<<< OG CODE
            calc_type=1 # this line might change depending on Natasha's new function
            
            #for iubar,weights in zip(ugauss_angles,ugauss_weights):
            flux_minus_all_i, flux_plus_all_i, flux_minus_midpt_all_i, flux_plus_midpt_all_i=get_thermal_1d_gfluxi(nlevel,wno,nwno,ng,nt,temperature,DTAU_OG[:,:,ig], W0_no_raman[:,:,ig], COSB_OG[:,:,ig], pressure,ubar1,surf_reflect, ugauss_angles,ugauss_weights, tridiagonal,calc_type, dwni)#,bb , y2, tp, tmin, tmax)

            flux_plus += flux_plus_all_i*gauss_wts[ig]#*weights
            flux_minus += flux_minus_all_i*gauss_wts[ig]#*weights

            flux_plus_midpt += flux_plus_midpt_all_i*gauss_wts[ig]#*weights
            flux_minus_midpt += flux_minus_midpt_all_i*gauss_wts[ig]#*weights
            """

        #"""<<<<<<< NEWCLIMA
        #compresses in gauss-chebyshev angle space 
        #the integration over the "disk" of the planet opposed to the 
        #other gauss angles which are for the correlatedk tables
        #gweight = np.array([0.01574791, 0.07390887, 0.14638699, 0.16717464, 0.09678159])
        #tweight = np.array([1])#[6.28318531])
        flux_plus = compress_thermal(nwno, flux_plus, gweight, tweight)
        flux_minus= compress_thermal(nwno, flux_minus, gweight, tweight)
        flux_plus_midpt= compress_thermal(nwno, flux_plus_midpt, gweight, tweight)
        flux_minus_midpt= compress_thermal(nwno, flux_minus_midpt, gweight, tweight)
        #"""

        for wvi in range(nwno):
            flux_net_ir_layer += (flux_plus_midpt[:,wvi]-flux_minus_midpt[:,wvi]) * dwni[wvi]
            flux_net_ir += (flux_plus[:,wvi]-flux_minus[:,wvi]) * dwni[wvi]

            flux_plus_ir[:,wvi] += flux_plus[:,wvi] * dwni[wvi]
            flux_minus_ir[:,wvi] += flux_minus[:,wvi] * dwni[wvi]
        """
        print('debug fluxes in get_fluxes', temperature)
        for wvi in range(nwno):
            for il in range(len(flux_plus_midpt[:,0])):
                print(wvi, dwni[wvi],flux_plus_midpt[il,wvi],flux_minus_midpt[il,wvi] )
        """

        #if full output is requested add in flux at top for 3d plots
    
    return flux_net_v_layer, flux_net_v, flux_plus_v, flux_minus_v , flux_net_ir_layer, flux_net_ir, flux_plus_ir, flux_minus_ir

#soon I will deprecate the function name "climate" as it is really confusing with what it 
#actually does, which is just run the RT to get fluxes
climate = get_fluxes

def calculate_atm(bundle, opacityclass):

    inputs = bundle.inputs

    wno = opacityclass.wno
    nwno = opacityclass.nwno
    ngauss = opacityclass.ngauss
    gauss_wts = opacityclass.gauss_wts #for opacity

    #check to see if we are running in test mode
    test_mode = inputs['test_mode']

    ############# DEFINE ALL APPROXIMATIONS USED IN CALCULATION #############
    #see class `inputs` attribute `approx`

    #set approx numbers options (to be used in numba compiled functions)
    single_phase = inputs['approx']['rt_params']['toon']['single_phase']
    multi_phase = inputs['approx']['rt_params']['toon']['multi_phase']
    raman_approx =inputs['approx']['rt_params']['common']['raman']
    method = inputs['approx']['rt_method']
    stream = inputs['approx']['rt_params']['common']['stream']
    tridiagonal = 0 

    #parameters needed for the two term hg phase function. 
    #Defaults are set in config.json
    f = inputs['approx']['rt_params']['common']['TTHG_params']['fraction']
    frac_a = f[0]
    frac_b = f[1]
    frac_c = f[2]
    constant_back = inputs['approx']['rt_params']['common']['TTHG_params']['constant_back']
    constant_forward = inputs['approx']['rt_params']['common']['TTHG_params']['constant_forward']

    #define delta eddington approximinations 
    delta_eddington = inputs['approx']['rt_params']['common']['delta_eddington']

    #pressure assumption
    p_reference =  inputs['approx']['p_reference']

    ############# DEFINE ALL GEOMETRY USED IN CALCULATION #############
    #see class `inputs` attribute `phase_angle`
    

    #phase angle 
    phase_angle = inputs['phase_angle']
    #get geometry
    geom = inputs['disco']

    #""" NEWCLIMA
    ng, nt = geom['num_gangle'], geom['num_tangle']#1,1 #
    gangle,gweight,tangle,tweight = geom['gangle'], geom['gweight'],geom['tangle'], geom['tweight']
    lat, lon = geom['latitude'], geom['longitude']
    cos_theta = geom['cos_theta']
    ubar0, ubar1 = geom['ubar0'], geom['ubar1']
    #"""
    """ OG Code
    ng, nt = 1,1
    gangle,gweight,tangle,tweight = geom['gangle'], geom['gweight'],geom['tangle'], geom['tweight']
    lat, lon = geom['latitude'], geom['longitude']
    cos_theta = geom['cos_theta']
    ubar0,ubar1 = np.zeros((5,1)),np.zeros((5,1))
    ubar0 += 0.5
    ubar1 += 0.5
    """
    #set star parameters
    radius_star = inputs['star']['radius']

    #semi major axis
    sa = inputs['star']['semi_major']

    #begin atm setup
    atm = ATMSETUP(inputs)

    #Add inputs to class 
    ##############################
    atm.surf_reflect = 0#inputs['surface_reflect']
    ##############################
    atm.wavenumber = wno
    atm.planet.gravity = inputs['planet']['gravity']
    atm.planet.radius = inputs['planet']['radius']
    atm.planet.mass = inputs['planet']['mass']

    #if dimension == '1d':
    atm.get_profile()
    #elif dimension == '3d':
    #    atm.get_profile_3d()

    #now can get these 
    atm.get_mmw()
    atm.get_density()
    atm.get_altitude(p_reference = p_reference)#will calculate altitude if r and m are given (opposed to just g)
    atm.get_column_density()

    #gets both continuum and needed rayleigh cross sections 
    #relies on continuum molecules are added into the opacity 
    #database. Rayleigh molecules are all in `rayleigh.py` 
    
    atm.get_needed_continuum(opacityclass.rayleigh_molecules,
                             opacityclass.avail_continuum)

    #get cloud properties, if there are any and put it on current grid 
    atm.get_clouds(wno)

    #Make sure that all molecules are in opacityclass. If not, remove them and add warning
    no_opacities = [i for i in atm.molecules if i not in opacityclass.molecules]
    atm.add_warnings('No computed opacities for: '+','.join(no_opacities))
    atm.molecules = np.array([ x for x in atm.molecules if x not in no_opacities ])

    nlevel = atm.c.nlevel
    nlayer = atm.c.nlayer
    
    
    opacityclass.get_opacities(atm)
    
    DTAU, TAU, W0, COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman, f_deltaM= compute_opacity(
            atm, opacityclass, ngauss=ngauss, stream=stream, delta_eddington=delta_eddington,test_mode=test_mode,raman=raman_approx,
            full_output=False, plot_opacity=False)
    

    #mmw = np.mean(atm.layer['mmw'])
    mmw = atm.layer['mmw']
    
    return DTAU, TAU, W0, COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman , atm.surf_reflect, ubar0,ubar1,cos_theta, single_phase,multi_phase,frac_a,frac_b,frac_c,constant_back,constant_forward, tridiagonal , wno,nwno,ng,nt, nlevel, ngauss, gauss_wts, mmw,gweight,tweight


def calculate_atm_deq(bundle, opacityclass,on_fly=False,gases_fly=None):

    inputs = bundle.inputs

    wno = opacityclass.wno
    nwno = opacityclass.nwno
    ngauss = opacityclass.ngauss
    gauss_wts = opacityclass.gauss_wts #for opacity

    #check to see if we are running in test mode
    test_mode = inputs['test_mode']

    ############# DEFINE ALL APPROXIMATIONS USED IN CALCULATION #############
    #see class `inputs` attribute `approx`

    #set approx numbers options (to be used in numba compiled functions)
    single_phase = inputs['approx']['rt_params']['toon']['single_phase']
    multi_phase = inputs['approx']['rt_params']['toon']['multi_phase']
    raman_approx =inputs['approx']['rt_params']['common']['raman']
    method = inputs['approx']['rt_method']
    stream = inputs['approx']['rt_params']['common']['stream']
    tridiagonal = 0 

    #parameters needed for the two term hg phase function. 
    #Defaults are set in config.json
    f = inputs['approx']['rt_params']['common']['TTHG_params']['fraction']
    frac_a = f[0]
    frac_b = f[1]
    frac_c = f[2]
    constant_back = inputs['approx']['rt_params']['common']['TTHG_params']['constant_back']
    constant_forward = inputs['approx']['rt_params']['common']['TTHG_params']['constant_forward']

    #define delta eddington approximinations 
    delta_eddington = inputs['approx']['rt_params']['common']['delta_eddington']

    #pressure assumption
    p_reference =  inputs['approx']['p_reference']

    ############# DEFINE ALL GEOMETRY USED IN CALCULATION #############
    #see class `inputs` attribute `phase_angle`
    

    #phase angle 
    phase_angle = inputs['phase_angle']
    #get geometry
    geom = inputs['disco']

    ng, nt = 1,1 #geom['num_gangle'], geom['num_tangle']
    gangle,gweight,tangle,tweight = geom['gangle'], geom['gweight'],geom['tangle'], geom['tweight']
    lat, lon = geom['latitude'], geom['longitude']
    cos_theta = geom['cos_theta']
    #ubar0, ubar1 = geom['ubar0'], geom['ubar1']
    #print(np.shape(ubar0),ubar0[0])
    ubar0,ubar1 = np.zeros((5,1)),np.zeros((5,1))
    ubar0 += 0.5
    ubar1 += 0.5
    #print(ubar0,ubar1)

    #set star parameters
    radius_star = inputs['star']['radius']
    #F0PI = np.zeros(nwno) + 1.
    #semi major axis
    sa = inputs['star']['semi_major']

    #begin atm setup
    atm = ATMSETUP(inputs)

    #Add inputs to class 
    ##############################
    atm.surf_reflect = 0#inputs['surface_reflect']
    ##############################
    atm.wavenumber = wno
    atm.planet.gravity = inputs['planet']['gravity']
    atm.planet.radius = inputs['planet']['radius']
    atm.planet.mass = inputs['planet']['mass']

    #if dimension == '1d':
    atm.get_profile()
    #elif dimension == '3d':
    #    atm.get_profile_3d()

    #now can get these 
    atm.get_mmw()
    atm.get_density()
    atm.get_altitude(p_reference = p_reference)#will calculate altitude if r and m are given (opposed to just g)
    atm.get_column_density()

    #gets both continuum and needed rayleigh cross sections 
    #relies on continuum molecules are added into the opacity 
    #database. Rayleigh molecules are all in `rayleigh.py` 
    
    atm.get_needed_continuum(opacityclass.rayleigh_molecules,
                             opacityclass.avail_continuum)

    #get cloud properties, if there are any and put it on current grid 
    atm.get_clouds(wno)

    #Make sure that all molecules are in opacityclass. If not, remove them and add warning
    no_opacities = [i for i in atm.molecules if i not in opacityclass.molecules]
    atm.add_warnings('No computed opacities for: '+','.join(no_opacities))
    atm.molecules = np.array([ x for x in atm.molecules if x not in no_opacities ])

    nlevel = atm.c.nlevel
    nlayer = atm.c.nlayer
    
    if on_fly == False:
        opacityclass.get_opacities_deq(bundle,atm)
    else:
        opacityclass.get_opacities_deq_onfly(bundle,atm,gases_fly=gases_fly)
    
    DTAU, TAU, W0, COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman, f_deltaM= compute_opacity(
            atm, opacityclass, ngauss=ngauss, stream=stream, delta_eddington=delta_eddington,test_mode=test_mode,raman=raman_approx,
            full_output=False, plot_opacity=False)
    

    #mmw = np.mean(atm.layer['mmw'])
    mmw = atm.level['mmw']
    
    return DTAU, TAU, W0, COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman , atm.surf_reflect, ubar0,ubar1,cos_theta, single_phase,multi_phase,frac_a,frac_b,frac_c,constant_back,constant_forward, tridiagonal , wno,nwno,ng,nt, nlevel, ngauss, gauss_wts, mmw