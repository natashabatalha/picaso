import numpy as np 
from numba import jit, vectorize
from numpy import exp, zeros, where, sqrt, cumsum , pi, outer, sinh, cosh, min, dot, array,log,log10


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
        array of Pressure values with 53 entries
    p_table : array 
        array of Pressure value with 26 entries
    grad : array 
        array of gradients of dimension 53*26
    cp : array 
        array of cp of dimension 53*26
    calc_type : int 
        if 0 will return both gradx,cpx , if 1 will return only gradx, if 2 will return only cpx
    
    Returns
    -------
    float 
        if calc_type= 0 grad_x,cp_x
        if calc_type= 1 grad_x 
        if calc_type= 2 cp_x
    
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
    
    if calc_type == 0:
        return grad_x,cp_x
    elif calc_type == 1:
        return grad_x
    elif calc_type == 2 :
        return cp_x


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
            raise ValueError("Array is singular, cannot be decomposed")
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



@jit(nopython=True, cache=True)
def t_start(nofczns,nstr,it_max,conv,x_max_mult, bundle, opacityclass, 
            rfaci, rfacv, nlevel, temp, pressure, p_table, t_table, 
            grad, cp, tidal, tmin,tmax):
    """
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

    bundle : dict 
        This input dict is built by loading the input = `justdoit.load_inputs()` 
    opacityclass : class
        Opacity class from `justdoit.opannection`
    rfaci : float 
        IR flux addition fraction 
    rfacv : float
        Visible flux addition fraction
    nlevel : int
        # of levels
    temp : array
        Temperature array, dimension is nlevel
    pressure : array
        Pressure array
    p_table : array

    t_table : array

    grad : array

    cp : array

    tidal : array
        Tidal Fluxes dimension = nlevel   
        
    Returns
    -------
    array 
        B array of dimension n*n

    """
    #     Routine to iteratively solve for T(P) profile.  Uses a Newton-
    #     Raphson iteration to zero out the net flux in the radiative
    #     zone above NSTRAT.  Repeats until average temperature change
    #     is less than CONV or until ITMAX is reached.

    # -- SM -- needs a lot of documentation

    # -- SM -- lots of array indexing so fortran to python conversion might not
    # be perfect. check that while benchmarking. 
    #  will have to debug by running line by line and printing indices between f an py

    eps=1e-4

    n_top_r=nstr[0]-1

    # here are other  convergence and tolerance criterias

    step_max = 0.03e0 # scaled maximum step size in line searches
    alf = 1.e-4    # ? 
    alam2 = 0.0   # ? 
    tolmin=1.e-5   # ?
    tolf = 5e-3    # tolerance in fractional Flux we are aiming for
    tolx = tolf    # tolerance in fractional T change we are aiming for

    # first we need a call to toon modules to get RT fluxes
    # using climate = True and calculation = ['reflected','thermal'] is
    # equivalent to calling SFLUXI and SFLUXV
    returns= climate(bundle,opacityclass, pressure, temp, dimension = '1d',calculation=['reflected','thermal'], climate = True,                        full_output=False, plot_opacity= False, as_dict=True)

    
    # extract visible fluxes
    flux_net_v_layer = returns['flux_vis_net_layer']  #fmnetv
    flux_net_v = returns['flux_vis_net_level']        #fnetv
    flux_plus_v = returns['flux_vis_plus_level'] 
    flux_minus_v = returns['flux_vis_minus_level'] 

    # extract ir fluxes

    flux_net_ir_layer = returns['flux_ir_net_layer'] #fmneti
    flux_net_ir = returns['flux_ir_net_level']     #fneti
    flux_plus_ir = returns['flux_ir_plus_level']   
    flux_minus_ir = returns['flux_ir_minus_level'] 
    
    # arrays for total net fluxes = optical+ir + tidal
    flux_net=np.zeros(shape=(nlevel))
    flux_net_midpt=np.zeros(shape=(nlevel))
    dflux=np.zeros(shape=(nlevel))
    f_vec=np.zeros(shape=(nlevel)) #fvec
    p=np.zeros(shape=(nlevel)) #p
    g=np.zeros(shape=(nlevel))
    
    #--SM-- jacobian?
    A= np.zeros(shape=(nlevel,nlevel)) 
    
    for its in range(it_max):

        # the total net flux = optical + ir + tidal component
        flux_net = rfaci* flux_net_ir + rfacv* flux_net_v +tidal #fnet
        flux_net_midpt = rfaci* flux_net_ir_layer + rfacv* flux_net_v_layer +tidal #fmnet
        beta= temp # beta vector
        
       
        # store old fluxes and temp before iteration
        # do not store the ir+vis flux because we are going to perturb only thermal structure

        
        temp_old= temp
        flux_net_old = flux_net_ir #fnetip
        flux_net_midpt_old= flux_net_ir_layer #fmip

        nao = n_top_r
        n_total = 0 #ntotl

        sum = 0.0
        sum_1 = 0.0 # sum1
        test = 0.0

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
            if nao < 0 :
                nao_temporary = nao
                nao == 0 # to avoid negative of negative index leading to wrong indexing, fort vs. py
                flag_nao = 1 # so that it can be reversed back to previous value after loop
            for j in range(n_top_a+1, n_strt_a):
                
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
            print(" We are already at a root, tolf , test = " +str(tolf)+", " + str(test))
            flag_converge = 2
        
        # define maximum T step size
        step_max *= np.amax([sqrt(sum_1),float(n_total)])

        no =n_top_r
        
        i_count= 1 #icount
       
        for nz in range(0, 3*nofczns, 3):

            n_top = nstr[nz] +1 #ntop
            n_strt = nstr[nz+1] #nstrt
            n_conv_top = n_strt + 1 #ncnvtop
            n_conv_bot= nstr[nz+2] +1 #ncnvbot

            if nz == 0 :
                n_top -= 1
            
        # begin jacobian calculation here
            for jm in range(n_top, n_strt):

                # chose perturbation for each level

                i_count += 1

                del_t = eps * temp_old[jm] # perturbation

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
                        calc_type =  1 # only need grad_x in return
                        grad_x = did_grad_cp( beta[j1-1], press, t_table, p_table, grad, cp, calc_type)
                        
                        temp[j1]= exp(log(temp[j1-1]) + grad_x*(log(pressure[j1]) - log(pressure[j1-1])))
                
                

                # temperature has been perturbed
                # now recalculate the IR fluxes, so call picaso with only thermal

                returns= climate(bundle,opacityclass, pressure, temp, dimension = '1d',calculation=['thermal'], climate = True,                               full_output=False, plot_opacity= False, as_dict=True)
                
                # new_fluxes after perturbations (old are stored in diff array)
                flux_net_ir_layer = returns['flux_ir_net_layer'] #fmneti
                flux_net_ir = returns['flux_ir_net_level']     #fneti
                flux_plus_ir = returns['flux_ir_plus_level']   
                flux_minus_ir = returns['flux_ir_minus_level'] 
                
                # now calculate jacobian terms in the same way as dflux
                nco = n_top_r

                for nc in range(0,3*nofczns, 3):
                    
                    n_top_c = nstr[nc] +1 # ntopc

                    if nc ==1:
                        n_top_c -= 1
                    n_strt_c = nstr[nc+1]
                    n_bot_c = nstr[nc+2] +1
                    
                    # -ve nco and no will mess indexing
                    # so we want to set them to 0 temporarily if that occurs
                    if nco < 0 :
                        nco_temporary = nco
                        nco == 0 # to avoid negative of negative index leading to wrong indexing, fort vs. py
                        flag_nco = 1 # so that it can be reversed back to previous value after loop
                    if no < 0 :
                        no_temporary = nco
                        no == 0 # to avoid negative of negative index leading to wrong indexing, fort vs. py
                        flag_no = 1 # so that it can be reversed back to previous value after loop
                    
                    if n_top_c == n_top_r+1 :

                        A[n_top_c-nco,jm-no] = (flux_net_ir[n_top_c]-flux_net_old[n_top_c])/del_t

                    else:

                        A[n_top_c-nco,jm-no] = (flux_net_ir_layer[n_top_c-1]-flux_net_midpt_old[n_top_c-1])/del_t
                    
                    
                    
                    # omitted -1 to include last element 
                    for im in range(n_top_c,n_strt_c):
                        
                        A[im+1-nco,jm-no] = (flux_net_ir_layer[im]-flux_net_midpt_old[im])/del_t
                    
                    # changing them back to what they were
                    if flag_nco == 1 :
                        nco= nco_temporary
                        flag_nco = 0
                    if flag_no == 1 :
                        no= no_temporary
                        flag_no = 0

                    nco+= n_bot_c-n_strt_c
                

                # undo beta vector perturbation
                beta[jm] = beta[jm] - del_t
            no += n_conv_bot-n_strt
        
        # a long print statement here in original. dont know if needed

        
        for i in range(n_total):
            sum=0.0
            for j in range(n_total):
                sum += A[j,i]*f_vec[j]
            
            g[i] = sum

            p[i] = -f_vec[i]
        
        f_old = f #fold

        A, p = mat_sol(A, nlevel, n_total, p)

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
        if slope >= 0.0 :
            raise ValueError("roundoff problem in linen search")
        
        ## checked till here -- SM
        test = 0.0
        
        for i in range(n_total):
            tmp = abs(p[i])/temp_old[i]
            if tmp > test :
                test= tmp 

        alamin = tolx/test
        alam = 1.0

        ## stick a while loop here maybe for the weird fortran goto 1
        # you have in tstart.
        flag_converge = 0
        # instead of the goto statement here
        while flag_converge == 0 :
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
            
            err= err/(n_total*scalt)

            if jmx > nstr[1] :
                jmx+= nstr[2]-nstr[1]
            
            ndo = n_top_r
            #+1 to include
            for nd in range(0,3*nofczns, 3):
                n_top_d = nstr[nd] +1
                if nd == 0:
                    n_top_d -= 1

                n_strt_d = nstr[nd+1]

                n_bot_d= nstr[nd+2] +1
                
                if ndo < 0 :
                    ndo_temporary = ndo
                    ndo == 0 # to avoid negative of negative index leading to wrong indexing, fort vs. py
                    flag_ndo = 1 # so that it can be reversed back to previous value after loop
                   
                #+1 for fort to py
                for j in range(n_top_d,n_strt_d+1):
                    temp[j]= beta[j]+ alam*p[j-ndo]
                #+1 for fort to py
                for j1 in range(n_strt_d+1, n_bot_d+1):

                    press = sqrt(pressure[j1-1]*pressure[j1])
                    calc_type =  1 # only need grad_x in return
                    grad_x = did_grad_cp( temp[j1-1], press, t_table, p_table, grad, cp, calc_type)
                            
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
            returns= climate(bundle,opacityclass, pressure, temp, dimension = '1d',calculation=['thermal'], climate = True,                                 full_output=False, plot_opacity= False, as_dict=True)                    
                    # new_fluxes after perturbations (old are stored in diff array)
            flux_net_ir_layer = returns['flux_ir_net_layer'] #fmneti
            flux_net_ir = returns['flux_ir_net_level']     #fneti
            flux_plus_ir = returns['flux_ir_plus_level']   
            flux_minus_ir = returns['flux_ir_minus_level'] 
            
            # re calculate net fluxes
            flux_net = rfaci* flux_net_ir + rfacv* flux_net_v +tidal #fnet
            flux_net_midpt = rfaci* flux_net_ir_layer + rfacv* flux_net_v_layer +tidal #fmnet
            
            sum = 0.0
            nao = n_top_r

            for nca in range(0,3*nofczns,3):
                n_top_a = nstr[nca] + 1
                if nca ==0 :
                    n_top_a -= 1
                
                n_strt_a=nstr[nca+1]
                n_bot_a = nstr[nca+2] + 1

                if nao < 0 :
                    nao_temporary = nao
                    nao == 0 # to avoid negative of negative index leading to wrong indexing, fort vs. py
                    flag_nao = 1 # so that it can be reversed back to previous value after loop
                   

                if n_top_a == n_top_r +1 :
                    
                    f_vec[0]= flux_net[n_top_r +1]
                    sum += f_vec[0]**2
                else:
                    f_vec[n_top_a-nao] = flux_net_midpt[n_top_a -1]
                    sum += f_vec[n_top_a - nao]**2
                
                for j in range(n_top_a+1,n_strt_a):
                    
                    f_vec[j-nao] = flux_net_midpt[j -1]
                    sum += f_vec[j-nao]**2
                
                if flag_nao == 1 :
                        nao= nao_temporary
                        flag_nao = 0
                
                nao+= n_bot_a - n_strt_a
            
            f= 0.5*sum
            # check_convergence is fortran from line indexed 9995 till next line of 19
            if alam < alamin :
                check = True
                
                flag_converge, check = check_convergence(f_vec, n_total, tolf, check, f, dflux, tolmin, temp, temp_old, tolx)
 
            
            elif f <= f_old :
                flag_converge, check = check_convergence(f_vec, n_total, tolf, check, f, dflux, tolmin, temp, temp_old, tolx)

            
            else:
                # we backtrack

                if alam == 1:
                    tmplam= -slope/ (2*(f-f_old-slope))
                else:
                    rhs_1 = f- f_old - alam*slope
                    rhs_2 = f2 - f_old - alam2*slope
                    anr= ((rhs_1/alam**2)-(rhs_2/alam2**2))/(alam-alam2)
                    b= ((-alam2*rhs_1/alam**2)+(alam*rhs_2/alam2**2))/(alam-alam2)
                    
                    if anr == 0 :
                        tmplam= -slope/(2*b)
                    else:
                        disc= b*b - 3.0*anr*slope

                        if disc < 0.0 :
                            tmplam= 0.5*alam
                        elif b <= 0.0:
                            tmplam=(-b + sqrt(disc))/(3*anr)
                        else:
                            tmplam= -slope/(b+sqrt(disc))
                    
                    if tmplam > 0.5*alam:
                        tmplam= 0.5*alam
            
            alam2=alam
            f2=f

            alam = np.amax([tmplam,0.1*alam])

        if flag_converge == 2 : # converged
            # calculate adiabatic lapse rate
            dtdp=np.zeros(shape=(nlevel-1))
            for j in range(nlevel -1):
                dtdp[j] = (log( temp[j]) - dlog( temp[j+1]))/(log(pressure[j]) - log(pressure[J+1]))
            
            print("Converged Solution")
            return   temp , dtdp 

    print("Iterations exceeded it_max ! sorry")
    return temp, temp    



def check_convergence(f_vec, n_total, tolf, check, f, dflux, tolmin, temp, temp_old, g , tolx):

    test = 0.0
    for i in range(n_total):
        if abs(f_vec[i]) > test:
            test=abs(f_vec[i])
    
    if test < tolf :
        check = False
        flag = 2
        return flag , check
    if check == True :
        test = 0.0
        den1 = np.amax([f,0.5*float(n_total)])
        
        for i in range(n_total):
            tmp= abs(g[i])*dflux[i]/den1
            if tmp > test:
                test= tmp
        
        if test < tolmin :
            check= True
        else :
            check= False
        flag = 2
        return flag, check
    
    test = 0.0
    for i in range(n_total):
        tmp = (abs(temp[i]-temp_old[i]))/temp_old[i]
        if tmp > test:
            test=tmp
    if test < tolx :
        flag = 2

        return flag, check
    

    flag = 1
    return flag , check


def climate(bundle,opacityclass, pressure, temperature, dimension = '1d',calculation='reflected', climate = False, full_output=False, 
    plot_opacity= False, as_dict=True):
    """
    Currently top level program to run RT for climate calculations.
    If opacities need to change while
    calling this function one has to make those changes in bundle and opacityclass

    Parameters 
    ----------
    bundle : dict 
        This input dict is built by loading the input = `justdoit.load_inputs()` 
    opacityclass : class
        Opacity class from `justdoit.opannection`
    dimension : str 
        (Optional) Dimensions of the calculation. Default = '1d'. But '3d' is also accepted. 
        In order to run '3d' calculations, user must build 3d input (see tutorials)
    climate: bool 
        (Optional) if True then module is used for climate calculation
    full_output : bool 
        (Optional) Default = False. Returns atmosphere class, which enables several 
        plotting capabilities. 
    plot_opacity : bool 
        (Optional) Default = False, Creates pop up of the weighted opacity
    as_dict : bool 
        (Optional) Default = True. If true, returns a condensed dictionary to the user. 
        If false, returns the atmosphere class, which can be used for debugging. 
        The class is clunky to navigate so if you are consiering navigating through this, ping one of the 
        developers. 

    Return
    ------
    dictionary with albedos or fluxes or both (depending on what calculation type)
    """
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
    single_phase = inputs['approx']['single_phase']
    multi_phase = inputs['approx']['multi_phase']
    raman_approx =inputs['approx']['raman']
    method = inputs['approx']['method']
    stream = inputs['approx']['stream']
    tridiagonal = 0 

    #parameters needed for the two term hg phase function. 
    #Defaults are set in config.json
    f = inputs['approx']['TTHG_params']['fraction']
    frac_a = f[0]
    frac_b = f[1]
    frac_c = f[2]
    constant_back = inputs['approx']['TTHG_params']['constant_back']
    constant_forward = inputs['approx']['TTHG_params']['constant_forward']

    #define delta eddington approximinations 
    delta_eddington = inputs['approx']['delta_eddington']

    #pressure assumption
    p_reference =  inputs['approx']['p_reference']

    ############# DEFINE ALL GEOMETRY USED IN CALCULATION #############
    #see class `inputs` attribute `phase_angle`
    

    #phase angle 
    phase_angle = inputs['phase_angle']
    #get geometry
    geom = inputs['disco']

    ng, nt = geom['num_gangle'], geom['num_tangle']
    gangle,gweight,tangle,tweight = geom['gangle'], geom['gweight'],geom['tangle'], geom['tweight']
    lat, lon = geom['latitude'], geom['longitude']
    cos_theta = geom['cos_theta']
    ubar0, ubar1 = geom['ubar0'], geom['ubar1']

    #set star parameters
    radius_star = inputs['star']['radius']
    F0PI = np.zeros(nwno) + 1.
    #semi major axis
    sa = inputs['star']['semi_major']

    #begin atm setup
    atm = ATMSETUP(inputs)

    #Add inputs to class 
    atm.surf_reflect = inputs['surface_reflect']
    atm.wavenumber = wno
    atm.planet.gravity = inputs['planet']['gravity']
    atm.planet.radius = inputs['planet']['radius']
    atm.planet.mass = inputs['planet']['mass']

    if dimension == '1d':
        atm.get_profile()
    elif dimension == '3d':
        atm.get_profile_3d()

    #now can get these 
    atm.get_mmw()
    atm.get_density()
    atm.get_altitude(p_reference = p_reference)#will calculate altitude if r and m are given (opposed to just g)
    atm.get_column_density()

    #gets both continuum and needed rayleigh cross sections 
    #relies on continuum molecules are added into the opacity 
    #database. Rayleigh molecules are all in `rayleigh.py` 
    atm.get_needed_continuum(opacityclass.rayleigh_molecules)

    #get cloud properties, if there are any and put it on current grid 
    atm.get_clouds(wno)

    #Make sure that all molecules are in opacityclass. If not, remove them and add warning
    no_opacities = [i for i in atm.molecules if i not in opacityclass.molecules]
    atm.add_warnings('No computed opacities for: '+','.join(no_opacities))
    atm.molecules = np.array([ x for x in atm.molecules if x not in no_opacities ])

    nlevel = atm.c.nlevel
    nlayer = atm.c.nlayer
    

    if dimension == '1d':
        #lastly grab needed opacities for the problem
        opacityclass.get_opacities(atm)
        #only need to get opacities for one pt profile

        #There are two sets of dtau,tau,w0,g in the event that the user chooses to use delta-eddington
        #We use HG function for single scattering which gets the forward scattering/back scattering peaks 
        #well. We only really want to use delta-edd for multi scattering legendre polynomials. 
        DTAU, TAU, W0, COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman= compute_opacity(
            atm, opacityclass, ngauss=ngauss, stream=stream, delta_eddington=delta_eddington,test_mode=test_mode,raman=raman_approx,
            full_output=full_output, plot_opacity=plot_opacity)


        if  'reflected' in calculation:
            #use toon method (and tridiagonal matrix solver) to get net cumulative fluxes 
            
            

            flux_net_v = np.zeros(ng,nt,nlevel) #net level visible fluxes
            flux_net_v_layer=np.zeros(ng,nt,nlevel) #net layer visible fluxes
        
            flux_plus_v= np.zeros(ng,nt,nlevel,nwno) # level plus visible fluxes
            flux_minus_v= np.zeros(ng,nt,nlevel,nwno) # level minus visible fluxes
       
               
            for ig in range(ngauss): # correlated - loop (which is different from gauss-tchevychev angle)
                nlevel = atm.c.nlevel
                
                
                calc_type=1
                    # this line might change depending on Natasha's new function
                flux_minus_all, flux_plus_all, flux_minus_midpt_all, flux_plus_midpt_all = get_reflected_1d(nlevel, wno,nwno,ng,nt,
                                DTAU[:,:,ig], TAU[:,:,ig], W0[:,:,ig], COSB[:,:,ig],
                                GCOS2[:,:,ig],ftau_cld[:,:,ig],ftau_ray[:,:,ig],
                                DTAU_OG[:,:,ig], TAU_OG[:,:,ig], W0_OG[:,:,ig], COSB_OG[:,:,ig],
                                atm.surf_reflect, ubar0,ubar1,cos_theta, F0PI,
                                single_phase,multi_phase,
                                frac_a,frac_b,frac_c,constant_back,constant_forward, tridiagonal,calc_type)
               
                
                

                flux_net_v_layer += (np.sum(flux_plus_midpt_all,axis=3)-np.sum(flux_minus_midpt_all,axis=3))*gauss_wts[ig]
                flux_net_v += (np.sum(flux_plus_all,axis=3)-np.sum(flux_minus_all,axis=3))*gauss_wts[ig]
                
                flux_plus_v += flux_plus_all*gauss_wts[ig]
                flux_minus_v += flux_minus_all*gauss_wts[ig]
                
            #if full output is requested add in xint at top for 3d plots
            

        if 'thermal' in calculation:

            #use toon method (and tridiagonal matrix solver) to get net cumulative fluxes 
             
            
                # total corr gauss weighted fluxes
            flux_plus_midpt = np.zeros(ng,nt,nlevel,nwno)
            flux_minus_midpt = np.zeros(ng,nt,nlevel,nwno)
            
            flux_plus = np.zeros(ng,nt,nlevel,nwno)
            flux_minus = np.zeros(ng,nt,nlevel,nwno)

            # outputs needed for climate
            flux_net_ir = np.zeros(ng,nt,nlevel) #net level visible fluxes
            flux_net_ir_layer=np.zeros(ng,nt,nlevel) #net layer visible fluxes
        
            flux_plus_ir= np.zeros(ng,nt,nlevel,nwno) # level plus visible fluxes
            flux_minus_ir= np.zeros(ng,nt,nlevel,nwno) # level minus visible fluxes
        

            for ig in range(ngauss): # correlated - loop (which is different from gauss-tchevychev angle)
                
                #remember all OG values (e.g. no delta eddington correction) go into thermal as well as 
                #the uncorrected raman single scattering 
                
                calc_type=1
                    # this line might change depending on Natasha's new function
                flux_minus_all, flux_plus_all, flux_minus_midpt_all, flux_plus_midpt_all= get_thermal_1d(nlevel, wno,nwno,ng,nt,temperature,
                                        DTAU_OG[:,:,ig], W0_no_raman[:,:,ig], COSB_OG[:,:,ig], 
                                        pressure,ubar1,
                                        atm.surf_reflect, tridiagonal,calc_type)
                
                
                flux_plus += flux_plus_all*gauss_wts[ig]
                flux_minus += flux_minus_all*gauss_wts[ig]

                flux_plus_midpt += flux_plus_midpt*gauss_wts[ig]
                flux_minus_midpt += flux_minus_midpt*gauss_wts[ig]
                
                    
            
            # SM - What is this dwni interval correction ?
            for wvi in range(nwno):
                flux_net_ir_layer += (flux_plus_midpt_all[:,:,:,wvi]-flux_minus_midpt_all[:,:,:,wvi]) * dwni[wvi]
                flux_net_ir += (flux_plus_all[:,:,:,wvi]-flux_minus_all[:,:,:,wvi]) * dwni[wvi]
            
                flux_plus_ir += flux_plus_all[:,wvi] * dwni[wvi]
                flux_minus_ir += flux_minus_all[:,wvi] * dwni[wvi]
                
            #if full output is requested add in flux at top for 3d plots
            
        
        


    #COMPRESS FULL TANGLE-GANGLE FLUX OUTPUT ONTO 1D FLUX GRID

    #set up initial returns
    returns = {}
    returns['wavenumber'] = wno
    

        
    if ((dimension == '1d') & ('reflected' in calculation)):
        returns['flux_vis_net_layer'] = flux_net_v_layer
        returns['flux_vis_net_level'] = flux_net_v
        returns['flux_vis_plus_level'] = flux_plus_v
        returns['flux_vis_minus_level'] = flux_minus_v
    if ((dimension == '1d') & ('thermal' in calculation)):
        returns['flux_ir_net_layer'] = flux_net_ir_layer
        returns['flux_ir_net_level'] = flux_net_ir
        returns['flux_ir_plus_level'] = flux_plus_ir
        returns['flux_ir_minus_level'] = flux_minus_ir

    return returns
@jit(nopython=True, cache=True)
def get_thermal_1d(nlevel, wno,nwno, numg,numt,tlevel, dtau, w0,cosb,plevel, ubar1,surf_reflect, tridiagonal, calc_type):
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
    tridiagonal : int 
        0 for tridiagonal, 1 for pentadiagonal
    calc_type : int 
        0 for outgoing flux at top level output, 1 for upward and downward layer and level flux outputs

    Returns
    -------
    numpy.ndarray
        Thermal flux in CGS units (erg/cm3/s) in a matrix that is 
        numg x numt x nwno if calc_type=0
    
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Thermal flux in CGS units (erg/cm3/s) in four matrices if calc_type=0: that is 
        level downward flux and level upward flux numg x numt x nlevel x nwno and then
        layer downward flux and layer upward flux numg x numt x nlayer x nwno  
    """
    nlayer = nlevel - 1 #nlayers 

    # Initialising Output Arrays
    flux_at_top = zeros((numg, numt, nwno)) # output when calc_type=0
    # outputs when calc_type =1
    flux_minus_all = zeros((numg, numt,nlevel, nwno)) ## level downwelling fluxes
    flux_plus_all = zeros((numg, numt, nlevel, nwno)) ## level upwelling fluxes
    flux_minus_midpt_all = zeros((numg, numt, nlayer, nwno)) ##  layer downwelling fluxes
    flux_plus_midpt_all = zeros((numg, numt, nlayer, nwno))  ## layer upwelling fluxes

    mu1 = 0.5#0.88#0.5 #from Table 1 Toon  
    twopi = pi#+pi #NEB REMOVING A PI FROM HERE BECAUSE WE ASSUME NO SYMMETRY! 

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
    exptrm_minus = 1.0/exptrm_positive

    tau_top = dtau[0,:]*plevel[0]/(plevel[1]-plevel[0]) #tried this.. no luck*exp(-1)# #tautop=dtau[0]*np.exp(-1)
    b_top = (1.0 - exp(-tau_top / mu1 )) * all_b[0,:]  # Btop=(1.-np.exp(-tautop/ubari))*B[0]
    #b_surface = all_b[-1,:] #for terrestrial, hard surface  
    b_surface=all_b[-1,:] + b1[-1,:]*mu1 #(for non terrestrial)

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
    L = nlayer+nlayer
    for w in range(nwno):
        #coefficient of posive and negative exponential terms 
        if tridiagonal==0:
            X = tri_diag_solve(L, A[:,w], B[:,w], C[:,w], D[:,w])
            #unmix the coefficients
            positive[:,w] = X[::2] + X[1::2] 
            negative[:,w] = X[::2] - X[1::2]
        #else:
        #   X = pent_diag_solve(L, A_[:,w], B_[:,w], C_[:,w], D_[:,w], E_[:,w], F_[:,w])
        #   positive[:,w] = exptrm_minus[:,w] * (X[::2] + X[1::2])
        #   negative[:,w] = X[::2] - X[1::2]

    #if you stop here this is regular ole 2 stream
    f_up = pi*(positive * exptrm_positive + gama * negative * exptrm_minus + c_plus_up)


    #calculate everyting from Table 3 toon
    alphax = ((1.0-w0)/(1.0-w0*cosb))**0.5
    G = twopi*w0*positive*(1.0+cosb*alphax)/(1.0+alphax)#
    H = twopi*w0*negative*(1.0-cosb*alphax)/(1.0+alphax)#
    J = twopi*w0*positive*(1.0-cosb*alphax)/(1.0+alphax)#
    K = twopi*w0*negative*(1.0+cosb*alphax)/(1.0+alphax)#
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
    
    

    #work through building eqn 55 in toon (tons of bookeeping exponentials)
    for ng in range(numg):
        for nt in range(numt): 

            iubar = ubar1[ng,nt]

            #flux_plus[-1,:] = twopi * (b_surface )# terrestrial
            flux_plus[-1,:] = twopi*( all_b[-1,:] + b1[-1,:] * iubar) #no hard surface
            flux_minus[0,:] = twopi * (1 - exp(-tau_top / iubar)) * all_b[0,:]
            
            exptrm_angle = exp( - dtau / iubar)
            exptrm_angle_mdpt = exp( -0.5 * dtau / iubar) 

            for itop in range(nlayer):

                #disbanning this for now because we dont need it in the thermal emission code
                flux_minus[itop+1,:]=(flux_minus[itop,:]*exptrm_angle[itop,:]+
                                     (J[itop,:]/(lamda[itop,:]*iubar+1.0))*(exptrm_positive[itop,:]-exptrm_angle[itop,:])+
                                     (K[itop,:]/(lamda[itop,:]*iubar-1.0))*(exptrm_angle[itop,:]-exptrm_minus[itop,:])+
                                     sigma1[itop,:]*(1.-exptrm_angle[itop,:])+
                                     sigma2[itop,:]*(iubar*exptrm_angle[itop,:]+dtau[itop,:]-iubar) )

                flux_minus_mdpt[itop,:]=(flux_minus[itop,:]*exptrm_angle_mdpt[itop,:]+
                                        (J[itop,:]/(lamda[itop,:]*iubar+1.0))*(exptrm_positive_mdpt[itop,:]-exptrm_angle_mdpt[itop,:])+
                                        (K[itop,:]/(-lamda[itop,:]*iubar+1.0))*(exptrm_minus_mdpt[itop,:]-exptrm_angle_mdpt[itop,:])+
                                        sigma1[itop,:]*(1.-exptrm_angle_mdpt[itop,:])+
                                        sigma2[itop,:]*(iubar*exptrm_angle_mdpt[itop,:]+0.5*dtau[itop,:]-iubar))

                ibot=nlayer-1-itop

                flux_plus[ibot,:]=(flux_plus[ibot+1,:]*exptrm_angle[ibot,:]+
                                  (G[ibot,:]/(lamda[ibot,:]*iubar-1.0))*(exptrm_positive[ibot,:]*exptrm_angle[ibot,:]-1.0)+
                                  (H[ibot,:]/(lamda[ibot,:]*iubar+1.0))*(1.0-exptrm_minus[ibot,:] * exptrm_angle[ibot,:])+
                                  alpha1[ibot,:]*(1.-exptrm_angle[ibot,:])+
                                  alpha2[ibot,:]*(iubar-(dtau[ibot,:]+iubar)*exptrm_angle[ibot,:]) )

                flux_plus_mdpt[ibot,:]=(flux_plus[ibot+1,:]*exptrm_angle_mdpt[ibot,:]+
                                       (G[ibot,:]/(lamda[ibot,:]*iubar-1.0))*(exptrm_positive[ibot,:]*exptrm_angle_mdpt[ibot,:]-exptrm_positive_mdpt[ibot,:])-
                                       (H[ibot,:]/(lamda[ibot,:]*iubar+1.0))*(exptrm_minus[ibot,:]*exptrm_angle_mdpt[ibot,:]-exptrm_minus_mdpt[ibot,:])+
                                       alpha1[ibot,:]*(1.-exptrm_angle_mdpt[ibot,:])+
                                       alpha2[ibot,:]*(iubar+0.5*dtau[ibot,:]-(dtau[ibot,:]+iubar)*exptrm_angle_mdpt[ibot,:])  )

            flux_at_top[ng,nt,:] = flux_plus_mdpt[0,:] #nlevel by nwno 
            
            flux_minus_all[ng,nt,:,:]=flux_minus[:,:]
            flux_plus_all[ng,nt,:,:]=flux_plus[:,:]

            flux_minus_midpt_all[ng,nt,:,:]=flux_minus_mdpt[:,:]
            flux_plus_midpt_all[ng,nt,:,:]=flux_plus_mdpt[:,:]

            #to get the convective heat flux 
            #flux_minus_mdpt_disco[ng,nt,:,:] = flux_minus_mdpt #nlevel by nwno
            #flux_plus_mdpt_disco[ng,nt,:,:] = flux_plus_mdpt #nlevel by nwno
    if calc_type == 0:
        return flux_at_top #, flux_down# numg x numt x nwno
    elif calc_type == 1:
        return flux_minus_all, flux_plus_all, flux_minus_midpt_all, flux_plus_midpt_all


@jit(nopython=True, cache=True)
def get_reflected_1d(nlevel, wno,nwno, numg,numt, dtau, tau, w0, cosb,gcos2, ftau_cld, ftau_ray,
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
    
    xint_at_top = zeros((numg, numt, nlevel, nwno))
    
    

    flux_minus_all = zeros((numg, numt,nlevel, nwno)) ## level downwelling fluxes
    flux_plus_all = zeros((numg, numt, nlevel, nwno)) ## level upwelling fluxes
    flux_minus_midpt_all = zeros((numg, numt, nlevel, nwno)) ##  layer downwelling fluxes
    flux_plus_midpt_all = zeros((numg, numt, nlevel, nwno))  ## layer upwelling fluxes

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

    xint_at_top[ng,nt,:] = flux_plus_all[0,0,0,:] /pi      ## Upward Intensity at top       
    if calc_type == 0:        
        return xint_at_top
    elif calc_type == 1:
        return flux_minus_all, flux_plus_all, flux_minus_midpt_all, flux_plus_midpt_all 
