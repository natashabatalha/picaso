import numpy as np
from numba import jit

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