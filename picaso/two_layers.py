from numpy import exp, zeros, where, sqrt, cumsum , pi, outer, sinh, cosh, min, dot, array,log, stack, ones, floor
import numpy as np
import time
import pickle as pk
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve_banded
from numpy.linalg import solve
from numpy.linalg import inv as npinv
from scipy.linalg import inv as spinv


def two_layers(w0_og, F0PI, u0, dtau_og,tau_og, cosb_og, u1, P): 

    DE = True
    nlayer = 2
    nlevel = nlayer+1
    F0PI = F0PI[-1]
    w0_og = w0_og[:,-1]
    cosb_og = cosb_og[:,-1]
    tau_og = tau_og[:,-1]
    dtau_og = dtau_og[:,-1]
    b_top = 0
    if DE:
        f2 = cosb_og**2
        w02 = (1-f2)*w0_og/(1-w0_og*f2)
        dtau2 = (1-w0_og*f2)*dtau_og
        tau2 = np.zeros(tau_og.shape)
        tau2[1:] = np.cumsum(dtau2)
        f4 = cosb_og**4
        w04 = (1-f4)*w0_og/(1-w0_og*f4)
        dtau4 = (1-w0_og*f4)*dtau_og
        tau4 = np.zeros(tau_og.shape)
        tau4[1:] = np.cumsum(dtau4)
    else:
        f2 = 0.
        w02 = w0_og
        tau2 = tau_og
        dtau2 = tau2
        f4 = 0.
        w04 = w0_og
        tau4 = tau_og
        dtau4 = tau4
    
    a2 = []; b2 = []
    a4 = []; b4 = []
    w2 = []; w4 = []
    for l in range(4):
        w2.append((2*l+1) * (cosb_og**l - f2) / (1 - f2))
        w4.append((2*l+1) * (cosb_og**l - f4) / (1 - f4))
        a2.append((2*l + 1) -  w02 * w2[l])
        a4.append((2*l + 1) -  w04 * w4[l])
        if l < 3:
            b2.append(( F0PI * (w02 * w2[l])) * P(-u0)[l] / (4*pi))
            b4.append(( F0PI * (w04 * w4[l])) * P(-u0)[l] / (4*pi))
        else:
            b2.append((0*w02))
            b4.append((0*w04))

    
    # 2-stream
    a = a2; b = b2
    tau = tau2; dtau = dtau2
    Del = ((1 / u0)**2 - a[0]*a[1])
    eta = [(b[1] /u0 - a[1]*b[0]) / Del,
            (b[0] /u0 - a[0]*b[1]) / Del]
    
    lam = sqrt(a[0]*a[1])
    expo = lam*dtau
    e0 = exp(-expo)
    e1 = exp(expo)
    exptau = exp(-tau/u0)
    if np.any(dtau*lam > 35.0) or np.any(tau/u0 > 35.0):
        print('check exponentials')
        import sys; sys.exit()
    
    q = lam/a[1]
    # intensity matrix
    a00 = ones(2);    a01 = ones(2)
    a10 = -q;   a11 = q;

    Q1 = 2*pi*(0.5 - q)
    Q2 = 2*pi*(0.5 + q)
    # flux matrix
    f00 = Q2;   f01 = Q1
    f10 = Q1;   f11 = Q2

    # flux RHS entries
    z0 = 2*pi*(0.5*eta[0] - eta[1]) 
    z1 = 2*pi*(0.5*eta[0] + eta[1])

    F2 = np.zeros((2*nlevel, 2*nlayer))
    M2 = np.zeros((2*nlayer, 2*nlayer))
    A2 = np.zeros((2*nlevel, 2*nlayer))
    G2 = np.zeros(2*nlevel)
    B2 = np.zeros(2*nlayer)
    N2 = np.zeros(2*nlevel)

    # top of atmosphere
    F2[0,0] = f00[0]
    F2[0,1] = f01[0]
    F2[1,0] = f10[0]
    F2[1,1] = f11[0]

    G2[0] = z0[0]
    G2[1] = z1[0]

    M2[0,0] = f00[0]
    M2[0,1] = f01[0]

    B2[0] = b_top -z0[0]

    A2[0,0] = a00[0]
    A2[0,1] = a01[0]
    A2[1,0] = a10[0]
    A2[1,1] = a11[0]

    N2[0] = eta[0][0]
    N2[1] = eta[1][0]

    # bottom of layer 1
    F2[2,0] = f00[0]*e0[0]
    F2[2,1] = f01[0]*e1[0]
    F2[3,0] = f10[0]*e0[0]
    F2[3,1] = f11[0]*e1[0]

    G2[2] = z0[0]*exptau[1]
    G2[3] = z1[0]*exptau[1]

    M2[1,0] = f00[0]*e0[0]
    M2[1,1] = f01[0]*e1[0]
    M2[2,0] = f10[0]*e0[0]
    M2[2,1] = f11[0]*e1[0]

    M2[1,2] = -f00[1]
    M2[1,3] = -f01[1]
    M2[2,2] = -f10[1]
    M2[2,3] = -f11[1]

    B2[1] = (z0[1]-z0[0])*exptau[1]
    B2[2] = (z1[1]-z1[0])*exptau[1]

    A2[2,0] = a00[0]*e0[0]
    A2[2,1] = a01[0]*e1[0]
    A2[3,0] = a10[0]*e0[0]
    A2[3,1] = a11[0]*e1[0]

    N2[2] = eta[0][0]*exptau[1]
    N2[3] = eta[1][0]*exptau[0]

    # bottom of atmosphere
    F2[4,2] = f00[1]*e0[1]
    F2[4,3] = f01[1]*e1[1]
    F2[5,2] = f10[1]*e0[1]
    F2[5,3] = f11[1]*e1[1]

    G2[4] = z0[1]*exptau[2]
    G2[5] = z1[1]*exptau[2]

    M2[3,2] = f10[1]*e0[1]
    M2[3,3] = f11[1]*e1[1]

    B2[3] = - z1[1]*exptau[2]

    A2[4,2] = a00[1]*e0[1]
    A2[4,3] = a01[1]*e1[1]
    A2[5,2] = a10[1]*e0[1]
    A2[5,3] = a11[1]*e1[1]

    N2[4] = eta[0][1]*exptau[2]
    N2[5] = eta[1][1]*exptau[2]

    X2 = spsolve(M2, B2)

    Flux2 = F2.dot(X2)+G2

    Int2 = A2.dot(X2)+N2

    
    # 4-stream
    a = a4; b = b4
    tau = tau4; dtau = dtau4
    beta = a[0]*a[1] + a[2]*a[3]/9 + 4*a[0]*a[3]/9
    gama = a[0]*a[1]*a[2]*a[3]/9
    
    lam1 = sqrt((beta + sqrt(beta**2 - 4*gama)) / 2)
    lam2 = sqrt((beta - sqrt(beta**2 - 4*gama)) / 2)

    e0 = exp(-dtau * lam1)
    e1 = exp(dtau * lam1)
    e2 = exp(-dtau * lam2)
    e3 = exp(dtau * lam2)
    exptau = exp(-tau/u0)
    if np.any(dtau*lam1 > 35.0) or np.any(dtau*lam2 > 35.0) or np.any(tau/u0 > 35.0):
        print('check exponentials')
        import sys; sys.exit()
    
    R1 = -a[0]/lam1
    R2 = -a[0]/lam2
    Q1 = (a[0]*a[1] / (lam1**2) - 1) / 2
    Q2 = (a[0]*a[1] / (lam2**2) - 1) / 2
    S1 = -3*(a[0]*a[1] / lam1 - lam1) / (2*a[3])
    S2 = -3*(a[0]*a[1] / lam2 - lam2) / (2*a[3])

    a00 = ones(2);    a01 = ones(2);    a02 = ones(2);    a03 = ones(2);
    a10 = R1;   a11 = -R1;  a12 = R2;   a13 = -R2;
    a20 = Q1;   a21 = Q1;   a22 = Q2;   a23 = Q2;
    a30 = S1;   a31 = -S1;  a32 = S2;   a33 = -S2;

    p1mn = 2*pi*(1/2 - R1 + 5*Q1/8);    p1pl = 2*pi*(1/2 + R1 + 5*Q1/8);
    p2mn = 2*pi*(1/2 - R2 + 5*Q2/8);    p2pl = 2*pi*(1/2 + R2 + 5*Q2/8);
    q1mn = 2*pi*(-1/8 + 5*Q1/8 - S1);   q1pl = 2*pi*(-1/8 + 5*Q1/8 + S1);
    q2mn = 2*pi*(-1/8 + 5*Q2/8 - S2);   q2pl = 2*pi*(-1/8 + 5*Q2/8 + S2);

    # flux matrix entries
    f00 = p1mn;    f01 = p1pl;    f02 = p2mn;    f03 = p2pl
    f10 = q1mn;    f11 = q1pl;    f12 = q2mn;    f13 = q2pl
    f20 = p1pl;    f21 = p1mn;    f22 = p2pl;    f23 = p2mn
    f30 = q1pl;    f31 = q1mn;    f32 = q2pl;    f33 = q2mn

    Del = 9 * (1/(u0**4) - beta / (u0**2) + gama)
    eta = []
    eta.append(((a[1]*b[0] - b[1]/u0)*(a[2]*a[3] - 9/(u0**2))
                    + 2*(a[3]*b[2] - 2*a[3]*b[0] - 3*b[3]/u0) / (u0**2)) / Del)
    eta.append(((a[0]*b[1] - b[0]/u0)*(a[2]*a[3] - 9/(u0**2))
                    - 2*a[0]*(a[3]*b[2] - 3*b[3]/u0) / u0) / Del)
    eta.append(((a[3]*b[2] - 3*b[3]/u0)*(a[0]*a[1] - 1/(u0**2))
                    - 2*a[3]*(a[0]*b[1] - b[0]/u0) / u0) / Del )
    eta.append(((a[2]*b[3] - 3*b[2]/u0)*(a[0]*a[1] - 1/(u0**2))
                    + 2*(3*a[0]*b[1] -2*a[0]*b[3] - 3*b[0]/u0) / (u0**2)) / Del )

    # flux RHS entries
    z0 = 2*pi*(eta[0]/2 - eta[1] + 5*eta[2]/8)
    z1 = 2*pi*(-eta[0]/8 + 5*eta[2]/8 - eta[3])
    z2 = 2*pi*(eta[0]/2 + eta[1] + 5*eta[2]/8)
    z3 = 2*pi*(-eta[0]/8 + 5*eta[2]/8 + eta[3])

    F = np.zeros((4*nlevel, 4*nlayer))
    M = np.zeros((4*nlayer, 4*nlayer))
    A = np.zeros((4*nlevel, 4*nlayer))

    ## top of atmosphere
    # downward flux
    F[0,0] = f00[0]
    F[0,1] = f01[0]
    F[0,2] = f02[0]
    F[0,3] = f03[0]

    M[0,0] = f00[0]
    M[0,1] = f01[0]
    M[0,2] = f02[0]
    M[0,3] = f03[0]

    A[0,0] = a00[0]
    A[0,1] = a01[0]
    A[0,2] = a02[0]
    A[0,3] = a03[0]

    # downward other flux
    F[1,0] = f10[0]
    F[1,1] = f11[0]
    F[1,2] = f12[0]
    F[1,3] = f13[0]

    M[1,0] = f10[0]
    M[1,1] = f11[0]
    M[1,2] = f12[0]
    M[1,3] = f13[0]

    A[1,0] = a10[0]
    A[1,1] = a11[0]
    A[1,2] = a12[0]
    A[1,3] = a13[0]

    # upward flux
    F[2,0] = f20[0]
    F[2,1] = f21[0]
    F[2,2] = f22[0]
    F[2,3] = f23[0]

    A[2,0] = a20[0]
    A[2,1] = a21[0]
    A[2,2] = a22[0]
    A[2,3] = a23[0]

    # upward other flux
    F[3,0] = f30[0]
    F[3,1] = f31[0]
    F[3,2] = f32[0]
    F[3,3] = f33[0]

    A[3,0] = a30[0]
    A[3,1] = a31[0]
    A[3,2] = a32[0]
    A[3,3] = a33[0]

    ## bottom of layer 1
    # downward flux
    F[4,0] = f00[0]*e0[0]
    F[4,1] = f01[0]*e1[0]
    F[4,2] = f02[0]*e2[0]
    F[4,3] = f03[0]*e3[0]

    A[4,0] = a00[0]*e0[0]
    A[4,1] = a01[0]*e1[0]
    A[4,2] = a02[0]*e2[0]
    A[4,3] = a03[0]*e3[0]

    M[2,0] = f00[0]*e0[0]
    M[2,1] = f01[0]*e1[0]
    M[2,2] = f02[0]*e2[0]
    M[2,3] = f03[0]*e3[0]

    M[2,4] = -f00[1]
    M[2,5] = -f01[1]
    M[2,6] = -f02[1]
    M[2,7] = -f03[1]

    # downward other flux
    F[5,0] = f10[0]*e0[0]
    F[5,1] = f11[0]*e1[0]
    F[5,2] = f12[0]*e2[0]
    F[5,3] = f13[0]*e3[0]

    A[5,0] = a10[0]*e0[0]
    A[5,1] = a11[0]*e1[0]
    A[5,2] = a12[0]*e2[0]
    A[5,3] = a13[0]*e3[0]

    M[3,0] = f10[0]*e0[0]
    M[3,1] = f11[0]*e1[0]
    M[3,2] = f12[0]*e2[0]
    M[3,3] = f13[0]*e3[0]

    M[3,4] = -f10[1]
    M[3,5] = -f11[1]
    M[3,6] = -f12[1]
    M[3,7] = -f13[1]

    # upward flux
    F[6,0] = f20[0]*e0[0]
    F[6,1] = f21[0]*e1[0]
    F[6,2] = f22[0]*e2[0]
    F[6,3] = f23[0]*e3[0]

    A[6,0] = a20[0]*e0[0]
    A[6,1] = a21[0]*e1[0]
    A[6,2] = a22[0]*e2[0]
    A[6,3] = a23[0]*e3[0]

    M[4,0] = f20[0]*e0[0]
    M[4,1] = f21[0]*e1[0]
    M[4,2] = f22[0]*e2[0]
    M[4,3] = f23[0]*e3[0]

    M[4,4] = -f20[1]
    M[4,5] = -f21[1]
    M[4,6] = -f22[1]
    M[4,7] = -f23[1]

    # upward other flux
    F[7,0] = f30[0]*e0[0]
    F[7,1] = f31[0]*e1[0]
    F[7,2] = f32[0]*e2[0]
    F[7,3] = f33[0]*e3[0]

    A[7,0] = a30[0]*e0[0]
    A[7,1] = a31[0]*e1[0]
    A[7,2] = a32[0]*e2[0]
    A[7,3] = a33[0]*e3[0]

    M[5,0] = f30[0]*e0[0]
    M[5,1] = f31[0]*e1[0]
    M[5,2] = f32[0]*e2[0]
    M[5,3] = f33[0]*e3[0]

    M[5,4] = -f30[1]
    M[5,5] = -f31[1]
    M[5,6] = -f32[1]
    M[5,7] = -f33[1]

    ## bottom of atmosphere
    # downward flux
    F[8,4] = f00[1]*e0[1]
    F[8,5] = f01[1]*e1[1]
    F[8,6] = f02[1]*e2[1]
    F[8,7] = f03[1]*e3[1]

    A[8,4] = a00[1]*e0[1]
    A[8,5] = a01[1]*e1[1]
    A[8,6] = a02[1]*e2[1]
    A[8,7] = a03[1]*e3[1]

    # downward other flux
    F[9,4] = f10[1]*e0[1]
    F[9,5] = f11[1]*e1[1]
    F[9,6] = f12[1]*e2[1]
    F[9,7] = f13[1]*e3[1]
         
    A[9,4] = a10[1]*e0[1]
    A[9,5] = a11[1]*e1[1]
    A[9,6] = a12[1]*e2[1]
    A[9,7] = a13[1]*e3[1]

    # upward flux
    F[10,4] = f20[1]*e0[1]
    F[10,5] = f21[1]*e1[1]
    F[10,6] = f22[1]*e2[1]
    F[10,7] = f23[1]*e3[1]

    A[10,4] = a20[1]*e0[1]
    A[10,5] = a21[1]*e1[1]
    A[10,6] = a22[1]*e2[1]
    A[10,7] = a23[1]*e3[1]

    M[6,4] = f20[1]*e0[1]
    M[6,5] = f21[1]*e1[1]
    M[6,6] = f22[1]*e2[1]
    M[6,7] = f23[1]*e3[1]

    # upward other flux
    F[11,4] = f30[1]*e0[1]
    F[11,5] = f31[1]*e1[1]
    F[11,6] = f32[1]*e2[1]
    F[11,7] = f33[1]*e3[1]

    A[11,4] = a30[1]*e0[1]
    A[11,5] = a31[1]*e1[1]
    A[11,6] = a32[1]*e2[1]
    A[11,7] = a33[1]*e3[1]

    M[7,4] = f30[1]*e0[1]
    M[7,5] = f31[1]*e1[1]
    M[7,6] = f32[1]*e2[1]
    M[7,7] = f33[1]*e3[1]

    G = np.zeros(4*nlevel)
    B = np.zeros(4*nlayer)
    N = np.zeros(4*nlevel)

    ## top of cwatmosphere
    # downward flux
    G[0] = z0[0]
    B[0] = b_top - z0[0]
    N[0] = eta[0][0]
    # downward other flux
    G[1] = z1[0]
    B[1] = - z1[0]
    N[1] = eta[1][0]
    # upward flux
    G[2] = z2[0]
    N[2] = eta[2][0]
    # upward other flux
    G[3] = z3[0]
    N[3] = eta[3][0]

    ## bottom of layer 1
    # downward flux
    G[4] = z0[0]*exptau[1]
    N[4] = eta[0][0]*exptau[1]
    B[2] = (z0[1]- z0[0])*exptau[1]
    # downward other flux
    G[5] = z1[0]*exptau[1]
    N[5] = eta[1][0]*exptau[1]
    B[3] = (z1[1]- z1[0])*exptau[1]
    # upward flux
    G[6] = z2[0]*exptau[1]
    B[4] = (z2[1]- z2[0])*exptau[1]
    N[6] = eta[2][0]*exptau[1]
    # upward other flux
    G[7] = z3[0]*exptau[1]
    B[5] = (z3[1]- z3[0])*exptau[1]
    N[7] = eta[3][0]*exptau[1]

    ## bottom of atmosphere
    # downward flux
    G[8] = z0[1]*exptau[2]
    N[8] = eta[0][1]*exptau[2]
    # downward other flux
    G[9] = z1[1]*exptau[2]
    N[9] = eta[1][1]*exptau[2]
    # upward flux
    G[10] = z2[1]*exptau[2]
    B[6] = - z2[1]*exptau[2]
    N[10] = eta[2][1]*exptau[2]
    # upward other flux
    G[11] = z3[1]*exptau[2]
    B[7] = - z3[1]*exptau[2]
    N[10] = eta[3][1]*exptau[2]

    X4 = spsolve(M, B)

    Flux4 = F.dot(X4)+G

    Int4 = A.dot(X4)+N


    Intensity2 = zeros(nlevel)
    for i in range(nlevel):
        for l in range(2):
            Intensity2[i] = Intensity2[i] + (2*l+1)*Int2[2*i+l]*P(u1)[l]
    Intensity4 = zeros(nlevel)
    Flux4_up = zeros(nlevel)
    Flux4_dwn = zeros(nlevel)
    for i in range(nlevel):
        Flux4_up[i] = 2*pi*(Int4[4*i]/2 + Int4[4*i+1]  + 5*Int4[4*i+2]/8)
        Flux4_dwn[i] = 2*pi*(Int4[4*i]/2 - Int4[4*i+1] + 5*Int4[4*i+2]/8)
        for l in range(4):
            Intensity4[i] = Intensity4[i] + (2*l+1)*Int4[4*i+l]*P(u1)[l]
    import IPython; IPython.embed()
    import sys; sys.exit()


