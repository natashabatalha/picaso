from numpy import exp, zeros, where, sqrt, cumsum , pi, outer, sinh, cosh, min, dot, array,log, stack, ones, floor
import numpy as np
import time
import pickle as pk
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve_banded
from numpy.linalg import solve
from numpy.linalg import inv as npinv
from scipy.linalg import inv as spinv


def single_layer(w0_og, F0PI, u0, dtau_og,tau_og, cosb_og, u1, P): 

    DE = True
    nlayer = 1
    nlevel = nlayer+1
    F0PI = F0PI[0]
    w0_og = w0_og[0,0]
    cosb_og = 0#cosb_og[0,0]
    tau_og = tau_og[-1,0]
    b_top = 1
    if DE:
        f2 = cosb_og**2
        w02 = (1-f2)*w0_og/(1-w0_og*f2)
        tau2 = (1-w0_og*f2)*tau_og
        dtau2 = tau2
        f4 = cosb_og**4
        w04 = (1-f4)*w0_og/(1-w0_og*f4)
        tau4 = (1-w0_og*f4)*tau_og
        dtau4 = tau4
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
        if l < 0:
            b2.append(( F0PI * (w02 * w2[l])) * P(-u0)[l] / (4*pi))
            b4.append(( F0PI * (w04 * w4[l])) * P(-u0)[l] / (4*pi))
        else:
            b2.append((0))
            b4.append((0))

    
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
    a00 = 1;    a01 = 1;
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
    F2[0,0] = f00
    F2[0,1] = f01
    F2[1,0] = f10
    F2[1,1] = f11

    G2[0] = z0
    G2[1] = z1

    M2[0,0] = f00
    M2[0,1] = f01

    B2[0] = b_top -z0

    A2[0,0] = a00
    A2[0,1] = a01
    A2[1,0] = a10
    A2[1,1] = a11

    N2[0] = eta[0]
    N2[1] = eta[1]

    # bottom of atmosphere
    F2[2,0] = f00*e0
    F2[2,1] = f01*e1
    F2[3,0] = f10*e0
    F2[3,1] = f11*e1

    G2[2] = z0*exptau
    G2[3] = z1*exptau

    M2[1,0] = f10*e0
    M2[1,1] = f11*e1

    B2[1] = -z1*exptau

    A2[2,0] = a00*e0
    A2[2,1] = a01*e1
    A2[3,0] = a10*e0
    A2[3,1] = a11*e1

    N2[2] = eta[0]*exptau
    N2[3] = eta[1]*exptau

    X2 = spsolve(M2, B2)

    Flux2 = F2.dot(X2)+G2

    Int2 = A2.dot(X2)+N2

    lamdas = [lam, -lam]
    temp = 0
    for l in range(2):
        temp2 = 0   
        for i in range(2):
            temp2 = temp2 + X2[i] * (1-exp(-dtau*(lamdas[i]+1/u1))) / (u1*lamdas[i]+1)
        temp = temp + w02*w2[l]*P(u1)[l] * temp2

    
    # 4-stream
    a = a4; b = b4
    tau = tau4; dtau = tau4
    beta = a[0]*a[1] + a[2]*a[3]/9 + 4*a[0]*a[3]/9
    gama = a[0]*a[1]*a[2]*a[3]/9
    
    lam1 = sqrt((beta + sqrt(beta**2 - 4*gama)) / 2)
    lam2 = sqrt((beta - sqrt(beta**2 - 4*gama)) / 2)

    e0 = exp(-tau * lam1)
    e1 = exp(tau * lam1)
    e2 = exp(-tau * lam2)
    e3 = exp(tau * lam2)
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

    a00 = 1;    a01 = 1;    a02 = 1;    a03 = 1;
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
    F[0,0] = f00
    F[0,1] = f01
    F[0,2] = f02
    F[0,3] = f03

    M[0,0] = f00
    M[0,1] = f01
    M[0,2] = f02
    M[0,3] = f03

    A[0,0] = a00
    A[0,1] = a01
    A[0,2] = a02
    A[0,3] = a03

    # downward other flux
    F[1,0] = f10
    F[1,1] = f11
    F[1,2] = f12
    F[1,3] = f13

    M[1,0] = f10
    M[1,1] = f11
    M[1,2] = f12
    M[1,3] = f13

    A[1,0] = a10
    A[1,1] = a11
    A[1,2] = a12
    A[1,3] = a13

    # upward flux
    F[2,0] = f20
    F[2,1] = f21
    F[2,2] = f22
    F[2,3] = f23

    A[2,0] = a20
    A[2,1] = a21
    A[2,2] = a22
    A[2,3] = a23

    # upward other flux
    F[3,0] = f30
    F[3,1] = f31
    F[3,2] = f32
    F[3,3] = f33

    A[3,0] = a30
    A[3,1] = a31
    A[3,2] = a32
    A[3,3] = a33
    
    ## bottom of atmosphere
    # downward flux
    F[4,0] = f00*e0
    F[4,1] = f01*e1
    F[4,2] = f02*e2
    F[4,3] = f03*e3

    A[4,0] = a00*e0
    A[4,1] = a01*e1
    A[4,2] = a02*e2
    A[4,3] = a03*e3

    # downward other flux
    F[5,0] = f10*e0
    F[5,1] = f11*e1
    F[5,2] = f12*e2
    F[5,3] = f13*e3

    A[5,0] = a10*e0
    A[5,1] = a11*e1
    A[5,2] = a12*e2
    A[5,3] = a13*e3

    # upward flux
    F[6,0] = f20*e0
    F[6,1] = f21*e1
    F[6,2] = f22*e2
    F[6,3] = f23*e3

    A[6,0] = a20*e0
    A[6,1] = a21*e1
    A[6,2] = a22*e2
    A[6,3] = a23*e3

    M[2,0] = f20*e0
    M[2,1] = f21*e1
    M[2,2] = f22*e2
    M[2,3] = f23*e3

    # upward other flux
    F[7,0] = f30*e0
    F[7,1] = f31*e1
    F[7,2] = f32*e2
    F[7,3] = f33*e3

    A[7,0] = a30*e0
    A[7,1] = a31*e1
    A[7,2] = a32*e2
    A[7,3] = a33*e3

    M[3,0] = f30*e0
    M[3,1] = f31*e1
    M[3,2] = f32*e2
    M[3,3] = f33*e3

    G = np.zeros(4*nlevel)
    B = np.zeros(4*nlayer)
    N = np.zeros(4*nlevel)
    ## top of atmosphere
    # downward flux
    G[0] = z0
    B[0] = b_top - z0
    N[0] = eta[0]
    # downward other flux
    G[1] = z1
    B[1] = - z1
    N[1] = eta[1]
    # upward flux
    G[2] = z2
    N[2] = eta[2]
    # upward other flux
    G[3] = z3
    N[3] = eta[3]

    ## bottom of atmosphere
    # downward flux
    G[4] = z0*exptau
    N[4] = eta[0]*exptau
    # downward other flux
    G[5] = z1*exptau
    N[5] = eta[1]*exptau
    # upward flux
    G[6] = z2*exptau
    B[2] = - z2*exptau
    N[6] = eta[2]*exptau
    # upward other flux
    G[7] = z3*exptau
    B[3] = - z3*exptau
    N[7] = eta[3]*exptau

    X4 = spsolve(M, B)

    Flux4 = F.dot(X4)+G

    Int4 = A.dot(X4)+N


    Intensity2 = zeros(nlevel)
    Flux2_up = zeros(nlevel)
    Flux2_dwn = zeros(nlevel)
    for i in range(nlevel):
        Flux2_up[i] = 2*pi*(Int2[2*i]/2 + Int2[2*i+1] )
        Flux2_dwn[i] = 2*pi*(Int2[2*i]/2 - Int2[2*i+1])
        for l in range(2):
            Intensity2[i] = Intensity2[i] + (2*l+1)*Int2[2*i+l]*P(u1)[l]
    Intensity4 = zeros(nlevel)
    Flux4_up = zeros(nlevel)
    Flux4_dwn = zeros(nlevel)
    HFlux4_up = zeros(nlevel)
    HFlux4_dwn = zeros(nlevel)
    for i in range(nlevel):
        Flux4_up[i] = 2*pi*(Int4[4*i]/2 + Int4[4*i+1]  + 5*Int4[4*i+2]/8)
        Flux4_dwn[i] = 2*pi*(Int4[4*i]/2 - Int4[4*i+1] + 5*Int4[4*i+2]/8)
        HFlux4_up[i] = 2*pi*(-Int4[4*i]/8 + 5*Int4[4*i+2]/8 + Int4[4*i+3])
        HFlux4_dwn[i] = 2*pi*(-Int4[4*i]/8 + 5*Int4[4*i+2]/8 - Int4[4*i+3])
        for l in range(4):
            Intensity4[i] = Intensity4[i] + (2*l+1)*Int4[4*i+l]*P(u1)[l]
        #Intensity4[i] = (Intensity4[i] + 7*Int4[4*i+l]*(P(u1)[3] + 1.6*sqrt(7)*P(u1)[4] 
        #                    + 3*sqrt(7)*(1-u1)**4))

    lamdas = [lam1, -lam1, lam2, -lam2]
    temp_ = 0
    for l in range(4):
        temp2 = 0   
        for i in range(4):
            temp2 = temp2 + X4[i] * (1-exp(-dtau*(lamdas[i]+1/u1))) / (u1*lamdas[i]+1)
        temp_ = temp_ + w04*w4[l]*P(u1)[l] * temp2

    import IPython; IPython.embed()
    import sys; sys.exit()

    def w(g,l):
        return (2*l+1) * g**l

    def w_(g,l,st):
        return (2*l+1) * (g**l - g**st) / (1 - g**st)

    def P2(g):
        temp = 0
        for l in range(2):
            temp = temp + w(g,l) * P(u1)[l]
        return temp

    def P2_(g):
        temp = 0
        for l in range(2):
            temp = temp + w_(g,l,2) * P(u1)[l]
        return temp

    def P4(g):
        temp = 0
        for l in range(4):
            temp = temp + w(g,l) * P(u1)[l]
        return temp

    def P4_(g):
        temp = 0
        for l in range(4):
            temp = temp + w_(g,l,4) * P(u1)[l]
        return temp

    def P6(g):
        temp = 0
        for l in range(6):
            temp = temp + w(g,l) * P(u1)[l]
        return temp

    def P6_(g):
        temp = 0
        for l in range(6):
            temp = temp + w_(g,l,6) * P(u1)[l]
        return temp

    def phase(g):
        return (1-g**2)/(sqrt((1+g**2-2*g*u1)**3))

    g = np.linspace(0,.999,100)


    import matplotlib.pyplot as plt
    plt.plot(g, phase(g), ':', color='k')
    plt.plot(g, P2(g), '-', color='r')
    plt.plot(g, P4(g), '-', color='b')
    #plt.loglog(g, P6(g), '-', color='g')
    plt.plot(g, P2_(g), '--', color='r')
    plt.plot(g, P4_(g), '--', color='b')
    #plt.loglog(g, P6_(g), '--', color='g')
    plt.show()

    import IPython; IPython.embed()
    import sys; sys.exit()


