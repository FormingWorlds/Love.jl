import numpy as np 
from numpy import pi
import matplotlib.pyplot as plt
import cmocean as cm

G = 6.67408e-11

WATT_TO_MILLIWATT = 1e3
KM_TO_METRE = 1e3


n = 2.0 

Ic = np.zeros((6,3), dtype=np.complex64)
D = np.diag( np.ones(6, dtype=np.float64) )
Ybar = np.zeros((6,6), dtype=np.complex64)
Y = np.zeros((6,6), dtype=np.complex64)

P = np.zeros((3, 6), dtype=np.complex64)
P[0, 2] = 1.0
P[1, 3] = 1.0
P[2, 5] = 1.0

def get_D(r):
    global D

    D[0, 0] = ( n+1 ) / r**( n+1 )
    D[1, 1] = n * ( n+1 ) * r**(-n + 1) / (2 * (2*n - 1) )
    D[2, 2] = r**( -(n-1) )
    D[3, 3] = n * r**n
    D[4, 4] = n * ( n+1 ) * r**(n+2) / ( 2 * (2*n + 3) )
    D[5, 5] = -r**(n+1)

    D /= 2*n + 1

    return D 

def get_Ybar(r, den, rigid, g):
    global Ybar

    # g = 4*pi*G*den*r/3

    mu = rigid
    ef_mu = den * g * r / rigid

    # First column:

    Ybar[0, 0] = ef_mu - 2*( n + 2 )
    Ybar[1, 0] = -ef_mu + 2*( n**2 + 3*n - 1) / ( n+1 )
    Ybar[2, 0] = 4*pi*G*den
    Ybar[3, 0] = ef_mu + 2*( n - 1 )
    Ybar[4, 0] = -ef_mu - 2*(n**2 - n - 3) / n
    Ybar[5, 0] = 4*pi*G*den*r

    # Second column

    Ybar[0, 1] = 2*n*( n+2 )
    Ybar[1, 1] = -2*( n**2 - 1 )
    Ybar[3, 1] =  2*( n**2 - 1 )
    Ybar[4, 1] = -2*n*( n + 2 )

    # Third column

    Ybar[0, 2] = -r/mu
    Ybar[1, 2] =  r/mu
    Ybar[3, 2] = -r/mu
    Ybar[4, 2] =  r/mu

    # Fourth column

    Ybar[0, 3] = n*r / mu
    Ybar[1, 3] = ( 2-n ) * r / mu
    Ybar[3, 3] =  -( n+1 ) * r / mu
    Ybar[4, 3] = ( n+3 ) * r /mu

    # Fifth column

    Ybar[0, 4] =  den * r / mu
    Ybar[1, 4] = -den * r / mu
    Ybar[3, 4] =  den * r / mu
    Ybar[4, 4] = -den * r / mu
    Ybar[5, 4] = 2*n + 1 

    # Sixth column 

    Ybar[2, 5] = -1.0
    Ybar[5, 5] = -r

    return Ybar

def get_Y(r, den, rigid, g):
    # Y = np.zeros((6,6), dtype=np.complex64)
    global Y
    # g = 4*pi*G*den*r/3

    mu = rigid
    pgr = den * g * r 

    # First column:

    Y[0, 0] = n*r**( n+1 ) / ( 2*( 2*n + 3) )
    Y[1, 0] = ( n+3 )*r**( n+1 ) / ( 2*( 2*n+3 ) * ( n+1 ) )
    Y[2, 0] = ( n*pgr + 2*( n**2 - n - 3)*mu ) * r**n / ( 2*( 2*n + 3) )
    Y[3, 0] = n *( n+2 ) * mu * r**n / ( ( 2*n + 3 )*( n+1 ) )
    Y[5, 0] = 2*pi*G*den*n*r**( n+1 ) / ( 2*n + 3 )

    # Second column

    Y[0, 1] = r**( n-1 )
    Y[1, 1] = r**( n-1 ) / n
    Y[2, 1] = ( pgr + 2*( n-1 )*mu ) * r**( n-2 )
    Y[3, 1] = 2*( n-1 ) * mu * r**( n-2 ) / n
    Y[5, 1] = 4*pi*G*den*r**( n-1 )

    # Third column

    Y[2, 2] = -den * r**n
    Y[4, 2] = -r**n
    Y[5, 2] = -( 2*n + 1) * r**( n-1 )

    # Fourth column

    Y[0, 3] = ( n+1 ) * r**( -n ) / ( 2*( 2*n - 1 ) )
    Y[1, 3] = ( 2 - n ) * r**( -n ) / ( 2*n * ( 2*n - 1 ) )
    Y[2, 3] = ( ( n+1 ) * pgr - 2 * ( n**2 + 3*n - 1 ) * mu ) / ( 2*( 2*n -1 ) * r**(n+1) )
    Y[3, 3] = ( n**2 - 1 ) * mu / ( n*( 2*n - 1 ) * r**( n+1) )
    Y[5, 3] = 2*pi*G*den*( n+1 ) / ( ( 2*n - 1 ) * r**n )

    # Fifth column

    Y[0, 4] =  r**( -n - 2 )
    Y[1, 4] = -r**( -n - 2 ) / ( n+1 )
    Y[2, 4] = (pgr - 2*( n+2 )*mu ) / r**(n+3)
    Y[3, 4] = 2 * ( n+2 ) * mu / ( (n+1) * r**(n+3) )
    Y[5, 4] = 4*pi*G*den / r**( n+2 )

    # Sixth column 

    Y[2, 5] = - den / r**( n+1 )
    Y[4, 5] = - 1.0 / r**( n+1 )

    return Y


def get_Ic(r, den, rigid, g):
    global Ic

    # g = 4*pi*G*den*r/3

    mu = rigid
    pgr = den * g * r 

    # First column:

    Ic[0, 0] = n*r**( n+1 ) / ( 2*( 2*n + 3) )
    Ic[1, 0] = ( n+3 )*r**( n+1 ) / ( 2*( 2*n+3 ) * ( n+1 ) )
    Ic[2, 0] = ( n*pgr + 2*( n**2 - n - 3)*mu ) * r**n / ( 2*( 2*n + 3) )
    Ic[3, 0] = n *( n+2 ) * mu * r**n / ( ( 2*n + 3 )*( n+1 ) )
    Ic[5, 0] = 2*pi*G*den*n*r**( n+1 ) / ( 2*n + 3 )

    # Second column

    Ic[0, 1] = r**( n-1 )
    Ic[1, 1] = r**( n-1 ) / n
    Ic[2, 1] = ( pgr + 2*( n-1 )*mu ) * r**( n-2 )
    Ic[3, 1] = 2*( n-1 ) * mu * r**( n-2 ) / n
    Ic[5, 1] = 4*pi*G*den*r**( n-1 )

    # Third column

    Ic[2, 2] = -den * r**n
    Ic[4, 2] = -r**n
    Ic[5, 2] = -( 2*n + 1) * r**( n-1 )

    return Ic


def get_Yinv(r, den, rigid, g):

    D = get_D(r)
    Ybar = get_Ybar(r, den, rigid, g) 

    return D.dot(Ybar)


def CalcLoveNumbers(radii, density, mu, visc, rot):
    # Currently only works for an incompressible solid core.

    rs = radii.copy()
    
    rx = radii

    mu = 1j*rot*mu / (1j*rot + mu/visc)

    # Give the fluid layer a small, but non-zero, complex shear modulus.
    mu[np.isnan(mu)] = 100

    muc = mu[-1]
    rc = radii[-1]
    denc = density[-1]
    R = radii[0]

    denx = np.zeros_like(rx)
    mux = np.zeros(len(rx), dtype=np.complex64)
    for i in range(len(rx)):
        for j in range(len(rs)-1):
            if rx[i] > radii[j+1]:
                denx[i] = density[j]
                mux[i] = mu[j]
                break

    
    gc = 4*pi*G*denc*rc/3

    N = len(rs)
    M = np.zeros(N)

    # Get mass of each layer
    for i in range(0, N-1):
        M[i] = 4./3. * pi * (rs[i]**3.0 - rs[i+1]**3.0) * density[i]

    M[-1] = 4./3. * pi * (rs[-1]**3.0) * density[-1]

    gR = G*np.sum(M)/R**2.0
    A = np.eye(6)
    for i in range(0, len(rx)-1):
        r_t = rx[i]
        r_b = rx[i+1]

        gt = G*np.sum(M[i:])/r_t**2.0
        gb = G*np.sum(M[i+1:])/r_b**2.0

        Y = get_Y(r_t, denx[i], mux[i], gt)
        Yinv = get_Yinv(r_b, denx[i], mux[i], gb)

        A = np.matmul(A, np.matmul(Y, Yinv) )


    # Set boundary condition vector!
    b = np.zeros(3)
    b[-1] = -(2*n+1) / R


    Ic = get_Ic(rc, denc, muc, gc)
    PAIinv = np.linalg.inv( np.matmul( np.matmul(P, A), Ic) )
    
    y = np.matmul( np.matmul(A, Ic), PAIinv.dot(b) )


    k = -y[4] - 1
    h = gR*y[0]
    l = gR*y[1]
    
    Q = k.real / (-k.imag)

    return k, h, l, Q



if __name__=='__main__':

    den = np.array([3e3, 3e3,3e3])
    r = np.array([2501e3, 2500e3, 1e3])
    mu = np.array([1e30, 3e9,3e9]) * 1e-9
    visc = np.array([1e30, 1e30,1e30])
    rot = 2e-7
    print(CalcLoveNumbers(r, den, mu, visc, rot))
    print("hi")