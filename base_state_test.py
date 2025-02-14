import numpy as np 

mu = 1e9 
kappa = 100e9
R = 1800e3
r = np.linspace(1, R, 101)
den = 1000.0
lam = kappa - 2/3 * mu
beta = 2*mu + lam

def get_A(r):
    A = np.zeros((2,2))
    A[0,0] = -2*lam / (r*beta)
    A[0,1] = 1/beta 
    A[1,0] = 12*kappa*mu / (beta * r**2)
    A[1,1] = -4*mu / (beta * r)

    return A 


I2 = np.identity(2)
ystart = np.zeros((2))
ystart[1] = 1.0

Bprod = np.zeros((2,2))
Bprod[0,0] = 1.0
Bprod[1,1] = 1.0

y = np.zeros((len(r), 2))
for i in range(len(r)-1):
    dr = r[i+1]-r[i]
    A1 = get_A(r[i])
    A2 = get_A(r[i+1])

    k1 = dr*A1 
    k2 = np.matmul(A2, dr*(I2 + k1) )

    B = I2 + 0.5* (k1+k2)

    # c1 = 1/6. * dr
    # c2 = 2/6. * dr

    # A1 = get_A(r[i])
    # A2 = get_A(r[i] + 0.5*dr)
    # A3 = get_A(r[i+1])
        
        
    # k1 = np.copy(A1) 
    # k2 = np.matmulA2 * (I2 + 0.5*dr * k1)
    # k3 =  A2 * (I2 + 0.5*dr * k2)
    # k4 =  A3 * (I2 + dr * k3) 

    Bprod = np.matmul(Bprod, B)
    
    # y[i,:] = np.matmul(B, ystart)
    # ystart[:] = y[i,:]



print(np.matmul(Bprod, ystart))
# print(y)


    