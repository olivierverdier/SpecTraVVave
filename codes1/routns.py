#This file contains functions which compute the
#flux, it's derivative; the linear operator (dispersion) and
#the initial guess (value) which will be employed in the solver.

import math
import numpy as np
import scipy.fftpack 
def initval(x):
    #The function computes the initial value (u(x,t=0))
    guess = -0.015-0.1*np.cos(x)
    return guess

def D(u):
    # This function returnes the df(u)/du.
    return 3*(u+1)**(0.5)-3

def flux(u):
    # This function returnes the f(u).
    return 2*(u+1)**(1.5)-3*u-2

def lin_op(lngth, gridnum):
    # The function takes the half of the period of wave and the grid number. 
    # Then it returns the matrix Tau which is used in the dispersion term.
    L = lngth
    N = gridnum
    h = float(L)/N
    x = np.arange(h/2.,L,h)
    xi = range(0,N,1)
    
    ww = math.sqrt(2/float(N)) * np.ones((N,1))
    ww[0,0] = math.sqrt(1/float(N))
    
    Tau = np.zeros((N,N))
    for m in range(N):
        for n in range(N):
            Tau[m,n] = ww[0]*ww[0]*math.cos(x[n]*xi[0])*math.cos(x[m]*xi[0])
            for k in range(1,N):
                Tau[m,n] = Tau[m,n] + math.sqrt(1./xi[k]*math.tanh(xi[k]))*ww[k]*ww[k]*math.cos(x[n]*xi[k])*math.cos(x[m]*xi[k])
    return Tau

def interp(u, L, N):
    # The code interpolates the values of the wave
    # branch at discerete nodes
    h = L/float(N)
    xx = np.arange(0, 2*L, h)
    xi = range(0,N,1)

    ww = math.sqrt(2/float(N)) * np.ones((N,1))
    ww[0,0] = math.sqrt(1/float(N))

    Y = np.zeros(N)
    Y = 0.5*np.array([ww[i]*scipy.fftpack.dct(u, type = 2)[i] for i in range(N)])
     
    uuu = np.zeros(2*N)
    for m in range(2*N):
        for n in range(N):
            uuu[m] += ww[n]*Y[n]*math.cos(xx[m]*xi[n])
            
    return uuu 