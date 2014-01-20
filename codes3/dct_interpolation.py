from __future__ import division
import math
import numpy as np
#import solver
#import sys
#import equation as eq
#from matplotlib.pyplot import plot, show
import scipy.fftpack 
#import fftpack


L = math.pi
N = 64
h = L/N
x = np.arange(h/2, L, h)

p = 2

dct = lambda x: scipy.fftpack.dct(x, norm='ortho')
idct = lambda x: scipy.fftpack.idct(x, norm='ortho')

ww = math.sqrt(2/float(N)) * np.ones((N,1))
ww[0,0] = math.sqrt(1/float(N))

Y = np.zeros(N)
Y = dct(np.cos(x)**p)
for n in range(N):
    if Y[n] < 1e-15:
        Y[n] = 0
        
#print Y[0:10]

xx = np.arange(0, 2*L, h)
k = range(N)
NN = 2*N

uu = np.zeros(NN)

for m in range(NN):
    uu[m] = 0
    for n in range(N):
        uu[m] = uu[m] + ww[n]*Y[n]*np.cos(xx[m]*k[n])
        
        
