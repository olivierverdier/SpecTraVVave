# Evolution code which tests whether the computed
# periodic travelling-wave branch is the solution 
# for the given PDE.

import math
import numpy as np
import routns         # file contains function computing nonlinearity and disperion 
#import time
import scipy.fftpack

def evoltest(c, vini, N, dt, num):
    NN = 2*N
        
    k = np.concatenate((np.arange(1, NN/2+1, 1), np.arange(1-NN/2, 0, 1)))
    whitha = 1j*np.array([k[i]*math.sqrt(math.tanh(k[i])/float(k[i])) for i in range(len(k))])
    whitham = np.concatenate(([0], whitha))
    whitham[NN/2] = 0
    k = np.concatenate((np.arange(0, NN/2, 1), [0], np.arange(1-NN/2, 0, 1)))
    
    m = 0.5*dt*whitham
    d1 = np.array([(1-m[i])/(1+m[i]) for i in range(len(m))])
    d1[NN/2] = 0
    d2 = -0.5*1j*dt*np.array([k[i]/(1+m[i]) for i in range(len(m))])
    u = vini
    
    T = 2*math.pi/float(c)
    eps = 1e-10
    t = dt
    it = 0
    
    while t < num*T:
        fftu = scipy.fftpack.fft(u)
        fftuu = scipy.fftpack.fft(routns.flux(u))
        
        v = np.real(scipy.fftpack.ifft([d1[i]*fftu[i] + d2[i]*fftuu[i] for i in range(len(d1))]))
        w = np.real(scipy.fftpack.ifft([d1[i]*fftu[i] + 2*d2[i]*fftuu[i] for i in range(len(d1))]))
        
        w_old = w 
        err = 1
        while err > eps:
            it += 1
            if it > 10000:
                break
                
            w = v +  np.real(scipy.fftpack.ifft([d2[i]*scipy.fftpack.fft(routns.flux(w)) for i in range(len(d2))]))
            err = np.linalg.norm([w[i] - w_old[i] for i in range(len(w))])
            w_old = w
            
        u = w
        t = t + dt
        
    return u