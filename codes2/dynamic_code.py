from __future__ import division

import math
import numpy as np
from scipy.fftpack import fft, ifft, dct 


class dynamic_code(object):
    def __init__(self, Equation, wave):
        self.eq = Equation
        self.u = wave
        
    def evolution(self, dt = 0.002, periods = 1):
        #dct = lambda x: scipy.fftpack.dct(x, norm='ortho')                 # short declaration of the dct and idct operators
        
        #idct = lambda x: scipy.fftpack.idct(x, norm='ortho')
        
        N = self.eq.size
        L = self.eq.length
        #p = 2 #self.eq.degree
        

        ww = math.sqrt(2/float(N)) * np.ones((N,1))
        ww[0,0] = math.sqrt(1/float(N))

        Y = np.zeros(N)
        Y = dct(self.u, norm='ortho')
#        for n in range(N):
#            if np.abs(Y[n]) < 1e-15:
#                Y[n] = 0
        
        xx = np.arange(0, 2*L, L/N)
        k = range(N)
        NN = 2*N

        uu = np.zeros(NN)

        for m in range(NN):
            uu[m] = 0
            for n in range(N):
                uu[m] = uu[m] + ww[n]*Y[n]*np.cos(xx[m]*k[n])
              
        u = uu    
        k = np.concatenate((np.arange(1, NN/2+1, 1), np.arange(1-NN/2, 0, 1)))
        whitha = 1j*k*np.sqrt(np.tanh(k)/k)
        whitham = np.concatenate(([0], whitha))
        whitham[NN/2] = 0
        k = np.concatenate((np.arange(0, NN/2, 1), [0], np.arange(1-NN/2, 0, 1)))
        
        m = 0.5*dt*whitham
        d1 = (1-m)/(1+m)
        d1[NN/2] = 0
        d2 = -0.5*1j*dt*k/(1+m)
        
        T = 2*self.eq.length/self.eq.velocity
        eps = 1e-10
        t = dt
        it = 0
        
        while t < periods*T:
            fftu = fft(u)
            fftuu = fft(self.eq.flux(u))
            z = d1*fftu + d2*fftuu
            v = np.real(ifft(z))
            z = d1*fftu + 2*d2*fftuu 
            w = np.real(ifft(z))
            
            w_old = w 
            err = 1
            while err > eps:
                it += 1
                if it > 10000:
                    break
                
                z = ifft(d2*fft(self.eq.flux(w_old)))    
                w = v + z.real
                z = w - w_old
                err = np.linalg.norm(z)
                w_old = w
                
            u = w
            t = t + dt
        
        print 'The error for', periods,' periods is E = ', np.linalg.norm(u-uu)     
        return u