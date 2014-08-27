from __future__ import division

import math
import numpy as np
from scipy.fftpack import fft, ifft, dct 


class Defrutos_sanzserna(object):
    def __init__(self, Equation, wave, velocity):
        self.eq = Equation
        self.u = wave
        self.velocity = velocity

    def interpolation(self, symm = 0):                  # symm = 1 means that the branch is symmetric w.r.t. L
        N = self.eq.size                                # symm = 0 means that the branch is symmetric w.r.t. 0
        L = self.eq.length
                
        ww = math.sqrt(2/float(N)) * np.ones((N,1))
        ww[0,0] = math.sqrt(1/float(N))

        Y = np.zeros(N)
        Y = dct(self.u, norm='ortho')
        for n in range(N):
            if np.abs(Y[n]) < 1e-15:
                Y[n] = 0
        if symm == 1:
            xx = np.arange(0, 2*L, L/N)
        else:
            xx = np.arange(-L, L, L/N)
        k = range(N)
        NN = 2*N

        uu = np.zeros(NN)

        for m in range(NN):
            uu[m] = 0
            for n in range(N):
                uu[m] = uu[m] + ww[n]*Y[n]*np.cos(xx[m]*k[n])
        
        return uu 
    
    def multipliers(self, dt = 0.001):
        beta = ( 2 + 2**(1/3) + 2**(-1/3) )/3
        N = 2*self.eq.size
        p = self.eq.degree()-1
        
        k = np.concatenate((np.arange(1, N/2+1, 1), np.arange(1-N/2, 0, 1)))
        kerne = 1j*k*self.eq.compute_kernel(k) 
        kernel = np.concatenate(([0], kerne))
        kernel[N/2] = 0
        k = np.concatenate((np.arange(0, N/2, 1), [0], np.arange(1-N/2, 0, 1)))    
        m = -0.5*dt*kernel
        
        m1 = 1./( 1 - beta*m );
        m2 = ( 1.5*beta*dt*1j*k/(2*(p+1)) )*m1; 
        mm1 = 1./( 1 - (1-2*beta)*m );
        mm2 = ( 1.5*(1-2*beta)*dt*1j*k/(2*(p+1)) )*mm1;
        
        return m1, m2, mm1, mm2
    
    def integrator(self, wave_profile, m1, m2, mm1, mm2):
        beta = ( 2 + 2**(1/3) + 2**(-1/3) )/3
        
        p = self.eq.degree()-1
        u = wave_profile

        #  ---------- STEP ONE ------------ #

        Y = fft(u)
        Z = Y
        LP = m1*Y
        for j in range(5):
            Z = LP - m2*( fft( ifft(Z).real**(p+1) )  )
            
        Y = 2*Z - Y
        Z = Y
        LP = mm1*Y    
        for j in range(5):
            Z = LP - mm2*( fft( ifft(Z).real**(p+1) )  )        
                        
        Y = 2*Z - Y
        Z = Y
        LP = m1*Y
        for j in range(5):
            Z = LP - m2*( fft( ifft(Z).real**(p+1) )  )
            
        unew = ifft( 2*Z - Y ).real
        
        #  ---------- STEP TWO ------------ #                                     
                                                                  
        Y = fft(unew)                                                                                                
        Z = .5*( (2 + beta)*Y - beta*fft(u) )
        LP = m1*Y                                                                                                                    
        for j in range(5):
            Z = LP - m2*( fft( ifft(Z).real**(p+1) )  )
        
        Z = .5*( Y + (2-beta)*fft(unew) - (1-beta)*fft(u) )
        LP = mm1*Y                                                                                                                                                                                                                                                                                                                                                            
        for j in range(5):
            Z = LP - mm2*( fft( ifft(Z).real**(p+1) )  )
    
        Y = 2*Z - Y
        Z = .5*( Y + 2*fft(unew) - fft(u) )
        LP = m1*Y
        for j in range(5):
            Z = LP - m2*( fft( ifft(Z).real**(p+1) )  )
    
        uold = u
        u = unew
        unew = ifft( 2*Z - Y ).real
    
        #  ---------- STEP THREE ------------ #
    
        q1 = .5*beta*(1+beta); q2 = beta*(2+beta); q3 = .5*(2+beta)*(1+beta);
        qq1 = .5*(2-beta)*(1-beta); qq2 = (3-beta)*(1-beta); qq3 = .5*(3-beta)*(2-beta);
        
        for j in range(1):
            Q1 = fft(q1*uold - q2*u + q3*unew)
            Q2 = fft(qq1*uold - qq2*u + qq3*unew)
            Q3 = fft(uold - 3*u + 3*unew)
            
            Y = fft(unew)
            Z = .5*( Y + Q1 )
            LP = m1*Y
            for k in range(2):
                Z = LP - m2*( fft( ifft(Z).real**(p+1) )  )
            
            Y = 2*Z - Y
            Z = .5*( Y + Q2 )
            LP = mm1*Y
            for k in range(2):
                Z = LP - mm2*( fft( ifft(Z).real**(p+1) )  )
    
            Y = 2*Z - Y
            Z = .5*( Y + Q3 )
            LP = m1*Y
            for k in range(2):
                Z = LP - m2*( fft( ifft(Z).real**(p+1) )  )
    
            uold = u; u = unew
            unew = ifft( 2*Z - Y ).real
        
        u = unew
        return u
    
    def evolution(self, solution, dt = 0.001, periods = 1):
        u = solution    
              
        T = 2*self.eq.length/self.velocity
        t = dt
        m1, m2, mm1, mm2 = self.multipliers(dt=dt)
        while t < periods*T+dt:
            w = self.integrator(wave_profile = u, m1=m1, m2=m2, mm1=mm1, mm2=mm2)
            u = w
            t = t + dt
        print 'The error for', periods,' periods is E = ', np.linalg.norm(u-solution)     
        return u