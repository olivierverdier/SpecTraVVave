from __future__ import division

import math
import numpy as np
from scipy.fftpack import fft, ifft, dct 


class Trapezoidal_rule(object):
    """
    Dynamic integrator based on trapezoidal rule, 2nd order precision in dt (Li, Sattinger, 1998).  
    """
    def __init__(self, Equation, wave, velocity):
        self.eq = Equation
        self.u = wave
        self.velocity = velocity

    def interpolation(self, symm = 0):                  
        """
        Uses dct to create a symmetric image of the given solution wave
        and contruct a full wave profile. Parameter symm = 0 means that the branch is symmetric w.r.t. x = 0; 
        symm = 1 yields a branch symmetric w.r.t. x = L.
        """
        N = self.eq.size                                
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
              
    def evolution(self, solution, dt = 0.001, periods = 1):
        """
        The main body of the integrator's code. Takes full wave profile as input. Returns the result of integration of the
        given equation with input as the initial value. 
        """
        u = solution    
        NN = len(u) 
        
        k = np.concatenate((np.arange(1, NN/2+1, 1), np.arange(1-NN/2, 0, 1)))
        kerne = 1j*k*self.eq.compute_kernel(k) 
        kernel = np.concatenate(([0], kerne))
        kernel[NN/2] = 0
        k = np.concatenate((np.arange(0, NN/2, 1), [0], np.arange(1-NN/2, 0, 1)))
        
        m = 0.5*dt*kernel
        d1 = (1-m)/(1+m)
        d1[NN/2] = 0
        d2 = -0.5*1j*dt*k/(1+m)
        
        T = 2*self.eq.length/self.velocity
        eps = 1e-10
        t = dt
        it = 0
        while t < periods*T+dt:
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
        print 'The error for', periods,' periods is E = ', np.linalg.norm(u-solution)     
        return u


class DeFrutos_SanzSerna(Trapezoidal_rule):
    """
    4th order dynamic integrator based on the method of de Frutos and Sanz-Serna (1992).
    """
    def multipliers(self, timestep = 0.001):
        """
        Constructs operators used in integration.
        """
        beta = ( 2 + 2**(1/3) + 2**(-1/3) )/3
        N = 2*self.eq.size
        p = self.eq.degree()-1
        dt = timestep
        
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
    
    def iterate1(self, fftvector, coeffs1, coeffs2, p):
        """
        Used in the first step of integration.
        """
        Z = fftvector
        LP = coeffs1*fftvector
        for j in range(5):
            Z = LP - coeffs2*( fft( np.power(ifft(Z).real, p+1 ) )  )        
        return 2*Z-fftvector
    
    def iterate3(self, fftvector, coeffs1, coeffs2, Q, p):
        """
        Used in the third step of integration.
        """
        Z = .5*( fftvector + Q )   
        LP = coeffs1*fftvector
        for j in range(2):
            Z = LP - coeffs2*( fft( np.power(ifft(Z).real, p+1 ) )  )        
        return 2*Z-fftvector
        
    def integrator(self, wave_profile, m1, m2, mm1, mm2):
        beta = ( 2 + 2**(1/3) + 2**(-1/3) )/3
        
        p = self.eq.degree()-1
        u = wave_profile

        #  ---------- STEP ONE ------------ #

        Y = fft(u)
        Y = self.iterate1(Y, m1, m2, p)
        Y = self.iterate1(Y, mm1, mm2, p)        
        Y = self.iterate1(Y, m1, m2, p)
        unew = ifft( Y ).real
        
        #  ---------- STEP TWO ------------ #                                     
                                                                  
        Y = fft(unew)                                                                                                
        Z = .5*( (2 + beta)*Y - beta*fft(u) )
        LP = m1*Y                                                                                                                    
        for j in range(5):
            Z = LP - m2*( fft( np.power(ifft(Z).real, p+1 ) )  )
        
        Z = .5*( Y + (2-beta)*fft(unew) - (1-beta)*fft(u) )
        LP = mm1*Y                                                                                                                                                                                                                                                                                                                                                            
        for j in range(5):
            Z = LP - mm2*( fft( np.power(ifft(Z).real, p+1 ) )  )
    
        Y = 2*Z - Y
        Z = .5*( Y + 2*fft(unew) - fft(u) )
        LP = m1*Y
        for j in range(5):
            Z = LP - m2*( fft( np.power(ifft(Z).real, p+1 ) )  )
    
        uold = u; u = unew
        unew = ifft( 2*Z - Y ).real
    
        #  ---------- STEP THREE ------------ #
    
        q1 = .5*beta*(1+beta); q2 = beta*(2+beta); q3 = .5*(2+beta)*(1+beta);
        qq1 = .5*(2-beta)*(1-beta); qq2 = (3-beta)*(1-beta); qq3 = .5*(3-beta)*(2-beta);
        
        Q1 = fft(q1*uold - q2*u + q3*unew)
        Q2 = fft(qq1*uold - qq2*u + qq3*unew)
        Q3 = fft(uold - 3*u + 3*unew)
            
        Y = fft(unew)
        Y = self.iterate3(Y, m1, m2, Q1, p)
        Y = self.iterate3(Y, mm1, mm2, Q2, p)
        Y = self.iterate3(Y, m1, m2, Q3, p)
        
        uold = u; u = unew
        unew = ifft( Y ).real
        
        u = unew
        return u
    
    def evolution(self, solution, dt = 0.001, periods = 1):
        u = solution    
              
        T = 2*self.eq.length/self.velocity
        t = dt
        m1, m2, mm1, mm2 = self.multipliers(timestep=dt)
        while t < periods*T+dt:
            w = self.integrator(wave_profile = u, m1=m1, m2=m2, mm1=mm1, mm2=mm2)
            u = w
            t = t + dt
        print 'The error for', periods,' periods is E = ', np.linalg.norm(u-solution)     
        return u