#!/usr/bin/env python
# coding: utf-8
from __future__ import division

import numpy as np
from scipy.fftpack import fft, ifft, dct 

from .trapezoidal import Trapezoidal_rule

class DeFrutos_SanzSerna(Trapezoidal_rule):
    """
    4th order dynamic integrator based on the method of de Frutos and Sanz-Serna (1992). The class changes the evolution() method of the
    Trapezoidal_rule class and adds its own methods. 
    """
    def multipliers(self, timestep = 0.001):
        """
        Constructs operators used in integration.
        """
        beta = ( 2 + 2**(1/3) + 2**(-1/3) )/3
        NN = 2*len(self.u)
        p = self.equation.degree()
        dt = timestep
        scale = self.equation.length/np.pi
        
        centered_frequencies = 1/scale * np.concatenate((np.arange(1, NN/2+1, 1), np.arange(1-NN/2, 0, 1)))
        kerne = 1j * centered_frequencies * self.equation.compute_kernel(centered_frequencies) 
        kernel = np.concatenate(([0], kerne))
        kernel[NN/2] = 0

        shifted_frequencies = 1/scale * np.concatenate((np.arange(0, NN/2, 1), [0], np.arange(1-NN/2, 0, 1)))
        
        m = -0.5*dt*kernel
        
        m1 = 1./( 1 - beta*m );
        m2 = ( self.equation.flux_prime(1) * beta * dt * 1j * shifted_frequencies/(2*p) )*m1; 
        mm1 = 1./( 1 - (1-2*beta)*m );
        mm2 = ( self.equation.flux_prime(1) * (1-2*beta) * dt * 1j * shifted_frequencies/(2*p) )*mm1;
        
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
        
    def iterate2(self, fftvector, Z, coeffs1, coeffs2, p):
        """
        Used in the second step of integration.
        """
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
        """
        The main algorithm for integration based on De Frutos and Sanz-Serna findings.
        """
        beta = ( 2 + 2**(1/3) + 2**(-1/3) )/3
        
        p = self.equation.degree()-1
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
       
        Y = self.iterate2(Y, Z, m1, m2, p)                      
        Z = .5*( Y + (2-beta)*fft(unew) - (1-beta)*fft(u) )
        Y = self.iterate2(Y, Z, mm1, mm2, p)
        
        Z = .5*( Y + 2*fft(unew) - fft(u) )
        Y = self.iterate2(Y, Z, m1, m2, p)
    
        uold = u; u = unew
        unew = ifft( Y ).real
    
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
    
    def evolution(self, solution, nb_steps=1000, periods = 1):
        """
        
        """
        u = solution                  

        T = 2*self.equation.length/self.velocity
        dt = periods*T/nb_steps
        m1, m2, mm1, mm2 = self.multipliers(timestep=dt)
        for i in range(nb_steps):
            w = self.integrator(wave_profile = u, m1=m1, m2=m2, mm1=mm1, mm2=mm2)
            u = w
        return u
