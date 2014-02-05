from __future__ import division

import numpy as np
import scipy.fftpack 

"""
    Transforms the eqution 
                    A*xi = -flux(xi)
with the Discrete Cosine Transform and computes the first two elements of the cosine expansion
for xi -- xi1 and xip, and for velocity c -- cp (or cp_1).
    Operator A  = (-c0 + Fourier_Multiplier_Operator_L)
    Vector   xi = surface elevation.
"""
class init_data(object):
    def __init__(self, Equation):
        self.eq = Equation

    def compute_init_guess(self, e = 0.1):
        dct = lambda x: scipy.fftpack.dct(x, norm='ortho')                 # short declaration of the dct and idct operators
        idct = lambda x: scipy.fftpack.idct(x, norm='ortho')

        fluxCoef      =     self.eq.fluxCoef                # 1/p! * flux^(p)(0) 
        fluxDegree    =     self.eq.degree                  # the degree of zeros of flux 
        fourierFreqs  =     np.arange(self.eq.size)         # Fourier modes same as Fourier frequencies
        fourierKernel =     self.eq.kernel(fourierFreqs)    # Fourier multiplier function of the linear operator
        
        c0            =     fourierKernel[1]                # bifurcation point in velocity        
        operatorA     =     np.diag(-c0 + fourierKernel)    # diagonal matrix of the Fourier multiplier operator acting on xi_p on the LHS        
        operatorA[1,1] = 1                                  # the second diagonal element of the matrix is set to 1 in order to invert the matrix
                                                            # later this element will be multiplied by zero, and the effect will be balanced
        
        xi1 = np.cos(self.eq.nodes)                         # xi_1 which is always the cosine of x
        dctFactor = dct(xi1)[1]                                # this factor is used to compute pure dct coefficients without weights of transformation

        if fluxDegree == 2:                                 # fluxDegree = 2 is special case becasue xi2 != cos(x)
            eqRHShat   =     -fluxCoef*dct(xi1**fluxDegree) # take the dct of the right-hand-side of the equation
            eqRHShat[1] = 0                                 # actually it holds true, but this stands here as compensation 
                                                            # b/w c1 and rhs_hat(1) to cancel the cos(x) terms  
            
            xi2hat = np.linalg.solve(operatorA, eqRHShat)   # solving the equation in the Fourier domain
            
            xi2 = idct(xi2hat)                              # finding  the solution of the equation , the correction to velocity here is zero -> c1 = 0.
            
            eqRHShat = -fluxCoef*dct(fluxDegree*xi1**(fluxDegree-1)*xi2)  # computing next epsilon level of approximation to compute the c2
            c2 = -eqRHShat[1]/dctFactor                                   # computing the corretion to the velocity.
            init_guess = e*xi1 + e**fluxDegree*xi2                        # declaring the initial guess for wave shape
            init_velocity = c0 + e**fluxDegree*c2                         # declaring the initial guess for the velocity

        if fluxDegree > 2 and fluxDegree%2 == 0:                         # fluxDegree==even and > 2 CANNOT BE HANDLED AT THE MOMENT! 
            eqRHShat   =     -fluxCoef*dct(xi1**fluxDegree)
            eqRHShat[1] = 0
            xiphat = np.linalg.solve(operatorA, eqRHShat)
            xip = idct(xiphat)
            eqRHShat = -fluxCoef*dct(fluxDegree*xi1**fluxDegree)                      # THIS HAS TO BE REVISED!
            cp = -eqRHShat[1]/dctFactor
            init_guess = e*xi1 + e**fluxDegree*xip
            init_velocity = c0 + e**fluxDegree*cp
            print "This case cannot be handled a the moment."
            
        if fluxDegree%2 == 1:                                            # in case fluxDegree==odd there are fewer steps to compute the initial shape and velocity
            eqRHShat   =     -fluxCoef*dct(xi1**fluxDegree)
            cp_1 = -eqRHShat[1]/dctFactor
            eqRHShat[1] = 0
            xiphat = np.linalg.solve(operatorA, eqRHShat)
            xip = idct(xiphat)
            init_guess = e*xi1 + e**fluxDegree*xip
            init_velocity = c0 + e**(fluxDegree-1)*cp_1
            
        return (init_guess, init_velocity)