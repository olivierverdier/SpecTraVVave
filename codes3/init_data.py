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

import functools

dct = functools.partial(scipy.fftpack.dct, norm='ortho')
idct = functools.partial(scipy.fftpack.idct, norm='ortho')


class InitData(object):
    def __init__(self, Equation):
        self.eq = Equation

    def compute_init_guess(self, e = 0.1):

        fluxCoef      =     self.eq.fluxCoef                # 1/p! * flux^(p)(0) 
        fluxDegree    =     self.eq.degree                  # the degree of zeros of flux 

        # move the following to the Equation class
        operatorA     =     self.eq.shifted_kernel()    # diagonal matrix of the Fourier multiplier operator acting on xi_p on the LHS        

        operatorA[1,1] = 1                                  # the second diagonal element of the matrix is set to 1 in order to invert the matrix
                                                            # later this element will be multiplied by zero, and the effect will be balanced

        xi1 = np.cos(self.eq.nodes)                         # xi_1 which is always the cosine of x

        eqRHShat   =     -fluxCoef*dct(xi1**fluxDegree) # take the dct of the right-hand-side of the equation
        eqRHShat[1] = 0                                 # actually it holds true, but this stands here as compensation 
                                                        # b/w c1 and rhs_hat(1) to cancel the cos(x) terms  

        xiphat = np.linalg.solve(operatorA, eqRHShat)   # solving the equation in the Fourier domain

        xip = idct(xiphat)                              # finding  the solution of the equation , the correction to velocity here is zero -> c1 = 0.

        init_guess = e*xi1 + e**fluxDegree*xip

        return init_guess
