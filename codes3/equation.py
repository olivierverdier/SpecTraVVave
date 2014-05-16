# -*- coding: UTF-8 -*-
from __future__ import division

import math
import numpy as np

class Equation(object):
    def __init__(self,  size, length):
#       self.velocity = velocity
        self.size = size
        self.length = length
        
        self.weights = self.compute_weights()
        self.nodes = self.compute_nodes()

        self.linear_operator = self.compute_linear_operator()

         
    def compute_shifted_operator(self):
        return (-1)*self.parameters[0]*np.eye(self.size) + self.linear_operator

    def initialize(self, parameters):
        self.parameters = parameters

    @classmethod
    def general_tensor(self, w, x, f):
        tensor = np.dstack([wk*(f(k*x) * f(k*x).reshape(-1,1)) for k, wk in enumerate(w)])
        return tensor

    @classmethod
    def general_linear_operator(self, w, x, f):
        ten = self.general_tensor(w, x, f)
        return np.sum(ten, axis=2)
        
    def Jacobian(self, u):
        return self.compute_shifted_operator + np.diag(self.flux_prime(u))

    def residual(self, u): 
        return np.dot(self.compute_shifted_operator, u) + self.flux(u)
    
    def frequencies(self):
        return np.arange(self.size, dtype=float)

    def image(self):
        return self.compute_kernel(self.frequencies())

    def bifurcation_velocity(self):
        return self.image()[1]

    def shifted_kernel(self):
        # note: the image method is called twice
        return np.diag(-self.bifurcation_velocity() + self.image())

    def compute_nodes(self):
        h = self.length/self.size
        nodes = np.arange(h/2.,self.length,h)
        return nodes

    def compute_initial_guess(self, e=0.):

        fluxCoef      =     self.fluxCoef                # 1/p! * flux^(p)(0) 
        fluxDegree    =     self.degree                  # the degree of zeros of flux 

        # move the following to the Equation class
        operatorA     =     self.shifted_kernel()    # diagonal matrix of the Fourier multiplier operator acting on xi_p on the LHS        

        operatorA[1,1] = 1                                  # the second diagonal element of the matrix is set to 1 in order to invert the matrix
                                                            # later this element will be multiplied by zero, and the effect will be balanced

        xi1 = self.cosine()

        eqRHShat   =     -fluxCoef*dct(xi1**fluxDegree) # take the dct of the right-hand-side of the equation
        eqRHShat[1] = 0                                 # actually it holds true, but this stands here as compensation 
                                                        # b/w c1 and rhs_hat(1) to cancel the cos(x) terms  

        xiphat = np.linalg.solve(operatorA, eqRHShat)   # solving the equation in the Fourier domain

        xip = idct(xiphat)                              # finding  the solution of the equation , the correction to velocity here is zero -> c1 = 0.

        init_guess = e*xi1 + e**fluxDegree*xip

        return init_guess

class Whitham(Equation):
    def degree(self):
        return 2    
    
    def flux_coefficient(self):
        return 3/4
    
    def compute_kernel(self,k):
        if k[0] == 0:
            k1 = k[1:]
            whitham = np.concatenate( ([1], np.sqrt(1./k1*np.tanh(k1))))
        else:    
            whitham  = np.sqrt(1./k*np.tanh(k))
        
        return whitham
        
    def compute_weights(self):
        ks = np.arange(self.size, dtype=float)
        ww = 2/self.size
        weights = self.compute_kernel(ks)*ww
        weights[0] = 1/self.size
        return weights
        

    def compute_linear_operator(self):
        return self.general_linear_operator(w=self.weights, x=self.nodes, f=np.cos)

    def flux(self, u):
        """
        Return the flux f(u)
        """
        return 2*(u+1)**(1.5)-3*u-2

    def flux_prime(self, u):
        """
        Derivative of the flux
        """
        return 3*(u+1)**(0.5)-3


class KDV(Equation):
    """
    The equation is :     -c*u + 3/4u^2 + (u + 1/6u")=0
    """
    def degree(self):
        return 2

    def flux_coefficient(self):
        return 3/4

    def compute_kernel(self, k):
        return 1.0-1.0/6*k**2
            
    def compute_weights(self):
        ks = np.arange(self.size, dtype=float)
        ww = 2/self.size
        weights = self.compute_kernel(ks)*ww  
        weights[0] = weights[0]/2  
        return weights
        
        
    def compute_linear_operator(self):
        return self.general_linear_operator(w=self.weights, x=self.nodes, f=np.cos)

    def flux(self, u):
        return 0.75*u**2  

    def flux_prime(self, u):
        return 1.5*u


class KDV1(KDV):
    # The equation is :     -c*u + 3/4u^3 + (u + 1/6u")=0
    def degree(self):
        return 3
    
    def flux(self, u):
        return 0.75*u**3  

    def flux_prime(self, u):
        return 2.25*u**2

