# -*- coding: UTF-8 -*-
from __future__ import division

import math
import numpy as np
import numba

def _make_linear_operator(linop, weights, fik):
    """
    fik: array(n,k)
    linop: array(n,n) of zeros
    weights: array(k)
    """
    size = len(fik)
    for k in range(len(weights)):
        wk = weights[k]
        fk = fik[:,k]
        for i in range(size):
            for j in range(size):
                linop[i,j] += wk * fk[i] * fk[j]
    
_fast_make_linear_operator = numba.jit('void(f8[:,:], f8[:], f8[:,:])', nopython=True)(_make_linear_operator)

class Equation(object):
    def __init__(self,  size, length):
#       self.velocity = velocity
        self.size = size
        self.length = length
        
        self.weights = self.compute_weights()
        self.nodes = self.compute_nodes()

        self.linear_operator = self.compute_linear_operator()

         
    def compute_shifted_operator(self):                                             # gives the (-c*I + L) operator of the equation
        return (-1)*self.parameters[0]*np.eye(self.size) + self.linear_operator

    def initialize(self, parameters):                                               # needed to initialize the equation with required parameters (c,a)
        self.parameters = parameters

    @classmethod
    def general_linear_operator(self, weights, nodes):
        f = np.cos
        size = len(nodes)
        ik = nodes.reshape(-1,1) * np.arange(len(weights))
        fik = f(ik) # should us dct instead
        linop = np.zeros([size, size])
        _fast_make_linear_operator(linop, weights, fik)
        return linop
        
    def compute_linear_operator(self):
        return self.general_linear_operator(weights=self.weights, nodes=self.nodes)

    def Jacobian(self, u):
        return self.compute_shifted_operator + np.diag(self.flux_prime(u))

    def residual(self, u): 
        return np.dot(self.linear_operator, u) - self.parameters[0]*u + self.flux(u)
    
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

    def compute_initial_guess(self, e=0.01):
        xi1 = np.cos(self.nodes)
        init_guess = e*xi1 
        return init_guess

    def boundary(self, wave):
        return self.parameters[1] - wave[0] + wave[-1]

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

