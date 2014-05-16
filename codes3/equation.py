# -*- coding: UTF-8 -*-
from __future__ import division

import math
import numpy as np

class Equation(object):
    def __init__(self,  size, length, parameters):
#       self.velocity = velocity
        self.size = size
        self.length = length
        self. parameters = parameters      # contains: [bifurcation velocity, wave amplitude]
        
        self.weights = self.compute_weights()
        self.nodes = self.compute_nodes()

        self.linear_operator = self.compute_linear_operator()

        self.shifted_operator = self.compute_shifted_operator()
         
    def compute_shifted_operator(self):
        return (-1)*self.parameters[0]*np.eye(self.size) + self.linear_operator

    @classmethod
    def general_tensor(self, w, x, f):
        tensor = np.dstack([wk*(f(k*x) * f(k*x).reshape(-1,1)) for k, wk in enumerate(w)])
        return tensor

    @classmethod
    def general_linear_operator(self, w, x, f):
        ten = self.general_tensor(w, x, f)
        return np.sum(ten, axis=2)
        
    def Jacobian(self, u):
        return self.shifted_operator + np.diag(self.flux_prime(u))

    def residual(self, u): 
        return np.dot(self.shifted_operator, u) + self.flux(u)
    
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

