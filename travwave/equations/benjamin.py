from __future__ import division

from base import Equation
import numpy as np

class Benjamin_Ono(Equation):
    """
    The equation is :     -c*u + u + 1/2*u^2 + H(u_x)=0
    """
    def degree(self):
        return 2

    def compute_kernel(self, k):        
        return  1.0 - np.abs(k)
            
    def flux(self, u):
        return 0.5*u*u

    def flux_prime(self, u):
        return u

class modified_Benjamin_Ono(Benjamin_Ono):
    """
    The equation is :     -c*u + u + 1/3*u^3 + H(u_x)=0
    """
    def degree(self):
        return 3

    def flux(self, u):
        return 1/3*u*u*u

    def flux_prime(self, u):
        return u*u