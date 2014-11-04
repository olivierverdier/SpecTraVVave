#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

from base import Equation
import numpy as np

class Whitham(Equation):
    def degree(self):
        return 2    

    def compute_kernel(self,k):
        if k[0] == 0:
            k1 = k[1:]
            whitham = np.concatenate( ([1], np.sqrt(1./k1*np.tanh(k1))))
        else:    
            whitham  = np.sqrt(1./k*np.tanh(k))        
        return whitham
        
    def flux(self, u):
        return 0.75*u*u  

    def flux_prime(self, u):
        return 1.5*u

class Whitham_scaled(Whitham):
    def compute_kernel(self,k):
        scale = self.length/np.pi
        if k[0] == 0:
            k1 = k[1:]
            whitham = np.sqrt(scale) * np.concatenate( ([1], np.sqrt(1./k1*np.tanh(1/scale * k1))))
        else:    
            whitham  = np.sqrt(scale) * np.sqrt(1./k*np.tanh(1/scale * k))
       
        return whitham    

class Whitham3(Whitham):
    def degree(self):
        return 3  
        
    def flux(self, u):
        return 0.5*u**3  

    def flux_prime(self, u):
        return 1.5*u**2
        
class Whitham5(Whitham):
    def degree(self):
        return 5
          
    def flux(self, u):
        return 0.5*u**5  

    def flux_prime(self, u):
        return 1.5*u**4
        
class Whithamsqrt (Whitham):
    def degree(self):
        return 1.5
 
    def flux(self, u):
        return 2*np.power(u+1, 1.5) - 3*u - 2

    def flux_prime(self, u):
        return 3*(u+1)**(0.5)-3
