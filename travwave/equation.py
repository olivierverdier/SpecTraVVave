from __future__ import division

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
        self.size = size
        self.length = length
        
        self.weights = self.compute_weights()
        self.nodes = self.compute_nodes()

        self.linear_operator = self.compute_linear_operator()

         
    def compute_shifted_operator(self, size):                                             
        """
        Only used for testing purposes
        """
        return (-1)*self.parameters[0]*np.eye(size) + self.linear_operator

    def initialize(self, parameters):                                               
        self.parameters = parameters

    def general_linear_operator(self, weights, nodes):
        f = np.cos
        size = len(nodes)
        ik = nodes.reshape(-1,1) * np.arange(len(weights))
        fik = f(np.pi/self.length*ik) # should be replaced by a dct
        linop = np.zeros([size, size])
        _fast_make_linear_operator(linop, weights, fik)
        return linop
        
    def compute_linear_operator(self):
        return self.general_linear_operator(weights=self.weights, nodes=self.nodes)

    def residual(self, u, integrconst): 
        return np.dot(self.linear_operator, u) - self.parameters[0]*u + self.flux(u) - integrconst

    def frequencies(self):
        return np.pi/self.length*np.arange(self.size, dtype=float)

    def image(self):
        return self.compute_kernel(self.frequencies())

    def bifurcation_velocity(self):
        return self.image()[1] # check this

    def shifted_kernel(self):
        return np.diag(-self.bifurcation_velocity() + self.image())

    def compute_nodes(self):
        nodes = self.length*(np.linspace(0, 1, self.size, endpoint=False) + 1/2/self.size)
        return nodes

    def compute_initial_guess(self, e=0.01):
        xi1 = np.cos(np.pi/self.length*self.nodes)
        init_guess = e*xi1 
        return init_guess

    def compute_weights(self):
        image = self.image()
        weights = image*2/(len(image))
        weights[0] /= 2  
        return weights
        

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
        
    def flux(self, u):
        return 0.75*u**2  

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
    
    def flux(self, u):
        scale = self.length/np.pi
        return 0.75*u**2  

    def flux_prime(self, u):
        scale = self.length/np.pi
        return 1.5*u

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
            
    def flux(self, u):
        return 0.75*u*u

    def flux_prime(self, u):
        return 1.5*u

class KDV3 (KDV):
    def degree(self):
        return 3
          
    def flux(self, u):
        return 0.5*u**3  

    def flux_prime(self, u):
        return 1.5*u**2

class KDV5 (KDV):
    def degree(self):
        return 5
          
    def flux(self, u):
        return 0.5*u**5  

    def flux_prime(self, u):
        return 1.5*u**4
        
class Kawahara (Equation):
    """
    The equation is :     -c*u + 3/4u^2 + (u - 1/2*kp*u" + 1/90*u'''')=0
    """
    def degree(self):
        return 2

    def flux_coefficient(self):
        return 3/4

    def compute_kernel(self, k):
        kp = 1
        return 1.0+0.5*kp*k**2 + 1.0/90*k**4
            
    def flux(self, u):
        return 0.75*u**2  

    def flux_prime(self, u):
        return 1.5*u

class Whithamsqrt (Whitham):
    def degree(self):
        return 1.5
 
    def flux(self, u):
        return 2*np.power(u+1, 1.5) - 3*u - 2

    def flux_prime(self, u):
        return 3*(u+1)**(0.5)-3
