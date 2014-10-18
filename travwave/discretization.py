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

class Discretization(object):
    def __init__(self, equation, size):
        self.equation = equation
        self.size = size
        
        self.weights = self.compute_weights()
        self.nodes = self.compute_nodes()

        self.linear_operator = self.compute_linear_operator()

         
    def compute_shifted_operator(self, size, parameters):                                             
        """
        Only used for testing purposes
        """
        return (-1)*parameters[0]*np.eye(size) + self.linear_operator

    def general_linear_operator(self, weights, nodes):
        f = np.cos
        size = len(nodes)
        ik = nodes.reshape(-1,1) * np.arange(len(weights))
        fik = f(np.pi/self.equation.length*ik) # should be replaced by a dct
        linop = np.zeros([size, size])
        _fast_make_linear_operator(linop, weights, fik)
        return linop
        
    def compute_linear_operator(self):
        return self.general_linear_operator(weights=self.weights, nodes=self.nodes)

    def residual(self, u, parameters, integrconst): 
        return np.dot(self.linear_operator, u) - parameters[0]*u + self.equation.flux(u) - integrconst

    def frequencies(self):
        return np.pi/self.equation.length*np.arange(self.size, dtype=float)

    def image(self):
        return self.equation.compute_kernel(self.frequencies())

    def bifurcation_velocity(self):
        return self.image()[1] # check this

    def shifted_kernel(self):
        return np.diag(-self.bifurcation_velocity() + self.image())

    def compute_nodes(self):
        nodes = self.equation.length*(np.linspace(0, 1, self.size, endpoint=False) + 1/2/self.size)
        return nodes

    def compute_initial_guess(self, e=0.01):
        xi1 = np.cos(np.pi/self.equation.length*self.nodes)
        init_guess = e*xi1 
        return init_guess

    def compute_weights(self):
        image = self.image()
        weights = image*2/(len(image))
        weights[0] /= 2  
        return weights
        

        
