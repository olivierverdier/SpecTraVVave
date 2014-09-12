# -*- coding: utf-8 -*-
"""
The file contains some boundary conditions which the given equation is solved with.
NOTE: In case the user creates their own boundary conditions, the balance of the number of 
equations and unknowns should be kept. Boundary conditions should not lead to a singular
system. 
"""

from __future__ import division
import numpy as np
from equation import Equation

class ConstZero(object):
    """
    The boundary condition under which the constant of integration (B) is not considered in
    the system and always set to zero.
    """
    def enforce(self, wave, variables, parameters):
        return np.hstack([variables[0] , parameters[1] - wave[0] + wave[-1]])
    
    def variables_num(self):
        return 1

class MeanZero(ConstZero):
    """
    The boundary condition under which the constant of integration (B) is considered as an unknown and
    is a part of the solution of the system. The new unknown balanced with the requirement that the mean 
    of the solution wave is zero.
    """    
    def enforce(self, wave, variables, parameters):
        return np.hstack([variables[0] - sum(wave), parameters[1] - wave[0] + wave[-1]])

class MinimumZero(ConstZero, Equation):
    """
    The boundary condition under which the constant of integration (B) is not considered in
    the system and always set to zero. The right-most element of the solution wave is always considered to be zero,
    this feature allows computing solitary waves.
    """   
    def __init__(self, Equation):
        self.equation = Equation
        
    def reducedresidual(self, u): 
        output = np.dot(self.equation.linear_operator, u) - self.equation.parameters[0]*u + self.equation.flux(u) 
        return output[:-1]
    
    def variables_num(self):
        return 1
  
    def enforce(self, wave, variables, parameters):
        wave[-1] = 0 
        self.equation.residual = self.reducedresidual
        return np.hstack([variables[0] - sum(wave), parameters[1] - wave[0] + wave[-1]])
