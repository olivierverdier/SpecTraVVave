# -*- coding: utf-8 -*-
"""
The file contains some boundary conditions which the given equation is solved with.
NOTE: In case the user creates their own boundary conditions, the balance of the number of 
equations and unknowns should be kept. Boundary conditions should not lead to a singular
system. 
"""

from __future__ import division
import numpy as np

def const_zero(wave, parameters):
    """
    The boundary condition under which the constant of integration (B) is not considered in
    the system and always set to zero.
    """
    return np.hstack([wave[-1] , parameters[1] - wave[0] + wave[-2]])
    
def mean_zero(wave, parameters):
    """
    The boundary condition under which the constant of integration (B) is considered as an unknown and
    is a part of the solution of the system. The new unknown balanced with the requirement that the mean 
    of the solution wave is zero.
    """    
    return np.hstack([wave[-1] - sum(wave[:-1]), parameters[1] - wave[0] + wave[-2]])

def minimum_zero(wave, parameters):
    """
    The boundary condition under which the constant of integration (B) is not considered in
    the system and always set to zero.
    """    
    return np.hstack([wave[-2], parameters[1] - wave[0] + wave[-2]])
