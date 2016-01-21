# -*- coding: utf-8 -*-
"""
The file contains some boundary conditions which the given equation is solved with.
NOTE: In case the user creates their own boundary conditions, the balance of the number of
equations and unknowns should be kept. Boundary conditions should not lead to a singular
system.
"""

from __future__ import division
import numpy as np
from scipy.integrate import trapz

class Const(object):
    """
    The boundary condition under which the constant of integration (B) is not considered in
    the system and always set to zero.
    """
    def __init__(self, level=0):
        self.level = level

    def enforce(self, wave, variables, parameters):
        """
        Enforces the ConstZero boundary condition. Requires a dummy variable to be zero (1st condition)
        and a constraint for navigation (2nd condition).
        """
        return np.hstack([variables[0] - self.level])

    def variables_num(self):
        """
        The number of additional variables that are required to construct the ConstZero boundary conditions.
        """
        return 1

class Mean(Const):
    """
    The boundary condition under which the constant of integration (B) is considered as an unknown and
    is a part of the solution of the system. The new unknown balanced with the requirement that the mean
    of the solution wave is zero.
    """
    def enforce(self, wave, variables, parameters):
        return np.hstack([sum(wave) - self.level])

class Minimum(Const):
    """
    The boundary condition under which the constant of integration (B) is not considered in
    the system and always set to zero. The right-most element of the solution wave is always considered to be zero,
    this feature allows computing solitary waves.
    """
    def variables_num(self):
        return 1

    def enforce(self, wave, variables, parameters):
        return np.hstack([wave[-1] - self.level])
