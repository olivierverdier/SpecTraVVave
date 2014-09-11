
from __future__ import division

import numpy as np
import navigation
import boundary

# various nonlinear solvers
from scipy.optimize import fsolve
import newton


def compute_parameter(parameter, direction, extent):
    """
    Computes gamma_0 + theta * direction.
    """
    return (parameter[0] + extent*direction[0],
            parameter[1] + extent*direction[1])

class Solver(object):
    def __init__(self, equation, boundary):
        self.equation = equation
        self.boundary = boundary

    def construct(self, wave, extent):
        """
        Attaches the solution wave and extent parameter together for further computation.
        """
        return np.hstack([wave, extent])

    def destruct(self, vector):
        """
        Separates the solution wave and the extent parameter.
        """
        wave = vector[:-1]
        extent = vector[-1]
        return wave, extent

    def solve(self, guess_wave, parameter_anchor, direction):
        """
        Runs a Newton solver on a system of nonlinear equations once. Takes the residual(vector) as the system to solve. 
        """
        def residual(vector):
            """
            Contructs a system of nonlinear equations. First part, main_residual, is from given wave equation; 
            second part, boundary_residual, comes from the chosen boundary conditions.
            """
            wave, extent = self.destruct(vector)
            parameter = compute_parameter(parameter_anchor , direction, extent)
            self.equation.initialize(parameter)
            boundary_residual = self.boundary(wave, parameter)
            main_residual = self.equation.residual(wave[:-1])
            return np.hstack([main_residual, boundary_residual])
        
        guess = self.construct(guess_wave, 0)
        nsolver = newton.MultipleSolver(residual)
        computed = nsolver.run(guess)
        wave, extent = self.destruct(computed)
        new_parameter = compute_parameter(parameter_anchor, direction, extent)
        
        return wave, new_parameter
