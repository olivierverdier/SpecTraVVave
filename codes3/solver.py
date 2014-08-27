
from __future__ import division

import numpy as np
import navigation

# various nonlinear solvers
from scipy.optimize import fsolve
import newton


def compute_parameter(parameter, direction, extent):
    """
    Compute gamma_0 + theta * direction
    """
    return (parameter[0] + extent*direction[0],
            parameter[1] + extent*direction[1])

class Solver(object):
    def __init__(self, equation):
        self.equation = equation

    def construct(self, wave, extent):
        return np.hstack([wave, extent])

    def destruct(self, vector):
        wave = vector[:-1]
        extent = vector[-1]
        return wave, extent

    def solve(self, guess_wave, parameter_anchor, direction):
        
        def residual(vector):
            wave, extent = self.destruct(vector)
            parameter = compute_parameter( parameter_anchor , direction, extent)
            self.equation.initialize(parameter)
            main_residual = self.equation.residual(wave)
            boundary_residual = self.equation.boundary(wave)
            return np.hstack([main_residual, boundary_residual])
        
        guess = self.construct(guess_wave, 0)
        nsolver = newton.MultipleSolver(residual)
        computed = nsolver.run(guess)
        wave, extent = self.destruct(computed)
        new_parameter = compute_parameter(parameter_anchor, direction, extent)
        
        return wave, new_parameter
