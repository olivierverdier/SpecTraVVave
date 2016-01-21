from __future__ import division

import numpy as np
from . import newton

def compute_parameter(parameter, direction, extent):
    """
    Computes gamma_0 + theta * direction.
    """
    return (parameter[0] + extent*direction[0],
            parameter[1] + extent*direction[1])

class Solver(object):
    def __init__(self, discretization, boundary):
        self.discretization = discretization
        self.boundary = boundary

    def construct(self, wave, variables, extent):
        """
        Attaches the solution wave and extent parameter together for further computation.
        """
        return np.hstack([wave, variables, extent])

    def destruct(self, vector):
        """
        Separates the solution wave and the extent parameter.
        """
        n = self.boundary.variables_num()
        wave = vector[:-(1+n)]
        variables = vector[-(1+n):-1]
        extent = vector[-1]
        return wave, variables, extent

    def solve(self, guess_wave, parameter_anchor, direction):
        """
        Runs a Newton solver on a system of nonlinear equations once. Takes the residual(vector) as the system to solve.
        """
        size = len(guess_wave)
        self.discretization.size = size

        def residual(vector):
            """
            Contructs a system of nonlinear equations. First part, main_residual, is from given wave equation;
            second part, boundary_residual, comes from the chosen boundary conditions.
            """
            wave, variables, extent = self.destruct(vector)
            parameter = compute_parameter(parameter_anchor, direction, extent)
            boundary_residual = self.boundary.enforce(wave, variables, parameter)
            amplitude_residual = np.array([parameter[1] - wave[0] + wave[-1]])
            main_residual = self.discretization.residual(wave, parameter, variables)
            return np.hstack([main_residual, boundary_residual, amplitude_residual])

        guess = self.construct(guess_wave, np.zeros(self.boundary.variables_num()), 0)
        nsolver = newton.MultipleSolver(residual)
        computed = nsolver.run(guess)
        wave, variables, extent = self.destruct(computed)
        new_parameter = compute_parameter(parameter_anchor, direction, extent)

        return wave, variables, new_parameter
