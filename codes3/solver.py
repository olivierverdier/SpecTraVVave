# The file includes the solver class. 
# wave = solver(matrix, guess_vector) 
# wave.matrix = the matrix of the system bieng solved
# wave.guess = the initial guess with which the system is solved
# wave.compute() = the solution of the system
from __future__ import division

import numpy as np
import navigation
from scipy.optimize import fsolve



def compute_parameter(parameter, direction, extent):
    """
    Compute gamma_0 + theta * direction
    """
    return parameter + extent*direction

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
            boundary_residual = self.equation.boundary(wave, parameter)
            return np.hstack([main_residual, boundary_residual])
        
        guess = self.construct(guess_wave, 0)
        print residual(guess)
        computed = fsolve(residual, guess)
        wave, extent = self.destruct(computed)
        new_parameter = compute_parameter(parameter_anchor, direction, extent)
        
        return wave, new_parameter
 




""" 
    def compute_extended_jacobian(jacobian, ortho, u):
    N = len(jacobian)
    ah = np.zeros(N)
    ah[0] = 1
    ah[-1] = -1

    ch = np.zeros(N)

    # adding 2 rows from below to Jacobian
    Vmatrix = np.vstack((jacobian, ah, ch))

    av0 = np.zeros(N)
    av = np.hstack((av0,[-1,ortho[1]]))

    cv0 = (-1)*u
    cv = np.hstack((cv0, [0, ortho[0]]))

    # adding 2 columns
    extended_jacobian = np.hstack((Vmatrix, av.reshape(N+2,1), cv.reshape(N+2,1)))
    return extended_jacobian
    def residual(self):
        equation = equation_class(parameters)
        ## return residual = [self.equation.residual, np.dot(self.equation.parameters, navigation.Navigator(????))] 
        residual = [equation.residual(), equation.boundary()]
        
    def compute_fsolve(self):
        
        return fsolve(residual(), [self.guess, step])
        # in the case of fsolve being slow, it is to be replaced by Newton method       
    def compute_newton(self, c1, a1, c2, a2, tol=1e-12):
        u = self.guess
        # N = self.equation.size
        
        nav = navigation.Navigation((c1, a1), (c2, a2))
        (pstar, ortho) = nav.compute_line()                       # computing the init guess for (c, a) 
        
        
        for it in xrange(10000):

            extended_jacobian = compute_extended_jacobian(self.equation.Jacobian(u), ortho, u)
             
            du = np.linalg.solve(extended_jacobian, np.hstack((-self.equation.residual(u), [u[0] - u[-1] - pstar[1], ortho[1]*pstar[1] + ortho[0]*pstar[0]])) )
            
            unew = u + du[:-2]
            cnew = pstar[0] + du[-1]
            anew = pstar[1] + du[-2]
            change = np.abs(du).max()
            u = unew
            pstar[0] = cnew
            pstar[1] = anew
            print it, change
            if change < tol:             # Newton iterative solver for the obtained system of equations
                break

        else:
            print 'Iterations\' limit reached: 10000'

            
        return (u, pstar[0], pstar[1])
"""
