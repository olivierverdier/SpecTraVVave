# The file includes the solver class. 
# wave = solver(matrix, guess_vector) 
# wave.matrix = the matrix of the system bieng solved
# wave.guess = the initial guess with which the system is solved
# wave.compute() = the solution of the system
from __future__ import division

import numpy as np
from scipy.optimize import fsolve


class solver(object):
    def __init__(self, Equation, guess):
        self.guess = guess
        self.equation = Equation

    def compute_fsolve(self):
       
            
        return fsolve(self.equation.residual, self.guess, fprime=self.equation.Jacobian)
        # in the case of fsolve being slow, it is to be replaced by Newton method 
        
    
    def compute_newton(self, tol=1e-12):
        u = self.guess
        for it in range(10000):
         #   DFu = self.equation.matrix + np.diag(self.equation.flux_prime(u))
            du = np.linalg.solve(self.equation.Jacobian(u), -self.equation.residual(u))
            #corr = np.dot(-np.linalg.inv(DFu),(np.dot(self.equation.matrix,u) + self.equation.flux(u)))
            unew = u + du
            change = np.abs(du).max()
            u = unew
            if change < tol:             # Newton iterative solver for the obtained system of equations
                break

        else:
            print 'Iterations\' limit reached: 10000'

            
        return u
