# The file includes the solver class. 
# wave = solver(matrix, guess_vector) 
# wave.matrix = the matrix of the system bieng solved
# wave.guess = the initial guess with which the system is solved
# wave.compute() = the solution of the system


import numpy as np
from scipy.optimize import fsolve

class solver(object):
    def __init__(self, Matrx, flux, guess):
        self.matrix = Matrx
        self.guess = guess
        self.flux = flux
        
    def compute_fsolve(self):
        solution = fsolve(np.dot(self.matrix, self.guess) + self.flux)
        # in the case of fsolve being slow, it is to be replaced by Newton method 
        return solution
    
    def compute_newton(self):
        