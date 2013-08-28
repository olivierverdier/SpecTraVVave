import math
import numpy as np
from scipy.optimize import fsolve
import routns

def init_guess(x):
    return -0.015-0.1*np.cos(x)
    
    
def sys_matrix(dimension, speed_c):
    c = speed_c
    Tau = routns.lin_op(math.pi, 64)
    MatrixOfSystem = -c*np.eye(dimension)+Tau
    return MatrixOfSystem

def solve_fsolve(matrix, flux, guess):
    equation = np.dot(matrix, guess) + flux(guess)
    return fsolve(equation, guess)

def solve_newton(matrix, flux, dflux, guess):
    change = 2
    it = 0
    while change > 1e-12:             # Newton iterative solver for the obtained system of equations
        if it > 10000:                 # stopping criteria: reaching tolerance level or 10000 iterations
            break
        DFu = matrix + np.diag(dflux(guess))
        corr = np.dot(-np.linalg.inv(DFu),(np.dot(matrix,guess) + flux(guess)))
        unew = guess + corr
        change = corr.max()
        guess = unew
        it = it + 1
    return guess

            ###### testing the above defined functions

# basic parameters' values to initiate computation
speed = 0.99*math.sqrt(math.tanh(1))
x_nodes = np.arange(math.pi/128., math.pi, math.pi/64)
u0 = init_guess(x_nodes) # initial guess

            # solutions with given parameters
# solution via Netwon method
x0 = solve_newton(sys_matrix(len(u0), speed), routns.flux, routns.D, u0)

# solution with fsolve
x1 = solve_fsolve(sys_matrix(len(u0), speed), routns.flux, u0)

print(x0)
print(x1)