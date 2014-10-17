from __future__ import division
import math
import numpy as np
import solver
import init_data
import dynamic_code
import equation as eq
import navigation
from matplotlib.pyplot import plot, show

             ###### testing the solver.py functions

# basic parameters' values to initiate computation
N = 64          #number of nodes 
L = math.pi     #half of the wave length 
eps = 0.1
whitham = eq.Whitham(N, L)
speed = whitham.bifurcation_velocity()
u0 = whitham.compute_initial_guess()
amplitude = u0[0]-u0[-1]
initial_parameters = [speed, amplitude]

x = whitham.compute_nodes()             
deg = whitham.degree()
whitham = whitham.initialize(initial_parameters)

solution = solver.Solver(whitham)
nav = navigation.Navigator(solution)
p1 = (speed, 0)
epsilon = 0.1
p0 = (speed, - epsilon)
nav.initialize([u0, 0.2 ], p1, p0)
nav.run(1)