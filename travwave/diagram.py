from __future__ import division

from travwave import navigation, solver, discretization 

class BifurcationDiagram(object):
    
    def __init__(self, equation, size = 256):
        self.equation = equation
        self.size = size
        self.result = []
   
    def set_boundary(self, boundary_condition):
        self.boundary = boundary_condition
    
    def run(self, iter_numb = 50):
        discrete = discretization.Discretization(self.equation, self.size)
        solve = solver.Solver(discrete, boundary=self.boundary)
        nav = navigation.Navigator(solve.solve)
        initial_guess = discrete.compute_initial_guess(.01)
        initial_velocity = discrete.bifurcation_velocity()
        step = 0.001
        p1 = (initial_velocity, 0)
        p0 = (initial_velocity, -step)
        nav.initialize(initial_guess, p1, p0)
        nav.run(iter_numb)
        self.result = nav
   
         