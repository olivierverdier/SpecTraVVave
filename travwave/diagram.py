from __future__ import division

from travwave import navigation, solver, discretization

import numpy as np

class BifurcationDiagram(object):

    def __init__(self, equation, boundary_condition, size=32, init_size=256):
        self.discretization = discretization.Discretization(equation, init_size)
        solve = solver.Solver(self.discretization, boundary=boundary_condition)
        nav = navigation.Navigator(solve.solve, size=size)
        self.navigation = nav

    def initialize(self, amplitude=0.01, step=0.005):
        initial_guess = self.discretization.compute_initial_guess(amplitude)
        initial_velocity = self.discretization.bifurcation_velocity()
        p1 = (initial_velocity, 0)
        p0 = (initial_velocity, -step)
        self.navigation.initialize(initial_guess, p1, p0)

    def plot_data(self):
        parameters = [result['current'] for result in self.navigation]
        aparameters = np.array(parameters)
        return aparameters.T

    def plot_diagram(self):
        from matplotlib.pyplot import plot, xlabel, ylabel
        import numpy as np
        nav = self.navigation
        parameters = [result['current'] for result in nav]
        aparameters = np.array(parameters)
        plot(aparameters[:,0], aparameters[:,1], '.--')
        xlabel('Wavespeed')
        ylabel('Waveheight')

    def plot_solution(self, index = [-1]):
        from matplotlib.pyplot import plot, xlabel, ylabel
        import numpy as np
        counter = np.arange(np.size(index))
        for i in counter:
            solution = self.navigation[index[i]]['solution']
            size = len(solution)
            self.discretization.size = size
            nodes = self.discretization.get_nodes()
            plot(nodes, solution)
        xlabel('x')
        ylabel('Surface Elevation')
