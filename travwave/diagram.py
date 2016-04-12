from __future__ import division

from travwave import navigation, solver, discretization

import numpy as np
import matplotlib.pyplot as plt

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
        aparameters = self.plot_data()
        plt.plot(aparameters[0], aparameters[1], '.--')
        plt.xlabel('Wavespeed')
        plt.ylabel('Waveheight')

    def plot_solution(self, solution):
        size = len(solution)
        self.discretization.size = size
        nodes = self.discretization.get_nodes()
        plt.plot(nodes, solution)
        plt.xlabel('x')
        plt.ylabel('Surface Elevation')

    def plot_solutions(self, index = [-1]):
        counter = np.arange(np.size(index))
        for i in counter:
            solution = self.navigation[index[i]]['solution']
            self.plot_solution(solution)
