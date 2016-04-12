from __future__ import division

import unittest

from travwave.diagram import BifurcationDiagram
import travwave.equations as teq
import travwave.boundary as tbc

from travwave.discretization import Discretization
from travwave.solver import Solver
from travwave.navigation import Navigator


import numpy.testing as npt

class TestRefine(unittest.TestCase):
    def setUp(self):
        length = 5
        equation = teq.kdv.KDV(length)
        boundary_cond = tbc.Const()
        self.bd = bd = BifurcationDiagram(equation, boundary_cond)
        bd.initialize()
        bd.navigation.run(10)

    def test_refine(self):
        new_size = 500
        n,v,p = self.bd.navigation.refine_at(new_size)
        self.assertEqual(len(n), new_size)

    def test_plot_diagram(self):
        self.bd.plot_diagram()

    def test_plot_solutions(self):
        self.bd.plot_solutions()

class TestGeneral(unittest.TestCase):
    """
    Only tests that the problem can be manually set up and run without raising any exception.
    No tests are actually performed.
    """
    def get_size(self):
        return 50

    def get_length(self):
        return 100

    def get_equation_class(self):
        return teq.kdv.KDV

    def get_boundary(self):
        return tbc.Minimum(0)

    def get_nbsteps(self):
        return 10

    def test_manual_initialization(self):
        size = self.get_size()
        length = self.get_length()
        self.equation = self.get_equation_class()(length)
        self.boundary = self.get_boundary()

        self.diagram = BifurcationDiagram(self.equation, self.boundary, size, size)

        self.discretization = Discretization(self.equation, size)
        solver = Solver(self.discretization, self.boundary)
        nb_steps = self.get_nbsteps()
        nav = Navigator(solver.solve, size=size)
        initial_velocity = self.discretization.bifurcation_velocity()
        p1 = (initial_velocity, 0)
        epsilon = .1/nb_steps
        p0 = (initial_velocity, -epsilon)
        initial_guess = self.discretization.compute_initial_guess(epsilon/10)
        nav.initialize(initial_guess, p1, p0)
        nav.run(1)
        self.nav = nav
        store = nav[-1]
        self.B = store['integration constant']
        self.c = store['current'][nav.velocity_]
        self.amplitude = store['current'][nav.amplitude_]
        self.xs = self.discretization.get_nodes()
        self.computed = store['solution']
