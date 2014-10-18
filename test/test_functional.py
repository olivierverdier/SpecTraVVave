#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import unittest

from travwave.navigation import *
from travwave.solver import *
from travwave.equations import *
from travwave.discretization import Discretization
from travwave.boundary import *

import numpy as np

class TestGeneral(unittest.TestCase):
    def test_run(self):
        size = 10
        length = np.pi
        equation = kdv.KDV(length)
        discretization = Discretization(equation, size)
        boundary = MeanZero()
        solver = Solver(discretization, boundary)
        nav = Navigator(solver.solve)
        initial_velocity = discretization.bifurcation_velocity()
        p1 = (initial_velocity, 0)
        epsilon = .1
        initial_guess = discretization.compute_initial_guess(epsilon)
        p0 = (initial_velocity, -epsilon)
        nav.initialize(initial_guess, p1, p0)
        nav.run(1)
        print nav.store[1]

if __name__ == '__main__':
    unittest.main()
#
