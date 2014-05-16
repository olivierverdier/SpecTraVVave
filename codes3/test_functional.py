#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import unittest

from navigation import *
from solver import *
from equation import *

import numpy as np

class TestGeneral(unittest.TestCase):
    def test_run(self):
        size = 10
        length = np.pi
        equation = KDV(size, length)
        solver = Solver(equation)
        nav = Navigator(solver)
        initial_guess = equation.compute_initial_guess()
        initial_velocity = equation.bifurcation_velocity()
        p1 = (initial_velocity, 0)
        epsilon = .1
        p0 = (initial_velocity, -epsilon)
        nav.initialize(initial_guess, p, p0)
        nav.run(1)
        print nav.store[1]
