#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import unittest
import numpy.testing as npt

from equation import *
from boundary import *
from solver import *
from navigation import *
from dynamic_code import *

class TestTrapezoidal(unittest.TestCase):
    def test(self):
        size = 128
        length = 3*np.pi
        equation = KDV(size, length)
        boundary = Minimum()
        solver = Solver(equation, boundary)
        nav = Navigator(solver.solve)
        initial_guess = equation.compute_initial_guess()
        initial_velocity = equation.bifurcation_velocity()
        p1 = (initial_velocity, 0)
        epsilon = .01
        p0 = (initial_velocity, -epsilon)
        nav.initialize(initial_guess, p1, p0)
        nav.run(10)

        u = nav.store[-1][0]
        velocity = nav.store[-1][2][0]
        #dyn = DeFrutos_SanzSerna(equation, u, velocity)
        dyn = Trapezoidal_rule(equation, u, velocity)
        uu = dyn.interpolation()
        #uu = np.hstack([u[::-1],u])
        t_wave = dyn.evolution(solution = uu, dt = 0.001, periods = 1)

        xx = np.arange(-length, length, length/size)
        error = t_wave - uu
        print max(abs(error))
        npt.assert_allclose(t_wave, uu, atol=1e-4)
