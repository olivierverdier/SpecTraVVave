#!/usr/bin/env python
# coding: utf-8
from __future__ import division

import unittest
import numpy.testing as npt

from travwave.equations import *
from travwave.boundary import *
from travwave.dynamic import *
from travwave.diagram import *


class TestTrapezoidal(unittest.TestCase):
    def test(self):
        length = 3*np.pi
        equation = kdv.KDV(length)
        boundary_cond = Const()
        bd = BifurcationDiagram(equation, boundary_cond, doublings=1)
        bd.navigation.run(10)

        u = bd.navigation[-2]['solution']
        velocity = bd.navigation[-2]['current'][bd.navigation.velocity_]
        dyn = Trapezoidal_rule(equation, u, velocity)
        uu = dyn.mirror()
        t_wave = dyn.evolution(solution = uu, nb_steps=int(1e2), periods = 1)

        error = t_wave - uu
        print(max(abs(error)))
        npt.assert_allclose(t_wave, uu, atol=1e-3, err_msg="Wave is equal to itself after travelling one period")

class TestDeFrutos(unittest.TestCase):
    @unittest.skip("code duplication")
    def test(self):
        size = 128
        length = 3*np.pi
        equation = kdv.KDV(length)
        boundary_cond = Const()
        bd = BifurcationDiagram(equation, boundary_cond)
        bd.navigation.run(10)

        u = bd.navigation[-1]['solution']
        velocity = bd.navigation[-1]['current'][bd.navigation.velocity_]
        dyn = Trapezoidal_rule(equation, u, velocity)
        uu = dyn.mirror()
        t_wave = dyn.evolution(solution = uu, nb_steps=int(1e2), periods = 1)

        error = t_wave - uu
        print(max(abs(error)))
        npt.assert_allclose(t_wave, uu, atol=1e-3, err_msg="Wave is equal to itself after travelling one period")

