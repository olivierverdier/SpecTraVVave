#!/usr/bin/env python
# coding: utf-8
from __future__ import division

import unittest
import numpy.testing as npt

import backend

from travwave.equations import *
from travwave.boundary import *
from travwave.dynamic import *
from travwave.diagram import *


class TestDynamic(unittest.TestCase):
    def test_dynamid(self):
        length = 3*np.pi
        equation = kdv.KDV(length)
        boundary_cond = Const()
        bd = BifurcationDiagram(equation, boundary_cond)
        bd.initialize()
        bd.navigation.run(10)

        u = bd.navigation[-1]['solution']
        velocity = bd.navigation[-1]['parameter'][bd.navigation.velocity_]
        dyn = self.get_dynamic_class()(equation, u, velocity)
        uu = dyn.mirror()
        t_wave = dyn.evolution(solution = uu, nb_steps=int(1e2), periods = 1)

        error = t_wave - uu
        print(max(abs(error)))
        npt.assert_allclose(t_wave, uu, atol=1e-3, err_msg="Wave is equal to itself after travelling one period")

    def get_dynamic_class(self):
        return Trapezoidal_rule

class TestDefrutos(TestDynamic):
    def get_dynamic_class(self):
        return DeFrutos_SanzSerna
