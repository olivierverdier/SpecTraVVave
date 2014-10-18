#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import unittest
import numpy.testing as npt
import numpy as np

from travwave.equations import *
from travwave.discretization import Discretization

class HarnessEquation(object):
    def setUp(self):
        self.equation = self.get_equation()
        self.discretization = Discretization(self.equation, 8)

    def test_flux(self):
        """
        Test that flux_prime works
        """
        u = .5
        eps = 1e-5
        expected = (self.equation.flux(u+eps) - self.equation.flux(u))/eps
        computed = self.equation.flux_prime(u)
        npt.assert_allclose(computed, expected, rtol=1e-4)

    def test_linop(self):
        weights = self.discretization.get_weights()
        f = lambda x: np.cos(np.pi/self.equation.length*x)
        x = self.discretization.get_nodes()
        tensor = np.dstack([wk*(f(k*x) * f(k*x).reshape(-1,1)) for k, wk in enumerate(weights)])
        expected = np.sum(tensor, axis=2)
        computed = self.discretization.compute_linear_operator()
        npt.assert_allclose(computed, expected)

    def test_residual(self):
        parameter = (2.,3.)
        u = np.random.rand(self.discretization.size)
        expected = np.dot(self.discretization.compute_shifted_operator(self.discretization.size, parameter), u) + self.equation.flux(u)
        computed = self.discretization.residual(u, parameter, 0)
        npt.assert_allclose(computed, expected)

class TestKDV(HarnessEquation, unittest.TestCase):
    def get_equation(self):
        return kdv.KDV(1)

class TestWhitham(HarnessEquation, unittest.TestCase):
    def get_equation(self):
        return whitham.Whitham(1)
