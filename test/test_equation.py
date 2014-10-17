#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import unittest
import numpy.testing as npt

from travwave.equation import *


class HarnessEquation(object):
    pass

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
        weights = self.equation.weights
        f = lambda x: np.cos(np.pi/self.equation.length*x)
        x = self.equation.nodes
        tensor = np.dstack([wk*(f(k*x) * f(k*x).reshape(-1,1)) for k, wk in enumerate(weights)])
        expected = np.sum(tensor, axis=2)
        computed = self.equation.compute_linear_operator()
        npt.assert_allclose(computed, expected)

    def test_residual(self):
        self.equation.initialize((2.,3.))
        u = np.random.rand(self.equation.size)
        expected = np.dot(self.equation.compute_shifted_operator(), u) + self.equation.flux(u)
        computed = self.equation.residual(u, 0)
        npt.assert_allclose(computed, expected)

class TestKDV(HarnessEquation, unittest.TestCase):
    def setUp(self):
        self.equation = KDV(8,1)

class TestWhitham(HarnessEquation, unittest.TestCase):
    def setUp(self):
        self.equation = Whitham(8,1)
