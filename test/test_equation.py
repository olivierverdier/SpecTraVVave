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
        """
        The numba implementation of the linear operator gives the same results as the previous one.
        """
        weights = self.discretization.get_weights()
        f = lambda x: np.cos(np.pi/self.equation.length*x)
        x = self.discretization.get_nodes()
        tensor = np.dstack([wk*(f(k*x) * f(k*x).reshape(-1,1)) for k, wk in enumerate(weights)])
        expected = np.sum(tensor, axis=2)
        computed = self.discretization.compute_linear_operator()
        npt.assert_allclose(computed, expected)

    def test_residual(self):
        """
        The discretization residual is equal to the linear operator plus the flux.
        """
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

class DummyDiscretization(Discretization):
    """
    A mock Discretization object.
    """
    def __init__(self, *args, **kwargs):
        self._count = {}
        super(DummyDiscretization, self).__init__(*args, **kwargs)

    def compute_linear_operator(self):
        self._count[self.size] = self._count.get(self.size, 0) + 1
        return super(DummyDiscretization, self).compute_linear_operator()
        
class TestCaching(unittest.TestCase):
    def test_caching(self):
        """
        Caching works.
        """
        start_size = 8
        d = DummyDiscretization(kdv.KDV(1), start_size)
        self.assertEqual(d._count[8], 1, msg="Operator is computed in __init__")
        self.assertNotIn(16, d._count, msg="Other operators are not computed yet")
        d.size = 16
        self.assertEqual(d._count[16], 1, msg="Operator of size 16 is computed")
        d.size = 8
        self.assertEqual(d._count[8], 1, "Operator of size 8 is not recomputed")

