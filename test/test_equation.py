#!/usr/bin/env python
# coding: utf-8
from __future__ import division

import unittest
import numpy.testing as npt
import numpy as np

from travwave.equations import kdv, whitham, benjamin, kawahara
from travwave.discretization import DiscretizationOperator

class HarnessEquation(object):
    def setUp(self):
        self.equation = self.get_equation()
        self.discretization = DiscretizationOperator(self.equation, 8)

    def test_flux(self):
        """
        Test that flux_prime works
        """
        u = .5
        eps = 1e-5
        expected = (self.equation.flux(u+eps) - self.equation.flux(u))/eps
        computed = self.equation.flux_prime(u)
        npt.assert_allclose(computed, expected, rtol=1e-4)

    def test_degree(self):
        """
        The degree is correct.
        """
        A = 1e10
        degree = np.log(self.equation.flux(A)/self.equation.flux(1))/np.log(A)
        npt.assert_allclose(self.equation.degree(), degree)

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

class TestKDV3(HarnessEquation, unittest.TestCase):
    def get_equation(self):
        return kdv.KDV3(1)

class TestKDV5(HarnessEquation, unittest.TestCase):
    def get_equation(self):
        return kdv.KDV5(1)

class TestWhitham(HarnessEquation, unittest.TestCase):
    def get_equation(self):
        return whitham.Whitham(1)

class TestWhitham3(HarnessEquation, unittest.TestCase):
    def get_equation(self):
        return whitham.Whitham3(1)

class TestWhitham5(HarnessEquation, unittest.TestCase):
    def get_equation(self):
        return whitham.Whitham5(1)

class TestWhithamsqrt(HarnessEquation, unittest.TestCase):
    def get_equation(self):
        return whitham.Whithamsqrt(1)

    def test_degree(self):
        """The flux is not a monomial"""
        pass

class TestKawahara(HarnessEquation, unittest.TestCase):
    def get_equation(self):
        return kawahara.Kawahara(1)

class TestBenjaminOno(HarnessEquation, unittest.TestCase):
    def get_equation(self):
        return benjamin.Benjamin_Ono(1)

class TestMBenjaminOno(HarnessEquation, unittest.TestCase):
    def get_equation(self):
        return benjamin.modified_Benjamin_Ono(1)


def whitham_kernel_(k):
    whitham = np.zeros(len(k))
    for i in range(len(k)):
        if k[i] == 0:
            whitham[i] = 1
        else:
            whitham[i]  = np.sqrt(1./k[i]*np.tanh(k[i]))
    return whitham

def test_whitham_refactor():
    w = whitham.Whitham(1.)
    ks = np.arange(32, dtype=float)
    new = w.compute_kernel(ks)
    old = whitham_kernel_(ks)
    npt.assert_allclose(new, old)
