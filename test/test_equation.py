#!/usr/bin/env python
# coding: utf-8
from __future__ import division

import unittest
import pytest
import numpy.testing as npt
import numpy as np

from tools import equations, poly_equations


@pytest.fixture(params=equations, ids=repr)
def equation(request):
    return request.param

@pytest.fixture(params=poly_equations, ids=repr)
def poly_equation(request):
    return request.param

def test_flux(equation):
    """
    Test that flux_prime works
    """
    u = .5
    eps = 1e-5
    expected = (equation.flux(u+eps) - equation.flux(u))/eps
    computed = equation.flux_prime(u)
    npt.assert_allclose(computed, expected, rtol=1e-4)

def test_degree(poly_equation):
    """
    The degree is correct.
    """
    equation = poly_equation
    A = 1e10
    degree = np.log(equation.flux(A)/equation.flux(1))/np.log(A)
    npt.assert_allclose(equation.degree(), degree)


from travwave.equations import whitham

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
