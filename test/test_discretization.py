import unittest
import numpy as np
import numpy.testing as npt

import pytest

from travwave.equations.kdv import KDV
from travwave.discretization import DiscretizationOperator

from tools import equations

@pytest.fixture(params=equations, ids=repr)
def discretization(request):
    return DiscretizationOperator(equation=request.param, size=8)

class TestLinear(unittest.TestCase):
    def test_compare(self):
        size = 64
        d = DiscretizationOperator(equation=KDV(np.pi), size=size)
        op = d.compute_linear_operator()
        u = np.random.randn(size)
        # u_ = # same with an initial guess
        npt.assert_allclose(np.dot(op, u), d.apply_operator(u))


def test_linop(discretization):
    """
    The numba implementation of the linear operator gives the same results as the previous one.
    """
    weights = discretization.get_weights()
    f = lambda x: np.cos(np.pi/discretization.equation.length*x)
    x = discretization.get_nodes()
    tensor = np.dstack([wk*(f(k*x) * f(k*x).reshape(-1,1)) for k, wk in enumerate(weights)])
    expected = np.sum(tensor, axis=2)
    computed = discretization.compute_linear_operator()
    npt.assert_allclose(computed, expected)

def test_residual(discretization):
    """
    The discretization residual is equal to the linear operator plus the flux.
    """
    parameter = (2.,3.)
    u = np.random.rand(discretization.size)
    expected = np.dot(discretization.compute_shifted_operator(discretization.size, parameter), u) + discretization.equation.flux(u)
    computed = discretization.residual(u, parameter, 0)
    npt.assert_allclose(computed, expected)
