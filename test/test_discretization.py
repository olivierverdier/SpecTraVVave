import unittest
import numpy as np
import numpy.testing as npt

from travwave.equations.kdv import KDV
from travwave.discretization import Discretization

class TestLinear(unittest.TestCase):
    def test_compare(self):
        size = 64
        d = Discretization(equation=KDV(np.pi), size=size)
        op = d.compute_linear_operator()
        u = np.random.randn(size)
        # u_ = # same with an initial guess
        npt.assert_allclose(np.dot(op, u), d.apply_operator(u))

        


