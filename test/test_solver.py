#!/usr/bin/env python
# coding: utf-8
from __future__ import division

import unittest
import numpy.testing as npt

from travwave.solver import *

@unittest.skip("The Jacobian is not implemented")
class TestJacobianExtension(unittest.TestCase):
    def test_extension(self):
        jacobian = np.zeros((3,3))
        N = len(jacobian)
        ortho = (1.,2.)
        extended = compute_extended_jacobian(jacobian, ortho)
        self.assertEqual(extended.shape, (N+2,N+2))
        last_rows = extended[-2:]
        ## last_columns = extended[:, -2:]
        npt.assert_allclose(last_rows[:, -2:], np.array([[-1, 0], [ortho[1], ortho[0]]]))
