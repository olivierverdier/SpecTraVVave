#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import unittest
import numpy.testing as npt

from navigation import *

class TestNavigation(unittest.TestCase):
    def test_main(self):
        p0 = (0,0)
        p1 = (1,0)
        computed = compute_line(p0, p1, d=1)
        expected_pstar = (2, 0)
        computed_pstar = computed[0]
        npt.assert_allclose(computed_pstar, expected_pstar)
        expected_ortho_orthogonal = (0,1)
        ortho_computed = computed[1]
        self.assertEqual(ortho_computed[0]*expected_ortho_orthogonal[0] + ortho_computed[1]*expected_ortho_orthogonal[1], 0)

