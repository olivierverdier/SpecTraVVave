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
        dp = (p1[0]-p0[0],p1[1]-p0[1])
        computed = compute_line(p0, p1, step=1)
        expected_pstar = (2, 0)
        computed_pstar = computed[0]
        npt.assert_allclose(computed_pstar, expected_pstar)
        expected_direction = (0,1)
        computed_direction = computed[1]
        self.assertEqual(computed_direction[0]*dp[0] + computed_direction[1]*dp[1], 0)

