#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import unittest
import numpy.testing as npt

import itertools

from navigation import *

def dummy_solver(x, pstar, direction):
    return x + 1., pstar


class TestDummyNavigation(unittest.TestCase):
    def test_ortho_direction(self):
        p0 = (0,0)
        p1 = (1,0)
        dp = (p1[0]-p0[0],p1[1]-p0[1])
        computed = ortho_direction(p0, p1, step=1)
        expected_pstar = (2, 0)
        computed_pstar = computed[0]
        self.assertIsInstance(computed_pstar, tuple)
        npt.assert_allclose(computed_pstar, expected_pstar)
        expected_direction = (0,1)
        computed_direction = computed[1]
        self.assertIsInstance(computed_direction, tuple)
        self.assertEqual(computed_direction[0]*dp[0] + computed_direction[1]*dp[1], 0)

    def test_nav(self):
        nav = Navigator(dummy_solver)
        N = 20
        x0 = 1.
        nav.initialize(x0, (1.,0), (0.,0))
        nav.run(N)
        points = [a[0] for a in nav.store]
        npt.assert_allclose(np.array(points), np.arange(N+1) + x0)
        for (x, p2, p1), (x_,p2_,p1_) in zip(nav.store[:-1], nav.store[1:]):
            self.assertEqual(p2, p1_) # parameter is properly passed at the next stage
        py = [a[1][1] for a in nav.store]
        npt.assert_allclose(np.array(py), 0.) # no move in the y direction because of our dummy solver
        px = [a[2][0] for a in nav.store]
        npt.assert_allclose(np.array(px), np.arange(N+1)) # steady move because of step function

