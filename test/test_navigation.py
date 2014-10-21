#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import unittest
import numpy.testing as npt
import numpy as np

import itertools

from travwave.navigation import *

def dummy_solver(x, pstar, direction):
    return x + 1., 0, pstar


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
        N = 20
        corr_rate = 10
        doublings = 1
        nav = Navigator(dummy_solver, doublings=doublings, correction_rate=corr_rate)
        x0 = np.ones(10)
        nav.initialize(x0, (1.,0), (0.,0))
        nav.run(N)
        points = [a['solution'] for a in nav]
        self.assertEqual(len(nav), len(points))
        ## npt.assert_allclose(np.array(points), np.arange(N+1) + x0)
      
        for i in range(N):
            p2 = nav[i]['current']
            p1_ = nav[i+1]['previous']
            self.assertEqual(p2, p1_) # parameter is properly passed at the next stage

        py = [a['current'][nav.amplitude_] for a in nav]
        npt.assert_allclose(np.array(py), 0.) # no move in the y direction because of our dummy solver
        px = [a['previous'][nav.velocity_] for a in nav]
        npt.assert_allclose(np.array(px), np.arange(N*corr_rate+1)) # steady move because of step function
        def assign_nav():
            nav[0] = None
        self.assertRaises(TypeError, assign_nav, msg="Assignment not possible for Navigation object")


