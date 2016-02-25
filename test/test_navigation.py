#!/usr/bin/env python
# coding: utf-8
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
        self.assertIsInstance(computed_pstar, tuple, msg="stored parameter is a tuple")
        npt.assert_allclose(computed_pstar, expected_pstar, err_msg="computed parameters are right")
        expected_direction = (0,1)
        computed_direction = computed[1]
        self.assertIsInstance(computed_direction, tuple, msg="direction is a tuple")
        self.assertEqual(computed_direction[0]*dp[0] + computed_direction[1]*dp[1], 0, msg="direction is orthogonal to the line joining the two previous parameter points")

    def setUp(self):
        low_size = 24
        self.nav = Navigator(dummy_solver, size=low_size)
        init_size = 50
        x0 = np.ones(init_size)
        self.nav.initialize(x0, (1.,0), (0.,0))

    def test_nav(self):
        N = 20
        nav = self.nav
        nav.run(N)
        points = [a['solution'] for a in nav]
        self.assertEqual(len(nav), len(points), msg="Possible to compute the length of a Navigation object")
        self.assertEqual(len(nav), N+1) # Total size is number of steps plus one
        self.assertEqual(len(nav[0]['solution']), nav.size) # Right size of the initialized navigation point
        ## npt.assert_allclose(np.array(points), np.arange(N+1) + x0)
        for i in range(N):
            p2 = nav[i]['current']
            p1_ = nav[i+1]['previous']
            if len(nav[i]['solution']) == len(nav[i+1]['solution']):
                self.assertEqual(p2, p1_) # parameter is properly passed at the next stage

        py = [a['current'][nav.amplitude_] for a in nav]
        npt.assert_allclose(np.array(py), 0., err_msg="no move in the y direction because of our dummy solver")
        px = [a['previous'][nav.velocity_] for a in nav]
        npt.assert_allclose(np.array(px), np.arange(N+1), err_msg="steady move because of step function")
        ## npt.assert_allclose(px, 0., err_msg="steady move because of step function")
        def assign_nav():
            nav[0] = None
        self.assertRaises(TypeError, assign_nav, msg="Assignment not possible for Navigation object")


