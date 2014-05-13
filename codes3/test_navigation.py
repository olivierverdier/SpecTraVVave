#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import unittest
import numpy.testing as npt

import itertools

from navigation import *

def dummy_solver(x, pstar, direction):
    return x + 1., (pstar[0]+.5*direction[0], pstar[1]+.5*direction[1])

def dummy_step_function(*args, **kwargs):
    return 1.

class TestDummyNavigation(unittest.TestCase):
    def test_main(self):
        p0 = (0,0)
        p1 = (1,0)
        dp = (p1[0]-p0[0],p1[1]-p0[1])
        computed = ortho_direction(p0, p1, step=1)
        expected_pstar = (2, 0)
        computed_pstar = computed[0]
        npt.assert_allclose(computed_pstar, expected_pstar)
        expected_direction = (0,1)
        computed_direction = computed[1]
        self.assertEqual(computed_direction[0]*dp[0] + computed_direction[1]*dp[1], 0)

    def test_stepper(self):
        stepper = get_stepper(solver=dummy_solver, compute_line=ortho_direction, step_function=dummy_step_function)
        print stepper(0., (0.,1.), (0., 2))

    def test_iterator(self):
        stepper = get_stepper(solver=dummy_solver, compute_line=ortho_direction, step_function=dummy_step_function)
        iterator = get_iterator(0., (0.,1.), (0.,2.), stepper=stepper)
        for val in itertools.islice(iterator, None, 5):
            print val


