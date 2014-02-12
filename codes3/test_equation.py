#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import unittest
import numpy.testing as npt

from equation import *


class HarnessEquation(object):
    pass

    def test_flux(self):
        u = .5
        eps = 1e-5
        expected = (self.equation.flux(u+eps) - self.equation.flux(u))/eps
        computed = self.equation.flux_prime(u)
        npt.assert_allclose(computed, expected, rtol=1e-4)


class TestKDV(HarnessEquation, unittest.TestCase):
    def setUp(self):
        self.equation = KDV(1,1,1)

class TestKDV1(HarnessEquation, unittest.TestCase):
    def setUp(self):
        self.equation = KDV1(1,1,1)

class TestWhitham(HarnessEquation, unittest.TestCase):
    def setUp(self):
        self.equation = Whitham(1,1,1)
