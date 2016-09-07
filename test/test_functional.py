#!/usr/bin/env python
# coding: utf-8
from __future__ import division

import unittest

from travwave.equations import *
from travwave.boundary import *
from travwave.diagram import BifurcationDiagram

import numpy as np

class TestGeneral(unittest.TestCase):
    """
    Only tests that the problem can be set up and run without raising any exception.
    No tests are actually performed.
    """
    def test_run(self):
        length = np.pi
        equation = kdv.KDV(length)
        boundary = Mean()
        epsilon = .1
        bd = BifurcationDiagram(equation=equation, boundary_condition=boundary)
        bd.initialize(amplitude=epsilon, step=epsilon)
        bd.navigation.run(8)

