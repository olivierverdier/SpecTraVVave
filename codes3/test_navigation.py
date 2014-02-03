#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import unittest

from navigation import *

class TestNavigation(unittest.TestCase):
    def test_main(self):
        nav = Navigation((0,0), (1,0))
        computed = nav.compute_line(d=1)
        expected_pstar = (2, 0)
        computed_pstar = computed[0]
        self.assertEqual(computed_pstar, expected_pstar)
        expected_ortho_orthogonal = (0,1)
        ortho_computed = computed[1]
        self.assertEqual(ortho_computed[0]*expected_ortho_orthogonal[0] + ortho_computed[1]*expected_ortho_orthogonal[1], 0)

