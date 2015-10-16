from __future__ import division

import unittest

from travwave.equations import *
from travwave.diagram import *
from travwave.boundary import *
import numpy.testing as npt

class TestGeneral(unittest.TestCase):
    """
    Only tests that the problem can be set up and run without raising any exception.
    No tests are actually performed.
    """
    def test_run(self):
        length = 5
        equation = kdv.KDV(length)
        boundary_cond = Const()
        bd = BifurcationDiagram(equation, boundary_cond)
        bd.navigation.run(10)
        print('Amplitude = ', bd.navigation[-1]['current'][bd.navigation.amplitude_])

if __name__ == '__main__':
    unittest.main()

