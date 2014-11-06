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
        bd = BifurcationDiagram(equation)
        bd.set_boundary(ConstZero())
        bd.run(iter_numb=10)
        print 'Amplitude = ', bd.result[-1]['current'][1]

if __name__ == '__main__':
    unittest.main()

