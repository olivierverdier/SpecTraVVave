import numpy as np

from .base import Boundary

class Mean(Boundary):
    """
    The boundary condition under which the constant of integration (B) is considered as an unknown and
    is a part of the solution of the system. The new unknown balanced with the requirement that the mean
    of the solution wave is zero.
    """
    def enforce(self, wave, variables, parameters):
        return np.hstack([sum(wave) - self.level])

    def variables_num(self):
        return 1
