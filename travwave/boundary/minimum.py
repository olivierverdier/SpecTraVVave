import numpy as np

from .base import Boundary

class Minimum(Boundary):
    """
    The boundary condition under which the constant of integration (B) is not considered in
    the system and always set to zero. The right-most element of the solution wave is always considered to be zero,
    this feature allows computing solitary waves.
    """
    def variables_num(self):
        return 1

    def enforce(self, wave, variables, parameters):
        return np.hstack([wave[-1] - self.level])
