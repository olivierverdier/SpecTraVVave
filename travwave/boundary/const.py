import numpy as np

from .base import Boundary

class Const(Boundary):
    """
    The boundary condition under which the constant of integration (B) is not considered in
    the system and always set to zero.
    """

    def enforce(self, wave, variables, parameters):
        """
        Enforces the ConstZero boundary condition. Requires a dummy variable to be zero (1st condition)
        and a constraint for navigation (2nd condition).
        """
        return np.hstack([variables[0] - self.level])

    def variables_num(self):
        """
        The number of additional variables that are required to construct the ConstZero boundary conditions.
        """
        return 1
