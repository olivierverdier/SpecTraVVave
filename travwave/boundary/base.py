class Boundary(object):
    def __init__(self, level=0):
        self.level = level

    def enforce(self, wave, variables, parameters):
        """
        Enforces the boundary condition.
        """
        raise NotImplementedError()

    def variables_num(self):
        """
        The number of additional variables that are required to construct the ConstZero boundary conditions.
        """
        raise NotImplementedError()
