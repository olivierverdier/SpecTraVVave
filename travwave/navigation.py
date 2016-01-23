from __future__ import division

from .discretization import resample

def ortho_direction(p1, p2, step):
    """
    Returns pstar such that
        pstar = p2 + step*(p2-p1)
    """
    dp = (p2[0] - p1[0], p2[1] - p1[1])
    # orthogonal direction
    direction = (-dp[1], dp[0])

    pstar = (p2[0] + dp[0]*step, p2[1] + dp[1]*step)
    return pstar, direction

class Navigator(object):
    """
    Runs the iterator and stores the result.
    """

    def __init__(self, solve, size=32, doublings=1, correction_rate=10):
        """
        solve: solve function
        size: smallest size for the cheap steps
        doublings: the number of size doublings for the corrections
        correction_rate: how many steps between two corrections of highest size
        """
        self.solve = solve
        self.size = size
        self.doublings = doublings
        self.correction_rate = correction_rate

    # the indices for velocity and amplitude
    velocity_, amplitude_ = (0, 1)

    def __getitem__(self, index):
        return self._stored_values[index]

    def __len__(self):
        return len(self._stored_values)

    def initialize (self, current, p, p0):
        """
        Creates a list for solutions and stores the first solution (initial guess).
        """
        self._stored_values = []
        variables = [0]
        self._stored_values.append({'solution': resample(current, self.size), 'integration constant': variables, 'current': p, 'previous': p0 })

    def compute_direction(self, p1, p2, step):
        """
        Strategy for a new parameter direction search.
        """
        return ortho_direction(p1, p2, step)

    def run(self, N):
        """
        Iterates the solver N times, navigating over the bifurcation branch and storing found solutions.
        """
        for i in range(N):
            # steps with low resolution
            for j in range(self.correction_rate):
                self.step()
            # correction steps
            for k in range(self.doublings):
                self.correction(resampling=self.size*2**(k+1))
            # back to low resolution:
            self.correction(resampling=self.size)

    def prepare_step(self,step):
        """
        Return the necessary variables to run the solver.
        """
        p2 = self._stored_values[-1]['current']
        p1 = self._stored_values[-1]['previous']
        pstar, direction = self.compute_direction(p1, p2, step)
        return pstar, direction, p1, p2

    def run_solver(self, current, pstar, direction):
        new, variables, p3 = self.solve(current, pstar, direction)
        return new, variables, p3

    def correction(self, resampling):
        current = self._stored_values[-1]['solution']
        current = resample(current, resampling)
        pstar, direction, p1, p2 = self.prepare_step(0.)
        new, variables, p3 = self.run_solver(current, pstar, direction)
        self._stored_values.append({'solution': new, 'integration constant': variables, 'current': p3, 'previous': p1})

    def step(self):
        pstar, direction, _, p2 = self.prepare_step(1.)
        current = self._stored_values[-1]['solution']
        new, variables, p3 = self.run_solver(current, pstar, direction)
        self._stored_values.append({'solution': new, 'integration constant': variables, 'current': p3, 'previous': p2})
