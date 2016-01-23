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

    def __init__(self, solve, size=32):
        """
        solve: solve function
        size: size for the navigation
        """
        self.solve = solve
        self.size = size

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
            self.step()

    def prepare_step(self, step, index=-1):
        """
        Return the necessary variables to run the solver.
        """
        p2 = self._stored_values[index]['current']
        p1 = self._stored_values[index]['previous']
        pstar, direction = self.compute_direction(p1, p2, step)
        return pstar, direction, p1, p2

    def run_solver(self, current, pstar, direction):
        new, variables, p3 = self.solve(current, pstar, direction)
        return new, variables, p3

    def refine(self, resampling, index=-1):
        sol = self._stored_values[index]['solution']
        sol = resample(sol, resampling)
        pstar, direction, p1, p2 = self.prepare_step(0., index)
        new, variables, p3 = self.run_solver(sol, pstar, direction)
        return new, variables, p3

    def step(self):
        pstar, direction, _, p2 = self.prepare_step(1.)
        current = self._stored_values[-1]['solution']
        new, variables, p3 = self.run_solver(current, pstar, direction)
        self._stored_values.append({'solution': new, 'integration constant': variables, 'current': p3, 'previous': p2})

