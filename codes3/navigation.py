from __future__ import division
    
def ortho_direction(p1, p2, step):
    """
    Return pstar such that
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

    def __init__(self, solve):
        self.solve = solve

    def initialize (self, current, p, p0):
        """
        Creates a list for solutions and stores the first solution (initial guess).
        """
        self.store = []
        variables = [0]
        self.store.append((current, variables, p, p0))

    def parameter_step(self):
        """
        Estimate a parameter step.
        """
        return 1.

    def compute_direction(self, p1, p2):
        """
        Strategy for a new parameter direction search.
        """
        return ortho_direction(p1, p2, self.parameter_step())

    def run(self, N):
        """
        Iterates the solver N times, navigating over the bifurcation brach and storing found solutions.
        """
        for i in range(N):
            current, variables, p2, p1 = self.store[-1]
            pstar, direction = self.compute_direction(p1, p2)
            new, variables, p3 = self.solve(current, pstar, direction)
            self.store.append((new, variables, p3, p2))