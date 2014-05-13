from __future__ import division

import numpy as np
from functools import partial
import itertools

    
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

def step(solver, current, p1, p2, step_function, compute_line):
    pstar, direction = compute_line(p1, p2, step_function())
    new, p3 = solver(current, pstar, direction)
    return new, p2, p3

def get_stepper(solver, compute_line, step_function):
    def stepper(current, p1, p2):
         return step(current=current, p1=p1, p2=p2, solver=solver, compute_line=compute_line, step_function=step_function)
    return stepper

def get_iterator(initial, p1, p2, stepper):
    current = initial
    yield current, p1, p2
    for i in itertools.count(): # infinite iterator
        current, p2, p3 = stepper(current, p1, p2)
        yield current, p2, p3

class Navigator(object):
    """
    Run the iterator and stores the result.
    """

    def __init__(self, stepper):
        self.stepper = stepper

    def initialize (current, p, p0):
        self.store = []
        self.store.append((current, p, p0))

    def run(self, N):
        current, p2, p1 = self.store[-1]
        iterator = get_iterator(current, p1, p2, self.stepper)
        for result in islice(iterator, N):
            self.state.append(result[0], result[2], result[1])


