#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import unittest

from navigation import *
from solver import *
from equation import *
from boundary import *

import numpy.testing as npt

import matplotlib.pyplot as plt

def sech(x):
    return 1/np.cosh(x)

def exact_kdv(amplitude, x):
    s = sech(np.sqrt(3*amplitude)/2*x)
    return amplitude*s*s

def identity(xs, us):
    return us[1:-1]

def derivative(xs, us):
    du = us[2:] - us[:-2]
    dx = xs[2:] - xs[:-2]
    return du/dx

def derivative2(xs, us):
    ddu = us[2:] - 2*us[1:-1] + us[:-2]
    dx = xs[1]-xs[0]
    return ddu/(dx*dx)

class TestKDV(unittest.TestCase):

    @classmethod
    def get_equation_class(self):
        return KDV

    def residual(self, c, B, xs, us):
        u = identity(xs,us)
        res = (1-c) * u + 3/4*u*u + 1/6*derivative2(xs,us) - B
        return res

    def test(self):
        size = 128
        length = 10
        ## length = np.pi
        self.equation = self.get_equation_class()(size, length)
        boundary = MeanZero()
        solver = Solver(self.equation, boundary)
        nav = Navigator(solver.solve)
        initial_guess = self.equation.compute_initial_guess()
        initial_velocity = self.equation.bifurcation_velocity()
        p1 = (initial_velocity, 0)
        epsilon = .1
        p0 = (initial_velocity, -epsilon)
        nav.initialize(initial_guess, p1, p0)
        nav.run(1)
        self.nav = nav
        store = nav.store[-1]
        B = store[1][0]
        c = store[2][0]
        xs = self.equation.nodes
        computed = store[0]
        res = self.residual(c, B, xs, computed)
        npt.assert_allclose(res, 0, atol=1e-6)
        
