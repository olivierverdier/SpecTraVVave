#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import unittest

from travwave.navigation import *
from travwave.solver import *
from travwave.equations import *
from travwave.discretization import Discretization
from travwave.boundary import *

import numpy.testing as npt

import matplotlib.pyplot as plt

def sech(x):
    return 1/np.cosh(x)

def soliton(amplitude, x):
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

    def get_length(self):
        return 10

    def get_nbsteps(self):
        return 1

    def get_size(self):
        return 128

    def setUp(self):
        size = self.get_size()
        length = self.get_length()
        self.equation = self.get_equation_class()(length)
        self.discretization = Discretization(self.equation, size)
        self.boundary = self.get_boundary()
        solver = Solver(self.discretization, self.boundary)
        nb_steps = self.get_nbsteps()
        nav = Navigator(solver.solve, size=size, doublings=0, correction_rate=nb_steps)
        initial_velocity = self.discretization.bifurcation_velocity()
        p1 = (initial_velocity, 0)
        epsilon = .1/nb_steps
        p0 = (initial_velocity, -epsilon)
        initial_guess = self.discretization.compute_initial_guess(epsilon/10)
        nav.initialize(initial_guess, p1, p0)
        nav.run(1)
        self.nav = nav
        store = nav.store[-1]
        self.B = store['integration constant']
        self.c = store['current'][nav.velocity_]
        self.amplitude = store['current'][nav.amplitude_]
        self.xs = self.discretization.get_nodes()
        self.computed = store['solution']

    def get_residual_tolerance(self):
        return 1e-6

    def test_residual(self):
        res = self.residual(self.c, self.B, self.xs, self.computed)
        npt.assert_allclose(res, 0, atol=self.get_residual_tolerance())

    def test_boundary(self):
        level = self.boundary.level
        npt.assert_allclose(min(self.computed), level, atol=1e-15)
        

    @classmethod
    def get_equation_class(self):
        return kdv.KDV

    def get_boundary(self, level=.1):
        return Minimum(level)

    def residual(self, c, B, xs, us):
        u = identity(xs,us)
        res = (1-c) * u + 3/4*u*u + 1/6*derivative2(xs,us) - B
        return res
        

class TestKDVMean(TestKDV):
    def get_boundary(self):
        return MeanZero()

    @unittest.skip("Remove when Mean boundary condition is implemented")
    def test_boundary(self):
        pass

class TestKDVConstZero(TestKDVMean):
    def get_boundary(self):
        return ConstZero()


class TestKDVSoliton(TestKDV):
    def get_length(self):
        return 50

    def get_nbsteps(self):
        return 10

    def get_residual_tolerance(self):
        return 1e-4

    def get_boundary(self):
        return Minimum(0)

    def test_soliton(self):
        expected = soliton(self.amplitude, self.xs)
        npt.assert_allclose(self.computed, expected, atol=1e-3)

@unittest.skip("Takes too long")
class TestKDVSolitonLarge(TestKDVSoliton):
    def get_size(self):
        return 32

    def get_length(self):
        return 75

    def get_nbsteps(self):
        return 50

