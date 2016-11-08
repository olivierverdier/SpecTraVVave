#!/usr/bin/env python
# coding: utf-8
from __future__ import division

import unittest

import backend

import numpy as np

from travwave.equations import kdv
from travwave.boundary import Minimum, Mean, Const
from travwave.diagram import BifurcationDiagram

import numpy.testing as npt

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

    def setUp(self):
        size = 128
        length = self.get_length()
        self.equation = self.get_equation_class()(length)
        self.boundary = self.get_boundary()
        nb_steps = self.get_nbsteps()
        epsilon = .01

        bd = BifurcationDiagram(equation=self.equation, boundary_condition=self.boundary)
        bd.initialize(amplitude=epsilon/10, step=epsilon)

        nav = self.nav = bd.navigation

        nav.run(nb_steps)
        store = nav[-1]
        self.computed, self.B, self.parameter = nav.refine_at(resampling=size)
        self.c = self.parameter[nav.velocity_]
        self.amplitude = self.parameter[nav.amplitude_]

        self.xs = bd.discretization.get_nodes()

    def get_residual_tolerance(self):
        return 1e-4

    def test_residual(self):
        """
        Discretized residual is below tolerance when applied to the computed solution.
        """
        res = self.residual(self.c, self.B, self.xs, self.computed)
        npt.assert_allclose(res, 0, atol=self.get_residual_tolerance())

    def test_boundary(self):
        """
        Correct boundary conditions are enforced in the solution.
        """
        level = self.boundary.level
        npt.assert_allclose(min(self.computed), level, atol=1e-7)


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
        return Mean()

    @unittest.skip("Remove when Mean boundary condition is implemented")
    def test_boundary(self):
        pass

class TestKDVConstZero(TestKDVMean):
    def get_boundary(self):
        return Const()


class TestKDVSoliton(TestKDV):
    def get_length(self):
        return 50

    def get_nbsteps(self):
        return 10

    def get_residual_tolerance(self):
        return 1e-3

    def get_boundary(self):
        return Minimum(0)

    def test_soliton(self):
        """
        With a sufficiently great length, the travelling wave is close to a soliton.
        """
        expected = soliton(self.amplitude, self.xs)
        npt.assert_allclose(self.computed, expected, atol=1e-2)

    def test_diagram(self):
        """
        The theoretical relation between a and c is a = 2(c-1)
        """
        npt.assert_allclose(self.amplitude, 2*(self.c - 1), rtol=1e-1)

class TestKDVSolitonLarge(TestKDVSoliton):

    def get_length(self):
        return 75

    def get_nbsteps(self):
        return 30
