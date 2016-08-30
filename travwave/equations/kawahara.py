#!/usr/bin/env python
# coding: utf-8
from __future__ import division

from .base import Equation

class Kawahara (Equation):
    """
    The equation is :     -c*u + 3/4u^2 + (u - 1/2*kp*u" + 1/90*u'''')=0
    """
    def degree(self):
        return 2

    def compute_kernel(self, k):
        kp = 1
        return 1.0+0.5*kp*k**2 + 1.0/90*k**4

    def flux(self, u):
        return 0.75*u**2

    def flux_prime(self, u):
        return 1.5*u
