from __future__ import division

from .base import Equation

class KDV(Equation):
    """
    The equation is :     -c*u + 3/4u^2 + (u + 1/6u")=0
    """
    def degree(self):
        return 2

    def compute_kernel(self, k):
        return 1.0-1.0/6*k**2

    def flux(self, u):
        return 0.75*u*u

    def flux_prime(self, u):
        return 1.5*u

class KDV3 (KDV):
    def degree(self):
        return 3

    def flux(self, u):
        return 0.5*u**3

    def flux_prime(self, u):
        return 1.5*u**2

class KDV5 (KDV):
    def degree(self):
        return 5

    def flux(self, u):
        return 0.5*u**5

    def flux_prime(self, u):
        return 2.5*u**4
