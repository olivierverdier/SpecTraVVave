#!/usr/bin/env python
# coding: utf-8
from __future__ import division

import numpy as np
from scipy.fftpack import fft, ifft, dct

class Trapezoidal_rule(object):
    """
    Dynamic integrator based on trapezoidal rule, 2nd order precision in dt (Li, Sattinger, 1998).
    """
    def __init__(self, equation, wave, velocity):
        self.equation = equation
        self.u = wave
        self.velocity = velocity

    def mirror(self):
        """
        Mirrors the half-period profile to a full period profile.
        """
        u = self.u
        uu = np.hstack([u[::-1],u])
        return uu

    def evolution(self, solution, nb_steps=1000, periods = 1):
        """
        The main body of the integrator's code. Takes full wave profile as input. Returns the result of integration
        of the given equation with input as the initial value. Integrates over the given number of time periods
        with given time-step dt.
        """
        u = solution
        NN = len(u)
        scale = self.equation.length/np.pi

        # todo: use fft shift and freq instead
        centered_frequencies = 1/scale * np.concatenate((np.arange(1, NN/2+1, 1), np.arange(1-NN/2, 0, 1)))
        kerne = 1j * centered_frequencies * self.equation.compute_kernel(centered_frequencies)
        kernel = np.concatenate(([0], kerne))
        kernel[NN/2] = 0

        shifted_frequencies = 1/scale * np.concatenate((np.arange(0, NN/2, 1), [0], np.arange(1-NN/2, 0, 1)))

        T = 2 * self.equation.length / self.velocity
        dt = periods * T / nb_steps

        m = 0.5 * dt * kernel
        d1 = (1-m)/(1+m)
        d1[NN/2] = 0
        d2 = -0.5 * 1j * dt * shifted_frequencies/(1+m)

        eps = 1e-10

        for i in range(nb_steps):
            fftu = fft(u)
            fftuu = fft(self.equation.flux(u))

            z1 = d1*fftu + d2*fftuu
            v = np.real(ifft(z1))

            z2 = d1*fftu + 2*d2*fftuu
            w = np.real(ifft(z2))

            def fixedpoint(w):
                z = ifft(d2*fft(self.equation.flux(w)))
                w_new = v + z.real
                return w_new

            maxit = 10000

            for it in xrange(maxit):
                w_new = fixedpoint(w)
                diff = w_new - w
                err = abs(np.max(diff))
                if err < eps:
                    break
                w = w_new
            else:
                raise Exception("Fixed point did not converge in %d iterations" % maxit)

            u = w
        return u
