#!/usr/bin/env python
# coding: utf-8
from __future__ import division

import numpy as np

class Equation(object):
    def __init__(self, length=np.pi):
        self.length = length

    def compute_kernel(self, k):
        raise NotImplementedError()

    def flux(self, u):
        raise NotImplementedError()
