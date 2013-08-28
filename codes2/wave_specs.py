import math
import numpy as np

class wave(object):
    def __init__(self, period, func_g):
        self.period = period
        self.g = func_g
        