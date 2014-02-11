from __future__ import division

import numpy as np

class Navigation(object):
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
    
    def compute_line(self, d = 1):
        """
        Provide ortho, and pstar
        such that the line on the bifurcation plane has equation
            ortho*(p - p*) = 0

        Returns: (pstar, ortho)
        """
        
        ortho = np.array([self.p2[0] - self.p1[0], self.p2[1] - self.p1[1]])
        if ortho[0] == 0 and ortho[1]==0:
            ortho[1] = 1
        
        pstar = np.array([self.p2[0] + ortho[0]*d, self.p2[1] + ortho[1]*d])
        return pstar, ortho