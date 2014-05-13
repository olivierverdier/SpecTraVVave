from __future__ import division

import numpy as np

    
def compute_line(p1, p2, d = 1):
    """
    Provide ortho, and pstar
    such that the line on the bifurcation plane has equation
        ortho*(p - p*) = 0

    Returns: (pstar, ortho)
    """
    
    ortho = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    if ortho[0] == 0 and ortho[1]==0:
        ortho[1] = 1
    
    pstar = np.array([p2[0] + ortho[0]*d, p2[1] + ortho[1]*d])
    return pstar, ortho
