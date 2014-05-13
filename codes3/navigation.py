from __future__ import division

import numpy as np

    
def compute_line(p1, p2, step):
    """
    Return pstar such that
        pstar = p2+(1+d)*(p2-p1)
    """
    dp = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    # orthogonal direction
    ortho = (-dp[1], dp[0])
    
    pstar = np.array([p2[0] + dp[0]*step, p2[1] + dp[1]*step])
    return pstar, ortho


