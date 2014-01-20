from __future__ import division

import math
import numpy as np

"""
The navigation.compute_line(...) provides 3 coefficients aplha, beta and gamma
to specify the line on bifurcation plane (c,a) : alpha*a + beta*c + gamma = 0 
on which the solution is looked for. 
"""
class navigation(object):
    def __init__(self, c1, a1, c2, a2):
        self.c1 = c1
        self.c2 = c2
        self.a1 = a1
        self.a2 = a2
        
    def compute_line(self, d = 1):
        distance = d*math.sqrt((self.c2-self.c1)**2+(self.a2-self.a1)**2)       # distance b/w P1 and P2, then we want same distance b/w P2 and P3
        if distance == 0:
            distance = 0.0025
            
        if self.c1 == self.c2:        # check if c0 == c1  --> vertical line
            alpha = 1
            beta = 0
            c3 = self.c2
            if self.a1 > self.a2: 
                a3 = self.a2 - distance
            else: 
                a3 = self.a2 + distance
            gamma = -a3        
            return (c3, a3, alpha, beta, gamma)
            
        elif self.a1 == self.a2:     # check if a0 == a1 --> horizontal line
            alpha = 0
            beta = 1
            if self.c1 < self.c2:
                c3 = self.c2 + distance
            else:
                c3 = self.c2 - distance
                                                           # might be reasonable to have just d (e.g. d = 0.0025) so as to make a step      
            gamma = -c3                                    # forward without computing each solution in a small neighbourhood 
            a3 = self.a2
            return (c3, a3, alpha, beta, gamma)
        
        else:
            slope = (self.a2-self.a1)/(self.c2-self.c1)                             # slope of the P1-P2-line
            b1 = -slope*self.c1 + self.a1                                           # intercept of the P1-P2-line
            b2 = distance*math.sqrt(1 + (1/slope)**2) - 1*self.a2 - 1/slope*self.c2 # from the distance from point (P2) to a line formula (line of interest)
            A = np.matrix([[1, -slope],[1, 1/slope]])                                   
            b = [b1, b2]
            p3 = np.linalg.solve(A, b)                                              # now find P3
            a3 = p3[0] 
            c3 = p3[1]
            
            alpha = 1
            beta = 1/slope
            gamma = -b2
            
            dp1p2 = distance/d
            dp1p3 = math.sqrt((c3 - self.c1)**2 + (a3 - self.a1)**2)
            
            if dp1p2 < dp1p3:                                                       # check that P3 is not b/w P1 and P2
                return (c3, a3, alpha, beta, gamma)
            else:
                b2 = distance*math.sqrt(1 + (1/slope)**2) + 1*self.a2 + 1/slope*self.c2
                A = np.matrix([[1, -slope],[1, 1/slope]])
                b = [b1, b2]
                p3 = np.linalg.solve(A, b)
                a3 = p3[0] 
                c3 = p3[1]
            
                alpha = 1
                beta = 1/slope
                gamma = -b2
                return (c3, a3, alpha, beta, gamma)
        
            
            