from __future__ import division

import numpy as np
import scipy.fftpack 

class init_data(object):
    def __init__(self, Equation):
        self.eq = Equation

    def compute_init_guess(self, e = 0.1):
        dct = lambda x: scipy.fftpack.dct(x, norm='ortho')                 # short declaration of the dct and idct operators
        idct = lambda x: scipy.fftpack.idct(x, norm='ortho')

        F         =     3/4                                                # f^(p)(0)/p! 
        p         =     2                         #self.eq.degree
        lin_op    =     self.eq.kernel                                     # Fourier multiplier function of the linear operator
        c0        =     lin_op[1]                                          # bifurcation point in velocity
        Oper_A    =     np.diag(-c0 + lin_op)                              # diagonal matrix of the operator acting on xi_p on the LHS
        Oper_A[1,1]    = 1                                                 # the second diagonal element of the matrix is set to 1 in order to invert the matrix
                                                                           # later this element will be multiplied by zero, and the effect will be balanced
        xi1 = np.cos(self.eq.nodes)                                        # xi_1 which is always the cosine of x
        norm = dct(xi1)[1]                                           # this norm is used to compute pure dct coefficients without weights of transformation

        if p == 2:                                            # p = 2 is special case becasue xi2 != cos(x)
            rhs_hat   =     -F*dct(xi1**p)                    # take the dct of the right-hand-side of the equation
            rhs_hat[1] = 0                                    # actually it holds true, but this stands here as compensation b/w c1 and rhs_hat(1) to cancel the cos(x) terms  
            xi2_hat = np.linalg.solve(Oper_A, rhs_hat)        # solving the equation in the Fourier domain
            xi2 = idct(xi2_hat)                               # finding  the solution of the equation , the correction to velocity here is zero -> c1 = 0.
            rhs_hat = -F*dct(p*xi1**(p-1)*xi2)                # computing next epsilon level of approximation to compute the c2
            c2 = -rhs_hat[1]/norm                             # computing the corretion to the velocity.
            init_guess = e*xi1 + e**p*xi2                     # declaring the initial guess for wave shape
            init_velocity = c0 + e**p*c2                      # declaring the initial guess for the velocity

        if p > 2 and p%2 == 0:                               # p==even and > 2 CANNOT BE HANDLED AT THE MOMENT! 
            rhs_hat   =     -F*dct(xi1**p)
            rhs_hat[1] = 0
            xip_hat = np.linalg.solve(Oper_A, rhs_hat)
            xip = idct(xip_hat)
            rhs_hat = -F*dct(p*xi1**p)                      # THIS HAS TO BE REVISED!
            cp = -rhs_hat[1]/norm
            init_guess = e*xi1 + e**p*xip
            init_velocity = c0 + e**p*cp
            print "This case cannot be handled a the moment."
            
        if p%2 == 1:                                         # in case p==odd there are fewer steps to compute the initial shape and velocity
            rhs_hat   =     -F*dct(xi1**p)
            cp_1 = -rhs_hat[1]/norm
            rhs_hat[1] = 0
            xip_hat = np.linalg.solve(Oper_A, rhs_hat)
            xip = idct(xip_hat)
            init_guess = e*xi1 + e**p*xip
            init_velocity = c0 + e**(p-1)*cp_1
            
        return (init_guess, init_velocity)