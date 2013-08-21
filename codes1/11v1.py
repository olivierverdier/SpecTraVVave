# This version of 11.py uses routns.py to compute 
# the nonlinearity and linear operator for dispersion.

import math
import numpy as np
import routns                        # file contains function computing nonlinearity and disperion 
import time

start_time = time.time()
L = math.pi                            # half of the period
N = 2*2*16                             # number of grid points on [0,L]
h = float(L)/N                         # spacial step size

x = np.arange(h/2.,L,h)                # grid points (nodes) on [0,L]
cstar = math.sqrt(math.tanh(1))        # phase speed bifurcation point of the wave branch
c = 0.99*cstar                         
vini = -0.015-0.1*np.cos(x)            # initial guess for the wave shape u(x,t=0)
tolerance =  1e-12
  
xi = range(0,N,1)                      # 
    
Tau = routns.lin_op(L,N)               # dispersion matrix
                
u=vini
for i in range(1,40):                  # a for-loop which decreases the wave speed
    c += -0.0025                       # so as to travel along the bifurcation branch
    change = 2
    it = 0
    ScriptL = -c*np.eye(N) + Tau
    while change > tolerance:          # Newton iterative solver for the obtained system of equations
        if it > 10000:                 # stopping criteria: reaching tolerance level or 10000 iterations
            break
        DFu = ScriptL + np.diag(routns.D(u))
        corr = np.dot(-np.linalg.inv(DFu),(np.dot(ScriptL,u) + routns.flux(u)))
        unew = u + corr
        change = corr.max()
        u = unew
        it = it + 1

elapsed_time = time.time() - start_time             # time counter 
print u                                            # print out the solution
print 'Elapsed time: ', elapsed_time, 'seconds.'   # print out elapsed time 

uu = routns.interp(u, L, N)

print uu