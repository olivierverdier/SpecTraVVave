from __future__ import division
import math
import numpy as np
import solver
import init_data
import dynamic_code
import equation as eq
from matplotlib.pyplot import plot, show

             ###### testing the solver.py functions

# basic parameters' values to initiate computation
N = 64          #number of nodes 
L = math.pi     #half of the wave length 
eps = 0.1
#Equation = Whitham
#speed = eq.Whitham.speed()    #phase speed
#u0 = eq.Whitham(N, L, speed).init_guess()    #-0.015-0.1*np.cos(x_nodes) # initial guess
data = init_data.init_data(eq.KDV(N, L, 1))
(u0, speed) = data.compute_init_guess()

Eq = eq.KDV(N, L, speed)
x = Eq.nodes             
deg = Eq.degree
            
            # solutions with given parameters

#solver_kind = 'compute_fsolve'
solver_kind = 'compute_newton'

us = []
bifur = []
print solver_kind

# solution via Netwon method
u = u0
c = speed
a1 = u[0]-u[-1]; c1 = c
bifur.append(np.array([c1,a1]))
a2 = a1; c2 = c1;
wave = solver.solver(eq.KDV(N, L, c), u)
(u, c, a) = wave.compute_newton(c1,a1,c2,a2) # getattr(wave, solver_kind)(c1,a1,c2,a2)
a2 = a; c2 = c
bifur.append(np.array([c2,a2]))
#for i in range(1):
    #print i
    #c += (-1)**(deg + 1)*0.0025
    #if c < 0:
     #   break
#    a1 = a2
#    c1 = c2
#    a2 = u[0]-u[-1]
#    c2 = c
#    wave = solver.solver(eq.KDV1(N, L, c), u)
#    (u, c, a) = getattr(wave, solver_kind)(c1,a1,c2,a2)
#    a2 = a; c2 = c
#    us.append(u);   bifur.append(np.array([c2,a2]))
#else:
#    print 'not stopped at zero'

#print solver_kind, ' \n', u
#plot(x, u0)
#aus = np.array(us)
#for i in range(10):
#    plot(x, aus[i*2,:])
    
show()

#xx0 = np.arange(-L, L, L/N)
#xx1 = np.arange(0, 2*L, L/N)
#dyn = dynamic_code.dynamic_code(eq.KDV1(N, L, c), u)
#uu = dyn.interpolation(symm = 0)
#t_wave = dyn.evolution(solution = uu)