import math
import numpy as np
from fractions import Fraction


N=64
L = math.pi
h = L/N


cstar = math.sqrt(math.tanh(1))
c = 0.95*cstar
x = np.arange(h/2,L,h)
xi = range(0,N,1)
ww = math.sqrt(Fraction(1, N)) * np.ones((N,1))
ww[0,0] = math.sqrt(Fraction(1, N))
uini = 0.015 - 0.1*np.cos(x)

Tau = np.zeros((N,N))
for m in range(N):
    for n in range(N):
        Tau[m,n] = ww[0]*ww[0]*math.cos(x[n]*xi[0])*math.cos(x[m]*xi[0])
        for k in range(1,N):
            Tau[m,n] = Tau[m,n]+math.sqrt(Fraction(1,xi[k])*math.tanh(xi[k]))*ww[k]*ww[k]*math.cos(x[n]*xi[k])*math.cos(x[m]*xi[k])
            
ScriptL = -c*np.eye(N) + Tau

u=uini  
change = 2
it = 0
for kk in range(50): 
    DFu = ScriptL + np.diag(3*(u+1)**(0.5)-3)
    corr = -np.linalg.solve(DFu, np.dot(ScriptL,u) + 2*(u+1)**(1.5)-3*u-2)
    unew = u + corr
    change = corr.max()
    u = unew
    
