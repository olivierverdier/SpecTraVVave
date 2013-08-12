import math
import numpy as np
from scipy import linalg

L = math.pi
N = 2*2*16
h = float(L)/N

x = np.arange(h/2.,L,h)
cstar = math.sqrt(math.tanh(1))
c = 0.99*cstar
vini = -0.015-0.1*np.cos(x)
tolerance =  1e-12
  
xi = range(0,N,1)
    
ww = math.sqrt(2/float(N)) * np.ones((N,1))
ww[0,0] = math.sqrt(1/float(N))

Tau = np.zeros((N,N))
for m in range(N):
    for n in range(N):
        Tau[m,n] = ww[0]*ww[0]*math.cos(x[n]*xi[0])*math.cos(x[m]*xi[0])
        for k in range(1,N):
            Tau[m,n] = Tau[m,n] + math.sqrt(1./xi[k]*math.tanh(xi[k]))*ww[k]*ww[k]*math.cos(x[n]*xi[k])*math.cos(x[m]*xi[k])
                
u=vini
for i in range(1,40):
    c += -0.0025
    change = 2
    it = 0
    ScriptL = -c*np.eye(N) + Tau
    while change > tolerance:
        if it > 10000:
            break
        DFu = ScriptL + np.diag(3*(u+1)**(0.5)-3)
        corr = np.dot(-linalg.inv(DFu),(np.dot(ScriptL,u) + 2*(u+1)**(1.5)-3*u-2))
        unew = u + corr
        change = corr.max()
        u = unew
        it = it + 1

print u