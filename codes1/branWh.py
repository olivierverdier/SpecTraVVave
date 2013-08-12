import math
import numpy as np
from travWh import travWh
def branWh():
    L = math.pi
    N = 2*2*16
    h = float(L)/N

    x = np.arange(h/2.,L,h)
    cstar = math.sqrt(math.tanh(1))
    c = 0.99*cstar
    vini = -0.015-0.1*np.cos(x)
    tolerance =  1e-12
    
    u = travWh(c,vini,N,tolerance)

    for i in range(1,40):
        c = c - 0.0025
        u = travWh(c,u,N,tolerance)
    return u
    
def main():
    wave = branWh()
    print wave
    
if __name__ == '__main__':
  main()