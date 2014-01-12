# -*- coding: UTF-8 -*-
from __future__ import division

import math
import numpy as np

class Equation(object):
    def __init__(self,  size, length, velocity):
        self.velocity = velocity
        self.size = size
        self.length = length

        self.weights = self.compute_weights()
        self.nodes = self.compute_nodes()
        self.linear_operator = self.compute_linear_operator()
        self.matrix = self.compute_matrix()
        #self.kernel = self.compute_kernel()
        self.degree = self.degree()

    def compute_matrix(self):
        return -self.velocity*np.eye(self.size) + self.linear_operator

    @classmethod
    def general_tensor(self, w, x, f):
        tensor = np.dstack([wk*(f(k*x) * f(k*x).reshape(-1,1)) for k, wk in enumerate(w)])
        return tensor

    @classmethod
    def general_linear_operator(self, w, x, f):
        ten = self.general_tensor(w, x, f)
        return np.sum(ten, axis=2)
        
    def Jacobian(self, u):
        return self.matrix + np.diag(self.flux_prime(u))

    def residual(self, u): 
        return np.dot(self.matrix, u) + self.flux(u)
    
    def kernel(self, k):
        return self.compute_kernel(k)

class Whitham(Equation):
    def degree(self):
        return 2    
    
    def compute_kernel(self,k):
        if k[0] == 0:
            k1 = k[1:]
            whitham = np.concatenate( ([1], np.sqrt(1./k1*np.tanh(k1))))
        else:    
            whitham  = np.sqrt(1./k*np.tanh(k))
        
        return whitham
        
    def compute_weights(self):
        ks = np.arange(self.size, dtype=float)
        ww = 2/self.size
        weights = self.compute_kernel(ks)*ww
        weights[0] = 1/self.size
        return weights
        
    def compute_matrix(self):                                           #!!!!!!!!!! 
        return -self.velocity*np.eye(self.size) + self.linear_operator #!!!!!!!!!!!

    def compute_nodes(self):
        h = self.length/self.size
        nodes = np.arange(h/2.,self.length,h)
        return nodes

    def compute_linear_operator(self):
        return self.general_linear_operator(w=self.weights, x=self.nodes, f=np.cos)

    def init_guess(self):
        #The function computes the initial value (u(x,t=0))
        L = self.length
        N = self.size
        h = float(L)/N
        guess = -0.015-0.1*np.cos(np.arange(h/2.,L,h))
        return guess

    def flux(self, u):
        # This function returnes the f(u).
        return 2*(u+1)**(1.5)-3*u-2

    def flux_prime(self, u):
        # This function returnes the df(u)/du.
        return 3*(u+1)**(0.5)-3

    def lin_op(self):
        # The function takes the half of the period of wave and the grid number. 
        # Then it returns the matrix Tau which is used in the dispersion term.
        L = self.length
        N = self.size
        h = float(L)/N
        x = np.arange(h/2.,L,h)
               
        Tau = np.zeros((N,N))
        for m in range(N):
            for n in range(N):
                Tau[m,n] = 1/N
                for k in range(1,N):
                    Tau[m,n] = Tau[m,n] + math.sqrt(1./k*math.tanh(k))*2/N*math.cos(x[n]*k)*math.cos(x[m]*k)
        return Tau
    
    @classmethod
    def speed(self):
        return 0.99*math.sqrt(math.tanh(1))
            
##############################################################################################################################
class KDV(Equation):
    # The equation is :     -c*u + 3/4u^2 + (u + 1/6u")=0
    def degree(self):
        return 2

    def compute_kernel(self, k):
        return 1.0-1.0/6*k**2
            
    def compute_weights(self):
        ks = np.arange(self.size, dtype=float)
        ww = 2/self.size
        weights = self.compute_kernel(ks)*ww  
        weights[0] = weights[0]/2  
        return weights
        
   # def compute_matrix(self):                                          
   #     return -self.velocity*np.eye(self.size) + self.linear_operator 
        
    def compute_nodes(self):
        h = self.length/self.size
        nodes = np.arange(h/2.,self.length,h)
        return nodes

    def compute_linear_operator(self):
        return self.general_linear_operator(w=self.weights, x=self.nodes, f=np.cos)

    def init_guess(self, e = 0.1):
        #The function computes the initial value (u(x,t=0))
        #c = self.velocity
        guess = e*np.cos(self.nodes)+e*e*(3.0/4*np.cos(2*self.nodes)-9.0/4) #(-3/4)*(1.0/(1.0-c-0.66666)*np.cos(2*self.nodes) + 1.0/(1-c))  # k is not kappa, kappa = 1.0
        return guess

    def flux(self, u):
        # This function returnes the f(u).
        return 0.75*u**2  

    def flux_prime(self, u):
        # This function returnes the df(u)/du.
        return 1.5*u

    def lin_op(self):
        # The function takes the half of the period of wave and the grid number. 
        # Then it returns the matrix Tau which is used in the dispersion term.
        L = self.length
        N = self.size
        h = float(L)/N
        x = np.arange(h/2.,L,h)
               
        Tau = np.zeros((N,N))
        for m in range(N):
            for n in range(N):
                Tau[m,n] = 0
                for k in range(1,N):
                    Tau[m,n] = Tau[m,n] + (-k**2)*2/N*math.cos(x[n]*k)*math.cos(x[m]*k)
        return Tau
    
    @classmethod
    def speed(self, e=0.1):
        cstar = 1.0-1.0/6 #bifurcation point
        c2 = -45./16 #2*(-0.375)*1.0/(1-cstar)      #second order term
                #u1 = np.cos(self.nodes)
                #u2 = np.linalg.solve(self.matrix, -self.flux(u1))
                #c2 = 2*1/np.sqrt(self.size)*np.sum(u2)
        return cstar + e*e*c2
        #return (1-e)*cstar

##############################################################################################################################
class KDV1(Equation):
    # The equation is :     -c*u + 3/4u^3 + (u + 1/6u")=0
    def degree(self):
        return 3
    
    def compute_kernel(self, k):
        return 1.0-1.0/6*k**2
            
    def compute_weights(self):
        ks = np.arange(self.size, dtype=float)
        ww = 2/self.size
        weights = self.compute_kernel(ks)*ww  
        weights[0] = weights[0]/2  
        return weights
        
   # def compute_matrix(self):                                          
   #     return -self.velocity*np.eye(self.size) + self.linear_operator 
        
    def compute_nodes(self):
        h = self.length/self.size
        nodes = np.arange(h/2.,self.length,h)
        return nodes

    def compute_linear_operator(self):
        return self.general_linear_operator(w=self.weights, x=self.nodes, f=np.cos)

    def init_guess(self, e = 0.1):
        #The function computes the initial value (u(x,t=0))
        #c = self.velocity
        guess = e*(1+e)*np.cos(self.nodes)+e**3*(9.0/64*np.cos(3*self.nodes)) #(-3/4)*(1.0/(1.0-c-0.66666)*np.cos(2*self.nodes) + 1.0/(1-c))  # k is not kappa, kappa = 1.0
        return guess

    def flux(self, u):
        # This function returnes the f(u).
        return 3/4*u**3  

    def flux_prime(self, u):
        # This function returnes the df(u)/du.
        return 9/4*u**2

    def lin_op(self):
        # The function takes the half of the period of wave and the grid number. 
        # Then it returns the matrix Tau which is used in the dispersion term.
        L = self.length
        N = self.size
        h = float(L)/N
        x = np.arange(h/2.,L,h)
               
        Tau = np.zeros((N,N))
        for m in range(N):
            for n in range(N):
                Tau[m,n] = 0
                for k in range(1,N):
                    Tau[m,n] = Tau[m,n] + (-k**2)*2/N*math.cos(x[n]*k)*math.cos(x[m]*k)
        return Tau
    
    @classmethod
    def speed(self, e=0.1):
        cstar = 1.0-1.0/6 #bifurcation point
        c2 = 9./16 #2*(-0.375)*1.0/(1-cstar)      #second order term
                #u1 = np.cos(self.nodes)
                #u2 = np.linalg.solve(self.matrix, -self.flux(u1))
                #c2 = 2*1/np.sqrt(self.size)*np.sum(u2)
        return cstar + e*e*c2
        #return (1-e)*cstar

#####################################################################################

class WAVE_EQ(Equation):
    # The equation is :     -c*u + 3/4u^3 + (u + 1/6u")=0

    def compute_kernel(self):
        ks = np.arange(self.size, dtype=float)
        return 1.0-1.0/6*ks**2

    def compute_weights(self):
        ks = np.arange(self.size, dtype=float)
        ww = 2/self.size
        weights = (1.0-1.0/6*ks**2)*ww  
        weights[0] = (1.0-1.0/6*ks[0]**2)*ww/2  
        return weights
        
   # def compute_matrix(self):                                          
   #     return -self.velocity*np.eye(self.size) + self.linear_operator 
        
    def compute_nodes(self):
        h = self.length/self.size
        nodes = np.arange(h/2.,self.length,h)
        return nodes

    def compute_linear_operator(self):
        return self.general_linear_operator(w=self.weights, x=self.nodes, f=np.cos)

    def init_guess(self, e = 0.1):
        #The function computes the initial value (u(x,t=0))
        #c = self.velocity
        guess = e*(1+e)*np.cos(self.nodes)+e**3*(9.0/64*np.cos(3*self.nodes)) #(-3/4)*(1.0/(1.0-c-0.66666)*np.cos(2*self.nodes) + 1.0/(1-c))  # k is not kappa, kappa = 1.0
        return guess

    def flux(self, u):
        # This function returnes the f(u).
        return 3/4*u**3  

    def flux_prime(self, u):
        # This function returnes the df(u)/du.
        return 9/4*u**2

    def lin_op(self):
        # The function takes the half of the period of wave and the grid number. 
        # Then it returns the matrix Tau which is used in the dispersion term.
        L = self.length
        N = self.size
        h = float(L)/N
        x = np.arange(h/2.,L,h)
               
        Tau = np.zeros((N,N))
        for m in range(N):
            for n in range(N):
                Tau[m,n] = 0
                for k in range(1,N):
                    Tau[m,n] = Tau[m,n] + (-k**2)*2/N*math.cos(x[n]*k)*math.cos(x[m]*k)
        return Tau
    
    @classmethod
    def speed(self, e=0.1):
        cstar = 1.0-1.0/6 #bifurcation point
        c2 = 9./16 #2*(-0.375)*1.0/(1-cstar)      #second order term
                #u1 = np.cos(self.nodes)
                #u2 = np.linalg.solve(self.matrix, -self.flux(u1))
                #c2 = 2*1/np.sqrt(self.size)*np.sum(u2)
        return cstar + e*e*c2
        #return (1-e)*cstar