from __future__ import division

from travwave import navigation, solver, discretization 

class BifurcationDiagram(object):
    
    def __init__(self, equation, boundary_condition, size = 32, doublings = 1, correction_rate = 10):
        self.equation = equation
        self.boundary = boundary_condition
        self.doublings = doublings
        self.size = size
        self.correction_rate = correction_rate
                                
        self.discretization = discretization.Discretization(self.equation, size = 256)
        solve = solver.Solver(self.discretization, boundary = self.boundary)
        nav = navigation.Navigator(solve.solve, size = self.size, doublings = self.doublings, correction_rate = self.correction_rate)
        initial_guess = self.discretization.compute_initial_guess(.01)
        initial_velocity = self.discretization.bifurcation_velocity()
        step = 0.005
        p1 = (initial_velocity, 0)
        p0 = (initial_velocity, -step)
                        
        nav.initialize(initial_guess, p1, p0)
        self.navigation = nav
   
    def plot_diagram(self):
        from matplotlib.pyplot import plot, xlabel, ylabel
        import numpy as np
        nav = self.navigation
        parameters = [result['current'] for result in nav]
        aparameters = np.array(parameters)        
        sizes = [len(result['solution']) for result in nav]        
        plot(aparameters[:,0], aparameters[:,1], '.--')
        for i,s in enumerate(sizes):
            if s > 50:
                plot(aparameters[i,0], aparameters[i,1], 'o', color='green')
            if s > 100:
                plot(aparameters[i,0], aparameters[i,1], 'o', color='red')
        xlabel('Wavespeed')
        ylabel('Waveheight')
        
    def plot_solution(self, index = [-1]):
        from matplotlib.pyplot import plot, xlabel, ylabel
        import numpy as np
        counter = np.arange(np.size(index))        
        for i in counter:
            solution = self.navigation[index[i]]['solution']
            size = len(solution)
            nodes = self.equation.length * (np.linspace(0, 1, size, endpoint = False) + 1/2/size)            
            plot(nodes, solution)
        xlabel('x')
        ylabel('Surface Elevation')
    
    