import numpy as np


class Simulation:

    def __init__(self):
        self.nt = 3

    
    def preallocate_variables(self):
        self.U = np.zeros((2*self.nt,))
        


    '''
    Evaluate the model

    returns: [f, c, df_dx, dc_dx, ...etc. I will fill this out later]
    '''
    def evaluate(self, x):
        pass



    def simulate_dynamics(self, u):
        pass


    def evaluate_objective(self, u):
        pass



    def evaluate_jacobian(self, u):
        pass



    def evaluate_hessian(self, u):
        pass

