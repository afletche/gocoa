import numpy as np

#hello there

class Simulation:

    def __init__(self):
        self.nt = 3

        self.g = 9.81

    
    def preallocate_variables(self):
        self.U = np.zeros((2*self.nt,))
        self.x = np.zeros((self.nt+1))
        self.xdot = np.zeros((self.nt+1))
        self.y = np.zeros((self.nt+1))
        self.ydot = np.zeros((self.nt+1))
        self.theta = np.zeros((self.nt+1))
        self.thetadot = np.zeros((self.nt+1))


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

