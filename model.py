import numpy as np

#hello there

class Simulation:

    def __init__(self):
        self.nt = 3

        self.x_target = 10.
        self.xdot_target = 0.
        self.y_target = 10.
        self.ydot_target = 0.
        self.theta_target = 0.
        self.thetadot_target = 0.

        self.g = 9.81

    
    def preallocate_variables(self):
        self.u = np.zeros((2*self.nt,))
        self.x = np.zeros((self.nt+1))
        self.xdot = np.zeros((self.nt+1))
        self.y = np.zeros((self.nt+1))
        self.ydot = np.zeros((self.nt+1))
        self.theta = np.zeros((self.nt+1))
        self.thetadot = np.zeros((self.nt+1))

        self.c = np.zeros((6,)) # 6 constrained final states


    '''
    Evaluate the model

    Inputs:
        - x : np.ndarray : vector of design variables (both control inputs and lagrange multipliers)

    Outputs:
        - model_outputs : List : [f, c, df_dx, dc_dx, ...etc. I will fill this out later]
    '''
    def evaluate(self, x):
        self.u = x[:(2*self.nt)]
        self.lagrange_multiliers = x[(2*self.nt):]
        # TODO



    def simulate_dynamics(self, u):
        pass



    def evaluate_objective(self, u):
        return u.dot(u)


    def evaluate_constraints(self, u):
        W = np.zeros((self.nt+1))

        self.c[0] = W.dot(self.x) - self.x_target
        self.c[1] = W.dot(self.xdot) - self.xdot_target
        self.c[2] = W.dot(self.y) - self.y_target
        self.c[3] = W.dot(self.ydot) - self.ydot_target
        self.c[4] = W.dot(self.theta) - self.theta_target
        self.c[5] = W.dot(self.thetadot) - self.thetadot_target

        return self.c



    def evaluate_gradient(self, u):
        objective_gradient = self.evaluate_objective_gradient(u)
        constraint_jacobian = self.evaluate_constraint_jacobian(u)
        constraints = self.evaluate_constraints(u)



    def evaluate_objective_gradient(self, u):
        return u


    def evaluate_constraint_jacobian(self, u):
        pass


    '''
    KKT Matrix
    '''
    def evaluate_hessian(self, u):
        pass


    '''
    d^f/dx^2
    '''
    def evaluate_objective_hessian(self, u):
        pass


    '''
    3rd order tensor of constraint second derivatives.
    '''
    def evaluate_constraint_hessian(self, u):
        pass