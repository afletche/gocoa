import numpy as np

#hello there

class Simulation:

    def __init__(self):
        self.nt = 3

        self.g = 9.81
        self.tf = 30
        self.deltat = self.tf/self.nt
        self.mass = 40
        self.inertia = 10

    
    def preallocate_variables(self):
        self.U = np.zeros((2*self.nt,))
        self.x = np.zeros((self.nt+1))
        self.xdot = np.zeros((self.nt+1))
        self.xdotdot = np.zeros((self.nt+1))
        self.y = np.zeros((self.nt+1))
        self.ydot = np.zeros((self.nt+1))
        self.theta = np.zeros((self.nt+1))
        self.thetadot = np.zeros((self.nt+1))
        self.thetadotdot = np.zeros((self.nt+1))


    '''
    Evaluate the model

    returns: [f, c, df_dx, dc_dx, ...etc. I will fill this out later]
    '''
    def evaluate(self, x):
        pass

    def simulate_one_dynamics(self, u, tindex):
        self.x[tindex + 1] = self.deltat*self.xdot[tindex]+self.x[tindex]
        self.xdot[tindex + 1] = self.deltat*self.xdotdot[tindex]+self.xdot[tindex]
        self.xdotdot[tindex + 1] = (u[2*tindex]+u[2*tindex+1])*np.cos(self.theta[tindex])/self.mass
        self.theta[tindex + 1] = self.deltat*self.thetadot[tindex]+self.theta[tindex]
        self.thetadot[tindex + 1] = self.deltat*self.thetadotdot[tindex]+self.thetadot[tindex]
        self.thetadotdot[tindex + 1] = (-u[2*tindex]+u[2*tindex+1])/self.inertia

    def simulate_dynamics(self, u):
        for tind in range(self.nt):
            self.simulate_one_dynamics(u,tind)



    def evaluate_objective(self, u):
        pass



    def evaluate_jacobian(self, u):
        pass



    def evaluate_hessian(self, u):
        pass

