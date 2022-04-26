import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative

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
        self.tf = 30
        self.deltat = self.tf/self.nt
        self.mass = 40
        self.inertia = 10

        self.preallocate_variables()

    '''
    Preallocates memory for vectors across time.
    '''
    def preallocate_variables(self):
        self.u = np.zeros((2*self.nt,))
        self.x = np.zeros((self.nt+1))
        self.xdot = np.zeros((self.nt+1))
        self.xdotdot = np.zeros((self.nt+1))
        self.y = np.zeros((self.nt+1))
        self.ydot = np.zeros((self.nt+1))
        self.theta = np.zeros((self.nt+1))
        self.thetadot = np.zeros((self.nt+1))
        self.thetadotdot = np.zeros((self.nt+1))

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



    '''
    Evaluates the lagrangian objective (f + lambda*c) which is equivalent to the original objective (f) because c=0
    '''
    def evaluate_objective(self, u):
        return u.dot(u)


    '''
    Evaluates constraint functions to get the constraint vector.
    '''
    def evaluate_constraints(self, u):
        W = np.zeros((self.nt+1))
        W[-1] = 1

        self.c[0] = W.dot(self.x) - self.x_target
        self.c[1] = W.dot(self.xdot) - self.xdot_target
        self.c[2] = W.dot(self.y) - self.y_target
        self.c[3] = W.dot(self.ydot) - self.ydot_target
        self.c[4] = W.dot(self.theta) - self.theta_target
        self.c[5] = W.dot(self.thetadot) - self.thetadot_target

        return self.c


    '''
    Lagrangian gradient. [df_dx + lambda*dc_dx, c].T
    '''
    def evaluate_gradient(self, u, lagrangian_multipliers):
        objective_gradient = self.evaluate_objective_gradient(u)
        constraint_jacobian = self.evaluate_constraint_jacobian(u)
        constraints = self.evaluate_constraints(u)


    '''
    df_dx
    '''
    def evaluate_objective_gradient(self, u):
        return u


    '''
    dc_dx
    '''
    def evaluate_constraint_jacobian(self, u):
        W = np.zeros((self.nt+1))
        W[-1] = 1
        pc_px = W

        px_pxdot = np.tril(np.ones((self.nt, self.nt)), -1)*self.deltat

        pacceleration_pinput = np.zeros((self.nt, 2*self.nt))
        for i in range(self.nt):
            pacceleration_pinput[i, 2*i] = 1
            pacceleration_pinput[i, 2*i+1] = 1
        pacceleration_pinput_translational = pacceleration_pinput/self.mass
        pacceleration_pinput_rotational = pacceleration_pinput/self.inertia
        
        pxdotdot_ptheta = np.sin(self.theta)

        dc_du = pc_px + px_pxdot.dot(px_pxdot).dot(pacceleration_pinput_translational + np.dot(pxdotdot_ptheta).dot(px_pxdot).dot(px_pxdot).dot(pacceleration_pinput_rotational))
        print(dc_du)



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





    def plot_rigid_body_displacement(self, x_axis='x', y_axis='y', show=True):
        t_data = self.t_eval
        x_data = self.rigid_body_displacement[:,0]
        y_data = self.rigid_body_displacement[:,1]
        rot_z_data = self.rigid_body_displacement[:,2]
        plot_points = np.zeros((self.nt+1, 2))  # 2D plot

        if x_axis == 't':
            x_coords = t_data
        elif x_axis == 'x':
            x_coords = x_data
        elif x_axis == 'y':
            x_coords = y_data
        elif x_axis == 'rot_z':
            x_coords = rot_z_data

        if y_axis == 't':
            y_coords = t_data
        elif y_axis == 'x':
            y_coords = x_data
        elif y_axis == 'y':
            y_coords = y_data
        elif y_axis == 'rot_z':
            y_coords = rot_z_data

        # plt.plot(plot_points[:,0], plot_points[:,1], '-bo')
        plt.plot(x_coords, y_coords, '-bo')
        plt.title(f'Rigid Body Dynamics: {y_axis} vs. {x_axis}')
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        if show:
            plt.show()


    '''
    Generates a video.
    '''
    def generate_video(self, video_file_name, video_fps):
        print('Creating Video...')
        image_folder = 'plots'
        images = [f'video_plot_at_t_{t:9.9f}.png' for t in self.t_eval]
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_file_name, cv2.VideoWriter_fourcc(*'XVID'), video_fps, (width,height))

        for image in images:
            image_frame = cv2.imread(os.path.join(image_folder, image))
            frame_resized = cv2.resize(image_frame, (width, height)) 
            video.write(frame_resized)

        cv2.destroyAllWindows()
        video.release()