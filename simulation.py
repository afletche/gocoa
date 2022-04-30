import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy.misc import derivative

#hello there

class Simulation:

    def __init__(self):
        self.nt = 50

        self.x_target = 2.
        self.xdot_target = 0.
        self.y_target = 2.
        self.ydot_target = 0.
        self.theta_target = np.pi / 2
        self.thetadot_target = 0.

        self.g = 9.81
        self.tf = 5
        self.deltat = self.tf/self.nt
        self.mass = 40
        self.inertia = 10
        self.r = 0.5
        self.preallocate_variables()


    '''
    Preallocates memory for vectors across time.
    '''
    def preallocate_variables(self):
        self.num_control_inputs = 2*self.nt
        self.num_constraints = 1    # 2D, each dof has a final condition constraint.

        self.u = np.zeros((2*self.nt,))
        self.x = np.zeros((self.nt+1))
        self.xdot = np.zeros((self.nt+1))
        self.xdotdot = np.zeros((self.nt+1))
        self.y = np.zeros((self.nt+1))
        self.ydot = np.zeros((self.nt+1))
        self.ydotdot = np.zeros((self.nt+1))
        self.theta = np.zeros((self.nt+1))
        self.thetadot = np.zeros((self.nt+1))
        self.thetadotdot = np.zeros((self.nt+1))
    def set_initial_cond(self):
        self.theta[0] = np.pi / 2


    '''
    Setup method that will precompute anything that can be precomputed

    Examples could be automatic moment of inertia calculation, etc.
    '''
    def setup(self):
        self.preallocate_variables()
        self.set_initial_cond()
        self.W = np.zeros((self.nt + 1))
        self.W[-1] = 1
    '''
    Evaluate the model

    Inputs:
        - x : np.ndarray : vector of design variables (both control inputs and lagrange multipliers)

    Outputs:
        - model_outputs : List : [f, c, df_dx, dc_dx, d2f_dx2, dl_dx, kkt]
    '''
    def evaluate(self, x, rho=0.):
        self.u = x[:self.num_control_inputs]
        self.lagrange_multipliers = x[self.num_control_inputs:]

        self.simulate_dynamics(self.u)

        f = self.evaluate_objective(self.u)
        c = self.evaluate_constraints()
        # df_dx = self.evaluate_objective_gradient(self.u)
        df_dx = self.evaluate_gradient(self.u, self.lagrange_multipliers)
        dc_dx = self.evaluate_constraint_jacobian()
        d2f_dx2 = self.evaluate_objective_hessian()
        dl_dx = self.evaluate_gradient(self.u, self.lagrange_multipliers)
        kkt = self.evaluate_hessian(self.lagrange_multipliers)

        #print("ending x position",self.x[-1]," c = ",c,"dc_dx=",dc_dx)
        print("ending x position",self.x[-1]," c = ",c)
        #dc_dx=None
        return [f, c, df_dx, dc_dx, d2f_dx2, dl_dx, kkt]


    

    '''
    Apply for explicit Euler update for one time step.
    '''
    def simulate_one_dynamics(self, u, tindex):
        self.x[tindex + 1] = self.deltat*self.xdot[tindex]+self.x[tindex]
        self.xdot[tindex + 1] = self.deltat*self.xdotdot[tindex]+self.xdot[tindex]
        self.xdotdot[tindex + 1] = (u[2*tindex]+u[2*tindex+1])*np.cos(self.theta[tindex])/self.mass
        self.y[tindex + 1] = self.deltat*self.ydot[tindex]+self.y[tindex]
        self.ydot[tindex + 1] = self.deltat*self.ydotdot[tindex]+self.ydot[tindex]-self.deltat*self.g
        self.ydotdot[tindex + 1] = (u[2*tindex]+u[2*tindex+1])*np.sin(self.theta[tindex])/self.mass
        #print("setp",tindex,"y",self.y[tindex + 1])
        #print("setp",tindex,"ydot",self.ydot[tindex + 1])
        #print("setp",tindex,"ydotdot",self.ydotdot[tindex + 1])

        self.theta[tindex + 1] = self.deltat*self.thetadot[tindex]+self.theta[tindex]
        self.thetadot[tindex + 1] = self.deltat*self.thetadotdot[tindex]+self.thetadot[tindex]
        self.thetadotdot[tindex + 1] = self.r*(-u[2*tindex]+u[2*tindex+1])/self.inertia
        #print("setp", tindex, "theeta", self.theta[tindex + 1])
        #print("setp", tindex, "theetadot", self.thetadot[tindex + 1])
        #print("setp", tindex, "theetadotdot", self.thetadotdot[tindex + 1])


    '''
    Integrate the dynamics.
    '''
    def simulate_dynamics(self, u):
        for tind in range(self.nt):
            self.simulate_one_dynamics(u,tind)



    '''
    Evaluates the lagrangian objective (f + lambda*c) which is equivalent to the original objective (f) because c=0
    '''
    def evaluate_objective(self, u):
        penalty = np.exp(-5*(self.theta+np.pi/6.0))+np.exp(-5*(7.0*np.pi/6.0-self.theta))
        return u.dot(u)+np.sum(penalty)
        #return self.evaluate_constraints()


    '''
    Evaluates constraint functions to get the constraint vector.
    '''
    def evaluate_constraints(self):


        c = np.zeros((self.num_constraints,))
        c[0] = self.W.dot(self.x) - self.x_target
        #c[1] = self.W.dot(self.xdot) - self.xdot_target
        #c[1] = self.W.dot(self.y) - self.y_target
        #c[3] = self.W.dot(self.ydot) - self.ydot_target
        #c[1] = self.W.dot( np.mod(self.theta,np.pi*2) ) - self.theta_target
        #c[5] = self.W.dot(self.thetadot) - self.thetadot_target

        return c


    '''
    Lagrangian gradient. [df_dx + lambda*dc_dx, c].T
    '''
    def evaluate_gradient(self, u, lagrangian_multipliers):
        objective_gradient = self.evaluate_objective_gradient(u)
        constraint_jacobian = self.evaluate_constraint_jacobian()
        constraints = self.evaluate_constraints()

        gradient = np.zeros((self.num_control_inputs + self.num_constraints))
        gradient[:self.num_control_inputs] = objective_gradient + lagrangian_multipliers.dot(constraint_jacobian)
        gradient[self.num_control_inputs:] = constraints
        return gradient

    '''
    df_dx
    '''
    def evaluate_objective_gradient(self, u):
        return u


    '''
    dc_dx
    '''
    def evaluate_constraint_jacobian(self):
        dc_du = np.zeros((self.num_constraints, 2*self.nt))

        W = np.zeros((self.nt))
        W[-1] = 1
        pc_px = W

        px_pxdot = np.tril(np.ones((self.nt, self.nt)),-1)*self.deltat

        pxdotdot_pinput = np.zeros((self.nt, 2*self.nt))
        pydotdot_pinput = np.zeros((self.nt, 2*self.nt))
        pthetadotdot_pinput = np.zeros((self.nt, 2*self.nt))
        for i in range(self.nt):
            cos_theta = np.cos(self.theta[i])
            pxdotdot_pinput[i, 2*i] = cos_theta
            pxdotdot_pinput[i, 2*i+1] = cos_theta
            sin_theta = np.sin(self.theta[i])
            pydotdot_pinput[i, 2*i] = sin_theta
            pydotdot_pinput[i, 2*i+1] = sin_theta

            pthetadotdot_pinput[i, 2*i] = -1.
            pthetadotdot_pinput[i, 2*i+1] = 1.

        pacceleration_pinput_translational = pxdotdot_pinput/self.mass
        pacceleration_pinput_rotational = pthetadotdot_pinput*self.r/self.inertia
        
        pxdotdot_ptheta = np.diag(-np.sin(self.theta[1:]))
        pydotdot_ptheta = np.diag(np.cos(self.theta[1:]))

        dc_du[0,:] = -pc_px.dot(px_pxdot).dot(px_pxdot).dot(pacceleration_pinput_translational + pxdotdot_ptheta.dot(px_pxdot).dot(px_pxdot).dot(pacceleration_pinput_rotational))
        #dc_du[1,:] = pc_px.dot(px_pxdot).dot(pacceleration_pinput_translational + pxdotdot_ptheta.dot(px_pxdot).dot(px_pxdot).dot(pacceleration_pinput_rotational))
        #dc_du[1,:] = pc_px.dot(px_pxdot).dot(px_pxdot).dot(pacceleration_pinput_translational + pydotdot_ptheta.dot(px_pxdot).dot(px_pxdot).dot(pacceleration_pinput_rotational))
        #dc_du[3,:] = pc_px.dot(px_pxdot).dot(pacceleration_pinput_translational + pydotdot_ptheta.dot(px_pxdot).dot(px_pxdot).dot(pacceleration_pinput_rotational))
        #dc_du[1,:] = W.dot(px_pxdot).dot(px_pxdot).dot(pacceleration_pinput_rotational)
        #dc_du[5,:] = W.dot(px_pxdot).dot(pacceleration_pinput_rotational)
        
        return dc_du


    '''
    KKT Matrix
    '''
    def evaluate_hessian(self, lagrangian_multipliers):
        return None


    '''
    d^f/dx^2
    '''
    def evaluate_objective_hessian(self):
        d2f_dx2 = 2*self.deltat*np.eye(self.num_control_inputs)
        return d2f_dx2



    '''
    3rd order tensor of constraint second derivatives.
    '''
    def evaluate_constraint_hessian(self, u):
        pass



    def savefigures(self, xmin,xmax,ymin,ymax,x_axis='x', y_axis='y', show=True):
        x_data = self.x
        y_data = self.y
        x_coords = x_data
        y_coords = y_data
        for intex in range(self.nt+1):
            plt.plot(x_coords[intex], y_coords[intex], '-bo')
            plt.title(f'Rigid Body Dynamics: {y_axis} vs. {x_axis}')
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)
            # plot right arm red. left arm green
            plt.plot(x_coords[intex] + 0.5 * np.cos(self.theta[intex] - np.pi / 2), y_coords[intex] + 0.5 * np.sin(self.theta[intex] - np.pi / 2),
                     'ro')
            plt.plot(x_coords[intex] + 0.5 * np.cos(self.theta[intex] + np.pi / 2), y_coords[intex] + 0.5 * np.sin(self.theta[intex] + np.pi / 2),
                     'go')

            # plot above blue
            plt.plot(x_coords[intex] + 0.5 * np.cos(self.theta[intex]), y_coords[intex] + 0.5 * np.sin(self.theta[intex]), 'b*')
            # plot target
            plt.plot(self.x_target, self.y_target, color=[1.0, 0.7, 0.0, 1.0], marker='*')
            plt.xlim([xmin,xmax])
            plt.ylim([ymin,ymax])

            plt.savefig(f'plots/video_plot_at_t_{intex:9.9f}.png', bbox_inches='tight')
            plt.close()

    def plot_rigid_body_displacement(self, x_axis='x', y_axis='y', show=True):
        t_data = np.linspace(0,self.tf,self.nt)
        #y_data = self.rigid_body_displacement[:,1]
        x_data = self.x
        y_data = self.y
        plot_points = np.zeros((self.nt+1, 2))  # 2D plot

        if x_axis == 't':
            x_coords = t_data
        elif x_axis == 'x':
            x_coords = x_data
        elif x_axis == 'y':
            x_coords = y_data
        elif x_axis == 'rot_z':
            rot_z_data = self.rigid_body_displacement[:, 2]
            x_coords = rot_z_data

        if y_axis == 't':
            y_coords = t_data
        elif y_axis == 'x':
            y_coords = x_data
        elif y_axis == 'y':
            y_coords = y_data
        elif y_axis == 'rot_z':
            rot_z_data = self.rigid_body_displacement[:, 2]
            y_coords = rot_z_data

        # plt.plot(plot_points[:,0], plot_points[:,1], '-bo')
        plt.plot(x_coords, y_coords, '-bo')

        #plot right arm red. left arm green
        plt.plot(x_coords+0.5*np.cos(self.theta-np.pi/2), y_coords+0.5*np.sin(self.theta-np.pi/2), 'ro')
        plt.plot(x_coords+0.5*np.cos(self.theta+np.pi/2), y_coords+0.5*np.sin(self.theta+np.pi/2), 'go')

        #plot above blue
        plt.plot(x_coords+0.5*np.cos(self.theta), y_coords+0.5*np.sin(self.theta), 'b*')

        #plot target
        plt.plot(self.x_target, self.y_target, color = [1.0,0.7,0.0,1.0],marker ='*')

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
        images = [f'video_plot_at_t_{t:9.9f}.png' for t in range(self.nt+1)]
        print("image names",images[0])
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_file_name, cv2.VideoWriter_fourcc(*'XVID'), video_fps, (width,height))

        for image in images:
            image_frame = cv2.imread(os.path.join(image_folder, image))
            frame_resized = cv2.resize(image_frame, (width, height)) 
            video.write(frame_resized)

        cv2.destroyAllWindows()
        video.release()


    def plot(self, stress_type=None, time_step=None, dof=None, show_dislpacements=False, show_nodes=False, show_connections=False, show_undeformed=False,
                save_plots=False, video_file_name=None, video_fps=1, show=True):
        nodes = self.mesh.nodes
        U_reshaped = self.U.reshape((-1, self.num_dimensions))
        max_x_dist = np.linalg.norm(max(nodes[:,0]) - min(nodes[:,0]))
        max_y_dist = np.linalg.norm(max(nodes[:,1]) - min(nodes[:,1]))
        scale_dist = np.linalg.norm(np.array([max_x_dist, max_y_dist]))
        if np.linalg.norm(self.U) != 0:
            visualization_scaling_factor = scale_dist*0.1/max(np.linalg.norm(U_reshaped, axis=1))
        else:
            visualization_scaling_factor = 0
        self.visualization_scaling_factor = visualization_scaling_factor

        self.element_midpoints = np.zeros((self.nt+1, self.num_elements, self.num_dimensions))
        self.element_midpoints_plot = np.zeros((self.nt+1, self.num_elements, self.num_dimensions))
        for i, element in enumerate(self.mesh.elements):
            element_dofs = self.nodes_to_dof_indices(element.node_map)
            element_U = self.U[np.ix_(element_dofs)]

            self.element_midpoints[:,i,:] = element.calc_midpoint(element_U)
            self.element_midpoints_plot[:,i,:] = element.calc_midpoint(element_U*self.visualization_scaling_factor)

        if time_step is None:
            time_step = range(len(self.t_eval))
        elif type(time_step) == int:
            time_step = [time_step]
        if dof is not None and type(dof) == int:
            dof = [dof]

        if stress_type is not None:
            self.evaluate_stress_points(visualization_scaling_factor)

            if stress_type == 'x' or stress_type == 'xx':
                stresses = self.stresses_dict['xx']
                stress_eval_points = self.stress_eval_points_dict['xx']
            elif stress_type == 'y' or stress_type == 'yy':
                stresses = self.stresses_dict['yy']
                stress_eval_points = self.stress_eval_points_dict['yy']
            elif stress_type == 'tao' or stress_type == 'xy':
                stresses = self.stresses_dict['xy']
                stress_eval_points = self.stress_eval_points_dict['xy']
            elif stress_type == 'von_mises' or stress_type == 'vm':
                stresses = self.von_mises_stresses
                stress_eval_points = self.stress_eval_points_dict['xx']
            elif stress_type == 'averaged_von_mises' or stress_type == 'avm':
                stresses = self.averaged_von_mises_stresses
                stress_eval_points = self.element_midpoints_plot
            elif stress_type == 'axial' or stress_type == 'tension' or stress_type == 'compression' or stress_type == '11':
                stresses = self.stresses_dict['axial']
                stress_eval_points = self.stress_eval_points_dict['axial']

        if dof is None:
            print('Plotting...')
            for t_step in time_step:
                t = self.t_eval[t_step]
                plt.figure()
                if show_dislpacements:
                    self.plot_displacements(show_nodes=show_nodes, show_connections=show_connections, show_undeformed=show_undeformed,
                                             time_step=t_step, visualization_scaling_factor=visualization_scaling_factor, show=False)
                if stress_type is not None:
                    self.plot_stresses(stresses=stresses, stress_eval_points=stress_eval_points, time_step=t_step, show=False)

                if stress_type is None:
                    # plt.title(f'Structure at t ={t: 9.5f}')
                    plt.title(f'Structure at t ={t: 1.2e}')
                else:
                    plt.title(f'Stress (sigma_{stress_type}) Colorplot of Structure at t ={t:1.2e}')
                plt.xlabel(f'x (m*{visualization_scaling_factor:3.0e})')
                plt.ylabel(f'y (m*{visualization_scaling_factor:3.0e})')
                plt.gca().set_aspect('equal')
                if save_plots or video_file_name is not None:
                    plt.savefig(f'plots/video_plot_at_t_{t:9.9f}.png', bbox_inches='tight')
                if show:
                    plt.show()
                plt.close()

            if video_file_name is not None:
                self.generate_video(video_file_name=video_file_name, video_fps=video_fps)

        elif dof is not None:
            if stress_type is not None:
                visualization_scaling_factor = max(np.linalg.norm(stresses, axis=1))*0.01/max(np.linalg.norm(U_reshaped, axis=1))
            else:
                visualization_scaling_factor = 1

            plt.figure()
            for index in dof:
                if show_dislpacements:
                    plt.plot(self.t_eval, self.U[index, :]*visualization_scaling_factor*50, '-', label=f'Displacement of node {index}')
                if stress_type is not None:
                    plot_stresses = stresses[:, index]   # (time_step, dof)
                    plt.plot(self.t_eval, plot_stresses, '-o', label=f'Stress of node {index}')
            
            if show_dislpacements and stress_type is not None:
                plt.title(f'Stress (sigma_{stress_type}) and Scaled Displacement vs. Time')
                plt.ylabel(f'Stress (Pa) and Displacement (m /{visualization_scaling_factor:3.0e})')
            elif show_dislpacements:
                plt.title(f'Y-Displacement of Node(s)')
                plt.ylabel('Displacement (m)')
            elif stress_type is not None:
                plt.title(f'Stress (sigma_{stress_type}) vs. Time')
                plt.ylabel('Stress (Pa)')

            plt.xlabel('Time (s)')
            plt.legend()

            if show:
                plt.show()


if __name__ == "__main__":
    sim1 = Simulation()
    sim1.theta[0] = np.pi/2
    sim1.u[0] = 3000
    sim1.u[1] = 3000
    sim1.u[2] = 1500
    sim1.u[3] = 1500
    sim1.u[8] = 1000
    #print(sim1.y)
    sim1.simulate_dynamics(sim1.u)

    print(sim1.deltat)
    print(sim1.y)

    sim1.plot_rigid_body_displacement()
    #sim1.savefigures(-5,25,-5,25)
    #sim1.generate_video("testplot1.avi",10)
    print("hello end of file")
