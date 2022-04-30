import numpy as np

from simulation import Simulation
from optimization_framework.optimization_problem import OptimizationProblem
from optimization_framework.optimizers.gradient_descent_optimizer import GradientDescentOptimizer
from optimization_framework.optimizers.finite_difference import finite_difference

sim1 = Simulation()
sim1.theta[0] = np.pi/2
sim1.u[0] = 1
sim1.u[1] = 1.1
sim1.u[2] = 7
sim1.u[6] = 6
#print(sim1.y)
sim1.simulate_dynamics(sim1.u)

print(sim1.deltat)
print(sim1.y)

# print('objective', sim1.evaluate_objective(sim1.u))
# print('constraint jacobian', sim1.evaluate_constraint_jacobian())

# sim1.plot_rigid_body_displacement()
# print("hello end of file")

def evaluate_fd(x0):
    sim1.simulate_dynamics(x0[:-1])

    c = np.zeros((sim1.num_constraints,))
    c[0] = sim1.W.dot(sim1.y) - sim1.y_target
    
    lagrange_multipliers = np.array([x0[-1]])

    # return [sim1.theta[-1]]
    return c
    # return [sim1.ydotdot[-1]]
    # return [sim1.xdotdot[-1]]
    # print(lagrange_multipliers.dot(c))
    # return [lagrange_multipliers.dot(c)]


# sim1.setup()
# sim1.theta[0] = np.pi/2
# x0 = np.ones(sim1.num_control_inputs + sim1.num_constraints,)*200.
# x0[2] = 100
# sim1.simulate_dynamics(x0[:-1])
# sim1.lagrange_multipliers = x0[-1]

# print('FD', finite_difference(evaluate_fd, x0)[:-1])
# print('ANALYTIC', sim1.evaluate_analytic_test(x0))
# print('JAC', sim1.evaluate(x0)[3])


control_optimization = OptimizationProblem()
steepest_descent_optimizer = GradientDescentOptimizer(alpha=1e-3)
control_optimization.set_model(model=sim1)
control_optimization.set_optimizer(steepest_descent_optimizer)
control_optimization.setup()
x0 = np.ones(sim1.num_control_inputs + sim1.num_constraints,)*200.
steepest_descent_optimizer.set_initial_guess(x0)
print("model outputs",control_optimization.evaluate_model(x0))
control_optimization.run(line_search='GFD', grad_norm_abs_tol=1.e-2, delta_x_abs_tol=1e-5, updating_penalty=True, max_iter=100000)
solution = control_optimization.report(history=True)
control_optimization.plot()
sim1.plot_rigid_body_displacement()
