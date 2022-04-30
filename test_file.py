import numpy as np

from simulation import Simulation
from optimization_framework.optimization_problem import OptimizationProblem
from optimization_framework.optimizers.gradient_descent_optimizer import GradientDescentOptimizer

sim1 = Simulation()
#print(sim1.y)
sim1.simulate_dynamics(sim1.u)

print(sim1.deltat)
print(sim1.y)

# print('objective', sim1.evaluate_objective(sim1.u))
# print('constraint jacobian', sim1.evaluate_constraint_jacobian())

# sim1.plot_rigid_body_displacement()
# print("hello end of file")

control_optimization = OptimizationProblem()
steepest_descent_optimizer = GradientDescentOptimizer(alpha=1e-3)
control_optimization.set_model(model=sim1)
control_optimization.set_optimizer(steepest_descent_optimizer)
control_optimization.setup()
x0 = np.ones(sim1.num_control_inputs + sim1.num_constraints,)*200.
steepest_descent_optimizer.set_initial_guess(x0)

control_optimization.run(line_search='GFD', grad_norm_abs_tol=1.e-2, delta_x_abs_tol=1e-5, updating_penalty=True, max_iter=15000)
solution = control_optimization.report(history=True)
#control_optimization.plot()
sim1.plot_rigid_body_displacement()
print("final generic cost evaluation",sim1.deltat*sim1.u.dot(sim1.u))

sim1.savefigures(-5,25,-5,25)
sim1.generate_video("testplot1.avi",10)

print("end of file")