import numpy as np

from simulation import Simulation

sim1 = Simulation()
sim1.theta[0] = np.pi/2
sim1.u[0] = 1
sim1.u[1] = 1
sim1.u[2] = 5
sim1.u[6] = 5
#print(sim1.y)
sim1.simulate_dynamics(sim1.u)

print(sim1.deltat)
print(sim1.y)

print('objective', sim1.evaluate_objective(sim1.u))
print('constraint jacobian', sim1.evaluate_constraint_jacobian(sim1.u))

# sim1.plot_rigid_body_displacement()
# print("hello end of file")
