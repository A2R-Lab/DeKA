import math
import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy as sqrt
from autograd.numpy.linalg import norm
from autograd.numpy.linalg import inv
from autograd import jacobian
from autograd.test_util import check_grads
from collections import deque
from simulator import Sim


class DoublePendulum():
    def __init__(self):
        self.sim = Sim()
        self.g = 9.81
        self.m1 = 1.0
        self.m2 = 1.0
        self.l1 = 1.0
        self.l2 = 1.0
        self.num_states = 4
        self.num_controls = 1
        self.A_jac = jacobian(self.double_pen_rk4, 0)  # jacobian wrt x
        self.B_jac = jacobian(self.double_pen_rk4, 1)  # jacobian wrt u
        self.max_dev_x = np.array([np.pi / 2, 1.0, np.pi / 2, 1.0])  # max deviations for states
        self.max_dev_u = np.array([10.0])  # max control effort
        self.Q = np.diag(1.0 / self.max_dev_x**2)
        self.R = np.diag(1.0 / self.max_dev_u**2)
    
    def double_pendulum_dynamics(self, state, m1, m2, l1, l2, g, u):
        """
        Compute the derivatives for the double pendulum using the provided formulation.
        
        Args:
            state: [theta1, omega1, theta2, omega2] - state variables
            m1: Mass of the first pendulum
            m2: Mass of the second pendulum
            l1: Length of the first pendulum
            l2: Length of the second pendulum
            g: Acceleration due to gravity
            
        Returns:
            dstate_dt: Derivatives [dtheta1/dt, domega1/dt, dtheta2/dt, domega2/dt]
        """
        state = np.ravel(state)
        theta1, omega1, theta2, omega2 = state
        delta = theta2 - theta1

        # Compute alpha1, alpha2
        alpha1 = (l2 / l1) * (m2 / (m1 + m2)) * np.cos(delta)
        alpha2 = (l1 / l2) * np.cos(delta)

        # Compute f1 and f2
        f1 = (-l2 / l1) * (m2 / (m1 + m2)) * omega2**2 * np.sin(delta) - (g / l1) * np.sin(theta1)
        # print("f2", (l1 / l2) * omega1**2 * np.sin(delta) - (g / l2) * np.sin(theta2), "u", u/l2)
        f2 = (l1 / l2) * omega1**2 * np.sin(delta) - (g / l2) * np.sin(theta2)+(u[0]/l2)
    

        # Matrix A and its determinant
        detA = 1 - alpha1 * alpha2
        A_inv = (1 / detA) * np.array([[1, -alpha1], [-alpha2, 1]])

        # Solve for angular accelerations
        rhs = np.array([f1, f2])
        angular_accels = A_inv @ rhs

        omega1_dot, omega2_dot = angular_accels

        # Return the derivatives
        return np.array([omega1, omega1_dot, omega2, omega2_dot])
    def double_pen_rk4(self, x, u, m1, m2, l1, l2, g):
        k1 = self.double_pendulum_dynamics(x, m1, m2, l1, l2, g, u)
        k2 = self.double_pendulum_dynamics(x + 0.5 * self.sim.h * k1, m1, m2, l1, l2, g, u)
        k3 = self.double_pendulum_dynamics(x + 0.5 * self.sim.h * k2, m1, m2, l1, l2, g, u)
        k4 = self.double_pendulum_dynamics(x + self.sim.h * k3, m1, m2, l1, l2, g, u)

        return x + (self.sim.h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)