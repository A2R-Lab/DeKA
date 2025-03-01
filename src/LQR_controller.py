import math
import autograd.numpy as np
import autograd.numpy as sqrt
from autograd.numpy.linalg import norm
from autograd.numpy.linalg import inv
from autograd import jacobian

class LQRController():
    def __init__(self, delta_x_quat):
        self.delta_x_quat = delta_x_quat

    def get_QR_bryson(self):
        max_dev_x = np.array([0.1, 0.1, 0.01,  0.5, 0.5, 0.05,  0.5, 0.5, 0.5,  0.7, 0.7, 0.2])
        max_dev_u = np.array([0.5, 0.5, 0.5, 0.5])/6
        # max_dev_x = np.array([0.05, 0.05, 0.005,  0.5, 0.5, 0.05,  0.5, 0.5, 0.5,  0.7, 0.7, 0.2]) # increase = increase tolerance for state deviation
        # max_dev_u = np.array([0.8, 0.8, 0.8, 0.8])/6 # increase = trust the control less
        Q = np.diag(1./max_dev_x**2)
        R = np.diag(1./max_dev_u**2)

        return Q, R

    def update_linearized_dynamics (self, Anp, Bnp, Q, R):
        self.Anp = Anp.copy()
        self.Bnp = Bnp.copy()
        self.Q = Q.copy()
        self.R = R.copy()

    # Riccati recursion on the linearized dynamics
    def dlqr(self, A, B, Q, R, n_steps = 500):
        P = Q
        for i in range(n_steps):
            K = inv(R + B.T @ P @ B) @ B.T @ P @ A
            P = Q + A.T @ P @ (A - B @ K)
        return K, P

    # Drive the system from initial state to the hovering state using the LQR controller:
    # LQR controller, input is 13x1 state vector
    def compute(self, x_curr, x_nom, u_nom):
        K_lqr, P_lqr = self.dlqr(self.Anp, self.Bnp, self.Q, self.R)
        delta_x = self.delta_x_quat(x_curr, x_nom)
        return u_nom - K_lqr @ delta_x
