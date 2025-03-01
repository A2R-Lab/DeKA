import math
import autograd.numpy as np
import autograd.numpy as sqrt
from autograd.numpy.linalg import norm
from autograd.numpy.linalg import inv
from autograd import jacobian

class LQRController_pen():
    def __init__(self ):
        ...

    def get_QR_bryson(self):
        max_dev_x = np.array([np.pi / 2, 1.0, np.pi / 2, 1.0])
        max_dev_u = np.array([10.0])
        Q = np.diag(1./max_dev_x**2)
        R = np.diag(1./max_dev_u**2)

        return Q, R

    def update_linearized_dynamics (self, Anp, Bnp, Q, R):
        self.Anp = Anp
        self.Bnp = Bnp
        self.Q = Q
        self.R = R

    # Riccati recursion on the linearized dynamics
    def dlqr(self, A, B, Q, R, n_steps = 500):
        P = Q
        for i in range(n_steps):
            K = inv(R + B.T @ P @ B) @ B.T @ P @ A
            P = Q + A.T @ P @ (A - B @ K)
        return K , P

    # Drive the system from initial state to the hovering state using the LQR controller:
    # LQR controller, input is 13x1 state vector
    def compute(self, x_curr, x_nom, u_nom):
        K_lqr, P_lqr = self.dlqr(self.Anp, self.Bnp, self.Q, self.R)
        delta_x = x_curr-x_nom
        print("control",u_nom - K_lqr @ delta_x )
        return u_nom - K_lqr @ delta_x

