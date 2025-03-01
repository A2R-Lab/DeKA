import math

import autograd.numpy as np
import autograd.numpy as sqrt
from autograd.numpy.linalg import norm
from autograd.numpy.linalg import inv
from autograd import jacobian
import pdb

from utilities import quaternion_to_rotation_matrix

from simulator import Sim

class Quadrotor():
    def __init__(self):
        self.sim = Sim()

        self.T = np.diag([1.0, -1, -1, -1])
        self.H = np.vstack([np.zeros((1,3)), np.eye(3)])

        # Quadrotor parameters
        self.mass = 0.035  # mass
        self.J = np.array([[1.66e-5, 0.83e-6, 0.72e-6], 
                           [0.83e-6, 1.66e-5, 1.8e-6], 
                           [0.72e-6, 1.8e-6, 2.93e-5]], dtype=np.float64)  # inertia
        self.g = 9.81  # gravity
        # thrustToTorque = 0.005964552
        self.thrustToTorque = 0.0008  # thrust to torque ratio
        self.el = 0.046/1.414213562  # arm length
        self.scale = 65535  # PWM scale
        self.kt = 2.245365e-6*self.scale # thrust coefficient, u is PWM in range [0...1], 0 is no thrust, 1 is max thrust
        self.km = self.kt*self.thrustToTorque # moment coefficient

        # Configuration Space
        self.num_states = 13
        self.num_controls = 4

        # Linearize Dynamics w.r.t x and u
        # f(x,u,theta) = A(theta) * x + B(theta) * u
        self.A_jac = jacobian(self.quad_dynamics_rk4, 0)  # jacobian wrt x
        self.B_jac = jacobian(self.quad_dynamics_rk4, 1)  # jacobian wrt u

        #  Choose Q and R matrices based on Bryson's rule
        max_dev_x = np.array([0.1, 0.1, 0.1,  0.5, 0.5, 0.05,  0.5, 0.5, 0.5,  0.7, 0.7, 0.2])
        max_dev_u = np.array([0.5, 0.5, 0.5, 0.5])/6
        
        self.Q = np.diag(1./max_dev_x**2)
        self.R = np.diag(1./max_dev_u**2)
    
    def hat(self, v):
        return np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0.0]])
    
    def L(self, q):
        s = q[0]
        v = q[1:4]
        up = np.hstack([s, -v])
        down = np.hstack([v.reshape(3,1), s*np.eye(3) + self.hat(v)])
        L = np.vstack([up,down])
        return L

    def qtoQ(self, q):
        return self.H.T @ self.T @ self.L(q) @ self.T @ self.L(q) @ self.H

    def G(self, q):
        return self.L(q) @ self.H

    def rptoq(self, phi):
        return (1./math.sqrt(1+phi.T @ phi)) * np.hstack([1, phi])

    def qtorp(self, q):
        return q[1:4]/q[0]
    
    def delta_x_quat(self, x_curr, x_nom=None):
        # if not x_nom:
        #     x_nom = self.xg
        q = x_curr[3:7]
        phi = self.qtorp(self.L(self.qg).T @ q)
        delta_x = np.hstack([x_curr[0:3]-x_nom[0:3], phi, x_curr[7:10]-x_nom[7:10], x_curr[10:13]-x_nom[10:13]])
        return delta_x

    def E(self, q):
        up = np.hstack([np.eye(3), np.zeros((3,3)), np.zeros((3,6))])
        mid = np.hstack([np.zeros((4,3)), self.G(q), np.zeros((4,6))])
        down = np.hstack([np.zeros((6,3)), np.zeros((6,3)), np.eye(6)])
        E = np.vstack([up, mid, down])
        return E
    
    # Quadrotor dynamics -- single rigid body dynamics
    def quad_dynamics(self, x, u, theta):
        mass = theta[0].copy()
        Ixx,Ixy,Ixz,Iyy,Iyz,Izz = theta[1:7]
        J = np.array([[Ixx, Ixy, Ixz],
                      [Ixy, Iyy, Iyz],
                      [Ixz, Iyz, Izz]], dtype=np.float64)

        r = x[0:3]  # position
        q = x[3:7]/norm(x[3:7])  # normalize quaternion
        v = x[7:10]  # linear velocity
        omg = x[10:13]  # angular velocity
        Q = self.qtoQ(q)  # quaternion to rotation matrix

        dr = v
        dq = 0.5*self.L(q)@self.H@omg
        dv = np.array([0, 0, -self.g]) + (1/mass)*Q@np.array([[0, 0, 0, 0], [0, 0, 0, 0], [self.kt, self.kt, self.kt, self.kt]])@u
        domg = inv(J)@(-self.hat(omg)@J@omg + np.array([[-self.el*self.kt, -self.el*self.kt, self.el*self.kt, self.el*self.kt], [-self.el*self.kt, self.el*self.kt, self.el*self.kt, -self.el*self.kt], [-self.km, self.km, -self.km, self.km]])@u)
        return np.hstack([dr, dq, dv, domg])

    # RK4 integration with zero-order hold on u
    def quad_dynamics_rk4(self, x, u, theta):
        f1 = self.quad_dynamics(x, u, theta)
        f2 = self.quad_dynamics(x + 0.5*self.sim.h*f1, u, theta)
        f3 = self.quad_dynamics(x + 0.5*self.sim.h*f2, u, theta)
        f4 = self.quad_dynamics(x + self.sim.h*f3, u, theta)
        xn = x + (self.sim.h/6.0)*(f1 + 2*f2 + 2*f3 + f4)
        xnormalized = xn[3:7]/norm(xn[3:7])  # normalize quaternion
        return np.hstack([xn[0:3], xnormalized, xn[7:13]])
    
    def get_data_matrix_km(self, x_curr, dx):
        a_x, a_y, a_z = dx[7:10]
        a_p, a_q, a_r = dx[10:13]
        w_x, w_y, w_z = x_curr[10:13]
        
        a_x = a_x * 1e4
        a_y = a_y * 1e4
        a_z = a_z * 1e4


        A = np.array([
            [a_x, 0, 0, 0, 0, 0, 0],  # fx equation (acceleration)
            [a_y, 0, 0, 0, 0, 0, 0],  # fy equation
            [a_z, 0, 0, 0, 0, 0, 0],  # fz equation
            [0, a_p, a_q-w_x*w_z, a_r+w_x*w_y, -w_y*w_z, w_y**2+w_z**2, w_y*w_z],  # τx equation (angular acceleration)
            [0, w_x*w_z, a_p+w_y*w_z, w_z**2-w_x**2, a_q, a_r-w_x*w_y, -w_x*w_z],  # τy equation
            [0, -w_x*w_y, w_x**2-w_y**2, a_p-w_y*w_z, w_x*w_y, a_q+w_x*w_z, a_r]  # τz equation
        ], dtype=np.float64)
        return A
    
    def get_force_vector_km(self, x_curr, u_curr, theta):
        R = self.qtoQ(x_curr[3:7])

        F_b = np.array([[0], 
                      [0],  
                      [np.sum(u_curr) * self.kt]]) # TODO: fix this dim
        F_w = R @ F_b
        F_w[2] -= theta[0] * self.g

        tau_b = np.array([[self.el*self.kt*(-u_curr[0]-u_curr[1]+u_curr[2]+u_curr[3])], # - - + +
                        [self.el*self.kt*(-u_curr[0]+u_curr[1]+u_curr[2]-u_curr[3])],  # - + + -
                        [self.km*(-u_curr[0] + u_curr[1] - u_curr[2] + u_curr[3])]]) # - + - +
        
        return np.vstack([F_w, tau_b])

    def get_data_matrix(self, x_curr, dx):
        a_x, a_y, a_z = dx[7:10]
        a_p, a_q, a_r = dx[10:13]
        w_x, w_y, w_z = x_curr[10:13]

        A = np.array([
            [a_x, 0, 0, 0, 0, 0, 0],  # fx equation (acceleration)
            [a_y, 0, 0, 0, 0, 0, 0],  # fy equation
            [a_z, 0, 0, 0, 0, 0, 0],  # fz equation
            [0, a_p, a_q-w_x*w_z, a_r+w_x*w_y, -w_y*w_z, w_y**2+w_z**2, w_y*w_z],  # τx equation (angular acceleration)
            [0, w_x*w_z, a_p+w_y*w_z, w_z**2-w_x**2, a_q, a_r-w_x*w_y, -w_x*w_z],  # τy equation
            [0, -w_x*w_y, w_x**2-w_y**2, a_p-w_y*w_z, w_x*w_y, a_q+w_x*w_z, a_r]  # τz equation
        ], dtype=np.float64)
        return A
    
    def get_force_vector(self, x_curr, u_curr, theta):
        R = self.qtoQ(x_curr[3:7])

        F_b = np.array([[0], 
                      [0],  
                      [np.sum(u_curr) * self.kt]]) # TODO: fix this dim
        F_w = R @ F_b
        F_w[2] -= theta[0] * self.g

        tau_b = np.array([[self.el*self.kt*(-u_curr[0]-u_curr[1]+u_curr[2]+u_curr[3])], # - - + +
                        [self.el*self.kt*(-u_curr[0]+u_curr[1]+u_curr[2]-u_curr[3])],  # - + + -
                        [self.km*(-u_curr[0] + u_curr[1] - u_curr[2] + u_curr[3])]]) # - + - +
        
        return np.vstack([F_w, tau_b])
    
    def get_hover_goals(self, theta):
        mass = theta[0].copy()

        # Hovering state and control input
        self.rg = np.array([0.0, 0, 0.0])
        self.qg = np.array([1.0, 0, 0, 0])
        self.vg = np.zeros(3)
        self.omgg = np.zeros(3)
        self.xg = np.hstack([self.rg, self.qg, self.vg, self.omgg])
        self.uhover = (mass*self.g/self.kt/4)*np.ones(4)  # ~each motor thrust to compensate for gravity
        # print("Hovering Initial State and Control")
        # print(self.xg, self.uhover)

        return self.xg, self.uhover
    
    #TODO: add kt as param?
    def get_linearized_dynamics(self, xg, uhover, theta):
        self.rg = np.array([0.0, 0, 0.0])
        self.qg = np.array([1.0, 0, 0, 0])
        Anp = self.A_jac(xg, uhover, theta)
        Bnp = self.B_jac(xg, uhover, theta)
        self.Anp = self.E(self.qg).T @ Anp @ self.E(self.qg)
        self.Bnp = self.E(self.qg).T @ Bnp

        return self.Anp, self.Bnp
    
    # TODO: add this method
    def perturb_initial_state(self):
        pass
