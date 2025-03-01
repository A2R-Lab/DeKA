
import math

import autograd.numpy as np
import autograd.numpy as sqrt
from autograd.numpy.linalg import norm
from autograd.numpy.linalg import inv
from autograd import jacobian
import pdb

from simulator import Sim

class Quadrotor():
    def __init__(self):
        self.sim = Sim()

        self.T = np.diag([1.0, -1, -1, -1])
        self.H = np.vstack([np.zeros((1,3)), np.eye(3)])

        # Quadrotor parameters
        self.mass = 0.035  # center mass
        self.g = 9.81  # gravity
        # thrustToTorque = 0.005964552
        self.thrustToTorque = 0.0008  # thrust to torque ratio
        self.el = 0.046/1.414213562  # arm length
        
        self.scale = 65535  # PWM scale
        self.kt = 2.245365e-6*self.scale # thrust coefficient, u is PWM in range [0...1], 0 is no thrust, 1 is max thrust
        self.km = self.kt*self.thrustToTorque # moment coefficient
        self.Jx = self.el**2*(self.mass/4)*2
        self.Jy = self.el**2*(self.mass/4)*2
        self.Jz = self.el**2*self.mass
        self.J = np.array([[self.el**2*(self.mass/4)*2, 0, 0], [0, self.el**2*(self.mass/4)*2, 0], [0, 0, self.el**2*self.mass]])  # inertia
        # Configuration Space
        self.num_states = 12
        self.num_controls = 4

        #  Choose Q and R matrices based on Bryson's rule
        max_dev_x = np.array([0.1, 0.1, 0.1,  0.5, 0.5, 0.05,  0.5, 0.5, 0.5,  0.7, 0.7, 0.2])
        max_dev_u = np.array([0.5, 0.5, 0.5, 0.5])/6
        
        self.Q = np.diag(1./max_dev_x**2)
        self.R = np.diag(1./max_dev_u**2)
        self.A_jac = jacobian(self.quad_dynamics_rk4, 0)  # jacobian wrt x
        self.B_jac = jacobian(self.quad_dynamics_rk4, 1)  # jacobian wrt u

      
    # Quadrotor dynamics -- single rigid body dynamics
    def quad_dynamics(self, x, u, mass,Jx,Jy,Jz):
        '''
        State vector x (12x1):
            x[0] : x-position (vehicle-1 frame, forward)
            x[1] : y-position (vehicle-1 frame, lateral)
            x[2] : z-position (altitude)
            x[3] : x_dot (velocity along x)
            x[4] : y_dot (velocity along y)
            x[5] : z_dot (velocity along z)
            x[6] : φ (roll angle)
            x[7] : θ (pitch angle)
            x[8] : ψ (yaw angle)
            x[9] : p (roll rate)
            x[10]: q (pitch rate)
            x[11]: r (yaw rate)
        '''
        
        Ff,Fr,Fb,Fl = u
        pn,pe = x[0:2]        # Position
        h = x[2]
        u_,v,w = x[3:6]        # Velocity
        phi,theta,psi = x[6:9]   # Euler angles: [phi, theta, psi]
        p,q,r = x[9:12]   # Angular velocity
        F = (Ff+Fr+Fb+Fl)

        # The following matrix maps motor inputs to body torques.
        tau = np.array([self.el*(Fl-Fr),self.el*(Ff-Fb),(Fr+Fl-Ff-Fb)/self.thrustToTorque])
        x_dot,y_dot,z_dot = u_,v,w
        phi_dot,theta_dot,psi_dot = p,q,r

        u_dot = (np.sin(psi)*np.sin(phi)+np.cos(psi)*np.sin(theta)*np.cos(phi) )*F/mass
        v_dot = (-np.sin(phi)*np.cos(psi)+np.sin(psi)*np.sin(theta)*np.cos(phi))*F/mass
        w_dot = -(self.g - np.cos(phi)*np.cos(theta)*F/mass)
       
        p_dot = (Jy-Jz)/Jx *q*r +tau[0]/Jx
        q_dot =  (Jz-Jx)/Jy *p*r + tau[1] / Jy
        r_dot = (Jx-Jy)/Jz *q*p + tau[2] / Jz
        print("tau: ", tau)
        print("rdot: ", r_dot)
        return np.hstack([x_dot,y_dot,z_dot,u_dot,v_dot,w_dot,phi_dot,theta_dot,psi_dot,p_dot,q_dot,r_dot])
    
    def get_tau(self,u,el,thrustToTorque):
        Ff,Fr,Fb,Fl = u
        tau = np.array([el*(Fl-Fr),el*(Ff-Fb),(Fr+Fl-Ff-Fb)/thrustToTorque])
        return tau
    def get_linearized_dynamics(self, xg, uhover, mass,Jx,Jy,Jz):
   
        Anp = self.A_jac(xg, uhover, mass,Jx,Jy,Jz)
        Bnp = self.B_jac(xg, uhover, mass,Jx,Jy,Jz)
        self.Anp = Anp
        self.Bnp = Bnp

        return self.Anp, self.Bnp
    
    def get_linear_system(self, Vb, wB, dVb, dwB, F, tau):
        """
        Build the linear system A * [m, Jx, Jy, Jz]^T = b
        from one time instant of Newton-Euler data.
        
        Inputs:
        Vb, wB : 3D arrays for linear velocity (m/s) and angular velocity (rad/s) in body frame
        dVb, dwB : 3D arrays for linear acceleration and angular acceleration in body frame
        F, tau : 3D arrays for net force (N) and torque (N·m) in body frame

        Returns:
        A : shape (6,4)
        b : shape (6,)
        """
        # Cross terms for translational eq
        a = dVb + np.cross(wB, Vb)  
        A = np.zeros((6, 4))
        b = np.zeros((6,1))

        print("a3: ", a[2],"F: ", F[2])
        # 1) Translational
        A[0,:] = [ a[0], 0,    0,    0 ]
        A[1,:] = [ a[1], 0,    0,    0 ]
        A[2,:] = [ a[2], 0,    0,    0 ]
        b[0,0] = F[0]
        b[1,0] = F[1]
        b[2,0] = F[2]

        # 2) Rotational
        # For diagonal I => Jx, Jy, Jz. Let p,q,r = wB, dp,dq,dr = dwB
        p,q,r = wB
        dp, dq, dr = dwB
        tau_x, tau_y, tau_z = tau

        # eq for tau_x = Jx*dp - (Jy - Jz)*q*r
        # => row is [0, dp, q*r, -q*r]
        A[3,:] = [ 0, dp, -q*r, q*r ]
        b[3,0] = tau_x

        # eq for tau_y = Jy*dq - (Jz + Jx)*r*p
        # => row is [0, -r*p, dq, r*p]
        A[4,:] = [ 0, r*p, dq, -r*p ]
        b[4,0] = tau_y

        # eq for tau_z = Jz*dr - (Jx - Jy)*p*q
        # => row is [0, p*q, -p*q, dr]
        A[5,:] = [ 0, -p*q, p*q, dr ]
        b[5,0] = tau_z

        
        return A, b
    
    def delta_x_quat(self, x_curr, x_nom):
        # Compute the state deviation
        delta_x = x_curr-x_nom
        return delta_x
    
    def get_hover_goals(self, mass, kt):
        # Hovering state and control input
        self.xg =  np.zeros(self.num_states)
        self.uhover = (mass*self.g/4 )*np.ones(4) # ~each motor thrust to compensate for gravity
        # print("Hovering Initial State and Control")
        # print(self.xg, self.uhover)

        return self.xg, self.uhover
    
  # RK4 integration with zero-order hold on u
    def quad_dynamics_rk4(self, x, u, mass,Jx,Jy,Jz):
         # Compute RK6 intermediate step
     
        f1 = self.quad_dynamics(x, u, mass,Jx,Jy,Jz)
        f2 = self.quad_dynamics(x + 0.5*self.sim.h*f1, u, mass,Jx,Jy,Jz)
        f3 = self.quad_dynamics(x + 0.5*self.sim.h*f2, u, mass,Jx,Jy,Jz)
        f4 = self.quad_dynamics(x + self.sim.h*f3, u, mass,Jx,Jy,Jz)
        xn = x + (self.sim.h/6.0)*(f1 + 2*f2 + 2*f3 + f4)
        # # print("f1", f1,"f2", f2,"f3", f3,"f4", f4)
        # # print("xn:",xn)
        return xn
