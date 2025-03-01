import math
import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy as sqrt
from autograd.numpy.linalg import norm
from autograd.numpy.linalg import inv
from autograd import jacobian
from autograd.test_util import check_grads
from collections import deque
np.set_printoptions(precision=4, suppress=True)
# Note: autograd does not work with np.block

from quadrotor_dynamics import Quadrotor
from double_pendulum_dynamics import DoublePendulum
from double_pen_LQR import LQRController_pen
from LQR_controller import LQRController
from estimation_methods import *
from utilities import *

import argparse


class OnlineParamEst:
    def __init__(self):
        # Quadrotor Tasks:
        self.quadrotor = Quadrotor()
        self.quadrotor_controller = LQRController(self.quadrotor.delta_x_quat)
        self.double_pendulum =  DoublePendulum() 
        self.double_pendulum_controller = LQRController_pen() #TODO: use one controller

    def simulate_quadrotor_hover_with_Naive_MPC(self, update_status: str, NSIM: int =300):
        
        np.set_printoptions(suppress=False, precision=10)
        # initialize quadrotor parameters
        Ixx, Iyy, Izz = self.quadrotor.J[0,0], self.quadrotor.J[1,1], self.quadrotor.J[2,2]
        Ixy, Ixz, Iyz = self.quadrotor.J[0,1], self.quadrotor.J[0,2], self.quadrotor.J[1,2]
        theta = np.array([self.quadrotor.mass,Ixx,Ixy,Ixz,Iyy,Iyz,Izz])
        
        # get tasks goals
        x_nom, u_nom = self.quadrotor.get_hover_goals(theta)
        Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta)
        Q, R = self.quadrotor_controller.get_QR_bryson()
        self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

        # randomly perturb initial state
        x0 = np.copy(x_nom)
        x0[0:3] += np.array([0.2, 0.2, -0.2])  # disturbed initial position
        x0[3:7] = self.quadrotor.rptoq(np.array([1.0, 0.0, 0.0]))  # disturbed initial attitude #TODO: move math methods to utilities class
        print("Perturbed Intitial State: ")
        print(x0)

        x_all = []
        u_all = []
        theta_all = []

        x_curr = np.copy(x0)
        x_prev = np.copy(x0)

        change_params = False
        changing_steps = [100] #np.random.choice(range(20,180),size=3, replace=False)

        # simulate the dynamics with the LQR controller
        for i in range(NSIM):
            # change mass
            if i in changing_steps:
                update_scale = np.random.uniform(1,2)
                theta *= update_scale

            # MPC controller
            if update_status == "immediate_update":
                if i in changing_steps:
                    x_nom, u_nom = self.quadrotor.get_hover_goals(theta)
                    Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta)
                    self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)
                u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)
            elif update_status == "never_update":
                u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)
            elif update_status == "late_update":
                if i == 120: # TODO: hardcoded
                    x_nom, u_nom = self.quadrotor.get_hover_goals(theta)
                    Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta)
                    self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)
                u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)
            else:
                print("invalid update status. Choose from 'immediate_update', 'never_update', or 'late_update'.")
                exit()

            print("step: ", i, "\n", 
                  "u_k: ", u_curr, "\n", 
                  "x_k: ", x_prev[:3], "\n", 
                  "theta_k: ", theta, "\n", 
                  )

            # postponing the dynamics model by telling it the correct parameters after several steps
            x_prev = x_curr
            x_curr = self.quadrotor.quad_dynamics_rk4(x_curr, u_curr, theta)

            x_curr = x_curr.reshape(x_curr.shape[0]).tolist()
            u_curr = u_curr.reshape(u_curr.shape[0]).tolist()

            x_all.append(x_curr)
            u_all.append(u_curr)
            theta_all.append(theta.copy())
        return x_all, u_all, theta_all 
    
    def simulate_quadrotor_hover_with_RLS(self, NSIM: int =300): #TODO: add RLS parameters here!
        # np.random.seed(42)
        self.quadrotor = Quadrotor()
        self.quadrotor_controller = LQRController(self.quadrotor.delta_x_quat)
        # initialize quadrotor parameters
        np.set_printoptions(suppress=False, precision=10)
        Ixx, Iyy, Izz = self.quadrotor.J[0,0], self.quadrotor.J[1,1], self.quadrotor.J[2,2]
        Ixy, Ixz, Iyz = self.quadrotor.J[0,1], self.quadrotor.J[0,2], self.quadrotor.J[1,2]
        # theta = np.array([self.quadrotor.mass,Ixx,Ixy,Ixz,Iyy,Iyz,Izz])
        theta = np.array([self.quadrotor.mass, Ixx,Ixy,Ixz,Iyy,Iyz,Izz])
        theta_hat = theta.copy()
       # print("RLS theta: ", theta)
        
        # get tasks goals
        x_nom, u_nom = self.quadrotor.get_hover_goals(theta_hat)
        Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat)
        Q, R = self.quadrotor_controller.get_QR_bryson()
        self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

        # randomly perturb initial state
        x0 = np.copy(x_nom)
        x0[0:3] += np.array([0.1, -0.15, -0.1])  # disturbed initial position
        x0[3:7] = self.quadrotor.rptoq(np.array([0.3, 0.4, 0.2]))  # disturbed initial attitude #TODO: move math methods to utilities class
        # print("Perturbed Intitial State: ")
        # print(x0)

        # get initial control and state
        u_curr = self.quadrotor_controller.compute(x0, x_nom, u_nom)
        x_curr = np.copy(x0)

        x_all = []
        u_all = []
        theta_all = []
        theta_hat_all = []

        changing_steps = [100]#np.random.choice(range(20,180),size=2, replace=False)
 
        rls = RLS(num_params=7, forgetting_factor=0.7, theta_hat=theta.copy().reshape(-1,1))
        n = 10
        A_tot = np.zeros((6*n,7))
        b_tot = np.zeros((6*n,1))

        # simulate the dynamics with the LQR controller
        for i in range(NSIM):
            # noising the parameters
            # if i % 10 == 0:
            #     theta += np.random.normal(0, process_noise_std, size=theta.shape)

            # Change system parameters at specific step
            if i in changing_steps:
                update_scale = np.random.uniform(2, 3)
                theta[0] *= update_scale
    
            # update goals
            x_nom, u_nom = self.quadrotor.get_hover_goals(theta_hat.copy())
            Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat.copy())
            self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

            # compute controls
            u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)                    # at t=k

            # step
            if i > 0:
                process_noise_std = 0.02 * np.abs(theta[0])  # Process noise
                theta[0] += np.random.normal(0, process_noise_std)

            x_curr = self.quadrotor.quad_dynamics_rk4(x_curr, u_curr, theta)       # at t=k+1
            # formulate measurement model
            A = self.quadrotor.get_data_matrix(x_curr, self.quadrotor.quad_dynamics(x_curr, u_curr, theta))
            b = self.quadrotor.get_force_vector(x_curr, u_curr, theta)
            measurement_noise_std = 0.05 * np.abs(b)
            b += np.random.normal(0, measurement_noise_std, size = b.shape)

            A_tot = np.roll(A_tot, shift=3, axis=0)
            A_tot[:6] = A
            b_tot = np.roll(b_tot, shift=3, axis=0)
            b_tot[:6] = b

            if i % 5 == 0 and i > 0:
                theta_hat = rls.iterate(A_tot,b_tot).reshape(-1,)
            
            # print(A@(theta.reshape(-1,1)))
            # print(b)

            # print("step: ", i, "\n", 
                # "prediction_err: ", np.linalg.norm(theta-theta_hat)/7, "\n"
                #   )


            x_curr = x_curr.reshape(x_curr.shape[0]).tolist() #TODO: x one step in front of controls
            u_curr = u_curr.reshape(u_curr.shape[0]).tolist()

            x_all.append(x_curr)
            u_all.append(u_curr)
            theta_all.append(theta.copy())
            theta_hat_all.append(theta_hat.copy())

        return x_all, u_all, theta_all, theta_hat_all
    
    def simulate_quadrotor_hover_with_KF(self, Q_noise, R_noise, NSIM: int =300): #TODO: add RLS parameters here!
        # np.random.seed(42)
        self.quadrotor = Quadrotor()
        self.quadrotor_controller = LQRController(self.quadrotor.delta_x_quat)
        # initialize quadrotor parameters
        np.set_printoptions(suppress=False, precision=10)
        Ixx, Iyy, Izz = self.quadrotor.J[0,0], self.quadrotor.J[1,1], self.quadrotor.J[2,2]
        Ixy, Ixz, Iyz = self.quadrotor.J[0,1], self.quadrotor.J[0,2], self.quadrotor.J[1,2]
        # theta = np.array([self.quadrotor.mass,Ixx,Ixy,Ixz,Iyy,Iyz,Izz])
      #  print(Ixx)
        theta = np.array([self.quadrotor.mass, Ixx,Ixy,Ixz,Iyy,Iyz,Izz])
        theta_hat = theta.copy()
       # print("KF theta: ", theta)
        
        # get tasks goals
        x_nom, u_nom = self.quadrotor.get_hover_goals(theta_hat)
        Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat)
        Q, R = self.quadrotor_controller.get_QR_bryson()
        self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

        # randomly perturb initial state
        x0 = np.copy(x_nom)
        x0[0:3] += np.array([0.1, -0.15, -0.1])  # disturbed initial position
        x0[3:7] = self.quadrotor.rptoq(np.array([0.3, 0.4, 0.2]))  # disturbed initial attitude #TODO: move math methods to utilities class
       # print("Perturbed Intitial State: ")
        #print(x0)

        # get initial control and state
        u_curr = self.quadrotor_controller.compute(x0, x_nom, u_nom)
        x_curr = np.copy(x0)

        x_all = []
        u_all = []
        theta_all = []
        theta_hat_all = []

        changing_steps = [100]#np.random.choice(range(20,180),size=2, replace=False)

        n = 10
        A_tot = np.zeros((6*n,7))
        b_tot = np.zeros((6*n,1))
        Q_ekf = 1e-5 * np.eye(len(theta))
        R_ekf = 1e-5 * np.eye(6*n)


        kf = EKF(num_params=7, process_noise=Q_noise, measurement_noise=R_noise)#, theta_hat=theta.copy().reshape(-1,1))

        # simulate the dynamics with the LQR controller
        for i in range(NSIM):
            # noising the parameters
            # if i % 10 == 0:
            #     theta += np.random.normal(0, process_noise_std, size=theta.shape)

            # Change system parameters at specific step
            if i in changing_steps:
                update_scale = np.random.uniform(2, 3)
                theta[0] *= update_scale

            # update goals
            x_nom, u_nom = self.quadrotor.get_hover_goals(theta_hat.copy())
            Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat.copy())
            self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

            # compute controls
            u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)                    # at t=k

            # step
            if i > 0 and i % 15 == 0:
                process_noise_std = 0.1 * np.abs(theta[0])  # Process noise
                theta[0] += np.random.normal(0, process_noise_std)
            x_curr = self.quadrotor.quad_dynamics_rk4(x_curr, u_curr, theta)      # at t=k+1
            # formulate measurement model
            A = self.quadrotor.get_data_matrix(x_curr, self.quadrotor.quad_dynamics(x_curr, u_curr, theta))
            b = self.quadrotor.get_force_vector(x_curr, u_curr, theta)
            measurement_noise_std = 0.05 * np.abs(b)
            b += np.random.normal(0, measurement_noise_std, size = b.shape)

            A_tot = np.roll(A_tot, shift=3, axis=0)
            A_tot[:6] = A
            b_tot = np.roll(b_tot, shift=3, axis=0)
            b_tot[:6] = b

            if i % 5 == 0 and i > 0:
                theta_hat = kf.iterate(A_tot,b_tot).reshape(-1,)
            
            # print(A@(theta.reshape(-1,1)))
            # print(b)

            # print("step: ", i, "\n", 
                # "prediction_err: ", np.linalg.norm(theta-theta_hat)/7, "\n"
                #   )

            x_curr = x_curr.reshape(x_curr.shape[0]).tolist() #TODO: x one step in front of controls
            u_curr = u_curr.reshape(u_curr.shape[0]).tolist()

            x_all.append(x_curr)
            u_all.append(u_curr)
            theta_all.append(theta.copy())
            theta_hat_all.append(theta_hat.copy())

        return x_all, u_all, theta_all, theta_hat_all
    
    # def simulate_quadrotor_hover_with_no_estimation(self, NSIM: int =200): #TODO: add RLS parameters here!
    #     # initialize quadrotor parameters
    #     np.set_printoptions(suppress=False, precision=10)
    #     Ixx, Iyy, Izz = self.quadrotor.J[0,0], self.quadrotor.J[1,1], self.quadrotor.J[2,2]
    #     Ixy, Ixz, Iyz = self.quadrotor.J[0,1], self.quadrotor.J[0,2], self.quadrotor.J[1,2]
    #     # theta = np.array([self.quadrotor.mass,Ixx,Ixy,Ixz,Iyy,Iyz,Izz])
    #     theta = np.array([Ixx,Ixy,Ixz,Iyy,Iyz,Izz])
    #     theta_hat = theta.copy()
        
    #     # get tasks goals
    #     x_nom, u_nom = self.quadrotor.get_hover_goals()
    #     Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat)
    #     Q, R = self.quadrotor_controller.get_QR_bryson()
    #     self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

    #     # randomly perturb initial state
    #     x0 = np.copy(x_nom)
    #     x0[0:3] += np.array([0.2, 0.2, -0.2])  # disturbed initial position
    #     x0[3:7] = self.quadrotor.rptoq(np.array([1.0, 0.0, 0.0]))  # disturbed initial attitude #TODO: move math methods to utilities class
    #     print("Perturbed Intitial State: ")
    #     print(x0)

    #     # get initial control and state
    #     u_curr = self.quadrotor_controller.compute(x0, x_nom, u_nom)
    #     x_curr = np.copy(x0)

    #     x_all = []
    #     u_all = []
    #     theta_all = []
    #     theta_hat_all = []

    #     changing_steps = [50]#np.random.choice(range(20,180),size=2, replace=False)
 
    #     kf = EKF(num_params=6, measurement_noise=np.random.normal(0, 1e-6, size = (30,1)), process_noise=np.random.normal(0, 1e-6, size = (30,6)))
    #     n = 10
    #     A_tot = np.zeros((3*n,6))
    #     b_tot = np.zeros((3*n,1))

    #     # simulate the dynamics with the LQR controller
    #     for i in range(NSIM):
    #         # noising the parameters
    #         # if i % 10 == 0:
    #         #     theta += np.random.normal(0, process_noise_std, size=theta.shape)

    #         # Change system parameters at specific step
    #         if i in changing_steps:
    #             update_scale = np.random.uniform(2, 4)
    #             theta *= update_scale

    #         # update goals
    #         x_nom, u_nom = self.quadrotor.get_hover_goals()
    #         Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat)
    #         self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

    #         # compute controls
    #         u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)                    # at t=k

    #         # step
    #         process_noise_std = 0.05 * np.abs(x_curr)  # Process noise

    #         x_curr = self.quadrotor.quad_dynamics_rk4(x_curr, u_curr, theta)+ np.random.normal(0, process_noise_std, size=(13,))       # at t=k+1
            
    #         # print(A@(theta.reshape(-1,1)))
    #         # print(b)

    #         print("step: ", i, "\n", 
    #             # "prediction_err: ", np.linalg.norm(theta-theta_hat)/7, "\n"
    #               )


    #         x_curr = x_curr.reshape(x_curr.shape[0]).tolist() #TODO: x one step in front of controls
    #         u_curr = u_curr.reshape(u_curr.shape[0]).tolist()

    #         x_all.append(x_curr)
    #         u_all.append(u_curr)
    #         theta_all.append(theta.copy())
    #         theta_hat_all.append(theta_hat.copy())

    #     return x_all, u_all, theta_all, theta_hat_all
    
    # def simulate_quadrotor_hover_with_EKF(self, NSIM: int =200): #TODO: add RLS parameters here!
    #     # initialize quadrotor parameters
    #     np.set_printoptions(suppress=False, precision=10)
    #     Ixx, Iyy, Izz = self.quadrotor.J[0,0], self.quadrotor.J[1,1], self.quadrotor.J[2,2]
    #     Ixy, Ixz, Iyz = self.quadrotor.J[0,1], self.quadrotor.J[0,2], self.quadrotor.J[1,2]
    #     # theta = np.array([self.quadrotor.mass,Ixx,Ixy,Ixz,Iyy,Iyz,Izz])
    #     theta = np.array([Ixx,Ixy,Ixz,Iyy,Iyz,Izz])
    #     theta_hat = theta.copy()
        
    #     # get tasks goals
    #     x_nom, u_nom = self.quadrotor.get_hover_goals(theta)
    #     Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta)
    #     Q, R = self.quadrotor_controller.get_QR_bryson()
    #     self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

    #     # randomly perturb initial state
    #     x0 = np.copy(x_nom)
    #     x0[0:3] += np.array([0.2, 0.2, -0.2])  # disturbed initial position
    #     x0[3:7] = self.quadrotor.rptoq(np.array([1.0, 0.0, 0.0]))  # disturbed initial attitude #TODO: move math methods to utilities class
    #     print("Perturbed Intitial State: ")
    #     print(x0)

    #     # get initial control and state
    #     u_curr = self.quadrotor_controller.compute(x0, x_nom, u_nom)
    #     x_curr = np.copy(x0)

    #     x_all = []
    #     u_all = []
    #     theta_all = []
    #     theta_hat_all = []

    #     changing_steps = [50]#np.random.choice(range(20,180),size=2, replace=False)

      
    #     n = 10
    #     A_tot = np.zeros((3*n,6))
    #     b_tot = np.zeros((3*n,1))
    #     Q_ekf = 1e-4 * np.eye(6)
    #     R_ekf = 1e-4 * np.eye(3*n)

    #     ekf = EKF(num_params=6, process_noise=Q_ekf, measurement_noise=R_ekf)
    #     # simulate the dynamics with the LQR controller
    #     for i in range(NSIM):
    #         # noising the parameters
    #         # if i % 10 == 0:
    #         #     theta += np.random.normal(0, process_noise_std, size=theta.shape)

    #         # Change system parameters at specific step
    #         if i in changing_steps:
    #             update_scale = np.random.uniform(2, 4)
    #             theta[0] *= update_scale

    #         # update goals
    #         x_nom, u_nom = self.quadrotor.get_hover_goals(theta)
    #         Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta)
    #         self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

    #         # compute controls
    #         u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)                    # at t=k

    #         # step
    #         process_noise_std = 0.05 * np.abs(x_curr)  # Process noise

    #         x_curr = self.quadrotor.quad_dynamics_rk4(x_curr, u_curr, theta)+ np.random.normal(0, process_noise_std, size=(13,))       # at t=k+1
    #         # formulate measurement model
    #         A = self.quadrotor.get_data_matrix(x_curr, self.quadrotor.quad_dynamics(x_curr, u_curr, theta))
    #         b = self.quadrotor.get_force_vector(x_curr, u_curr, theta)
    #         measurement_noise_std = 0.05 * np.abs(b)
    #         b += np.random.normal(0, measurement_noise_std, size = b.shape)

    #         A_tot = np.roll(A_tot, shift=3, axis=0)
    #         A_tot[:3] = A
    #         b_tot = np.roll(b_tot, shift=3, axis=0)
    #         b_tot[:3] = b

    #         if i % 5 == 0 and i > 0:
    #             theta_hat = ekf.iterate(A_tot,b_tot).reshape(-1,)


    #         print("step: ", i, "\n", 
    #             # "prediction_err: ", np.linalg.norm(theta-theta_hat)/7, "\n"
    #               )


    #         x_curr = x_curr.reshape(x_curr.shape[0]).tolist() #TODO: x one step in front of controls
    #         u_curr = u_curr.reshape(u_curr.shape[0]).tolist()

    #         x_all.append(x_curr)
    #         u_all.append(u_curr)
    #         theta_all.append(theta.copy())
    #         theta_hat_all.append(theta_hat.copy())

    #     return x_all, u_all, theta_all, theta_hat_all

    def simulate_quadrotor_hover_with_DEKA(self, NSIM: int =300): #TODO: add RLS parameters here!
        # np.random.seed(42)
        self.quadrotor = Quadrotor()
        self.quadrotor_controller = LQRController(self.quadrotor.delta_x_quat)

        # initialize quadrotor parameters
        np.set_printoptions(suppress=False, precision=10)
        Ixx, Iyy, Izz = self.quadrotor.J[0,0], self.quadrotor.J[1,1], self.quadrotor.J[2,2]
        Ixy, Ixz, Iyz = self.quadrotor.J[0,1], self.quadrotor.J[0,2], self.quadrotor.J[1,2]
        # theta = np.array([self.quadrotor.mass,Ixx,Ixy,Ixz,Iyy,Iyz,Izz])
      #  print(Ixx)
        mass = np.copy(self.quadrotor.mass)
        theta = np.array([mass,Ixx,Ixy,Ixz,Iyy,Iyz,Izz])
        theta_hat = theta.copy()
       # print("DEKA theta: ", theta)
        # get tasks goals
        x_nom, u_nom = self.quadrotor.get_hover_goals(theta.copy())
        Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat)
        Q, R = self.quadrotor_controller.get_QR_bryson()
        self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

        # randomly perturb initial state
        x0 = np.copy(x_nom)
        x0[0:3] += np.array([0.1, -0.15, -0.1])  # disturbed initial position
        x0[3:7] = self.quadrotor.rptoq(np.array([0.3, 0.4, 0.2]))  # disturbed initial attitude #TODO: move math methods to utilities class
       # print("Perturbed Intitial State: ")
      #  print(x0)

        # get initial control and state
        u_curr = self.quadrotor_controller.compute(x0, x_nom, u_nom)
        x_curr = np.copy(x0)

        x_all = []
        u_all = []
        theta_all = []
        theta_hat_all = []

        changing_steps = [100]#np.random.choice(range(20,180),size=2, replace=False)
 
        deka = DEKA(num_params=7, x0=theta.reshape(-1,1), damping=0.7, regularization=1e-15, smoothing_factor=0.2, tol_max=1e-5, tol_min=0.01)

        n = 10

        A_tot = np.zeros((6*n,7))
        b_tot = np.zeros((6*n,1))

        # simulate the dynamics with the LQR controller
        for i in range(NSIM):
            # noising the parameters
            # if i % 10 == 0:
            #     theta += np.random.normal(0, process_noise_std, size=theta.shape)

            # Change system parameters at specific step
            if i in changing_steps:
                update_scale = np.random.uniform(2, 3)
                theta = np.array([mass*update_scale,Ixx,Ixy,Ixz,Iyy,Iyz,Izz])

            # update goals
            x_nom, u_nom = self.quadrotor.get_hover_goals(theta_hat.copy())
            Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat.copy())
            self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

            # compute controls
            u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)                    # at t=k

            # step
            if i > 0 and i % 15 == 0:
                process_noise_std = 0.1 * np.abs(theta[0])  # Process noise
                theta[0] += np.random.normal(0, process_noise_std)
            x_curr = self.quadrotor.quad_dynamics_rk4(x_curr, u_curr, theta)    # at t=k+1
            # formulate measurement model
            A = self.quadrotor.get_data_matrix(x_curr, self.quadrotor.quad_dynamics(x_curr, u_curr, theta))
            b = self.quadrotor.get_force_vector(x_curr, u_curr, theta)
            measurement_noise_std = 0.05 * np.abs(b)
            b += np.random.normal(0, measurement_noise_std, size = b.shape)

            # print(A@(theta.reshape(-1,1)))
            # print(b)
            # import pdb; pdb.set_trace()

            A_tot = np.roll(A_tot, shift=3, axis=0)
            A_tot[:6] = A
            b_tot = np.roll(b_tot, shift=3, axis=0)
            b_tot[:6] = b

            if i % 5 == 0 and i > 0:
                # print("step", i)
                theta_hat_prev = theta_hat
                theta_hat = deka.iterate(A_tot,b_tot, x_0=theta_hat_prev.reshape(-1,1),num_iterations=int(9/6*(n**2)))[0].reshape(-1,)
                
                if (np.linalg.norm(theta_hat - theta_hat_prev) / np.linalg.norm(theta_hat)) > 0.2:
                    theta_hat = 0.8 * theta_hat + 0.2 * theta_hat_prev
                    # print(f"smoothed at step {i} with value {(np.linalg.norm(theta_hat - theta_hat_prev) / np.linalg.norm(theta_hat))}: ")
                # print("theta: ", theta, "\n"
                #     "theta_hat", theta_hat, "\n",
                #     )
            # print(A@(theta.reshape(-1,1)))
            # print(b)




            x_curr = x_curr.reshape(x_curr.shape[0]).tolist() #TODO: x one step in front of controls
            u_curr = u_curr.reshape(u_curr.shape[0]).tolist()

            x_all.append(x_curr)
            u_all.append(u_curr)
            theta_all.append(theta.copy())
            theta_hat_all.append(theta_hat.copy())

        return x_all, u_all, theta_all, theta_hat_all
    
    # def simulate_quadrotor_tracking_with_RLS(self, NSIM: int =200): 
    #     np.random.seed(0) 
    #     # initialize quadrotor parameters
    #     np.set_printoptions(suppress=False, precision=10)
    #     Ixx, Iyy, Izz = self.quadrotor.J[0,0], self.quadrotor.J[1,1], self.quadrotor.J[2,2]
    #     Ixy, Ixz, Iyz = self.quadrotor.J[0,1], self.quadrotor.J[0,2], self.quadrotor.J[1,2]
    #     theta = np.array([self.quadrotor.mass,Ixx,Ixy,Ixz,Iyy,Iyz,Izz])
    #     theta_hat = theta.copy()
        
    #     self.quadrotor.rg = np.array([0.0, 0, 0.0])
    #     self.quadrotor.qg = np.array([1.0, 0, 0, 0])
    #     self.quadrotor.vg = np.zeros(3)
    #     self.quadrotor.omgg = np.zeros(3)
    #     x_nom_lower = np.hstack([self.quadrotor.qg, self.quadrotor.vg, self.quadrotor.omgg])
    #     num_points = 600
    #     angles = np.linspace(0, 6*np.pi, num_points, endpoint=False)
    #     # Create the figure-8 in the XY-plane
    #     x = np.sin(angles)
    #     y = np.sin(2 * angles)
    #     z = np.zeros_like(x)  # Initially flat in XY-plane

    #     # Stack as a (3, num_points) matrix
    #     traj = np.vstack((x, y, z))  # Shape: (3, num_points)

    #     # Define rotation matrix (tilt around the Y-axis)
    #     rot_ang= np.radians(30)  # Tilt angle in degrees
    #     R = np.array([
    #         [np.cos(rot_ang), 0, np.sin(rot_ang)],  # Rotation matrix for X-Z plane
    #         [0, 1, 0],  # Keep Y unchanged
    #         [-np.sin(rot_ang), 0, np.cos(rot_ang)]
    #     ])

    #     # Apply rotation
    #     traj_rotated = (R @ traj).T  # Matrix multiplication
    #     traj = traj.T  # Transpose back to original shape

    #     # get tasks goals
    #     x_nom = np.hstack([traj[0], x_nom_lower])
    #     # print("x_nom: ", x_nom)
    #     u_nom = (theta[0]*self.quadrotor.g/self.quadrotor.kt/4)*np.ones(4)
    #     Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat)
    #     Q, R = self.quadrotor_controller.get_QR_bryson()
    #     self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

    #     # randomly perturb initial state
    #     x0 = np.copy(x_nom)
    #    # x0[0:3] += np.array([0.2, 0.2, -0.2])  # disturbed initial position
    #     x0[3:7] = self.quadrotor.rptoq(np.array([1.0, 0.0, 0.0]))  # disturbed initial attitude #TODO: move math methods to utilities class
    #     print("Perturbed Intitial State: ")
    #     print(x0)

    #     # get initial control and state
    #     u_curr = self.quadrotor_controller.compute(x0, x_nom, u_nom)
    #     x_curr = np.copy(x0)

    #     x_all = []
    #     u_all = []
    #     theta_all = []
    #     theta_hat_all = []

    #     changing_steps = [50]#np.random.choice(range(20,180),size=2, replace=False)
 
    #     rls = RLS(num_params=7)

    #     # simulate the dynamics with the LQR controller
    #     for i in range(NSIM):
    #         # processing noise
            
    #         # Change system parameters at specific step
    #         if i in changing_steps:
    #             update_scale = np.random.uniform(1, 2, size = theta.shape)
    #             theta *= update_scale

    #         # update goals
    #         x_nom = np.hstack([traj[i], x_nom_lower])
    #         # print("x_nom: ", x_nom)
    #         u_nom = (theta[0]*self.quadrotor.g/self.quadrotor.kt/4)*np.ones(4)
    #         Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat)
    #         self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

    #         # compute controls
    #         u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)                    # at t=k

    #         # step
    #         x_curr = self.quadrotor.quad_dynamics_rk4(x_curr, u_curr, theta)       # at t=k+1
            
    #         process_noise_std = 0.001 * np.abs(x_curr)  # Process noise
    #         x_curr = self.quadrotor.quad_dynamics_rk4(x_curr, u_curr, theta) + np.random.normal(0, process_noise_std, size=(13,))  # at t=k+1
    #         # measurement noise
    #         measurement_noise_std = 0.05 * np.abs(x_curr)
    #         x_curr += np.random.normal(0, measurement_noise_std, size = x_curr.shape)

    #         # formulate measurement model
    #         A = self.quadrotor.get_data_matrix(x_curr, self.quadrotor.quad_dynamics(x_curr, u_curr, theta))
    #         b = self.quadrotor.get_force_vector(x_curr, u_curr, theta)
            
    #         # import pdb; pdb.set_trace()
    #         theta_hat = rls.iterate(A,b).reshape(-1,)
    #         print("step: ", i, "\n", 
    #             "prediction_err: ", np.linalg.norm(theta-theta_hat)/7, "\n"
    #               )


    #         x_curr = x_curr.reshape(x_curr.shape[0]).tolist() #TODO: x one step in front of controls
    #         u_curr = u_curr.reshape(u_curr.shape[0]).tolist()

    #         x_all.append(x_curr)
    #         u_all.append(u_curr)
    #         theta_all.append(theta.copy())
    #         theta_hat_all.append(theta_hat.copy())

    #     return x_all, u_all, theta_all, theta_hat_all

    # def simulate_quadrotor_tracking_with_EKF(self, NSIM: int =200): 
    #     np.random.seed(0) 
    #     # initialize quadrotor parameters
    #     np.set_printoptions(suppress=False, precision=10)
    #     Ixx, Iyy, Izz = self.quadrotor.J[0,0], self.quadrotor.J[1,1], self.quadrotor.J[2,2]
    #     Ixy, Ixz, Iyz = self.quadrotor.J[0,1], self.quadrotor.J[0,2], self.quadrotor.J[1,2]
    #     theta = np.array([self.quadrotor.mass,Ixx,Ixy,Ixz,Iyy,Iyz,Izz])
    #     theta_hat = theta.copy()
        
    #     self.quadrotor.rg = np.array([0.0, 0, 0.0])
    #     self.quadrotor.qg = np.array([1.0, 0, 0, 0])
    #     self.quadrotor.vg = np.zeros(3)
    #     self.quadrotor.omgg = np.zeros(3)
    #     x_nom_lower = np.hstack([self.quadrotor.qg, self.quadrotor.vg, self.quadrotor.omgg])
    #     num_points = 600
    #     angles = np.linspace(0, 6*np.pi, num_points, endpoint=False)
    #     # Create the figure-8 in the XY-plane
    #     x = np.sin(angles)
    #     y = np.sin(2 * angles)
    #     z = np.zeros_like(x)  # Initially flat in XY-plane

    #     # Stack as a (3, num_points) matrix
    #     traj = np.vstack((x, y, z))  # Shape: (3, num_points)

    #     # Define rotation matrix (tilt around the Y-axis)
    #     rot_ang= np.radians(30)  # Tilt angle in degrees
    #     R = np.array([
    #         [np.cos(rot_ang), 0, np.sin(rot_ang)],  # Rotation matrix for X-Z plane
    #         [0, 1, 0],  # Keep Y unchanged
    #         [-np.sin(rot_ang), 0, np.cos(rot_ang)]
    #     ])

    #     # Apply rotation
    #     traj_rotated = (R @ traj).T  # Matrix multiplication
    #     traj = traj.T  # Transpose back to original shape

    #     # get tasks goals
    #     x_nom = np.hstack([traj[0], x_nom_lower])
    #     # print("x_nom: ", x_nom)
    #     u_nom = (theta[0]*self.quadrotor.g/self.quadrotor.kt/4)*np.ones(4)
    #     Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat)
    #     Q, R = self.quadrotor_controller.get_QR_bryson()
    #     self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

    #     # randomly perturb initial state
    #     x0 = np.copy(x_nom)
    #    # x0[0:3] += np.array([0.2, 0.2, -0.2])  # disturbed initial position
    #     x0[3:7] = self.quadrotor.rptoq(np.array([1.0, 0.0, 0.0]))  # disturbed initial attitude #TODO: move math methods to utilities class
    #     print("Perturbed Intitial State: ")
    #     print(x0)

    #     # get initial control and state
    #     u_curr = self.quadrotor_controller.compute(x0, x_nom, u_nom)
    #     x_curr = np.copy(x0)

    #     x_all = []
    #     u_all = []
    #     theta_all = []
    #     theta_hat_all = []

    #     changing_steps = [50]#np.random.choice(range(20,180),size=2, replace=False)
    #     Q_ekf = 1e-4 * np.eye(len(theta))
    #     R_ekf = 1e-4 * np.eye(6)


    #     ekf = EKF(num_params=7, process_noise=Q_ekf, measurement_noise=R_ekf)

    #     # simulate the dynamics with the LQR controller
    #     for i in range(NSIM):      
    #         # Change system parameters at specific step
    #         if i in changing_steps:
    #             update_scale = np.random.uniform(1, 2, size = theta.shape)
    #             theta *= update_scale

    #         # update goals
    #         x_nom = np.hstack([traj[i], x_nom_lower])
    #         # print("x_nom: ", x_nom)
    #         u_nom = (theta[0]*self.quadrotor.g/self.quadrotor.kt/4)*np.ones(4)
    #         Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat)
    #         self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

    #         # compute controls
    #         u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)                    # at t=k

    #         # step
    #         process_noise_std = 0.001 * np.abs(x_curr)  # Process noise
    #         x_curr = self.quadrotor.quad_dynamics_rk4(x_curr, u_curr, theta) + np.random.normal(0, process_noise_std, size=(13,))  # at t=k+1
    #         # measurement noise
    #         measurement_noise_std = 0.05 * np.abs(x_curr)
    #         x_curr += np.random.normal(0, measurement_noise_std, size = x_curr.shape)

    #         # formulate measurement model
    #         A = self.quadrotor.get_data_matrix(x_curr, self.quadrotor.quad_dynamics(x_curr, u_curr, theta))
    #         b = self.quadrotor.get_force_vector(x_curr, u_curr, theta)
            
    #         # import pdb; pdb.set_trace()
    #         theta_hat = ekf.iterate(A,b).reshape(-1,)
    #         print("step: ", i, "\n", 
    #             "prediction_err: ", np.linalg.norm(theta-theta_hat)/7, "\n"
    #               )


    #         x_curr = x_curr.reshape(x_curr.shape[0]).tolist() #TODO: x one step in front of controls
    #         u_curr = u_curr.reshape(u_curr.shape[0]).tolist()

    #         x_all.append(x_curr)
    #         u_all.append(u_curr)
    #         theta_all.append(theta.copy())
    #         theta_hat_all.append(theta_hat.copy())

    #     return x_all, u_all, theta_all, theta_hat_all

    # def simulate_quadrotor_tracking_with_DEKA(self, NSIM: int =200): 
    #     # initialize quadrotor parameters
    #     Ixx, Iyy, Izz = self.quadrotor.J[0,0], self.quadrotor.J[1,1], self.quadrotor.J[2,2]
    #     Ixy, Ixz, Iyz = self.quadrotor.J[0,1], self.quadrotor.J[0,2], self.quadrotor.J[1,2]
    #     theta = np.array([self.quadrotor.mass,Ixx,Ixy,Ixz,Iyy,Iyz,Izz])
    #     theta_hat = theta.copy()
        
    #     self.quadrotor.rg = np.array([0.0, 0, 0.0])
    #     self.quadrotor.qg = np.array([1.0, 0, 0, 0])
    #     self.quadrotor.vg = np.zeros(3)
    #     self.quadrotor.omgg = np.zeros(3)
    #     x_nom_lower = np.hstack([self.quadrotor.qg, self.quadrotor.vg, self.quadrotor.omgg])
    #     num_points = 600
    #     angles = np.linspace(0, 6*np.pi, num_points, endpoint=False)
    #     # Create the figure-8 in the XY-plane
    #     x = np.sin(angles)
    #     y = np.sin(2 * angles)
    #     z = np.zeros_like(x)  # Initially flat in XY-plane

    #     # Stack as a (3, num_points) matrix
    #     traj = np.vstack((x, y, z))  # Shape: (3, num_points)

    #     # Define rotation matrix (tilt around the Y-axis)
    #     rot_ang= np.radians(30)  # Tilt angle in degrees
    #     R = np.array([
    #         [np.cos(rot_ang), 0, np.sin(rot_ang)],  # Rotation matrix for X-Z plane
    #         [0, 1, 0],  # Keep Y unchanged
    #         [-np.sin(rot_ang), 0, np.cos(rot_ang)]
    #     ])

    #     # Apply rotation
    #     traj_rotated = (R @ traj).T  # Matrix multiplication
    #     traj = traj.T  # Transpose back to original shape

    #     # get tasks goals
    #     x_nom = np.hstack([traj[0], x_nom_lower])
    #     # print("x_nom: ", x_nom)
    #     u_nom = (theta[0]*self.quadrotor.g/self.quadrotor.kt/4)*np.ones(4)
    #     Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat)
    #     Q, R = self.quadrotor_controller.get_QR_bryson()
    #     self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

    #     # randomly perturb initial state
    #     x0 = np.copy(x_nom)
    #     #x0[0:3] += np.array([0.2, 0.2, -0.2])  # disturbed initial position
    #     x0[3:7] = self.quadrotor.rptoq(np.array([1.0, 0.0, 0.0]))  # disturbed initial attitude #TODO: move math methods to utilities class
    #     print("Perturbed Intitial State: ")
    #     print(x0)

    #     # get initial control and state
    #     u_curr = self.quadrotor_controller.compute(x0, x_nom, u_nom)
    #     x_curr = np.copy(x0)

    #     x_all = []
    #     u_all = []
    #     theta_all = []
    #     theta_hat_all = []

    #     changing_steps = [50, 100, 150]#np.random.choice(range(20,180),size=2, replace=False)
 
    #     deka = DEKA_new(num_params=7,x0=theta_hat, damping=1.0, smoothing_factor=0.0)

    #     # simulate the dynamics with the LQR controller
    #     for i in range(NSIM):
    #         # Change system parameters at specific step
    #         if i in changing_steps:
    #             update_scale = np.random.uniform(1, 1.5)
    #             theta *= update_scale

    #         # update goals
    #         x_nom = np.hstack([traj[i], x_nom_lower])
    #        # print("x_nom: ", x_nom)
    #         u_nom = (theta[0]*self.quadrotor.g/self.quadrotor.kt/4)*np.ones(4)
    #         # theta_hat[0] *= 1e5
    #         Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat)
    #         self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

    #         # compute controls
    #         u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)                    # at t=k

    #         # step
    #         process_noise_std = 0.001 * np.abs(x_curr)  # Process noise
    #         x_curr = self.quadrotor.quad_dynamics_rk4(x_curr, u_curr, theta)# + np.random.normal(0, process_noise_std, size=(13,))  # at t=k+1
            
    #         # measurement noise
    #         measurement_noise_std = 0.05 * np.abs(x_curr)
    #         x_curr += np.random.normal(0, measurement_noise_std, size = x_curr.shape)
    #         # formulate measurement model
    #         A = self.quadrotor.get_data_matrix(x_curr, self.quadrotor.quad_dynamics(x_curr, u_curr, theta))
    #         # A[0] *= 1e5
    #         b = self.quadrotor.get_force_vector(x_curr, u_curr, theta)

    #         # import pdb; pdb.set_trace()
    #         print(A@(theta.reshape(-1,1)))
    #         print(b)
    #         import pdb; pdb.set_trace()

    #         # if i % 2 == 0:
    #         #     measurement_noise_std = 0.005 * np.abs(b)
    #         #     b += np.random.normal(0, measurement_noise_std, size=b.shape)
            
    #         # import pdb; pdb.set_trace()
    #         #print(deka.iterate(A,b)[0])
    #         theta_hat = deka.iterate(A,b, num_iterations=1000, tol=1e-3)[0].reshape(-1,)
    #         # theta_hat[0] *= 1e-5

    #         #np.linalg.lstsq(A,b)[0].reshape(-1,)
            

    #         print("step: ", i, "\n", 
    #             # "prediction_err: ", np.linalg.norm(theta-theta_hat)/7, "\n"
    #               )


    #         x_curr = x_curr.reshape(x_curr.shape[0]).tolist() #TODO: x one step in front of controls
    #         u_curr = u_curr.reshape(u_curr.shape[0]).tolist()

    #         x_all.append(x_curr.copy())
    #         u_all.append(u_curr)
    #         theta_all.append(theta.copy())
    #         theta_hat_all.append(theta_hat.copy())

    #     return x_all, u_all, theta_all, theta_hat_all

    # def simulate_quadrotor_hover_with_Adaptive_RLS(self, NSIM: int =200): #TODO: add RLS parameters here!
    #     # initialize quadrotor parameters
    #     theta = np.array([self.quadrotor.mass,self.quadrotor.g])
    #     theta_hat = np.copy(theta)

    #     # get tasks goals
    #     x_nom, u_nom = self.quadrotor.get_hover_goals(theta[0], theta[1], self.quadrotor.kt)
    #     check_grads(self.quadrotor.quad_dynamics_rk4, modes=['rev'], order=2)(x_nom, u_nom, self.quadrotor.mass, self.quadrotor.g)
    #     Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta[0], theta[1])
    #     Q, R = self.quadrotor_controller.get_QR_bryson()
    #     self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

    #     # randomly perturb initial state
    #     x0 = np.copy(x_nom)
    #     x0[0:3] += np.array([0.2, 0.2, -0.2])  # disturbed initial position
    #     x0[3:7] = self.quadrotor.rptoq(np.array([1.0, 0.0, 0.0]))  # disturbed initial attitude #TODO: move math methods to utilities class
    #     print("Perturbed Intitial State: ")
    #     print(x0)

    #     # get initial control and state
    #     u_curr = self.quadrotor_controller.compute(x0, x_nom, u_nom)
    #     x_curr = np.copy(x0)

    #     x_all = []
    #     u_all = []
    #     theta_all = []
    #     theta_hat_all = []

    #     changing_steps = np.random.choice(range(20,180),size=2, replace=False)
    #     process_noise_std=np.zeros((13,1))+0.0001

    #     df_dm = jacobian(self.quadrotor.quad_dynamics_rk4, 2)
    #     df_dg = jacobian(self.quadrotor.quad_dynamics_rk4, 3)

    #     rls = AdaptiveLambdaRLS(num_params=2)

    #     # simulate the dynamics with the LQR controller
    #     for i in range(NSIM):
    #         # noising the parameters
    #         # if i % 10 == 0:
    #         #     theta += np.random.normal(0, process_noise_std, size=theta.shape)

    #         # Change system parameters at specific step
    #         if i in changing_steps:
    #             mass_update_scale = np.random.uniform(0.8, 1.3)
    #             gravity_update_scale = np.random.uniform(0.8, 1.3)
    #             theta = np.array([self.quadrotor.mass * mass_update_scale, self.quadrotor.g * gravity_update_scale])

    #         # update goals
    #         x_nom, u_nom = self.quadrotor.get_hover_goals(theta_hat[0], theta_hat[1], self.quadrotor.kt) #TODO: use moving ave filter?
    #         Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat[0], theta_hat[1])
    #         self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

    #         # compute controls
            
    #         u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)                    # at t=k

    #         # step
    #         x_prev = x_curr                                                                     # at t=k
    #         x_curr = self.quadrotor.quad_dynamics_rk4(x_prev, u_curr, theta[0], theta[1])       # at t=k+1

    #         # estimate parameters
    #         theta_hat_prev = theta_hat                                                          # at t=k
    #         A = np.column_stack((df_dm(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1]),  # (num_states, num_params)
    #                              df_dg(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1])))
    #         f_at_theta_prev = self.quadrotor.quad_dynamics_rk4(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1]) # (num_states, 1)
    #         b = (x_curr - f_at_theta_prev).reshape(-1,1) # (num_states, 1)
            
    #         if i % 10 == 0:
    #             b += np.random.normal(0, process_noise_std, size=b.shape)

    #         # print("normed residual at step: ", i, " is " np.linalg.norm(df_dtheta_at_theta_prev @ theta_hat- b))
    #         # if i == 0:
    #         #     rls.initialize(df_dtheta_at_theta_prev, b)
    #         estimate = rls.iterate(A, b)
    #         delta_theta_hat = np.copy(estimate.reshape(2))                                       # at t=k+1
            
    #         # using LSQ SOLN
    #         # delta_theta_hat = np.linalg.lstsq(A,b)[0].reshape(2)
    #         theta_hat = delta_theta_hat + theta_hat_prev

    #         print("step: ", i, "\n", 
    #               "u_k: ", u_curr, "\n", 
    #               "x_k: ", x_prev[:3], "\n", 
    #               "theta_k: ", theta, "\n", 
    #               "theta_hat_k: ", theta_hat, "\n",
    #               "delta_theta_hat", delta_theta_hat, "\n",
    #             #   "x_k+1: ", x_curr[:3], 
    #               )
    #         # print("A: ", A)
    #         # print("b: ", b)

    #         x_curr = x_curr.reshape(x_curr.shape[0]).tolist() #TODO: x one step in front of controls
    #         u_curr = u_curr.reshape(u_curr.shape[0]).tolist()

    #         x_all.append(x_curr)
    #         u_all.append(u_curr)
    #         theta_all.append(theta)
    #         theta_hat_all.append(theta_hat)

    #     return x_all, u_all, theta_all, theta_hat_all
    
    # def simulate_quadrotor_hover_with_EKF(self, NSIM:int=200): #TODO: add RLS parameters here!
    #     # initialize quadrotor parameters
    #     theta = np.array([self.quadrotor.mass,self.quadrotor.g])
    #     theta_hat = np.copy(theta)

    #     # get tasks goals
    #     x_nom, u_nom = self.quadrotor.get_hover_goals(theta[0], theta[1], self.quadrotor.kt)
    #     check_grads(self.quadrotor.quad_dynamics_rk4, modes=['rev'], order=2)(x_nom, u_nom, self.quadrotor.mass, self.quadrotor.g)
    #     Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta[0], theta[1])
    #     Q, R = self.quadrotor_controller.get_QR_bryson()
    #     self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

    #     # randomly perturb initial state
    #     x0 = np.copy(x_nom)
    #     x0[0:3] += np.array([0.2, 0.2, -0.2])  # disturbed initial position
    #     x0[3:7] = self.quadrotor.rptoq(np.array([1.0, 0.0, 0.0]))  # disturbed initial attitude #TODO: move math methods to utilities class
    #     print("Perturbed Intitial State: ")
    #     print(x0)

    #     # get initial control and state
    #     u_curr = self.quadrotor_controller.compute(x0, x_nom, u_nom)
    #     x_curr = np.copy(x0)

    #     x_all = []
    #     u_all = []
    #     theta_all = []
    #     theta_hat_all = []

    #     changing_steps = np.random.choice(range(20,180),size=2, replace=False)
    #     process_noise_std=np.zeros((13,1))+0.0001

    #     df_dm = jacobian(self.quadrotor.quad_dynamics_rk4, 2)
    #     df_dg = jacobian(self.quadrotor.quad_dynamics_rk4, 3)

    #     ekf = EKF(num_params=2, process_noise=np.eye(2)*0.005 ** 2, measurement_noise=np.eye(13)*0.005 ** 2)

    #     # simulate the dynamics with the LQR controller
    #     for i in range(NSIM):
    #         # noising the parameters
    #         # if i % 10 == 0:
    #         #     theta += np.random.normal(0, process_noise_std, size=theta.shape)

    #         # Change system parameters at specific step
    #         if i in changing_steps:
    #             mass_update_scale = np.random.uniform(0.8, 1.3)
    #             gravity_update_scale = np.random.uniform(0.8, 1.3)
    #             theta = np.array([self.quadrotor.mass * mass_update_scale, self.quadrotor.g * gravity_update_scale])

    #         # update goals
    #         x_nom, u_nom = self.quadrotor.get_hover_goals(theta_hat[0], theta_hat[1], self.quadrotor.kt) #TODO: use moving ave filter?
    #         Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat[0], theta_hat[1])
    #         self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

    #         # compute controls
            
    #         u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)                    # at t=k

    #         # step
    #         x_prev = x_curr                                                                     # at t=k
    #         x_curr = self.quadrotor.quad_dynamics_rk4(x_prev, u_curr, theta[0], theta[1])       # at t=k+1

    #         # estimate parameters
    #         theta_hat_prev = theta_hat                                                          # at t=k
    #         A = np.column_stack((df_dm(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1]),  # (num_states, num_params)
    #                              df_dg(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1])))
    #         f_at_theta_prev = self.quadrotor.quad_dynamics_rk4(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1]) # (num_states, 1)
    #         b = (x_curr - f_at_theta_prev).reshape(-1,1) # (num_states, 1)
            
    #         if i % 10 == 0:
    #             b += np.random.normal(0, process_noise_std, size=b.shape)

    #         estimate = ekf.iterate(A, b)
    #         delta_theta_hat = np.copy(estimate.reshape(2))                                       # at t=k+1
            
    #         # using LSQ SOLN
    #         # delta_theta_hat = np.linalg.lstsq(A,b)[0].reshape(2)
    #         theta_hat = delta_theta_hat + theta_hat_prev

    #         print("step: ", i, "\n", 
    #               "u_k: ", u_curr, "\n", 
    #               "x_k: ", x_prev[:3], "\n", 
    #               "theta_k: ", theta, "\n", 
    #               "theta_hat_k: ", theta_hat, "\n",
    #               "delta_theta_hat", delta_theta_hat, "\n",
    #             #   "x_k+1: ", x_curr[:3], 
    #               )
    #         # print("A: ", A)
    #         # print("b: ", b)

    #         x_curr = x_curr.reshape(x_curr.shape[0]).tolist() #TODO: x one step in front of controls
    #         u_curr = u_curr.reshape(u_curr.shape[0]).tolist()

    #         x_all.append(x_curr)
    #         u_all.append(u_curr)
    #         theta_all.append(theta)
    #         theta_hat_all.append(theta_hat)

    #     return x_all, u_all, theta_all, theta_hat_all
    
    # def simulate_quadrotor_hover_with_RKAS(self, NSIM: int =200): #TODO: add RLS parameters here!
    #     # initialize quadrotor parameters
    #     theta = np.array([self.quadrotor.mass,self.quadrotor.g])
    #     theta_hat = np.copy(theta)

    #     # get tasks goals
    #     x_nom, u_nom = self.quadrotor.get_hover_goals(theta[0], theta[1], self.quadrotor.kt)
    #     check_grads(self.quadrotor.quad_dynamics_rk4, modes=['rev'], order=2)(x_nom, u_nom, self.quadrotor.mass, self.quadrotor.g)
    #     Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta[0], theta[1])
    #     Q, R = self.quadrotor_controller.get_QR_bryson()
    #     self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

    #     # randomly perturb initial state
    #     x0 = np.copy(x_nom)
    #     x0[0:3] += np.array([0.2, 0.2, -0.2])  # disturbed initial position
    #     x0[3:7] = self.quadrotor.rptoq(np.array([1.0, 0.0, 0.0]))  # disturbed initial attitude #TODO: move math methods to utilities class
    #     print("Perturbed Intitial State: ")
    #     print(x0)

    #     # get initial control and state
    #     u_curr = self.quadrotor_controller.compute(x0, x_nom, u_nom)
    #     x_curr = np.copy(x0)

    #     x_all = []
    #     u_all = []
    #     theta_all = []
    #     theta_hat_all = []

    #     changing_steps = np.random.choice(range(20,180),size=2, replace=False)
    #     process_noise_std=np.zeros((13,1))+0.0001

    #     df_dm = jacobian(self.quadrotor.quad_dynamics_rk4, 2)
    #     df_dg = jacobian(self.quadrotor.quad_dynamics_rk4, 3)

    #     rkas = RKAS()

    #     # simulate the dynamics with the LQR controller
    #     for i in range(NSIM):
    #         # noising the parameters
    #         # if i % 10 == 0:
    #         #     theta += np.random.normal(0, process_noise_std, size=theta.shape)

    #         # Change system parameters at specific step
    #         # if i in changing_steps:
    #         #     mass_update_scale = np.random.uniform(0.8, 1.3)
    #         #     gravity_update_scale = np.random.uniform(0.8, 1.3)
    #         #     theta = np.array([self.quadrotor.mass * mass_update_scale, self.quadrotor.g * gravity_update_scale])

    #         # update goals
    #         x_nom, u_nom = self.quadrotor.get_hover_goals(theta_hat[0], theta_hat[1], self.quadrotor.kt) #TODO: use moving ave filter?
    #         Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat[0], theta_hat[1])
    #         self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

    #         # compute controls
            
    #         u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)                    # at t=k

    #         # step
    #         x_prev = x_curr                                                                     # at t=k
    #         x_curr = self.quadrotor.quad_dynamics_rk4(x_prev, u_curr, theta[0], theta[1])       # at t=k+1

    #         # estimate parameters
    #         theta_hat_prev = theta_hat                                                          # at t=k
    #         A = np.column_stack((df_dm(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1]),  # (num_states, num_params)
    #                              df_dg(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1])))
    #         f_at_theta_prev = self.quadrotor.quad_dynamics_rk4(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1]) # (num_states, 1)
    #         b = (x_curr - f_at_theta_prev).reshape(-1,1) # (num_states, 1)
            
    #         if i % 10 == 0:
    #             b += np.random.normal(0, process_noise_std, size=b.shape)

    #         # print("normed residual at step: ", i, " is " np.linalg.norm(df_dtheta_at_theta_prev @ theta_hat- b))
    #         # if i == 0:
    #         #     rls.initialize(df_dtheta_at_theta_prev, b)
            
    #         delta_theta_hat = np.copy(rkas.solve(A, b, theta_hat_prev.reshape(-1,1), tol=1e-2)).reshape(-1,)                                       # at t=k+1
            
    #         # using LSQ SOLN
    #         # delta_theta_hat = np.linalg.lstsq(A,b)[0].reshape(2)
    #         theta_hat = delta_theta_hat + theta_hat_prev

    #         print("step: ", i, "\n", 
    #               "u_k: ", u_curr, "\n", 
    #               "x_k: ", x_prev[:3], "\n", 
    #               "theta_k: ", theta, "\n", 
    #               "theta_hat_k: ", theta_hat, "\n",
    #               "delta_theta_hat", delta_theta_hat, "\n",
    #               )
    #         # print("A: ", A)
    #         # print("b: ", b)

    #         x_curr = x_curr.reshape(x_curr.shape[0]).tolist() #TODO: x one step in front of controls
    #         u_curr = u_curr.reshape(u_curr.shape[0]).tolist()

    #         x_all.append(x_curr)
    #         u_all.append(u_curr)
    #         theta_all.append(theta)
    #         theta_hat_all.append(theta_hat)

    #     return x_all, u_all, theta_all, theta_hat_all
    
    # def simulate_quadrotor_hover_with_KM(self, NSIM: int = 200):
    #     """
    #     Simulates a quadrotor hover maneuver with the Kaczmarz method (KM) for parameter estimation.

    #     Args:
    #         NSIM (int): Number of simulation steps.

    #     Returns:
    #         tuple: x_all, u_all, theta_hat_all, theta_all (lists of states, controls, estimated parameters, and true parameters).
    #     """

    #     theta = np.array([self.quadrotor.mass, self.quadrotor.g])
    #     theta_hat = np.copy(theta)

    #     # get tasks goals
    #     x_nom, u_nom = self.quadrotor.get_hover_goals(theta[0], theta[1], self.quadrotor.kt)
    #     Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta[0], theta[1])
    #     Q, R = self.quadrotor_controller.get_QR_bryson()
    #     self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

    #     # randomly perturb initial state
    #     x0 = np.copy(x_nom)
    #     x0[0:3] += np.array([0.2, 0.2, -0.2])  # disturbed initial position
    #     x0[3:7] = self.quadrotor.rptoq(np.array([1.0, 0.0, 0.0]))  # disturbed initial attitude
    #     print("Perturbed Initial State: ")
    #     print(x0)

    #     # get initial control and state
    #     u_curr = self.quadrotor_controller.compute(x0, x_nom, u_nom)
    #     x_curr = np.copy(x0)
        
    #     x_all = []
    #     u_all = []
    #     theta_all = []
    #     theta_hat_all = []

    #     rk = RK()     

    #     changing_steps = np.random.choice(range(20, 180), size=3, replace=False) # TODO: avoid frequent/consecutive changes

    #     debug: bool = False
    #     debug_history_length = 0

    #     df_dm = jacobian(self.quadrotor.quad_dynamics_rk4, 2)
    #     df_dg = jacobian(self.quadrotor.quad_dynamics_rk4, 3)

    #     # Simulation loop
    #     for i in range(NSIM):
    #         # Change system parameters at specific step
    #         if i in changing_steps:
    #             mass_update_scale = np.random.uniform(1, 2)
    #             gravity_update_scale = np.random.uniform(1, 2)
    #             theta = np.array([self.quadrotor.mass * mass_update_scale, self.quadrotor.g * gravity_update_scale])
    #             # debug_history_length = 4

    #         # update goals based on the estimated states
    #         x_nom, u_nom = self.quadrotor.get_hover_goals(theta_hat[0], theta_hat[1], self.quadrotor.kt)
    #         Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat[0], theta_hat[1])
    #         self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)
            
    #         # compute controls
    #         u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)

    #         # step
    #         x_prev = x_curr  # at t=k
    #         x_curr = self.quadrotor.quad_dynamics_rk4(x_prev, u_curr, theta[0], theta[1])

    #         # estimate parameters
    #         theta_hat_prev = theta_hat                                                          # at t=k
    #         A = np.column_stack((df_dm(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1]),  # (num_states, num_params)
    #                              df_dg(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1])))
    #         f_at_theta_prev = self.quadrotor.quad_dynamics_rk4(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1]) # (num_states, 1)
    #         b = x_curr - f_at_theta_prev # (num_states, 1)

    #         theta_km = rk.iterate(A, b, theta_hat_prev.reshape(-1,1), 10000, 0.01).reshape(2) + theta_hat_prev
    #         delta_theta_hat = np.linalg.lstsq(A, b)[0]
    #         theta_hat = delta_theta_hat + theta_hat_prev

    #         if debug_history_length>0:
    #             print("A: ", A)
    #             print("b: ", b.T)
    #             print("gt residual: ", np.linalg.norm(np.dot(A,theta)-b))
    #             debug_history_length -= 1

    #         print("step: ", i, "\n",
    #               "u_k: ", u_curr, "\n",
    #               "x_k: ", x_prev[:3], "\n",
    #               "theta_k: ", theta, "\n",
    #               "theta_km: ", theta_km, "\n",
    #               "theta_lstsq: ", theta_hat
    #               )

    #         x_curr = x_curr.reshape(x_curr.shape[0]).tolist()
    #         u_curr = u_curr.reshape(u_curr.shape[0]).tolist()

    #         # Store results
    #         x_all.append(x_curr)
    #         u_all.append(u_curr)
    #         theta_all.append(theta)
    #         theta_hat_all.append(theta_hat)

    #     return x_all, u_all, theta_hat_all, theta_all
    
    # def simulate_quadrotor_hover_with_DEKA(self, NSIM: int =200):
    #     theta = np.array([self.quadrotor.mass,self.quadrotor.g])
    #     theta_hat = np.copy(theta)
    #     delta_theta_hat = np.array([[0.0,0.0]])

    #     # get tasks goals
    #     x_nom, u_nom = self.quadrotor.get_hover_goals(theta[0], theta[1], self.quadrotor.kt)
    #     # check_grads(self.quadrotor.quad_dynamics_rk4, modes=['rev'], order=2)(x_nom, u_nom, self.quadrotor.mass, self.quadrotor.g)
    #     Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta[0], theta[1])
    #     Q, R = self.quadrotor_controller.get_QR_bryson()
    #     self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

    #     # randomly perturb initial state
    #     x0 = np.copy(x_nom)
    #     x0[0:3] += np.array([0.2, 0.2, -0.2])  # disturbed initial position
    #     x0[3:7] = self.quadrotor.rptoq(np.array([1.0, 0.0, 0.0]))  # disturbed initial attitude #TODO: move math methods to utilities class
    #     print("Perturbed Intitial State: ")
    #     print(x0)

    #     # get initial control and state
    #     u_curr = self.quadrotor_controller.compute(x0, x_nom, u_nom)
    #     x_curr = np.copy(x0)

    #     x_all = []
    #     u_all = []
    #     theta_all = []
    #     theta_hat_all = []

    #     changing_steps = [30,100]#np.random.choice(range(20,180),size=1, replace=False)
        

    #     df_dm = jacobian(self.quadrotor.quad_dynamics_rk4, 2)
    #     df_dg = jacobian(self.quadrotor.quad_dynamics_rk4, 3)

    #     deka = DEKA(num_params=2, smoothing_factor=0)

    #     # Priority queue for history (fixed size)
    #     queue_size = 3
    #     A_queue, b_queue = deque(maxlen=queue_size), deque(maxlen=queue_size)

    #     # Keep track of scores for removal
    #     score_queue = deque(maxlen=queue_size)

    #     # Simulation loop
    #     for i in range(NSIM):
    #         # Change system parameters at specific step
    #         if i in changing_steps:
    #             mass_update_scale = np.random.uniform(0.5, 2.)
    #             gravity_update_scale = np.random.uniform(0.9, 1.1)
    #             theta = np.array([self.quadrotor.mass * mass_update_scale, self.quadrotor.g * gravity_update_scale])

    #         # update goals
    #         x_nom, u_nom = self.quadrotor.get_hover_goals(theta_hat[0], theta_hat[1], self.quadrotor.kt)
    #         Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat[0], theta_hat[1])
    #         self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

    #         # compute controls
    #         u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom) 

    #         # step
    #         x_prev = x_curr                                                                     # at t=k
    #         x_curr = self.quadrotor.quad_dynamics_rk4(x_prev, u_curr, theta[0], theta[1]) 

    #         # estimate parameters
    #         theta_hat_prev = theta_hat                                                          # at t=k
    #         A = np.column_stack((df_dm(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1]),  # (num_states, num_params)
    #                              df_dg(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1])))
    #         f_at_theta_prev = self.quadrotor.quad_dynamics_rk4(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1]) # (num_states, 1)
    #         b = x_curr - f_at_theta_prev # (num_states, 1)

    #         # # Add to history queue
    #         # A_queue.append(A)
    #         # b_queue.append(b)

    #         # if i == 30 or i == 100:
    #         #     A_queue.clear()
    #         #     b_queue.clear() 

    #         # lsq_delta = np.linalg.lstsq(A, b)[0]
    #         # if len(A_queue) == queue_size:
    #         #     A_hist = np.vstack(A_queue)  # Shape (13*3, 2)
    #         #     b_hist = np.vstack(b_queue)  # Shape (13*3, 1)

    #         delta_theta_hat_prev = delta_theta_hat
    #         delta_theta_hat = deka.iterate(A.reshape(-1,2), b.reshape(-1,1), delta_theta_hat_prev.reshape(-1,1), 1000, 1e-4).reshape(2)
    #         theta_hat = delta_theta_hat + theta_hat_prev

    #         print("step: ", i, "\n",
    #               "u_k: ", u_curr, "\n",
    #               "x_k: ", x_prev[:3], "\n",
    #               "theta_k: ", theta, "\n",
    #               "theta_lstsq: ", theta_hat, "\n",
    #               "delta_theta: ", delta_theta_hat, "\n"
    #               )
            
    #         x_curr = x_curr.reshape(x_curr.shape[0]).tolist()
    #         u_curr = u_curr.reshape(u_curr.shape[0]).tolist()
            
    #         # Store results
    #         x_all.append(x_curr)
    #         u_all.append(u_curr)
    #         theta_all.append(theta)
    #         theta_hat_all.append(theta_hat)

    #     return x_all, u_all, theta_hat_all, theta_all
    
    # def simulate_quadrotor_hover_with_DEKA(self, NSIM: int =200): #TODO: add RLS parameters here!
    #     # initialize quadrotor parameters
    #     np.set_printoptions(suppress=False, precision=10)
    #     Ixx, Iyy, Izz = self.quadrotor.J[0,0], self.quadrotor.J[1,1], self.quadrotor.J[2,2]
    #     Ixy, Ixz, Iyz = self.quadrotor.J[0,1], self.quadrotor.J[0,2], self.quadrotor.J[1,2]
    #     theta = np.array([self.quadrotor.mass,Ixx,Ixy,Ixz,Iyy,Iyz,Izz])
    #     theta_hat = theta.copy()
        
    #     # get tasks goals
    #     x_nom, u_nom = self.quadrotor.get_hover_goals(theta)
    #     Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta)
    #     Q, R = self.quadrotor_controller.get_QR_bryson()
    #     self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

    #     # randomly perturb initial state
    #     x0 = np.copy(x_nom)
    #     x0[0:3] += np.array([0.2, 0.2, -0.2])  # disturbed initial position
    #     x0[3:7] = self.quadrotor.rptoq(np.array([1.0, 0.0, 0.0]))  # disturbed initial attitude #TODO: move math methods to utilities class
    #     print("Perturbed Intitial State: ")
    #     print(x0)

    #     # get initial control and state
    #     u_curr = self.quadrotor_controller.compute(x0, x_nom, u_nom)
    #     x_curr = np.copy(x0)

    #     x_all = []
    #     u_all = []
    #     theta_all = []
    #     theta_hat_all = []

    #     changing_steps = [50]#np.random.choice(range(20,180),size=2, replace=False)
 
    #     deka = DEKA_new()

    #     # simulate the dynamics with the LQR controller
    #     for i in range(NSIM):
    #         # noising the parameters
    #         # if i % 10 == 0:
    #         #     theta += np.random.normal(0, process_noise_std, size=theta.shape)

    #         # Change system parameters at specific step
    #         if i in changing_steps:
    #             update_scale = np.random.uniform(1, 2)
    #             theta *= update_scale

    #         # update goals
    #         x_nom, u_nom = self.quadrotor.get_hover_goals(theta)
    #         Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta)
    #         self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

    #         # compute controls
    #         u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)                    # at t=k

    #         # step
    #         x_curr = self.quadrotor.quad_dynamics_rk4(x_curr, u_curr, theta)       # at t=k+1
            
    #         # formulate measurement model
    #         A = self.quadrotor.get_data_matrix(x_curr, self.quadrotor.quad_dynamics(x_curr, u_curr, theta))
    #         b = self.quadrotor.get_force_vector(x_curr, u_curr, theta)
            
    #         # import pdb; pdb.set_trace()
    #         theta_hat = rls.iterate(A,b).reshape(-1,)
    #         # theta_hat = np.linalg.lstsq(A, b)[0]

    #         print("step: ", i, "\n", 
    #             "prediction_err: ", np.linalg.norm(theta-theta_hat)/7, "\n"
    #               )


    #         x_curr = x_curr.reshape(x_curr.shape[0]).tolist() #TODO: x one step in front of controls
    #         u_curr = u_curr.reshape(u_curr.shape[0]).tolist()

    #         x_all.append(x_curr)
    #         u_all.append(u_curr)
    #         theta_all.append(theta.copy())
    #         theta_hat_all.append(theta_hat.copy())

    #     return x_all, u_all, theta_all, theta_hat_all
    
    
    # def simulate_quadrotor_hover_with_REK(self, NSIM: int =200):
    #     theta = np.array([self.quadrotor.mass,self.quadrotor.g])
    #     theta_hat = np.copy(theta)
    #     delta_theta_hat = np.array([[0.0,0.0]])

    #     # get tasks goals
    #     x_nom, u_nom = self.quadrotor.get_hover_goals(theta[0], theta[1], self.quadrotor.kt)
    #     # check_grads(self.quadrotor.quad_dynamics_rk4, modes=['rev'], order=2)(x_nom, u_nom, self.quadrotor.mass, self.quadrotor.g)
    #     Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta[0], theta[1])
    #     Q, R = self.quadrotor_controller.get_QR_bryson()
    #     self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

    #     # randomly perturb initial state
    #     x0 = np.copy(x_nom)
    #     x0[0:3] += np.array([0.2, 0.2, -0.2])  # disturbed initial position
    #     x0[3:7] = self.quadrotor.rptoq(np.array([1.0, 0.0, 0.0]))  # disturbed initial attitude #TODO: move math methods to utilities class
    #     print("Perturbed Intitial State: ")
    #     print(x0)

    #     # get initial control and state
    #     u_curr = self.quadrotor_controller.compute(x0, x_nom, u_nom)
    #     x_curr = np.copy(x0)

    #     x_all = []
    #     u_all = []
    #     theta_all = []
    #     theta_hat_all = []

    #     changing_steps = [30,100]#np.random.choice(range(20,180),size=1, replace=False)
        

    #     df_dm = jacobian(self.quadrotor.quad_dynamics_rk4, 2)
    #     df_dg = jacobian(self.quadrotor.quad_dynamics_rk4, 3)

    #     rek = REK()

    #     # Priority queue for history (fixed size)
    #     queue_size = 3
    #     A_queue, b_queue = deque(maxlen=queue_size), deque(maxlen=queue_size)

    #     # Keep track of scores for removal
    #     score_queue = deque(maxlen=queue_size)

    #     # Simulation loop
    #     for i in range(NSIM):
    #         # Change system parameters at specific step
    #         if i in changing_steps:
    #             mass_update_scale = np.random.uniform(1, 3/2)
    #             gravity_update_scale = np.random.uniform(1, 3/2)
    #             theta = np.array([self.quadrotor.mass * mass_update_scale, self.quadrotor.g * gravity_update_scale])

    #         # update goals
    #         x_nom, u_nom = self.quadrotor.get_hover_goals(theta_hat[0], theta_hat[1], self.quadrotor.kt)
    #         Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat[0], theta_hat[1])
    #         self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

    #         # compute controls
    #         u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom) 

    #         # step
    #         x_prev = x_curr                                                                     # at t=k
    #         x_curr = self.quadrotor.quad_dynamics_rk4(x_prev, u_curr, theta[0], theta[1]) 

    #         # estimate parameters
    #         theta_hat_prev = theta_hat                                                          # at t=k
    #         A = np.column_stack((df_dm(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1]),  # (num_states, num_params)
    #                              df_dg(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1])))
    #         f_at_theta_prev = self.quadrotor.quad_dynamics_rk4(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1]) # (num_states, 1)
    #         b = x_curr - f_at_theta_prev # (num_states, 1)

    #         # # Add to history queue
    #         # A_queue.append(A)
    #         # b_queue.append(b)

    #         # if i == 30 or i == 100:
    #         #     A_queue.clear()
    #         #     b_queue.clear() 

    #         # lsq_delta = np.linalg.lstsq(A, b)[0]
    #         # if len(A_queue) == queue_size:
    #         #     A_hist = np.vstack(A_queue)  # Shape (13*3, 2)
    #         #     b_hist = np.vstack(b_queue)  # Shape (13*3, 1)

    #         delta_theta_hat_prev = delta_theta_hat
    #         delta_theta_hat = rek.iterate(A.reshape(-1,2), b.reshape(-1,1), delta_theta_hat_prev.reshape(-1,1), 1000, 1e-5).reshape(2)
    #         theta_hat = delta_theta_hat + theta_hat_prev

    #         print("step: ", i, "\n",
    #               "u_k: ", u_curr, "\n",
    #               "x_k: ", x_prev[:3], "\n",
    #               "theta_k: ", theta, "\n",
    #               "theta_km: ", theta_hat, "\n",
    #               )
            
    #         x_curr = x_curr.reshape(x_curr.shape[0]).tolist()
    #         u_curr = u_curr.reshape(u_curr.shape[0]).tolist()
            
    #         # Store results
    #         x_all.append(x_curr)
    #         u_all.append(u_curr)
    #         theta_all.append(theta)
    #         theta_hat_all.append(theta_hat)

    #     return x_all, u_all, theta_hat_all, theta_all

    def main(): 
        np.set_printoptions(suppress=False, precision=10)

        parser = argparse.ArgumentParser(description="An online parameter estimation interface with different robots, control tasks, estimation methods.")
        parser.add_argument("--robot", type=str, required=True, help="choose from quadrotor, cartpole, double_pendulum")
        parser.add_argument("--task", type=str, required=True, help="choose from hover, tracking, control") 
        parser.add_argument("--method", type=str, required=True, help="choose from Naive_MPC, KM, RLS, EKF, DEKA")  
        parser.add_argument("--update_type", type=str, required=False, help="choose from immediate_update, late_update, never_update") 
        args = parser.parse_args()

        method_name = f"simulate_{args.robot}_{args.task}_with_{args.method}"
        ParamEst = OnlineParamEst()

        if hasattr(ParamEst, method_name):
            if args.method == "Naive_MPC":              # without parameter estimation
                update_type = args.update_type
                x_history, u_history, theta_history = getattr(ParamEst, method_name)(update_type)
                title = f"{args.robot} {args.task} with {args.method} and update type: " + update_type 
                visualize_trajectory_inertia(x_history, u_history, theta_history, title) #TODO: different state space for different robots!
            else:                                       # with parameter estimation 
                x_history, u_history, theta_history, theta_hat_history = getattr(ParamEst, method_name)()
                title = f"Online Parameter Estimation for {args.robot} {args.task} with {args.method}"
                visualize_trajectory_with_est(x_history, u_history, theta_history, theta_hat_history, title)
                # theta_hat_deka_array = np.array(theta_hat_history)
                # theta_deka_array = np.array(theta_history)
                # visualize_all_parameter(theta_deka_array, theta_hat_deka_array)

        else: 
            print(f"Error: Method {method_name} not found. Use --help to see available option.")


if __name__ == "__main__":
    OnlineParamEst.main()