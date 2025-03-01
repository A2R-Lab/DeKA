import math
import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy as sqrt
from autograd.numpy.linalg import norm
from autograd.numpy.linalg import inv
from autograd import jacobian
from autograd.test_util import check_grads
from collections import deque
from ne_quadrotor import Quadrotor
from LQR_controller import LQRController
from estimation_methods import *
from utilities import *
import argparse
# Note: autograd does not work with np.block



class OnlineParamEst:
    def __init__(self):
        # Quadrotor Tasks:
        self.quadrotor = Quadrotor()
        self.quadrotor_controller = LQRController(self.quadrotor.delta_x_quat)

    def simulate_quadrotor_hover_with_Naive_MPC(self, update_status: str, NSIM: int =200):
        # initialize quadrotor parameters
        theta = np.array([self.quadrotor.mass,self.quadrotor.Jx,self.quadrotor.Jy,self.quadrotor.Jz])
        theta_hat = np.copy(theta)
        x_nom, u_nom = self.quadrotor.get_hover_goals(theta[0], self.quadrotor.kt)
        x0 = np.copy(x_nom)
        x0[0:3] += np.array([0.2, 0.2, -0.2])  # disturbed initial position
        print("Perturbed Intitial State: ")
        print(x0)
        u0 = np.array([0,0,0,0])
    
        # get tasks goals
        Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta[0],theta[1],theta[2],theta[3])
        Q, R = self.quadrotor_controller.get_QR_bryson()
        self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

        x_all = []
        u_all = []
        theta_all = []
        theta_hat_all=[]

        x_all.append(x0)
        u_all.append(u0)
        theta_all.append(theta)
        theta_hat_all.append(theta_hat)
        
        x_curr = np.copy(x0)
        x_prev = np.copy(x0)

        changing_steps = np.random.choice(range(20,180),size=3, replace=False)

        # simulate the dynamics with the LQR controller
        for i in range(NSIM):
            # change mass
            if i in changing_steps:
                mass_update_scale = np.random.uniform(1,2)
                Jx_update_scale = np.random.uniform(1,2)
                Jy_update_scale = np.random.uniform(1,2)
                Jz_update_scale = np.random.uniform(1,2)

                theta = np.array([self.quadrotor.mass * mass_update_scale,self.quadrotor.Jx*Jx_update_scale,self.quadrotor.Jy*Jy_update_scale,self.quadrotor.Jz*Jz_update_scale])#,self.quadrotor.g * gravity_update_scale])
            # MPC controller
            if update_status == "immediate_update":
                if i in changing_steps:
                    x_nom, u_nom = self.quadrotor.get_hover_goals(theta[0], self.quadrotor.kt)                   
                    Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta[0], theta[1],theta[2],theta[3])
                    self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)
                u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)
            elif update_status == "never_update":
                u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)
            elif update_status == "late_update":
                if i in changing_steps+10: # when to tell controller the right parameters
                    x_nom, u_nom = self.quadrotor.get_hover_goals(theta[0], self.quadrotor.kt)
                    Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta[0], theta[1],theta[2],theta[3])
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

            x_prev = x_curr
            x_curr = self.quadrotor.quad_dynamics_rk4(x_curr, u_curr, theta[0],theta[1],theta[2],theta[3])

            x_curr = x_curr.reshape(x_curr.shape[0]).tolist()
            u_curr = u_curr.reshape(u_curr.shape[0]).tolist()

            x_all.append(x_curr)
            u_all.append(u_curr)
            theta_all.append(theta)
        return x_all, u_all, theta_all
    
    def simulate_quadrotor_hover_with_RLS(self, NSIM: int =200):
        # initialize quadrotor parameters
        theta = np.array([self.quadrotor.mass,self.quadrotor.Jx,self.quadrotor.Jy,self.quadrotor.Jz])
        print("J: ", theta[1], theta[2], theta[3])
        theta_hat = np.copy(theta)
        x_nom, u_nom = self.quadrotor.get_hover_goals(theta[0], self.quadrotor.kt)
        x0 = np.copy(x_nom)
        x0[0:3] += np.array([0.2, 0.2, -0.2])  # disturbed initial position
        # print("Perturbed Intitial State: ")
        # print(x0)
        u0 = u_nom
        x0_dot = self.quadrotor.quad_dynamics(x0,u0,self.quadrotor.mass,self.quadrotor.Jx,self.quadrotor.Jy,self.quadrotor.Jz)
        Vb  = x0_dot[0:3]         # linear velocity
        wB  = x0_dot[6:9]        # angular velocity
        dVb = x0_dot[3:6]      # linear accel
        dwB = x0_dot[9:12]     # angular accel
        print("x_0_dot: ", x0_dot)
        F = np.array([0,0,np.sum(u0)])
        tau = self.quadrotor.get_tau(u0, self.quadrotor.el, self.quadrotor.thrustToTorque)
        A,b = self.quadrotor.get_linear_system(Vb, wB, dVb, dwB, F, tau)
        
        # get tasks goals
        Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta[0],theta[1],theta[2],theta[3])
        Q, R = self.quadrotor_controller.get_QR_bryson()
        self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

              
        x_all = []
        u_all = []
        theta_all = []
        theta_hat_all=[]
        lsq_solns = []
        x_all.append(x0)
        u_all.append(u0)
        theta_all.append(theta)
        theta_hat_all.append(theta_hat)
        lsq_solns.append(theta_hat)
        
        x_curr = np.copy(x0)
        x_prev = np.copy(x0)

        changing_steps = np.random.choice(range(20,380),size=3, replace=False)

        rls = RLS(num_params=4)

        # simulate the dynamics with the LQR controller
        for i in range(NSIM):
            # change mass
            
            if i in changing_steps:
                mass_update_scale = np.random.uniform(1,2)
                Jx_update_scale = np.random.uniform(1,2)
                Jy_update_scale = np.random.uniform(1,2)
                Jz_update_scale = np.random.uniform(1,2)

                theta = np.array([self.quadrotor.mass * mass_update_scale,self.quadrotor.Jx*Jx_update_scale,self.quadrotor.Jy*Jy_update_scale,self.quadrotor.Jz*Jz_update_scale])#,self.quadrotor.g * gravity_update_scale])
            
        
            # MPC controller
            x_nom, u_nom = self.quadrotor.get_hover_goals(theta_hat[0], self.quadrotor.kt)
            x_dot = self.quadrotor.quad_dynamics(x_all[-1],u_all[-1],theta[0],theta[1],theta[2],theta[3])
            Vb  = x_all[-1][3:6]         # linear velocity
            wB  = x_all[-1][9:12]        # angular velocity
            dVb = x_dot[3:6]      # linear accel
            dwB = x_dot[9:12]     # angular accel
            F = np.array([0,0,np.sum(u_all[-1])])
            tau = self.quadrotor.get_tau(u_all[-1],self.quadrotor.el, self.quadrotor.thrustToTorque)
            # print(theta_hat)
            Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat[0],theta_hat[1],theta_hat[2],theta_hat[3])
            # print(Anp, Bnp)
            self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

            u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)
            
            # postponing the dynamics model by telling it the correct parameters after several steps
            x_prev = x_curr
            x_curr = self.quadrotor.quad_dynamics_rk4(x_curr, u_curr, theta[0],theta[1],theta[2],theta[3])

            # estimate parameters
            theta_hat_prev = theta_hat
            A,b = self.quadrotor.get_linear_system(Vb, wB, dVb, dwB, F, tau)
            #print("A: ", A, "b: ", b)
            estimate = rls.iterate(A, b)
            
            lsq_soln = np.linalg.lstsq(A,b)[0]
            lsq = np.linalg.lstsq(A,b)[0]
          
            theta_hat =estimate
            # if theta_hat[0] == 0:
            #     theta_hat[0] +=0.02
            if theta_hat[1]==0:
                theta_hat[1]+=1e-5
            if theta_hat[2]==0:
                theta_hat+=1e-5
            if theta_hat[3]==0:
                theta_hat[3]+=1e-5
           
            print("step: ", i, "\n", 
                  "u_k: ", u_curr, "\n", 
                  "x_k: ", x_prev[:3], "\n", 
                  "theta_k: ", theta, "\n", 
                  "theta_hat_k: ", theta_hat, "\n",
                  "least squares solution: ", lsq_soln, "\n",
                  )
            x_curr = x_curr.reshape(x_curr.shape[0]).tolist()
            u_curr = u_curr.reshape(u_curr.shape[0]).tolist()

            x_all.append(x_curr)
            u_all.append(u_curr)
            theta_all.append(theta)
            theta_hat_all.append(theta_hat.copy())
            lsq_solns.append(lsq_soln)
        print("theta", theta_all)
        print("theta_hat", theta_hat_all)

        return x_all, u_all, theta_all, theta_hat_all, lsq_solns

    def simulate_quadrotor_tracking_with_RLS(self,  NSIM: int =200):
        # initialize quadrotor parameters
        theta = np.array([self.quadrotor.mass,self.quadrotor.Jx,self.quadrotor.Jy,self.quadrotor.Jz])
      #  print("J: ", theta[1], theta[2], theta[3])
        theta_hat = np.copy(theta)

        #figure 8 trajectory 
        num_points = 601
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        traj = [np.array([np.sin(a), np.sin(a) * np.cos(a), 0]) for a in angles]

        x_nom = np.zeros(12)
        # get tasks goals
        x_nom[0:3 ]= traj[0]
        u_nom = (theta[0]*self.quadrotor.g/4)*np.ones(4)

        Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta[0],theta[1],theta[2],theta[3])
        Q, R = self.quadrotor_controller.get_QR_bryson()
        self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

        x0 = np.copy(x_nom)
        x0[0:3] += np.array([0.2, 0.2, -0.2])  # disturbed initial position
        # print("Perturbed Intitial State: ")
        # print(x0)
        
        u_curr = self.quadrotor_controller.compute(x0, x_nom, u_nom)
        x_curr = np.copy(x0)
              
        x_all = []
        u_all = []
        theta_all = []
        theta_hat_all=[]
        lsq_solns = []
        x_all.append(x0)
        u_all.append(u_curr)
        theta_all.append(theta)
        theta_hat_all.append(theta_hat)
        lsq_solns.append(theta_hat)
        
        x_curr = np.copy(x0)

        changing_steps = np.random.choice(range(20,300),size=3, replace=False)

        rls = RLS(num_params=4)

        # simulate the dynamics with the LQR controller
        for i in range(NSIM):
            # change mass
            
            if i in changing_steps:
                mass_update_scale = 1#np.random.uniform(1,2)
                Jx_update_scale =1# np.random.uniform(1,2)
                Jy_update_scale =1 #np.random.uniform(1,2)
                Jz_update_scale = 1#np.random.uniform(1,2)

                theta = np.array([self.quadrotor.mass * mass_update_scale,self.quadrotor.Jx*Jx_update_scale,self.quadrotor.Jy*Jy_update_scale,self.quadrotor.Jz*Jz_update_scale])#,self.quadrotor.g * gravity_update_scale])
            
            
            # MPC controller
            x_nom = x_nom = np.zeros(12)
            x_nom[0:3 ]= traj[i+1]
            #print("xnom: ", x_nom)
            u_nom = (theta_hat[0]*self.quadrotor.g/4)*np.ones(4)
            Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat[0],theta_hat[1],theta_hat[2],theta_hat[3])
            self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)
            
            
            x_dot = self.quadrotor.quad_dynamics(x_all[-1],u_all[-1],theta[0],theta[1], theta[2], theta[3])
            Vb  = x_all[-1][3:6]         # linear velocity
            wB  = x_all[-1][9:12]        # angular velocity
            dVb = x_dot[3:6]      # linear accel
            dwB = x_dot[9:12]     # angular accel
            F = np.array([0,0,np.sum(u_all[-1])])
            tau = self.quadrotor.get_tau(u_all[-1],self.quadrotor.el, self.quadrotor.thrustToTorque)
            u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)
            
            x_curr = self.quadrotor.quad_dynamics_rk4(x_curr, u_curr, theta[0],theta[1],theta[2],theta[3])

            # estimate parameters
            A,b = self.quadrotor.get_linear_system(Vb, wB, dVb, dwB, F, tau)
        #    print("A: ", A, "b: ", b)
            estimate = rls.iterate(A, b)
            
            lsq_soln = np.linalg.lstsq(A,b)[0]
            theta_hat =estimate
            print("mass_est_lsq: ", lsq_soln[0])
            if theta_hat[0] <= 1e-4:
                theta_hat[0] +=0.02
            if theta_hat[1]<= 1e-6:
                theta_hat[1]+=1e-5
            if theta_hat[2]<= 1e-6:
                theta_hat+=1e-5
            if theta_hat[3]<= 1e-6:
                theta_hat[3]+=1e-5
            print("----step-----: ", i)
            # print("step: ", i, "\n", 
            #       "u_k: ", u_curr, "\n", 
            #       "x_k: ", x_prev[:3], "\n", 
            #       "theta_k: ", theta, "\n", 
            #       "theta_hat_k: ", theta_hat, "\n",
            #       "least squares solution: ", lsq_soln, "\n",
            #       )
            x_curr = x_curr.reshape(x_curr.shape[0]).tolist()
            u_curr = u_curr.reshape(u_curr.shape[0]).tolist()

            x_all.append(x_curr)
            u_all.append(u_curr)
            theta_all.append(theta)
            theta_hat_all.append(theta_hat.copy())
            lsq_solns.append(lsq_soln)
        print("theta", theta_all)
        print("theta_hat", theta_hat_all)

        return x_all, u_all, theta_all, theta_hat_all, lsq_solns

    def simulate_quadrotor_tracking_with_Naive_MPC(self, update_status: str, NSIM: int =200):
        # initialize quadrotor parameters
        theta = np.array([self.quadrotor.mass,self.quadrotor.Jx,self.quadrotor.Jy,self.quadrotor.Jz])
        print("J: ", theta[1], theta[2], theta[3])
      

        #figure 8 trajectory 
        num_points = 600
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        traj = [np.array([np.sin(a), np.sin(a) * np.cos(a), 0]) for a in angles]

        x_nom = np.zeros(12)
        # get tasks goals
        x_nom[0:3 ]= traj[0]
        u_nom = (theta[0]*self.quadrotor.g/4)*np.ones(4)

        Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta[0],theta[1],theta[2],theta[3])
        Q, R = self.quadrotor_controller.get_QR_bryson()
        self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

        x0 = np.copy(x_nom)
        u_curr = self.quadrotor_controller.compute(x0, x_nom, u_nom)
        x_curr = np.copy(x0)

              
        x_all = []
        u_all = []
        theta_all = []
        x_all.append(x0)
        u_all.append(u_curr)
        theta_all.append(theta)
        
        x_curr = np.copy(x0)

        changing_steps = np.random.choice(range(20,300),size=3, replace=False)
        # simulate the dynamics with the LQR controller
        for i in range(NSIM):
            # change mass
            
            if i in changing_steps:
                mass_update_scale = np.random.uniform(1,2)
                Jx_update_scale = np.random.uniform(1,2)
                Jy_update_scale = np.random.uniform(1,2)
                Jz_update_scale = np.random.uniform(1,2)

                theta = np.array([self.quadrotor.mass * mass_update_scale,self.quadrotor.Jx*Jx_update_scale,self.quadrotor.Jy*Jy_update_scale,self.quadrotor.Jz*Jz_update_scale])#,self.quadrotor.g * gravity_update_scale])
            
            
            # MPC controller
            x_nom = x_nom = np.zeros(12)
            x_nom[0:3 ]= traj[i]
            print("xnom: ", x_nom)
            print("theta: ", theta)
            u_nom = (theta[0]*self.quadrotor.g/4)*np.ones(4)

            Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta[0],theta[1],theta[2],theta[3])
            self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)
            
     
            u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)
            
            # postponing the dynamics model by telling it the correct parameters after several steps
            x_prev = x_curr
            x_curr = self.quadrotor.quad_dynamics_rk4(x_curr, u_curr, theta[0],theta[1],theta[2],theta[3])
            
           
            print("step: ", i, "\n", 
                  "u_k: ", u_curr, "\n", 
                  "x_k: ", x_prev[:3], "\n", 
                  "theta_k: ", theta, "\n", 
                  )
            x_curr = x_curr.reshape(x_curr.shape[0]).tolist()
            u_curr = u_curr.reshape(u_curr.shape[0]).tolist()

            x_all.append(x_curr)
            u_all.append(u_curr)
            theta_all.append(theta)
   
        print("theta", theta_all)

        return x_all, u_all, theta_all


def main(): 
    parser = argparse.ArgumentParser(description="An online parameter estimation interface with different robots, control tasks, estimation methods.")
    parser.add_argument("--robot", type=str, required=True, choices=["quadrotor", "cartpole", "double_pendulum"],
                        help="Choose from quadrotor, cartpole, double_pendulum")
    parser.add_argument("--task", type=str, required=True, choices=["hover", "tracking", "control"],
                        help="Choose from hover, tracking, control") 
    parser.add_argument("--method", type=str, required=True, choices=["Naive_MPC", "KM", "RLS", "EKF", "DEKA"],
                        help="Choose from Naive_MPC, KM, RLS, EKF, DEKA")  
    parser.add_argument("--update_type", type=str, choices=["immediate_update", "late_update", "never_update"], default="immediate_update",
                        help="Choose from immediate_update, late_update, never_update (only for some methods)") 
    args = parser.parse_args()

    method_name = f"simulate_{args.robot}_{args.task}_with_{args.method}"
    estimator = OnlineParamEst()

    if hasattr(estimator, method_name):
        if args.method == "Naive_MPC":  # Without parameter estimation
            x_all, u_all, theta_all = getattr(estimator, method_name)(args.update_type)
            title = f"{args.robot} {args.task} with {args.method} (Update Type: {args.update_type})"
            plot_trajectory(x_all, u_all, theta_all, title)
        else:  # With parameter estimation
            x_all, u_all, theta_all, theta_hat_all, lsq_solns = getattr(estimator, method_name)()
            title = f"Online Parameter Estimation for {args.robot} {args.task} with {args.method}"
            plot_trajectory_with_est(x_all, u_all, theta_all, theta_hat_all, lsq_solns, title)
    else:
        print(f"Error: Method {method_name} not found. Use --help to see available options.")

if __name__ == "__main__":
    main()