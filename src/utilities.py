import matplotlib.pyplot as plt
import autograd.numpy as np

def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion [qw, qx, qy, qz] to a 3x3 rotation matrix.

    Args:
        q (np.ndarray): Quaternion as [qw, qx, qy, qz]

    Returns:
        np.ndarray: 3x3 Rotation matrix
    """
    qw, qx, qy, qz = q

    # Compute the elements of the rotation matrix
    R = np.array([
        [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qw*qz),  2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),  1 - 2*(qx**2 + qz**2),  2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),  2*(qy*qz + qw*qx),  1 - 2*(qx**2 + qy**2)]
    ])

    return R

def visualize_trajectory_inertia(x_all, u_all, theta_all, title):
    # Set up the figure and axis for plotting
    fig, ax = plt.subplots(3, 1)

    # Plot the trajectory
    x_all = np.array(x_all)
    nsteps = len(x_all)
    steps = np.arange(nsteps)
    ax[0].plot(steps, x_all[:, 0], label="x", linewidth=2)
    ax[0].plot(steps, x_all[:, 1], label="y", linewidth=2)
    ax[0].plot(steps, x_all[:, 2], label="z", linewidth=2)
    ax[0].legend()
    ax[0].title.set_text("Position")

    theta_all = np.array(theta_all)
    nsteps = len(theta_all)
    steps = np.arange(nsteps)
    ax[1].plot(steps, theta_all[:, 0], label="mass", linewidth=2)
    # ax[1].plot(steps, theta_all[:, 1], label="Jx", linewidth=2)
    # ax[1].plot(steps, theta_all[:, 2], label="Jx", linewidth=2)
    # ax[1].plot(steps, theta_all[:, 3], label="Jx", linewidth=2)
    ax[1].legend()
    ax[1].title.set_text("Parameters: [0:3]")
   
    u_all = np.array(u_all)
    nsteps = len(u_all)
    steps = np.arange(nsteps)
    ax[2].plot(steps, u_all[:, 0], label="u1", linewidth=2)
    ax[2].plot(steps, u_all[:, 1], label="u2", linewidth=2)
    ax[2].plot(steps, u_all[:, 2], label="u3", linewidth=2)
    ax[2].plot(steps, u_all[:, 3], label="u4", linewidth=2)
    ax[2].legend()
    ax[2].title.set_text("Controls")

    plt.suptitle(title)
    plt.show()

def visualize_all_parameter(theta_all, theta_hat_all):
    t = np.arange(len(theta_all))  # time steps 0..N-1

    fig, axs = plt.subplots(4, 2, figsize=(12, 10))
    axs = axs.flatten()  # Make it a 1D list of axes for easy iteration

    for j in range(7):
        axs[j].plot(t, theta_hat_all[:, j], label=f'deka param {j}')
        axs[j].plot(t, theta_all[:, j], '--', label=f'True param {j}')
        axs[j].set_title(f'Parameter {j}')
        axs[j].set_xlabel('Time Step')
        axs[j].set_ylabel('Value')
        axs[j].legend()
        axs[j].grid(True)

    # Hide the 8th subplot if you only have 7 params
    axs[-1].set_visible(False)

    plt.tight_layout()
    plt.show()

def visualize_trajectory(x_all, u_all, theta_all, title):
    # Set up the figure and axis for plotting
    fig, ax = plt.subplots(3, 1)

    # Plot the trajectory
    x_all = np.array(x_all)
    nsteps = len(x_all)
    steps = np.arange(nsteps)
    ax[0].plot(steps, x_all[:, 0], label="x", linewidth=2)
    ax[0].plot(steps, x_all[:, 1], label="y", linewidth=2)
    ax[0].plot(steps, x_all[:, 2], label="z", linewidth=2)
    ax[0].legend()
    ax[0].title.set_text("Position")

    theta_all = np.array(theta_all)
    nsteps = len(theta_all)
    steps = np.arange(nsteps)
    ax[1].plot(steps, theta_all[:, 0], label="mass", linewidth=2)
    ax[1].plot(steps, theta_all[:, 1], label="Ixx", linewidth=2)
    ax[1].plot(steps, theta_all[:, 2], label="Iyy", linewidth=2)
    ax[1].plot(steps, theta_all[:, 3], label="Izz", linewidth=2)
    ax[1].legend()
    ax[1].title.set_text("Inertia Parameters")
   
    u_all = np.array(u_all)
    nsteps = len(u_all)
    steps = np.arange(nsteps)
    ax[2].plot(steps, u_all[:, 0], label="u1", linewidth=2)
    ax[2].plot(steps, u_all[:, 1], label="u2", linewidth=2)
    ax[2].plot(steps, u_all[:, 2], label="u3", linewidth=2)
    ax[2].plot(steps, u_all[:, 3], label="u4", linewidth=2)
    ax[2].legend()
    ax[2].title.set_text("Controls")

    plt.suptitle(title)
    plt.show()

def visualize_trajectory_with_est(x_all, u_all, theta_all, theta_hat_all, title):
    # Set up the figure and axis for plotting
    fig, ax = plt.subplots(8, 1)

    # Plot the trajectory
    x_all = np.array(x_all)
    nsteps = len(x_all)
    steps = np.arange(nsteps)
    ax[0].plot(steps, x_all[:, 0], label="x", linewidth=1)
    ax[0].plot(steps, x_all[:, 1], label="y", linewidth=1)
    ax[0].plot(steps, x_all[:, 2], label="z", linewidth=1)
    ax[0].legend()
    ax[0].title.set_text("Position Tracking")

    # theta_all = np.array(theta_all)
    # theta_hat_all = np.array(theta_hat_all)
    # nsteps = len(theta_all)
    # steps = np.arange(nsteps)
    # ax[1].plot(steps, np.mean(theta_all, axis=1), label="ave param gt", linewidth=1)
    # ax[1].plot(steps, np.mean(theta_hat_all, axis=1), label="ave param est", linewidth=1)
    # ax[1].legend()
    # ax[1].title.set_text("Parameter Tracking")



    # ax[2].plot(steps, theta_all[:,1], label="g", linewidth=1)
    # ax[2].plot(steps, theta_hat_all[:,1], label="g_hat", linewidth=1)
    # ax[2].legend()
    # ax[2].title.set_text("Param: Gravity")

    u_all = np.array(u_all)
    nsteps = len(u_all)
    steps = np.arange(nsteps)
    ax[1].plot(steps, u_all[:, 0], label="u1", linewidth=1)
    ax[1].plot(steps, u_all[:, 1], label="u2", linewidth=1)
    ax[1].plot(steps, u_all[:, 2], label="u3", linewidth=1)
    ax[1].plot(steps, u_all[:, 3], label="u4", linewidth=1)
    ax[1].legend()
    ax[1].title.set_text("Controls")

    theta_all = np.array(theta_all)
    theta_hat_all = np.array(theta_hat_all)
    nsteps = len(theta_all)
    steps = np.arange(nsteps)
    for i in range(6):
        ax[i+2].plot(steps, theta_all[:,i], label=f"param {i+1} gt", linewidth=1)
        ax[i+2].plot(steps, theta_hat_all[:,i], label=f"param {i+1} est", linewidth=1)
        ax[i+2].legend()
        ax[i+2].title.set_text(f"Parameter {i+1} Tracking")

    plt.tight_layout()
    plt.suptitle(title)
    plt.show()

def plot_trajectory_with_est(x_all, u_all, theta_all, theta_hat_all, lsq_solns, title="Quadrotor Simulation"):
        """
        Plots quadrotor state trajectory, control inputs, and estimated parameters.
        """
        time = range(len(x_all))  # 0,1,2,... up to NSIM-1
        
        # Extract position
        pos_x = [state[0] for state in x_all]
        pos_y = [state[1] for state in x_all]
        pos_z = [state[2] for state in x_all]

        # Extract controls
        u1 = [ctrl[0] for ctrl in u_all]
        u2 = [ctrl[1] for ctrl in u_all]
        u3 = [ctrl[2] for ctrl in u_all]
        u4 = [ctrl[3] for ctrl in u_all]

        # Extract ground truth parameters
        mass, Jx, Jy, Jz = [], [], [], []
        for theta in theta_all:
            mass.append(theta[0].item())
            Jx.append(theta[1].item())
            Jy.append(theta[2].item())
            Jz.append(theta[3].item())

        # Extract estimated parameters
        mass_est, Jx_est, Jy_est, Jz_est = [], [], [], []
        for theta_hat in theta_hat_all:
            mass_est.append(theta_hat[0].item())
            Jx_est.append(theta_hat[1].item())
            Jy_est.append(theta_hat[2].item())
            Jz_est.append(theta_hat[3].item())

        # Extract LSQ solutions
        mass_lsq, Jx_lsq, Jy_lsq, Jz_lsq = [], [], [], []
        for lsq in lsq_solns:
            mass_lsq.append(lsq[0].item())
            Jx_lsq.append(lsq[1].item())
            Jy_lsq.append(lsq[2].item())
            Jz_lsq.append(lsq[3].item())

        # Plot everything
        plt.figure(figsize=(10, 14))

        # Position
        plt.subplot(6,1,1)
        plt.plot(time, pos_x, label='x-position')
        plt.plot(time, pos_y, label='y-position')
        plt.plot(time, pos_z, label='z-position')
        plt.title(title)
        plt.ylabel("Position (m)")
        plt.legend()
        plt.grid(True)

        # Control Inputs
        plt.subplot(6,1,2)
        plt.plot(time, u1, label='u1')
        plt.plot(time, u2, label='u2')
        plt.plot(time, u3, label='u3')
        plt.plot(time, u4, label='u4')
        plt.ylabel("Control Inputs")
        plt.legend()
        plt.grid(True)

        # Jx
        plt.subplot(6,1,3)
        plt.plot(time, Jx, label='Jx_true')
        plt.plot(time, Jx_est, label='Jx_est')
        plt.plot(time, Jx_lsq, label='Jx_lsq')
        plt.ylabel("Theta")
        plt.legend()
        plt.grid(True)

        # Jy
        plt.subplot(6,1,4)
        plt.plot(time, Jy, label='Jy_true')
        plt.plot(time, Jy_est, label='Jy_est')
        plt.plot(time, Jy_lsq, label='Jy_lsq')
        plt.ylabel("Theta")
        plt.legend()
        plt.grid(True)

        # Jz
        plt.subplot(6,1,5)
        plt.plot(time, Jz, label='Jz_true')
        plt.plot(time, Jz_est, label='Jz_est')
        plt.plot(time, Jz_lsq, label='Jz_lsq')
        plt.ylabel("Theta")
        plt.legend()
        plt.grid(True)

        # Mass
        plt.subplot(6,1,6)
        plt.plot(time, mass, label='mass_true')
        plt.plot(time, mass_est, label='mass_est')
        plt.plot(time, mass_lsq, label='mass_lsq')
        plt.xlabel("Time Step")
        plt.ylabel("Theta")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

def plot_trajectory(x_all, u_all, theta_all, title="Quadrotor Simulation"):
        """
        Plots quadrotor state trajectory, control inputs, and estimated parameters.
        """
        time = range(len(x_all))  # 0,1,2,... up to NSIM-1
        
        # Extract position
        pos_x = [state[0] for state in x_all]
        pos_y = [state[1] for state in x_all]
        pos_z = [state[2] for state in x_all]

        # Extract controls
        u1 = [ctrl[0] for ctrl in u_all]
        u2 = [ctrl[1] for ctrl in u_all]
        u3 = [ctrl[2] for ctrl in u_all]
        u4 = [ctrl[3] for ctrl in u_all]

        # Extract ground truth parameters
        mass, Jx, Jy, Jz = [], [], [], []
        for theta in theta_all:
            mass.append(theta[0].item())
            Jx.append(theta[1].item())
            Jy.append(theta[2].item())
            Jz.append(theta[3].item())    

        # Plot everything
        plt.figure(figsize=(10, 14))

        # Position
        plt.subplot(6,1,1)
        plt.plot(time, pos_x, label='x-position')
        plt.plot(time, pos_y, label='y-position')
        plt.plot(time, pos_z, label='z-position')
        plt.title(title)
        plt.ylabel("Position (m)")
        plt.legend()
        plt.grid(True)

        # Control Inputs
        plt.subplot(6,1,2)
        plt.plot(time, u1, label='u1')
        plt.plot(time, u2, label='u2')
        plt.plot(time, u3, label='u3')
        plt.plot(time, u4, label='u4')
        plt.ylabel("Control Inputs")
        plt.legend()
        plt.grid(True)

        # Jx
        plt.subplot(6,1,3)
        plt.plot(time, Jx, label='Jx_true')
        plt.ylabel("Theta")
        plt.legend()
        plt.grid(True)

        # Jy
        plt.subplot(6,1,4)
        plt.plot(time, Jy, label='Jy_true')
        plt.ylabel("Theta")
        plt.legend()
        plt.grid(True)

        # Jz
        plt.subplot(6,1,5)
        plt.plot(time, Jz, label='Jz_true')
        plt.ylabel("Theta")
        plt.legend()
        plt.grid(True)

        # Mass
        plt.subplot(6,1,6)
        plt.plot(time, mass, label='mass_true')
        plt.xlabel("Time Step")
        plt.ylabel("Theta")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

# @staticmethod
# def visualize_trajectory(x_all, u_all, title):
#     # Set up the figure and axis for plotting
#     fig, ax = plt.subplots(2, 1)

#     # Plot the trajectory
#     x_all = np.array(x_all)
#     nsteps = len(x_all)
#     steps = np.arange(nsteps)
#     ax[0].plot(steps, x_all[:, 0], label="x", linewidth=2)
#     ax[0].plot(steps, x_all[:, 1], label="y", linewidth=2)
#     ax[0].plot(steps, x_all[:, 2], label="z", linewidth=2)
#     ax[0].legend()
#     ax[0].title.set_text("Position")

#     u_all = np.array(u_all)
#     nsteps = len(u_all)
#     steps = np.arange(nsteps)
#     ax[1].plot(steps, u_all[:, 0], label="u1", linewidth=2)
#     ax[1].plot(steps, u_all[:, 1], label="u2", linewidth=2)
#     ax[1].plot(steps, u_all[:, 2], label="u3", linewidth=2)
#     ax[1].plot(steps, u_all[:, 3], label="u4", linewidth=2)
#     # ax[1].legend()
#     ax[1].title.set_text("Controls")

#     plt.suptitle(title)
#     plt.show()