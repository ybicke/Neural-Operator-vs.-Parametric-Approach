


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# Define the domain
x1 = np.linspace(-1, 1, 100)
x2 = np.linspace(-1, 1, 100)
X1, X2 = np.meshgrid(x1, x2)

# Set the parameters
# use 0.5 for more advanced setting
# use 0.1 for a comparable setting
t = 0.7  # final time
d = 3 # number of terms in the sum
np.random.seed(42)
mu = np.random.uniform(-1, 1, d)  # mu values are random numbers between -1 and 1

# Define the initial condition function
def u0(x1, x2, mu):
    u = 0
    for m in range(1, d+1):
        u -= mu[m-1] * np.sin(np.pi * m * x1) * np.sin(np.pi * m * x2) / np.sqrt(m)
    return u / d

# Define the solution function
def u(t, x1, x2, mu):
    u = 0
    for m in range(1, d+1):
        u += mu[m-1] * np.sin(np.pi * m * x1) * np.sin(np.pi * m * x2) * np.cos(np.pi * m * np.sqrt(2) * t) / np.sqrt(m)
    return u / d



# n_plots = 10
# n_points = 100
# n_rows = 2
# n_cols = 5
#
# # Create two separate figures for initial conditions and solutions
# fig1 = plt.figure(figsize=(20, 10))  # Adjust as needed
# fig1.suptitle('Initial conditions')
# fig2 = plt.figure(figsize=(20, 10))  # Adjust as needed
# fig2.suptitle('Solutions at t={}'.format(t))
#
# np.random.seed(42)
#
# for plot_number in range(n_plots):
#     # Generate new parameters for each plot
#
#     mu = np.random.uniform(-1, 1, d)
#
#     # Compute the initial condition and the solution with the new parameters
#     U0 = u0(X1, X2, mu)
#     U = u(t, X1, X2, mu)
#
#     # Compute the global minimum and maximum of the solution
#     U_min = min(U0.min(), U.min())
#     U_max = max(U0.max(), U.max())
#
#     # Add subplot for each initial condition
#     ax = fig1.add_subplot(n_rows, n_cols, plot_number + 1, projection='3d')
#     surf = ax.plot_surface(X1, X2, U0, cmap='viridis')#, vmin=U_min, vmax=U_max)
#     ax.set_xlabel('X1')
#     ax.set_ylabel('X2')
#     ax.set_zlabel('T')
#     ax.set_title('Initial condition {}'.format(plot_number+1))
#     #ax.set_zlim(U_min, U_max)
#
#     # Add subplot for each solution
#     ax = fig2.add_subplot(n_rows, n_cols, plot_number + 1, projection='3d')
#     surf = ax.plot_surface(X1, X2, U, cmap='viridis' )#, vmin=U_min, vmax=U_max)
#     ax.set_xlabel('X1')
#     ax.set_ylabel('X2')
#     ax.set_zlabel('T')
#     ax.set_title('Solution {}'.format(plot_number+1))
#     #ax.set_zlim(U_min, U_max)
#
# plt.tight_layout()
# plt.show()



#
#
# # Compute and plot the solution at several time points
# time_points = np.linspace(0, t, 10)  # 10 time points between 0 and T
#
# # Calculate the number of rows needed for your plots
# n_rows = int(np.ceil(len(time_points) / 2.0))
#
# fig, axs = plt.subplots(n_rows, 2, figsize=(10, n_rows*5), subplot_kw={'projection': '3d'})
#
# # Flatten the array of axes so that you can iterate over them
# axs = axs.flatten()
#
# for ax, t in zip(axs, time_points):
#     U = u(t, X1, X2, mu)
#     surf = ax.plot_surface(X1, X2, U, cmap='viridis' , vmin=U_min, vmax=U_max)
#     ax.set_xlabel('X1')
#     ax.set_ylabel('X2')
#     ax.set_zlabel('U')
#     ax.set_title('Solution at t={:.3f}'.format(t))
#     ax.set_zlim(U_min, U_max)
#
# # Remove any extra subplots
# if len(time_points) % 2 != 0:
#     fig.delaxes(axs[-1])
#
# plt.tight_layout()
# plt.show()



mu = np.random.uniform(-1, 1, d)


# Compute and plot the solution at several time points
time_points = np.linspace(0, t, 5)  # 5 time points between 0 and T
fig, axs = plt.subplots(1, len(time_points), figsize=(15, 5), subplot_kw={'projection': '3d'})

for ax, t in zip(axs, time_points):

    U0 = u0(X1, X2, mu)
    U = u(t, X1, X2, mu)

    # Compute the global minimum and maximum of the solution
    U_min = min(U0.min(), U.min())
    U_max = max(U0.max(), U.max())

    surf = ax.plot_surface(X1, X2, U, cmap='viridis' , vmin=U_min, vmax=U_max)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('U')
    ax.set_title('Solution at t={:.3f}'.format(t))
    ax.set_zlim(U_min, U_max)
plt.show()
plt.tight_layout()



