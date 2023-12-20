import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
n_points = 32
# Define the domain
x1 = np.linspace(-1, 1, n_points)
x2 = np.linspace(-1, 1, n_points)
X1, X2 = np.meshgrid(x1, x2)

# Set the parameters
t = 0.1 # final time 0.05 for d=3
d = 3 # number of terms in the sum
mu = np.random.uniform(-1, 1, d)  # mu values are random numbers between -1 and 1, each time the solution looks a bit different
#mu=np.ones(d)

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
        u -= np.exp(-(np.pi * m) ** 2 * t) * mu[m-1] * np.sin(np.pi * m * x1) * np.sin(np.pi * m * x2) / np.sqrt(m)
    return u / d




n_plots = 4
n_rows = 2
n_cols = 2


# Create two separate figures for initial conditions and solutions
fig1 = plt.figure(figsize=(10, 5))  # Adjust as needed
fig1.suptitle('Initial conditions')
fig2 = plt.figure(figsize=(10, 5))  # Adjust as needed
fig2.suptitle('Solutions at t={}'.format(t))

np.random.seed(42)

for plot_number in range(n_plots):
    # Generate new parameters for each plot
    mu = np.random.uniform(-1, 1, d)

    # Compute the initial condition and the solution with the new parameters
    U0 = u0(X1, X2, mu)
    U = u(t, X1, X2, mu)

    # Compute the global minimum and maximum of the solution
    U_min = min(U0.min(), U.min())
    U_max = max(U0.max(), U.max())

    # Add subplot for each initial condition
    ax = fig1.add_subplot(n_rows, n_cols, plot_number + 1, projection='3d')
    surf = ax.plot_surface(X1, X2, U0, cmap='viridis')# , vmin=U_min, vmax=U_max)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('T')
    ax.set_title('Initial condition {}'.format(mu))
    # ax.set_zlim(U_min, U_max)


    # Add subplot for each solution
    ax = fig2.add_subplot(n_rows, n_cols, plot_number + 1, projection='3d')
    surf = ax.plot_surface(X1, X2, U, cmap='viridis', vmin=U_min, vmax=U_max)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('T')
    ax.set_title('Solution {}'.format(plot_number+1))
    ax.set_zlim(U_min, U_max)

plt.tight_layout()
plt.show()



#
#
#
mu = np.random.uniform(-1, 1, d)
U0 = u0(X1, X2, mu)
U = u(t, X1, X2, mu)

# Compute the global minimum and maximum of the solution
U_min = min(U0.min(), U.min())
U_max = max(U0.max(), U.max())
#
# Compute and plot the solution at several time points
time_points = np.linspace(0, t, 5)  # 5 time points between 0 and T
fig, axs = plt.subplots(1, len(time_points), figsize=(15, 5), subplot_kw={'projection': '3d'})

for ax, t in zip(axs, time_points):
    U = u(t, X1, X2, mu)
    surf = ax.plot_surface(X1, X2, U, cmap='viridis', vmin=U_min, vmax=U_max)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('U')
    ax.set_title('t={:.3f}'.format(t))
    #ax.set_zlim(U_min, U_max)
#
#
plt.show()
