import numpy as np
import matplotlib.pyplot as plt
import os

if not os.path.exists('heat'):
    os.makedirs('heat')


# Define the function
def t_to_string(t, N, dim):
    str_t = "{:.3f}".format(t)
    str_t = str_t.replace(".", "_")
    str_t = str_t.zfill(6)
    str_N = str(N).zfill(4)  # Assumes N is less than 10,000
    str_dim = str(dim)
    return str_t + "_" + str_N + "_" + str_dim


# Define the domain
n_points= 50
x1 = np.linspace(-1, 1, n_points)
x2 = np.linspace(-1, 1, n_points)
X1, X2 = np.meshgrid(x1, x2)

# Set the parameters
T = 0.1
d = 1 # number of terms in the sum
N_dataSamples = 1000  # number of data samples
t_string = t_to_string(T, N_dataSamples, d)


# Define the initial condition function
def u0(x1, x2, mu):
    u = 0
    for m in range(1, d + 1):
        u -= mu[m - 1] * np.sin(np.pi * m * x1) * np.sin(np.pi * m * x2) / np.sqrt(m)
    return u / d


# Define the solution function
def u(t, x1, x2, mu):
    u = 0
    for m in range(1, d + 1):
        u -= np.exp(-(np.pi * m) ** 2 * t) * mu[m - 1] * np.sin(np.pi * m * x1) * np.sin(np.pi * m * x2) / np.sqrt(m)
    return u / d


# Generate a set of mu values
np.random.seed(42)
mu_values = np.random.uniform(-1, 1, (N_dataSamples, d))

# Generate the data
data = []
for mu in mu_values:
    U0 = u0(X1, X2, mu)
    U = u(T, X1, X2, mu)
    data.append((U0, U, mu))

# Format the data for the parametric approach (spatial structure should inherently be captured by the PDE)
outputs_parametric = []
inputs_parametric = []
for U0, U, mu in data:
    for i in range(n_points):
        for j in range(n_points):
            inputs_parametric.append([X1[i, j], X2[i, j]] + list(mu))
            outputs_parametric.append(U[i, j])


# Format the data for the operator approach
inputs_operator = []
outputs_operator = []
for U0, U, mu in data:
    inputs_operator.append(np.dstack([X1, X2, U0]))
    outputs_operator.append(np.expand_dims(U, axis=-1))


#inputs_parametric_np = np.array(inputs_parametric)
#outputs_parametric_np = np.array(outputs_parametric)

# Save the numpy arrays
np.save(f"heat/inputs_parametric_heat_{t_string}.npy", inputs_parametric)
np.save(f"heat/outputs_parametric_heat_{t_string}.npy", outputs_parametric)

# Unnormalized Data: Convert to numpy arrays
# Assuming inputs_operator and outputs_operator are your data
np.save(f'heat/inputs_operator_heat_{t_string}.npy', inputs_operator)
np.save(f'heat/outputs_operator_heat_{t_string}.npy', outputs_operator)


#
# # Normalizing the input
# inputs_operator_np = np.array(inputs_operator)
# U0_values = inputs_operator_np[..., 2] # Extract the U0 values
#
# mean_U0 = np.mean(U0_values)
# std_U0 = np.std(U0_values)
# inputs_operator_np[..., 2] = (U0_values - mean_U0) / std_U0
#
# # Normalizing the output
# outputs_operator_np = np.array(outputs_operator)
# U_values = outputs_operator_np[..., 0]
#
# mean_output = np.mean(U_values)
# std_output = np.std(U_values)
# outputs_operator_np[..., 0] = (U_values - mean_output) / std_output
#
# # Save the normalized data
# np.save(f'heat/inputs_operator_normalized_heat_{t_string}.npy', inputs_operator_np)
# np.save(f'heat/outputs_operator_normalized_heat_{t_string}.npy', outputs_operator_np)
#
# # Save the mean and standard deviation for later use (renormalization)
# np.save(f'heat/mean_output_operator_heat_{t_string}.npy', mean_output)
# np.save(f'heat/std_output_operator_heat_{t_string}.npy', std_output)
#
#
#
# # Denormalize the predicted output data
# #U_predicted_denormalized = U_predicted_normalized * std_output_operator + mean_output_operator
#
#
# # Normalize the parametric data
# inputs_parametric_np = np.array(inputs_parametric)
# outputs_parametric_np = np.array(outputs_parametric)
#
# # Normalizing the parameter vector mu
# mu_values = inputs_parametric_np[..., 2:] # Extract the mu values
# mean_mu = np.mean(mu_values, axis=0)
# std_mu = np.std(mu_values, axis=0)
# inputs_parametric_np[..., 2:] = (mu_values - mean_mu) / std_mu
#
# # Normalizing the output
# mean_output = np.mean(outputs_parametric_np)
# std_output = np.std(outputs_parametric_np)
# outputs_parametric_np = (outputs_parametric_np - mean_output) / std_output
#
#
# # Save the normalized data
# np.save(f'heat/inputs_parametric_normalized_heat_{t_string}.npy', inputs_parametric_np)
# np.save(f'heat/outputs_parametric_normalized_heat_{t_string}.npy', outputs_parametric_np)
#
# # Save the mean and standard deviation for later use (renormalization)
# np.save(f'heat/mean_output_parametric_heat_{t_string}.npy', mean_output)
# np.save(f'heat/std_output_parametric_heat_{t_string}.npy', std_output)
# np.save(f'heat/mean_mu_heat_{t_string}.npy', mean_mu)
# np.save(f'heat/std_mu_heat_{t_string}.npy', std_mu)
#
# # Load the mean and standard deviation
# mean_output = np.load(f'heat/mean_output_parametric_heat_{t_string}.npy')
# std_output = np.load(f'heat/std_output_parametric_heat_{t_string}.npy')




## Sanity Data Generation, plot normalized parametric data
n_plots = 10
n_rows = 2
n_cols = 5

# Define the domain
x1 = np.linspace(-1, 1, n_points)
x2 = np.linspace(-1, 1, n_points)
X, Y = np.meshgrid(x1, x2)

fig = plt.figure(figsize=(10, 5))  # Adjust as needed

for plot_number in range(n_plots):
    start_index = n_points**2 * plot_number
    end_index = n_points**2 * (plot_number + 1)

    output_all = []
    for i in range(start_index, end_index):
        output = outputs_parametric[i] # here normalized parametric data
        output_all.append(output)

    # Reshape the output back into a 2D grid
    output_2D = np.array(output_all).reshape(n_points, n_points)

    # Add subplot for each solution
    ax = fig.add_subplot(n_rows, n_cols, plot_number + 1, projection='3d')  # 2 rows, 5 columns, index
    surf = ax.plot_surface(X, Y, output_2D, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('U')
    ax.set_title(f'Solution at t={T}, Plot {plot_number+1}')

plt.tight_layout()
plt.show()


# Plot initial Condition
fig = plt.figure(figsize=(10, 5))  # Adjust as needed

for plot_number in range(n_plots):
    # Select the data for this plot
    input_3D = inputs_operator[plot_number]

    # Extract the initial condition U0
    U0_2D = input_3D[:, :, 2]

    # Add subplot
    ax = fig.add_subplot(n_rows, n_cols, plot_number + 1, projection='3d')
    surf = ax.plot_surface(X, Y, U0_2D, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('U')
    ax.set_title(f'Initial Condition, Plot {plot_number+1}')

# plt.tight_layout()
plt.show()
