import numpy as np
import matplotlib.pyplot as plt

# Number of brain regions
x = 10

# Time step
dt = 0.1
# Total simulation time
T = 5000
# Number of steps
steps = int(T/dt)

# Initialize excitatory and inhibitory activities
E = np.zeros((steps, x))
I = np.zeros((steps, x))

# Wilson-Cowan parameters
a_e, a_i = 1.0, 1.0
b_e, b_i = 4.0, 4.0
tau_e, tau_i = 10.0, 10.0
I_e, I_i = 0.0, 0.0
w_ee, w_ei, w_ie, w_ii = 12.0, 4.0, 13.0, 11.0

# Sigmoid function
def sigmoid(x, a, b):
    return 1 / (1 + np.exp(-(x - a) / b))

# Wilson-Cowan model for x regions
def wc_model(E, I):
    dE = (-E + sigmoid(w_ee * E - w_ei * I + I_e, a_e, b_e)) / tau_e
    dI = (-I + sigmoid(w_ie * E - w_ii * I + I_i, a_i, b_i)) / tau_i
    return dE, dI

# Runge-Kutta 4th order integration for Wilson-Cowan model
for t in range(steps - 1):
    k1_E, k1_I = wc_model(E[t], I[t])
    k2_E, k2_I = wc_model(E[t] + dt * k1_E / 2, I[t] + dt * k1_I / 2)
    k3_E, k3_I = wc_model(E[t] + dt * k2_E / 2, I[t] + dt * k2_I / 2)
    k4_E, k4_I = wc_model(E[t] + dt * k3_E, I[t] + dt * k3_I)
    E[t + 1] = E[t] + dt * (k1_E + 2 * k2_E + 2 * k3_E + k4_E) / 6
    I[t + 1] = I[t] + dt * (k1_I + 2 * k2_I + 2 * k3_I + k4_I) / 6

# Balloon-Windkessel model to convert neuronal activity to BOLD signal
# Here we will use a simplified representation
alpha = 0.32
BOLD = np.zeros((steps, x))
for t in range(1, steps):
    BOLD[t] = BOLD[t-1] + dt * (E[t] - alpha * BOLD[t-1])

# Compute functional connectivity matrix using Pearson correlation
FC = np.corrcoef(BOLD.T)

# Plot results
plt.imshow(FC, cmap='hot', interpolation='nearest')
plt.title('Functional Connectivity Matrix')
plt.colorbar(label='Correlation')
plt.show()
