import numpy as np
from scipy.signal import hilbert
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Number of brain regions
X = 10
# Time steps
T = 5000
# Parameters for the Wilson-Cowan equations
tau_e, tau_i = 10, 10
alpha_e, alpha_i = 1.3, 2
theta_e, theta_i = 4, 3.7
beta = 0.4
# Global coupling
G = 0.1
# Time delay (in time steps)
delay = 2
# Structural connectivity matrix
SC = np.random.rand(X, X)
# Excitatory and inhibitory variables
E = np.random.rand(X)
I = np.random.rand(X)
# Time-step size
dt = 0.1

# Time series container
time_series_E = np.zeros((X, T))

# Wilson-Cowan equations update
def update(E, I, past_E, SC):
    dE = (-E + (1 - E) * alpha_e * np.tanh(E - beta * I + theta_e + G * SC.dot(past_E))) / tau_e
    dI = (-I + alpha_i * np.tanh(E - I + theta_i)) / tau_i
    return E + dE * dt, I + dI * dt

# Simulating Wilson-Cowan dynamics
for t in range(T):
    if t < delay:
        past_E = E  # Use current state for first time steps
    else:
        past_E = time_series_E[:, t - delay]
    
    E, I = update(E, I, past_E, SC)
    time_series_E[:, t] = E

# Converting to BOLD signal using the Hilbert transform
BOLD_signal = np.abs(hilbert(time_series_E))

# Computing the functional connectivity matrix
FC = np.zeros((X, X))
for i in range(X):
    for j in range(X):
        corr, _ = pearsonr(BOLD_signal[i, :], BOLD_signal[j, :])
        FC[i, j] = corr

plt.imshow(FC, cmap='hot')
plt.title('Functional Connectivity Matrix')
plt.colorbar()
plt.show()
