import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Number of brain regions
X = 10
# Time steps
T = 1000
# Synaptic gating variable
S = np.zeros((X, T))
# Parameters
I_0 = 0.3
J = 0.5
dt = 0.1

# Runge-Kutta 4th order integration for Wong-Wang dynamics
def wong_wang_dynamics(S_prev):
    k1 = -S_prev/80 + (1-S_prev)*(I_0 + J*S_prev)
    k2 = -S_prev/80 + (1-S_prev)*(I_0 + J*(S_prev + k1*dt/2))
    k3 = -S_prev/80 + (1-S_prev)*(I_0 + J*(S_prev + k2*dt/2))
    k4 = -S_prev/80 + (1-S_prev)*(I_0 + J*(S_prev + k3*dt))
    return S_prev + (k1 + 2*k2 + 2*k3 + k4) * dt / 6

for t in range(1, T):
    for x in range(X):
        S[x, t] = wong_wang_dynamics(S[x, t-1])

# Converting neural activity to BOLD signal using the Balloon-Windkessel model
BOLD = np.zeros_like(S)
tau = 0.8 # Time constant for hemodynamics
for x in range(X):
    for t in range(1, T):
        BOLD[x, t] = BOLD[x, t-1] + dt * (S[x, t-1] - BOLD[x, t-1]) / tau

# Computing the functional connectivity matrix
FC = np.zeros((X, X))
for i in range(X):
    for j in range(X):
        corr, _ = pearsonr(BOLD[i, :], BOLD[j, :])
        FC[i, j] = corr

plt.imshow(FC, cmap='hot')
plt.title('Functional Connectivity Matrix')
plt.colorbar()
plt.show()
