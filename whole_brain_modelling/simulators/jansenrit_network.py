import numpy as np

def jansen_rit(SC, GC, time_delay, simulation_time, dt):
    num_regions = SC.shape[0]
    
    # Parameters
    A = 3.25
    B = 22
    a = 100
    b = 50
    v0 = 6
    e0 = 2.5
    r = 0.56
    C = 135
    
    # State variables
    y = np.zeros((num_regions, 6, simulation_time))

    # Output
    S = np.zeros((num_regions, simulation_time))

    for t in range(1, simulation_time - time_delay):
        delayed_y = y[:, :, t - time_delay]
        I = SC.dot(GC * S[:, t-1, None])
        for i in range(num_regions):
            y[i, 0, t] = y[i, 3, t-1]
            m = e0 * (1 + r * np.sin(np.pi * y[i, 1, t-1] / v0))
            y[i, 1, t] = y[i, 4, t-1]
            y[i, 2, t] = y[i, 5, t-1]
            y[i, 3, t] = A * a * Sigmoid(y[i, 1, t-1] - y[i, 2, t-1]) - 2 * a * y[i, 0, t-1] - a**2 * y[i, 3, t-1] + I[i]
            y[i, 4, t] = A * a * (p + C * Sigmoid(C * m)) - 2 * a * y[i, 1, t-1] - a**2 * y[i, 4, t-1]
            y[i, 5, t] = B * b * C * Sigmoid(C * m) - 2 * b * y[i, 2, t-1] - b**2 * y[i, 5, t-1]
            S[i, t] = y[i, 0, t-1] - y[i, 1, t-1]
    return S

def Sigmoid(v):
    e0 = 2.5
    v0 = 6
    r = 0.56
    return 2 * e0 / (1 + np.exp(r * v0 / v))

def balloon_windkessel(S, dt):
    num_regions, simulation_time = S.shape
    BOLD = np.zeros_like(S)
    k1 = 7
    k2 = 2
    k3 = 2
    alpha = 0.32
    tau_s = 0.8
    tau_f = 0.4
    tau_o = 0.4
    E = 0.4

    s = np.zeros((num_regions, simulation_time))
    f = np.zeros_like(s)
    v = np.zeros_like(s)
    q = np.zeros_like(s)
    
    for t in range(1, simulation_time):
        s[:, t] = s[:, t-1] + dt * (S[:, t] - (k1 + 1) * s[:, t-1] + k1 * f[:, t-1])
        f[:, t] = f[:, t-1] + dt * s[:, t-1]
        v[:, t] = v[:, t-1] + dt * (alpha * (S[:, t] - f[:, t]) - 1/tau_s * v[:, t-1])
        q[:, t] = q[:, t-1] + dt * (alpha * (S[:, t] - E * f[:, t]) - 1/tau_f * q[:, t-1])
        BOLD[:, t] = v[:, t] - q[:, t]
        
    return BOLD

SC = np.random.rand(10, 10) # Structural connectivity for 10 regions
GC = 0.2 # Global coupling
time_delay = 5
dt = 0.1
simulation_time = 1000

S = jansen_rit(SC, GC, time_delay, simulation_time, dt)
BOLD = balloon_windkessel(S, dt)
