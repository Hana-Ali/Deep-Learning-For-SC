####### WILSON-COWAN MODEL

#%% Import libraries
import os
os.add_dll_directory(r"C:\src\vcpkg\installed\x64-windows\bin")
os.add_dll_directory(r"C:\cpp_libs\include\bayesopt\build\bin\Release")
from sklearn.preprocessing import MinMaxScaler
from py_helpers.helper_funcs import *
from py_helpers.wilson_interface import *
import matplotlib.pyplot as plt
import scipy.signal as signal
import multiprocessing as mp
import scipy.stats as stats
import scipy.io as sio
import numpy as np
import time
import os

# Defining paths
root_path = 'C:\\Users\\shahi\\OneDrive - Imperial College London\\Documents\\imperial\\Spring Sem\\iso_dubai\\ISO\\HCP_DTI_BOLD'
write_path = "C:\\Users\\shahi\\OneDrive - Imperial College London\\Documents\\imperial\\Dissertation\\Notebooks\\MyCodes\\results"

# Defining optimization parameters
coupling_strength = 0.1
delay = 0.1

# Defining fixed parameters
c_ee = 16.0
c_ei = 12.0
c_ie = 15.0
c_ii = 3.0

tau_e = 8.0
tau_i = 8.0

r_e = 1.0
r_i = 1.0
k_e = 1.0
k_i = 1.0

alpha_e = 1.0
alpha_i = 1.0
theta_e = 4.0
theta_i = 3.7

external_e = 0.1
external_i = 0.1

# Defining integration parameters
time_simulated = 510.000 # seconds
integration_step_size = 0.002 # seconds

# Defining rate of downsampling
downsampling_rate = 400

# Defining when to start saving data
save_data_start = 150.0 # seconds

# Defining the number of threads to use
number_threads_needed = 48

# Defining noise specifications
noise_type = 1
noise_amplitude = 0.001

# Defining the number of oscillators
number_oscillators = 100

# Defining filter parameters
order = 2
cutoffLow = 0.01
cutoffHigh = 0.1
sampling_rate = 1 / 0.7

# Defining Bayesian Optimization parameters
n_iterations = 50
n_inner_iterations = 10
n_init_samples = 10
n_iter_relearn = 10
init_method = 1
verbose_level = 2
log_file = "wilson_bo_log.txt"
surr_name = 1 # sGaussianProcessML
sc_type = 0
l_type = 3
l_all = False
epsilon = 0.01
force_jump = 0
crit_name = 0 # cExpectedImprovement

#%% Start main program
if __name__ == "__main__":

    # %% Initial operations - checking, printing, etc.

    # Printing messages for start and beginning timer
    print('Running Wilson-Cowan model...')
    start_time = time.time()

    #%% Load empirical data
    SC_matrix = get_empirical_SC(root_path)
    FC_matrix = get_empirical_FC(root_path)
    BOLD_signals = get_empirical_BOLD(root_path)

    #%% Check number of available threads - multiprocessing tingz

    # Get number of available threads
    number_threads_available = mp.cpu_count()

    # Check if number of threads is greater than available threads
    if number_threads_needed > number_threads_available:
        # If so, set number of threads to available threads
        number_threads_needed = number_threads_available
        # Print message to confirm
        print('Number of threads needed is greater than available threads. Setting number of threads to available threads.')
        print('Number of threads needed: ' + str(number_threads_needed))
        print('Number of threads available: ' + str(number_threads_available))
    else:
        # Otherwise, print message to confirm
        print('Number of threads needed is less than or equal to available threads. Setting number of threads to number of threads needed.')
        print('Number of threads needed: ' + str(number_threads_needed))
        print('Number of threads available: ' + str(number_threads_available))

    #%% Defining the parameters, and beginning the simulation
    number_integration_steps = int(time_simulated / integration_step_size)
    start_save_idx = int(save_data_start / integration_step_size) + downsampling_rate

    initial_conditions_e = np.random.rand(number_oscillators)
    initial_conditions_i = np.random.rand(number_oscillators)

    wilson_params = [
        coupling_strength,
        delay,
        number_oscillators,
        c_ee,
        c_ei,
        c_ie,
        c_ii,
        tau_e,
        tau_i,
        r_e,
        r_i,
        alpha_e,
        alpha_i,
        theta_e,
        theta_i,
        external_e,
        external_i,
        number_integration_steps,
        integration_step_size,
        start_save_idx,
        downsampling_rate,
        initial_conditions_e,
        initial_conditions_i,
        SC_matrix,
        FC_matrix,
        BOLD_signals,
        noise_type,
        noise_amplitude,
        write_path,
        order,
        cutoffLow,
        cutoffHigh,
        sampling_rate,
        n_iterations,
        n_inner_iterations,
        n_init_samples,
        n_iter_relearn,
        init_method,
        verbose_level,
        log_file,
        surr_name,
        sc_type,
        l_type,
        l_all,
        epsilon,
        force_jump,
        crit_name
    ]

    #%% Run the simulation and get results
    
    # Define start time before simulation
    start_time = time.time()
    
    # Run the simulation
    wilson_results = wilson_electrical_sim(wilson_params)

    # Define end time after simulation
    end_time = time.time()

    # Print the time taken for the simulation
    print('Time taken for entire operation: ' + str(end_time - start_time) + ' seconds')
