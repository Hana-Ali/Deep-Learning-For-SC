# Imports
import matplotlib.pyplot as plt
from py_helpers import *
from wilson_sim_pytorch import *
import pandas as pd
import numpy as np
import sys
import os
import time
import torch

# Defining paths
hpc = False
(SC_FC_root, WBM_main_path, WBM_results_path, config_path, NUMPY_root_path, 
    SC_numpy_root, FC_numpy_root) = define_paths(hpc, wbm_type="kuramoto")

# Define the parameters as a dictionary
params = {
    'c_ee': 16.0,
    'c_ei': 12.0,
    'c_ie': 15.0,
    'c_ii': 3.0,
    'tau_e': 8.0,
    'tau_i': 8.0,
    'r_e': 1.0,
    'r_i': 1.0,
    'k_e': 1.0,
    'k_i': 1.0,
    'alpha_e': 1.0,
    'alpha_i': 1.0,
    'theta_e': 4.0,
    'theta_i': 3.7,
    'external_e': 0.1,
    'external_i': 0.1,
    'coupling_strength': 0.5,
    'delay': 2.0
}

# Defining integration parameters
time_simulated = 510.000 # seconds
integration_step_size = 0.002 # seconds
integration_steps = int(time_simulated / integration_step_size)

# Defining the number of oscillators
number_oscillators = 360

# Defining initial conditions
initial_conditions = torch.tensor(np.random.rand(number_oscillators, 2))

# Choose current subject to do processing for
(SUBJECT_SC_PATH, SUBJECT_FC_PATH) = choose_random_subject(SC_FC_root, NUMPY_root_path)

# Get the SC and FC matrices
(SC_matrix, SC_type) = get_empirical_SC(SUBJECT_SC_PATH, HCP=False)
FC_matrix = get_empirical_FC(SUBJECT_FC_PATH, config_path, HCP=False)

# Get the write path
write_path = get_write_path(SUBJECT_SC_PATH, SC_type, wbm_type="kuramoto")

# Get the number of oscillators
number_oscillators = SC_matrix.shape[0]

# Store the numpy matrices in the numpy arrays folder
(SC_matrix_path, FC_matrix_path) = store_subject_numpy_arrays(SC_matrix, FC_matrix, SUBJECT_SC_PATH, NUMPY_root_path)

# Creating the WC model instance
wc_model = wilson_model(params)

# Defining the simulation parameters as a dictionary
sim_params = {
    'integration_steps' : integration_steps,
    'integration_step_size': integration_step_size,
    'initial_conditions': initial_conditions,
    'number_of_regions': number_oscillators,
    'SC': SC_matrix_path
}

# Start timer
start = time.time()

# Run the simulation
simulation = wc_model.simulator(sim_params)

# End timer
end = time.time()

# Print the time taken
print("Time taken: " + str(end - start) + " seconds")
