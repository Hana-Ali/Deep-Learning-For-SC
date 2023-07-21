# Imports
import matplotlib.pyplot as plt
from py_helpers import *
from simulators import *
import pandas as pd
import numpy as np
import sys
import os
import time
import torch

def initialize_model(N, neighbor_radius, random_edges, variance):

    wcn, edge_list = nazemi_jamali_network(N, neighbor_radius=neighbor_radius, random_edges=random_edges)

    θE = -2
    θI = 8
    wcn.excitatory_firing_rate = lambda x: relu(x - θE)
    wcn.inhibitory_firing_rate = lambda x: relu(x - θI)

    wcn.excitatory_variance = variance
    wcn.inhibitory_variance = 0
    
    return wcn, edge_list

# Defining paths
hpc = False
(SC_FC_root, WBM_main_path, WBM_results_path, config_path, NUMPY_root_path, 
    SC_numpy_root, FC_numpy_root) = define_paths(hpc, wbm_type="kuramoto")

# Define the parameters as a dictionary
params = {
    'c_ee': 8.0,
    'c_ei': 16.0,
    'c_ie': 8.0,
    'c_ii': 4.0,
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

# Choose current subject to do processing for
(SUBJECT_SC_PATH, SUBJECT_FC_PATH) = choose_random_subject(SC_FC_root, NUMPY_root_path)

# Get the SC and FC matrices
(SC_matrix, SC_type) = get_empirical_SC(SUBJECT_SC_PATH, HCP=False)
FC_matrix = get_empirical_FC(SUBJECT_FC_PATH, config_path, HCP=False)

# Get the write path
write_path = get_write_path(SUBJECT_SC_PATH, SC_type, wbm_type="kuramoto")

# Get the number of oscillators
number_oscillators = SC_matrix.shape[0]

# Define the noise type
noise_type = "normal"
noise_amplitude = 0.01

# Defining initial conditions
initial_conditions = torch.tensor(np.random.rand(number_oscillators, 2))

# Store the numpy matrices in the numpy arrays folder
(SC_matrix_path, FC_matrix_path) = store_subject_numpy_arrays(SC_matrix, FC_matrix, SUBJECT_SC_PATH, NUMPY_root_path)

Ns = [10]
radii = [1]
random_edges_list = [5]
variances = [0.01]
trials = 10

t_final = 10
time_span = (t_final/2, t_final)

num_experiments = len(list(product(Ns, radii, random_edges_list, variances, range(trials))))
results_key = [
    'experiment_index', 
    'trial',
    'N', 
    'radius', 
    'random_edges', 
    'variance', 
    'trial',
    'kuramoto',
    'pearson',
    'clustering_coefficient',
    'mean_path_length',
    'diameter'
]

errors_list = []
results_list = []

for experiment_index, (N, radius, random_edges, variance, trial) in enumerate(product(Ns, radii, random_edges_list, variances, range(trials))):
    print(f'{experiment_index+1}/{num_experiments}' + ' '*100, end='\r')
    m = radius*N
    if random_edges is 'max':
        random_edges = m
    elif random_edges >= m:
        errors_list.append(experiment_index)
        continue #skip if too many random edges
    wcn, edge_list = initialize_model(N, radius, random_edges, variance)
    ts, Es, Is = wcn.simulate(t_final, Δt = 1e-3)
    print("Es: ", Es)
    results_list.append([ts, Es, Is])
    print("Es.shape: ", Es.shape)
    # Balloon
    bold_model = BOLDModel(Es.shape[0], Es)
    print("bold_model.N: ", bold_model.N)
    bold_model.run(Es)
    print("ran bold")
    print("bold_model.BOLD.shape: ", bold_model.BOLD.shape)
    # Save plot of BOLD
    plt.figure(figsize=(20, 10))
    plt.plot(bold_model.BOLD)
    plt.savefig("BOLD{index}.png".format(index=experiment_index))

# # Creating the WC model instance
# wc_model = wilson_model(params)

# # Defining the simulation parameters as a dictionary
# sim_params = {
#     'integration_steps' : integration_steps,
#     'integration_step_size': integration_step_size,
#     'initial_conditions': initial_conditions,
#     'number_of_regions': number_oscillators,
#     'noise_amplitude': noise_amplitude,
#     'noise_type' : noise_type,
#     'SC': SC_matrix
# }

# # Start timer
# start = time.time()

# # Run the simulation
# simulation = wc_model.simulator(sim_params)

# # End timer
# end = time.time()

# # Print the time taken
# print("Time taken: " + str(end - start) + " seconds")

# downsampling_rate = 400
# save_data_start = 150.0 # seconds
# start_save_idx = int(save_data_start / integration_step_size) + downsampling_rate

# bold_down1 = simulation.numpy()[:, start_save_idx - downsampling_rate + 1 :]

# print("simulation shape", bold_down1.shape)
# print("simulation[0,:].shape", bold_down1[0, :].shape)
# # Save the figure of the simulation
# plt.figure(figsize=(20, 10))
# plt.plot(bold_down1[0, :])
# plt.savefig("simulation.png")

# # Get the BOLD signal
# (BOLD, s, f, v, q) = balloonWindkessel(simulation[0, :].numpy(), sampling_rate=0.72)

# # Recast it into a numpy array
# BOLD = np.array(BOLD)

# print("BOLD shape: ", BOLD.shape)

# # Concatenate along the first axis
# BOLD = np.concatenate(BOLD, axis=0)

# print("BOLD shape: ", BOLD.shape)

# print("BOLD: ", BOLD)


# # Find the correlation
# corr = np.corrcoef(BOLD)

# print("Correlation shape: ", corr.shape)

# # Plot the correlation and save image
# plt.imshow(corr)
# plt.savefig(write_path + "corr.png")
