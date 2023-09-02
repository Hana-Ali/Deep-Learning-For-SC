####### WILSON-COWAN MODEL

#%% Import libraries
import os

import argparse

# Define allowed types for the streamline_type argument
allowed_streamline_types = ["model", "traditional", "tracer"]
# Define allowed types for the species argument
allowed_species = ["marmoset"]
# Define allowed types for the atlas_type argument
allowed_atlas_types = ["MBM"]

parser = argparse.ArgumentParser(description="Define stuff for running the model")
parser.add_argument("-st", "--streamline_type", help="whether to do WBM for model, traditional tractography or tracer streamlines",
                    default="tracer", required=True,
                    type=str,
                    choices=allowed_streamline_types)
parser.add_argument("-s", "--species", help="what species we're predicting for", 
                    default="marmoset", required=False,
                    type=str,
                    choices=allowed_species)
parser.add_argument("-a", "--atlas_type", help="what type of atlas to use", 
                    default="MBCA", required=True,
                    type=str,
                    choices=allowed_atlas_types)
parser.add_argument("-sym", "--symmetric", help="whether to use symmetric SC matrix or not",
                    action='store_true')

args = parser.parse_args()

streamline_type = args.streamline_type
species = args.species
atlas_type = args.atlas_type
symmetric = args.symmetric

hpc=False

if not hpc:
    os.add_dll_directory(r"C:\src\vcpkg\installed\x64-windows\bin")
    
from py_helpers import *
from interfaces import *

from collections import OrderedDict
import numpy as np

try:
    np.distutils.__config__.blas_opt_info = np.distutils.__config__.blas_ilp64_opt_info
except Exception:
    pass

np.bool = np.bool_

import time


from pyGPGO.GPGO import GPGO
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.covfunc import matern52

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

# Defining filter parameters
order = 6
cutoffLow = 0.01
cutoffHigh = 0.1
TR = 0.7

# Defining Bayesian Optimization parameters
n_iterations = 300


#%% Start main program
if __name__ == "__main__":

    # %% Initial operations - making config file, starting timer, etc.

    # Get the main paths
    (SC_root_path, FC_root_path, write_folder, 
     config_folder) = define_paths(hpc, wbm_type="wilson")

    # Derive some parameters for simulation
    number_integration_steps = int(time_simulated / integration_step_size)
    start_save_idx = int(save_data_start / integration_step_size) + downsampling_rate


    # Parameters for JSON file
    wilson_params = [
        c_ee, # 0
        c_ei, # 1
        c_ie, # 2
        c_ii, # 3
        tau_e, # 4
        tau_i, # 5
        r_e, # 6
        r_i, # 7
        alpha_e, # 8
        alpha_i, # 9
        theta_e, # 10
        theta_i, # 11
        external_e, # 12
        external_i, # 13
        number_integration_steps, # 14
        integration_step_size, # 15
        start_save_idx, # 16
        downsampling_rate, # 17
        noise_type, # 18
        noise_amplitude, # 19
        write_folder, # 20
        order, # 21
        cutoffLow, # 22
        cutoffHigh, # 23
        TR, # 24
        species, # 25
        streamline_type, # 26
        atlas_type, # 27
        symmetric # 28
    ]

    print('Create initial config of parameters...')
    config_path = write_initial_config_wilson(wilson_params, config_folder)

    # Choose current subject to do processing for
    (SUBJECT_SC_PATH, SUBJECT_FC_PATH,
     SUBJECT_LENGTH_PATH) = get_subject_matrices(SC_root_path, FC_root_path, write_folder, 
                                                 streamline_type=streamline_type, 
                                                 atlas_type=atlas_type)

    # Getting the SC matrix just to get number of oscillators
    SC_matrix = get_empirical_SC(SUBJECT_SC_PATH, HPC=hpc, species_type=species, symmetric=symmetric)
    number_of_oscillators = SC_matrix.shape[0]

    # Append the SC and FC matrix paths to the config file
    wilson_params = [number_of_oscillators, SUBJECT_SC_PATH, SUBJECT_FC_PATH, SUBJECT_LENGTH_PATH]
    append_SC_FC_to_config(wilson_params, config_path)

    #%% Run the simulation and get results
    
    # Define start time before simulation
    print('Running Wilson-Cowan model...')
    start_time = time.time()
    
    # Bayesian Optimisation
    print("Define Bayesian Optimization parameters...")
    bo_params = OrderedDict()
    bo_params['coupling_strength'] = ('cont', [0.0, 1.0])
    bo_params['delay'] = ('cont', [0.0, 100.0])

    print("Define acquisition function...")
    acq = Acquisition(mode='ExpectedImprovement')

    print("Define covariance function...")
    cov = matern52()

    print("Define surrogate model...")
    gp = GaussianProcess(covfunc=cov,
                         optimize=True,
                         usegrads=True)
    
    np.random.seed(20)

    print("Define Bayesian Optimization object...")
    gpgo = GPGO(gp, acq, wilson_simulator, bo_params)
    gpgo.run(max_iter=n_iterations)

    print("Get results...")
    print(gpgo.getResult())

    # Define end time after simulation
    end_time = time.time()

    # Print the time taken for the simulation
    print('Time taken for entire operation: ' + str(end_time - start_time) + ' seconds')
