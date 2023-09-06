####### KURAMOTO MODEL

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
parser.add_argument("-bayes", "--bayesian_optimization", help="whether to use Bayesian Optimization or not",
                    action='store_true')

args = parser.parse_args()

streamline_type = args.streamline_type
species = args.species
atlas_type = args.atlas_type
symmetric = args.symmetric
bayesian_optimization = args.bayesian_optimization
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

# Defining integration parameters
time_simulated = 510.000 # seconds
integration_step_size = 0.002 # seconds

# Specifications of coupling strengths
coupling_i = 0.000
coupling_f = 0.945
coupling_n = 64

# Specifications of delays
delay_i = 0.0
delay_f = 47.0
delay_n = 48

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
TR = 0.72

# Defining Bayesian Optimization parameters
n_iterations = 300


#%% Start main program
if __name__ == "__main__":

    # %% Initial operations - making config file, starting timer, etc.

    # Get the main paths
    (SC_root_path, FC_root_path, write_folder, 
     config_folder) = define_paths(hpc, wbm_type="kuramoto")

    # Derive some parameters for simulation
    number_integration_steps = int(time_simulated / integration_step_size)
    start_save_idx = int(save_data_start / integration_step_size) + downsampling_rate

    # Write the initial parameters for the JSON file
    kuramoto_params = [
        number_integration_steps, # 0
        integration_step_size, # 1
        start_save_idx, # 2
        downsampling_rate, # 3
        noise_type, # 4
        noise_amplitude, # 5
        write_folder, # 6
        order, # 7
        cutoffLow, # 8
        cutoffHigh, # 9
        TR, # 10
        species, # 11
        streamline_type, # 12
        atlas_type, # 13
        symmetric # 14
    ]

    print('Create initial config of parameters...')
    config_path = write_initial_config_kura(kuramoto_params, config_folder)

    # Choose current subject to do processing for
    (SUBJECT_SC_PATH, SUBJECT_FC_PATH,
     SUBJECT_LENGTH_PATH) = get_subject_matrices(SC_root_path, FC_root_path, write_folder, 
                                                 streamline_type=streamline_type, 
                                                 atlas_type=atlas_type)

    # Getting the SC matrix just to get number of oscillators
    SC_matrix = get_empirical_SC(SUBJECT_SC_PATH, HPC=hpc, species_type=species, symmetric=symmetric)
    number_of_oscillators = SC_matrix.shape[0]

    # Append the SC and FC matrix paths to the config file
    kuramoto_params = [number_of_oscillators, SUBJECT_SC_PATH, SUBJECT_FC_PATH, SUBJECT_LENGTH_PATH]
    append_SC_FC_to_config(kuramoto_params, config_path)

    #%% Run the simulation and get results
    
    # Define start time before simulation
    print('Running Kuramoto model...')
    start_time = time.time()

    if bayesian_optimization:
        # Bayesian Optimisation
        print("Define Bayesian Optimization parameters...")
        bo_params = OrderedDict()
        bo_params['coupling_strength'] = ('cont', [0.0, 1.0])
        bo_params['delay'] = ('cont', [0.0, 50.0])

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
        gpgo = GPGO(gp, acq, kuramoto_simulator, bo_params)
        gpgo.run(max_iter=n_iterations)

        print("Get results...")
        print(gpgo.getResult())
    
    else:
        # Determine coupling and delay arrays
        coupling_array = np.linspace(coupling_i, coupling_f, coupling_n)
        delay_array = np.linspace(delay_i, delay_f, delay_n)

        # Define the max correlation
        max_corr = 0.0

        for idx, coupling_strength in enumerate(coupling_array):
            for idx2, delay in enumerate(delay_array):

                # Run the simulation
                corr = kuramoto_simulator(coupling_strength, delay)

                # If the correlation is greater than the max correlation, save the coupling and delay
                if corr > max_corr:
                    max_corr = corr

                # Print the correlation
                print("Step {idx1} of {len1} and {idx2} of {len2} completed. Corr: {corr}, Max Corr: {max}".format(idx1=idx, len1=len(coupling_array), 
                                                                                                                    idx2=idx2, len2=len(delay_array), 
                                                                                                                    corr=corr, max=max_corr))
                

    # Define end time after simulation
    end_time = time.time()

    # Print the time taken for the simulation
    print('Time taken for entire operation: ' + str(end_time - start_time) + ' seconds')
