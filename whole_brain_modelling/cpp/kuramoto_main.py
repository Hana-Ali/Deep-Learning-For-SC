####### KURAMOTO MODEL

#%% Import libraries
import os

hpc = False

if hpc == False:
    os.add_dll_directory(r"C:\src\vcpkg\installed\x64-windows\bin")
    os.add_dll_directory(r"C:\cpp_libs\include\bayesopt\build\bin\Release")
    
from py_helpers import *
from interfaces import *

from collections import OrderedDict
import multiprocessing as mp
import numpy as np
import time

try:
    np.distutils.__config__.blas_opt_info = np.distutils.__config__.blas_ilp64_opt_info
except Exception:
    pass

np.bool = np.bool_

# Bayesian Optimization
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
order = 2
cutoffLow = 0.01
cutoffHigh = 0.1
TR = 0.7

# Defining Bayesian Optimization parameters
n_iterations = 50


#%% Start main program
if __name__ == "__main__":

    # %% Initial operations - making config file, starting timer, etc.

    # Get the main paths
    hpc = False
    (SC_FC_root, WBM_main_path, WBM_results_path, config_path, NUMPY_root_path, 
     SC_numpy_root, FC_numpy_root) = define_paths(hpc, wbm_type="kuramoto")

    # Derive some parameters for simulation
    number_integration_steps = int(time_simulated / integration_step_size)
    start_save_idx = int(save_data_start / integration_step_size) + downsampling_rate

    # Write the initial parameters for the JSON file
    kuramoto_params = [
        number_integration_steps,
        integration_step_size,
        start_save_idx,
        downsampling_rate,
        noise_type,
        noise_amplitude,
        order,
        cutoffLow,
        cutoffHigh,
        TR
    ]

    print('Create initial config of parameters...')
    write_initial_config_kura(kuramoto_params, config_path)

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

    # Append the SC and FC matrix paths to the config file
    kuramoto_params = [number_oscillators, write_path, SC_matrix_path, FC_matrix_path]
    append_SC_FC_to_config(kuramoto_params, config_path)

    #%% Check number of available threads - multiprocessing tingz

    # Get number of available threads
    # number_threads_available = mp.cpu_count()

    # # Check if number of threads is greater than available threads
    # if number_threads_needed > number_threads_available:
    #     # If so, set number of threads to available threads
    #     number_threads_needed = number_threads_available
    #     # Print message to confirm
    #     print('Number of threads needed is greater than available threads. Setting number of threads to available threads.')
    #     print('Number of threads needed: ' + str(number_threads_needed))
    #     print('Number of threads available: ' + str(number_threads_available))
    # else:
    #     # Otherwise, print message to confirm
    #     print('Number of threads needed is less than or equal to available threads. Setting number of threads to number of threads needed.')
    #     print('Number of threads needed: ' + str(number_threads_needed))
    #     print('Number of threads available: ' + str(number_threads_available))


    #%% Run the simulation and get results
    
    # Define start time before simulation
    print('Running Kuramoto model...')
    start_time = time.time()
    
    # Bayesian Optimisation
    print("Define Bayesian Optimization parameters...")
    bo_params = OrderedDict()
    bo_params['coupling_strength'] = ('cont', [0.0, 1.0])
    bo_params['delay'] = ('cont', [0.0, 20.0])

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
    best_result = gpgo.getResult()
    
    # Print best result
    print("Best result: " + str(best_result))

    # Save the best result to a file
    print("Save best result to file...")
    best_result_txt = os.path.join(write_path, "best_result.txt")
    with open(best_result_txt, "w") as best_result_file:
        best_result_file.write(str(best_result))

    # Define end time after simulation
    end_time = time.time()

    # Print the time taken for the simulation
    print('Time taken for entire operation: ' + str(end_time - start_time) + ' seconds')
