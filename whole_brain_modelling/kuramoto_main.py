####### KURAMOTO MODEL

#%% Import libraries
import os

hpc = False

if not hpc:
    os.add_dll_directory(r"C:\src\vcpkg\installed\x64-windows\bin")
    os.add_dll_directory(r"C:\cpp_libs\include\bayesopt\build\bin\Release")
    
from sklearn.preprocessing import MinMaxScaler

from py_helpers import *

from collections import OrderedDict
import matplotlib.pyplot as plt
import scipy.signal as signal
import multiprocessing as mp
import scipy.stats as stats
import scipy.io as sio
import numpy as np

try:
    np.distutils.__config__.blas_opt_info = np.distutils.__config__.blas_ilp64_opt_info
except Exception:
    pass

np.bool = np.bool_

import pyGPGO
import time
import json


from pyGPGO.GPGO import GPGO
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.covfunc import matern52
import pymc3 as pm

# Defining paths
if not hpc:
    root_path = 'C:\\Users\\shahi\\OneDrive - Imperial College London\\Documents\\imperial\\Spring Sem\\iso_dubai\\ISO\\HCP_DTI_BOLD'
    write_path = "C:\\Users\\shahi\\OneDrive - Imperial College London\\Documents\\imperial\\Dissertation\\Notebooks\\MyCodes\\whole_brain_modelling\\results\\wilson"
else:
    root_path = '/rds/general/user/hsa22/ephemeral/HCP_DTI_BOLD'
    write_path = os.path.join(os.getcwd(), "results/wilson")

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

# Defining the number of oscillators
number_oscillators = 100

# Defining filter parameters
order = 2
cutoffLow = 0.01
cutoffHigh = 0.1
TR = 0.7

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

# Defining config path
config_path = os.path.join(os.getcwd(), os.path.join("configs", "kuramoto_config.json"))

#%% Start main program
if __name__ == "__main__":

    # %% Initial operations - making config file, starting timer, etc.

    # Derive some parameters for simulation
    number_integration_steps = int(time_simulated / integration_step_size)
    start_save_idx = int(save_data_start / integration_step_size) + downsampling_rate

    # Defining the paths with this data
    SC_path = os.path.join(os.getcwd(), os.path.join("emp_data", "SC_matrix.npy"))
    FC_path = os.path.join(os.getcwd(), os.path.join("emp_data", "FC_matrix.npy"))

    # Parameters for JSON file
    kuramoto_params = [
        number_oscillators,
        number_integration_steps,
        integration_step_size,
        start_save_idx,
        downsampling_rate,
        SC_path,
        FC_path,
        noise_type,
        noise_amplitude,
        write_path,
        order,
        cutoffLow,
        cutoffHigh,
        TR
    ]

    print('Create config of parameters...')
    # Create a JSON file with the parameters
    write_json_config_kura(kuramoto_params, config_path)

    # Get empirical matrices
    print('Getting SC, FC and BOLD matrices...')
    SC_matrix = get_empirical_SC(root_path)
    FC_matrix = get_empirical_FC(root_path, config_path)
    BOLD_signals = get_empirical_BOLD(root_path)

    # Store matrices in .npy files
    print('Storing matrices in .npy files...')
    np.save(os.path.join('emp_data', 'SC_matrix.npy'), SC_matrix)
    np.save(os.path.join('emp_data', 'FC_matrix.npy'), FC_matrix)
    np.save(os.path.join('emp_data', 'BOLD_signals.npy'), BOLD_signals)



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
    # gp = GaussianProcessMCMC(covfunc=cov, 
    #                         niter=300, 
    #                         burnin=100, 
    #                         step=pm.Slice)
    gp = GaussianProcess(covfunc=cov,
                         optimize=True,
                         usegrads=True)
    
    np.random.seed(20)

    print("Define Bayesian Optimization object...")
    gpgo = GPGO(gp, acq, kuramoto_simulator, bo_params)
    gpgo.run(max_iter=n_iterations)

    gpgo.GP.posteriorPlot()

    print("Get results...")
    print(gpgo.getResult())

    # wilson_results = wilson_simulator(coupling_strength=coupling_strength, delay=delay)

    # Define end time after simulation
    end_time = time.time()

    # Print the time taken for the simulation
    print('Time taken for entire operation: ' + str(end_time - start_time) + ' seconds')
