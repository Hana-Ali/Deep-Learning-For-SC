# Here we want to do some analysis on the topology of the networks
# We want to see how the topology of the networks changes the WBM results

import os
from py_helpers import *

# Function to grab all the SC matrices to do analysis on
def get_sc_fc_matrices(wbm_type="kuramoto"):

    # Get the main paths
    (SC_FC_root, WBM_main_path, WBM_results_path, config_path, NUMPY_root_path, 
     SC_numpy_root, FC_numpy_root) = define_paths(hpc=True, wbm_type=wbm_type)
    
    # Grab all the npy files in the numpy root path
    npy_files = glob_files(NUMPY_root_path, "npy")

    # Grab the SC matrices
    SC_files = [file for file in npy_files if "SC" in file]

    # Grab the FC matrices
    FC_files = [file for file in npy_files if "FC" in file]

    # Return the SC and FC files
    return SC_files, FC_files

# Function to get the WBM results for the SC matrices
def get_wbm_results(SC_files, FC_files, wbm_type="kuramoto"):

    # 
    pass



