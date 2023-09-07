import numpy as np
import json
import glob
import shutil
import os

################################# GENERAL FUNCTIONS #################################

# Check type of array and raise error if wrong
def check_type(inp, input_type, input_name):
    if not isinstance(inp, input_type):
        raise TypeError('The input ' + input_name + ' must be a ' + input_type.__name__ + ', is a ' + type(inp).__name__)

# Check shape of array and raise error if wrong
def check_shape(inp, input_shape, input_name):
    if not inp.shape == input_shape:
        raise ValueError('The input ' + input_name + ' must have shape ' + str(input_shape) + ', has shape ' + str(inp.shape))

def check_all_types(inputs):
    for (inputs, input_type, input_name) in inputs:
        check_type(inputs, input_type, input_name)

def check_all_shapes(inputs):
    for (inputs, input_shape, input_name) in inputs:
        check_shape(inputs, input_shape, input_name)

# Retrieve (GLOB) files
def glob_files(PATH_NAME, file_format):
    INPUT_FILES = []
    for file in glob.glob(os.path.join(PATH_NAME, os.path.join("**", "*.{}".format(file_format))), recursive=True):
        INPUT_FILES.append(file)
    return INPUT_FILES

# Check that output folders are in suitable shape
def check_output_folders(folder, name, wipe=False, verbose=False):
    if not os.path.exists(folder):
        if verbose:
            print("--- {} folder not found. Created folder: {}".format(name, folder))
        os.makedirs(folder)
    # If it has no content, either continue or delete, depending on wipe
    else:
        # If it has content, delete it
        if wipe:
            if verbose:
                print("--- {} folder found. Wiping...".format(name))
            # If the folder has content, delete it
            if len(os.listdir(folder)) != 0:
                if verbose:
                    print("{} folder has content. Deleting content...".format(name))
                # Since this can have both folders and files, we need to check if it's a file or folder to remove
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                if verbose:
                    print("Content deleted. Continuing...")
        else:
            if verbose:
                print("--- {} folder found. Continuing without wipe...".format(name))

# Function to define paths, based on hpc
def define_paths(hpc=False, wbm_type="kuramoto"):
    
    # Defining paths
    if hpc:
        # root: SC_FC root | write: where to write BO results | config: where to write model config
        SC_FC_root = '/rds/general/user/hsa22/ephemeral/CAMCAN/sc_fc_matrices'
        WBM_main_path = '/rds/general/user/hsa22/ephemeral/WBM/'
        WBM_results_path = os.path.join(WBM_main_path, 'results')
        # write_folder = os.path.join(WBM_main_path, 'results', '{wbm}'.format(wbm=wbm_type))
        config_path = os.path.join(WBM_main_path, 'configs', '{wbm}_config.json'.format(wbm=wbm_type))
        # Paths for SC, FC numpy arrays
        NUMPY_root_path = os.path.join(WBM_main_path, 'numpy_arrays')
        SC_numpy_root = ""
        FC_numpy_root = ""
    else:
        SC_root_path = "D:\marmoset_sc_fc\connectomes"
        FC_root_path = "D:\marmoset_sc_fc\mbm_fc"
        # Defining some main folders
        WBM_folder = "D:\WBM"
        write_folder = os.path.join(WBM_folder, "results", "{wbm}".format(wbm=wbm_type))
        config_folder = os.path.join(WBM_folder, "configs", "{wbm}".format(wbm=wbm_type))
        # Ensure that the write and config folders exist
        check_output_folders(write_folder, "Write folder", wipe=False)
        check_output_folders(config_folder, "Config folder", wipe=False)

    # Return the paths
    return (SC_root_path, FC_root_path, write_folder, config_folder)

# Check that output folders with subfolders are in suitable shape
def check_output_folders(folder, name, wipe=True, verbose=False):
    if not os.path.exists(folder):
        if verbose:
            print("--- {} folder not found. Created folder: {}".format(name, folder))
        os.makedirs(folder, exist_ok=True)
    # If it has content, delete it
    else:
        if wipe:
            if verbose:
                print("--- {} folder found. Continuing...".format(name))
            if len(os.listdir(folder)) != 0:
                if verbose:
                    print("{} folder has content. Deleting content...".format(name))
                # Since this can have both folders and files, we need to check if it's a file or folder to remove
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                if verbose:
                    print("Content deleted. Continuing...")
        else:
            if verbose:
                print("--- {} folder found. Continuing without wipe...".format(name))

# Function to get the write path
def get_write_folder(SUBJECT_SC_PATH, SC_type, wbm_type="kuramoto"):

    # Get the subject name
    subject_name = SUBJECT_SC_PATH.split(os.sep)[-1]

    # Get the general paths
    (SC_FC_root, WBM_main_path, WBM_results_path, config_path, NUMPY_root_path, 
     SC_numpy_root, FC_numpy_root) = define_paths(hpc=False, wbm_type=wbm_type)

    # Get the write path
    write_folder = os.path.join(WBM_results_path, subject_name, SC_type, '{wbm}'.format(wbm=wbm_type))
    check_output_folders(write_folder, "Write path", wipe=False)

    # Return the write path
    return write_folder 


# Function to choose a random subject from the path
def get_subject_matrices(SC_root_path, FC_root_path, write_folder,
                        streamline_type="tracer", atlas_type="MBCA"):

    # Grab all the csv files in the SC root path and npy files in the FC root path
    SC_files = glob.glob(os.path.join(SC_root_path, "*.csv"))
    FC_files = glob.glob(os.path.join(FC_root_path, "*.npy"))

    # Get all the path length files
    LENGTH_files = [file for file in SC_files if "pathlength" in file]

    # Remove all path length files from the list
    SC_files = [file for file in SC_files if file not in LENGTH_files]

    # Get filtered SC, FC and LENGTH files
    SC_files = filter_sc_fc_data(SC_files, streamline_type, atlas_type, "both")
    FC_files = filter_sc_fc_data(FC_files, streamline_type, atlas_type, "atlas")
    LENGTH_files = filter_sc_fc_data(LENGTH_files, streamline_type, atlas_type, "both")

    # Make sure we don't get the BOLD files in the FC files - get the correlation files
    FC_files = [file for file in FC_files if "corr" in file]

    # Assert that all of them got only ONE file
    assert len(SC_files) == 1, "SC files must have length 1, has length {}".format(len(SC_files))
    assert len(FC_files) == 1, "FC files must have length 1, has length {}".format(len(FC_files))
    assert len(LENGTH_files) == 1, "LENGTH files must have length 1, has length {}".format(len(LENGTH_files))

    # Get the first item of the SC, FC and LENGTH files
    SC_path = SC_files[0]
    FC_path = FC_files[0]
    LENGTH_path = LENGTH_files[0]

    return (SC_path, FC_path, LENGTH_path)

# Function to filter for a specific streamline type and connectome type
def filter_sc_fc_data(files, streamline_type, atlas_type, what_to_filter="atlas"):

    # If we're filtering for the atlas type
    if what_to_filter == "atlas":
        # Filter for the atlas type
        files = [file for file in files if atlas_type in file]
    # If we're filtering for the streamline type
    elif what_to_filter == "streamline":
        # If the streamline type is optimized
        if streamline_type == "optimized":
            # Filter for ones that don't have unoptimized but have optimized
            files = [file for file in files if "unoptimized" not in file and "optimized" in file]
        # If the streamline type is unoptimized, continue as normal
        files = [file for file in files if streamline_type in file]
    # If we're filtering for both
    elif what_to_filter == "both":
        # If the streamline type is optimized
        if streamline_type == "optimized":
            # Filter for ones that don't have unoptimized but have optimized
            files = [file for file in files if "unoptimized" not in file and "optimized" in file]
        # If the streamline type is unoptimized, continue as normal
        files = [file for file in files if streamline_type in file]
        # Filter for the connectome type
        files = [file for file in files if atlas_type in file]
    else:
        raise ValueError("what_to_filter must be either 'atlas', 'streamline' or 'both'")

    # Return the filtered files
    return files

# Function to get a random subject
def choose_random_subject(files, write_folder):

    # Get the list of directories in the write folder to see what subjects have already been processed
    processed_subjects = os.listdir(write_folder)

    # Remove the processed subjects from the list of subjects
    remaining_subjects = [file for file in files if file.split(os.sep)[-2] not in processed_subjects]

    # Choose a random subject
    current_subject = np.random.choice(remaining_subjects)

    # Return the current subject
    return current_subject

# Function to flip the matrix
def flip_matrix(matrix):

    # Flip the matrix
    flipped_matrix = np.flip(matrix, axis=0)

    # Flip the matrix again
    flipped_matrix = np.flip(flipped_matrix, axis=1)

    # Add the original to the flipped
    flipped_matrix = flipped_matrix + matrix

    # Return the flipped matrix
    return flipped_matrix

# Function to store the numpy arrays
def store_subject_numpy_arrays(SC_matrix, FC_matrix, SUBJECT_SC_PATH, NUMPY_root_path):

    # Get the subject name
    subject_name = SUBJECT_SC_PATH.split(os.sep)[-1]

    # Create the numpy arrays folder if it doesn't exist
    SUBJECT_NUMPY_PATH = os.path.join(NUMPY_root_path, subject_name)
    check_output_folders(SUBJECT_NUMPY_PATH, "Subject numpy arrays", wipe=False)

    # Define the SC and FC matrix paths
    SC_matrix_path = os.path.join(SUBJECT_NUMPY_PATH, 'SC_matrix.npy')
    FC_matrix_path = os.path.join(SUBJECT_NUMPY_PATH, 'FC_matrix.npy')

    # Store the numpy arrays
    np.save(SC_matrix_path, SC_matrix)
    np.save(FC_matrix_path, FC_matrix)

    # Return the paths
    return (SC_matrix_path, FC_matrix_path)