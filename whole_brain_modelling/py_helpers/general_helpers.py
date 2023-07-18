import numpy as np
import json
import glob
import shutil
import os

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

# Function to define paths, based on hpc
def define_paths(hpc=True, wbm_type="kuramoto"):
    
    # Defining paths
    if hpc:
        # root: SC_FC root | write: where to write BO results | config: where to write model config
        SC_FC_root = '/rds/general/user/hsa22/ephemeral/CAMCAN/sc_fc_matrices'
        WBM_main_path = '/rds/general/user/hsa22/ephemeral/WBM/'
        WBM_results_path = os.path.join(WBM_main_path, 'results')
        # write_path = os.path.join(WBM_main_path, 'results', '{wbm}'.format(wbm=wbm_type))
        config_path = os.path.join(WBM_main_path, 'configs', '{wbm}_config.json'.format(wbm=wbm_type))
        # Paths for SC, FC numpy arrays
        NUMPY_root_path = os.path.join(WBM_main_path, 'numpy_arrays')
        SC_numpy_root = ""
        FC_numpy_root = ""
    else:
        # root: SC_FC root | write: where to write BO results | config: where to write model config
        SC_FC_root = 'C:\\Users\\shahi\\OneDrive - Imperial College London\\Documents\\imperial\\Spring Sem\\iso_dubai\\ISO\\HCP_DTI_BOLD'
        # write_path = "C:\\Users\\shahi\\OneDrive - Imperial College London\\Documents\\imperial\\Dissertation\\Notebooks\\MyCodes\\whole_brain_modelling\\results\\wilson"
        config_path = os.path.join(os.getcwd(), os.path.join("configs", "{wbm}_config.json".format(wbm=wbm_type)))
        # Paths for SC, FC numpy arrays
        NUMPY_root_path = os.path.join(os.getcwd(), "emp_data")
        SC_numpy_root = os.path.join(NUMPY_root_path, "SC_matrix.npy")
        FC_numpy_root = os.path.join(NUMPY_root_path, "FC_matrix.npy")

    # Return the paths
    return (SC_FC_root, WBM_main_path, WBM_results_path, config_path, NUMPY_root_path, SC_numpy_root, FC_numpy_root)

# Function to get the write path
def get_write_path(SUBJECT_SC_PATH, SC_type, wbm_type="kuramoto"):

    # Get the subject name
    subject_name = SUBJECT_SC_PATH.split(os.sep)[-1]

    # Get the general paths
    (SC_FC_root, WBM_main_path, WBM_results_path, config_path, NUMPY_root_path, 
     SC_numpy_root, FC_numpy_root) = define_paths(hpc=True, wbm_type=wbm_type)

    # Get the write path
    write_path = os.path.join(WBM_results_path, subject_name, SC_type, '{wbm}'.format(wbm=wbm_type))
    check_output_folders(write_path, "Write path", wipe=False)

    # Return the write path
    return write_path 


# Function to choose a random subject from the path
def choose_random_subject(SC_FC_root, NUMPY_root_path):

    # Define the main SC and main FC paths
    MAIN_PATH_SC = os.path.join(SC_FC_root, 'structural')
    MAIN_PATH_FC = os.path.join(SC_FC_root, 'functional')

    # Get the list of subjects - doesn't matter whether we use structural or function right now
    subjects = os.listdir(MAIN_PATH_SC)

    # Get the list of subjects that have already been processed
    processed_subjects = os.listdir(NUMPY_root_path)

    # Remove the processed subjects from the list of subjects
    subjects = [subject for subject in subjects if subject not in processed_subjects]

    # Choose a random subject
    current_subject = np.random.choice(subjects)

    # Get the path to the current subject
    SUBJECT_STRUCTURAL_PATH = os.path.join(MAIN_PATH_SC, current_subject)
    SUBJECT_FUNCTIONAL_PATH = os.path.join(MAIN_PATH_FC, current_subject)

    return (SUBJECT_STRUCTURAL_PATH, SUBJECT_FUNCTIONAL_PATH)

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

# Function to create the JSON config file
def write_initial_config_wilson(params, config_path):

    # Get the parameters
    number_oscillators = params[0]
    c_ee = params[1]
    c_ei = params[2]
    c_ie = params[3]
    c_ii = params[4]
    tau_e = params[5]
    tau_i = params[6]
    r_e = params[7]
    r_i = params[8]
    alpha_e = params[9]
    alpha_i = params[10]
    theta_e = params[11]
    theta_i = params[12]
    external_e = params[13]
    external_i = params[14]
    number_integration_steps = params[15]
    integration_step_size = params[16]
    start_save_idx = params[17]
    downsampling_rate = params[18]
    noise_type = params[19]
    noise_amplitude = params[20]
    write_path = params[21]
    order = params[22]
    cutoffLow = params[23]
    cutoffHigh = params[24]
    TR = params[25]

    # Check that the input arguments are of the correct type
    check_all_types([
        (number_oscillators, int, 'number_oscillators'),
        (c_ee, float, 'c_ee'),
        (c_ei, float, 'c_ei'),
        (c_ie, float, 'c_ie'),
        (c_ii, float, 'c_ii'),
        (tau_e, float, 'tau_e'),
        (tau_i, float, 'tau_i'),
        (r_e, float, 'r_e'),
        (r_i, float, 'r_i'),
        (alpha_e, float, 'alpha_e'),
        (alpha_i, float, 'alpha_i'),
        (theta_e, float, 'theta_e'),
        (theta_i, float, 'theta_i'),
        (external_e, float, 'external_e'),
        (external_i, float, 'external_i'),
        (number_integration_steps, int, 'number_integration_steps'),
        (integration_step_size, float, 'integration_step_size'),
        (start_save_idx, int, 'start_save_idx'),
        (downsampling_rate, int, 'downsampling_rate'),
        (noise_type, int, 'noise_type'),
        (noise_amplitude, float, 'noise_amplitude'),
        (write_path, str, 'write_path'),
        (order, int, 'order'),
        (cutoffLow, float, 'cutoffLow'),
        (cutoffHigh, float, 'cutoffHigh'),
        (TR, float, 'TR')
    ])

    # Create the dictionary
    config = {
        "number_oscillators": number_oscillators,
        "c_ee": c_ee,
        "c_ei": c_ei,
        "c_ie": c_ie,
        "c_ii": c_ii,
        "tau_e": tau_e,
        "tau_i": tau_i,
        "r_e": r_e,
        "r_i": r_i,
        "alpha_e": alpha_e,
        "alpha_i": alpha_i,
        "theta_e": theta_e,
        "theta_i": theta_i,
        "external_e": external_e,
        "external_i": external_i,
        "number_integration_steps": number_integration_steps,
        "integration_step_size": integration_step_size,
        "start_save_idx": start_save_idx,
        "downsampling_rate": downsampling_rate,
        "noise_type": noise_type,
        "noise_amplitude": noise_amplitude,
        "write_path": write_path,
        "order": order,
        "cutoffLow": cutoffLow,
        "cutoffHigh": cutoffHigh,
        "TR": TR
    }

    # Dump as a string
    config_string = json.dumps(config, indent=4)

    # Write the dictionary to a JSON file
    with open(config_path, 'w') as outfile:
        outfile.write(config_string)

# Function to create the JSON config file - Kuramoto
def write_initial_config_kura(params, config_path):
    # Get the parameters
    number_integration_steps = params[0]
    integration_step_size = params[1]
    start_save_idx = params[2]
    downsampling_rate = params[3]
    noise_type = params[4]
    noise_amplitude = params[5]
    order = params[6]
    cutoffLow = params[7]
    cutoffHigh = params[8]
    TR = params[9]

    # Check that the input arguments are of the correct type
    check_all_types([
        (number_integration_steps, int, 'number_integration_steps'),
        (integration_step_size, float, 'integration_step_size'),
        (start_save_idx, int, 'start_save_idx'),
        (downsampling_rate, int, 'downsampling_rate'),
        (noise_type, int, 'noise_type'),
        (noise_amplitude, float, 'noise_amplitude'),
        (order, int, 'order'),
        (cutoffLow, float, 'cutoffLow'),
        (cutoffHigh, float, 'cutoffHigh'),
        (TR, float, 'TR')
    ])

    # Create the dictionary
    config = {
        "number_integration_steps": number_integration_steps,
        "integration_step_size": integration_step_size,
        "start_save_idx": start_save_idx,
        "downsampling_rate": downsampling_rate,
        "noise_type": noise_type,
        "noise_amplitude": noise_amplitude,
        "order": order,
        "cutoffLow": cutoffLow,
        "cutoffHigh": cutoffHigh,
        "TR": TR
    }

    # Dump as a string
    config_string = json.dumps(config, indent=4)

    # Write the dictionary to a JSON file
    with open(config_path, 'w') as outfile:
        outfile.write(config_string)

# Function to add SC and FC to the JSON config file
def append_SC_FC_to_config(params, config_path):

    # Get the parameters
    number_oscillators = params[0]
    write_path = params[1]
    SC_path = params[2]
    FC_path = params[3]

    # Check that the input arguments are of the correct type
    check_all_types([
        (number_oscillators, int, 'number_oscillators'),
        (write_path, str, 'write_path'),
        (SC_path, str, 'SC_path'),
        (FC_path, str, 'FC_path')
    ])

    # Create the dictionary
    config = {
        "number_oscillators": number_oscillators,
        "write_path": write_path,
        "SC_path": SC_path,
        "FC_path": FC_path
    }

    with open(config_path,'r') as f:
        current_dictionary = json.load(f)

    current_dictionary.update(config)

    with open(config_path,'w') as f:
        json.dump(current_dictionary, f, indent=4)

# Function to read the JSON config file
def read_json_config(config_path):
    
    # Check that config path exists
    if not os.path.exists(config_path):
        raise ValueError('The input config_path does not exist')
    
    # Read the JSON file
    with open(config_path) as json_file:
        config = json.load(json_file)
    
    # Check that the input arguments are of the correct type
    check_all_types([
        (config["number_oscillators"], int, 'number_oscillators'),
        (config["c_ee"], float, 'c_ee'),
        (config["c_ei"], float, 'c_ei'),
        (config["c_ie"], float, 'c_ie'),
        (config["c_ii"], float, 'c_ii'),
        (config["tau_e"], float, 'tau_e'),
        (config["tau_i"], float, 'tau_i'),
        (config["r_e"], float, 'r_e'),
        (config["r_i"], float, 'r_i'),
        (config["alpha_e"], float, 'alpha_e'),
        (config["alpha_i"], float, 'alpha_i'),
        (config["theta_e"], float, 'theta_e'),
        (config["theta_i"], float, 'theta_i'),
        (config["external_e"], float, 'external_e'),
        (config["external_i"], float, 'external_i'),
        (config["number_integration_steps"], int, 'number_integration_steps'),
        (config["integration_step_size"], float, 'integration_step_size'),
        (config["start_save_idx"], int, 'start_save_idx'),
        (config["downsampling_rate"], int, 'downsampling_rate'),
        (config["SC_path"], str, 'SC_path'),
        (config["FC_path"], str, 'FC_path'),
        (config["noise_type"], int, 'noise_type'),
        (config["noise_amplitude"], float, 'noise_amplitude'),
        (config["write_path"], str, 'write_path'),
        (config["order"], int, 'order'),
        (config["cutoffLow"], float, 'cutoffLow'),
        (config["cutoffHigh"], float, 'cutoffHigh'),
        (config["TR"], float, 'TR')
    ])

    return config

# Function to read the JSON config file - Kuramoto
def read_json_config_kura(config_path):

    # Check that config path exists
    if not os.path.exists(config_path):
        raise ValueError('The input config_path does not exist')
    
    # Read the JSON file
    with open(config_path) as json_file:
        config = json.load(json_file)

    # Check that the input arguments are of the correct type
    check_all_types([
        (config["number_oscillators"], int, 'number_oscillators'),
        (config["number_integration_steps"], int, 'number_integration_steps'),
        (config["integration_step_size"], float, 'integration_step_size'),
        (config["start_save_idx"], int, 'start_save_idx'),
        (config["downsampling_rate"], int, 'downsampling_rate'),
        (config["noise_type"], int, 'noise_type'),
        (config["noise_amplitude"], float, 'noise_amplitude'),
        (config["write_path"], str, 'write_path'),
        (config["order"], int, 'order'),
        (config["cutoffLow"], float, 'cutoffLow'),
        (config["cutoffHigh"], float, 'cutoffHigh'),
        (config["TR"], float, 'TR'),
        (config["SC_path"], str, 'SC_path'),
        (config["FC_path"], str, 'FC_path'),
    ])

    return config