import json
from .general_helpers import check_all_types, check_output_folders
import os

# Function to create the JSON config file
def write_initial_config_wilson(params, config_folder):

    # Get the parameters
    c_ee = params[0]
    c_ei = params[1]
    c_ie = params[2]
    c_ii = params[3]
    tau_e = params[4]
    tau_i = params[5]
    r_e = params[6]
    r_i = params[7]
    alpha_e = params[8]
    alpha_i = params[9]
    theta_e = params[10]
    theta_i = params[11]
    external_e = params[12]
    external_i = params[13]
    number_integration_steps = params[14]
    integration_step_size = params[15]
    start_save_idx = params[16]
    downsampling_rate = params[17]
    noise_type = params[18]
    noise_amplitude = params[19]
    write_folder = params[20]
    order = params[21]
    cutoffLow = params[22]
    cutoffHigh = params[23]
    TR = params[24]
    species = params[25]
    streamline_type = params[26]
    connectome_type = params[27]
    symmetric = params[28]

    # Check that the input arguments are of the correct type
    check_all_types([
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
        (write_folder, str, 'write_folder'),
        (order, int, 'order'),
        (cutoffLow, float, 'cutoffLow'),
        (cutoffHigh, float, 'cutoffHigh'),
        (TR, float, 'TR'),
        (species, str, 'species'),
        (streamline_type, str, 'streamline_type'),
        (connectome_type, str, 'connectome_type'),
        (symmetric, bool, 'symmetric')
    ])

    # Create the dictionary
    config = {
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
        "write_folder": write_folder,
        "order": order,
        "cutoffLow": cutoffLow,
        "cutoffHigh": cutoffHigh,
        "TR": TR,
        "species": species,
        "streamline_type": streamline_type,
        "connectome_type": connectome_type,
        "symmetric": symmetric
    }

    # Dump as a string
    config_string = json.dumps(config, indent=4)

    # Define the symmetric string
    symmetric_str = "symm" if symmetric else "half"

    # Add to the config folder the species, streamline_type, connectome_type, and symmetric
    config_folder = os.path.join(config_folder, species, streamline_type, connectome_type, symmetric_str)
    check_output_folders(config_folder, "config folder", wipe=False)

    # Define the config path
    config_path = os.path.join(config_folder, "config.json")

    # Write the dictionary to a JSON file
    with open(config_path, 'w') as outfile:
        outfile.write(config_string)

    # Return the config path
    return config_path

# Function to create the JSON config file - Kuramoto
def write_initial_config_kura(params, config_folder):
    # Get the parameters
    number_integration_steps = params[0]
    integration_step_size = params[1]
    start_save_idx = params[2]
    downsampling_rate = params[3]
    noise_type = params[4]
    noise_amplitude = params[5]
    write_folder = params[6]
    order = params[7]
    cutoffLow = params[8]
    cutoffHigh = params[9]
    TR = params[10]
    species = params[11]
    streamline_type = params[12]
    connectome_type = params[13]
    symmetric = params[14]

    # Check that the input arguments are of the correct type
    check_all_types([
        (number_integration_steps, int, 'number_integration_steps'),
        (integration_step_size, float, 'integration_step_size'),
        (start_save_idx, int, 'start_save_idx'),
        (downsampling_rate, int, 'downsampling_rate'),
        (noise_type, int, 'noise_type'),
        (noise_amplitude, float, 'noise_amplitude'),
        (write_folder, str, 'write_folder'),
        (order, int, 'order'),
        (cutoffLow, float, 'cutoffLow'),
        (cutoffHigh, float, 'cutoffHigh'),
        (TR, float, 'TR'),
        (species, str, 'species'),
        (streamline_type, str, 'streamline_type'),
        (connectome_type, str, 'connectome_type'),
        (symmetric, bool, 'symmetric')
    ])

    # Create the dictionary
    config = {
        "number_integration_steps": number_integration_steps,
        "integration_step_size": integration_step_size,
        "start_save_idx": start_save_idx,
        "downsampling_rate": downsampling_rate,
        "noise_type": noise_type,
        "noise_amplitude": noise_amplitude,
        "write_folder": write_folder,
        "order": order,
        "cutoffLow": cutoffLow,
        "cutoffHigh": cutoffHigh,
        "TR": TR,
        "species": species,
        "streamline_type": streamline_type,
        "connectome_type": connectome_type,
        "symmetric": symmetric
    }

    # Dump as a string
    config_string = json.dumps(config, indent=4)

    # Define the symmetric string
    symmetric_str = "symm" if symmetric else "half"

    # Add to the config folder the species, streamline_type, connectome_type, and symmetric
    config_folder = os.path.join(config_folder, species, streamline_type, connectome_type, symmetric_str)
    check_output_folders(config_folder, "config folder", wipe=False)

    # Define the config path
    config_path = os.path.join(config_folder, "config.json")

    # Write the dictionary to a JSON file
    with open(config_path, 'w') as outfile:
        outfile.write(config_string)

    # Return the config path
    return config_path

# Function to read the JSON config file
def read_json_config_wilson(config_folder):

    # Grab the json files in the config folder
    json_files = []
    for root, dirs, files in os.walk(config_folder):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))

    # Check that there are JSON files in the config folder
    if not json_files:
        raise ValueError("No JSON files found in the provided directory and its subdirectories.")

    # Sort the JSON files based on modification time and get the most recent one
    try:
        json_files.sort(key=os.path.getmtime)
        config_path = json_files[-1]
    except Exception as e:
        print(f"Error while sorting or accessing the JSON files: {e}")
    
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
        (config["LENGTH_path"], str, 'LENGTH_path'),
        (config["noise_type"], int, 'noise_type'),
        (config["noise_amplitude"], float, 'noise_amplitude'),
        (config["write_folder"], str, 'write_folder'),
        (config["order"], int, 'order'),
        (config["cutoffLow"], float, 'cutoffLow'),
        (config["cutoffHigh"], float, 'cutoffHigh'),
        (config["TR"], float, 'TR'),
        (config["species"], str, 'species'),
        (config["streamline_type"], str, 'streamline_type'),
        (config["connectome_type"], str, 'connectome_type'),
        (config["symmetric"], bool, 'symmetric')
    ])

    return config, config_path

# Function to read the JSON config file - Kuramoto
def read_json_config_kuramoto(config_folder):

    # Grab the json files in the config folder
    json_files = []
    for root, dirs, files in os.walk(config_folder):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))

    # Check that there are JSON files in the config folder
    if not json_files:
        raise ValueError("No JSON files found in the provided directory and its subdirectories.")

    # Sort the JSON files based on modification time and get the most recent one
    try:
        json_files.sort(key=os.path.getmtime)
        config_path = json_files[-1]
    except Exception as e:
        print(f"Error while sorting or accessing the JSON files: {e}")
    
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
        (config["SC_path"], str, 'SC_path'),
        (config["FC_path"], str, 'FC_path'),
        (config["LENGTH_path"], str, 'LENGTH_path'),
        (config["noise_type"], int, 'noise_type'),
        (config["noise_amplitude"], float, 'noise_amplitude'),
        (config["write_folder"], str, 'write_folder'),
        (config["order"], int, 'order'),
        (config["cutoffLow"], float, 'cutoffLow'),
        (config["cutoffHigh"], float, 'cutoffHigh'),
        (config["TR"], float, 'TR'),
        (config["species"], str, 'species'),
        (config["streamline_type"], str, 'streamline_type'),
        (config["connectome_type"], str, 'connectome_type'),
        (config["symmetric"], bool, 'symmetric')
    ])

    return config, config_path

# Function to add SC and FC to the JSON config file
def append_SC_FC_to_config(params, config_path):

    # Get the parameters
    number_oscillators = params[0]
    SC_path = params[1]
    FC_path = params[2]
    LENGTH_path = params[3]

    # Check that the input arguments are of the correct type
    check_all_types([
        (number_oscillators, int, 'number_oscillators'),
        (SC_path, str, 'SC_path'),
        (FC_path, str, 'FC_path'),
        (LENGTH_path, str, 'LENGTH_path')
    ])

    # Create the dictionary
    config = {
        "number_oscillators": number_oscillators,
        "SC_path": SC_path,
        "FC_path": FC_path,
        "LENGTH_path": LENGTH_path
    }

    with open(config_path,'r') as f:
        current_dictionary = json.load(f)

    current_dictionary.update(config)

    with open(config_path,'w') as f:
        json.dump(current_dictionary, f, indent=4)