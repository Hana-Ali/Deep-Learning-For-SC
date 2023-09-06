import simulations as sim
from py_helpers import *
import numpy as np
import time

hpc=False

def kuramoto_simulator(coupling_strength, delay):
    """"
    This function will simulate the Kuramoto model with electrical coupling
    for a given set of parameters. It will return the Phases array

    Parameters
    ----------
    args : tuple, input model arguments
        args[0] : str, path to the config file

    Returns
    -------
    results : dict, dictionary of results of BO
    
    Equation
    --------
    
    """

    print('----------------- In Kuramoto Simulator -----------------')

    # --------- Get the main paths
    (SC_root_path, FC_root_path, write_folder, 
     config_folder) = define_paths(hpc, wbm_type="kuramoto")

     # --------- Read the config file
    print('Reading config file...')
    config, config_path = read_json_config_kuramoto(config_folder)

    # --------- Extract the parameters
    print('Extracting parameters...')
    # Extract the parameters
    number_oscillators = config['number_oscillators']
    number_integration_steps = config['number_integration_steps']
    integration_step_size = config['integration_step_size']
    start_save_idx = config['start_save_idx']
    downsampling_rate = config['downsampling_rate']
    SC_path = config['SC_path']
    FC_path = config['FC_path']
    LENGTH_path = config['LENGTH_path']
    noise_type = config['noise_type']
    noise_amplitude = config['noise_amplitude']
    write_folder = config['write_folder']
    order = config['order']
    cutoffLow = config['cutoffLow']
    cutoffHigh = config['cutoffHigh']
    TR = config['TR']
    species = config['species']
    streamline_type = config['streamline_type']
    connectome_type = config['connectome_type']
    symmetric = config['symmetric']

    # --------- Get the SC and FC matrices
    print('Getting SC and FC matrices...')
    SC_matrix = get_empirical_SC(SC_path, HPC=hpc, species_type=species, symmetric=symmetric)
    FC_matrix = get_empirical_FC(FC_path, config_path, HPC=hpc, species_type=species)
    LENGTH_matrix = get_empirical_LENGTH(LENGTH_path, HPC=hpc, species_type=species)

    # --------- Check the shape of the SC and FC matrices
    inputs = [
        (SC_matrix, (number_oscillators, number_oscillators), 'SC'),
        (FC_matrix, (number_oscillators, number_oscillators), 'FC')
    ]
    check_all_shapes(inputs)

    # --------- Check the type of data in the SC and FC matrices
    inputs = [
        (SC_matrix[0, 0], np.float64, 'SC[0, 0]'),
        (FC_matrix[0, 0], np.float64, 'FC[0, 0]')
    ]
    check_all_types(inputs)

    # --------- Get the initial conditions and other parameters
    print('Getting initial conditions...')
    initial_conditions_e = np.random.rand(number_oscillators)
    initial_conditions_i = np.random.rand(number_oscillators)
    
    # --------- Check the type of data in the initial conditions and other parameters
    inputs = [
        (initial_conditions_e[0], np.float64, 'initial_cond_e[0]'),
        (initial_conditions_i[0], np.float64, 'initial_cond_i[0]'),
    ]
    check_all_types(inputs)
    
    # --------- Defining the coupling and delay matrices
    print('Defining coupling and delay matrices...')
    # COUPLING is just * SC here
    coupling_matrix = coupling_strength * SC_matrix    
    # DELAY is either 0, if local coupling, or delay * path lengths, if global coupling
    delay_matrix = delay * LENGTH_matrix
    delay_matrix += (np.diag(np.zeros((number_oscillators,))))

    # --------- Define the index matrices for integration (WHAT IS THIS)
    print('Defining index matrices...')
    upper_idx = np.floor(delay_matrix / integration_step_size).astype(int)
    lower_idx = upper_idx + 1

    # Start simulation time
    start_sim_time = time.time()

    print('Entering simulation...')
    
    # --------- SIMULATION TIME BABEY
    raw_phi = sim.parsing_kuramoto_inputs(
        coupling_strength,
        delay,
        SC_matrix,
        number_oscillators,
        number_integration_steps,
        integration_step_size,
        lower_idx,
        upper_idx,
        initial_conditions_e,
        initial_conditions_i,
        noise_type,
        noise_amplitude
    )

    print('----------------- After simulation -----------------')

    # End simulation time
    end_sim_time = time.time()
    print('Simulation time: ' + str(end_sim_time - start_sim_time), 'seconds')

    print('----------------- BOLD and FC processing -----------------')
    
    # --------- Ignore initialization
    print("Ignoring initialization...")
    downsample_phi = raw_phi[:, start_save_idx - downsampling_rate + 1 :]

    # --------- Calculate FC
    print("Calculating FC...")
    bold_filtered = process_BOLD_osc(downsample_phi)
    sim_FC = np.corrcoef(bold_filtered)
    np.fill_diagonal(sim_FC, 0.0)

     # --------- Determine the R order for the BOLD signal
    order_mean, order_std = determine_order_R(bold_filtered, number_oscillators, 
                                              start_index=int(1 / integration_step_size))

    # --------- Check the shape of the simulated FC matrix
    print("Checking simulated FC shape...")
    check_shape(sim_FC, (number_oscillators, number_oscillators), 'sim_FC')

    # --------- Calculate simFC <-> empFC correlation
    print("Calculating simFC <-> empFC correlation...")
    empFC_simFC_corr = determine_similarity(FC_matrix, sim_FC)

    print('----------------- Saving results -----------------')

    # --------- Define folder path for all simulations
    print("Creating folders...")
    symmetric_str = "symmetric" if symmetric else "asymmetric"
    write_folder = os.path.join(write_folder, species, connectome_type, streamline_type, symmetric_str)
    check_output_folders(write_folder, "kuramoto write path", wipe=False)

    folder_name = "Coupling {:.4f}, Delay{:.4f}\\".format(coupling_strength, delay)
    results_path_main = os.path.join(write_folder, folder_name)

    # Make sure folders exist
    check_output_folders(results_path_main, "results_path_main", wipe=False)

    FC_path = os.path.join(results_path_main, "sim_FC.csv")
    FC_matrix_img_path = os.path.join(results_path_main, "emp_FC.png")
    sim_FC_img_path = os.path.join(results_path_main, "sim_FC.png")
    empFC_simFC_corr_path = os.path.join(results_path_main, "empFC_simFC_corr.txt")
    bold_path = os.path.join(results_path_main, "bold_filtered.csv")
    order_data_path = os.path.join(results_path_main, "order_data.txt")

    # Save the results
    print("Saving the results...")
    np.savetxt(FC_path, sim_FC, fmt="% .8f", delimiter=",")
    np.savetxt(empFC_simFC_corr_path, np.array([empFC_simFC_corr]), fmt="% .8f")
    np.savetxt(order_data_path, np.array([order_mean, order_std]), fmt="% .8f")
    # np.savetxt(bold_path, bold_filtered, fmt="% .8f", delimiter=",")

    # Create and save images
    plt.figure()
    plt.imshow(FC_matrix)
    plt.savefig(FC_matrix_img_path)
    plt.close()
    plt.figure()
    plt.imshow(sim_FC)
    plt.savefig(sim_FC_img_path)
    plt.close()

    # Return the correlation
    return empFC_simFC_corr