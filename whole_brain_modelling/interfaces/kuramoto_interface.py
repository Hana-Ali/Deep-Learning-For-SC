import simulations as sim
from py_helpers import *
import numpy as np
import time

hpc = True

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
    (SC_FC_root, WBM_main_path, WBM_results_path, config_path, NUMPY_root_path, 
     SC_numpy_root, FC_numpy_root) = define_paths(hpc, wbm_type="kuramoto")

    # --------- Read the config file
    print('Reading config file...')
    config = read_json_config_kura(config_path)

    # --------- Extract the parameters
    print('Extracting parameters...')
    # Extract the parameters
    number_oscillators = config['number_oscillators']
    number_integration_steps = config['number_integration_steps']
    integration_step_size = config['integration_step_size']
    start_save_idx = config['start_save_idx']
    downsampling_rate = config['downsampling_rate']
    noise_type = config['noise_type']
    noise_amplitude = config['noise_amplitude']
    write_path = config['write_path']
    SC_path = config['SC_path']
    FC_path = config['FC_path']

    # --------- Get the SC and FC matrices
    print('Getting SC and FC matrices...')
    SC = np.load(SC_path, allow_pickle=True)
    emp_FC = np.load(FC_path, allow_pickle=True)

    # --------- Check the shape of the SC and FC matrices
    inputs = [
        (SC, (number_oscillators, number_oscillators), 'SC'),
        (emp_FC, (number_oscillators, number_oscillators), 'FC')
    ]
    check_all_shapes(inputs)

    # --------- Check the type of data in the SC and FC matrices
    inputs = [
        (SC[0, 0], np.float64, 'SC[0, 0]'),
        (emp_FC[0, 0], np.float64, 'FC[0, 0]')
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
    coupling_matrix = coupling_strength * SC    
    # DELAY is either 0, if local coupling, or delay * path lengths, if global coupling
    delay_matrix = delay * SC

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
        SC,
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
    
    # --------- Ignore initialization (and downsample?)
    print("Ignoring initialization and downsampling...")
    downsample_phi = raw_phi[:, start_save_idx - downsampling_rate + 1 :]

    # --------- Calculate FC
    print("Calculating FC...")
    sim_FC = np.corrcoef(downsample_phi)
    np.fill_diagonal(sim_FC, 0.0)

    # --------- Check the shape of the simulated FC matrix
    print("Checking simulated FC shape...")
    check_shape(sim_FC, (number_oscillators, number_oscillators), 'sim_FC')

    # --------- Calculate simFC <-> empFC correlation
    print("Calculating simFC <-> empFC correlation...")
    empFC_simFC_corr = determine_similarity(emp_FC, sim_FC)

    print('----------------- Saving results -----------------')

    # --------- Define folder path for all simulations
    print("Creating folders...")
    check_output_folders(write_path, "kuramoto write path", wipe=False)

    # Define the main results folder
    main_results_folder_name = "Coupling {:.4f}, Delay{:.4f}\\".format(coupling_strength, delay)
    main_results_folder = os.path.join(write_path, main_results_folder_name)
    check_output_folders(main_results_folder, "Main results folder", wipe=False)

    # Define the correlation results folder
    corr_results_folder = os.path.join(main_results_folder, "correlation")
    check_output_folders(corr_results_folder, "Correlation results folder", wipe=False)
    
    # Save the simulated FC in the main results folder
    sim_FC_path = os.path.join(main_results_folder, "sim_FC.csv")

    # Save the correlation results in the correlation results folder
    emp_FC_img_path = os.path.join(corr_results_folder, "emp_FC.png")
    sim_FC_img_path = os.path.join(corr_results_folder, "sim_FC.png")
    empFC_simFC_corr_path = os.path.join(corr_results_folder, "empFC_simFC_corr.txt")

    # Save the results
    print("Saving the results...")
    np.savetxt(sim_FC_path, sim_FC, fmt="% .8f", delimiter=",")
    np.savetxt(empFC_simFC_corr_path, np.array([empFC_simFC_corr]), fmt="% .8f")

    # Create and save images
    plt.figure()
    plt.imshow(emp_FC)
    plt.savefig(emp_FC_img_path)
    plt.figure()
    plt.imshow(sim_FC)
    plt.savefig(sim_FC_img_path)

    # Return the correlation
    return empFC_simFC_corr