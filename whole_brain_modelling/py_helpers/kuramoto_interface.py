import simulations as sim
import sys
import time
sys.path.insert(0, r"C:\Users\shahi\OneDrive - Imperial College London\Documents\imperial\Dissertation\Notebooks\MyCodes\whole_brain_modelling\py_helpers")
from helper_funcs import *
import numpy as np

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

    # --------- Read the config file
    print('Reading config file...')
    config_path = os.path.join(os.getcwd(), "configs\\kuramoto_config.json")
    config = read_json_config_kura(config_path)

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
    noise_type = config['noise_type']
    noise_amplitude = config['noise_amplitude']
    write_path = config['write_path']
    order = config['order']
    cutoffLow = config['cutoffLow']
    cutoffHigh = config['cutoffHigh']
    sampling_rate = config['sampling_rate']

    # --------- Get the SC and FC matrices
    print('Getting SC and FC matrices...')
    SC = np.load(SC_path)
    emp_FC = np.load(FC_path)

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

    # print("Saving raw phi...")
    # if not os.path.exists(write_path):
    #     os.makedirs(write_path)
    # raw_phi_path = os.path.join(write_path, "raw_phi.csv")
    # np.savetxt(raw_phi_path, raw_phi, fmt="% .4f", delimiter=",")
    
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
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    folder_name = "Coupling {:.4f}, Delay{:.4f}\\".format(coupling_strength, delay)
    # phi_path_main = os.path.join(write_path, folder_name)
    FC_path_main = os.path.join(write_path, folder_name)
    empFC_simFC_corr_path_main = os.path.join(write_path, folder_name)

    # if not os.path.exists(phi_path_main):
    #     os.makedirs(phi_path_main)
    if not os.path.exists(FC_path_main):
        os.makedirs(FC_path_main)
    if not os.path.exists(empFC_simFC_corr_path_main):
        os.makedirs(empFC_simFC_corr_path_main)

    # raw_phi_path = os.path.join(phi_path_main, "raw_phi.csv")
    # phi_downsample_path = os.path.join(phi_path_main, "downsample_phi.csv")
    FC_path = os.path.join(FC_path_main, "sim_FC.csv")
    emp_FC_img_path = os.path.join(FC_path_main, "emp_FC.png")
    sim_FC_img_path = os.path.join(FC_path_main, "sim_FC.png")
    empFC_simFC_corr_path = os.path.join(empFC_simFC_corr_path_main, "empFC_simFC_corr.txt")

    # # Save the results
    print("Saving the results...")
    # np.savetxt(raw_phi_path, raw_phi, fmt="% .4f", delimiter=",")
    # np.savetxt(phi_downsample_path, downsample_phi, fmt="% .4f", delimiter=",")
    np.savetxt(FC_path, sim_FC, fmt="% .8f", delimiter=",")
    np.savetxt(empFC_simFC_corr_path, np.array([empFC_simFC_corr]), fmt="% .8f")

    plt.figure()
    plt.imshow(emp_FC)
    plt.savefig(emp_FC_img_path)
    plt.figure()
    plt.imshow(sim_FC)
    plt.savefig(sim_FC_img_path)


    # Return the correlation
    return empFC_simFC_corr