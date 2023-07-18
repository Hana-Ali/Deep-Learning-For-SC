import simulations as sim
import sys
import time
from py_helpers import *
import numpy as np

def wilson_simulator(coupling_strength, delay):
    """"
    This function will simulate the Wilson-Cowan model with electrical coupling
    for a given set of parameters. It will return the simulated electrical activity

    Parameters
    ----------
    args : tuple, input model arguments
        args[0] : str, path to the config file

    Returns
    -------
    results : dict, dictionary of results of BO
    
    Equation
    --------
    tau_e * dE/dt = -E + (1 - r_e*E)*S_e(c_ee*E + c_ei*I + external_e)
    tau_i * dI/dt = -I + (1 - r_i*I)*S_I(c_ie*E + c_ii*I + external_i)

    where S_e and S_i are sigmoid functions
        S_e = 1 / (1 + exp(-alpha_e * (x - theta_e)))
        S_i = 1 / (1 + exp(-alpha_i * (x - theta_i)))
    """

    print('----------------- In wilson_electrical_sim -----------------')

    # --------- Read the config file
    print('Reading config file...')
    config_path = os.path.join(os.getcwd(), os.path.join("configs", "wilson_config.json"))
    config = read_json_config(config_path)

    # --------- Extract the parameters
    print('Extracting parameters...')
    # Extract the parameters
    number_oscillators = config['number_oscillators']
    c_ee = config['c_ee']
    c_ei = config['c_ei']
    c_ie = config['c_ie']
    c_ii = config['c_ii']
    tau_e = config['tau_e']
    tau_i = config['tau_i']
    r_e = config['r_e']
    r_i = config['r_i']
    alpha_e = config['alpha_e']
    alpha_i = config['alpha_i']
    theta_e = config['theta_e']
    theta_i = config['theta_i']
    external_e = config['external_e']
    external_i = config['external_i']
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
    TR = config['TR']

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
    # COUPLING is either c_ee, if local coupling, or SC, if global coupling
    coupling_matrix = coupling_strength * SC    
    coupling_matrix += (np.diag(np.ones((number_oscillators,)) * c_ee))
    # DELAY is either 0, if local coupling, or delay * path lengths, if global coupling
    delay_matrix = delay * SC
    delay_matrix += (np.diag(np.zeros((number_oscillators,))))

    # --------- Define the index matrices for integration (WHAT IS THIS)
    print('Defining index matrices...')
    upper_idx = np.floor(delay_matrix / integration_step_size).astype(int)
    lower_idx = upper_idx + 1

    # Start simulation time
    start_sim_time = time.time()

    print('Entering simulation...')
    
    # --------- SIMULATION TIME BABEY
    raw_sim_bold = sim.parsing_wilson_inputs(
        coupling_strength,
        delay,
        SC,
        number_oscillators,
        c_ee,
        c_ei,
        c_ie,
        c_ii,
        tau_e,
        tau_i,
        r_e,
        r_i,
        alpha_e,
        alpha_i,
        theta_e,
        theta_i,
        external_e,
        external_i,
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
    # Check results shape
    print("Checking raw BOLD output shape...")
    check_shape(raw_sim_bold, (number_oscillators, number_integration_steps), 'wilson_sim_bold')
    
    # --------- Ignore initialization (and downsample?)
    print("Ignoring initialization and downsampling...")
    bold_down1 = raw_sim_bold[:, start_save_idx - downsampling_rate + 1 :]

    # --------- Calculate FC
    print("Calculating filtered BOLD and FC...")
    bold_filter = process_BOLD(bold_down1, order, TR, cutoffLow, cutoffHigh)
    sim_FC = np.corrcoef(bold_filter)
    np.fill_diagonal(sim_FC, 0.0)

    # --------- Saving the downsampled BOLD only
    print("Downsampling BOLD again?...")
    bold_down2 = bold_filter[:, downsampling_rate - 1 :: downsampling_rate]

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
    # bold_path_main = os.path.join(write_path, folder_name)
    FC_path_main = os.path.join(write_path, folder_name)
    empFC_simFC_corr_path_main = os.path.join(write_path, folder_name)

    # if not os.path.exists(bold_path_main):
    #     os.makedirs(bold_path_main)
    if not os.path.exists(FC_path_main):
        os.makedirs(FC_path_main)
    if not os.path.exists(empFC_simFC_corr_path_main):
        os.makedirs(empFC_simFC_corr_path_main)

    # raw_bold_path = os.path.join(bold_path_main, "raw_bold.csv")
    # bold_down1_path = os.path.join(bold_path_main, "bold_down1.csv")
    # bold_filter_path = os.path.join(bold_path_main, "bold_filter.csv")
    # bold_down2_path = os.path.join(bold_path_main, "bold_down2.csv")
    FC_path = os.path.join(FC_path_main, "sim_FC.csv")
    emp_FC_img_path = os.path.join(FC_path_main, "emp_FC.png")
    sim_FC_img_path = os.path.join(FC_path_main, "sim_FC.png")
    empFC_simFC_corr_path = os.path.join(empFC_simFC_corr_path_main, "empFC_simFC_corr.txt")

    # # Save the results
    print("Saving the results...")
    # np.savetxt(raw_bold_path, raw_sim_bold, fmt="% .4f", delimiter=",")
    # np.savetxt(bold_down1_path, bold_down1, fmt="% .4f", delimiter=",")
    # np.savetxt(bold_filter_path, bold_filter, fmt="% .4f", delimiter=",")
    # np.savetxt(bold_down2_path, bold_down2, fmt="% .4f", delimiter=",")
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