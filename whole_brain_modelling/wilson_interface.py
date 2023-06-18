import simulations as sim
from helper_funcs import *
import numpy as np

# New

def wilson_electrical_sim(args):
    """"
    This function will simulate the Wilson-Cowan model with electrical coupling
    for a given set of parameters. It will return the simulated electrical activity

    Parameters
    ----------
    args : tuple, input model arguments
        args[0] : float, coupling strength
        args[1] : float, delay
        args[2] : int, number of oscillators
        args[3] : float, c_ee
        args[4] : float, c_ei
        args[5] : float, c_ie
        args[6] : float, c_ii
        args[7] : float, tau_e
        args[8] : float, tau_i
        args[9] : float, r_e
        args[10] : float, r_i
        args[11] : float, alpha_e
        args[12] : float, alpha_i
        args[13] : float, theta_e
        args[14] : float, theta_i
        args[15] : float, external_e
        args[16] : float, external_i
        args[17] : int, number of integration steps
        args[18] : float, integration step size
        args[19] : int, start_save_idx
        args[20] : int, downsampling_rate
        args[21] : array, initial conditions_e
        args[22] : array, initial conditions_i
        args[23] : array, SC matrix
        args[24] : array, FC matrix
        args[25] : array, BOLD matrix
        args[26] : int, noise type
        args[27] : float, noise amplitude
        args[28] : string, write path,
        args[29] : double, order of filter
        args[30] : double, low cutoff frequency of filter
        args[31] : double, high cutoff frequency of filter
        args[32] : double, sampling frequency of filter
        args[33] : int, number of iterations (BO)
        args[34] : int, number of inner iterations (BO)
        args[35] : int, number of initial samples (BO)
        args[36] : number of iterations to relearn (BO)
        args[37] : int, initial method (BO)
        args[38] : int, verbose level (BO)
        args[39] : string, log file (BO)
        args[40] : string, surrogate name (BO)
        args[41] : int, SC type (BO)
        args[42] : int, L type (BO)
        args[43] : bool, L all (BO)
        args[44] : double, epsilon (BO)
        args[45] : bool, force jump (BO)
        args[46] : string, criterion name (BO)

    Returns
    -------
    elec: array, simulated electrical activity

    Equation
    --------
    tau_e * dE/dt = -E + (1 - r_e*E)*S_e(c_ee*E + c_ei*I + external_e)
    tau_i * dI/dt = -I + (1 - r_i*I)*S_I(c_ie*E + c_ii*I + external_i)

    where S_e and S_i are sigmoid functions
        S_e = 1 / (1 + exp(-alpha_e * (x - theta_e)))
        S_i = 1 / (1 + exp(-alpha_i * (x - theta_i)))
    """

    print('------------ In wilson_electrical_sim ------------')
    # --------- Check length of the input arguments
    num_params_expected = 47

    if len(args) != num_params_expected:
        exception_msg = 'Exception in WC model. Expected {} arguments, got {}'.format(num_params_expected, str(len(args)))
        raise Exception(exception_msg)
    
    # --------- Unpack the arguments
    print('-- Unpacking arguments --')
    coupling_strength = args[0]
    delay = args[1]
    number_oscillators = args[2]
    c_ee = args[3]
    c_ei = args[4]
    c_ie = args[5]
    c_ii = args[6]
    tau_e = args[7]
    tau_i = args[8]
    r_e = args[9]
    r_i = args[10]
    alpha_e = args[11]
    alpha_i = args[12]
    theta_e = args[13]
    theta_i = args[14]
    external_e = args[15]
    external_i = args[16]
    number_integration_steps = args[17]
    integration_step_size = args[18]
    start_save_idx = args[19]
    downsampling_rate = args[20]
    initial_cond_e = args[21]
    initial_cond_i = args[22]
    SC = args[23]
    FC = args[24]
    BOLD = args[25]
    noise_type = args[26]
    noise_amplitude = args[27]
    write_path = args[28]
    order = args[29]
    cutoffLow = args[30]
    cutoffHigh = args[31]
    sampling_rate = args[32]
    n_iterations = args[33]
    n_inner_iterations = args[34]
    n_init_samples = args[35]
    n_iter_relearn = args[36]
    init_method = args[37]
    verbose_level = args[38]
    log_file = args[39]
    surr_name = args[40]
    sc_type = args[41]
    l_type = args[42]
    l_all = args[43]
    epsilon = args[44]
    force_jump = args[45]
    crit_name = args[46]


    # --------- Check the type of the input arguments
    print('-- Checking types --')
    check_type(coupling_strength, float, 'coupling_strength')
    check_type(delay, float, 'delay')
    check_type(number_oscillators, int, 'number_oscillators')
    check_type(c_ee, float, 'c_ee')
    check_type(c_ei, float, 'c_ei')
    check_type(c_ie, float, 'c_ie')
    check_type(c_ii, float, 'c_ii')
    check_type(tau_e, float, 'tau_e')
    check_type(tau_i, float, 'tau_i')
    check_type(r_e, float, 'r_e')
    check_type(r_i, float, 'r_i')
    check_type(alpha_e, float, 'alpha_e')
    check_type(alpha_i, float, 'alpha_i')
    check_type(theta_e, float, 'theta_e')
    check_type(theta_i, float, 'theta_i')
    check_type(external_e, float, 'external_e')
    check_type(external_i, float, 'external_i')
    check_type(number_integration_steps, int, 'number_integration_steps')
    check_type(integration_step_size, float, 'integration_step_size')
    check_type(start_save_idx, int, 'start_save_idx')
    check_type(downsampling_rate, int, 'downsampling_rate')
    check_type(initial_cond_e, np.ndarray, 'initial_cond_e')
    check_type(initial_cond_i, np.ndarray, 'initial_cond_i')
    check_type(SC, np.ndarray, 'SC')
    check_type(FC, np.ndarray, 'FC')
    check_type(BOLD, np.ndarray, 'BOLD')
    check_type(noise_type, int, 'noise_type')
    check_type(noise_amplitude, float, 'noise_amplitude')
    check_type(write_path, str, 'write_path')
    check_type(order, int, 'order')
    check_type(cutoffLow, float, 'cutoffLow')
    check_type(cutoffHigh, float, 'cutoffHigh')
    check_type(sampling_rate, float, 'sampling_rate')
    check_type(n_iterations, int, 'n_iterations')
    check_type(n_inner_iterations, int, 'n_inner_iterations')
    check_type(n_init_samples, int, 'n_init_samples')
    check_type(n_iter_relearn, int, 'n_iter_relearn')
    check_type(init_method, int, 'init_method')
    check_type(verbose_level, int, 'verbose_level')
    check_type(log_file, str, 'log_file')
    check_type(surr_name, str, 'surr_name')
    check_type(sc_type, int, 'sc_type')
    check_type(l_type, int, 'l_type')
    check_type(l_all, bool, 'l_all')
    check_type(epsilon, float, 'epsilon')
    check_type(force_jump, bool, 'force_jump')
    check_type(crit_name, str, 'crit_name')

    # --------- Check the type of data in the input arguments
    check_type(initial_cond_e[0], np.float64, 'initial_cond_e[0]')
    check_type(initial_cond_i[0], np.float64, 'initial_cond_i[0]')
    check_type(SC[0, 0], np.float64, 'SC[0, 0]')
    check_type(FC[0, 0], np.float64, 'FC[0, 0]')

    # --------- Check the shape of the input arguments
    check_shape(initial_cond_e, (number_oscillators,), 'initial_cond_e')
    check_shape(initial_cond_i, (number_oscillators,), 'initial_cond_i')
    check_shape(SC, (number_oscillators, number_oscillators), 'SC')
    check_shape(FC, (number_oscillators, number_oscillators), 'FC')


    # --------- Define initial values to be used in equation, COUPLING AND DELAY
    # COUPLING is either c_ee, if local coupling, or SC, if global coupling
    # DELAY is either 0, if local coupling, or delay * path lengths, if global coupling
    print('-- Defining initial values --')
    coupling_matrix = coupling_strength * SC
    # np.fill_diagonal(coupling_matrix, c_ee)
    coupling_matrix += (
        np.diag(np.ones((number_oscillators,)) * c_ee)
    )
    delay_matrix = delay * SC

    num_BOLD_subjects = BOLD.shape[0]
    num_BOLD_regions = BOLD.shape[1]
    num_BOLD_timepoints = BOLD.shape[2]

    # --------- Define the index matrices for integration (WHAT IS THIS)
    print('-- Defining index matrices --')
    upper_idx = np.floor(delay_matrix / integration_step_size).astype(int)
    lower_idx = upper_idx + 1
    
    print('------------ Before simulation ------------')

    # --------- SIMULATION TIME BABEY
    simulation_results = sim.parsing_wilson_inputs(
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
        initial_cond_e,
        initial_cond_i,
        noise_type,
        noise_amplitude,
        order,
        cutoffLow,
        cutoffHigh,
        sampling_rate,
        BOLD,
        num_BOLD_subjects,
        num_BOLD_regions,
        num_BOLD_timepoints,
        n_iterations,
        n_inner_iterations,
        n_iter_relearn,
        n_init_samples,
        init_method,
        verbose_level,
        log_file,
        surr_name,
        sc_type,
        l_type,
        l_all,
        epsilon,
        force_jump,
        crit_name
    )

    print('------------ After simulation ------------')
    # Check results shape
    # check_shape(simulation_results, (number_oscillators, number_integration_steps + 1), 'wilson_simulation_results')

    # --------- Convert electrical to BOLD
    # sim_bold = sim.electrical_to_bold()

    # np.savetxt('BOLD_array.csv', sim_bold, delimiter=",")
    # # Check results shape
    # check_shape(sim_bold, (number_oscillators, number_integration_steps + 1), 'wilson_simulation_bold')

    # # --------- Ignore initialization (and downsample?)
    # sim_bold = sim_bold[:, start_save_idx - downsampling_rate + 1 :]
    # # sim_bold = sim_bold[:, start_save_idx:]

    # # --------- Determine order parameter
    # R_mean, R_std = determine_order_R(simulation_results, number_oscillators, int(1 / integration_step_size))

    # # --------- Calculate FC
    # # sim_bold = process_BOLD(sim_bold)
    # sim_FC = np.corrcoef(sim_bold)
    # np.fill_diagonal(sim_FC, 0.0)
    # np.savetxt('bold.csv', sim_bold, fmt="% .4f", delimiter=",")
    # np.savetxt('sim_FC.csv', sim_FC, fmt="% .8f", delimiter=",")

    # # Check the same of the simulated FC matrix
    # check_shape(sim_FC, (number_oscillators, number_oscillators), 'sim_FC')
    
    # # --------- Calculate simFC <-> empFC correlation
    # empFC_simFC_corr = determine_similarity(FC, sim_FC)

    # # --------- Save the results
    # # Define folder path for all simulations
    # folder_name = "wilson_Coupling{:.4f}Delay{:.4f}\\".format(coupling_strength, delay)
    # # Define main paths for each thing
    # electric_path_main = os.path.join(write_path, folder_name)
    # bold_path_main = os.path.join(write_path, folder_name)
    # FC_path_main = os.path.join(write_path, folder_name)
    # R_path_main = os.path.join(write_path, folder_name)
    # empFC_simFC_corr_path_main = os.path.join(write_path, folder_name)
    # # Make paths if they don't exist
    # if not os.path.exists(electric_path_main):
    #     os.makedirs(electric_path_main)
    # if not os.path.exists(bold_path_main):
    #     os.makedirs(bold_path_main)
    # if not os.path.exists(FC_path_main):
    #     os.makedirs(FC_path_main)
    # if not os.path.exists(R_path_main):
    #     os.makedirs(R_path_main)
    # if not os.path.exists(empFC_simFC_corr_path_main):
    #     os.makedirs(empFC_simFC_corr_path_main)
    # # Define paths for this simulation
    # electric_path = os.path.join(electric_path_main, "electric.csv")
    # bold_path = os.path.join(bold_path_main, "bold.csv")
    # FC_path = os.path.join(FC_path_main, "FC.csv")
    # R_path = os.path.join(R_path_main, "R.csv")
    # empFC_simFC_corr_path = os.path.join(empFC_simFC_corr_path_main, "empFC_simFC_corr.csv")

    # print('paths are', electric_path, bold_path, FC_path, R_path, empFC_simFC_corr_path)

    # # Downsample BOLD
    # sim_bold = sim_bold[:, downsampling_rate - 1 :: downsampling_rate]
    # # Save the results
    # np.savetxt(electric_path, simulation_results, delimiter=",")
    # np.savetxt(bold_path, sim_bold, fmt="% .4f", delimiter=",")
    # np.savetxt(FC_path, sim_FC, fmt="% .8f", delimiter=",")
    # np.savetxt(R_path, np.array([R_mean, R_std]), delimiter=",")
    # np.savetxt(empFC_simFC_corr_path, np.array([empFC_simFC_corr]), delimiter=",")

    # # Save the plots
    # plt.figure()
    # print('sim_bold shape is', sim_bold.shape)
    # # print('After expand dims, the first is dim', np.expand_dims(sim_bold[0, :], axis=0).shape)
    # plt.imshow(np.expand_dims(sim_bold[0, :], axis=0))
    # # cmap
    # plt.set_cmap('jet')
    # plt.savefig(os.path.join(bold_path_main, "bold.png"))

    # plt.figure()
    # plt.imshow(sim_FC)
    # plt.savefig(os.path.join(FC_path_main, "FC.png"))
    
    # # --------- Return the results
    # # Create dictionary of results
    # results = {
    #     'coupling_strength': coupling_strength,
    #     'delay': delay,
    #     'R_mean': R_mean,
    #     'R_std': R_std,
    #     'empFC_simFC_corr': empFC_simFC_corr
    # }

    return 0
