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
    SC = args[0]
    BOLD = args[1]
    write_path = args[2]


    # --------- Check the type of the input arguments
    print('-- Checking types --')
    check_type(SC, np.ndarray, 'SC')
    check_type(BOLD, np.ndarray, 'BOLD')
    check_type(write_path, str, 'write_path')
    
    check_shape(SC, (BOLD.shape[0], BOLD.shape[0]), 'SC')
    

    # --------- Define initial values to be used in equation, COUPLING AND DELAY
    # COUPLING is either c_ee, if local coupling, or SC, if global coupling
    # DELAY is either 0, if local coupling, or delay * path lengths, if global coupling
    # print('-- Defining initial values --')
    # coupling_matrix = coupling_strength * SC
    # # np.fill_diagonal(coupling_matrix, c_ee)
    # coupling_matrix += (
    #     np.diag(np.ones((number_oscillators,)) * c_ee)
    # )
    # delay_matrix = delay * SC

    num_BOLD_subjects = BOLD.shape[0]
    num_BOLD_regions = BOLD.shape[1]
    num_BOLD_timepoints = BOLD.shape[2]

    # --------- Define the index matrices for integration (WHAT IS THIS)
    # print('-- Defining index matrices --')
    # upper_idx = np.floor(delay_matrix / integration_step_size).astype(int)
    # lower_idx = upper_idx + 1
    
    print('------------ Before simulation ------------')

    # --------- SIMULATION TIME BABEY
    simulation_results = sim.parsing_wilson_inputs(
        SC,
        BOLD
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
