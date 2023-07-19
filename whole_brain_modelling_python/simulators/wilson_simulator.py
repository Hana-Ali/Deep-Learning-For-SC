# Imports
from py_helpers import *
import numpy as np
import torch

###### Define the Wilson-Cowan model as a class
class wilson_model:
    # Constructor
    def __init__(self, params):
        """
        We assume that we get the WC parameters as a dictionary, and we set the
        model parameters as the attributes of the class

        The expected parameters are:
        - c_ee = 16.0
        - c_ei = 12.0
        - c_ie = 15.0
        - c_ii = 3.0
        - tau_e = 8.0
        - tau_i = 8.0
        - r_e = 1.0
        - r_i = 1.0
        - k_e = 1.0
        - k_i = 1.0
        - alpha_e = 1.0
        - alpha_i = 1.0
        - theta_e = 4.0
        - theta_i = 3.7
        - external_e = 0.1
        - external_i = 0.1
        - coupling_strength = 0.0
        - delay = 0.0
        """

        # Setting the parameters
        self.c_ee = params['c_ee']
        self.c_ei = params['c_ei']
        self.c_ie = params['c_ie']
        self.c_ii = params['c_ii']
        self.tau_e = params['tau_e']
        self.tau_i = params['tau_i']
        self.r_e = params['r_e']
        self.r_i = params['r_i']
        self.k_e = params['k_e']
        self.k_i = params['k_i']
        self.alpha_e = params['alpha_e']
        self.alpha_i = params['alpha_i']
        self.theta_e = params['theta_e']
        self.theta_i = params['theta_i']
        self.external_e = params['external_e']
        self.external_i = params['external_i']
        self.coupling_strength = params['coupling_strength']
        self.delay = params['delay']

        # Checking that the parameters are valid
        check_type(self.c_ee, float, "c_ee")
        check_type(self.c_ei, float, "c_ei")
        check_type(self.c_ie, float, "c_ie")
        check_type(self.c_ii, float, "c_ii")
        check_type(self.tau_e, float, "tau_e")
        check_type(self.tau_i, float, "tau_i")
        check_type(self.r_e, float, "r_e")
        check_type(self.r_i, float, "r_i")
        check_type(self.k_e, float, "k_e")
        check_type(self.k_i, float, "k_i")
        check_type(self.alpha_e, float, "alpha_e")
        check_type(self.alpha_i, float, "alpha_i")
        check_type(self.theta_e, float, "theta_e")
        check_type(self.theta_i, float, "theta_i")
        check_type(self.external_e, float, "external_e")
        check_type(self.external_i, float, "external_i")
        check_type(self.coupling_strength, float, "coupling_strength")
        check_type(self.delay, float, "delay")

    # Define the response function
    def sigmoidal(self, inp):
        """
        This function takes in the input and returns the sigmoidal response
        function output
        """
        return 1 / (1 + np.exp(-inp))

    # Define the simulator function
    def simulator(self, simulation_params):
        """
        This function takes in the simulation parameters and returns the
        simulated electrical activity time series

        The expected parameters are:
        - integration_steps: int, number of integration steps
        - integration_step_size: float, integration step size
        - initial_conditions: list of floats, initial conditions for the
        simulation
        - number_of_regions: int, number of regions in the model
        - SC: numpy array, structural connectivity matrix
        - noise_type: int, type of noise to be added to the simulation
        - time_steps: numpy array, time steps for the simulation
        """

        # Setting the parameters
        integration_steps = simulation_params['integration_steps']
        integration_step_size = simulation_params['integration_step_size']
        initial_conditions = simulation_params['initial_conditions']
        number_of_regions = simulation_params['number_of_regions']
        SC = simulation_params['SC']
        # noise_type = simulation_params['noise_type']
        # time_steps = simulation_params['time_steps']

        # Checking that the parameters are valid
        check_type(integration_steps, int, "integration_steps")
        check_type(integration_step_size, float, "integration_step_size")
        check_type(initial_conditions, "tensor", "initial_conditions")
        check_type(number_of_regions, int, "number_of_regions")
        check_type(SC, np.ndarray, "SC")
        # check_type(noise_type, int, "noise_type")
        # check_type(time_steps, np.ndarray, "time_steps")

        # Checking the shape of the parameters
        check_shape(initial_conditions, (number_of_regions, 2), "initial_conditions")
        check_shape(SC, (number_of_regions, number_of_regions), "SC")
        
        # Setting the initial conditions
        E = initial_conditions[:, 0]
        I = initial_conditions[:, 1]

        # Setting the output matrix
        electrical_activity = torch.tensor(np.zeros((number_of_regions, integration_steps)))

        # Setting the simulation loop - going from the max delay backwards
        for i in range(0, integration_steps):

            if i % 1000 == 0:
                print('Integration step', i)

            # If current integration step is greater than delay, then we can delay
            if i > self.delay:
                # Initialize the delay input
                self.delay = np.floor(self.delay).astype(int)
                delay_matrix = torch.tensor(electrical_activity[:, i - self.delay : i + 1])
            
                # Multiplying by the appropriate elements in the structural connectivity matrix
                delay_matrix = torch.tensor(SC) @ delay_matrix # Needs to be times path length - check w pedro
                delay_matrix = torch.sum(delay_matrix, axis=1)

                # Calculating the coupling matrix
                coupling_matrix = self.coupling_strength * torch.tensor(SC)
                coupling_matrix.fill_diagonal_(self.c_ee)
                coupling_matrix = coupling_matrix @ delay_matrix

            # Calculating the external input
            external_input = torch.tensor(np.zeros((number_of_regions, 2)))
            external_input[:, 0] = self.external_e
            external_input[:, 1] = self.external_i

            # Calculating the input
            inp = torch.tensor(np.zeros((number_of_regions, 2)))
            inp[:, 0] = self.c_ee * E - self.c_ei * I + external_input[:, 0]
            inp[:, 1] = self.c_ie * E - self.c_ii * I + external_input[:, 1]

            # If there is long range delay, add the coupling
            if i > self.delay:
                inp[:, 0] = inp[:, 0] + coupling_matrix
                inp[:, 1] = inp[:, 1] + coupling_matrix

            # Multipling input by alpha
            inp[:, 0] = self.alpha_e * inp[:, 0]
            inp[:, 1] = self.alpha_i * inp[:, 1]

            # Calculating the sigmoidal response
            E_response = self.sigmoidal(inp[:, 0])
            I_response = self.sigmoidal(inp[:, 1])

            # Calculating the derivative
            E_derivative = (-E + (self.k_e - self.r_e * E) * (E_response)) / self.tau_e
            I_derivative = (-I + (self.k_i - self.r_i * I) * (I_response)) / self.tau_i

            # Calculating the next state
            E = E + integration_step_size * E_derivative
            I = I + integration_step_size * I_derivative

            # Storing this in the output
            electrical_activity[:, i] = E

        # Returning the output
        return electrical_activity