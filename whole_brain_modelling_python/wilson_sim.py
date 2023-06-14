# Imports
import matplotlib.pyplot as plt
from helper_funcs import *
import pandas as pd
import numpy as np
import sys
import os

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
    def sigmoidal(inp):
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
        noise_type = simulation_params['noise_type']
        time_steps = simulation_params['time_steps']

        # Checking that the parameters are valid
        check_type(integration_steps, int, "integration_steps")
        check_type(integration_step_size, float, "integration_step_size")
        check_type(initial_conditions, np.ndarray, "initial_conditions")
        check_type(number_of_regions, int, "number_of_regions")
        check_type(SC, np.ndarray, "SC")
        check_type(noise_type, int, "noise_type")
        check_type(time_steps, np.ndarray, "time_steps")

        # Checking the shape of the parameters
        check_shape(initial_conditions, (number_of_regions, 2), "initial_conditions")
        check_shape(SC, (number_of_regions, number_of_regions), "SC")
        
        # Setting the initial conditions
        E = initial_conditions[:, 0]
        I = initial_conditions[:, 1]

        # Setting the output matrix
        electrical_activity = np.zeros((number_of_regions, integration_steps))

        # Remember that delays are for long range connections only!!
        # Find the long-range delays
        delays_lookup = np.floor(self.delay).astype(int)
        max_delay = int(np.max(delays_lookup))

        # Defining the type of noise and initializing it
        if noise_type == 0:
            noise = np.zeros((number_of_regions))
        elif noise_type == 1:
            noise = np.random.normal(0, 1, (number_of_regions))
        elif noise_type == 2:
            noise = np.random.uniform(0, 1, (number_of_regions))

        # Setting the simulation loop - going from the max delay backwards
        for i in time_steps[max_delay:-1]:

            # Initialize the delay input
            long_range_connections = E[:, :i]

            # Initialize the delay matrix
            

            # Calculating the external input
            external_input = np.zeros(number_of_regions, 2)
            external_input[:, 0] = self.external_e
            external_input[:, 1] = self.external_i

            # Calculating the input
            inp = np.zeros(number_of_regions, 2)
            inp[:, 0] = self.c_ee * E[0] - self.c_ei * I[0] + external_input[:, 0]
            inp[:, 1] = self.c_ie * E[1] - self.c_ii * I[1] + external_input[:, 1]

            # Calculating the sigmoidal response
            E_response = self.sigmoidal(inp[:, 0])
            I_response = self.sigmoidal(inp[:, 1])

            # Calculating the derivative
            E_derivative = (-E[0] + self.r_e * E_response * (self.alpha_e * E[0] - self.k_e) + coupling_matrix[0, 0] * E[0] + coupling_matrix[0, 1] * E[1] + delay_matrix[0, 0] * E[0] + delay_matrix[0, 1] * E[1]) / self.tau_e
            I_derivative = (-I[0] + self.r_i * I_response * (self.alpha_i * I[0] - self.k_i) + coupling_matrix[1, 0] * I[0] + coupling_matrix[1, 1] * I[1] + delay_matrix[1, 0] * I[0] + delay_matrix[1, 1] * I[1]) / self.tau_i

            # Calculating the next state
            E[0] = E[0] + integration_step_size * E_derivative
            I[0] = I[0] + integration_step_size * I_derivative

            # Storing the output
            output_matrix[0, i] = E[0]
            output_matrix[1, i] = I[0]


