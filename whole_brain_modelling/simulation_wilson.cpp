#define PY_SSIZE_T_CLEAN
#include <math.h>
#include <random>
#include <string>
#include <list>
#include <Python.h>
#include <iostream>
#include <fstream>
#include <typeinfo>
#include <boost/any.hpp>
#include <bayesopt/bayesopt.h>
#include <numpy/arrayobject.h>
#include <gsl/gsl_statistics.h>
#include <bayesopt/bayesopt.hpp>
#include "simulation_helpers.hpp"
#include "wilson_config.hpp"

// New

// Response function for Wilson Model
double response_function(double x, double alpha, double theta)
{
    /*
    Given a value x, this function returns the value of the sigmoid function with parameters alpha and theta.

    Parameters
    ----------
    x : double, input value
    alpha : double, sigmoid parameter (steepness)
    theta : double, sigmoid parameter (inflection point position)

    Returns
    -------
    y : double, sigmoid function value at x
    */
    // return 1 / (1 + exp(-alpha * (x - theta)));

    double S = 1 / (1 + exp(-alpha * (x - theta)));
    S -= 1 / (1 + exp(alpha * theta));

    return S;
}

std::vector<std::vector<double>> Wilson::electrical_to_bold(std::vector<std::vector<double>>& electrical_activity,
                                                            int number_of_oscillators,
                                                            int number_of_integration_steps,
                                                            float integration_step_size)
{
    /*
    This is a function that, given electrical activity, will convert it to BOLD signals.
    It does so using the Balloon-Windkessel model. Again, the differential equation follows
    Heun's Method

    Parameters
    ----------
    args : tuple, input model arguments
        args[0] : array, electrical activity of each node (input_e)
        args[1] : int, number of oscillators (number_of_oscillators)
        args[2] : int, number of integration steps (number_of_integration_steps)
        args[3] : float, integration step size (integration_step_size)

    Returns
    -------
    BOLD_signal : array, BOLD signal of each node
    */
    
    // ------------- Declare state variables
    struct BOLDParams {
        double f;
        double q;
        double s;
        double v;
        double differential_f;
        double differential_q;
        double differential_s;
        double differential_v;
        double differential_f2;
        double differential_q2;
        double differential_s2;
        double differential_v2;
        double activity_f;
        double activity_q;
        double activity_s;
        double activity_v;
        double alpha;
        double gamma;
        double kappa;
        double rho;
        double tau;
        double c1;
        double c3;
    };

    // Create an instance of the helper variables, of size number of oscillators
    auto bold_params = std::vector<BOLDParams>(number_of_oscillators);
    // Define random generator with constant seed
    std::default_random_engine generator(1);

    // Check the size of the inputted electrical activity
    if (electrical_activity.size() != number_of_oscillators ||
        electrical_activity[0].size() != number_of_integration_steps + 1)
        {
            std::string error_string = "The size of electrical_activity is (" + std::to_string(electrical_activity.size()) + " x " +
                                std::to_string(electrical_activity[0].size()) + ", but should be (" + std::to_string(number_of_oscillators) 
                                + " x " + std::to_string(number_of_integration_steps + 1) + ")";
            throw std::invalid_argument(error_string);
        }

    // ------------- Initialize values of state variables, [0, 0.1]
    for (auto& state : bold_params)
    {
        // Initialize state variables
        state.f = rand_std_uniform(generator) * 0.1;
        state.q = rand_std_uniform(generator) * 0.1;
        state.s = rand_std_uniform(generator) * 0.1;
        state.v = rand_std_uniform(generator) * 0.1;
    }

    // ------------- Initialize values of helper variables
    auto c2 = 2.000;
    auto V0 = 0.020;
    for (auto& helper : bold_params)
    {
        // Initialize helper variables
        helper.alpha = 1 / (0.320 + rand_std_normal(generator) * 0.039);
        helper.gamma = 0.410 + rand_std_normal(generator) * 0.045;
        helper.kappa = 0.650 + rand_std_normal(generator) * 0.122;
        helper.rho = 0.340 + rand_std_normal(generator) * 0.049;
        helper.tau = 0.980 + rand_std_normal(generator) * 0.238;
        helper.c1 = 7.0 * helper.rho;
        helper.c3 = 2.0 * helper.rho - 0.2;
    }

    // ------------- Declare output variables
    auto output_bold = std::vector<double>(number_of_oscillators);
    auto output_bold_matrix = std::vector<std::vector<double>>(number_of_oscillators, std::vector<double>(number_of_integration_steps + 1, 0));
    std::vector<double> input_e(number_of_oscillators);
    
    // ------------- Initialize output

    // ------------- CONVERSIONS BABEY
    for (int step = 1; step <= number_of_integration_steps; step++)
    {
        for (int i = 0; i < number_of_oscillators; i++)
        {
            input_e[i] = electrical_activity[i][step];
        }

        // ------------ Heun's Method - Step 1
        int i = 0;
        for(auto& state : bold_params)
        {
            // Calculate differentials
            state.differential_f = state.s;
            state.differential_q = 1 - pow(1 - state.rho, 1 / state.f);
            state.differential_q *= state.f / state.rho;
            state.differential_q -= state.q * pow(state.v, state.alpha - 1);
            state.differential_q /= state.tau;
            state.differential_s = input_e[i++];
            state.differential_s -= state.kappa * state.s + state.gamma * (state.f - 1);
            state.differential_v = (state.f - pow(state.v, state.alpha)) / state.tau;

            // First estimate of the new activity values
            state.activity_f = state.f + integration_step_size * state.differential_f;
            state.activity_q = state.q + integration_step_size * state.differential_q;
            state.activity_s = state.s + integration_step_size * state.differential_s;
            state.activity_v = state.v + integration_step_size * state.differential_v;
        }

        // ------------ Heun's Method - Step 2
        int j = 0;
        for(auto& state : bold_params)
        {
            // Calculate differentials
            state.differential_f2 = state.activity_s;
            state.differential_q2 = 1 - pow(1 - state.rho, 1 / state.activity_f);
            state.differential_q2 *= state.activity_f / state.rho;
            state.differential_q2 -= state.activity_q * pow(state.activity_v, state.alpha - 1);
            state.differential_q2 /= state.tau;
            state.differential_s2 = input_e[j++];
            state.differential_s2 -= state.kappa * state.activity_s + state.gamma * (state.activity_f - 1);
            state.differential_v2 = (state.activity_f - pow(state.activity_v, state.alpha)) / state.tau;

            // Second estimate of the new activity values
            state.f += integration_step_size / 2 * (state.differential_f + state.differential_f2);
            state.q += integration_step_size / 2 * (state.differential_q + state.differential_q2);
            state.s += integration_step_size / 2 * (state.differential_s + state.differential_s2);
            state.v += integration_step_size / 2 * (state.differential_v + state.differential_v2);
        }

        // Calculate BOLD signal
        int osc = 0;
        for (auto const& helper : bold_params)
        {
            output_bold[osc] = helper.c1 * (1 - helper.q);
            output_bold[osc] += c2 * (1 - helper.q / helper.v);
            output_bold[osc] += helper.c3 * (1 - helper.v);
            output_bold[osc] *= V0;
            output_bold_matrix[osc][step] = output_bold[osc];
            osc++;
        }
    }

    // ------------- Unpack the BOLD signal
    printf("----------- Unpacking BOLD signal -----------\n");
    // Printing the dimensions of the BOLD_array
    size_t BOLD_dims[2] = { output_bold_matrix.size(), output_bold_matrix[0].size() };

    auto& unpack_bold = output_bold_matrix;

    return unpack_bold;
}
   
// Moving config
Wilson::Wilson(WilsonConfig config)
    : config(std::move(config))
    , electrical_activity{
        std::vector<std::vector<double>>(config.number_of_oscillators,
                                         std::vector<double>(config.number_of_integration_steps + 1, nan("")))}
{
}

// Defining the main Wilson simulator function
double* Wilson::run_simulation()
{
    // Checking config is valid for everything we've saved
    if (!config.check_validity())
    {
        throw std::runtime_error("Not valid config");
    }
    printf("----------------- In CPP file for Wilson Function -----------------\n");

    // TODO: Not sure if we can use wilson_electrical_activity directly
    output_e = electrical_activity;
    // ------------- Convert input variables to C++ types
    printf("---- Converting input variables to C++ types ----\n");
    for (int i = 0; i < config.number_of_oscillators; i++)
    {
        // ------------ Initialize output matrix
        output_e[i][0] = config.e_values[i];
        // Other values in matrix are NaN
    }

    // ------------ Get the BOLD signals for processing
    printf("---- Get the empirical BOLD signals for processing ----\n");

    size_t emp_BOLD_dims[3] = { config.emp_BOLD_signals.size(),
                                config.emp_BOLD_signals[0].size(),
                                config.emp_BOLD_signals[0][0].size() };
    auto& unpack_emp_BOLD = config.emp_BOLD_signals;

    // Saving it just for a sanity check
    printf("----------- Saving unpacked empirical BOLD signal -----------\n");
    save_data_3D(unpack_emp_BOLD, "temp_arrays/unpacked_emp_BOLD.csv");

    printf("----------- Filtering the empirical BOLD signal -----------\n");
    // Create a vector that stores for ALL SUBJECTS
    std::vector<std::vector<std::vector<double>>> emp_bold_filtered;

// For each subject
    for (int subject = 0; subject < emp_BOLD_dims[0]; ++subject)
    {
        printf("In filtering subject %d\n", subject);

        // Add the subject to the vector of all subjects
        emp_bold_filtered.emplace_back(process_BOLD(unpack_emp_BOLD[subject],
                                                    emp_BOLD_dims[1],
                                                    emp_BOLD_dims[2],
                                                    config.order,
                                                    config.cutoffLow,
                                                    config.cutoffHigh,
                                                    config.sampling_rate));
    }

    // Saving it just for a sanity check
    printf("----------- Saving filtered empirical BOLD signal -----------\n");
    save_data_3D(emp_bold_filtered, "temp_arrays/filtered_emp_BOLD.csv");
    
    // ------------- Getting the empirical FC
    printf("----------- Getting the empirical FC -----------\n");
    // Create a vector of vectors of vectors for the FC for all subjects
    std::vector<std::vector<std::vector<double>>> unpack_emp_FC;

    // For each subject
    for (int subject = 0; subject < emp_BOLD_dims[0]; subject++)
    {
        // Add the subject to the vector of all subjects
        printf("subject: %d\n\r", subject);
        unpack_emp_FC.emplace_back(determine_FC(emp_bold_filtered[subject]));
    }

    // Saving it just for a sanity check
    printf("----------- Saving unpacked empirical FC -----------\n");
    save_data_3D(unpack_emp_FC, "temp_arrays/emp_FC_all.csv");

    // ------------- Finding the average across subjects
    printf("----------- Finding the average across subjects -----------\n");
    // Note that this average FC is what's gonna be stored in the empFC global variable

    // For each region
    for (int i = 0; i < emp_BOLD_dims[1]; i++)
    {
        // Create a vector of doubles for each *other* region
        std::vector<double> region_avg;

        // For each other region
        for (int j = 0; j < emp_BOLD_dims[1]; j++)
        {
        // Create a vector of doubles for each subject
        std::vector<double> subject_values;

        // For each subject
        for (int k = 0; k < emp_BOLD_dims[0]; k++)
        {
            subject_values.push_back(unpack_emp_FC[i][j][k]);
        }
        // Get the mean of the subject values
        double mean = gsl_stats_mean(subject_values.data(), 1, subject_values.size());
        region_avg.push_back(mean);
        }
        config.emp_FC.push_back(region_avg);
    }

    // Saving it just for a sanity check
    printf("----------- Saving average empirical FC -----------\n");
    save_data_2D(config.emp_FC, "temp_arrays/empFC.csv");

    // ------------ Run Bayesian Optimization
    printf("---- Define Bayesian Optimization Parameters ----\n");

    // Bayesian Optimization parameters
    bayesopt::Parameters bo_parameters = initialize_parameters_to_default();

    bo_parameters.n_iterations = config.BO_n_iter;
    bo_parameters.n_inner_iterations = config.BO_n_inner_iter;
    bo_parameters.n_init_samples = config.BO_init_samples;
    bo_parameters.n_iter_relearn = config.BO_n_inner_iter;
    bo_parameters.init_method = config.BO_init_method;
    bo_parameters.verbose_level = config.BO_verbose_level;
    bo_parameters.log_filename = config.BO_log_file;
    bo_parameters.surr_name = config.BO_surrogate;
    bo_parameters.sc_type = static_cast<score_type>(config.BO_sc_type);
    bo_parameters.l_type = static_cast<learning_type>(config.BO_l_type);
    bo_parameters.l_all = config.BO_l_all;
    bo_parameters.epsilon = config.BO_epsilon;
    bo_parameters.force_jump = config.BO_force_jump;
    bo_parameters.crit_name = config.BO_crit_name;

    // Call Bayesian Optimization
    // wilson_objective(2, NULL, NULL, NULL);
    const int num_dimensions = 2;
    double lower_bounds[num_dimensions] = { 0.0, 0.0 };
    double upper_bounds[num_dimensions] = { 1.0, 1.0 };

    double minimizer[num_dimensions] = { config.coupling_strength, config.delay };
    double minimizer_value[128];

    printf("---- Run Bayesian Optimization ----\n");
    int wilson_BO_output = bayes_optimization(num_dimensions,
                                                &wilson_objective,
                                                this, // can be used for pass class pointer if it will be a class
                                                lower_bounds,
                                                upper_bounds,
                                                minimizer,
                                                minimizer_value,
                                                bo_parameters.generate_bopt_params());

    // Note that the output of Bayesian Optimization will just be an error message, which we can output
    printf("---- Bayesian Optimization output ----\n");
    if (wilson_BO_output == 0) {
        printf("Bayesian Optimization was successful!\n");
    }
    else {
        printf("Bayesian Optimization was unsuccessful!. Output is %d\n", wilson_BO_output);
    }

    // Note that the output minimizer is stored in the minimizer array
    printf("---- Bayesian Optimization minimizer ----\n");
    for (int i = 0; i < num_dimensions; i++) {
        printf("Minimizer value for dimension %d is %f\n", i, minimizer[i]);
    }

    return minimizer;
}

// Define the objective function for the Wilson model
double Wilson::wilson_objective(unsigned int input_dim,
                                const double *initial_query,
                                double* gradient,
                                void *func_data)
{

    // IMPORTANT
    // ONE WAY TO THINK ABOUT REFACTORING THIS IS THAT THE COUPLING STRENGTH AND DELAY ARE IN THE INITIAL QUERY, AND HERE
    // WE CALCULATE THE MATRICES RATHER THAN IN THE PYTHON FILE

    /*
    This is the goal or objective function that will be used by Bayesian Optimization to find the optimal parameters for the Wilson model.

    Parameters
    ----------
    input_dim : unsigned int, number of parameters
    initial_query : array, initial parameter values
    gradient : array, gradient of the objective function
    func_data : void, additional data for the objective function (which I think means data for the wilson model)
    
    Returns
    -------
    objective_value : double, value of the objective function
    */

    // ------------- Getting function data
    auto& instance = *(static_cast<Wilson*>(func_data));

    // ------------- Declare input variables - arrays
    printf("---- Declare helper variables ----\n");
    long temp_long; //
    double node_input;
    double delay_difference;
    int index_lower;
    int index_upper;
    double input_lower;
    double input_upper;
    double input_final;
    auto differential_E = std::vector<double>(instance.config.number_of_oscillators);
    auto differential_I = std::vector<double>(instance.config.number_of_oscillators);
    auto differential_E2 = std::vector<double>(instance.config.number_of_oscillators);
    auto differential_I2 = std::vector<double>(instance.config.number_of_oscillators);
    auto activity_E = std::vector<double>(instance.config.number_of_oscillators);
    auto activity_I = std::vector<double>(instance.config.number_of_oscillators);
    auto noises_array = std::vector<double>(instance.config.number_of_oscillators);
    instance.delay_mat =
      std::vector<std::vector<double>>(instance.config.number_of_oscillators,
                                       std::vector<double>(instance.config.number_of_oscillators));
    instance.coupling_mat =
      std::vector<std::vector<double>>(instance.config.number_of_oscillators,
                                       std::vector<double>(instance.config.number_of_oscillators));
    // ------------ Random generation
    std::default_random_engine generator(1);

    // ------------ Create the coupling matrix
    printf("---- Create the coupling matrix ----\n");
    // ::wilson_coupling_mat

    // ------------ Create the delay matrix
    printf("---- Create the delay matrix ----\n");
    // ::wilson_delay_mat

    // ------------ TEMPORAL INTEGRATION
    printf("---- Temporal integration ----\n");
    for (int step = 1; step <= instance.config.number_of_integration_steps; step++)
    {
      if (step % 10000 == 0)
        printf("-- Temporal integration step %d --\n", step);
      // printf("-- Heun's Method - Step 1 --\n");
      // ------------ Heun's Method - Step 1
      for (int node = 0; node < instance.config.number_of_oscillators; node++)
      {
        // printf("-- Heun's 1: Node %d --\n", node);
        // ------------ Initializations
        // Initialize input to node as 0
        node_input = 0;

        // Initialize noise
        if ((int)instance.config.noise == 0)
        {
          noises_array[node] = 0;
        }
        else if((int)instance.config.noise == 1)
        {
          noises_array[node] = instance.config.noise_amplitude * (2 * instance.rand_std_uniform(generator) - 1);
        }
        else if((int)instance.config.noise == 2)
        {
          noises_array[node] = instance.config.noise_amplitude * instance.rand_std_normal(generator);
        }

        // printf("-- Heun's 1: Node %d - Noise: %f --\n", node, noises_array[node]);

        // ------------ Calculate input to node
        // Consider all other nodes, but only if the lower delay index is lower than the time point
        for (int other_node = 0; other_node < instance.config.number_of_oscillators; other_node++)
        {
          // printf("-- Heun's 1: Node %d - Other node %d --\n", node, other_node);
          if (step > instance.config.lower_idxs_mat[node][other_node])
          {
            // Retrieve the difference between the 'true' delay and the one corresponding to the upper index
            delay_difference = instance.delay_mat[node][other_node];
            delay_difference -= (double)instance.config.upper_idxs_mat[node][other_node] *
                                        instance.config.integration_step_size;

            // Retrieve the time point indices corresponding with the lower and upper delay indices
            index_lower = step - 1 - instance.config.lower_idxs_mat[node][other_node];
            index_upper = step - 1 - instance.config.upper_idxs_mat[node][other_node];

            // Retrieve the activities corresponding to the lower and upper delay indices
            input_lower = instance.output_e[other_node][index_lower];
            input_upper = instance.output_e[other_node][index_upper];

            // From the previously retrieved values, estimate the input to oscillator k from oscillator j
            input_final = input_upper;
            input_final += (input_lower - input_upper) / instance.config.integration_step_size * delay_difference;
            // From this estimation, determine the quantile, final input
            input_final *= instance.coupling_mat[node][other_node];
            // Add this to the total input to oscillator k
            node_input += input_final;
          }
        }

        // printf("-- Heun's 1: Node %d - Differential Equations --\n", node);
        // ------------ Calculate Equations
        // Excitatory population (without noise and time) differentials
        differential_E[node] = node_input -
                               instance.config.c_ei * instance.config.i_values[node] -
                               instance.config.external_e;
        differential_E[node] = - instance.config.e_values[node] +
                                 (1 - instance.config.r_e * instance.config.e_values[node]) *
                                 response_function(differential_E[node], instance.config.alpha_e, instance.config.theta_e);

        // Inhibitory population (without noise and time) differentials
        differential_I[node] = instance.config.c_ie * instance.config.e_values[node];
        differential_I[node] = - instance.config.i_values[node] +
                                 (1 - instance.config.r_i * instance.config.i_values[node]) *
                                 response_function(differential_I[node], instance.config.alpha_i, instance.config.theta_i);

        // First estimate of the new activity values
        activity_E[node] = instance.config.e_values[node] +
                           (instance.config.integration_step_size * differential_E[node] +
                            sqrt(instance.config.integration_step_size) * noises_array[node]) / instance.config.tau_e;
        activity_I[node] = instance.config.i_values[node] +
                           (instance.config.integration_step_size * differential_I[node] +
                            sqrt(instance.config.integration_step_size) * noises_array[node]) / instance.config.tau_i;

        // printf("-- Heun's 1: Node %d - Update ::wilson_output_e value --\n", node);
        instance.output_e[node][step] = activity_E[node];
      }

      // printf("-- Heun's Method - Step 2 --\n");
      // ------------ Heun's Method - Step 2
      for(int node = 0; node < instance.config.number_of_oscillators; node++)
      {
        // printf("-- Heun's 2: Node %d --\n", node);
        // Initialize input to node as 0
        node_input = 0;

        // ------------ Calculate input to node
        // Consider all other nodes, but only if the lower delay index is lower than the time point
        for (int other_node = 0; other_node < instance.config.number_of_oscillators; other_node++)
        {
          // printf("-- Heun's 2: Node %d - Other node %d --\n", node, other_node);
          if (step > instance.config.lower_idxs_mat[node][other_node])
          {
            // printf("Step > lowerIdx");
            // Retrieve the difference between the 'true' delay and the one corresponding to the upper index
            delay_difference = instance.delay_mat[node][other_node];
            delay_difference -= (double)instance.config.upper_idxs_mat[node][other_node] *
                                        instance.config.integration_step_size;

            // Retrieve the time point indices corresponding with the lower and upper delay indices
            index_lower = step - instance.config.lower_idxs_mat[node][other_node];
            index_upper = step - instance.config.upper_idxs_mat[node][other_node];

            // Retrieve the activities corresponding to the lower and upper delay indices
            input_lower = instance.output_e[other_node][index_lower];
            input_upper = instance.output_e[other_node][index_upper];

            // From the previously retrieved values, estimate the input to oscillator k from oscillator j
            input_final = input_upper;
            input_final += (input_lower - input_upper) / instance.config.integration_step_size * delay_difference;
            // From this estimation, determine the quantile, final input
            input_final *= instance.coupling_mat[node][other_node];
            // Add this to the total input to oscillator k
            node_input += input_final;
          }
        }

        // printf("-- Heun's 2: Node %d - Differential Equations --\n", node);
        // ------------ Calculate Equations
        // Excitatory population (without noise and time) differentials
        differential_E2[node] = node_input - instance.config.c_ei * activity_I[node] + instance.config.external_e;
        differential_E2[node] = - activity_E[node] +
                                  (1 - instance.config.r_e * activity_E[node]) *
                                  response_function(differential_E2[node], instance.config.alpha_e, instance.config.theta_e);

        // Inhibitory population (without noise and time) differentials
        differential_I2[node] = instance.config.c_ie * activity_E[node];
        differential_I2[node] = - activity_I[node] +
                                  (1 - instance.config.r_i * activity_I[node]) *
                                  response_function(differential_I2[node], instance.config.alpha_i, instance.config.theta_i);

        // Second estimate of the new activity values
        instance.config.e_values[node] += (instance.config.integration_step_size /
                                                    2 * (differential_E[node] + differential_E2[node]) +
                                                  sqrt(instance.config.integration_step_size) * noises_array[node]) / instance.config.tau_e;
        instance.config.i_values[node] += (instance.config.integration_step_size /
                                                    2 * (differential_I[node] + differential_I2[node]) +
                                                  sqrt(instance.config.integration_step_size) * noises_array[node]) / instance.config.tau_i;

        // printf("-- Heun's 2: Node %d - Calculate ::output_e values --\n", node);
        instance.output_e[node][step] = instance.config.e_values[node];
      }
    }

    instance.electrical_activity = instance.output_e;

    // ------------- Got electrical activity
    printf("---- Shape of electrical activity: %d x %d----\n",
           (int)instance.electrical_activity.size(),
           (int)instance.electrical_activity[0].size());

    // ------------- Convert the signal to BOLD
    printf("---- Converting electrical activity to BOLD ----\n");
    auto bold_signal = instance.electrical_to_bold(instance.electrical_activity,
                                                   instance.config.number_of_oscillators,
                                                   instance.config.number_of_integration_steps,
                                                   instance.config.integration_step_size);

    // Saving it just for a sanity check
    printf("----------- Saving unpacked BOLD signal -----------\n");
    save_data_2D(bold_signal, "temp_arrays/unpacked_bold.csv");

    // TODO: It had better do that outside of this function by principe SOLID
    printf("----------- Filtering the BOLD signal -----------\n");
    std::vector<std::vector<double>> bold_filtered = process_BOLD(bold_signal,
                                                                  bold_signal.size(),
                                                                  bold_signal[0].size(),
                                                                  instance.config.order,
                                                                  instance.config.cutoffLow,
                                                                  instance.config.cutoffHigh,
                                                                  instance.config.sampling_rate);


    // Saving it just for a sanity check
    printf("----------- Saving filtered BOLD signal -----------\n");
    save_data_2D(bold_filtered, "temp_arrays/filtered_bold.csv");

    // Printing shape of bold signal
    printf("---- Shape of BOLD signal: %zd x %zd----\n", bold_signal.size(), bold_signal[0].size());

    // ------------- Determining the FC from the BOLD signal
    printf("----------- Determining FC from BOLD signal -----------\n");
    std::vector<std::vector<double>> sim_FC = determine_FC(bold_signal);

    // Checking the size of the output
    printf("FC matrix of size %d x %d\n", sim_FC.size(), sim_FC[0].size());

    printf("----------- Saving FC from BOLD signal -----------\n");
    std::string sim_FC_filename = "temp_arrays/sim_FC.csv";
    save_data_2D(sim_FC, sim_FC_filename);

    printf("----------- Comparing sim_FC with emp_FC -----------\n");
    // First, flatten the arrays
    std::vector<double> flat_sim_FC = flatten(sim_FC);
    std::vector<double> flat_emp_FC = flatten(instance.config.emp_FC);

    // Then, calculate the correlation
    double objective_corr = gsl_stats_correlation(flat_sim_FC.data(), 1, flat_emp_FC.data(), 1, flat_sim_FC.size());
    
    // This is finally the objective value
    return objective_corr;
}