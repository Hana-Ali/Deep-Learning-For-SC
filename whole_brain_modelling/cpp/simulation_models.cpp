#define PY_SSIZE_T_CLEAN
#include <random>
#include <string>
#include <fstream>
#include <gsl/gsl_statistics.h>
#include <numpy/arrayobject.h>
#include "cpp_headers/wilson_config.hpp"
#include "cpp_headers/kuramoto_config.hpp"
#include "cpp_headers/simulation_helpers.hpp"

/**
 * @brief Helper function to find the response of the Wilson model
 * @param x Input value
 * @param alpha Alpha parameter
 * @param theta Theta parameter
 * @return Response of the Wilson model
*/
double response_function(double x, double alpha, double theta)
{
    double S = 1 / (1 + exp(-alpha * (x - theta)));
    S -= 1 / (1 + exp(alpha * theta));

    return S;
}

/**
 * @brief Changes electrical activity to a BOLD signal
 * @param electrical_activity Electrical activity of the brain
 * @param number_of_oscillators Number of oscillators in the brain
 * @param number_of_integration_steps Number of integration steps
 * @param integration_step_size Integration step size
 * @return BOLD signal
*/
std::vector<std::vector<double>> Wilson::electrical_to_bold(std::vector<std::vector<double>>& electrical_activity,
                                                            int number_of_oscillators,
                                                            int number_of_integration_steps,
                                                            float integration_step_size)
{   
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
    printf("Unpacking BOLD signal...\n");
    // Printing the dimensions of the BOLD_array
    size_t BOLD_dims[2] = { output_bold_matrix.size(), output_bold_matrix[0].size() };

    auto& unpack_bold = output_bold_matrix;

    return unpack_bold;
}
   
/**
 * @brief Constructor for the Wilson class
 * @param config WilsonConfig object that contains the parameters for the Wilson model
*/
Wilson::Wilson(WilsonConfig config)
    : config(std::move(config))
{
}


/**
 * @brief Runs the simulation for the Wilson model
 * @return BOLD signal
 * @details This function runs the simulation for the Wilson model, and returns the BOLD signal
 *         as a 2D vector. The first dimension is the number of oscillators, and the second
 *        dimension is the number of integration steps
*/
std::vector<std::vector<double>> Wilson::run_simulation()
{

    // ------------- Declare input variables - arrays
    printf("----------------- In Wilson Objective -----------------\n");
    long temp_long; //
    double node_input;
    double delay_difference;
    int index_lower;
    int index_upper;
    double input_lower;
    double input_upper;
    double input_final;
    auto differential_E = std::vector<double>((*this).config.number_of_oscillators);
    auto differential_I = std::vector<double>((*this).config.number_of_oscillators);
    auto differential_E2 = std::vector<double>((*this).config.number_of_oscillators);
    auto differential_I2 = std::vector<double>((*this).config.number_of_oscillators);
    auto activity_E = std::vector<double>((*this).config.number_of_oscillators);
    auto activity_I = std::vector<double>((*this).config.number_of_oscillators);
    auto noises_array = std::vector<double>((*this).config.number_of_oscillators);

    // ------------- Defining the matrices that will keep changing
    printf("Define matrices that will keep changing...\n");
    (*this).coupling_mat.resize((*this).config.number_of_oscillators,
                                    std::vector<double>((*this).config.number_of_oscillators));
    for (int i = 0; i < (*this).config.number_of_oscillators; i++)
    {
      for (int j = 0; j < (*this).config.number_of_oscillators; j++)
      {
        if (i == j)
          (*this).coupling_mat[i][j] = (*this).config.c_ee;
        else
          (*this).coupling_mat[i][j] = (*this).config.coupling_strength * (*this).config.structural_connectivity_mat[i][j];
      }
    }

    (*this).delay_mat.resize((*this).config.number_of_oscillators,
                                std::vector<double>((*this).config.number_of_oscillators));
    for (int i = 0; i < (*this).config.number_of_oscillators; i++)
    {
      for (int j = 0; j < (*this).config.number_of_oscillators; j++)
      {
        if (i == j)
          (*this).delay_mat[i][j] = 0;
        else
          (*this).delay_mat[i][j] = (*this).config.delay * (*this).config.structural_connectivity_mat[i][j];
      }
    }

    // Create the indices matrices
    (*this).config.lower_idxs_mat.resize((*this).config.number_of_oscillators,
                                std::vector<int>((*this).config.number_of_oscillators));
    
    for (int i = 0; i < (*this).config.number_of_oscillators; i++)
    {
      for (int j = 0; j < (*this).config.number_of_oscillators; j++)
      {
        temp_long = (*this).delay_mat[i][j] / (*this).config.integration_step_size;
        (*this).config.lower_idxs_mat[i][j] = (int)temp_long;
      }
    }
    (*this).config.upper_idxs_mat.resize((*this).config.number_of_oscillators,
                                  std::vector<int>((*this).config.number_of_oscillators));
    for (int i = 0; i < (*this).config.number_of_oscillators; i++)
    {
      for (int j = 0; j < (*this).config.number_of_oscillators; j++)
      {
        (*this).config.upper_idxs_mat[i][j] = (*this).config.lower_idxs_mat[i][j] + 1;
      }
    }

    // ------------ Random generation
    std::default_random_engine generator(1);

    // ------------ TEMPORAL INTEGRATION
    printf("-> Temporal integration\n");
    for (int step = 1; step <= (*this).config.number_of_integration_steps; step++)
    {
      if (step % 10000 == 0)
        printf("Temporal integration step %d...\n", step);
      // ------------ Heun's Method - Step 1
      for (int node = 0; node < (*this).config.number_of_oscillators; node++)
      {
        // ------------ Initializations
        // Initialize input to node as 0
        node_input = 0;

        // Initialize noise
        if ((*this).config.noise == WilsonConfig::Noise::NOISE_NONE)
        {
          noises_array[node] = 0;
        }
        else if((*this).config.noise == WilsonConfig::Noise::NOISE_UNIFORM)
        {
          noises_array[node] = (*this).config.noise_amplitude * (2 * (*this).rand_std_uniform(generator) - 1);
        }
        else if((*this).config.noise == WilsonConfig::Noise::NOISE_NORMAL)
        {
          noises_array[node] = (*this).config.noise_amplitude * (*this).rand_std_normal(generator);
        }
        else
        {
          throw std::invalid_argument("Invalid noise type");
        }


        // ------------ Calculate input to node
        // Consider all other nodes, but only if the lower delay index is lower than the time point
        for (int other_node = 0; other_node < (*this).config.number_of_oscillators; other_node++)
        {
          if (step > (*this).config.lower_idxs_mat[node][other_node])
          {
            // Retrieve the difference between the 'true' delay and the one corresponding to the upper index
            delay_difference = (*this).delay_mat[node][other_node];
            delay_difference -= (double)(*this).config.upper_idxs_mat[node][other_node] *
                                        (*this).config.integration_step_size;

            // Retrieve the time point indices corresponding with the lower and upper delay indices
            index_lower = step - 1 - (*this).config.lower_idxs_mat[node][other_node];
            index_upper = step - 1 - (*this).config.upper_idxs_mat[node][other_node];

            // Retrieve the activities corresponding to the lower and upper delay indices
            input_lower = (*this).config.output_e[other_node][index_lower];
            input_upper = (*this).config.output_e[other_node][index_upper];

            // From the previously retrieved values, estimate the input to oscillator k from oscillator j
            input_final = input_upper;
            input_final += (input_lower - input_upper) / (*this).config.integration_step_size * delay_difference;
            // From this estimation, determine the quantile, final input
            input_final *= (*this).coupling_mat[node][other_node];
            // Add this to the total input to oscillator k
            node_input += input_final;
          }
        }

        // ------------ Calculate Equations
        // Excitatory population (without noise and time) differentials
        differential_E[node] = node_input -
                               (*this).config.c_ei * (*this).config.i_values[node] -
                               (*this).config.external_e;
        differential_E[node] = - (*this).config.e_values[node] +
                                 (1 - (*this).config.r_e * (*this).config.e_values[node]) *
                                 response_function(differential_E[node], (*this).config.alpha_e, (*this).config.theta_e);

        // Inhibitory population (without noise and time) differentials
        differential_I[node] = (*this).config.c_ie * (*this).config.e_values[node];
        differential_I[node] = - (*this).config.i_values[node] +
                                 (1 - (*this).config.r_i * (*this).config.i_values[node]) *
                                 response_function(differential_I[node], (*this).config.alpha_i, (*this).config.theta_i);

        // First estimate of the new activity values
        activity_E[node] = (*this).config.e_values[node] +
                           ((*this).config.integration_step_size * differential_E[node] +
                            sqrt((*this).config.integration_step_size) * noises_array[node]) / (*this).config.tau_e;
        activity_I[node] = (*this).config.i_values[node] +
                           ((*this).config.integration_step_size * differential_I[node] +
                            sqrt((*this).config.integration_step_size) * noises_array[node]) / (*this).config.tau_i;

        (*this).config.output_e[node][step] = activity_E[node];
      }

      // ------------ Heun's Method - Step 2
      for(int node = 0; node < (*this).config.number_of_oscillators; node++)
      {
        // Initialize input to node as 0
        node_input = 0;

        // ------------ Calculate input to node
        // Consider all other nodes, but only if the lower delay index is lower than the time point
        for (int other_node = 0; other_node < (*this).config.number_of_oscillators; other_node++)
        {
          if (step > (*this).config.lower_idxs_mat[node][other_node])
          {
            // Retrieve the difference between the 'true' delay and the one corresponding to the upper index
            delay_difference = (*this).delay_mat[node][other_node];
            delay_difference -= (double)(*this).config.upper_idxs_mat[node][other_node] *
                                        (*this).config.integration_step_size;

            // Retrieve the time point indices corresponding with the lower and upper delay indices
            index_lower = step - (*this).config.lower_idxs_mat[node][other_node];
            index_upper = step - (*this).config.upper_idxs_mat[node][other_node];

            // Retrieve the activities corresponding to the lower and upper delay indices
            input_lower = (*this).config.output_e[other_node][index_lower];
            input_upper = (*this).config.output_e[other_node][index_upper];

            // From the previously retrieved values, estimate the input to oscillator k from oscillator j
            input_final = input_upper;
            input_final += (input_lower - input_upper) / (*this).config.integration_step_size * delay_difference;
            // From this estimation, determine the quantile, final input
            input_final *= (*this).coupling_mat[node][other_node];
            // Add this to the total input to oscillator k
            node_input += input_final;
          }
        }

        // ------------ Calculate Equations
        // Excitatory population (without noise and time) differentials
        differential_E2[node] = node_input - (*this).config.c_ei * activity_I[node] + (*this).config.external_e;
        differential_E2[node] = - activity_E[node] +
                                  (1 - (*this).config.r_e * activity_E[node]) *
                                  response_function(differential_E2[node], (*this).config.alpha_e, (*this).config.theta_e);

        // Inhibitory population (without noise and time) differentials
        differential_I2[node] = (*this).config.c_ie * activity_E[node];
        differential_I2[node] = - activity_I[node] +
                                  (1 - (*this).config.r_i * activity_I[node]) *
                                  response_function(differential_I2[node], (*this).config.alpha_i, (*this).config.theta_i);

        // Second estimate of the new activity values
        (*this).config.e_values[node] += ((*this).config.integration_step_size /
                                                    2 * (differential_E[node] + differential_E2[node]) +
                                                  sqrt((*this).config.integration_step_size) * noises_array[node]) / (*this).config.tau_e;
        (*this).config.i_values[node] += ((*this).config.integration_step_size /
                                                    2 * (differential_I[node] + differential_I2[node]) +
                                                  sqrt((*this).config.integration_step_size) * noises_array[node]) / (*this).config.tau_i;

        // printf("-- Heun's 2: Node %d - Calculate ::output_e values --\n", node);
        (*this).config.output_e[node][step] = (*this).config.e_values[node];
      }
    }

    printf("Shape of output_e %d x % d\n", (*this).config.output_e.size(),
            (*this).config.output_e[0].size());


    // ------------- Convert the signal to BOLD
    printf("Converting electrical activity to BOLD...\n");
    std::vector<std::vector<double>> bold_signal = (*this).electrical_to_bold((*this).config.output_e,
                                                   (*this).config.number_of_oscillators,
                                                   (*this).config.number_of_integration_steps,
                                                   (*this).config.integration_step_size);

    for(auto& row:bold_signal) row.erase(std::next(row.begin(), 0));

    // Saving it just for a sanity check
    // printf("Saving unpacked BOLD signal...\n");
    // save_data_2D(bold_signal, "temp_arrays/sim_bold.csv");
    
    // This is finally the objective value
    return bold_signal;
}

#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Wilson Model, gets the *electrical activity* equations
static PyObject* parsing_wilson_inputs(PyObject* self, PyObject* args)
{
    printf("----------------- In CPP method of parsing inputs -----------------\n");

    PyObject* lower_idxs;
    PyObject* upper_idxs;
    PyObject* initial_cond_e;
    PyObject* initial_cond_i;
    PyObject* structural_connec;

    WilsonConfig config;
    if (
        !PyArg_ParseTuple(
            args, "ddOiddddddddddddddidOOOOid",
            &config.coupling_strength, &config.delay, &structural_connec,
            &config.number_of_oscillators, &config.c_ee,
            &config.c_ei, &config.c_ie, &config.c_ii,
            &config.tau_e, &config.tau_i, &config.r_e,
            &config.r_i, &config.alpha_e, &config.alpha_i,
            &config.theta_e, &config.theta_i, &config.external_e,
            &config.external_i, &config.number_of_integration_steps,
            &config.integration_step_size,
            &lower_idxs, &upper_idxs,
            &initial_cond_e, &initial_cond_i,
            &config.noise, &config.noise_amplitude
        )
        )
    {
      PyErr_SetString(PyExc_TypeError, "Input should be a numpy array of numbers.");
      printf("Parsing input variables failed\n");
      return NULL;
    };

    printf("Parsing input variables succeeded. Continuing...\n");

    // ------------- Convert input objects to C++ types
    printf("Converting input objects to C++ types...\n");
    // Collect all python objects in their own container
    WilsonConfig::PythonObjects python_objects = {
        structural_connec,
        lower_idxs,
        upper_idxs,
        initial_cond_e,
        initial_cond_i,
    };
    // Invoke the conversion function
    printf("Setting config...\n");
    set_config(&config, python_objects);

    // Call run simulation
    printf("Calling run simulation...\n");
    Wilson wilson(config);
    std::vector<std::vector<double>> sim_bold = wilson.run_simulation();

    printf("Finished running simulation...\n");

    // Convert to Python object
    printf("Converting to Python object...\n");
    // Create a numpy array of the BOLD signal
    npy_intp dims[2];
    dims[0] = config.number_of_oscillators;
    dims[1] = config.number_of_integration_steps;
    PyObject *temp_variable;
    PyObject *output_bold = PyArray_EMPTY(2, dims, NPY_FLOAT64, 0);

    for (int i = 0; i < dims[0]; i++)
    {   
        for (int step = 0; step < dims[1]; step++)
        {
            temp_variable = PyFloat_FromDouble(sim_bold[i][step]);
            PyArray_SETITEM(output_bold, PyArray_GETPTR2(output_bold, i, step), temp_variable);
            // Decrease reference for next
            Py_DECREF(temp_variable);
        }
    }

    // Return the output
    printf("Returning the output...\n");
    return output_bold;
}

// ************************************************************************************************
// ******************************* Kuramoto Simulations Definitions *******************************
/**
 * @brief Constructor for the Kuramoto class
 * @param config KuramotoConfig object that contains the parameters for the Kuramoto model
*/
Kuramoto::Kuramoto(KuramotoConfig config)
    : config(std::move(config))
{
}


/**
 * @brief Runs the simulation for the Wilson model
 * @return BOLD signal
 * @details This function runs the simulation for the Wilson model, and returns the BOLD signal
 *         as a 2D vector. The first dimension is the number of oscillators, and the second
 *        dimension is the number of integration steps
*/
std::vector<std::vector<double>> Kuramoto::run_simulation()
{

    // ------------- Declare input variables - arrays
    printf("----------------- In Wilson Objective -----------------\n");
    long temp_long; //
    double node_input;
    double receiving_input;
    double delay_difference;
    int index_lower;
    int index_upper;
    double input_lower;
    double input_upper;
    double input_final;
    auto differential_Phi = std::vector<double>((*this).config.number_of_oscillators);
    auto differential_Phi2 = std::vector<double>((*this).config.number_of_oscillators);
    auto init_phis = std::vector<double>((*this).config.number_of_oscillators);
    auto noises_array = std::vector<double>((*this).config.number_of_oscillators);

    // ------------- Defining the matrices that will keep changing
    printf("Define matrices that will keep changing...\n");
    (*this).coupling_mat.resize((*this).config.number_of_oscillators,
                                    std::vector<double>((*this).config.number_of_oscillators));
    for (int i = 0; i < (*this).config.number_of_oscillators; i++)
    {
      for (int j = 0; j < (*this).config.number_of_oscillators; j++)
      {
        (*this).coupling_mat[i][j] = (*this).config.coupling_strength * (*this).config.structural_connectivity_mat[i][j];
      }
    }

    (*this).delay_mat.resize((*this).config.number_of_oscillators,
                                std::vector<double>((*this).config.number_of_oscillators));
    for (int i = 0; i < (*this).config.number_of_oscillators; i++)
    {
      for (int j = 0; j < (*this).config.number_of_oscillators; j++)
      {
        if (i == j)
          (*this).delay_mat[i][j] = 0;
        else
          (*this).delay_mat[i][j] = (*this).config.delay * (*this).config.structural_connectivity_mat[i][j];
      }
    }

    // Create the indices matrices
    (*this).config.lower_idxs_mat.resize((*this).config.number_of_oscillators,
                                std::vector<int>((*this).config.number_of_oscillators));
    
    for (int i = 0; i < (*this).config.number_of_oscillators; i++)
    {
      for (int j = 0; j < (*this).config.number_of_oscillators; j++)
      {
        temp_long = (*this).delay_mat[i][j] / (*this).config.integration_step_size;
        (*this).config.lower_idxs_mat[i][j] = (int)temp_long;
      }
    }
    (*this).config.upper_idxs_mat.resize((*this).config.number_of_oscillators,
                                  std::vector<int>((*this).config.number_of_oscillators));
    for (int i = 0; i < (*this).config.number_of_oscillators; i++)
    {
      for (int j = 0; j < (*this).config.number_of_oscillators; j++)
      {
        (*this).config.upper_idxs_mat[i][j] = (*this).config.lower_idxs_mat[i][j] + 1;
      }
    }

    // ------------ Random generation
    std::default_random_engine generator(1);

    // ------------ TEMPORAL INTEGRATION
    printf("-> Temporal integration\n");
    for (int step = 1; step <= (*this).config.number_of_integration_steps; step++)
    {
      if (step % 10000 == 0)
        printf("Temporal integration Kuramoto step %d...\n", step);
      // ------------ Heun's Method - Step 1
      for (int node = 0; node < (*this).config.number_of_oscillators; node++)
      {
        // ------------ Initializations
        // Initialize input to node as 0
        node_input = 0;
        receiving_input = (*this).config.output_phi[node][step - 1];

        // Initialize noise
        if ((*this).config.noise == KuramotoConfig::Noise::NOISE_NONE)
        {
          noises_array[node] = 0;
        }
        else if((*this).config.noise == KuramotoConfig::Noise::NOISE_UNIFORM)
        {
          noises_array[node] = (*this).config.noise_amplitude * (2 * (*this).rand_std_uniform(generator) - 1);
        }
        else if((*this).config.noise == KuramotoConfig::Noise::NOISE_NORMAL)
        {
          noises_array[node] = (*this).config.noise_amplitude * (*this).rand_std_normal(generator);
        }
        else
        {
          throw std::invalid_argument("Invalid noise type");
        }


        // ------------ Calculate input to node
        // Consider all other nodes, but only if the lower delay index is lower than the time point
        for (int other_node = 0; other_node < (*this).config.number_of_oscillators; other_node++)
        {
          if (step > (*this).config.lower_idxs_mat[node][other_node])
          {
            // Retrieve the difference between the 'true' delay and the one corresponding to the upper index
            delay_difference = (*this).delay_mat[node][other_node];
            delay_difference -= (double)(*this).config.upper_idxs_mat[node][other_node] *
                                        (*this).config.integration_step_size;

            // Retrieve the time point indices corresponding with the lower and upper delay indices
            index_lower = step - 1 - (*this).config.lower_idxs_mat[node][other_node];
            index_upper = step - 1 - (*this).config.upper_idxs_mat[node][other_node];

            // Retrieve the activities corresponding to the lower and upper delay indices
            input_lower = (*this).config.output_phi[other_node][index_lower];
            input_upper = (*this).config.output_phi[other_node][index_upper];

            // From the previously retrieved values, estimate the input to oscillator k from oscillator j
            input_final = input_upper;
            input_final += (input_lower - input_upper) / (*this).config.integration_step_size * delay_difference;
            // From this estimation, determine the quantile, final input
            input_final = sin(input_final - receiving_input);
            input_final *= (*this).coupling_mat[node][other_node];
            // Add this to the total input to oscillator k
            node_input += input_final;
          }
        }

        // ------------ Calculate Equations
        // Excitatory population (without noise and time) differentials
        differential_Phi[node] = node_input - (*this).config.freq_omega[node];
        // First estimate of the new activity values
        init_phis[node] = (*this).config.phases_phi[node] +
                        ((*this).config.integration_step_size * differential_Phi[node] +
                        sqrt((*this).config.integration_step_size) * noises_array[node]);
        
        (*this).config.output_phi[node][step] = init_phis[node];
      }

      // ------------ Heun's Method - Step 2
      for(int node = 0; node < (*this).config.number_of_oscillators; node++)
      {
        // Initialize input to node as 0
        node_input = 0;
        receiving_input = (*this).config.output_phi[node][step];

        // ------------ Calculate input to node
        // Consider all other nodes, but only if the lower delay index is lower than the time point
        for (int other_node = 0; other_node < (*this).config.number_of_oscillators; other_node++)
        {
          if (step > (*this).config.lower_idxs_mat[node][other_node])
          {
            // Retrieve the difference between the 'true' delay and the one corresponding to the upper index
            delay_difference = (*this).delay_mat[node][other_node];
            delay_difference -= (double)(*this).config.upper_idxs_mat[node][other_node] *
                                        (*this).config.integration_step_size;

            // Retrieve the time point indices corresponding with the lower and upper delay indices
            index_lower = step - (*this).config.lower_idxs_mat[node][other_node];
            index_upper = step - (*this).config.upper_idxs_mat[node][other_node];

            // Retrieve the activities corresponding to the lower and upper delay indices
            input_lower = (*this).config.output_phi[other_node][index_lower];
            input_upper = (*this).config.output_phi[other_node][index_upper];

            // From the previously retrieved values, estimate the input to oscillator k from oscillator j
            input_final = input_upper;
            input_final += (input_lower - input_upper) / (*this).config.integration_step_size * delay_difference;
            // From this estimation, determine the quantile, final input
            input_final = sin(input_final - receiving_input);
            input_final *= (*this).coupling_mat[node][other_node];
            // Add this to the total input to oscillator k
            node_input += input_final;
            }
        }

        // ------------ Calculate Equations
        // Excitatory population (without noise and time) differentials
        differential_Phi2[node] = (*this).config.freq_omega[node] + node_input;
        
        (*this).config.phases_phi[node] = (*this).config.phases_phi[node] +
                        ((*this).config.integration_step_size / 2) *
                        (differential_Phi[node] + differential_Phi2[node]) +
                        sqrt((*this).config.integration_step_size) * noises_array[node];
        
        (*this).config.output_phi[node][step] = (*this).config.phases_phi[node];
      
        }
    }

    printf("Shape of output_phi %d x % d\n", (*this).config.output_phi.size(),
            (*this).config.output_phi[0].size());
    
    // Save the output to a file
    // printf("Saving output to file...\n");
    // save_data_2D((*this).config.output_phi, "temp_arrays/output_phi.csv");
    
    // Return the output
    std::vector<std::vector<double>> output_phi = (*this).config.output_phi;

    return output_phi;
}

#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Wilson Model, gets the *electrical activity* equations
static PyObject* parsing_kuramoto_inputs(PyObject* self, PyObject* args)
{
    printf("----------------- In CPP method of parsing inputs -----------------\n");

    PyObject* lower_idxs;
    PyObject* upper_idxs;
    PyObject* phis_array;
    PyObject* omega_array;
    PyObject* structural_connec;

    KuramotoConfig config;
    if (
        !PyArg_ParseTuple(
            args, "ddOiidOOOOid",
            &config.coupling_strength, &config.delay, &structural_connec,
            &config.number_of_oscillators, &config.number_of_integration_steps,
            &config.integration_step_size,
            &lower_idxs, &upper_idxs,
            &phis_array, &omega_array,
            &config.noise, &config.noise_amplitude
        )
        )
    {
      PyErr_SetString(PyExc_TypeError, "Input should be a numpy array of numbers.");
      printf("Parsing input variables failed\n");
      return NULL;
    };

    printf("Parsing input variables succeeded. Continuing...\n");

    // ------------- Convert input objects to C++ types
    printf("Converting input objects to C++ types...\n");
    // Collect all python objects in their own container
    KuramotoConfig::PythonObjects python_objects = {
        structural_connec,
        lower_idxs,
        upper_idxs,
        phis_array,
        omega_array,
    };
    // Invoke the conversion function
    printf("Setting config...\n");
    set_kura_config(&config, python_objects);

    // Call run simulation
    printf("Calling run simulation...\n");
    Kuramoto kuramoto(config);
    std::vector<std::vector<double>> output_phi = kuramoto.run_simulation();

    printf("Size of output_phi %d x % d\n", output_phi.size(), output_phi[0].size());

    printf("Finished running simulation...\n");

    // Convert to Python object
    printf("Converting to Python object...\n");
    // Create a numpy array of the BOLD signal
    npy_intp dims[2];
    dims[0] = output_phi.size();
    dims[1] = output_phi[0].size();
    PyObject *temp_variable;
    PyObject *phi_history = PyArray_EMPTY(2, dims, NPY_FLOAT64, 0);

    for (int i = 0; i < dims[0]; i++)
    {   
        for (int step = 0; step < dims[1]; step++)
        {
            temp_variable = PyFloat_FromDouble(output_phi[i][step]);
            PyArray_SETITEM(phi_history, PyArray_GETPTR2(phi_history, i, step), temp_variable);
            // Decrease reference for next
            Py_DECREF(temp_variable);
        }
    }

    // Return the output
    printf("Returning the output...\n");
    return phi_history;
}

// ************************************************************************************************
// ******************************* Python Module Definitions **************************************

static PyMethodDef IntegrationMethods[] = {
    {
        "parsing_wilson_inputs",
        parsing_wilson_inputs,
        METH_VARARGS,
        "Solves the Wilson-Cowan model equations, and returns electrical activity"
    },
    {
      "parsing_kuramoto_inputs",
      parsing_kuramoto_inputs,
      METH_VARARGS,
      "Solves the Kuramoto model equations, and returns the phases"
    },
    { // Sentinel to properly exit function
        NULL, NULL, 0, NULL
    }
};

// Function that wraps the methods in a module
static struct PyModuleDef SimulationsModule = {
    PyModuleDef_HEAD_INIT,
    "simulations",
    "Module containing functions for simulation compiled in C",
    -1,
    IntegrationMethods
};

// Function that creates the modules
PyMODINIT_FUNC PyInit_simulations(void)
{
    import_array();
    return PyModule_Create(&SimulationsModule);
}