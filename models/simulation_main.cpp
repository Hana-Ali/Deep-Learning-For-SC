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
#include <bayesopt/bayesopt.hpp>
#include <numpy/arrayobject.h>
#include "simulation_helpers.hpp"

// ------------------------------------- DEFINING GLOBAL PARAMETERS -------------------------------------
//wilson
int *wilson_number_of_oscillators = new int;                   
double *wilson_c_ee = new double;                      
double *wilson_c_ei = new double;
double *wilson_c_ie = new double;
double *wilson_c_ii = new double;
double *wilson_tau_e = new double;
double *wilson_tau_i = new double;
double *wilson_r_e = new double;
double *wilson_r_i = new double;
double *wilson_alpha_e = new double;
double *wilson_alpha_i = new double;
double *wilson_theta_e = new double;
double *wilson_theta_i = new double;
double *wilson_external_e = new double;
double *wilson_external_i = new double;
int *wilson_number_of_integration_steps = new int;
double *wilson_integration_step_size = new double;
int *wilson_noise_type = new int;
double *wilson_noise_amplitude = new double;
double *wilson_e_values = NULL;
double *wilson_i_values = NULL;
double *wilson_coupling_mat = NULL;
double *wilson_delay_mat = NULL;
int *wilson_lower_idxs_mat = NULL;
int *wilson_upper_idxs_mat = NULL;
double *wilson_output_e = NULL;
PyObject *wilson_electrical_activity;


// Defining random distributions
std::normal_distribution<double> rand_std_normal (0, 1);
std::uniform_real_distribution<double> rand_std_uniform (0, 1);

// Function declarations
std::vector<std::vector<double>> electrical_to_bold();
double wilson_response_function(double x, double alpha, double theta);
static PyObject *parsing_wilson_inputs(PyObject *self, PyObject *args);
double wilson_objective(unsigned int input_dim, const double *initial_query, double* gradient, void *func_data);

std::vector<std::vector<double>> electrical_to_bold()
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

   // ------------- Declare input variables
    double *input_e = NULL;
    
    // ------------- Declare state variables
    double *f = NULL;
    double *q = NULL;
    double *s = NULL;
    double *v = NULL;
    double *differential_f = NULL;
    double *differential_q = NULL;
    double *differential_s = NULL;
    double *differential_v = NULL;
    double *differential_f2 = NULL;
    double *differential_q2 = NULL;
    double *differential_s2 = NULL;
    double *differential_v2 = NULL;
    double *activity_f = NULL;
    double *activity_q = NULL;
    double *activity_s = NULL;
    double *activity_v = NULL;

    // ------------- Declare helper variables
    double *alpha = NULL;
    double *gamma = NULL;
    double *kappa = NULL;
    double *rho = NULL;
    double *tau = NULL;
    double *c1 = NULL;
    double *c2 = new double;
    double *c3 = NULL;
    double *V0 = new double;
    PyObject *temp_variable;
    npy_intp *e_array_size;

    // Define input variables
    input_e = new double[*::wilson_number_of_oscillators];

    // ------------- Declare output variables
    double* output_bold = new double[*::wilson_number_of_oscillators];
    PyObject *BOLD_array;
    npy_intp dimensions[2] = {*::wilson_number_of_oscillators, *::wilson_number_of_integration_steps + 1};
    BOLD_array = PyArray_EMPTY(2, dimensions, NPY_FLOAT64, 0);
        
    // ------------- Define state variables
    f = new double[*::wilson_number_of_oscillators];
    q = new double[*::wilson_number_of_oscillators];
    s = new double[*::wilson_number_of_oscillators];
    v = new double[*::wilson_number_of_oscillators];
    differential_f = new double[*::wilson_number_of_oscillators];
    differential_q = new double[*::wilson_number_of_oscillators];
    differential_s = new double[*::wilson_number_of_oscillators];
    differential_v = new double[*::wilson_number_of_oscillators];
    differential_f2 = new double[*::wilson_number_of_oscillators];
    differential_q2 = new double[*::wilson_number_of_oscillators];
    differential_s2 = new double[*::wilson_number_of_oscillators];
    differential_v2 = new double[*::wilson_number_of_oscillators];
    activity_f = new double[*::wilson_number_of_oscillators];
    activity_q = new double[*::wilson_number_of_oscillators];
    activity_s = new double[*::wilson_number_of_oscillators];
    activity_v = new double[*::wilson_number_of_oscillators];

    // ------------- Define helper variables
    alpha = new double[*::wilson_number_of_oscillators];
    gamma = new double[*::wilson_number_of_oscillators];
    kappa = new double[*::wilson_number_of_oscillators];
    rho = new double[*::wilson_number_of_oscillators];
    tau = new double[*::wilson_number_of_oscillators];
    c1 = new double[*::wilson_number_of_oscillators];
    c3 = new double[*::wilson_number_of_oscillators];

    std::default_random_engine generator(1);

    // ------------- Check numpy array with electrical signal to ensure correct dimensions and type
    // Check it's a numpy array
    if (!PyArray_Check(::wilson_electrical_activity))
    {
        std::string warning_string = "Expected a numpy array for the electrical activity, but it is " + std::to_string(PyArray_Check(::wilson_electrical_activity));
        PyErr_SetString(PyExc_TypeError, warning_string.c_str());
        return {};
    }
    // Check it's a 2D array
    if (PyArray_NDIM(::wilson_electrical_activity) != 2)
    {
        std::string warning_string = "Expected a 2D numpy array for the electrical activity, but it is " + std::to_string(PyArray_NDIM(::wilson_electrical_activity)) + "D";
        PyErr_SetString(PyExc_TypeError, warning_string.c_str());
        return {};
    }
    // Check it's a float64 array
    if (PyArray_TYPE(::wilson_electrical_activity) != NPY_FLOAT64)
    {
        std::string warning_string = "Expected a float64 numpy array for the electrical activity, but it is " + std::to_string(PyArray_TYPE(::wilson_electrical_activity));
        PyErr_SetString(PyExc_TypeError, warning_string.c_str());
        return {};
    }
    // Check it has the correct dimensions
    e_array_size = PyArray_DIMS(::wilson_electrical_activity);
    if (e_array_size[0] != *::wilson_number_of_oscillators || e_array_size[1] != *::wilson_number_of_integration_steps + 1)
    {   
        std::string warning_string = "Expected a numpy array with the dimensions (*num_osc, *n_step + 1) for the electrical activity, but it's " + std::to_string(e_array_size[0]) + "x" + std::to_string(e_array_size[1]);
        PyErr_SetString(PyExc_TypeError, warning_string.c_str());
        return {};
    }

    // ------------- Initialize values of state variables, [0, 0.1]
    for (int i = 0; i < *::wilson_number_of_oscillators; i++)
    {
        // Initialize state variables
        f[i] = rand_std_uniform(generator) * 0.1;
        q[i] = rand_std_uniform(generator) * 0.1;
        s[i] = rand_std_uniform(generator) * 0.1;
        v[i] = rand_std_uniform(generator) * 0.1;
    }

    // ------------- Initialize values of helper variables
    *c2 = 2.000;
    *V0 = 0.020;
    for (int i = 0; i < *::wilson_number_of_oscillators; i++)
    {
        // Initialize helper variables
        alpha[i] = 1 / (0.320 + rand_std_normal(generator) * 0.039);
        gamma[i] = 0.410 + rand_std_normal(generator) * 0.045;
        kappa[i] = 0.650 + rand_std_normal(generator) * 0.122;
        rho[i] = 0.340 + rand_std_normal(generator) * 0.049;
        tau[i] = 0.980 + rand_std_normal(generator) * 0.238;
        c1[i] = 7.0 * rho[i];
        c3[i] = 2.0 * rho[i] - 0.2;
    }

    // ------------- Initialize output matrix
    for (int i = 0; i < *::wilson_number_of_oscillators; i++)
    {
        output_bold[i] = c1[i] * (1 - q[i]);
        output_bold[i] += *c2 * (1 - q[i] / v[i]);
        output_bold[i] += c3[i] * (1 - v[i]);
        output_bold[i] *= *V0;
    }

    // ------------- CONVERSIONS BABEY
    for (int step = 1; step <= *::wilson_number_of_integration_steps; step++)
    {
        // Get the electrical signal for this timestep from the Python input array
        for (int i = 0; i < *::wilson_number_of_oscillators; i++)
        {
            temp_variable = PyArray_GETITEM(::wilson_electrical_activity, PyArray_GETPTR2(::wilson_electrical_activity, i, step));
            input_e[i] = PyFloat_AsDouble(temp_variable);
            // Decrease reference for next
            Py_DECREF(temp_variable);
        }

        // ------------ Heun's Method - Step 1
        for (int i = 0; i < *::wilson_number_of_oscillators; i++)
        {
            // Calculate differentials
            differential_f[i] = s[i];
            differential_q[i] = 1 - pow(1 - rho[i], 1 / f[i]);
            differential_q[i] *= f[i] / rho[i];
            differential_q[i] -= q[i] * pow(v[i], alpha[i] - 1);
            differential_q[i] /= tau[i];
            differential_s[i] = input_e[i];
            differential_s[i] -= kappa[i] * s[i] + gamma[i] * (f[i] - 1);
            differential_v[i] = (f[i] - pow(v[i], alpha[i])) / tau[i];

            // First estimate of the new activity values
            activity_f[i] = f[i] + *::wilson_integration_step_size * differential_f[i];
            activity_q[i] = q[i] + *::wilson_integration_step_size * differential_q[i];
            activity_s[i] = s[i] + *::wilson_integration_step_size * differential_s[i];
            activity_v[i] = v[i] + *::wilson_integration_step_size * differential_v[i];
        }
        // ------------ Heun's Method - Step 2
        for (int j = 0; j < *::wilson_number_of_oscillators; j++)
        {
            // Calculate differentials
            differential_f2[j] = activity_s[j];
            differential_q2[j] = 1 - pow(1 - rho[j], 1 / activity_f[j]);
            differential_q2[j] *= activity_f[j] / rho[j];
            differential_q2[j] -= activity_q[j] * pow(activity_v[j], alpha[j] - 1);
            differential_q2[j] /= tau[j];
            differential_s2[j] = input_e[j];
            differential_s2[j] -= kappa[j] * activity_s[j] + gamma[j] * (activity_f[j] - 1);
            differential_v2[j] = (activity_f[j] - pow(activity_v[j], alpha[j])) / tau[j];

            // Second estimate of the new activity values
            f[j] += *::wilson_integration_step_size / 2 * (differential_f[j] + differential_f2[j]);
            q[j] += *::wilson_integration_step_size / 2 * (differential_q[j] + differential_q2[j]);
            s[j] += *::wilson_integration_step_size / 2 * (differential_s[j] + differential_s2[j]);
            v[j] += *::wilson_integration_step_size / 2 * (differential_v[j] + differential_v2[j]);
        }

        // Calculate BOLD signal
        for (int osc = 0; osc < *::wilson_number_of_oscillators; osc++)
        {
            output_bold[osc] = c1[osc] * (1 - q[osc]);
            output_bold[osc] += *c2 * (1 - q[osc] / v[osc]);
            output_bold[osc] += c3[osc] * (1 - v[osc]);
            output_bold[osc] *= *V0;
            // Put into temporary variable
            temp_variable = PyFloat_FromDouble(output_bold[osc]);
            // Set item in BOLD_array
            PyArray_SETITEM(BOLD_array, PyArray_GETPTR2(BOLD_array, osc, step), temp_variable);
            // Decrease reference for next
            Py_DECREF(temp_variable);
        }
    }

    // ------------- Unpack the BOLD signal
    printf("----------- Unpacking BOLD signal -----------\n");
    // Printing the dimensions of the BOLD_array
    npy_intp BOLD_dims[] = {PyArray_NDIM(BOLD_array)};
    BOLD_dims[0] = PyArray_DIM(BOLD_array, 0);
    BOLD_dims[1] = PyArray_DIM(BOLD_array, 1);

    // Create a vector of doubles, to store the entire BOLD signal
    std::vector<std::vector<double>> unpack_bold;
    PyObject* time_sample;
    
    for (int i = 0; i < BOLD_dims[0]; ++i)
    {   
        printf("In processing signal %d\n", i);
        
        // Create another vector of doubles, to store timesamples for each BOLD signal
        std::vector<double> samples_array;
        
        // For each BOLD signal in the BOLD signals, for each timestep
        for (int j = 0; j < BOLD_dims[1]; ++j)
        {   
            // if (j % 10000 == 0)
            //     printf("In timesample %d\n", j);
            
            // This will store the value in the bold array
            double value;

            // Get the time_sample point
            time_sample = PyArray_GETITEM(BOLD_array, PyArray_GETPTR2(BOLD_array, i, j));

            // Check thet each time sample is a float
            if(PyFloat_Check(time_sample))
                value = PyFloat_AsDouble(time_sample);
            else {
                printf("Not floats!!!");
                PyErr_SetString(PyExc_TypeError, "must pass in list of list of number");
                return {};
            }
            samples_array.push_back(value);
            // Decrement the pointer reference
            Py_DECREF(time_sample);
        }
        unpack_bold.push_back(samples_array);
    }

    // Saving it just for a sanity check
    printf("----------- Saving unpacked BOLD signal -----------\n");
    std::ofstream myfile;
    myfile.open("temp_arrays/unpacked_bold.csv");
    
    for (size_t i = 0; i < BOLD_dims[0]; ++i)
    {
        for (size_t j = 0; j < BOLD_dims[1]; ++j)
            if (j < (BOLD_dims[1] - 1)) {
                myfile << unpack_bold[i][j] << ",";
            }
            else if (j == (BOLD_dims[1] - 1)) {
                myfile << unpack_bold[i][j] << "\n";
            }
    }

    printf("----------- Filtering the BOLD signal -----------\n");
    // Defining filter parameters
    int order = 4;
    double cutoffLow = 1.5;
    double cutoffHigh = 2.5;
    double sampling_rate = 20;
    std::vector<std::vector<double>> bold_filtered = process_BOLD(unpack_bold, BOLD_dims[0], BOLD_dims[1], order,
                                                                    cutoffLow, cutoffHigh, sampling_rate);


    // Saving it just for a sanity check
    printf("----------- Saving filtered BOLD signal -----------\n");
    std::ofstream myfile2;
    myfile2.open("temp_arrays/filtered_bold.csv");
    
    for (size_t i = 0; i < BOLD_dims[0]; ++i)
    {
        for (size_t j = 0; j < BOLD_dims[1]; ++j)
            if (j < (BOLD_dims[1] - 1)) {
                myfile2 << bold_filtered[i][j] << ",";
            }
            else if (j == (BOLD_dims[1] - 1)) {
                myfile2 << bold_filtered[i][j] << "\n";
            }
    }

    // ------------- Free memory
    // Delete input variables
    delete[] input_e;

    // Delete state variables
    delete[] f;
    delete[] q;
    delete[] s;
    delete[] v;
    delete[] differential_f;
    delete[] differential_q;
    delete[] differential_s;
    delete[] differential_v;
    delete[] differential_f2;
    delete[] differential_q2;
    delete[] differential_s2;
    delete[] differential_v2;
    delete[] activity_f;
    delete[] activity_q;
    delete[] activity_s;
    delete[] activity_v;

    // Delete helper variables
    delete[] alpha;
    delete[] gamma;
    delete[] kappa;
    delete[] rho;
    delete[] tau;
    delete[] c1;
    delete c2;
    delete[] c3;
    delete V0;

    // ------------- Return output
    return unpack_bold;
}

// Response function for Wilson Model
double wilson_response_function(double x, double alpha, double theta)
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

// Wilson Model, gets the *electrical activity* equations
static PyObject* parsing_wilson_inputs(PyObject* self, PyObject *args)
{
    /*
    This function takes in the parameters of the WC model from Python, unpacks them to C++ objects,
    then sends them to the objective function, which then does the actual computation.
    
    Parameters
    ----------
    args : tuple, input model arguments
        args[0] : array, coupling strength
        args[1] : array, delay
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
        args[19] : array, lower_idxs
        args[20] : array, upper_idxs
        args[21] : array, initial conditions e
        args[22] : array, initial conditions i
        args[23] : int, noise type
        args[24] : float, noise amplitude
    
    Returns
    -------
    electrical_activity : array, electrical activity of each node wrapped into numpy array

    Equations
    ---------
    tau_e * dE/dt = -E + (1 - r_e*E)*S_e(c_ee*E + c_ei*I + external_e)
    tau_i * dI/dt = -I + (1 - r_i*I)*S_I(c_ie*E + c_ii*I + external_i)

    where S_e and S_i are sigmoid functions
        S_e = 1 / (1 + exp(-alpha_e * (x - theta_e)))
        S_i = 1 / (1 + exp(-alpha_i * (x - theta_i)))
    */

    // ------------- Declare input variables - not arrays
    printf("----------------- In CPP file for Wilson Function -----------------\n");
    // ------------- Declare input variables - arrays
    PyObject *coupling_strength;
    PyObject *delay;
    PyObject *lower_idxs;
    PyObject *upper_idxs;
    PyObject *initial_cond_e;
    PyObject *initial_cond_i;

    // ------------- Declare helper variables
    long *temp_long = new long;
    double *node_input = new double;
    double *delay_difference = new double;
    int *index_lower = new int;
    int *index_upper = new int;
    double *input_lower = new double;
    double *input_upper = new double;
    double *input_final = new double;
    double *differential_E = NULL;
    double *differential_I = NULL;
    double *differential_E2 = NULL;
    double *differential_I2 = NULL;
    double *activity_E = NULL;
    double *activity_I = NULL;
    double *noises_array = NULL;
    PyObject *temp_variable;

    // ------------- Declare output variables
    npy_intp dimensions[2];

    // ------------- Parse input variables
    printf("---- Parsing input variables ----\n");
    if(
        !PyArg_ParseTuple(
            args, "OOiddddddddddddddidOOOOid",
            &coupling_strength, &delay, 
            ::wilson_number_of_oscillators, ::wilson_c_ee, 
            ::wilson_c_ei, ::wilson_c_ie, ::wilson_c_ii, 
            ::wilson_tau_e, ::wilson_tau_i, ::wilson_r_e, 
            ::wilson_r_i, ::wilson_alpha_e, ::wilson_alpha_i, 
            ::wilson_theta_e, ::wilson_theta_i, ::wilson_external_e, 
            ::wilson_external_i, ::wilson_number_of_integration_steps, 
            ::wilson_integration_step_size,
            &lower_idxs, &upper_idxs, 
            &initial_cond_e, &initial_cond_i,
            ::wilson_noise_type, ::wilson_noise_amplitude
        )
    )
    {
        printf("---- Parsing input variables failed ----\n");
        return NULL;
    };

    // ------------- Allocate memory
    // Allocate memory for input and helper variables
    ::wilson_e_values = new double[*::wilson_number_of_oscillators];
    ::wilson_i_values = new double[*::wilson_number_of_oscillators];
    ::wilson_coupling_mat = new double[*::wilson_number_of_oscillators * *::wilson_number_of_oscillators];
    ::wilson_delay_mat = new double[*::wilson_number_of_oscillators * *::wilson_number_of_oscillators];
    ::wilson_lower_idxs_mat = new int[*::wilson_number_of_oscillators * *::wilson_number_of_oscillators];
    ::wilson_upper_idxs_mat = new int[*::wilson_number_of_oscillators * *::wilson_number_of_oscillators];

    // Allocate memory for output variables
    dimensions[0] = *::wilson_number_of_oscillators;
    dimensions[1] = *::wilson_number_of_integration_steps + 1;
    ::wilson_output_e = new double[*::wilson_number_of_oscillators * (*::wilson_number_of_integration_steps + 1)];
    ::wilson_electrical_activity = PyArray_EMPTY(2, dimensions, NPY_FLOAT64, 0);

    // ------------- Convert input variables to C++ types
    printf("---- Converting input variables to C++ types ----\n");
    for (int i = 0; i < *::wilson_number_of_oscillators; i++)
    {   
        // Get the initial conditions - EXCITATORY
        temp_variable = PyArray_GETITEM(initial_cond_e, PyArray_GETPTR1(initial_cond_e, i));
        ::wilson_e_values[i] = PyFloat_AsDouble(temp_variable);
        // Decrease reference for next
        Py_DECREF(temp_variable);

        // Get the initial conditions - INHIBITORY
        temp_variable = PyArray_GETITEM(initial_cond_i, PyArray_GETPTR1(initial_cond_i, i));
        ::wilson_i_values[i] = PyFloat_AsDouble(temp_variable);
        // Decrease reference for next
        Py_DECREF(temp_variable);

        // ------------ Matrices
        for (int j = 0; j < *::wilson_number_of_oscillators; j++)
        {   
            // Get the coupling strength matrix
            temp_variable = PyArray_GETITEM(coupling_strength, PyArray_GETPTR2(coupling_strength, i, j));
            ::wilson_coupling_mat[i * *::wilson_number_of_oscillators + j] = PyFloat_AsDouble(temp_variable);
            // Decrease reference for next
            Py_DECREF(temp_variable);

            // Get the delay matrix
            temp_variable = PyArray_GETITEM(delay, PyArray_GETPTR2(delay, i, j));
            ::wilson_delay_mat[i * *::wilson_number_of_oscillators + j] = PyFloat_AsDouble(temp_variable);
            // Decrease reference for next
            Py_DECREF(temp_variable);

            // Get the lower_idxs matrix
            temp_variable = PyArray_GETITEM(lower_idxs, PyArray_GETPTR2(lower_idxs, i, j));
            *temp_long = PyLong_AsLong(temp_variable);
            ::wilson_lower_idxs_mat[i * *::wilson_number_of_oscillators + j] = *temp_long;
            // Decrease reference for next
            Py_DECREF(temp_variable);

            // Get the upper_idxs matrix
            temp_variable = PyArray_GETITEM(upper_idxs, PyArray_GETPTR2(upper_idxs, i, j));
            *temp_long = PyLong_AsLong(temp_variable);
            ::wilson_upper_idxs_mat[i * *::wilson_number_of_oscillators + j] = *temp_long;
            // Decrease reference for next
            Py_DECREF(temp_variable);
        }

        // ------------ Initialize output matrix
        ::wilson_output_e[i * (*::wilson_number_of_integration_steps + 1)] = ::wilson_e_values[i];
        // Other values in matrix are NaN
        for (int step = 1; step <= *::wilson_number_of_integration_steps; step++)
        {
            ::wilson_output_e[i * (*::wilson_number_of_integration_steps + 1) + step] = nan("");
        }
    }

    // ------------ Send data to objective function
    printf("---- Send data to objective function ----\n");

    // Call the objective function
    wilson_objective(2, NULL, NULL, NULL);

    // ------------- Free memory
    printf("---- Free memory ----\n");
    // Delete input variables
        // Delete global wilson variables
    delete ::wilson_number_of_oscillators;
    delete ::wilson_c_ee;
    delete ::wilson_c_ei;
    delete ::wilson_c_ie;
    delete ::wilson_c_ii;
    delete ::wilson_tau_e;
    delete ::wilson_tau_i;
    delete ::wilson_r_e;
    delete ::wilson_r_i;
    delete ::wilson_alpha_e;
    delete ::wilson_alpha_i;
    delete ::wilson_theta_e;
    delete ::wilson_theta_i;
    delete ::wilson_external_e;
    delete ::wilson_external_i;
    delete ::wilson_number_of_integration_steps;
    delete ::wilson_integration_step_size;
    delete ::wilson_noise_type;
    delete ::wilson_noise_amplitude;

    // Delete helper variables
    delete[] ::wilson_e_values;
    delete[] ::wilson_i_values;
    delete[] ::wilson_coupling_mat;
    delete[] ::wilson_delay_mat;
    delete[] ::wilson_lower_idxs_mat;
    delete[] ::wilson_upper_idxs_mat;
    delete[] ::wilson_output_e;

    // Delete single-value helper variables
    delete temp_long;
    delete node_input;
    delete delay_difference;
    delete index_lower;
    delete index_upper;
    delete input_lower;
    delete input_upper;
    delete input_final;

    // Delete differential variables
    delete[] differential_E;
    delete[] differential_I;
    delete[] differential_E2;
    delete[] differential_I2;
    delete[] activity_E;
    delete[] activity_I;

    // Delete noise array
    delete[] noises_array;

    return ::wilson_electrical_activity;
}

// Define the objective function for the Wilson model
double wilson_objective(unsigned int input_dim, const double *initial_query, double* gradient, void *func_data)
{
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

    // ------------- Check the shape and type of global input variable values
    printf("---- Check global input variable values ----\n");
    check_type((boost::any)::wilson_number_of_oscillators, "int * __ptr64", "wilson_number_of_oscillators");

    // ------------- Declare input variables - arrays
    printf("---- Declare helper variables ----\n");
    PyObject *temp_variable;
    long *temp_long = new long;
    double *node_input = new double;
    double *delay_difference = new double;
    int *index_lower = new int;
    int *index_upper = new int;
    double *input_lower = new double;
    double *input_upper = new double;
    double *input_final = new double;
    double *differential_E = new double[*::wilson_number_of_oscillators];
    double *differential_I = new double[*::wilson_number_of_oscillators];
    double *differential_E2 = new double[*::wilson_number_of_oscillators];
    double *differential_I2 = new double[*::wilson_number_of_oscillators];
    double *activity_E = new double[*::wilson_number_of_oscillators];
    double *activity_I = new double[*::wilson_number_of_oscillators];
    double *noises_array = new double[*::wilson_number_of_oscillators];

    // ------------ Random generation
    std::default_random_engine generator(1);

    // ------------ TEMPORAL INTEGRATION
    printf("---- Temporal integration ----\n");
    for (int step = 1; step <= *::wilson_number_of_integration_steps; step++)
    {   
        if (step % 10000 == 0)
            printf("-- Temporal integration step %d --\n", step);
        // printf("-- Heun's Method - Step 1 --\n");
        // ------------ Heun's Method - Step 1
        for (int node = 0; node < *::wilson_number_of_oscillators; node++)
        {   
            // printf("-- Heun's 1: Node %d --\n", node);
            // ------------ Initializations
            // Initialize input to node as 0
            *node_input = 0;

            // Initialize noise
            if (*::wilson_noise_type == 0)
            {
                noises_array[node] = 0;
            }
            else if(*::wilson_noise_type == 1)
            {
                noises_array[node] = *::wilson_noise_amplitude * (2 *rand_std_uniform(generator) - 1);
            }
            else if(*::wilson_noise_type == 2)
            {
                noises_array[node] = *::wilson_noise_amplitude * rand_std_normal(generator);
            }

            // printf("-- Heun's 1: Node %d - Noise: %f --\n", node, noises_array[node]);

            // ------------ Calculate input to node
            // Consider all other nodes, but only if the lower delay index is lower than the time point
            for (int other_node = 0; other_node < *::wilson_number_of_oscillators; other_node++)
            {   
                // printf("-- Heun's 1: Node %d - Other node %d --\n", node, other_node);
                if (step > ::wilson_lower_idxs_mat[node * *::wilson_number_of_oscillators + other_node])
                {
                    // Retrieve the difference between the 'true' delay and the one corresponding to the upper index
                    *delay_difference = ::wilson_delay_mat[node * *::wilson_number_of_oscillators + other_node];
                    *delay_difference -= (double)::wilson_upper_idxs_mat[node * *::wilson_number_of_oscillators + other_node] * *::wilson_integration_step_size;

                    // Retrieve the time point indices corresponding with the lower and upper delay indices
                    *index_lower = step - 1 - ::wilson_lower_idxs_mat[node * *::wilson_number_of_oscillators + other_node];
                    *index_upper = step - 1 - ::wilson_upper_idxs_mat[node * *::wilson_number_of_oscillators + other_node];

                    // Retrieve the activities corresponding to the lower and upper delay indices
                    *input_lower = ::wilson_output_e[other_node * (*::wilson_number_of_integration_steps + 1) + *index_lower];
                    *input_upper = ::wilson_output_e[other_node * (*::wilson_number_of_integration_steps + 1) + *index_upper];

                    // From the previously retrieved values, estimate the input to oscillator k from oscillator j
                    *input_final = *input_upper;
                    *input_final += (*input_lower - *input_upper) / *::wilson_integration_step_size * *delay_difference;
                    // From this estimation, determine the quantile, final input
                    *input_final *= ::wilson_coupling_mat[node * *::wilson_number_of_oscillators + other_node];
                    // Add this to the total input to oscillator k
                    *node_input += *input_final;
                }
            }

            // printf("-- Heun's 1: Node %d - Differential Equations --\n", node);
            // ------------ Calculate Equations
            // Excitatory population (without noise and time) differentials
            differential_E[node] = *node_input - *::wilson_c_ei * ::wilson_i_values[node] - *::wilson_external_e;
            differential_E[node] = -::wilson_e_values[node] + (1 - *::wilson_r_e * ::wilson_e_values[node]) * wilson_response_function(differential_E[node], *::wilson_alpha_e, *::wilson_theta_e);

            // Inhibitory population (without noise and time) differentials
            differential_I[node] = *::wilson_c_ie * ::wilson_e_values[node];
            differential_I[node] = -::wilson_i_values[node] + (1 - *::wilson_r_i * ::wilson_i_values[node]) * wilson_response_function(differential_I[node], *::wilson_alpha_i, *::wilson_theta_i);
            
            // First estimate of the new activity values
            activity_E[node] = ::wilson_e_values[node] + (*::wilson_integration_step_size * differential_E[node] + sqrt(*::wilson_integration_step_size) * noises_array[node]) / *::wilson_tau_e;
            activity_I[node] = ::wilson_i_values[node] + (*::wilson_integration_step_size * differential_I[node] + sqrt(*::wilson_integration_step_size) * noises_array[node]) / *::wilson_tau_i;

            // printf("-- Heun's 1: Node %d - Update ::wilson_output_e value --\n", node);
            ::wilson_output_e[node * (*::wilson_number_of_integration_steps + 1) + step] = activity_E[node];
        }

        // printf("-- Heun's Method - Step 2 --\n");
        // ------------ Heun's Method - Step 2
        for(int node = 0; node < *::wilson_number_of_oscillators; node++)
        {   
            // printf("-- Heun's 2: Node %d --\n", node);
            // Initialize input to node as 0
            *node_input = 0;

            // ------------ Calculate input to node
            // Consider all other nodes, but only if the lower delay index is lower than the time point
            for (int other_node = 0; other_node < *::wilson_number_of_oscillators; other_node++)
            {   
                // printf("-- Heun's 2: Node %d - Other node %d --\n", node, other_node);
                if (step > ::wilson_lower_idxs_mat[node * *::wilson_number_of_oscillators + other_node])
                {   
                    // printf("Step > lowerIdx");
                    // Retrieve the difference between the 'true' delay and the one corresponding to the upper index
                    *delay_difference = ::wilson_delay_mat[node * *::wilson_number_of_oscillators + other_node];
                    *delay_difference -= (double)::wilson_upper_idxs_mat[node * *::wilson_number_of_oscillators + other_node] * *::wilson_integration_step_size;

                    // Retrieve the time point indices corresponding with the lower and upper delay indices
                    *index_lower = step - ::wilson_lower_idxs_mat[node * *::wilson_number_of_oscillators + other_node];
                    *index_upper = step - ::wilson_upper_idxs_mat[node * *::wilson_number_of_oscillators + other_node];

                    // Retrieve the activities corresponding to the lower and upper delay indices
                    *input_lower = ::wilson_output_e[other_node * (*::wilson_number_of_integration_steps + 1) + *index_lower];
                    *input_upper = ::wilson_output_e[other_node * (*::wilson_number_of_integration_steps + 1) + *index_upper];

                    // From the previously retrieved values, estimate the input to oscillator k from oscillator j
                    *input_final = *input_upper;
                    *input_final += (*input_lower - *input_upper) / *::wilson_integration_step_size * *delay_difference;
                    // From this estimation, determine the quantile, final input
                    *input_final *= ::wilson_coupling_mat[node * *::wilson_number_of_oscillators + other_node];
                    // Add this to the total input to oscillator k
                    *node_input += *input_final;
                }
            }

            // printf("-- Heun's 2: Node %d - Differential Equations --\n", node);
            // ------------ Calculate Equations
            // Excitatory population (without noise and time) differentials
            differential_E2[node] = *node_input - *::wilson_c_ei * activity_I[node] + *::wilson_external_e;
            differential_E2[node] = -activity_E[node] + (1 - *::wilson_r_e * activity_E[node]) * wilson_response_function(differential_E2[node], *::wilson_alpha_e, *::wilson_theta_e);

            // Inhibitory population (without noise and time) differentials
            differential_I2[node] = *::wilson_c_ie * activity_E[node];
            differential_I2[node] = -activity_I[node] + (1 - *::wilson_r_i * activity_I[node]) * wilson_response_function(differential_I2[node], *::wilson_alpha_i, *::wilson_theta_i);

            // Second estimate of the new activity values
            ::wilson_e_values[node] += (*::wilson_integration_step_size / 2 * (differential_E[node] + differential_E2[node]) + sqrt(*::wilson_integration_step_size) * noises_array[node]) / *::wilson_tau_e;
            ::wilson_i_values[node] += (*::wilson_integration_step_size / 2 * (differential_I[node] + differential_I2[node]) + sqrt(*::wilson_integration_step_size) * noises_array[node]) / *::wilson_tau_i;

            // printf("-- Heun's 2: Node %d - Calculate ::wilson_output_e values --\n", node);
            ::wilson_output_e[node * (*::wilson_number_of_integration_steps + 1) + step] = ::wilson_e_values[node];
        }
    }

    // ------------- Convert output variables to Python types
    printf("---- Converting output variables to Python types ----\n");
    for (int i = 0; i < *::wilson_number_of_oscillators; i++)
    {
        for (int step = 0; step <= *::wilson_number_of_integration_steps; step++)
        {
            temp_variable = PyFloat_FromDouble(::wilson_output_e[i * (*::wilson_number_of_integration_steps + 1) + step]);
            PyArray_SETITEM(::wilson_electrical_activity, PyArray_GETPTR2(::wilson_electrical_activity, i, step), temp_variable);
            // Decrease reference for next
            Py_DECREF(temp_variable);
        }
    }

    // ------------- Check that the output has the correct shape and type
    // printf("---- Check output variables ----\n");
    // check_type((boost::any)::wilson_electrical_activity, "struct __object * __ptr64", "wilson_electrical_activity");

    printf("---- Free memory - Electrical ----\n");

    // Delete single-value helper variables
    delete temp_long;
    delete node_input;
    delete delay_difference;
    delete index_lower;
    delete index_upper;
    delete input_lower;
    delete input_upper;
    delete input_final;

    // Delete differential variables
    delete[] differential_E;
    delete[] differential_I;
    delete[] differential_E2;
    delete[] differential_I2;
    delete[] activity_E;
    delete[] activity_I;

    // Delete noise array
    delete[] noises_array;

    // ------------- Got electrical activity
    printf("---- Shape of electrical activity: %d x %d----\n", PyArray_DIMS(::wilson_electrical_activity)[0], 
                                                                PyArray_DIMS(::wilson_electrical_activity)[1]);

    // ------------- Convert the signal to BOLD
    printf("---- Converting electrical activity to BOLD ----\n");
    std::vector<std::vector<double>> bold_signal = electrical_to_bold();

    // Printing shape of bold signal
    printf("---- Shape of BOLD signal: %zd x %zd----\n", bold_signal.size(), bold_signal[0].size());


    return 1;
}

// Function that wraps these functions into methods of a module
static PyMethodDef IntegrationMethods[] = {
    // { // BOLD model
    //     "electrical_to_bold",
    //     electrical_to_bold,
    //     METH_VARARGS,
    //     "Solves the BOLD model equations, and returns BOLD activity"
    // },
    {
        "wilson_model",
        parsing_wilson_inputs,
        METH_VARARGS,
        "Solves the Wilson-Cowan model equations, and returns electrical activity"
    },
    { // Sentinel to properly exit function
        NULL, NULL, 0, NULL
    }
};

// Function that wraps the methods in a module
static struct PyModuleDef simulations = {
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
    return PyModule_Create(&simulations);
}