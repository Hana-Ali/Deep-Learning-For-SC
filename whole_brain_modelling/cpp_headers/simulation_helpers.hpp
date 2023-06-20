// -------------------- Header file with some helper for the main simulation
#define PY_SSIZE_T_CLEAN

// Helper functions for the simulation
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <complex>
#include <stdexcept>
#include <boost/any.hpp>
#include <numpy/arrayobject.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_statistics.h>
#include "filtering.hpp"
#include "wilson_config.hpp"

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

void check_type(boost::any input, const std::string& expected_type, const std::string& input_name);
void save_data_2D(std::vector<std::vector<double>> data, std::string file_path);
void save_data_3D(std::vector<std::vector<std::vector<double>>> data, std::string file_path);
void set_config(WilsonConfig config, WilsonConfig::PythonObjects python_objects);
void set_emp_BOLD(WilsonConfig config, PyObject *BOLD_signal);
void filter_BOLD(WilsonConfig config);
void set_emp_FC(WilsonConfig config);
void set_avg_emp_FC(WilsonConfig config);

std::vector<std::vector<double>> process_BOLD(std::vector<std::vector<double>> BOLD_signal, int num_rows, int num_columns, int order, 
							double samplingFrequency, double cutoffFrequencyLow, double cutoffFrequencyHigh);
std::vector<std::vector<double>> determine_FC(std::vector<std::vector<double>> BOLD_signal);
double pearsoncoeff(std::vector<double> X, std::vector<double> Y);
std::vector<std::vector<double>> determine_FC_nogsl(std::vector<std::vector<double>> BOLD_signal);
std::vector<std::vector<double>> get_emp_FC(std::string &file_path);
template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>> & vec);

// Function to check the data type of a variable
void check_type(boost::any input, const std::string& expected_type, const std::string& input_name) {

    // Get the type of the variable
    std::string true_type = input.type().name();
    printf("The type of %s is %s\n", input_name.c_str(), true_type.c_str());

    // Check if the type is the expected one
    if (true_type != expected_type) {
        throw std::invalid_argument("The type of " + input_name + " is " + true_type + " but it should be " + expected_type);
    }
    else {
        printf("The type of %s is correct\n", input_name.c_str());
    }
}

// Function to flatten 2D arrays
template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>> & vec) {   
    std::vector<T> result;
    for (const auto & v : vec)
        result.insert(result.end(), v.begin(), v.end());                                                                                         
    return result;
}

// Saving 2D data in files
void save_data_2D(std::vector<std::vector<double>> data, std::string file_path) {
	// Open the file given in the path
	std::ofstream file(file_path);
	if (!file.is_open()) {
		throw std::invalid_argument("The file " + file_path + " could not be opened");
	}
	else {
		printf("The file %s was opened successfully\n", file_path.c_str());
	}

	// Write the data to the file
	for (int i = 0; i < data.size(); i++) {
		for (int j = 0; j < data[0].size(); j++) {
			file << data[i][j] << ",";
		}
		file << '\n';
	}
}

// Saving 3D data in files
void save_data_3D(std::vector<std::vector<std::vector<double>>> data, std::string file_path) {
	// Open the file given in the path
	std::ofstream file(file_path);
	if (!file.is_open()) {
		throw std::invalid_argument("The file " + file_path + " could not be opened");
	}
	else {
		printf("The file %s was opened successfully\n", file_path.c_str());
	}

	// Write the data to the file
	for (int i = 0; i < data.size(); i++) {
		for (int j = 0; j < data[0].size(); j++) {
			for (int k = 0; k < data[0][0].size(); k++) {
				file << data[i][j][k] << ",";
			}
			file << std::endl;
		}
		file << '\n';
	}
}

// Function to convert Python NumPY arrays to C++ vectors
void set_config(WilsonConfig* config, WilsonConfig::PythonObjects python_objects) {
	
	PyObject *temp_variable;
	long temp_long;

    // Allocate space in the matrices
    (*config).e_values.resize((*config).number_of_oscillators);
    (*config).i_values.resize((*config).number_of_oscillators);
    (*config).structural_connectivity_mat.resize((*config).number_of_oscillators, std::vector<double>((*config).number_of_oscillators));
    (*config).lower_idxs_mat.resize((*config).number_of_oscillators, std::vector<int>((*config).number_of_oscillators));
    (*config).upper_idxs_mat.resize((*config).number_of_oscillators, std::vector<int>((*config).number_of_oscillators));
    (*config).output_e.resize((*config).number_of_oscillators, std::vector<double>((*config).number_of_integration_steps + 1));

	for (int i = 0; i < (*config).number_of_oscillators; i++)
    {   
        srand(time(0));
		std::generate((*config).e_values.begin(), (*config).e_values.end(), rand);
		std::generate((*config).i_values.begin(), (*config).i_values.end(), rand);

        // ------------ Matrices
        for (int j = 0; j < (*config).number_of_oscillators; j++)
        {   
            // Get the structural connectivity matrix
            temp_variable = PyArray_GETITEM(python_objects.structural_connec, PyArray_GETPTR2(python_objects.structural_connec, i, j));
            (*config).structural_connectivity_mat[i][j] = PyFloat_AsDouble(temp_variable);
            // Decrease reference for next
            Py_DECREF(temp_variable);

            // Get the lower_idxs matrix
            temp_variable = PyArray_GETITEM(python_objects.lower_idxs, PyArray_GETPTR2(python_objects.lower_idxs, i, j));
            temp_long = PyLong_AsLong(temp_variable);
            (*config).lower_idxs_mat[i][j] = temp_long;
            // Decrease reference for next
            Py_DECREF(temp_variable);

            // Get the upper_idxs matrix
            temp_variable = PyArray_GETITEM(python_objects.upper_idxs, PyArray_GETPTR2(python_objects.upper_idxs, i, j));
            temp_long = PyLong_AsLong(temp_variable);
            (*config).upper_idxs_mat[i][j] = temp_long;
            // Decrease reference for next
            Py_DECREF(temp_variable);
        }

        // ------------ Initialize output matrix
        (*config).output_e[i][0] = (*config).e_values[i];
        // Other values in matrix are NaN
        for (int step = 1; step <= (*config).number_of_integration_steps; step++)
        {
            (*config).output_e[i][step] = nan("");
        }
    }

}

// Function to get the empirical BOLD signal
void set_emp_BOLD(WilsonConfig* config, PyObject *BOLD_signal) {

	npy_intp emp_BOLD_dims[3] = { PyArray_DIM(BOLD_signal, 0),
                                  PyArray_DIM(BOLD_signal, 1),
                                  PyArray_DIM(BOLD_signal, 2)};
    
    PyObject* time_sample;

    // // Allocate space in the matrices
    // config.emp_BOLD_signals.resize(emp_BOLD_dims[0], std::vector<std::vector<double>>(emp_BOLD_dims[1], std::vector<double>(emp_BOLD_dims[2])));

    // For each subject
    for (int subject = 0; subject < emp_BOLD_dims[0]; ++subject)
    {   
        // Create another vector of vector of doubles, to store each subject's 100 region signals
        std::vector<std::vector<double>> subject_regions;
        
        // For each BOLD signal in the BOLD signals, for each timestep
        for (int region = 0; region < emp_BOLD_dims[1]; ++region)
        {   
            // Create a last vector of doubles, to store the timesamples for each signal
            std::vector<double> region_timesamples;

            for (int timepoint = 0; timepoint < emp_BOLD_dims[2]; ++timepoint)
            {
                // This will store the value in the bold array
                double value;

                // Get the time_sample point
                time_sample = PyArray_GETITEM(BOLD_signal, PyArray_GETPTR3(BOLD_signal, subject, region, timepoint));

                // Check thet each time sample is a float
                if(PyFloat_Check(time_sample))
                    value = PyFloat_AsDouble(time_sample);
                else {
                    printf("Not floats!!!");
                    PyErr_SetString(PyExc_TypeError, "Empirical BOLD is not in the correct format");
					return;
                }

                region_timesamples.push_back(value);
                // Decrement the pointer reference
                Py_DECREF(time_sample);
            }
            subject_regions.push_back(region_timesamples);
        }
        (*config).emp_BOLD_signals.push_back(subject_regions);
    }

	// Saving it just for a sanity check
    printf("Size of BOLD signal is %d x %d x %d\n", (*config).emp_BOLD_signals.size(), 
        (*config).emp_BOLD_signals[0].size(), (*config).emp_BOLD_signals[0][0].size());
    printf("Saving unpacked empirical BOLD signal...\n");
    save_data_3D((*config).emp_BOLD_signals, "temp_arrays/unpacked_emp_BOLD.csv");

}

// Function to filter the BOLD signal
void filter_BOLD(WilsonConfig* config) {

    printf("Size of BOLD signal is %d x %d x %d\n", (*config).emp_BOLD_signals.size(), 
        (*config).emp_BOLD_signals[0].size(), (*config).emp_BOLD_signals[0][0].size()
    );

	// For each subject
    for (int subject = 0; subject < (*config).emp_BOLD_signals.size(); subject++)
    {
        // Add the subject to the vector of all subjects
        (*config).filtered_BOLD_signals.emplace_back(process_BOLD((*config).emp_BOLD_signals[subject],
                                                    		   (*config).emp_BOLD_signals[subject].size(),
                                                    		   (*config).emp_BOLD_signals[subject][0].size(),
                                                    		   (*config).order,
															   (*config).cutoffLow,
															   (*config).cutoffHigh,
															   (*config).sampling_rate));
    }

	// Saving it just for a sanity check
	printf("Saving filtered empirical BOLD signal...\n");
	save_data_3D((*config).filtered_BOLD_signals, "temp_arrays/filtered_emp_BOLD.csv");

}

// Function to find the empirical FC
void set_emp_FC(WilsonConfig* config) {
	// For each subject
    for (int subject = 0; subject < (*config).filtered_BOLD_signals.size(); subject++)
    {
        // Add the subject to the vector of all subjects
        printf("subject: %d\n\r", subject);
        (*config).all_emp_FC.emplace_back(determine_FC((*config).filtered_BOLD_signals[subject]));
    }

	// Saving it just for a sanity check
	printf("Saving empirical FC...\n");
	save_data_3D((*config).all_emp_FC, "temp_arrays/emp_FC.csv");

}

// Function to find the averaged empirical FC
void set_avg_emp_FC(WilsonConfig* config) {
	// For each region
    for (int i = 0; i < (*config).all_emp_FC[1].size(); i++)
    {
        // Create a vector of doubles for each *other* region
        std::vector<double> region_avg;

        // For each other region
        for (int j = 0; j < (*config).all_emp_FC[1].size(); j++)
        {
			// Create a vector of doubles for each subject
			std::vector<double> subject_values;

			// For each subject
			for (int k = 0; k < (*config).all_emp_FC[0].size(); k++)
			{
				subject_values.push_back((*config).all_emp_FC[i][j][k]);
			}
			// Get the mean of the subject values
			double mean = gsl_stats_mean(subject_values.data(), 1, subject_values.size());
			region_avg.push_back(mean);
		}
        (*config).emp_FC.push_back(region_avg);
    }

	// Saving it just for a sanity check
	printf("Saving averaged empirical FC...\n");
	save_data_2D((*config).emp_FC, "temp_arrays/avg_emp_FC.csv");
}

// Function to process the BOLD data - same as in Python helper_funcs.py file
std::vector<std::vector<double>> process_BOLD(std::vector<std::vector<double>> BOLD_signal, int num_rows, int num_columns, int order, 
                                                double samplingFrequency, double cutoffFrequencyLow, double cutoffFrequencyHigh)
{   
    // Create the filtered signal object
    std::vector<std::vector<double>> filteredSignal(num_rows, std::vector<double>(num_columns));

    // Create filter objects
    // These values are as a ratio of f/fs, where fs is sampling rate, and f is cutoff frequency
    double FrequencyBands[2] = {
        cutoffFrequencyLow/(samplingFrequency/2.0),
        cutoffFrequencyHigh/(samplingFrequency/2.0)
    };
    //Create the variables for the numerator and denominator coefficients
    std::vector<double> DenC;
    std::vector<double> NumC;

    // Find the mean across the columns
    printf("BOLD PROCESSING: Finding the mean across the columns...\n");
    std::vector<double> mean(num_columns);
    // Calculate the mean across the columns
    for (int row = 0; row < num_rows; row++) {
        double colSum = 0.0;
        for (int col = 0; col < num_columns; col++) {
            colSum += BOLD_signal[row][col];
            // printf("column %d, colSum is: %lf\n", col, colSum);
        }
        mean[row] = colSum / num_rows;
    }

    // Remove the mean from each column
    printf("BOLD PROCESSING: Removing the mean from each column...\n");
    for (int row = 0; row < num_rows; row++) {
        for (int col = 0; col < num_columns; col++) {
            BOLD_signal[row][col] -= mean[col];
        }
    }

    // Finding the coefficients of the filter
    printf("BOLD PROCESSING: Finding the coefficients of the filter...\n");
    DenC = ComputeDenCoeffs(order, FrequencyBands[0], FrequencyBands[1]);
    for(int k = 0; k < 2 * order + 1; k++)
        printf("DenC is: %lf\n", DenC[k]);

    NumC = ComputeNumCoeffs(order,FrequencyBands[0],FrequencyBands[1],DenC);
    for(int k = 0; k < 2 * order + 1; k++)
        printf("NumC is: %lf\n", NumC[k]);

	// Printing sizes of both vectors
	printf("BOLD_signal size is %d x %d\n", BOLD_signal.size(), BOLD_signal[0].size());
	printf("filteredSignal size is %d x %d\n", filteredSignal.size(), filteredSignal[0].size());

    // Applying the filter forwards and backwards
    printf("Applying the filter forwards and backwards\n");
    for (int row = 0; row < num_rows; row++)
        filteredSignal[row] = filter(NumC, DenC, filteredSignal[0].size(), 
                                        BOLD_signal[row], filteredSignal[row]);
    
    for (int row = num_rows - 1; row >= 0; row--)
        filteredSignal[row] = filter(NumC, DenC, filteredSignal[0].size(), 
                                        BOLD_signal[row], filteredSignal[row]);
    
    save_data_2D(filteredSignal, "temp_arrays/filtered_beforez.csv");

	// Z-scoring the final filtered signal
	printf("Z-scoring the final filtered signal\n");
	for (int row = 0; row < num_rows; row++) {
		double mean = 0.0;
		double std = 0.0;
		for (int col = 0; col < num_columns; col++) {
			mean += filteredSignal[row][col];
		}
		mean /= num_columns;
		for (int col = 0; col < num_columns; col++) {
			std += pow(filteredSignal[row][col] - mean, 2);
		}
		std /= num_columns;
		std = sqrt(std);
		for (int col = 0; col < num_columns; col++) {
			filteredSignal[row][col] = (filteredSignal[row][col] - mean) / std;
		}
	}

    return filteredSignal;
}

// Function to find the functional connectivity matrix from the BOLD signal
std::vector<std::vector<double>> determine_FC(std::vector<std::vector<double>> BOLD_signal)
{
	// Create a correlation matrix of size BOLD_signal.size() x BOLD_signal.size()
	std::vector<std::vector<double>> correlation_matrix(BOLD_signal.size(), std::vector<double>(BOLD_signal.size()));

	// For every row (brain region) of the BOLD_signal
	for (int i = 0; i < BOLD_signal.size(); i++)
	{
		// For every other row (brain region) of the BOLD_signal
		for (int j = 0; j < BOLD_signal.size(); j++)
		{
			// // Convert the signal vectors to the appropriate format for gsl
			// gsl_vector_const_view gsl_BOLD_i = gsl_vector_const_view_array(BOLD_signal[i].data(), BOLD_signal[i].size());
			// gsl_vector_const_view gsl_BOLD_j = gsl_vector_const_view_array(BOLD_signal[j].data(), BOLD_signal[j].size());
			double correlation = gsl_stats_correlation(BOLD_signal[i].data(), 1, BOLD_signal[j].data(), 1, BOLD_signal[i].size());
			correlation_matrix[i][j] = correlation;
		}
	}

	return correlation_matrix;
}

double pearsoncoeff(std::vector<double> X, std::vector<double> Y)
{
	double sum_X = 0, sum_Y = 0, sum_XY = 0;
    double squareSum_X = 0, squareSum_Y = 0;
 
    for (int i = 0; i < X.size(); i++)
    {
        // sum of elements of array X.
        sum_X += X[i];
 
        // sum of elements of array Y.
        sum_Y += Y[i];
 
        // sum of X[i] * Y[i].
        sum_XY = sum_XY + X[i] * Y[i];
 
        // sum of square of array elements.
        squareSum_X = squareSum_X + X[i] * X[i];
        squareSum_Y = squareSum_Y + Y[i] * Y[i];
    }
 
    // use formula for calculating correlation coefficient.
    float corr = (float)(X.size() * sum_XY - sum_X * sum_Y)
                  / sqrt((X.size() * squareSum_X - sum_X * sum_X)
                      * (X.size() * squareSum_Y - sum_Y * sum_Y));
 
    return corr;
}

// Function to find the functional connectivity matrix from the BOLD signal
std::vector<std::vector<double>> determine_FC_nogsl(std::vector<std::vector<double>> BOLD_signal)
{
	// Create a correlation matrix of size BOLD_signal.size() x BOLD_signal.size()
	std::vector<std::vector<double>> correlation_matrix(BOLD_signal.size(), std::vector<double>(BOLD_signal.size()));

	// For every row (brain region) of the BOLD_signal
	for (int i = 0; i < BOLD_signal.size(); i++)
	{
		// For every other row (brain region) of the BOLD_signal
		for (int j = 0; j < BOLD_signal.size(); j++)
		{
			double correlation = pearsoncoeff(BOLD_signal[i], BOLD_signal[j]);
			correlation_matrix[i][j] = correlation;
			printf("Correlation between %d and %d is %lf\n", i, j, correlation);
		}
	}

	return correlation_matrix;
}

// Function to find get the functional connectivity from input files
std::vector<std::vector<double>> get_emp_FC(std::string &file_path)
{
	// Open the file given in the path
	std::ifstream file(file_path);
	if (!file.is_open()) {
		throw std::invalid_argument("The file " + file_path + " could not be opened");
	}
	else {
		printf("The file %s was opened successfully\n", file_path.c_str());
	}

	// Create a vector to store the data
	std::vector<std::vector<double>> data;

	// Read the data from the file
	std::string line;
	while (std::getline(file, line)) {
		// Create a vector to store the data in the line
		std::vector<double> line_data;

		// Create a stringstream of the line
		std::stringstream ss(line);

		// Read the data from the stringstream
		double value;
		while (ss >> value) {
			line_data.push_back(value);
		}

		// Add the line data to the data vector
		data.push_back(line_data);
	}
}