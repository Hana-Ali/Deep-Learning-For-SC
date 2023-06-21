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
#include <functional>
#include "wilson_config.hpp"
#include "kuramoto_config.hpp"

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

void check_type(boost::any input, const std::string& expected_type, const std::string& input_name);
void set_kura_config(KuramotoConfig* config, KuramotoConfig::PythonObjects python_objects);
void save_data_2D(std::vector<std::vector<double>> data, std::string file_path);
void save_data_3D(std::vector<std::vector<std::vector<double>>> data, std::string file_path);
void set_config(WilsonConfig* config, WilsonConfig::PythonObjects python_objects);

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
        std::uniform_real_distribution<float> distribution(0.0f, 2.0f); //Values between 0 and 2
        std::mt19937 engine; // Mersenne twister MT19937
        auto generator = std::bind(distribution, engine);
        std::generate_n((*config).e_values.begin(), 
                        (*config).e_values.size(), 
                        generator); 
        std::generate_n((*config).i_values.begin(), 
                        (*config).i_values.size(),
                        generator);
		// std::generate((*config).e_values.begin(), (*config).e_values.end(), rand);
		// std::generate((*config).i_values.begin(), (*config).i_values.end(), rand);

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

// Function to convert Python NumPY arrays to C++ vectors - Kuramoto
void set_kura_config(KuramotoConfig* config, KuramotoConfig::PythonObjects python_objects) {
	
	PyObject *temp_variable;
	long temp_long;

    // Allocate space in the matrices
    (*config).phases_phi.resize((*config).number_of_oscillators);
    (*config).freq_omega.resize((*config).number_of_oscillators);
    (*config).structural_connectivity_mat.resize((*config).number_of_oscillators, std::vector<double>((*config).number_of_oscillators));
    (*config).lower_idxs_mat.resize((*config).number_of_oscillators, std::vector<int>((*config).number_of_oscillators));
    (*config).upper_idxs_mat.resize((*config).number_of_oscillators, std::vector<int>((*config).number_of_oscillators));
    (*config).output_phi.resize((*config).number_of_oscillators, std::vector<double>((*config).number_of_integration_steps + 1));

	for (int i = 0; i < (*config).number_of_oscillators; i++)
    {   
        std::uniform_real_distribution<float> distribution(0.0f, 2.0f); //Values between 0 and 2
        std::mt19937 engine; // Mersenne twister MT19937
        auto generator = std::bind(distribution, engine);
        std::generate_n((*config).phases_phi.begin(), 
                        (*config).phases_phi.size(), 
                        generator); 
        std::generate_n((*config).freq_omega.begin(), 
                        (*config).freq_omega.size(),
                        generator);
		// std::generate((*config).phases_phi.begin(), (*config).phases_phi.end(), rand);
		// std::generate((*config).freq_omega.begin(), (*config).freq_omega.end(), rand);

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
        (*config).output_phi[i][0] = (*config).phases_phi[i];
        // Other values in matrix are NaN
        for (int step = 1; step <= (*config).number_of_integration_steps; step++)
        {
            (*config).output_phi[i][step] = nan("");
        }
    }

}