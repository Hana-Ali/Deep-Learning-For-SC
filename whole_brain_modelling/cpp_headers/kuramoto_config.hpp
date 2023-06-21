#pragma once

#include <vector>
#include <string>
#include <random>
#include <map>
#include <numpy/arrayobject.h>

typedef std::map<int, std::string> MapIntToType;

// New

// Define a Kuramoto configuration class for the whole brain model
class KuramotoConfig 
{
    public:
        KuramotoConfig() = default; // Explicitly default

        // Maps for Bayesian Optimization
        // Enum for noise
        enum class Noise {
            NOISE_NONE = 0,
            NOISE_NORMAL = 1,
            NOISE_UNIFORM = 2,
            NOISE_ERROR = -1
        };
        
        // Inputs for the Kuramoto model from Python
        struct PythonObjects {
            PyObject *structural_connec;
            PyObject *lower_idxs;
            PyObject *upper_idxs;
            PyObject *phis_array;
            PyObject *omega_array;
        };

        // Parameters used by WC model
        double coupling_strength{};
        double delay{};
        int number_of_oscillators{};
        // Parameters used for the simulation
        double time_simulated{}; // seconds
        double integration_step_size{}; // seconds
        int number_of_integration_steps{};
        std::vector<std::vector<double>> output_phi{};
        std::vector<std::vector<int>> lower_idxs_mat{}; 
        std::vector<std::vector<int>> upper_idxs_mat{};
        std::vector<double> phases_phi{};
        std::vector<double> freq_omega{};
        Noise noise{};
        double noise_amplitude{};
        // Empirical stuff
        std::vector<std::vector<double>> structural_connectivity_mat{};

    // Method that checks the validity of the passed data
    bool check_validity() const {
        if(phases_phi.size() != number_of_oscillators) return false;
        if(freq_omega.size() != number_of_oscillators) return false;
        if(lower_idxs_mat.size() != number_of_oscillators) return false;
        if(lower_idxs_mat[0].size() != number_of_oscillators) return false;
        if(upper_idxs_mat.size() != number_of_oscillators) return false;
        if(upper_idxs_mat[0].size() != number_of_oscillators) return false;

        if(structural_connectivity_mat.size() != number_of_oscillators) return false;
        if(structural_connectivity_mat[0].size() != number_of_oscillators) return false;
        return true;
    }
};

// Defining the Kuramoto class, which borrows from the config class
class Kuramoto {
    public:
        explicit Kuramoto(KuramotoConfig config);

        // Objective function of the Bayesian Optimization
        std::vector<std::vector<double>> run_simulation();
    
    private:
        KuramotoConfig config;

        // Methods and params needed for the simulation
        std::normal_distribution<double> rand_std_normal{0, 1};
        std::uniform_real_distribution<double> rand_std_uniform{0, 1};

        std::vector<std::vector<double>> delay_mat;
        std::vector<std::vector<double>> coupling_mat;
};