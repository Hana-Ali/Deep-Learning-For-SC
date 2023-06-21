#pragma once

#include <vector>
#include <string>
#include <random>
#include <map>

typedef std::map<int, std::string> MapIntToType;

// New

// Define a Wilson configuration class for the whole brain model
class WilsonConfig 
{
    public:
        WilsonConfig() = default; // Explicitly default

        // Enum for noise
        enum class Noise {
            NOISE_NONE = 0,
            NOISE_NORMAL = 1,
            NOISE_UNIFORM = 2,
            NOISE_ERROR = -1
        };
        
        // Inputs for the Wilson model from Python
        struct PythonObjects {
            PyObject *structural_connec;
            PyObject *lower_idxs;
            PyObject *upper_idxs;
            PyObject *initial_cond_e;
            PyObject *initial_cond_i;
        };

        // Parameters used by WC model
        double coupling_strength{};
        double delay{};
        int number_of_oscillators{};
        double c_ee{};
        double c_ei{};
        double c_ie{};
        double c_ii{};
        double tau_e{};
        double tau_i{};
        double r_e{};
        double r_i{};
        double k_e{};
        double k_i{};
        double alpha_e{};
        double alpha_i{};
        double theta_e{};
        double theta_i{};
        double external_e{};
        double external_i{};
        // Parameters used for the simulation
        double time_simulated{}; // seconds
        double integration_step_size{}; // seconds
        int number_of_integration_steps{};
        std::vector<std::vector<double>> output_e{}; // TODO: Can be the same as electrical_activity
        std::vector<std::vector<int>> lower_idxs_mat{}; 
        std::vector<std::vector<int>> upper_idxs_mat{};
        std::vector<double> e_values{};
        std::vector<double> i_values{};
        Noise noise{};
        double noise_amplitude{};
        // Empirical stuff
        std::vector<std::vector<double>> structural_connectivity_mat{};

    // Method that checks the validity of the passed data
    bool check_validity() const {
        if(e_values.size() != number_of_oscillators) return false;
        if(i_values.size() != number_of_oscillators) return false;
        if(lower_idxs_mat.size() != number_of_oscillators) return false;
        if(lower_idxs_mat[0].size() != number_of_oscillators) return false;
        if(upper_idxs_mat.size() != number_of_oscillators) return false;
        if(upper_idxs_mat[0].size() != number_of_oscillators) return false;

        if(structural_connectivity_mat.size() != number_of_oscillators) return false;
        if(structural_connectivity_mat[0].size() != number_of_oscillators) return false;
        return true;
    }
};

// Defining the Wilson class, which borrows from the config class
class Wilson {
    public:
        explicit Wilson(WilsonConfig config);

        // Method that converts electrical to BOLD signals
        std::vector<std::vector<double>> electrical_to_bold(std::vector<std::vector<double>>& electrical_signals,
                                                            int number_of_oscillators,
                                                            int number_of_integration_steps,
                                                            float integration_step_size);

        // Objective function of the Bayesian Optimization
        std::vector<std::vector<double>> run_simulation();
    
    private:
        WilsonConfig config;

        // Methods and params needed for the simulation
        std::normal_distribution<double> rand_std_normal{0, 1};
        std::uniform_real_distribution<double> rand_std_uniform{0, 1};

        std::vector<std::vector<double>> delay_mat;
        std::vector<std::vector<double>> coupling_mat;
};