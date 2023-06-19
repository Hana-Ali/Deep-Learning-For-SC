#pragma once

#include <vector>
#include <string>
#include <random>

// New

// Define a Wilson configuration class for the whole brain model
class WilsonConfig 
{
    public:
        WilsonConfig() = default; // Explicitly default

        // Enums for Bayesian Optimization
        enum class ScoreType {
            SC_MTL = 0,
            SC_ML = 1, 
            SC_MAP = 2,
            SC_LOOCV = 3,
            SC_ERROR = -1
        };
        enum class LearningType {
            L_FIXED = 0,
            L_EMPIRICAL = 1,
            L_DISCRETE = 2,
            L_MCMC = 3,
            L_ERROR = -1
        };
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
            PyObject *BOLD_signals;
        };

        // Create a struct to hold both the minimizer and the minimum value
        struct BO_output {
            double *minimizer;
            double minimizer_value;
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
        // Parameters for the filder
        int order{};
        double cutoffLow{};
        double cutoffHigh{};
        double sampling_rate{};
        // Empirical stuff
        std::vector<std::vector<double>> structural_connectivity_mat{};
        std::vector<std::vector<std::vector<double>>> emp_BOLD_signals{};
        std::vector<std::vector<std::vector<double>>> filtered_BOLD_signals{};
        std::vector<std::vector<std::vector<double>>> all_emp_FC{};
        std::vector<std::vector<double>> emp_FC{};
        // Simulation BOLD stuff
        int num_BOLD_subjects{};
        int num_BOLD_regions{};
        int num_BOLD_timepoints{};
        // Bayesian Optimization stuff
        int BO_n_iter{}; ///< BO iterations
        int BO_n_inner_iter{}; ///< BO inner iterations
        int BO_iter_relearn{}; ///< BO iterations before relearning
        int BO_init_samples{}; ///< BO initial samples
        int BO_init_method{}; ///< BO initial method
        int BO_verbose_level{}; ///< BO verbose level
        std::string BO_log_file{}; ///< BO log file
        std::string BO_surrogate{}; ///< BO surrogate
        ScoreType BO_sc_type{}; ///< BO score type
        LearningType BO_l_type{}; ///< BO learning type
        bool BO_l_all{}; ///< BO learn all
        double BO_epsilon{}; ///< BO epsilon
        int BO_force_jump{}; ///< BO force jump
        std::string BO_crit_name{}; ///< BO criterion name

    // Method that checks the validity of the passed data
    bool check_validity() const {
        if(e_values.size() != number_of_oscillators) return false;
        if(i_values.size() != number_of_oscillators) return false;
        if(lower_idxs_mat.size() != number_of_oscillators) return false;
        if(lower_idxs_mat[0].size() != number_of_oscillators) return false;
        if(upper_idxs_mat.size() != number_of_oscillators) return false;
        if(upper_idxs_mat[0].size() != number_of_oscillators) return false;

        if(emp_BOLD_signals.size() != num_BOLD_subjects) return false;
        if(emp_BOLD_signals[0].size() != num_BOLD_regions) return false;
        if(emp_BOLD_signals[0][0].size() != num_BOLD_timepoints) return false;

        if(structural_connectivity_mat.size() != number_of_oscillators) return false;
        if(structural_connectivity_mat[0].size() != number_of_oscillators) return false;
        return true;
    }
};

// Defining the Wilson class, which borrows from the config class
class Wilson {
    public:
        explicit Wilson(WilsonConfig config); // Explicitly default

        // Method that runs the simulation
        WilsonConfig::BO_output run_simulation();

        // Method that converts electrical to BOLD signals
        std::vector<std::vector<double>> electrical_to_bold(std::vector<std::vector<double>>& electrical_signals,
                                                            int number_of_oscillators,
                                                            int number_of_integration_steps,
                                                            float integration_step_size);

        // Objective function of the Bayesian Optimization
        static double wilson_objective(unsigned int input_dim,
                                       const double *initial_query = nullptr,
                                       double* gradient = nullptr,
                                       void *func_data = nullptr);
    
    private:
        WilsonConfig config;

        // Methods and params needed for the simulation
        std::normal_distribution<double> rand_std_normal{0, 1};
        std::uniform_real_distribution<double> rand_std_uniform{0, 1};

        std::vector<std::vector<double>> delay_mat;
        std::vector<std::vector<double>> coupling_mat;
};