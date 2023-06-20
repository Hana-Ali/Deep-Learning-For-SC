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
        // Maps for Bayesian Optimization
        // Enum for noise
        enum class Noise {
            NOISE_NONE = 0,
            NOISE_NORMAL = 1,
            NOISE_UNIFORM = 2,
            NOISE_ERROR = -1
        };
        std::map<int, std::string> CriteriaName = {
            {0, "cEI"},
            {1, "cLCB"},
            {2, "cMI"},
            {3, "cPOI"},
            {4, "cExpReturn"},
            {5, "cAopt"},
            {6, "cHedge(cSum(cEI,cDistance),cLCB,cPOI,cOptimisticSampling)"},
            {-1, "Error"}
        };
        std::map<int, std::string> SurrogateName = {
            {0, "sGaussianProcess"},
            {1, "sGaussianProcessML"},
            {2, "sGaussianProcessNormal"},
            {3, "sStudentTProcessJef"},
            {4, "sStudentTProcessNIG"},
            {-1, "Error"}
        };
        std::map<int, std::string> Kernel = {
            {0, "kConst"},
            {1, "kLinear"},
            {2, "kMaternISO1"},
            {3, "kMaternISO3"},
            {4, "kPoly4"},
            {5, "kSEARD"},
            {6, "kRQISO"},
            {-1, "Error"}
        };
        
        // Inputs for the Kuramoto model from Python
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
        // Parameters for the filder
        int order{};
        double cutoffLow{};
        double cutoffHigh{};
        double sampling_rate{};
        // Empirical stuff
        std::vector<std::vector<double>> structural_connectivity_mat{};
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