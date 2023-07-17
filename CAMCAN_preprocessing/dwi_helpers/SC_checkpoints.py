import os

import sys
sys.path.append("..")
from py_helpers.general_helpers import *

from .SC_paths import *

# ------------------------------------------------- CHECKING MISSING FILES AND CHECKPOINTS ------------------------------------------------- #

# Function to check which files are missing, and where we should start the processing
def check_all_mrtrix_missing_files(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH, ATLAS):
    # --------------------- GENERAL VARIABLES NEEDED
    # Get the main MRTRIX folder paths
    (SUBJECT_FOLDER_NAME, RECON_FOLDER_NAME, RESPONSE_FOLDER_NAME, FOD_FOLDER_NAME, FOD_NORM_FOLDER_NAME,
        T1_REG_FOLDER_NAME, ATLAS_REG_FOLDER_NAME, PROB_TRACKING_FOLDER_NAME, GLOBAL_TRACKING_FOLDER_NAME,
            CONNECTIVITY_FOLDER_NAME) = main_mrtrix_folder_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)
    
    # --------------------- MRTRIX RESPONSE CHECK
    MRTRIX_RESPONSE = check_missing_mrtrix_response(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH, RESPONSE_FOLDER_NAME)

    # --------------------- MRTRIX FOD CHECK
    MRTRIX_FOD = check_missing_mrtrix_fod(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH, FOD_NORM_FOLDER_NAME)

    # --------------------- MRTRIX FOD NORM CHECK
    MRTRIX_FOD_NORM = check_missing_mrtrix_fod_norm(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH, FOD_NORM_FOLDER_NAME)
    
    # --------------------- MRTRIX REGISTRATION CHECK
    MRTRIX_REGISTRATION = check_missing_mrtrix_registration(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH, ATLAS, T1_REG_FOLDER_NAME, ATLAS_REG_FOLDER_NAME)
    
    # --------------------- MRTRIX PROBTRACK CHECK
    MRTRIX_PROBTRACK = check_missing_mrtrix_probtrack(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH, PROB_TRACKING_FOLDER_NAME)
    
    # --------------------- MRTRIX GLOBAL TRACKING CHECK
    MRTRIX_GLOBAL_TRACKING = check_missing_mrtrix_global_tracking(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH, GLOBAL_TRACKING_FOLDER_NAME)

    # --------------------- MRTRIX CONNECTOME CHECK
    MRTRIX_CONNECTOME = check_missing_mrtrix_connectome(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH, CONNECTIVITY_FOLDER_NAME)

    # Return the variables
    return (MRTRIX_RESPONSE, MRTRIX_FOD, MRTRIX_FOD_NORM, MRTRIX_REGISTRATION, MRTRIX_PROBTRACK, MRTRIX_GLOBAL_TRACKING, MRTRIX_CONNECTOME)
    
# Function to check missing FSL
def check_missing_fsl(MAIN_FSL_PATH, dwi_filename):
    # Define variable that stores whether or not we should do FSL cleaning
    FSL_CLEANING = True
    # Extract what we need here from the needed file paths
    FSL_SUBJECT_FOLDER = os.path.join(MAIN_FSL_PATH, dwi_filename)
    # Glob all the nii.gz files in the FSL folder
    FSL_FILES = glob_files(FSL_SUBJECT_FOLDER, "nii.gz")
    # If the length of the files is 3, then we have all the files we need
    if len(FSL_FILES) >= 3:
        print("--- FSL files found. Skipping FSL processing.")
        FSL_CLEANING = False
    
    # If we don't have all the files we need, then we clean the folder and start from scratch
    if FSL_CLEANING:
        print("--- FSL files not found. Cleaning FSL folder.")
        check_output_folders(FSL_SUBJECT_FOLDER, "FSL subject folder", wipe=True)
    
    # Return the variable
    return FSL_CLEANING

# Function to check missing MRtrix cleaning
def check_missing_mrtrix_cleaning(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH):
    MRTRIX_CLEANING = True
    # Get the MRtrix cleaning file paths
    (CLEANING_FOLDER_NAME, INPUT_MIF_PATH, DWI_DENOISE_PATH, DWI_NOISE_PATH, DWI_EDDY_PATH, 
        DWI_BIAS_PATH, DWI_CONVERT_PATH) = get_mrtrix_clean_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)
    # Glob all the nii.gz, bval and bvec files in the MRtrix cleaning folder
    CLEAN_DWI_FILES = glob_files(CLEANING_FOLDER_NAME, "nii.gz")
    CLEAN_BVAL_FILE = glob_files(CLEANING_FOLDER_NAME, "bval")
    CLEAN_BVEC_FILE = glob_files(CLEANING_FOLDER_NAME, "bvec")
    print("CLEAN_DWI_FILES: ", CLEAN_DWI_FILES)
    print("CLEAN_BVAL_FILE: ", CLEAN_BVAL_FILE)
    print("CLEAN_BVEC_FILE: ", CLEAN_BVEC_FILE)
    # Check that we have all the files we need
    if (any(DWI_CONVERT_PATH in clean_file for clean_file in CLEAN_DWI_FILES)
         and len(CLEAN_BVAL_FILE) > 0 and len(CLEAN_BVEC_FILE) > 0):
        print("--- MRtrix cleaning files found. Skipping MRtrix cleaning.")
        MRTRIX_CLEANING = False

    # If we don't have all the files we need, then we clean the folder and start from scratch
    if MRTRIX_CLEANING:
        print("--- MRtrix cleaning files not found. Cleaning MRtrix cleaning folder.")
        check_output_folders(CLEANING_FOLDER_NAME, "MRtrix cleaning folder", wipe=True)

    # Return the variable
    return MRTRIX_CLEANING

# Function to check missing DSI Studio processing
def check_missing_dsi_studio(NEEDED_FILE_PATHS, MAIN_STUDIO_PATH, dwi_filename):
    # Define variable that stores whether or not we should do DSI Studio processing
    STUDIO_PROCESSING = True
    # Get the DSI_STUDIO file paths
    (STUDIO_SRC_PATH, STUDIO_DTI_PATH, STUDIO_QSDR_PATH, SRC_LOG_PATH, DTI_LOG_PATH,
        DTI_EXP_LOG_PATH, QSDR_LOG_PATH, QSDR_EXP_LOG_PATH, TRACT_LOG_PATH) = get_dsi_studio_paths(NEEDED_FILE_PATHS, MAIN_STUDIO_PATH)
    # Creating folder for each subject in DSI_STUDIO folder. CAN WIPE as this is the only function that uses it
    SUBJECT_FOLDER_NAME = os.path.join(MAIN_STUDIO_PATH, dwi_filename)
    # Get the connectivity matrices in the DSI_STUDIO folder
    DSI_CONNECTIVITY = glob_files(SUBJECT_FOLDER_NAME, "mat")
    # Check that we have all the files we need
    if len(DSI_CONNECTIVITY) >= 1:
        print("--- DSI Studio files found. Skipping DSI Studio processing.")
        STUDIO_PROCESSING = False
    
    # If we don't have all the files we need, then we clean the folder and start from scratch
    if STUDIO_PROCESSING:
        print("--- DSI Studio files not found. Cleaning DSI Studio folder.")
        check_output_folders(SUBJECT_FOLDER_NAME, "DSI Studio subject folder", wipe=True)

    # Return the variable
    return STUDIO_PROCESSING

# Function to check missing MRtrix response function processing
def check_missing_mrtrix_response(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH, RESPONSE_FOLDER_NAME):
    # Define variable that stores whether or not we should do MRtrix response function processing
    MRTRIX_RESPONSE = True
    # Get the MRtrix response function paths
    (RESPONSE_WM_PATH, RESPONSE_GM_PATH, RESPONSE_CSF_PATH, RESPONSE_VOXEL_PATH,
        WM_FOD_PATH, GM_FOD_PATH, CSF_FOD_PATH, VF_FOD_PATH, WM_FOD_NORM_PATH, GM_FOD_NORM_PATH, 
            CSF_FOD_NORM_PATH) = get_mrtrix_fod_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)
    # Grab all the mif files
    MRTRIX_RESPONSE_MIF_FILES = glob_files(RESPONSE_FOLDER_NAME, "mif")
    print("MRTRIX_RESPONSE_MIF_FILES: ", MRTRIX_RESPONSE_MIF_FILES)
    # Check that we have all the files we need
    if (any(RESPONSE_VOXEL_PATH in mif_file for mif_file in MRTRIX_RESPONSE_MIF_FILES)):
        print("--- MRtrix response files found. Skipping MRtrix response estimation.")
        MRTRIX_RESPONSE = False

    # If we don't have all the files we need, then we clean the folder and start from scratch
    if MRTRIX_RESPONSE:
        print("--- MRtrix response files not found. Cleaning MRtrix response folder.")
        check_output_folders(RESPONSE_FOLDER_NAME, "MRtrix response folder", wipe=True)

    # Return the variable
    return MRTRIX_RESPONSE

# Function to check missing MRtrix FOD processing
def check_missing_mrtrix_fod(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH, FOD_FOLDER_NAME):
    # Define variable that stores whether or not we should do MRtrix FOD processing
    MRTRIX_FOD = True
    # Get the MRtrix FOD paths
    (RESPONSE_WM_PATH, RESPONSE_GM_PATH, RESPONSE_CSF_PATH, RESPONSE_VOXEL_PATH,
        WM_FOD_PATH, GM_FOD_PATH, CSF_FOD_PATH, VF_FOD_PATH, WM_FOD_NORM_PATH, GM_FOD_NORM_PATH, 
            CSF_FOD_NORM_PATH) = get_mrtrix_fod_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)
    # Grab all the mif files
    MRTRIX_FOD_MIF_FILES = glob_files(FOD_FOLDER_NAME, "mif")
    print("MRTRIX_FOD_MIF_FILES: ", MRTRIX_FOD_MIF_FILES)
    # Check that we have all the files we need
    if (any(WM_FOD_PATH in fod_mif for fod_mif in MRTRIX_FOD_MIF_FILES)
            and any(GM_FOD_PATH in fod_mif for fod_mif in MRTRIX_FOD_MIF_FILES)
            and any(CSF_FOD_PATH in fod_mif for fod_mif in MRTRIX_FOD_MIF_FILES)
            and any(VF_FOD_PATH in fod_mif for fod_mif in MRTRIX_FOD_MIF_FILES)):
        print("--- MRtrix FOD files found. Skipping MRtrix FOD processing.")
        MRTRIX_FOD = False

    # If we don't have all the files we need, then we clean the folder and start from scratch
    if MRTRIX_FOD:
        print("--- MRtrix FOD files not found. Cleaning MRtrix FOD folder.")
        check_output_folders(FOD_FOLDER_NAME, "MRtrix FOD folder", wipe=True)

    # Return the variable
    return MRTRIX_FOD


# Function to check missing MRtrix FOD processing
def check_missing_mrtrix_fod_norm(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH, FOD_NORM_FOLDER_NAME):
    # Define variable that stores whether or not we should do MRtrix FOD norm processing
    MRTRIX_FOD_NORM = True
    # Get the MRtrix FOD paths
    (RESPONSE_WM_PATH, RESPONSE_GM_PATH, RESPONSE_CSF_PATH, RESPONSE_VOXEL_PATH,
        WM_FOD_PATH, GM_FOD_PATH, CSF_FOD_PATH, VF_FOD_PATH, WM_FOD_NORM_PATH, GM_FOD_NORM_PATH, 
            CSF_FOD_NORM_PATH) = get_mrtrix_fod_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)
    # Grab all the mif files
    MRTRIX_FOD_NORM_MIF_FILES = glob_files(FOD_NORM_FOLDER_NAME, "mif")
    print("MRTRIX_FOD_NORM_MIF_FILES: ", MRTRIX_FOD_NORM_MIF_FILES)
    # Check that we have all the files we need
    if (any(WM_FOD_NORM_PATH in fod_mif for fod_mif in MRTRIX_FOD_NORM_MIF_FILES)
            and any(GM_FOD_NORM_PATH in fod_mif for fod_mif in MRTRIX_FOD_NORM_MIF_FILES)
            and any(CSF_FOD_NORM_PATH in fod_mif for fod_mif in MRTRIX_FOD_NORM_MIF_FILES)):
        print("--- MRtrix FOD NORM files found. Skipping MRtrix FOD processing.")
        MRTRIX_FOD_NORM = False

    # If we don't have all the files we need, then we clean the folder and start from scratch
    if MRTRIX_FOD_NORM:
        print("--- MRtrix FOD NORM files not found. Cleaning MRtrix FOD folder.")
        check_output_folders(FOD_NORM_FOLDER_NAME, "MRtrix FOD folder", wipe=True)
    
    # Return the variable
    return MRTRIX_FOD_NORM

# Function to check missing MRtrix registration processing
def check_missing_mrtrix_registration(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH, ATLAS, T1_REG_FOLDER_NAME, ATLAS_REG_FOLDER_NAME):
    # Define variable that stores whether or not we should do MRtrix registration processing
    MRTRIX_REGISTRATION = True
    # Get the MRtrix registration paths
    (T1_MIF_PATH, FIVETT_NOREG_PATH, DWI_B0_PATH, DWI_B0_NII, FIVETT_GEN_NII, T1_DWI_MAP_MAT,
        T1_DWI_CONVERT_INV, FIVETT_REG_PATH, ATLAS_DWI_MAP_MAT, ATLAS_DWI_CONVERT_INV, ATLAS_REG_PATH, 
            ATLAS_MIF_PATH) = get_mrtrix_registration_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH, ATLAS)
    # Grab all the mif files in the T1 and atlas registration folders
    MRTRIX_REG_MIF_FILES = glob_files(T1_REG_FOLDER_NAME, "mif")
    MRTRIX_ATLAS_REG_MIF_FILES = glob_files(ATLAS_REG_FOLDER_NAME, "mif")
    print("MRTRIX_REG_MIF_FILES: ", MRTRIX_REG_MIF_FILES)
    print("MRTRIX_ATLAS_REG_MIF_FILES: ", MRTRIX_ATLAS_REG_MIF_FILES)
    # Check that we have all the files we need
    if (any(FIVETT_REG_PATH in reg_mif for reg_mif in MRTRIX_REG_MIF_FILES) and 
        any(ATLAS_REG_PATH in reg_mif for reg_mif in MRTRIX_ATLAS_REG_MIF_FILES)):
        print("--- MRtrix registration files found. Skipping MRtrix registration processing.")
        MRTRIX_REGISTRATION = False
    
    # If we don't have all the files we need, then we clean the folder and start from scratch
    if MRTRIX_REGISTRATION:
        print("--- MRtrix registration files not found. Cleaning MRtrix registration folder.")
        check_output_folders(T1_REG_FOLDER_NAME, "MRtrix registration folder", wipe=True)
        check_output_folders(ATLAS_REG_FOLDER_NAME, "MRtrix atlas registration folder", wipe=True)

    # Return the variable
    return MRTRIX_REGISTRATION

# Function to check missing MRtrix probabilistic tracking processing
def check_missing_mrtrix_probtrack(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH, PROB_TRACKING_FOLDER_NAME):
    # Define variable that stores whether or not we should do MRtrix probabilistic tracking processing
    MRTRIX_PROBTRACK = True
    # Get the MRtrix probabilistic tracking paths
    (GM_WM_SEED_PATH, TRACT_TCK_PATH) = get_mrtrix_probtrack_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)
    # Grab all the tck files
    MRTRIX_PROBTRACK_TCK_FILES = glob_files(PROB_TRACKING_FOLDER_NAME, "tck")
    print("MRTRIX_PROBTRACK_TCK_FILES: ", MRTRIX_PROBTRACK_TCK_FILES)
    # Check that we have all the files we need
    if any(TRACT_TCK_PATH in tckfile for tckfile in MRTRIX_PROBTRACK_TCK_FILES):
        print("--- MRtrix probabilistic tracking files found. Skipping MRtrix probabilistic tracking processing.")
        MRTRIX_PROBTRACK = False

    # If we don't have all the files we need, then we clean the folder and start from scratch
    if MRTRIX_PROBTRACK:
        print("--- MRtrix probabilistic tracking files not found. Cleaning MRtrix probabilistic tracking folder.")
        check_output_folders(PROB_TRACKING_FOLDER_NAME, "MRtrix probabilistic tracking folder", wipe=True)

    # Return the variable
    return MRTRIX_PROBTRACK

# Function to check missing MRtrix global tracking processing
def check_missing_mrtrix_global_tracking(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH, GLOBAL_TRACKING_FOLDER_NAME):
    # Define variable that stores whether or not we should do MRtrix global tracking processing
    MRTRIX_GLOBAL_TRACKING = True
    # Get the MRtrix global tracking paths
    (GLOBAL_FOD_PATH, GLOBAL_FISO_PATH, GLOBAL_TRACT_PATH) = get_mrtrix_global_tracking_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)
    # Grab all the tck files
    MRTRIX_GLOBAL_TRACKING_TCK_FILES = glob_files(GLOBAL_TRACKING_FOLDER_NAME, "tck")
    print("MRTRIX_GLOBAL_TRACKING_TCK_FILES: ", MRTRIX_GLOBAL_TRACKING_TCK_FILES)
    # Check that we have all the files we need
    if any(GLOBAL_TRACT_PATH in global_tract for global_tract in MRTRIX_GLOBAL_TRACKING_TCK_FILES):
        print("--- MRtrix global tracking files found. Skipping MRtrix global tracking processing.")
        MRTRIX_GLOBAL_TRACKING = False

    # If we don't have all the files we need, then we clean the folder and start from scratch
    if MRTRIX_GLOBAL_TRACKING:
        print("--- MRtrix global tracking files not found. Cleaning MRtrix global tracking folder.")
        check_output_folders(GLOBAL_TRACKING_FOLDER_NAME, "MRtrix global tracking folder", wipe=True)
    
    # Return the variable
    return MRTRIX_GLOBAL_TRACKING

# Function to check missing MRtrix connectome processing
def check_missing_mrtrix_connectome(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH, CONNECTIVITY_FOLDER_NAME):
    # Define variable that stores whether or not we should do MRtrix connectome processing
    MRTRIX_CONNECTOME = True
    # Get the MRtrix connectome paths
    (CONNECTIVITY_PROB_PATH, CONNECTIVITY_GLOBAL_PATH) = get_mrtrix_connectome_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)
    # Grab all the csv files
    MRTRIX_CONNECTOME_CSV_FILES = glob_files(CONNECTIVITY_FOLDER_NAME, "csv")
    print("MRTRIX_CONNECTOME_CSV_FILES: ", MRTRIX_CONNECTOME_CSV_FILES)
    # Check that we have all the files we need
    if (any(CONNECTIVITY_PROB_PATH in csv_file for csv_file in MRTRIX_CONNECTOME_CSV_FILES) 
        and any(CONNECTIVITY_GLOBAL_PATH in csv_file for csv_file in MRTRIX_CONNECTOME_CSV_FILES)):
        print("--- MRtrix connectome files found. Skipping MRtrix connectome processing.")
        MRTRIX_CONNECTOME = False

    # If we don't have all the files we need, then we clean the folder and start from scratch
    if MRTRIX_CONNECTOME:
        print("--- MRtrix connectome files not found. Cleaning MRtrix connectome folder.")
        check_output_folders(CONNECTIVITY_FOLDER_NAME, "MRtrix connectome folder", wipe=True)

    # Return the variable
    return MRTRIX_CONNECTOME
