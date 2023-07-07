from .inj_paths import *

import sys
sys.path.append("..")
from py_helpers.general_helpers import *

# ------------------------------------------------- CHECKING MISSING FILES AND CHECKPOINTS ------------------------------------------------- #

# Function to check which files are missing from the atlas and streamline registration
def check_missing_atlas_streamline_registration():
    
    # Get the main paths
    (ATLAS_REG_FOLDER_NAME, COMBINED_TRACTS_FOLDER_NAME) = main_mrtrix_folder_paths()

    # --------------------- MRTRIX ATLAS REGISTRATION CHECK
    MRTRIX_ATLAS_REGISTRATION = check_missing_mrtrix_atlas_registration_ants(ATLAS_REG_FOLDER_NAME)

    # --------------------- MRTRIX STREAMLINE COMBINATION CHECK
    MRTRIX_STREAMLINE_COMBINATION = check_missing_mrtrix_streamline_combination(COMBINED_TRACTS_FOLDER_NAME)

    # Return the variables
    return (MRTRIX_ATLAS_REGISTRATION, MRTRIX_STREAMLINE_COMBINATION)


# Function to check missing mrtrix atlas registration
def check_missing_mrtrix_atlas_registration_ants(ATLAS_REG_FOLDER_NAME):
    # Define variable that stores whether or not we should do MRtrix general processing
    MRTRIX_ATLAS_REGISTRATION = True
    # Get the MRtrix atlas registration paths
    (ATLAS_REG_PATH, ATLAS_REG_MIF_PATH) = get_mrtrix_atlas_reg_paths_ants()
    # Grab all the nii.gz and mif files
    MRTRIX_ATLAS_REGISTRATION_NII_FILES = glob_files(ATLAS_REG_FOLDER_NAME, "nii.gz")
    MRTRIX_ATLAS_REGISTRATION_MIF_FILES = glob_files(ATLAS_REG_FOLDER_NAME, "mif")
    # Check that we have all the files we need
    if (any(ATLAS_REG_PATH in reg_nii_file for reg_nii_file in MRTRIX_ATLAS_REGISTRATION_NII_FILES)
        and any(ATLAS_REG_MIF_PATH in reg_mif_file for reg_mif_file in MRTRIX_ATLAS_REGISTRATION_MIF_FILES)):
        print("--- MRtrix atlas registration files found. Skipping MRtrix atlas registration.")
        MRTRIX_ATLAS_REGISTRATION = False

    # If we don't have all the files we need, then we clean the folder and start from scratch
    if MRTRIX_ATLAS_REGISTRATION:
        print("--- MRtrix atlas registration files not found. Cleaning MRtrix atlas registration folder.")
        check_output_folders(ATLAS_REG_FOLDER_NAME, "MRtrix atlas registration folder", wipe=True)

    # Return the variable
    return (MRTRIX_ATLAS_REGISTRATION)

# Function to check missing mrtrix streamline combination
def check_missing_mrtrix_streamline_combination(COMBINED_TRACTS_FOLDER_NAME):
    # Define variable that stores whether or not we should do MRtrix general processing
    MRTRIX_STREAMLINE_COMBINATION = True
    # Get the MRtrix streamline combination paths
    COMBINED_TRACTS_PATH = get_combined_tracts_path()
    # Grab all the tck files
    MRTRIX_STREAMLINE_COMBINATION_TCK_FILES = glob_files(COMBINED_TRACTS_FOLDER_NAME, "tck")
    # Check that we have all the files we need
    if any(COMBINED_TRACTS_PATH in streamline_file for streamline_file in MRTRIX_STREAMLINE_COMBINATION_TCK_FILES):
        print("--- MRtrix streamline combination files found. Skipping MRtrix streamline combination.")
        MRTRIX_STREAMLINE_COMBINATION = False

    # If we don't have all the files we need, then we clean the folder and start from scratch
    if MRTRIX_STREAMLINE_COMBINATION:
        print("--- MRtrix streamline combination files not found. Cleaning MRtrix streamline combination folder.")
        check_output_folders(COMBINED_TRACTS_FOLDER_NAME, "MRtrix streamline combination folder", wipe=True)

    # Return the variable
    return (MRTRIX_STREAMLINE_COMBINATION)
