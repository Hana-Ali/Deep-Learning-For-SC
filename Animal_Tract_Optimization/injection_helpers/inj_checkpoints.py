from .inj_paths import *

import sys
sys.path.append("..")
from py_helpers.general_helpers import *

# ------------------------------------------------- CHECKING MISSING FILES AND CHECKPOINTS ------------------------------------------------- #

# Function to check which files are missing from the atlas and streamline registration
def check_missing_general_files(ARGS):

    # Extract the arguments
    ATLAS_STPT = ARGS[0]
    
    # Get the main paths
    (GENERAL_MRTRIX_FOLDER, SPECIFIC_MRTRIX_FOLDER, ATLAS_REG_FOLDER_NAME, COMBINED_TRACTS_FOLDER_NAME, 
        COMBINED_INJECTIONS_FOLDER_NAME, COMBINED_ATLAS_INJECTIONS_FOLDER_NAME, COMBINED_CONNECTOME_FOLDER_NAME,
        INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME, 
        INDIVIDUAL_ROIS_MIF_FOLDER_NAME) = main_mrtrix_folder_paths()

    # --------------------- MRTRIX ATLAS REGISTRATION CHECK
    MRTRIX_ATLAS_REGISTRATION = check_missing_atlas_registration_ants(ATLAS_REG_FOLDER_NAME)

    # --------------------- MRTRIX STREAMLINE COMBINATION CHECK
    MRTRIX_STREAMLINE_COMBINATION = check_missing_mrtrix_streamline_combination(COMBINED_TRACTS_FOLDER_NAME)

    # --------------------- MRTRIX INJECTION COMBINATION CHECK
    MRTRIX_INJECTION_COMBINATION = check_missing_mrtrix_injection_combination(COMBINED_INJECTIONS_FOLDER_NAME)

    # --------------------- MRTRIX INJECTION AND ATLAS COMBINATION CHECK
    MRTRIX_INJECTION_ATLAS_COMBINATION = check_missing_mrtrix_injection_atlas_combination(COMBINED_ATLAS_INJECTIONS_FOLDER_NAME)

    # --------------------- MRTRIX CONNECTOME CHECK
    MRTRIX_CONNECTOME = check_missing_connectome(COMBINED_CONNECTOME_FOLDER_NAME)

    # Return the variables
    return (MRTRIX_ATLAS_REGISTRATION, MRTRIX_STREAMLINE_COMBINATION, MRTRIX_INJECTION_COMBINATION, 
            MRTRIX_INJECTION_ATLAS_COMBINATION, MRTRIX_CONNECTOME)

# Function to check which files are missing from the injection mifs
def check_missing_region_files(ARGS):

    # Extract the arguments
    REGION_ID = ARGS[0]
    ATLAS_STPT = ARGS[1]
    
    # Get the main paths
    (GENERAL_MRTRIX_FOLDER, SPECIFIC_MRTRIX_FOLDER, ATLAS_REG_FOLDER_NAME, COMBINED_TRACTS_FOLDER_NAME, 
        COMBINED_INJECTIONS_FOLDER_NAME, COMBINED_ATLAS_INJECTIONS_FOLDER_NAME, COMBINED_CONNECTOME_FOLDER_NAME,
        INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME, 
        INDIVIDUAL_ROIS_MIF_FOLDER_NAME) = main_mrtrix_folder_paths()
    (REGION_MRTRIX_FOLDER, INJECTION_MIF_FOLDER, INJECTION_ROI_MIF_FOLDER, 
     INJECTION_ROI_CONNECTOME_FOLDER) = region_mrtrix_folder_paths(REGION_ID)

    # --------------------- MRTRIX INJECTION MIFS CHECK
    INJECTION_MIFS = check_missing_injection_mifs(REGION_ID, INJECTION_MIF_FOLDER)

    # --------------------- MRTRIX ATLAS ROI CHECK
    MRTRIX_ATLAS_ROIS = check_missing_atlas_rois(ATLAS_STPT, INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME)

    # --------------------- MRTRIX ATLAS MIF CONVERSION CHECK
    MRTRIX_ATLAS_MIF_CONVERSION = check_missing_atlas_mif_conversion(ATLAS_STPT, INDIVIDUAL_ROIS_MIF_FOLDER_NAME)

    # --------------------- MRTRIX INJECTION <-> ROI COMBINATION CHECK
    INJECTION_ROI_COMBINATION = check_missing_injection_roi_combination(ATLAS_STPT, INJECTION_ROI_MIF_FOLDER)

    # --------------------- MRTRIX CONNECTOMES CHECK
    CONNECTOMES = check_missing_connectomes_region_roi(ATLAS_STPT, INJECTION_ROI_CONNECTOME_FOLDER)

    # Return the variables
    return (INJECTION_MIFS, MRTRIX_ATLAS_ROIS, MRTRIX_ATLAS_MIF_CONVERSION, INJECTION_ROI_COMBINATION, CONNECTOMES)

# Function to check missing mrtrix atlas registration
def check_missing_atlas_registration_ants(ATLAS_REG_FOLDER_NAME):
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

# Function to check if atlas ROIs are missing
def check_missing_atlas_rois(ATLAS_STPT, INDIVIDUAL_ROIS_NIFTI):
    # Define the variable that stores whether or not we should do MRtrix general processing
    MRTRIX_ATLAS_ROIS = False
    # Get the MRtrix atlas ROIs paths
    ATLAS_ROIS_PATH = get_individual_rois_from_atlas_path(ATLAS_STPT)
    # Grab all the nii.gz files
    EXISTING_ATLAS_ROIS_FILES = glob_files(INDIVIDUAL_ROIS_NIFTI, "nii.gz")
    # Check that we have all the files we need
    for roi_path in ATLAS_ROIS_PATH:
        if not any(roi_path in roi_file for roi_file in EXISTING_ATLAS_ROIS_FILES):
            MRTRIX_ATLAS_ROIS = True
            break
    
    # If we don't have all the files we need, then we clean the folder and start from scratch
    if MRTRIX_ATLAS_ROIS:
        print("--- MRtrix atlas ROIs not found. Cleaning MRtrix atlas ROIs folder.")
        check_output_folders(INDIVIDUAL_ROIS_NIFTI, "MRtrix atlas ROIs folder", wipe=True)
    else:
        print("--- MRtrix atlas ROIs found. Skipping MRtrix atlas ROIs.")
    
    # Return the variable
    return (MRTRIX_ATLAS_ROIS)

# Function to check whether or not we need to do the atlas mif conversion
def check_missing_atlas_mif_conversion(ATLAS_STPT, INDIVIDUAL_ROIS_MIF):
    # Define the variable that stores whether or not we should do MRtrix general processing
    MRTRIX_ATLAS_MIF_CONVERSION = False
    # Get the MRtrix atlas ROIs paths
    INDIVIDUAL_ROIS_MIF_PATHS = get_individual_rois_mif_path(ATLAS_STPT)
    # Grab all the mif files
    EXISTING_ATLAS_ROIS_MIF_FILES = glob_files(INDIVIDUAL_ROIS_MIF, "mif")
    # Check that we have all the files we need
    for mif_path in INDIVIDUAL_ROIS_MIF_PATHS:
        if not any(mif_path in mif_file for mif_file in EXISTING_ATLAS_ROIS_MIF_FILES):
            MRTRIX_ATLAS_MIF_CONVERSION = True
            break

    # If we don't have all the files we need, then we clean the folder and start from scratch
    if MRTRIX_ATLAS_MIF_CONVERSION:
        print("--- MRtrix atlas mif conversion not found. Cleaning MRtrix atlas mif conversion folder.")
        check_output_folders(INDIVIDUAL_ROIS_MIF, "MRtrix atlas mif conversion folder", wipe=True)
    else:
        print("--- MRtrix atlas mif conversion found. Skipping MRtrix atlas mif conversion.")

    # Return the variable
    return (MRTRIX_ATLAS_MIF_CONVERSION)


# Function to check missing mrtrix injection combination
def check_missing_mrtrix_injection_combination(COMBINED_INJECTIONS_FOLDER_NAME):
    # Define variable that stores whether or not we should do MRtrix general processing
    MRTRIX_INJECTION_COMBINATION = True
    # Get the MRtrix injection combination paths
    (COMBINED_INJECTIONS_PATH, COMBINED_INJECTIONS_MIF_PATH) = get_combined_injections_path()
    # Grab all the nii.gz and mif files
    MRTRIX_INJECTION_COMBINATION_NII_FILES = glob_files(COMBINED_INJECTIONS_FOLDER_NAME, "nii.gz")
    MRTRIX_INJECTION_COMBINATION_MIF_FILES = glob_files(COMBINED_INJECTIONS_FOLDER_NAME, "mif")
    # Check that we have all the files we need
    if (any(COMBINED_INJECTIONS_PATH in injection_file for injection_file in MRTRIX_INJECTION_COMBINATION_NII_FILES)
        and any(COMBINED_INJECTIONS_MIF_PATH in injection_file for injection_file in MRTRIX_INJECTION_COMBINATION_MIF_FILES)):
        print("--- MRtrix injection combination files found. Skipping MRtrix injection combination.")
        MRTRIX_INJECTION_COMBINATION = False

    # If we don't have all the files we need, then we clean the folder and start from scratch
    if MRTRIX_INJECTION_COMBINATION:
        print("--- MRtrix injection combination files not found. Cleaning MRtrix injection combination folder.")
        check_output_folders(COMBINED_INJECTIONS_FOLDER_NAME, "MRtrix injection combination folder", wipe=True)

    # Return the variable
    return (MRTRIX_INJECTION_COMBINATION)

# Function to check if the injection and atlas combination is missing
def check_missing_mrtrix_injection_atlas_combination(COMBINED_INJECTIONS_FOLDER_NAME):

    # Define variable that stores whether or not we should do MRtrix general processing
    MRTRIX_INJECTION_ATLAS_COMBINATION = True
    # Get the MRtrix injection and atlas combination paths
    (COMBINED_INJECTION_ATLAS_MIF_PATH, COMBINED_INJECTION_ATLAS_NII_PATH) = get_combined_injection_atlas_path()
    # Grab all the nii.gz and mif files
    COMBINED_INJECTION_ATLAS_NII_FILES = glob_files(COMBINED_INJECTIONS_FOLDER_NAME, "nii.gz")
    COMBINED_INJECTION_ATLAS_MIF_FILES = glob_files(COMBINED_INJECTIONS_FOLDER_NAME, "mif")
    # Check that we have all the files we need
    if (any(COMBINED_INJECTION_ATLAS_NII_PATH in injection_atlas_file for injection_atlas_file in COMBINED_INJECTION_ATLAS_NII_FILES)
        and any(COMBINED_INJECTION_ATLAS_MIF_PATH in injection_atlas_file for injection_atlas_file in COMBINED_INJECTION_ATLAS_MIF_FILES)):
        print("--- MRtrix injection and atlas combination files found. Skipping MRtrix injection and atlas combination.")
        MRTRIX_INJECTION_ATLAS_COMBINATION = False

    # If we don't have all the files we need, then we clean the folder and start from scratch
    if MRTRIX_INJECTION_ATLAS_COMBINATION:
        print("--- MRtrix injection and atlas combination files not found. Cleaning MRtrix injection and atlas combination folder.")
        check_output_folders(COMBINED_INJECTIONS_FOLDER_NAME, "MRtrix injection and atlas combination folder", wipe=True)

    # Return the variable
    return (MRTRIX_INJECTION_ATLAS_COMBINATION)

# Function to check if the injection mifs are missing
def check_missing_injection_mifs(REGION_ID, INJECTION_MIF_FOLDER):
    # Define variable that stores whether or not we should do MRtrix general processing
    INJECTION_MIFS = True
    # Get the MRtrix injection mifs paths
    INJECTION_MIF_PATH = get_injection_mif_path(REGION_ID)
    (REGION_MRTRIX_FOLDER, INJECTION_MIF_FOLDER, INJECTION_ROI_MIF_FOLDER, 
     INJECTION_ROI_CONNECTOME_FOLDER) = region_mrtrix_folder_paths(REGION_ID)
    # Grab all the mif files
    INJECTION_MIF_FILES = glob_files(INJECTION_MIF_FOLDER, "mif")
    # Check that we have all the files we need
    if any(INJECTION_MIF_PATH in injection_mif_file for injection_mif_file in INJECTION_MIF_FILES):
        print("--- MRtrix injection mifs found. Skipping MRtrix injection mifs.")
        INJECTION_MIFS = False

    # If we don't have all the files we need, then we clean the folder and start from scratch
    if INJECTION_MIFS:
        print("--- MRtrix injection mifs not found. Cleaning MRtrix injection mifs folder.")
        check_output_folders(INJECTION_MIF_FOLDER, "MRtrix injection mifs folder", wipe=True)

    # Return the variable
    return (INJECTION_MIFS)

# Function to check if the injection <-> ROI combination is missing
def check_missing_injection_roi_combination(ATLAS_STPT, INJECTION_ROI_CONNECTOME_FOLDER):
    # Define variable that stores whether or not we should do MRtrix general processing
    INJECTION_ROI_COMBINATION = True
    # Get all the atlas ROI mif paths
    (INDIVIDUAL_ROIS_MIF_PATHS) = get_individual_rois_mif_path(ATLAS_STPT)
    # Grab all the injection <-> ROI combination files
    INJECTION_ROI_FILES = glob_files(INJECTION_ROI_CONNECTOME_FOLDER, "mif")
    # Check that we have all the files we need - for every ROI, we have a combo from the region files
    for roi_mif_path in INDIVIDUAL_ROIS_MIF_PATHS:
        # Get the ROI name - FROM THE MAIN ROIS MIF FILE
        roi_mif_name = roi_mif_path.split("/")[-1]
        if not any(roi_mif_name in roi_file for roi_file in INJECTION_ROI_FILES):
            INJECTION_ROI_COMBINATION = True
            break

    # If we don't have all the files we need, then we clean the folder and start from scratch
    if INJECTION_ROI_COMBINATION:
        print("--- MRtrix injection <-> ROI combination not found. Cleaning MRtrix injection <-> ROI combination folder.")
        check_output_folders(INJECTION_ROI_CONNECTOME_FOLDER, "MRtrix injection <-> ROI combination folder", wipe=True)
    else:
        print("--- MRtrix injection <-> ROI combination found. Skipping MRtrix injection <-> ROI combination.")

    # Return the variable
    return (INJECTION_ROI_COMBINATION)

# Function to check if the connectomes are missing
def check_missing_connectomes_region_roi(ATLAS_STPT, INJECTION_ROI_CONNECTOME_FOLDER):
    # Define variable that stores whether or not we should do MRtrix general processing
    CONNECTOMES = True
    # Get all the atlas ROI mif paths
    (INDIVIDUAL_ROIS_MIF_PATHS) = get_individual_rois_mif_path(ATLAS_STPT)
    # Grab all the connectome files, and filter for the ones with the specific region ID
    INJECTION_ROI_CONNECTOMES_FILES = glob_files(INJECTION_ROI_CONNECTOME_FOLDER, "csv")
    # Check that we have all the files we need
    for roi_mif_path in INDIVIDUAL_ROIS_MIF_PATHS:
        # Get the ROI name - FROM THE MAIN ROIS MIF FILE
        roi_mif_name = roi_mif_path.split("/")[-1]
        if not any(roi_mif_name in connectome_file for connectome_file in INJECTION_ROI_CONNECTOMES_FILES):
            CONNECTOMES = True
            break

    # If we don't have all the files we need, then we clean the folder and start from scratch
    if CONNECTOMES:
        print("--- MRtrix connectomes not found. Cleaning MRtrix connectomes folder.")
        check_output_folders(INJECTION_ROI_CONNECTOME_FOLDER, "MRtrix connectomes folder", wipe=True)
    else:
        print("--- MRtrix connectomes found. Skipping MRtrix connectomes.")

    # Return the variable
    return (CONNECTOMES)

# Function to check for a missing connectome
def check_missing_connectome(COMBINED_CONNECTOME_FOLDER_NAME):

    # Define variable that stores whether or not we should do MRtrix general processing
    MRTRIX_CONNECTOME = True
    # Get the connectome paths
    (COMBINED_INJECTION_ATLAS_CONNECTOME_PATH) = get_combined_injection_atlas_connectome_path()
    # Grab all the connectome files
    COMBINED_INJECTION_ATLAS_CONNECTOME_FILES = glob_files(COMBINED_CONNECTOME_FOLDER_NAME, "csv")
    # Check that we have all the files we need
    if any(COMBINED_INJECTION_ATLAS_CONNECTOME_PATH in connectome_file for connectome_file in COMBINED_INJECTION_ATLAS_CONNECTOME_FILES):
        print("--- MRtrix connectome found. Skipping MRtrix connectome.")
        MRTRIX_CONNECTOME = False

    # If we don't have all the files we need, then we clean the folder and start from scratch
    if MRTRIX_CONNECTOME:
        print("--- MRtrix connectome not found. Cleaning MRtrix connectome folder.")
        check_output_folders(COMBINED_CONNECTOME_FOLDER_NAME, "MRtrix connectome folder", wipe=True)

    # Return the variable
    return (MRTRIX_CONNECTOME)

