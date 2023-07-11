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
        COMBINED_CONNECTOME_FOLDER_NAME, INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME, 
        INDIVIDUAL_ROIS_MIF_FOLDER_NAME) = main_mrtrix_folder_paths()

    # --------------------- MRTRIX ATLAS REGISTRATION CHECK
    MRTRIX_ATLAS_REGISTRATION = check_missing_atlas_registration_ants(ATLAS_REG_FOLDER_NAME)

    # --------------------- MRTRIX STREAMLINE COMBINATION CHECK
    MRTRIX_STREAMLINE_COMBINATION = check_missing_mrtrix_streamline_combination(COMBINED_TRACTS_FOLDER_NAME)

    # Return the variables
    return (MRTRIX_ATLAS_REGISTRATION, MRTRIX_STREAMLINE_COMBINATION)

# Function to check which files are missing from the injection mifs
def check_missing_region_files(ARGS):

    # Extract the arguments
    REGION_ID = ARGS[0]
    ATLAS_STPT = ARGS[1]
    
    # Get the main paths
    (GENERAL_MRTRIX_FOLDER, SPECIFIC_MRTRIX_FOLDER, ATLAS_REG_FOLDER_NAME, COMBINED_TRACTS_FOLDER_NAME, 
        COMBINED_CONNECTOME_FOLDER_NAME, INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME, 
        INDIVIDUAL_ROIS_MIF_FOLDER_NAME) = main_mrtrix_folder_paths()
    (REGION_MRTRIX_FOLDER, INJECTION_MIF_FOLDER, INJECTION_ROI_TRACTS_FOLDER, INJECTION_ROI_TRACTS_STATS_FOLDER,
    INJECTION_STATS_MATRIX_FOLDER) = region_mrtrix_folder_paths(REGION_ID)

    # --------------------- MRTRIX INJECTION MIFS CHECK
    INJECTION_MIFS = check_missing_injection_mifs(REGION_ID, INJECTION_MIF_FOLDER)

    # --------------------- MRTRIX ATLAS ROI CHECK
    MRTRIX_ATLAS_ROIS = check_missing_atlas_rois(ATLAS_STPT, INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME)

    # --------------------- MRTRIX ATLAS MIF CONVERSION CHECK
    MRTRIX_ATLAS_MIF_CONVERSION = check_missing_atlas_mif_conversion(ATLAS_STPT, INDIVIDUAL_ROIS_MIF_FOLDER_NAME)

    # --------------------- MRTRIX INJECTION ROI TRACTS CHECK
    # INJECTION_ROI_TRACTS = check_missing_injection_roi_tracts(REGION_ID, ATLAS_STPT, INJECTION_ROI_TRACTS_FOLDER)

    # --------------------- MRTRIX INJECTION ROI TRACTS STATS CHECK
    # INJECTION_ROI_TRACTS_STATS = check_missing_injection_roi_tracts_stats(REGION_ID, ATLAS_STPT, INJECTION_ROI_TRACTS_STATS_FOLDER)

    # Return the variables
    return (INJECTION_MIFS, MRTRIX_ATLAS_ROIS, MRTRIX_ATLAS_MIF_CONVERSION)

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

# Function to check if the injection mifs are missing
def check_missing_injection_mifs(REGION_ID, INJECTION_MIF_FOLDER):
    # Define variable that stores whether or not we should do MRtrix general processing
    INJECTION_MIFS = True
    # Get the MRtrix injection mifs paths
    INJECTION_MIF_PATH = get_injection_mif_path(REGION_ID)
    (REGION_MRTRIX_FOLDER, INJECTION_MIF_FOLDER, INJECTION_ROI_TRACTS_FOLDER, INJECTION_ROI_TRACTS_STATS_FOLDER,
    INJECTION_STATS_MATRIX_FOLDER) = region_mrtrix_folder_paths(REGION_ID)
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

# Check missing injection ROI tracts
def check_missing_injection_roi_tracts(REGION_ID, ATLAS_STPT, INJECTION_ROI_TRACTS_FOLDER):
    # Define variable that stores whether or not we should do MRtrix general processing
    INJECTION_ROI_TRACTS = False
    # Get the paths we need
    (INDIVIDUAL_ROIS_MIF_PATHS) = get_individual_rois_mif_path(ATLAS_STPT)
    # For every ROI, check whether or not we need to redo processing
    for idx, roi_mif_path in enumerate(INDIVIDUAL_ROIS_MIF_PATHS):
        # Get the ROI name or ID
        roi_name = roi_mif_path.split("/")[-1]
        # Get the injection ROI tracts and stats path
        (INJECTION_ROI_TRACTS_PATH) = get_injection_roi_tracts_path(REGION_ID, roi_name)
        # Grab all the tck files
        INJECTION_ROI_TRACTS_FILES = glob_files(INJECTION_ROI_TRACTS_FOLDER, "tck")
        # Check that we have all the files we need
        if not any(INJECTION_ROI_TRACTS_PATH in injection_roi_tracts_file for injection_roi_tracts_file in INJECTION_ROI_TRACTS_FILES):
            INJECTION_ROI_TRACTS = True
            break

    # If we don't have all the files we need, then we clean the folder and start from scratch
    if INJECTION_ROI_TRACTS:
        print("--- MRtrix injection ROI tracts not found. Cleaning MRtrix injection ROI tracts folder.")
        check_output_folders(INJECTION_ROI_TRACTS_FOLDER, "MRtrix injection ROI tracts folder", wipe=True)
    else:
        print("--- MRtrix injection ROI tracts found. Skipping MRtrix injection ROI tracts.")

    # Return the variable
    return (INJECTION_ROI_TRACTS)

# This function returns which rois have not been done yet
# Check missing injection ROI tracts - only the ones that haven't been done before
def not_done_yet_injection_roi_tckedit(REGION_ID, ATLAS_STPT, INJECTION_ROI_TRACTS_FOLDER):
    # Get the paths we need
    (INDIVIDUAL_ROIS_MIF_PATHS) = get_individual_rois_mif_path(ATLAS_STPT)
    # This holds which ROIs we haven't done yet
    INJECTION_ROIS_NOT_DONE = []
    # For every ROI, check whether or not we need to redo processing
    for idx, roi_mif_path in enumerate(INDIVIDUAL_ROIS_MIF_PATHS):
        # Get the ROI name or ID
        roi_name = roi_mif_path.split("/")[-1]
        # Get the injection ROI tracts and stats path
        (INJECTION_ROI_TRACTS_PATH) = get_injection_roi_tracts_path(REGION_ID, roi_name)
        # Grab all the tck files
        INJECTION_ROI_TRACTS_FILES = glob_files(INJECTION_ROI_TRACTS_FOLDER, "tck")
        # Save the globbed files to a text file
        INJECTION_ROI_TRACTS_FILES_PATH = os.path.join(INJECTION_ROI_TRACTS_FOLDER, "injection_roi_found.txt")
        with open(INJECTION_ROI_TRACTS_FILES_PATH, "w") as f:
            for injection_roi_tracts_file in INJECTION_ROI_TRACTS_FILES:
                f.write(injection_roi_tracts_file + "\n")
        # Check that we have all the files we need
        if not any(INJECTION_ROI_TRACTS_PATH in injection_roi_tracts_file for injection_roi_tracts_file in INJECTION_ROI_TRACTS_FILES):
            # Add the ROI to the list of ROIs we haven't done yet
            INJECTION_ROIS_NOT_DONE.append(roi_name)
    
    # Save them to a file
    INJECTION_ROIS_NOT_DONE_PATH = os.path.join(INJECTION_ROI_TRACTS_FOLDER, "injection_rois_not_done.txt")
    with open(INJECTION_ROIS_NOT_DONE_PATH, "w") as f:
        for roi_name in INJECTION_ROIS_NOT_DONE:
            f.write(roi_name + "\n")
            
    # Return the variable
    return (INJECTION_ROIS_NOT_DONE)

# Check missing injection ROI tracts stats
def check_missing_injection_roi_tracts_stats(REGION_ID, ATLAS_STPT, INJECTION_ROI_TRACTS_STATS_FOLDER):
    # Define variable that stores whether or not we should do MRtrix general processing
    INJECTION_ROI_TRACTS_STATS = False
    # Get the paths we need
    (INDIVIDUAL_ROIS_MIF_PATHS) = get_individual_rois_mif_path(ATLAS_STPT)
    # For every ROI, check whether or not we need to redo processing
    for idx, roi_mif_path in enumerate(INDIVIDUAL_ROIS_MIF_PATHS):
        # Get the ROI name or ID
        roi_name = roi_mif_path.split("/")[-1]
        # Get the injection ROI tracts and stats path
        (INJECTION_ROI_LENGTHS_PATH, INJECTION_ROI_COUNT_PATH, INJECTION_ROI_MEAN_PATH, 
         INJECTION_ROI_MEDIAN_PATH, INJECTION_ROI_STD_PATH, INJECTION_ROI_MIN_PATH, 
         INJECTION_ROI_MAX_PATH) = get_injection_roi_tracts_stats_path(REGION_ID, roi_name)
        # Grab all the txt files 
        INJECTION_ROI_STATS_FILES = glob_files(INJECTION_ROI_TRACTS_STATS_FOLDER, "txt")
        # Check that we have all the files we need
        if (not any(INJECTION_ROI_LENGTHS_PATH in injection_roi_stats_file for injection_roi_stats_file in INJECTION_ROI_STATS_FILES)
            or not any(INJECTION_ROI_COUNT_PATH in injection_roi_stats_file for injection_roi_stats_file in INJECTION_ROI_STATS_FILES)
            or not any(INJECTION_ROI_MEAN_PATH in injection_roi_stats_file for injection_roi_stats_file in INJECTION_ROI_STATS_FILES)
            or not any(INJECTION_ROI_MEDIAN_PATH in injection_roi_stats_file for injection_roi_stats_file in INJECTION_ROI_STATS_FILES)
            or not any(INJECTION_ROI_STD_PATH in injection_roi_stats_file for injection_roi_stats_file in INJECTION_ROI_STATS_FILES)
            or not any(INJECTION_ROI_MIN_PATH in injection_roi_stats_file for injection_roi_stats_file in INJECTION_ROI_STATS_FILES)
            or not any(INJECTION_ROI_MAX_PATH in injection_roi_stats_file for injection_roi_stats_file in INJECTION_ROI_STATS_FILES)):
            INJECTION_ROI_TRACTS_STATS = True
            break

    # If we don't have all the files we need, then we clean the folder and start from scratch
    if INJECTION_ROI_TRACTS_STATS:
        print("--- MRtrix injection ROI tracts stats not found. Cleaning MRtrix injection ROI tracts stats folder.")
        check_output_folders(INJECTION_ROI_TRACTS_STATS_FOLDER, "MRtrix injection ROI tracts stats folder", wipe=True)
    else:
        print("--- MRtrix injection ROI tracts stats found. Skipping MRtrix injection ROI tracts stats.")

    # Return the variable
    return (INJECTION_ROI_TRACTS_STATS)

# Check which injection ROIs haven't been done yet for stats
def not_done_yet_injection_roi_stats(REGION_ID, ATLAS_STPT, INJECTION_ROI_TRACTS_STATS_FOLDER):
    # Get the paths we need
    (INDIVIDUAL_ROIS_MIF_PATHS) = get_individual_rois_mif_path(ATLAS_STPT)
    # This holds which ROIs we haven't done yet
    INJECTION_ROIS_NOT_DONE = []
    # For every ROI, check whether or not we need to redo processing
    for idx, roi_mif_path in enumerate(INDIVIDUAL_ROIS_MIF_PATHS):
        # Get the ROI name or ID
        roi_name = roi_mif_path.split("/")[-1]
        # Get the injection ROI tracts and stats path
        (INJECTION_ROI_LENGTHS_PATH, INJECTION_ROI_COUNT_PATH, INJECTION_ROI_MEAN_PATH,
            INJECTION_ROI_MEDIAN_PATH, INJECTION_ROI_STD_PATH, INJECTION_ROI_MIN_PATH,
            INJECTION_ROI_MAX_PATH) = get_injection_roi_tracts_stats_path(REGION_ID, roi_name)
        # Grab all the txt files
        INJECTION_ROI_STATS_FILES = glob_files(INJECTION_ROI_TRACTS_STATS_FOLDER, "txt")
        # Check that we have all the files we need
        if (not any(INJECTION_ROI_LENGTHS_PATH in injection_roi_stats_file for injection_roi_stats_file in INJECTION_ROI_STATS_FILES)
            or not any(INJECTION_ROI_COUNT_PATH in injection_roi_stats_file for injection_roi_stats_file in INJECTION_ROI_STATS_FILES)
            or not any(INJECTION_ROI_MEAN_PATH in injection_roi_stats_file for injection_roi_stats_file in INJECTION_ROI_STATS_FILES)
            or not any(INJECTION_ROI_MEDIAN_PATH in injection_roi_stats_file for injection_roi_stats_file in INJECTION_ROI_STATS_FILES)
            or not any(INJECTION_ROI_STD_PATH in injection_roi_stats_file for injection_roi_stats_file in INJECTION_ROI_STATS_FILES)
            or not any(INJECTION_ROI_MIN_PATH in injection_roi_stats_file for injection_roi_stats_file in INJECTION_ROI_STATS_FILES)
            or not any(INJECTION_ROI_MAX_PATH in injection_roi_stats_file for injection_roi_stats_file in INJECTION_ROI_STATS_FILES)):
            # Add the ROI to the list of ROIs we haven't done yet
            INJECTION_ROIS_NOT_DONE.append(roi_name)

    # Return the variable
    return (INJECTION_ROIS_NOT_DONE)