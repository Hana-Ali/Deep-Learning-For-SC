from .dwi_paths import *

from py_helpers import *

# ------------------------------------------------- CHECKING MISSING FILES AND CHECKPOINTS ------------------------------------------------- #

# Function to check which files are missing, and where we should start the processing
def check_all_mrtrix_missing_files(ARGS):
    # Get the arguments
    REGION_ID = ARGS[0]
    DWI_FILES = ARGS[1]
    ATLAS_STPT = ARGS[2]

    # Extract the needed files
    ATLAS_NEEDED = ["atlas"]
    ATLAS = extract_from_input_list(ATLAS_STPT, ATLAS_NEEDED, "atlas_stpt")

    # --------------------- GENERAL VARIABLES NEEDED
    # Get the main MRTRIX folder paths
    (REGION_MRTRIX_PATH, GENERAL_FOLDER_NAME, RESPONSE_FOLDER_NAME, FOD_FOLDER_NAME, 
        FOD_NORM_FOLDER_NAME, ATLAS_REG_FOLDER_NAME, PROB_TRACKING_FOLDER_NAME, 
        CONNECTIVITY_FOLDER_NAME) = main_mrtrix_folder_paths(REGION_ID)

    # --------------------- MRTRIX GENERAL AND CLEAN CHECK
    MRTRIX_GENERAL = check_missing_general_clean(REGION_ID, DWI_FILES, GENERAL_FOLDER_NAME)

    # --------------------- MRTRIX FOD CHECK
    MRTRIX_FOD = check_missing_mrtrix_fod(REGION_ID, FOD_NORM_FOLDER_NAME)
    
    # --------------------- MRTRIX REGISTRATION CHECK
    MRTRIX_REGISTRATION = check_missing_mrtrix_registration(REGION_ID, ATLAS, ATLAS_REG_FOLDER_NAME)
    
    # --------------------- MRTRIX PROBTRACK CHECK
    MRTRIX_PROBTRACK = check_missing_mrtrix_probtrack(REGION_ID, PROB_TRACKING_FOLDER_NAME)
    
    # --------------------- MRTRIX CONNECTOME CHECK
    MRTRIX_CONNECTOME = check_missing_mrtrix_connectome(REGION_ID, CONNECTIVITY_FOLDER_NAME)

    # Return the variables
    return (MRTRIX_GENERAL, MRTRIX_FOD, MRTRIX_REGISTRATION, MRTRIX_PROBTRACK, MRTRIX_CONNECTOME)
    
# Function to check missing mrtrix general processing
def check_missing_general_clean(REGION_ID, DWI_FILES, GENERAL_FOLDER_NAME):
    # Define variable that stores whether or not we should do MRtrix general processing
    MRTRIX_GENERAL = True
    # Get the MRtrix general paths
    (INPUT_MIF_PATH, MASK_MIF_PATH, MASK_NII_PATH) = get_mrtrix_general_paths(REGION_ID, DWI_FILES)
    # Get the MRtrix clean paths
    (DWI_DENOISE_PATH, DWI_NOISE_PATH, DWI_EDDY_PATH, DWI_CLEAN_MIF_PATH, DWI_CLEAN_MASK_PATH,
        DWI_CLEAN_MASK_NII_PATH, DWI_CLEAN_NII_PATH, DWI_CLEAN_BVEC_PATH, DWI_CLEAN_BVAL_PATH) = get_mrtrix_clean_paths(REGION_ID)
    # Grab all the mif and nii files
    MRTRIX_GENERAL_MIF_FILES = glob_files(GENERAL_FOLDER_NAME, "mif")
    MRTRIX_GENERAL_NII_FILES = glob_files(GENERAL_FOLDER_NAME, "nii")
    # Check that we have all the files we need
    if (any(MASK_MIF_PATH in mif_file for mif_file in MRTRIX_GENERAL_MIF_FILES) 
        and any(MASK_NII_PATH in nii_file for nii_file in MRTRIX_GENERAL_NII_FILES)
        and any(DWI_CLEAN_MIF_PATH in mif_file for mif_file in MRTRIX_GENERAL_MIF_FILES)
        and any(DWI_CLEAN_MASK_NII_PATH in nii_file for nii_file in MRTRIX_GENERAL_NII_FILES)
        and any(DWI_CLEAN_NII_PATH in nii_file for nii_file in MRTRIX_GENERAL_NII_FILES)
        and any(DWI_CLEAN_BVEC_PATH in bvec_file for bvec_file in MRTRIX_GENERAL_MIF_FILES)
        and any(DWI_CLEAN_BVAL_PATH in bval_file for bval_file in MRTRIX_GENERAL_MIF_FILES)):
        print("--- MRtrix general files found. Skipping MRtrix general processing.")
        MRTRIX_GENERAL = False
    
    # If we don't have all the files we need, then we clean the folder and start from scratch
    if MRTRIX_GENERAL:
        print("--- MRtrix general files not found. Cleaning MRtrix general folder.")
        check_output_folders(GENERAL_FOLDER_NAME, "MRtrix general folder", wipe=True)

    # Return the variable
    return MRTRIX_GENERAL

# Function to check missing MRtrix FOD processing
def check_missing_mrtrix_fod(REGION_ID, FOD_NORM_FOLDER_NAME):
    # Define variable that stores whether or not we should do MRtrix FOD processing
    MRTRIX_FOD = True
    # Get the MRtrix FOD paths
    (RESPONSE_WM_PATH, RESPONSE_GM_PATH, RESPONSE_CSF_PATH, RESPONSE_VOXEL_PATH,
        WM_FOD_PATH, GM_FOD_PATH, CSF_FOD_PATH, VF_FOD_PATH, WM_FOD_NORM_PATH, 
            GM_FOD_NORM_PATH, CSF_FOD_NORM_PATH) = get_mrtrix_fod_paths(REGION_ID)
    # Grab all the mif files
    MRTRIX_FOD_MIF_FILES = glob_files(FOD_NORM_FOLDER_NAME, "mif")
    # Check that we have all the files we need
    if (any(WM_FOD_NORM_PATH in wm_fod_file for wm_fod_file in MRTRIX_FOD_MIF_FILES) and 
        any(GM_FOD_NORM_PATH in gm_fod_file for gm_fod_file in MRTRIX_FOD_MIF_FILES) and
        any(CSF_FOD_NORM_PATH in csf_fod_file for csf_fod_file in MRTRIX_FOD_MIF_FILES)):
        print("--- MRtrix FOD files found. Skipping MRtrix FOD processing.")
        MRTRIX_FOD = False

    # If we don't have all the files we need, then we clean the folder and start from scratch
    if MRTRIX_FOD:
        print("--- MRtrix FOD files not found. Cleaning MRtrix FOD folder.")
        check_output_folders(FOD_NORM_FOLDER_NAME, "MRtrix FOD folder", wipe=True)
    
    # Return the variable
    return MRTRIX_FOD

# Function to check missing MRtrix registration processing
def check_missing_mrtrix_registration(REGION_ID, ATLAS, ATLAS_REG_FOLDER_NAME):
    # Define variable that stores whether or not we should do MRtrix registration processing
    MRTRIX_REGISTRATION = True
    # Get the MRtrix registration paths
    (DWI_B0_PATH, DWI_B0_NII, ATLAS_DWI_MAP_MAT, ATLAS_DWI_CONVERT_INV, ATLAS_REG_PATH, 
        ATLAS_MIF_PATH) = get_mrtrix_registration_paths(REGION_ID, ATLAS)
    # Grab all the mif files in the T1 and atlas registration folders
    MRTRIX_ATLAS_REG_MIF_FILES = glob_files(ATLAS_REG_FOLDER_NAME, "mif")
    # Check that we have all the files we need
    if (any(ATLAS_REG_PATH in reg_mif_file for reg_mif_file in MRTRIX_ATLAS_REG_MIF_FILES)):
        print("--- MRtrix registration files found. Skipping MRtrix registration processing.")
        MRTRIX_REGISTRATION = False
    
    # If we don't have all the files we need, then we clean the folder and start from scratch
    if MRTRIX_REGISTRATION:
        print("--- MRtrix registration files not found. Cleaning MRtrix registration folder.")
        check_output_folders(ATLAS_REG_FOLDER_NAME, "MRtrix atlas registration folder", wipe=True)

    # Return the variable
    return MRTRIX_REGISTRATION

# Function to check missing MRtrix probabilistic tracking processing
def check_missing_mrtrix_probtrack(REGION_ID, PROB_TRACKING_FOLDER_NAME):
    # Define variable that stores whether or not we should do MRtrix probabilistic tracking processing
    MRTRIX_PROBTRACK = True
    # Get the MRtrix probabilistic tracking paths
    (GM_WM_SEED_PATH, TRACT_TCK_PATH) = get_mrtrix_probtrack_paths(REGION_ID)
    # Grab all the tck files
    MRTRIX_PROBTRACK_TCK_FILES = glob_files(PROB_TRACKING_FOLDER_NAME, "tck")
    # Check that we have all the files we need
    if (any(TRACT_TCK_PATH in tck_file for tck_file in MRTRIX_PROBTRACK_TCK_FILES)):
        print("--- MRtrix probabilistic tracking files found. Skipping MRtrix probabilistic tracking processing.")
        MRTRIX_PROBTRACK = False

    # If we don't have all the files we need, then we clean the folder and start from scratch
    if MRTRIX_PROBTRACK:
        print("--- MRtrix probabilistic tracking files not found. Cleaning MRtrix probabilistic tracking folder.")
        check_output_folders(PROB_TRACKING_FOLDER_NAME, "MRtrix probabilistic tracking folder", wipe=True)

    # Return the variable
    return MRTRIX_PROBTRACK

# Function to check missing MRtrix connectome processing
def check_missing_mrtrix_connectome(REGION_ID, CONNECTIVITY_FOLDER_NAME):
    # Define variable that stores whether or not we should do MRtrix connectome processing
    MRTRIX_CONNECTOME = True
    # Get the MRtrix connectome paths
    (CONNECTIVITY_PROB_PATH) = get_mrtrix_connectome_paths(REGION_ID)
    # Grab all the csv files
    MRTRIX_CONNECTOME_CSV_FILES = glob_files(CONNECTIVITY_FOLDER_NAME, "csv")
    # Check that we have all the files we need
    if (any(CONNECTIVITY_PROB_PATH in csv_file for csv_file in MRTRIX_CONNECTOME_CSV_FILES)):
        print("--- MRtrix connectome files found. Skipping MRtrix connectome processing.")
        MRTRIX_CONNECTOME = False

    # If we don't have all the files we need, then we clean the folder and start from scratch
    if MRTRIX_CONNECTOME:
        print("--- MRtrix connectome files not found. Cleaning MRtrix connectome folder.")
        check_output_folders(CONNECTIVITY_FOLDER_NAME, "MRtrix connectome folder", wipe=True)

    # Return the variable
    return MRTRIX_CONNECTOME
