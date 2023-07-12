import os

import sys
sys.path.append("..")
from py_helpers.general_helpers import *

# ------------------------------------------------- PATHS ------------------------------------------------- #
# Get the FSL file paths
def get_fsl_paths(NEEDED_FILE_PATHS, MAIN_FSL_PATH):

    # Extract what we need here from the needed file paths
    filename = NEEDED_FILE_PATHS["filename"]

    # Creating folder for each subject in FSL folder
    SUBJECT_FOLDER_NAME = os.path.join(MAIN_FSL_PATH, filename)
    check_output_folders(SUBJECT_FOLDER_NAME, "FSL subject folder", wipe=False)
    
    # Define the path for skull stripped T1
    SKULL_STRIP_PATH = os.path.join(SUBJECT_FOLDER_NAME, "_skull_strip")

    return (SKULL_STRIP_PATH)

# Get the MRtrix cleaning file paths
def get_mrtrix_clean_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH):

    # Extract what we need here from the needed file paths
    dwi_filename = NEEDED_FILE_PATHS["filename"]

    # Creating folder for each subject in MRTRIX folder. DON'T WIPE as it'll be used by other functions
    SUBJECT_FOLDER_NAME = os.path.join(MAIN_MRTRIX_PATH, dwi_filename)
    if not os.path.exists(SUBJECT_FOLDER_NAME):
        print("--- MRtrix subject folder not found. Created folder: {}".format(SUBJECT_FOLDER_NAME))
        os.makedirs(SUBJECT_FOLDER_NAME)
    
    # Creating folder for cleaning in MRTRIX folder. WIPE since this is the only function that uses it
    CLEANING_FOLDER_NAME = os.path.join(SUBJECT_FOLDER_NAME, "cleaning")
    check_output_folders(CLEANING_FOLDER_NAME, "MRtrix cleaning folder", wipe=False)

    # DWI nii -> mif filepath
    INPUT_MIF_PATH = os.path.join(CLEANING_FOLDER_NAME, "{}_clean_input".format(dwi_filename))
    # DWI denoising, eddy correction, and bias correction paths
    DWI_DENOISE_PATH = os.path.join(CLEANING_FOLDER_NAME, "{}_denoise".format(dwi_filename))
    DWI_NOISE_PATH = os.path.join(CLEANING_FOLDER_NAME, "{}_noise".format(dwi_filename))
    DWI_EDDY_PATH = os.path.join(CLEANING_FOLDER_NAME, "{}_eddy".format(dwi_filename))
    DWI_BIAS_PATH = os.path.join(CLEANING_FOLDER_NAME, "{}_bias".format(dwi_filename))
    # DWI mif -> nii filepath
    DWI_CONVERT_PATH = os.path.join(CLEANING_FOLDER_NAME, "{}_clean_output".format(dwi_filename))

    # Return the paths
    return (CLEANING_FOLDER_NAME, INPUT_MIF_PATH, DWI_DENOISE_PATH, DWI_NOISE_PATH, DWI_EDDY_PATH, 
            DWI_BIAS_PATH, DWI_CONVERT_PATH)

# Get the DSI_STUDIO file paths
def get_dsi_studio_paths(NEEDED_FILE_PATHS, MAIN_STUDIO_PATH):

    # Extract what we need here from the needed file paths
    dwi_filename = NEEDED_FILE_PATHS["filename"]

    # Creating folder for each subject in DSI_STUDIO folder. CAN WIPE as this is the only function that uses it
    SUBJECT_FOLDER_NAME = os.path.join(MAIN_STUDIO_PATH, dwi_filename)
    check_output_folders(SUBJECT_FOLDER_NAME, "DSI Studio subject folder", wipe=False)
    # Creating logs folder for each subject. CAN WIPE as this is the only function that uses it
    SUBJECT_LOGS_FOLDER = os.path.join(SUBJECT_FOLDER_NAME, "logs")
    check_output_folders(SUBJECT_LOGS_FOLDER, "DSI Studio logs folder", wipe=False)

    # Create folders and file paths for eachcommand
    STUDIO_SRC_FOLDER = os.path.join(SUBJECT_FOLDER_NAME, "src")
    STUDIO_DTI_FOLDER = os.path.join(SUBJECT_FOLDER_NAME, "dti")
    STUDIO_QSDR_FOLDER = os.path.join(SUBJECT_FOLDER_NAME, "qsdr")
    check_output_folders(STUDIO_SRC_FOLDER, "DSI Studio src folder", wipe=False)
    check_output_folders(STUDIO_DTI_FOLDER, "DSI Studio dti folder", wipe=False)
    check_output_folders(STUDIO_QSDR_FOLDER, "DSI Studio qsdr folder", wipe=False)

    STUDIO_SRC_PATH = os.path.join(STUDIO_SRC_FOLDER, "{}_clean".format(dwi_filename))
    STUDIO_DTI_PATH = os.path.join(STUDIO_DTI_FOLDER, "{}_dti".format(dwi_filename))
    STUDIO_QSDR_PATH = os.path.join(STUDIO_QSDR_FOLDER, "{}_qsdr".format(dwi_filename))

    # Log folders and file paths
    SRC_LOG_FOLDER = os.path.join(SUBJECT_LOGS_FOLDER, "dwi_to_src")
    DTI_LOG_FOLDER = os.path.join(SUBJECT_LOGS_FOLDER, "src_to_dti")
    QSDR_LOG_FOLDER = os.path.join(SUBJECT_LOGS_FOLDER, "src_to_qsdr")
    TRACT_LOG_FOLDER = os.path.join(SUBJECT_LOGS_FOLDER, "recon_to_tract")
    DTI_EXP_LOG_FOLDER = os.path.join(SUBJECT_LOGS_FOLDER, "dti_export")
    QSDR_EXP_LOG_FOLDER = os.path.join(SUBJECT_LOGS_FOLDER, "qsdr_export")
    check_output_folders(SRC_LOG_FOLDER, "DSI Studio dwi to src logs folder", wipe=False)
    check_output_folders(DTI_LOG_FOLDER, "DSI Studio src to dti logs folder", wipe=False)
    check_output_folders(QSDR_LOG_FOLDER, "DSI Studio src to qsdr logs folder", wipe=False)
    check_output_folders(TRACT_LOG_FOLDER, "DSI Studio recon to tract logs folder", wipe=False)
    check_output_folders(DTI_EXP_LOG_FOLDER, "DSI Studio dti export logs folder", wipe=False)
    check_output_folders(QSDR_EXP_LOG_FOLDER, "DSI Studio qsdr export logs folder", wipe=False)

    SRC_LOG_PATH = os.path.join(SRC_LOG_FOLDER, "src_log_{}.txt".format(dwi_filename))
    DTI_LOG_PATH = os.path.join(DTI_LOG_FOLDER, "dti_log_{}.txt".format(dwi_filename))
    DTI_EXP_LOG_PATH = os.path.join(DTI_EXP_LOG_FOLDER, "exporting_dti_log_{}.txt".format(dwi_filename))
    QSDR_LOG_PATH = os.path.join(QSDR_LOG_FOLDER, "qsdr_log_{}.txt".format(dwi_filename))
    QSDR_EXP_LOG_PATH = os.path.join(QSDR_EXP_LOG_FOLDER, "exporting_qsdr_log_{}.txt".format(dwi_filename))
    TRACT_LOG_PATH = os.path.join(TRACT_LOG_FOLDER, "tract_log_{}.txt".format(dwi_filename))

    # Return the paths
    return (STUDIO_SRC_PATH, STUDIO_DTI_PATH, STUDIO_QSDR_PATH, SRC_LOG_PATH, DTI_LOG_PATH, 
            DTI_EXP_LOG_PATH, QSDR_LOG_PATH, QSDR_EXP_LOG_PATH, TRACT_LOG_PATH)

# Define MRTRIX folder paths
def main_mrtrix_folder_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH):

    # Extract what we need here from the needed file paths
    dwi_filename = NEEDED_FILE_PATHS["filename"]

    # Creating folder for each subject in MRTRIX folder. DON'T WIPE as it'll be used by other functions
    SUBJECT_FOLDER_NAME = os.path.join(MAIN_MRTRIX_PATH, dwi_filename)
    check_output_folders(SUBJECT_FOLDER_NAME, "MRtrix subject folder", wipe=False)
    
    # Creating folder for cleaning in MRTRIX folder. DON'T WIPE as it'll be used by other functions
    RECON_FOLDER_NAME = os.path.join(SUBJECT_FOLDER_NAME, "reconstruction")
    check_output_folders(RECON_FOLDER_NAME, "MRtrix reconstruction folder", wipe=False)

    # Because there's a lot of commands, we define extra folder paths here. DON'T WIPE as it'll be used by other functions
    RESPONSE_FOLDER_NAME = os.path.join(RECON_FOLDER_NAME, "response")
    FOD_FOLDER_NAME = os.path.join(RECON_FOLDER_NAME, "fod")
    FOD_NORM_FOLDER_NAME = os.path.join(RECON_FOLDER_NAME, "fod_norm")
    T1_REG_FOLDER_NAME = os.path.join(RECON_FOLDER_NAME, "t1_and_Fivett_reg")
    ATLAS_REG_FOLDER_NAME = os.path.join(RECON_FOLDER_NAME, "atlas_reg")
    PROB_TRACKING_FOLDER_NAME = os.path.join(RECON_FOLDER_NAME, "prob_tracking")
    GLOBAL_TRACKING_FOLDER_NAME = os.path.join(RECON_FOLDER_NAME, "global_tracking")
    CONNECTIVITY_FOLDER_NAME = os.path.join(RECON_FOLDER_NAME, "connectivity")
    check_output_folders(RESPONSE_FOLDER_NAME, "MRtrix response folder", wipe=False)
    check_output_folders(FOD_FOLDER_NAME, "MRtrix fod folder", wipe=False)
    check_output_folders(FOD_NORM_FOLDER_NAME, "MRtrix fod norm folder", wipe=False)
    check_output_folders(T1_REG_FOLDER_NAME, "MRtrix t1 and Fivett reg folder", wipe=False)
    check_output_folders(ATLAS_REG_FOLDER_NAME, "MRtrix atlas reg folder", wipe=False)
    check_output_folders(PROB_TRACKING_FOLDER_NAME, "MRtrix probabilistic tracking folder", wipe=False)
    check_output_folders(GLOBAL_TRACKING_FOLDER_NAME, "MRtrix global tracking folder", wipe=False)
    check_output_folders(CONNECTIVITY_FOLDER_NAME, "MRtrix connectivity folder", wipe=False)

    # Return the paths
    return (SUBJECT_FOLDER_NAME, RECON_FOLDER_NAME, RESPONSE_FOLDER_NAME, FOD_FOLDER_NAME, FOD_NORM_FOLDER_NAME,
                T1_REG_FOLDER_NAME, ATLAS_REG_FOLDER_NAME, PROB_TRACKING_FOLDER_NAME, GLOBAL_TRACKING_FOLDER_NAME,
                    CONNECTIVITY_FOLDER_NAME)

# Define MRtrix general paths
def get_mrtrix_general_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH):

    # Extract what we need here from the needed file paths
    dwi_filename = NEEDED_FILE_PATHS["filename"]
    
    # Get the folder names
    (_, RECON_FOLDER_NAME, _, _, _, _, _, _, _, _) = main_mrtrix_folder_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)

    # DWI nii -> mif filepath
    INPUT_MIF_PATH = os.path.join(RECON_FOLDER_NAME, "{}_recon_input".format(dwi_filename))
    # Path of the brain mask
    MASK_MIF_PATH = os.path.join(RECON_FOLDER_NAME, "{}_mask".format(dwi_filename))

    # Return the paths
    return (INPUT_MIF_PATH, MASK_MIF_PATH)
    
# Define MRTrix FOD paths
def get_mrtrix_fod_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH):

    # Extract what we need here from the needed file paths
    dwi_filename = NEEDED_FILE_PATHS["filename"]
    
    # Get the folder names
    (_, _, RESPONSE_FOLDER_NAME, FOD_FOLDER_NAME, FOD_NORM_FOLDER_NAME, _, 
        _, _, _, _) = main_mrtrix_folder_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)

    # Finding the response function paths
    RESPONSE_WM_PATH = os.path.join(RESPONSE_FOLDER_NAME, "{}_wm".format(dwi_filename))
    RESPONSE_GM_PATH = os.path.join(RESPONSE_FOLDER_NAME, "{}_gm".format(dwi_filename))
    RESPONSE_CSF_PATH = os.path.join(RESPONSE_FOLDER_NAME, "{}_csf".format(dwi_filename))
    RESPONSE_VOXEL_PATH = os.path.join(RESPONSE_FOLDER_NAME, "{}_response_voxels".format(dwi_filename))
    # Finding the fiber orientation distributions (fODs)
    WM_FOD_PATH = os.path.join(FOD_FOLDER_NAME, "{}_wmfod".format(dwi_filename))
    GM_FOD_PATH = os.path.join(FOD_FOLDER_NAME, "{}_gmfod".format(dwi_filename))
    CSF_FOD_PATH = os.path.join(FOD_FOLDER_NAME, "{}_csffod".format(dwi_filename))
    VF_FOD_PATH = os.path.join(FOD_FOLDER_NAME, "{}_vf".format(dwi_filename))
    # Normalizing the fiber orientation distributions (fODs)
    WM_FOD_NORM_PATH = os.path.join(FOD_NORM_FOLDER_NAME, "{}_wmfod_norm".format(dwi_filename))
    GM_FOD_NORM_PATH = os.path.join(FOD_NORM_FOLDER_NAME, "{}_gmfod_norm".format(dwi_filename))
    CSF_FOD_NORM_PATH = os.path.join(FOD_NORM_FOLDER_NAME, "{}_csffod_norm".format(dwi_filename))
    
    # Return the paths
    return (RESPONSE_WM_PATH, RESPONSE_GM_PATH, RESPONSE_CSF_PATH, RESPONSE_VOXEL_PATH,
                WM_FOD_PATH, GM_FOD_PATH, CSF_FOD_PATH, VF_FOD_PATH, WM_FOD_NORM_PATH, GM_FOD_NORM_PATH, CSF_FOD_NORM_PATH)

# Define MRtrix Registration paths
def get_mrtrix_registration_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH, ATLAS):

    # Extract what we need here from the needed file paths
    dwi_filename = NEEDED_FILE_PATHS["filename"]
    
    # Get the folder names
    (_, _, _, _, _, T1_REG_FOLDER_NAME, ATLAS_REG_FOLDER_NAME, _, _, 
        _) = main_mrtrix_folder_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)

    # Convert T1 nii to mif, then create 5ttgen, register/map it to the B0 space of DWI, then convert back to nii
    T1_MIF_PATH = os.path.join(T1_REG_FOLDER_NAME, "{}_t1".format(dwi_filename))
    FIVETT_NOREG_PATH = os.path.join(T1_REG_FOLDER_NAME, "{}_5ttgen".format(dwi_filename))
    DWI_B0_PATH = os.path.join(T1_REG_FOLDER_NAME, "{}_b0".format(dwi_filename))
    DWI_B0_NII = os.path.join(T1_REG_FOLDER_NAME, "{}_b0.nii.gz".format(dwi_filename))
    FIVETT_GEN_NII = os.path.join(T1_REG_FOLDER_NAME, "{}_5ttgen.nii.gz".format(dwi_filename))
    T1_DWI_MAP_MAT = os.path.join(T1_REG_FOLDER_NAME, "{}_t12dwi_fsl".format(dwi_filename))
    T1_DWI_CONVERT_INV = os.path.join(T1_REG_FOLDER_NAME, "{}_t12dwi_mrtrix".format(dwi_filename))
    FIVETT_REG_PATH = os.path.join(T1_REG_FOLDER_NAME, "{}_5ttgenreg".format(dwi_filename))
    # Convert atlas nii to mif, then register/map it to the B0 space of DWI
    ATLAS_DWI_MAP_MAT = os.path.join(ATLAS_REG_FOLDER_NAME, "{}_atlas2dwi_fsl".format(dwi_filename))
    ATLAS_DWI_CONVERT_INV = os.path.join(ATLAS_REG_FOLDER_NAME, "{}_atlas2dwi_mrtrix".format(dwi_filename))
    ATLAS_REG_PATH = os.path.join(ATLAS_REG_FOLDER_NAME, "{}_atlasreg".format(dwi_filename))
    # Getting the name of the atlas without .nii.gz
    ATLAS_NAME = ATLAS.split("/")[-1].split(".")[0].split("_")[0]
    ATLAS_MIF_PATH = os.path.join(ATLAS_REG_FOLDER_NAME, "{}_atlas_mif".format(ATLAS_NAME))

    # Return the paths
    return (T1_MIF_PATH, FIVETT_NOREG_PATH, DWI_B0_PATH, DWI_B0_NII, FIVETT_GEN_NII, T1_DWI_MAP_MAT,
                T1_DWI_CONVERT_INV, FIVETT_REG_PATH, ATLAS_DWI_MAP_MAT, ATLAS_DWI_CONVERT_INV, ATLAS_REG_PATH, ATLAS_MIF_PATH)

# Define MRtrix probabilistic tracking paths
def get_mrtrix_probtrack_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH):

    # Extract what we need here from the needed file paths
    dwi_filename = NEEDED_FILE_PATHS["filename"]
    
    # Get the folder names
    (_, _, _, _, _, _, _, PROB_TRACKING_FOLDER_NAME, _, _) = main_mrtrix_folder_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)

    # Mask for streamline seeding paths and probabilistic tractography path
    GM_WM_SEED_PATH = os.path.join(PROB_TRACKING_FOLDER_NAME, "{}_gmwmseed".format(dwi_filename))
    TRACT_TCK_PATH = os.path.join(PROB_TRACKING_FOLDER_NAME, "{}_tract".format(dwi_filename))

    # Return the paths
    return (GM_WM_SEED_PATH, TRACT_TCK_PATH)

# Define MRtrix global tracking paths
def get_mrtrix_global_tracking_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH):
    
    # Extract what we need here from the needed file paths
    dwi_filename = NEEDED_FILE_PATHS["filename"]
    
    # Get the folder names
    (_, _, _, _, _, _, _, _, GLOBAL_TRACKING_FOLDER_NAME, _) = main_mrtrix_folder_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)

    # FOD and FISO for global tractography paths
    GLOBAL_FOD_PATH = os.path.join(GLOBAL_TRACKING_FOLDER_NAME, "{}_global_fod".format(dwi_filename))
    GLOBAL_FISO_PATH = os.path.join(GLOBAL_TRACKING_FOLDER_NAME, "{}_global_fiso".format(dwi_filename))
    # Global tractography path
    GLOBAL_TRACT_PATH = os.path.join(GLOBAL_TRACKING_FOLDER_NAME, "{}_global_tract".format(dwi_filename))

    # Return the paths
    return (GLOBAL_FOD_PATH, GLOBAL_FISO_PATH, GLOBAL_TRACT_PATH)

# Define MRtrix connectome paths
def get_mrtrix_connectome_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH):

    # Extract what we need here from the needed file paths
    dwi_filename = NEEDED_FILE_PATHS["filename"]

    # Get the folder names
    (_, _, _, _, _, _, _, _, _, CONNECTIVITY_FOLDER_NAME) = main_mrtrix_folder_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)

    # Connectivity matrix path
    CONNECTIVITY_PROB_PATH = os.path.join(CONNECTIVITY_FOLDER_NAME, "{}_prob_connectivity".format(dwi_filename))
    CONNECTIVITY_GLOBAL_PATH = os.path.join(CONNECTIVITY_FOLDER_NAME, "{}_global_connectivity".format(dwi_filename))

    # Return the paths
    return (CONNECTIVITY_PROB_PATH, CONNECTIVITY_GLOBAL_PATH)
