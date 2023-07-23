# Import py_helpers
from py_helpers import *

# ----------------------------------------------------- PATH DEFINITIONS ----------------------------------------------------- #

# Define MRTRIX folder paths
def main_mrtrix_folder_paths(MAIN_MRTRIX_FOLDER, REGION_ID):
    
    # Create the region's main folder - don't wipe as it'll be invoked more than once
    REGION_MRTRIX_PATH = os.path.join(MAIN_MRTRIX_FOLDER, REGION_ID)
    check_output_folders(REGION_MRTRIX_PATH, "REGION MRTRIX PATH", wipe=False)
    
    # Because there's a lot of commands, we define extra folder paths here. DON'T WIPE as it'll be used by other functions
    GENERAL_FOLDER_NAME = os.path.join(REGION_MRTRIX_PATH, "general")
    RESPONSE_FOLDER_NAME = os.path.join(REGION_MRTRIX_PATH, "response")
    FOD_FOLDER_NAME = os.path.join(REGION_MRTRIX_PATH, "fod")
    FOD_NORM_FOLDER_NAME = os.path.join(REGION_MRTRIX_PATH, "fod_norm")
    REGISTRATION_FOLDER_NAME = os.path.join(REGION_MRTRIX_PATH, "registration")
    PROB_TRACKING_FOLDER_NAME = os.path.join(REGION_MRTRIX_PATH, "prob_tracking")
    CONNECTIVITY_FOLDER_NAME = os.path.join(REGION_MRTRIX_PATH, "connectivity")

    # Check the output folders
    check_output_folders(GENERAL_FOLDER_NAME, "MRtrix general folder", wipe=False)
    check_output_folders(RESPONSE_FOLDER_NAME, "MRtrix response folder", wipe=False)
    check_output_folders(FOD_FOLDER_NAME, "MRtrix fod folder", wipe=False)
    check_output_folders(FOD_NORM_FOLDER_NAME, "MRtrix fod norm folder", wipe=False)
    check_output_folders(REGISTRATION_FOLDER_NAME, "MRtrix reg folder", wipe=False)
    check_output_folders(PROB_TRACKING_FOLDER_NAME, "MRtrix probabilistic tracking folder", wipe=False)
    check_output_folders(CONNECTIVITY_FOLDER_NAME, "MRtrix connectivity folder", wipe=False)

    # Return the paths
    return (REGION_MRTRIX_PATH, GENERAL_FOLDER_NAME, RESPONSE_FOLDER_NAME, FOD_FOLDER_NAME, 
            FOD_NORM_FOLDER_NAME, REGISTRATION_FOLDER_NAME, PROB_TRACKING_FOLDER_NAME, CONNECTIVITY_FOLDER_NAME)

# Deine the main paths depending on type of data working with
def define_main_mrtrix_folders_for_each_data(FOLDER_TYPE, REGION_ID):

    # Get the general paths
    (BMINDS_DATA_FOLDER, BMINDS_OUTPUTS_DMRI_FOLDER, BMINDS_OUTPUTS_DMRI_BMA_FOLDER, BMINDS_OUTPUTS_INVIVO_BMA_FOLDER,
    BMINDS_OUTPUTS_EXVIVO_BMA_FOLDER, BMINDS_OUTPUTS_INJECTIONS_FOLDER, BMINDS_BMA_MAIN_FOLDER,
    BMINDS_BMA_INVIVO_FOLDER, BMINDS_BMA_EXVIVO_FOLDER, BMINDS_BMA_INVIVO_DWI_FOLDER, BMINDS_BMA_EXVIVO_DWI_FOLDER,
    BMINDS_CORE_FOLDER, BMINDS_DWI_FOLDER, BMINDS_METADATA_FOLDER, BMINDS_TEMPLATES_FOLDER,
    BMINDS_ATLAS_FOLDER, BMINDS_STPT_TEMPLATE_FOLDER, BMINDS_TRANSFORMS_FOLDER, BMINDS_INJECTIONS_FOLDER,
    BMINDS_UNZIPPED_DWI_FOLDER, BMINDS_UNZIPPED_DWI_RESIZED_FOLDER,
    MAIN_MRTRIX_FOLDER_DMRI, MAIN_MRTRIX_FOLDER_BMA_DMRI_INVIVO, MAIN_MRTRIX_FOLDER_BMA_DMRI_EXVIVO,
    MAIN_MRTRIX_FOLDER_INJECTIONS) = get_main_paths(hpc=True)

    # Grab the folder paths depending on type
    if FOLDER_TYPE == "BMCR":
        (REGION_MRTRIX_PATH, GENERAL_FOLDER_NAME, RESPONSE_FOLDER_NAME, FOD_FOLDER_NAME, 
        FOD_NORM_FOLDER_NAME, REGISTRATION_FOLDER_NAME, PROB_TRACKING_FOLDER_NAME, 
        CONNECTIVITY_FOLDER_NAME) = main_mrtrix_folder_paths(MAIN_MRTRIX_FOLDER_DMRI, REGION_ID)
    elif FOLDER_TYPE == "BMA_INVIVO":
        (REGION_MRTRIX_PATH, GENERAL_FOLDER_NAME, RESPONSE_FOLDER_NAME, FOD_FOLDER_NAME, 
        FOD_NORM_FOLDER_NAME, REGISTRATION_FOLDER_NAME, PROB_TRACKING_FOLDER_NAME, 
        CONNECTIVITY_FOLDER_NAME) = main_mrtrix_folder_paths(MAIN_MRTRIX_FOLDER_BMA_DMRI_INVIVO, REGION_ID)
    elif FOLDER_TYPE == "BMA_EXVIVO":
        (REGION_MRTRIX_PATH, GENERAL_FOLDER_NAME, RESPONSE_FOLDER_NAME, FOD_FOLDER_NAME, 
        FOD_NORM_FOLDER_NAME, REGISTRATION_FOLDER_NAME, PROB_TRACKING_FOLDER_NAME, 
        CONNECTIVITY_FOLDER_NAME) = main_mrtrix_folder_paths(MAIN_MRTRIX_FOLDER_BMA_DMRI_EXVIVO, REGION_ID)
    # Error
    else:
        print("ERROR: FOLDER_TYPE must be either BMCR, BMA_INVIVO, or BMA_EXVIVO")
        sys.exit(1)

    # Return the paths
    return (REGION_MRTRIX_PATH, GENERAL_FOLDER_NAME, RESPONSE_FOLDER_NAME, FOD_FOLDER_NAME, 
            FOD_NORM_FOLDER_NAME, REGISTRATION_FOLDER_NAME, PROB_TRACKING_FOLDER_NAME, 
            CONNECTIVITY_FOLDER_NAME)

# Define MRtrix general paths
def get_mrtrix_general_paths(REGION_ID, DWI_FILES, FOLDER_TYPE):

    # Get the folder names
    (REGION_MRTRIX_PATH, GENERAL_FOLDER_NAME, RESPONSE_FOLDER_NAME, FOD_FOLDER_NAME, 
    FOD_NORM_FOLDER_NAME, REGISTRATION_FOLDER_NAME, PROB_TRACKING_FOLDER_NAME, 
    CONNECTIVITY_FOLDER_NAME) = define_main_mrtrix_folders_for_each_data(FOLDER_TYPE, REGION_ID)

    # Get the input MIF path
    NEEDED_FILES = ["dwi_mif"]
    INPUT_MIF_PATH = extract_from_input_list(DWI_FILES, NEEDED_FILES, "dwi")["dwi_mif"].replace(".mif", "")

    # DWI nii -> mif + mask filepaths
    MASK_MIF_PATH = os.path.join(GENERAL_FOLDER_NAME, "{}_mask_mif".format(REGION_ID))
    MASK_NII_PATH = os.path.join(GENERAL_FOLDER_NAME, "{}_mask_nii".format(REGION_ID))

    # Return the paths
    return (INPUT_MIF_PATH, MASK_MIF_PATH, MASK_NII_PATH)
    
# Define MRTrix clean paths
def get_mrtrix_clean_paths(REGION_ID, FOLDER_TYPE):

    # Get the folder names  
    (REGION_MRTRIX_PATH, GENERAL_FOLDER_NAME, RESPONSE_FOLDER_NAME, FOD_FOLDER_NAME, 
    FOD_NORM_FOLDER_NAME, REGISTRATION_FOLDER_NAME, PROB_TRACKING_FOLDER_NAME, 
    CONNECTIVITY_FOLDER_NAME) = define_main_mrtrix_folders_for_each_data(FOLDER_TYPE, REGION_ID)

    # Define skull stripping path as empty - only has a path if it's BMA_INVIVO
    SKULL_STRIP_PATH = ""
    SKULL_STRIP_MIF_PATH = ""
    
    # Depending on if it's BMA or not, we either do skull stripping or not
    if "BMA_INVIVO" == FOLDER_TYPE:
        DWI_B0_PATH = os.path.join(GENERAL_FOLDER_NAME, "{}_b0".format(REGION_ID))
        DWI_B0_NII = os.path.join(GENERAL_FOLDER_NAME, "{}_b0.nii.gz".format(REGION_ID))
        SKULL_STRIP_PATH = os.path.join(GENERAL_FOLDER_NAME, "{}_skullstrip".format(REGION_ID))
        SKULL_STRIP_MIF_PATH = os.path.join(GENERAL_FOLDER_NAME, "{}_skullstrip_mif".format(REGION_ID))
    
    # Define the denoising, eddy correction, and bias correction paths
    DWI_DENOISE_PATH = os.path.join(GENERAL_FOLDER_NAME, "{}_denoise".format(REGION_ID))
    DWI_NOISE_PATH = os.path.join(GENERAL_FOLDER_NAME, "{}_noise".format(REGION_ID))
    DWI_EDDY_PATH = os.path.join(GENERAL_FOLDER_NAME, "{}_eddy".format(REGION_ID))
    DWI_CLEAN_MIF_PATH = os.path.join(GENERAL_FOLDER_NAME, "{}_clean_mif".format(REGION_ID))
    DWI_CLEAN_MASK_PATH = os.path.join(GENERAL_FOLDER_NAME, "{}_clean_mask".format(REGION_ID))
    DWI_CLEAN_MASK_NII_PATH = os.path.join(GENERAL_FOLDER_NAME, "{}_clean_mask_nii".format(REGION_ID))
    DWI_CLEAN_NII_PATH = os.path.join(GENERAL_FOLDER_NAME, "{}_clean_nii".format(REGION_ID))
    DWI_CLEAN_BVEC_PATH = os.path.join(GENERAL_FOLDER_NAME, "{}_clean_bvec".format(REGION_ID))
    DWI_CLEAN_BVAL_PATH = os.path.join(GENERAL_FOLDER_NAME, "{}_clean_bval".format(REGION_ID))

    # Return the paths
    return (DWI_B0_PATH, DWI_B0_NII, SKULL_STRIP_PATH, SKULL_STRIP_MIF_PATH, DWI_DENOISE_PATH, DWI_NOISE_PATH, 
            DWI_EDDY_PATH, DWI_CLEAN_MIF_PATH, DWI_CLEAN_MASK_PATH, DWI_CLEAN_MASK_NII_PATH, DWI_CLEAN_NII_PATH, 
            DWI_CLEAN_BVEC_PATH, DWI_CLEAN_BVAL_PATH)

# Define MRTrix FOD paths
def get_mrtrix_fod_paths(REGION_ID, FOLDER_TYPE):
    
    # Get the folder names
    (REGION_MRTRIX_PATH, GENERAL_FOLDER_NAME, RESPONSE_FOLDER_NAME, FOD_FOLDER_NAME, 
    FOD_NORM_FOLDER_NAME, REGISTRATION_FOLDER_NAME, PROB_TRACKING_FOLDER_NAME, 
    CONNECTIVITY_FOLDER_NAME) = define_main_mrtrix_folders_for_each_data(FOLDER_TYPE, REGION_ID)

    # Finding the response function paths
    RESPONSE_WM_PATH = os.path.join(RESPONSE_FOLDER_NAME, "{}_wm".format(REGION_ID))
    RESPONSE_GM_PATH = os.path.join(RESPONSE_FOLDER_NAME, "{}_gm".format(REGION_ID))
    RESPONSE_CSF_PATH = os.path.join(RESPONSE_FOLDER_NAME, "{}_csf".format(REGION_ID))
    RESPONSE_VOXEL_PATH = os.path.join(RESPONSE_FOLDER_NAME, "{}_response_voxels".format(REGION_ID))
    # Finding the fiber orientation distributions (fODs)
    WM_FOD_PATH = os.path.join(FOD_FOLDER_NAME, "{}_wmfod".format(REGION_ID))
    GM_FOD_PATH = os.path.join(FOD_FOLDER_NAME, "{}_gmfod".format(REGION_ID))
    CSF_FOD_PATH = os.path.join(FOD_FOLDER_NAME, "{}_csffod".format(REGION_ID))
    VF_FOD_PATH = os.path.join(FOD_FOLDER_NAME, "{}_vf".format(REGION_ID))
    # Normalizing the fiber orientation distributions (fODs)
    WM_FOD_NORM_PATH = os.path.join(FOD_NORM_FOLDER_NAME, "{}_wmfod_norm".format(REGION_ID))
    GM_FOD_NORM_PATH = os.path.join(FOD_NORM_FOLDER_NAME, "{}_gmfod_norm".format(REGION_ID))
    CSF_FOD_NORM_PATH = os.path.join(FOD_NORM_FOLDER_NAME, "{}_csffod_norm".format(REGION_ID))
    
    # Return the paths
    return (RESPONSE_WM_PATH, RESPONSE_GM_PATH, RESPONSE_CSF_PATH, RESPONSE_VOXEL_PATH,
            WM_FOD_PATH, GM_FOD_PATH, CSF_FOD_PATH, VF_FOD_PATH, WM_FOD_NORM_PATH, GM_FOD_NORM_PATH, CSF_FOD_NORM_PATH)

# Define DWI registration path to the STPT template
def get_STPT_registration_paths(REGION_ID, FOLDER_TYPE):

    # Get the folder names
    (REGION_MRTRIX_PATH, GENERAL_FOLDER_NAME, RESPONSE_FOLDER_NAME, FOD_FOLDER_NAME,
    FOD_NORM_FOLDER_NAME, REGISTRATION_FOLDER_NAME, PROB_TRACKING_FOLDER_NAME,
    CONNECTIVITY_FOLDER_NAME) = define_main_mrtrix_folders_for_each_data(FOLDER_TYPE, REGION_ID)

    # Define the DWI registration folder
    DWI_REG_FOLDER = os.path.join(REGISTRATION_FOLDER_NAME, "{}_dwi_registration".format(REGION_ID))

    # Define the registration paths
    DWI_MAP_MAT = os.path.join(DWI_REG_FOLDER, "{}_dwi2stpt_fsl".format(REGION_ID))
    DWI_CONVERT_INV = os.path.join(DWI_REG_FOLDER, "{}_dwi2stpt_mrtrix".format(REGION_ID))
    DWI_REG_PATH = os.path.join(DWI_REG_FOLDER, "{}_dwi_reg".format(REGION_ID))
    
    # Define a mask for the final DWI
    DWI_MASK_PATH = os.path.join(DWI_REG_FOLDER, "{}_dwi_mask".format(REGION_ID))

    # Return the paths
    return (DWI_REG_FOLDER, DWI_MAP_MAT, DWI_CONVERT_INV, DWI_REG_PATH, DWI_MASK_PATH)


# Define MRtrix Registration paths
def get_atlas_registration_paths(REGION_ID, ATLAS_NEEDED_PATH, FOLDER_TYPE):
    
    # Get the folder names
    (REGION_MRTRIX_PATH, GENERAL_FOLDER_NAME, RESPONSE_FOLDER_NAME, FOD_FOLDER_NAME, 
    FOD_NORM_FOLDER_NAME, REGISTRATION_FOLDER_NAME, PROB_TRACKING_FOLDER_NAME, 
    CONNECTIVITY_FOLDER_NAME) = define_main_mrtrix_folders_for_each_data(FOLDER_TYPE, REGION_ID)

    # Getting the name of the atlas without .nii.gz
    ATLAS_NAME = ATLAS_NEEDED_PATH["atlas"].split("/")[-1].split(".")[0]

    # Define the atlas registration folder
    ATLAS_REG_FOLDER = os.path.join(REGISTRATION_FOLDER_NAME, "{}_atlas_registration".format(ATLAS_NAME))

    # Convert atlas nii to mif, then register/map it to the B0 space of DWI
    DWI_B0_PATH = os.path.join(ATLAS_REG_FOLDER, "{}_b0".format(REGION_ID))
    DWI_B0_NII = os.path.join(ATLAS_REG_FOLDER, "{}_b0.nii.gz".format(REGION_ID))

    ATLAS_DWI_MAP_MAT = os.path.join(ATLAS_REG_FOLDER, "{}_atlas2dwi_fsl".format(REGION_ID))
    ATLAS_DWI_CONVERT_INV = os.path.join(ATLAS_REG_FOLDER, "{}_atlas2dwi_mrtrix".format(REGION_ID))
    ATLAS_REG_PATH = os.path.join(ATLAS_REG_FOLDER, "{}_atlasreg".format(REGION_ID))
    
    # Get the mif path
    ATLAS_MIF_PATH = os.path.join(ATLAS_REG_FOLDER, "{}_atlas_mif".format(ATLAS_NAME))

    # Return the paths
    return (ATLAS_REG_FOLDER, DWI_B0_PATH, DWI_B0_NII, ATLAS_DWI_MAP_MAT, ATLAS_DWI_CONVERT_INV, ATLAS_REG_PATH, 
            ATLAS_MIF_PATH)

# Define MRtrix probabilistic tracking paths
def get_mrtrix_probtrack_paths(REGION_ID, FOLDER_TYPE):

    # Get the folder names
    (REGION_MRTRIX_PATH, GENERAL_FOLDER_NAME, RESPONSE_FOLDER_NAME, FOD_FOLDER_NAME, 
    FOD_NORM_FOLDER_NAME, REGISTRATION_FOLDER_NAME, PROB_TRACKING_FOLDER_NAME, 
    CONNECTIVITY_FOLDER_NAME) = define_main_mrtrix_folders_for_each_data(FOLDER_TYPE, REGION_ID)

    # Mask for streamline seeding paths and probabilistic tractography path
    GM_WM_SEED_PATH = os.path.join(PROB_TRACKING_FOLDER_NAME, "{}_gmwmseed".format(REGION_ID))
    TRACT_TCK_PATH = os.path.join(PROB_TRACKING_FOLDER_NAME, "{}_tract".format(REGION_ID))

    # Return the paths
    return (GM_WM_SEED_PATH, TRACT_TCK_PATH)

# Define MRtrix connectome paths
def get_mrtrix_connectome_paths(REGION_ID, FOLDER_TYPE):

    # Get the folder names
    (REGION_MRTRIX_PATH, GENERAL_FOLDER_NAME, RESPONSE_FOLDER_NAME, FOD_FOLDER_NAME, 
    FOD_NORM_FOLDER_NAME, REGISTRATION_FOLDER_NAME, PROB_TRACKING_FOLDER_NAME, 
    CONNECTIVITY_FOLDER_NAME) = define_main_mrtrix_folders_for_each_data(FOLDER_TYPE, REGION_ID)

    # Connectivity matrix path
    CONNECTIVITY_PROB_PATH = os.path.join(CONNECTIVITY_FOLDER_NAME, "{}_prob_connectivity".format(REGION_ID))

    # Return the paths
    return (CONNECTIVITY_PROB_PATH)
