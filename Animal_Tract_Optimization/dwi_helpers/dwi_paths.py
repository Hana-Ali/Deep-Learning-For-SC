import sys
sys.path.append("..")
from py_helpers.general_helpers import *
from py_helpers.shared_helpers import *

# ----------------------------------------------------- PATH DEFINITIONS ----------------------------------------------------- #

# Define MRTRIX folder paths
def main_mrtrix_folder_paths(REGION_ID):
    
    # Get the general paths
    (BMINDS_DATA_FOLDER, BMINDS_OUTPUTS_DMRI_FOLDER, BMINDS_OUTPUTS_INJECTIONS_FOLDER, BMINDS_CORE_FOLDER,
    BMINDS_DWI_FOLDER, BMINDS_METADATA_FOLDER, BMINDS_TEMPLATES_FOLDER, BMINDS_ATLAS_FOLDER, BMINDS_STPT_TEMPLATE_FOLDER, 
    BMINDS_TRANSFORMS_FOLDER, BMINDS_INJECTIONS_FOLDER, BMINDS_UNZIPPED_DWI_FOLDER, BMINDS_UNZIPPED_DWI_RESIZED_FOLDER,
    MAIN_MRTRIX_FOLDER_DMRI, MAIN_MRTRIX_FOLDER_INJECTIONS) = get_main_paths(hpc=True)

    # Create the region's main folder - don't wipe as it'll be invoked more than once
    REGION_MRTRIX_PATH = os.path.join(MAIN_MRTRIX_FOLDER_DMRI, REGION_ID)
    check_output_folders(REGION_MRTRIX_PATH, "REGION MRTRIX PATH", wipe=False)
    
    # Because there's a lot of commands, we define extra folder paths here. DON'T WIPE as it'll be used by other functions
    GENERAL_FOLDER_NAME = os.path.join(REGION_MRTRIX_PATH, "general")
    RESPONSE_FOLDER_NAME = os.path.join(REGION_MRTRIX_PATH, "response")
    FOD_FOLDER_NAME = os.path.join(REGION_MRTRIX_PATH, "fod")
    FOD_NORM_FOLDER_NAME = os.path.join(REGION_MRTRIX_PATH, "fod_norm")
    # T1_REG_FOLDER_NAME = os.path.join(REGION_MRTRIX_PATH, "t1_and_Fivett_reg")
    ATLAS_REG_FOLDER_NAME = os.path.join(REGION_MRTRIX_PATH, "atlas_reg")
    PROB_TRACKING_FOLDER_NAME = os.path.join(REGION_MRTRIX_PATH, "prob_tracking")
    # GLOBAL_TRACKING_FOLDER_NAME = os.path.join(REGION_MRTRIX_PATH, "global_tracking")
    CONNECTIVITY_FOLDER_NAME = os.path.join(REGION_MRTRIX_PATH, "connectivity")

    # Check the output folders
    check_output_folders(GENERAL_FOLDER_NAME, "MRtrix general folder", wipe=False)
    check_output_folders(RESPONSE_FOLDER_NAME, "MRtrix response folder", wipe=False)
    check_output_folders(FOD_FOLDER_NAME, "MRtrix fod folder", wipe=False)
    check_output_folders(FOD_NORM_FOLDER_NAME, "MRtrix fod norm folder", wipe=False)
    # check_output_folders(T1_REG_FOLDER_NAME, "MRtrix t1 and Fivett reg folder", wipe=False)
    check_output_folders(ATLAS_REG_FOLDER_NAME, "MRtrix atlas reg folder", wipe=False)
    check_output_folders(PROB_TRACKING_FOLDER_NAME, "MRtrix probabilistic tracking folder", wipe=False)
    # check_output_folders(GLOBAL_TRACKING_FOLDER_NAME, "MRtrix global tracking folder", wipe=False)
    check_output_folders(CONNECTIVITY_FOLDER_NAME, "MRtrix connectivity folder", wipe=False)

    # Return the paths
    return (REGION_MRTRIX_PATH, GENERAL_FOLDER_NAME, RESPONSE_FOLDER_NAME, FOD_FOLDER_NAME, 
            FOD_NORM_FOLDER_NAME, ATLAS_REG_FOLDER_NAME, PROB_TRACKING_FOLDER_NAME, CONNECTIVITY_FOLDER_NAME)

# Define MRtrix general paths
def get_mrtrix_general_paths(REGION_ID, DWI_FILES):
    
    # Get the folder names
    (REGION_MRTRIX_PATH, GENERAL_FOLDER_NAME, RESPONSE_FOLDER_NAME, FOD_FOLDER_NAME, 
        FOD_NORM_FOLDER_NAME, ATLAS_REG_FOLDER_NAME, PROB_TRACKING_FOLDER_NAME, 
        CONNECTIVITY_FOLDER_NAME) = main_mrtrix_folder_paths(REGION_ID)
    # Get the input MIF path
    NEEDED_FILES = ["dwi_mif"]
    INPUT_MIF_PATH = extract_from_input_list(DWI_FILES, NEEDED_FILES, "dwi")["dwi_mif"].replace(".mif", "")

    # DWI nii -> mif + mask filepaths
    MASK_MIF_PATH = os.path.join(GENERAL_FOLDER_NAME, "{}_mask_mif".format(REGION_ID))
    MASK_NII_PATH = os.path.join(GENERAL_FOLDER_NAME, "{}_mask_nii".format(REGION_ID))

    # Return the paths
    return (INPUT_MIF_PATH, MASK_MIF_PATH, MASK_NII_PATH)
    
# Define MRTrix FOD paths
def get_mrtrix_fod_paths(REGION_ID):
    
    # Get the folder names
    (REGION_MRTRIX_PATH, GENERAL_FOLDER_NAME, RESPONSE_FOLDER_NAME, FOD_FOLDER_NAME, 
        FOD_NORM_FOLDER_NAME, ATLAS_REG_FOLDER_NAME, PROB_TRACKING_FOLDER_NAME, 
        CONNECTIVITY_FOLDER_NAME) = main_mrtrix_folder_paths(REGION_ID)

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

# Define MRtrix Registration paths
def get_mrtrix_registration_paths(REGION_ID, ATLAS_NEEDED_PATH):
    
    # Get the folder names
    (REGION_MRTRIX_PATH, GENERAL_FOLDER_NAME, RESPONSE_FOLDER_NAME, FOD_FOLDER_NAME, 
        FOD_NORM_FOLDER_NAME, ATLAS_REG_FOLDER_NAME, PROB_TRACKING_FOLDER_NAME, 
            CONNECTIVITY_FOLDER_NAME) = main_mrtrix_folder_paths(REGION_ID)

    # Convert atlas nii to mif, then register/map it to the B0 space of DWI
    DWI_B0_PATH = os.path.join(ATLAS_REG_FOLDER_NAME, "{}_b0".format(REGION_ID))
    DWI_B0_NII = os.path.join(ATLAS_REG_FOLDER_NAME, "{}_b0.nii.gz".format(REGION_ID))

    ATLAS_DWI_MAP_MAT = os.path.join(ATLAS_REG_FOLDER_NAME, "{}_atlas2dwi_fsl".format(REGION_ID))
    ATLAS_DWI_CONVERT_INV = os.path.join(ATLAS_REG_FOLDER_NAME, "{}_atlas2dwi_mrtrix".format(REGION_ID))
    ATLAS_REG_PATH = os.path.join(ATLAS_REG_FOLDER_NAME, "{}_atlasreg".format(REGION_ID))
    # Getting the name of the atlas without .nii.gz
    print("ATLAS_NEEDED_PATH IN PATHS: {}".format(ATLAS_NEEDED_PATH))
    ATLAS_NAME = ATLAS_NEEDED_PATH["atlas"].split("/")[-1].split(".")[0]
    ATLAS_MIF_PATH = os.path.join(ATLAS_REG_FOLDER_NAME, "{}_atlas_mif".format(ATLAS_NAME))

    # Return the paths
    return (DWI_B0_PATH, DWI_B0_NII, ATLAS_DWI_MAP_MAT, ATLAS_DWI_CONVERT_INV, ATLAS_REG_PATH, ATLAS_MIF_PATH)

# Define MRtrix probabilistic tracking paths
def get_mrtrix_probtrack_paths(REGION_ID):

    # Get the folder names
    (REGION_MRTRIX_PATH, GENERAL_FOLDER_NAME, RESPONSE_FOLDER_NAME, FOD_FOLDER_NAME, 
        FOD_NORM_FOLDER_NAME, ATLAS_REG_FOLDER_NAME, PROB_TRACKING_FOLDER_NAME, 
            CONNECTIVITY_FOLDER_NAME) = main_mrtrix_folder_paths(REGION_ID)

    # Mask for streamline seeding paths and probabilistic tractography path
    GM_WM_SEED_PATH = os.path.join(PROB_TRACKING_FOLDER_NAME, "{}_gmwmseed".format(REGION_ID))
    TRACT_TCK_PATH = os.path.join(PROB_TRACKING_FOLDER_NAME, "{}_tract".format(REGION_ID))

    # Return the paths
    return (GM_WM_SEED_PATH, TRACT_TCK_PATH)

# Define MRtrix connectome paths
def get_mrtrix_connectome_paths(REGION_ID):

    # Get the folder names
    (REGION_MRTRIX_PATH, GENERAL_FOLDER_NAME, RESPONSE_FOLDER_NAME, FOD_FOLDER_NAME, 
        FOD_NORM_FOLDER_NAME, ATLAS_REG_FOLDER_NAME, PROB_TRACKING_FOLDER_NAME, 
            CONNECTIVITY_FOLDER_NAME) = main_mrtrix_folder_paths(REGION_ID)

    # Connectivity matrix path
    CONNECTIVITY_PROB_PATH = os.path.join(CONNECTIVITY_FOLDER_NAME, "{}_prob_connectivity".format(REGION_ID))

    # Return the paths
    return (CONNECTIVITY_PROB_PATH)
