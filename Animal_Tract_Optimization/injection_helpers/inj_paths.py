import os
import sys
sys.path.append("..")
from py_helpers.general_helpers import *
from py_helpers.shared_helpers import *


# Define the MRTRIX folder paths
def main_mrtrix_folder_paths():

    # Get the general paths
    (BMINDS_DATA_FOLDER, BMINDS_OUTPUTS_DMRI_FOLDER, BMINDS_OUTPUTS_INJECTIONS_FOLDER, BMINDS_CORE_FOLDER,
        BMINDS_DWI_FOLDER, BMINDS_METADATA_FOLDER, BMINDS_TEMPLATES_FOLDER, BMINDS_ATLAS_FOLDER, BMINDS_STPT_TEMPLATE_FOLDER, 
        BMINDS_TRANSFORMS_FOLDER, BMINDS_INJECTIONS_FOLDER, BMINDS_UNZIPPED_DWI_FOLDER, MAIN_MRTRIX_FOLDER_DMRI, 
        MAIN_MRTRIX_FOLDER_INJECTIONS) = get_main_paths(hpc=False)

    # Create the region's main folder - don't wipe as it'll be invoked more than once
    GENERAL_MRTRIX_FOLDER = os.path.join(MAIN_MRTRIX_FOLDER_INJECTIONS, "general_mrtrix")
    SPECIFIC_MRTRIX_FOLDER = os.path.join(MAIN_MRTRIX_FOLDER_INJECTIONS, "specific_mrtrix")
    ATLAS_REG_FOLDER_NAME = os.path.join(GENERAL_MRTRIX_FOLDER, "atlas_reg")
    COMBINED_TRACTS_FOLDER_NAME = os.path.join(GENERAL_MRTRIX_FOLDER, "combined_tracts")
    COMBINED_INJECTIONS_FOLDER_NAME = os.path.join(GENERAL_MRTRIX_FOLDER, "combined_injections")
    COMBINED_ATLAS_INJECTIONS_FOLDER_NAME = os.path.join(GENERAL_MRTRIX_FOLDER, "combined_atlas_injections")
    COMBINED_CONNECTOME_FOLDER_NAME = os.path.join(GENERAL_MRTRIX_FOLDER, "combined_connectome")

    INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME = os.path.join(GENERAL_MRTRIX_FOLDER, "individual_rois_from_atlas")
    INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME = os.path.join(INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, "nifti")
    INDIVIDUAL_ROIS_MIF_FOLDER_NAME = os.path.join(INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, "mif")
    INJECTION_ROI_FOLDER_NAME = os.path.join(SPECIFIC_MRTRIX_FOLDER, "injection_ROI_combined")
    INJECTION_ROI_CONNECTOMES_FOLDER_NAME = os.path.join(SPECIFIC_MRTRIX_FOLDER, "injection_ROI_connectomes")
    check_output_folders(GENERAL_MRTRIX_FOLDER, "MRtrix general folder", wipe=False)
    check_output_folders(SPECIFIC_MRTRIX_FOLDER, "MRtrix specific folder", wipe=False)
    check_output_folders(ATLAS_REG_FOLDER_NAME, "MRtrix atlas reg folder", wipe=False)
    check_output_folders(COMBINED_TRACTS_FOLDER_NAME, "MRtrix combined tracts folder", wipe=False)
    check_output_folders(COMBINED_INJECTIONS_FOLDER_NAME, "MRtrix combined injections folder", wipe=False)
    check_output_folders(COMBINED_ATLAS_INJECTIONS_FOLDER_NAME, "MRtrix combined atlas injections folder", wipe=False)
    check_output_folders(COMBINED_CONNECTOME_FOLDER_NAME, "MRtrix combined connectome folder", wipe=False)

    check_output_folders(INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, "MRtrix individual rois from atlas folder", wipe=False)
    check_output_folders(INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME, "MRtrix individual rois from atlas nifti folder", wipe=False)
    check_output_folders(INDIVIDUAL_ROIS_MIF_FOLDER_NAME, "MRtrix individual rois from atlas to mif folder", wipe=False)
    check_output_folders(INJECTION_ROI_FOLDER_NAME, "MRtrix injection and ROI combination folder", wipe=False)
    check_output_folders(INJECTION_ROI_CONNECTOMES_FOLDER_NAME, "MRtrix injection and ROI connectomes folder", wipe=False)

    # Return the paths
    return (GENERAL_MRTRIX_FOLDER, SPECIFIC_MRTRIX_FOLDER, ATLAS_REG_FOLDER_NAME, COMBINED_TRACTS_FOLDER_NAME, 
            COMBINED_INJECTIONS_FOLDER_NAME, COMBINED_ATLAS_INJECTIONS_FOLDER_NAME, COMBINED_CONNECTOME_FOLDER_NAME,
            INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME, INDIVIDUAL_ROIS_MIF_FOLDER_NAME, 
            INJECTION_ROI_FOLDER_NAME, INJECTION_ROI_CONNECTOMES_FOLDER_NAME)

# Define MRTRIX region-specific folder paths
def region_mrtrix_folder_paths(REGION_ID):
    
    # Get the main paths
    (GENERAL_MRTRIX_FOLDER, SPECIFIC_MRTRIX_FOLDER, ATLAS_REG_FOLDER_NAME, COMBINED_TRACTS_FOLDER_NAME, 
        COMBINED_INJECTIONS_FOLDER_NAME, COMBINED_ATLAS_INJECTIONS_FOLDER_NAME, COMBINED_CONNECTOME_FOLDER_NAME,
        INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME, INDIVIDUAL_ROIS_MIF_FOLDER_NAME, 
        INJECTION_ROI_FOLDER_NAME, INJECTION_ROI_CONNECTOMES_FOLDER_NAME) = main_mrtrix_folder_paths()
    
    # Create the region folder
    REGION_MRTRIX_FOLDER = os.path.join(SPECIFIC_MRTRIX_FOLDER, REGION_ID)
    INJECTION_MIF_FOLDER = os.path.join(REGION_MRTRIX_FOLDER, "injection_mifs")
    check_output_folders(REGION_MRTRIX_FOLDER, "REGION MRTRIX PATH", wipe=False)
    check_output_folders(INJECTION_MIF_FOLDER, "INJECTION MIF PATH", wipe=False)

    # Return the paths
    return (REGION_MRTRIX_FOLDER, INJECTION_MIF_FOLDER)

# Function to define the atlas registration USING ANTS AND TRANSFORMATION MATRIX files
def get_mrtrix_atlas_reg_paths_ants():

    # Get the folder names
    (GENERAL_MRTRIX_FOLDER, SPECIFIC_MRTRIX_FOLDER, ATLAS_REG_FOLDER_NAME, COMBINED_TRACTS_FOLDER_NAME, 
        COMBINED_INJECTIONS_FOLDER_NAME, COMBINED_ATLAS_INJECTIONS_FOLDER_NAME, COMBINED_CONNECTOME_FOLDER_NAME,
        INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME, INDIVIDUAL_ROIS_MIF_FOLDER_NAME, 
        INJECTION_ROI_FOLDER_NAME, INJECTION_ROI_CONNECTOMES_FOLDER_NAME) = main_mrtrix_folder_paths()

    # Immediately transform with the transformation matrix
    ATLAS_REG_PATH = os.path.join(ATLAS_REG_FOLDER_NAME, "atlasreg_ANTS")
    ATLAS_REG_MIF_PATH = os.path.join(ATLAS_REG_FOLDER_NAME, "atlasreg_ANTS_mif")

    # Return the paths
    return (ATLAS_REG_PATH, ATLAS_REG_MIF_PATH)

# Function to define the combined tracts path
def get_combined_tracts_path():

    # Get the folder names and paths
    (GENERAL_MRTRIX_FOLDER, SPECIFIC_MRTRIX_FOLDER, ATLAS_REG_FOLDER_NAME, COMBINED_TRACTS_FOLDER_NAME, 
        COMBINED_INJECTIONS_FOLDER_NAME, COMBINED_ATLAS_INJECTIONS_FOLDER_NAME, COMBINED_CONNECTOME_FOLDER_NAME,
        INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME, INDIVIDUAL_ROIS_MIF_FOLDER_NAME, 
        INJECTION_ROI_FOLDER_NAME, INJECTION_ROI_CONNECTOMES_FOLDER_NAME) = main_mrtrix_folder_paths()

    # Define the combined tracts path
    COMBINED_TRACTS_PATH = os.path.join(COMBINED_TRACTS_FOLDER_NAME, "combined_tracts")

    # Return the path
    return (COMBINED_TRACTS_PATH)

# Function to define the combined injection path
def get_combined_injections_path():

    # Get the folder names and paths
    (GENERAL_MRTRIX_FOLDER, SPECIFIC_MRTRIX_FOLDER, ATLAS_REG_FOLDER_NAME, COMBINED_TRACTS_FOLDER_NAME, 
        COMBINED_INJECTIONS_FOLDER_NAME, COMBINED_ATLAS_INJECTIONS_FOLDER_NAME, COMBINED_CONNECTOME_FOLDER_NAME,
        INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME, INDIVIDUAL_ROIS_MIF_FOLDER_NAME, 
        INJECTION_ROI_FOLDER_NAME, INJECTION_ROI_CONNECTOMES_FOLDER_NAME) = main_mrtrix_folder_paths()

    # Define the combined tracts path
    COMBINED_INJECTIONS_PATH = os.path.join(COMBINED_INJECTIONS_FOLDER_NAME, "combined_injections")
    COMBINED_INJECTIONS_MIF_PATH = os.path.join(COMBINED_INJECTIONS_FOLDER_NAME, "combined_injections_mif")

    # Return the path
    return (COMBINED_INJECTIONS_PATH, COMBINED_INJECTIONS_MIF_PATH)

# Function to define the combined injection atlas path (BIG)
def get_combined_injection_atlas_path():

    # Get the folder names and paths
    (GENERAL_MRTRIX_FOLDER, SPECIFIC_MRTRIX_FOLDER, ATLAS_REG_FOLDER_NAME, COMBINED_TRACTS_FOLDER_NAME, 
        COMBINED_INJECTIONS_FOLDER_NAME, COMBINED_ATLAS_INJECTIONS_FOLDER_NAME, COMBINED_CONNECTOME_FOLDER_NAME,
        INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME, INDIVIDUAL_ROIS_MIF_FOLDER_NAME, 
        INJECTION_ROI_FOLDER_NAME, INJECTION_ROI_CONNECTOMES_FOLDER_NAME) = main_mrtrix_folder_paths()

    # Define the combined injection atlas path
    COMBINED_INJECTION_ATLAS_MIF_PATH = os.path.join(COMBINED_ATLAS_INJECTIONS_FOLDER_NAME, "combined_injection_atlas_mif")
    COMBINED_INJECTION_ATLAS_NII_PATH = os.path.join(COMBINED_ATLAS_INJECTIONS_FOLDER_NAME, "combined_injection_atlas_nii")

    # Return the path
    return (COMBINED_INJECTION_ATLAS_MIF_PATH, COMBINED_INJECTION_ATLAS_NII_PATH)

# Function to define the combined injection atlas connectome path
def get_combined_injection_atlas_connectome_path():

    # Get the folder names and paths
    (GENERAL_MRTRIX_FOLDER, SPECIFIC_MRTRIX_FOLDER, ATLAS_REG_FOLDER_NAME, COMBINED_TRACTS_FOLDER_NAME, 
        COMBINED_INJECTIONS_FOLDER_NAME, COMBINED_ATLAS_INJECTIONS_FOLDER_NAME, COMBINED_CONNECTOME_FOLDER_NAME,
        INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME, INDIVIDUAL_ROIS_MIF_FOLDER_NAME, 
        INJECTION_ROI_FOLDER_NAME, INJECTION_ROI_CONNECTOMES_FOLDER_NAME) = main_mrtrix_folder_paths()
    
    # Define the combibed connectome paths
    COMBINED_INJECTION_ATLAS_CONNECTOME_PATH = os.path.join(COMBINED_CONNECTOME_FOLDER_NAME, "combined_injection_atlas_connectome")

    # Return the path
    return (COMBINED_INJECTION_ATLAS_CONNECTOME_PATH)

# Function to define the injection mif path
def get_injection_mif_path(REGION_ID):
    
    # Get the folder names and paths
    (REGION_MRTRIX_FOLDER, INJECTION_MIF_FOLDER) = region_mrtrix_folder_paths(REGION_ID)

    # Define the injection mif path
    INJECTION_MIF_PATH = os.path.join(INJECTION_MIF_FOLDER, "{}_injection".format(REGION_ID))

    # Return the path
    return (INJECTION_MIF_PATH)

# Function to define the individual rois from atlas path
def get_individual_rois_from_atlas_path(ATLAS_STPT):
    
    # Get the folder names and paths
    (GENERAL_MRTRIX_FOLDER, SPECIFIC_MRTRIX_FOLDER, ATLAS_REG_FOLDER_NAME, COMBINED_TRACTS_FOLDER_NAME, 
        COMBINED_INJECTIONS_FOLDER_NAME, COMBINED_ATLAS_INJECTIONS_FOLDER_NAME, COMBINED_CONNECTOME_FOLDER_NAME,
        INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME, INDIVIDUAL_ROIS_MIF_FOLDER_NAME, 
        INJECTION_ROI_FOLDER_NAME, INJECTION_ROI_CONNECTOMES_FOLDER_NAME) = main_mrtrix_folder_paths()

    # Get the atlas and atlas labels path
    NEEDED_FILES_ATLAS = ["atlas", "atlas_label"]
    ATLAS_LABEL_NEEDED_PATH = extract_from_input_list(ATLAS_STPT, NEEDED_FILES_ATLAS, "atlas_stpt")

    # This holds all the atlas ROI paths
    INDIVIDUAL_ROIS_FROM_ATLAS_PATH = []

    # For every line in the atlas label file, extract the ROI name
    with open(ATLAS_LABEL_NEEDED_PATH["atlas_label"], "r") as atlas_label_file:
        for line in atlas_label_file:
            # Get the ROI number and name
            LINE_SPLIT = [splits for splits in line.split("\t") if splits]
            ROI_NUM = LINE_SPLIT[0]
            ROI_NAME = LINE_SPLIT[-1].replace('"', '').replace(" ", "_").replace("\n", "").replace("(", "").replace(")", "")
            # Get the atlas ROI path
            filename = "NUMBER_" + ROI_NUM + "_NAME_" + ROI_NAME
            ATLAS_ROI_PATH = os.path.join(INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME, filename)
            # Append the atlas ROI path to the list
            INDIVIDUAL_ROIS_FROM_ATLAS_PATH.append(ATLAS_ROI_PATH)

    # Return the path
    return (INDIVIDUAL_ROIS_FROM_ATLAS_PATH)

# Function to define the individual rois mif path
def get_individual_rois_mif_path(ATLAS_STPT):

    # Get the folder names and paths
    (GENERAL_MRTRIX_FOLDER, SPECIFIC_MRTRIX_FOLDER, ATLAS_REG_FOLDER_NAME, COMBINED_TRACTS_FOLDER_NAME, 
        COMBINED_INJECTIONS_FOLDER_NAME, COMBINED_ATLAS_INJECTIONS_FOLDER_NAME, COMBINED_CONNECTOME_FOLDER_NAME,
        INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME, INDIVIDUAL_ROIS_MIF_FOLDER_NAME, 
        INJECTION_ROI_FOLDER_NAME, INJECTION_ROI_CONNECTOMES_FOLDER_NAME) = main_mrtrix_folder_paths()
    
    # Grab all the ROI paths
    (INDIVIDUAL_ROIS_FROM_ATLAS_PATH) = get_individual_rois_from_atlas_path(ATLAS_STPT)

    # This holds all the atlas ROI mif paths
    INDIVIDUAL_ROIS_MIF_PATHS = []

    # For every nifti file, create the mif filename
    for nifti_file in INDIVIDUAL_ROIS_FROM_ATLAS_PATH:
        # Get the filename filename
        filename = nifti_file.split("/")[-1]
        # Create the mif path
        mif_path = os.path.join(INDIVIDUAL_ROIS_MIF_FOLDER_NAME, filename)
        # Append the mif path to the list
        INDIVIDUAL_ROIS_MIF_PATHS.append(mif_path)

    # Return the path
    return (INDIVIDUAL_ROIS_MIF_PATHS)

# Function to define the injection roi path
def get_injection_roi_path(REGION_ID, ROI_ID):
    
    # Get the folder names and paths
    (GENERAL_MRTRIX_FOLDER, SPECIFIC_MRTRIX_FOLDER, ATLAS_REG_FOLDER_NAME, COMBINED_TRACTS_FOLDER_NAME, 
        COMBINED_INJECTIONS_FOLDER_NAME, COMBINED_ATLAS_INJECTIONS_FOLDER_NAME, COMBINED_CONNECTOME_FOLDER_NAME,
        INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME, INDIVIDUAL_ROIS_MIF_FOLDER_NAME, 
        INJECTION_ROI_FOLDER_NAME, INJECTION_ROI_CONNECTOMES_FOLDER_NAME) = main_mrtrix_folder_paths()
    
    # Define the combined tracts path
    INJECTION_ROI_PATH = os.path.join(INJECTION_ROI_FOLDER_NAME, "INJ_{}_ROI_{}".format(REGION_ID, ROI_ID))

    # Return the path
    return (INJECTION_ROI_PATH)

# Function to define the injection roi connectome path
def get_injection_roi_connectome_path(REGION_ID, ROI_ID):

    # Get the folder names and paths
    (GENERAL_MRTRIX_FOLDER, SPECIFIC_MRTRIX_FOLDER, ATLAS_REG_FOLDER_NAME, COMBINED_TRACTS_FOLDER_NAME, 
        COMBINED_INJECTIONS_FOLDER_NAME, COMBINED_ATLAS_INJECTIONS_FOLDER_NAME, COMBINED_CONNECTOME_FOLDER_NAME,
        INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME, INDIVIDUAL_ROIS_MIF_FOLDER_NAME, 
        INJECTION_ROI_FOLDER_NAME, INJECTION_ROI_CONNECTOMES_FOLDER_NAME) = main_mrtrix_folder_paths()
    
    # Define the combibed connectome paths
    INJECTION_ROI_CONNECTOME_PATH = os.path.join(INJECTION_ROI_CONNECTOMES_FOLDER_NAME, "INJ_{}_ROI_{}_CONNECTOME".format(REGION_ID, ROI_ID))

    # Return the path
    return (INJECTION_ROI_CONNECTOME_PATH)
