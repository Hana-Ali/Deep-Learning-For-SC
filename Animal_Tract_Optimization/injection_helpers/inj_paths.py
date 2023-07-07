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
    ATLAS_REG_FOLDER_NAME = os.path.join(MAIN_MRTRIX_FOLDER_INJECTIONS, "atlas_reg")
    COMBINED_TRACTS_FOLDER_NAME = os.path.join(MAIN_MRTRIX_FOLDER_INJECTIONS, "combined_tracts")
    check_output_folders(ATLAS_REG_FOLDER_NAME, "MRtrix atlas reg folder", wipe=False)
    check_output_folders(COMBINED_TRACTS_FOLDER_NAME, "MRtrix combined tracts folder", wipe=False)

    # Return the paths
    return (ATLAS_REG_FOLDER_NAME, COMBINED_TRACTS_FOLDER_NAME)

# Define MRTRIX region-specific folder paths
def region_mrtrix_folder_paths(REGION_ID):
    
    # Get the general paths
    (BMINDS_DATA_FOLDER, BMINDS_OUTPUTS_DMRI_FOLDER, BMINDS_OUTPUTS_INJECTIONS_FOLDER, BMINDS_CORE_FOLDER,
        BMINDS_DWI_FOLDER, BMINDS_METADATA_FOLDER, BMINDS_TEMPLATES_FOLDER, BMINDS_ATLAS_FOLDER, BMINDS_STPT_TEMPLATE_FOLDER, 
        BMINDS_TRANSFORMS_FOLDER, BMINDS_INJECTIONS_FOLDER, BMINDS_UNZIPPED_DWI_FOLDER, MAIN_MRTRIX_FOLDER_DMRI, 
        MAIN_MRTRIX_FOLDER_INJECTIONS) = get_main_paths(hpc=False)
    
    # Create the region folder
    REGION_MRTRIX_PATH = os.path.join(MAIN_MRTRIX_FOLDER_INJECTIONS, REGION_ID)
    check_output_folders(REGION_MRTRIX_PATH, "REGION MRTRIX PATH", wipe=False)

    # Return the paths
    return (REGION_MRTRIX_PATH)

# Function to define the atlas registration USING ANTS AND TRANSFORMATION MATRIX files
def get_mrtrix_atlas_reg_paths_ants():

    # Get the folder names
    (ATLAS_REG_FOLDER_NAME, COMBINED_TRACTS_FOLDER_NAME) = main_mrtrix_folder_paths()

    # Immediately transform with the transformation matrix
    ATLAS_REG_PATH = os.path.join(ATLAS_REG_FOLDER_NAME, "atlasreg_ANTS")
    ATLAS_REG_MIF_PATH = os.path.join(ATLAS_REG_FOLDER_NAME, "atlasreg_ANTS_mif")

    # Return the paths
    return (ATLAS_REG_PATH, ATLAS_REG_MIF_PATH)

# Function to define the combined tracts path
def get_combined_tracts_path():

    # Get the folder names and paths
    (ATLAS_REG_FOLDER_NAME, COMBINED_TRACTS_FOLDER_NAME) = main_mrtrix_folder_paths()

    # Define the combined tracts path
    COMBINED_TRACTS_PATH = os.path.join(COMBINED_TRACTS_FOLDER_NAME, "combined_tracts")

    # Return the path
    return (COMBINED_TRACTS_PATH)



