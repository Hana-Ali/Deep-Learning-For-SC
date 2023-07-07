# We want to create the injection matrix that was in the paper
# To do this, we have many different tracts, and we want to concatenate them all together
# We want to do this for each injection site
import os
import sys
from py_helpers.general_helpers import *
from injection_helpers.inj_general import *
import numpy as np

# Main function
def main():
    # --------------- Get main folder paths, check inputs/outputs, unzip necessary --------------- #
    # Get the main folder paths
    hpc = False
    (BMINDS_DATA_FOLDER, BMINDS_OUTPUTS_DMRI_FOLDER, BMINDS_OUTPUTS_INJECTIONS_FOLDER, BMINDS_ATLAS_STPT_FOLDER, 
            BMINDS_CORE_FOLDER, BMINDS_DWI_FOLDER, BMINDS_METADATA_FOLDER, BMINDS_INJECTIONS_FOLDER, BMINDS_UNZIPPED_DWI_FOLDER, 
                MAIN_MRTRIX_FOLDER_DMRI, MAIN_MRTRIX_FOLDER_INJECTIONS) = get_main_paths(hpc)
    
    # Check the input folders
    check_input_folders(BMINDS_DATA_FOLDER, "BMINDS_DATA_FOLDER")
    check_input_folders(BMINDS_ATLAS_STPT_FOLDER, "BMINDS_ATLAS_STPT_FOLDER")
    check_input_folders(BMINDS_CORE_FOLDER, "BMINDS_CORE_FOLDER")
    check_input_folders(BMINDS_DWI_FOLDER, "BMINDS_DWI_FOLDER")
    check_input_folders(BMINDS_METADATA_FOLDER, "BMINDS_METADATA_FOLDER")
    check_input_folders(BMINDS_INJECTIONS_FOLDER, "BMINDS_INJECTIONS_FOLDER")

    # Check the output folders
    check_output_folders(BMINDS_OUTPUTS_INJECTIONS_FOLDER, "BMINDS_OUTPUTS_FOLDER", wipe=False)
    check_output_folders(BMINDS_UNZIPPED_DWI_FOLDER, "BMINDS_UNZIPPED_DWI_FOLDER", wipe=False)
    check_output_folders(MAIN_MRTRIX_FOLDER_INJECTIONS, "MAIN_MRTRIX_FOLDER_INJECTIONS", wipe=True)

    # --------------- Get the injection files --------------- #
    BMINDS_UNZIPPED_DWI_FILES = glob_files(BMINDS_UNZIPPED_DWI_FOLDER, "nii")
    BMINDS_BVAL_FILES = glob_files(BMINDS_DWI_FOLDER, "bval")
    BMINDS_BVEC_FILES = glob_files(BMINDS_DWI_FOLDER, "bvec")
    BMINDS_STREAMLINE_FILES = glob_files(BMINDS_METADATA_FOLDER, "tck")
    BMINDS_INJECTION_FILES = glob_files(BMINDS_INJECTIONS_FOLDER, "nii.gz")
    BMINDS_ATLAS_STPT_FILES = glob_files(BMINDS_ATLAS_STPT_FOLDER, "nii.gz") # Gets both the atlas and stpt files, need to separate

    # Get the atlas and stpt files - separate from the mix above
    BMINDS_ATLAS_FILE = [file for file in BMINDS_ATLAS_STPT_FILES if "cortical_atlas" in file]
    BMINDS_STPT_FILE = [file for file in BMINDS_ATLAS_STPT_FILES if "STPT_template" in file]

    # Check the globbed files
    check_globbed_files(BMINDS_UNZIPPED_DWI_FILES, "BMINDS_UNZIPPED_DWI_FILES")
    check_globbed_files(BMINDS_BVAL_FILES, "BMINDS_BVAL_FILES")
    check_globbed_files(BMINDS_BVEC_FILES, "BMINDS_BVEC_FILES")
    check_globbed_files(BMINDS_STREAMLINE_FILES, "BMINDS_STREAMLINE_FILES")
    check_globbed_files(BMINDS_INJECTION_FILES, "BMINDS_INJECTION_FILES")

    # --------------- Create list of all data for each zone name --------------- #
    ALL_DATA_LIST = create_data_list(BMINDS_UNZIPPED_DWI_FILES, BMINDS_BVAL_FILES, BMINDS_BVEC_FILES, 
                                     BMINDS_STREAMLINE_FILES, BMINDS_INJECTION_FILES, BMINDS_ATLAS_FILE, 
                                     BMINDS_STPT_FILE)
    
    # Print out the length of all rows in the list
    SHAPE_ALL_DATA_LIST = np.shape(np.array(ALL_DATA_LIST))
    print("shape of ALL_DATA_LIST: ", SHAPE_ALL_DATA_LIST)
    

if __name__ == "__main__":
    main()
