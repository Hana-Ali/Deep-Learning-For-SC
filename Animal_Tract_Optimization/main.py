# Right now we just want to read the data using nibabel
import os
import sys
import numpy as np
import nibabel as nib
from py_helpers.general_helpers import *

# Main function
def main():

    # --------------- Get main folder paths, check inputs/outputs, unzip necessary --------------- #
    # Get the folder paths
    (BMINDS_DATA_FOLDER, BMINDS_METADATA_FOLDER, BMINDS_INJECTIONS_FOLDER, BMINDS_ATLAS_STPT_FOLDER,
        BMINDS_ZIPPED_DWI_FOLDER, BMINDS_UNZIPPED_DWI_FOLDER) = get_main_paths(hpc=False)
    
    # Check the input folders
    check_input_folders(BMINDS_DATA_FOLDER, "BMINDS_DATA_FOLDER")
    check_input_folders(BMINDS_METADATA_FOLDER, "BMINDS_METADATA_FOLDER")
    check_input_folders(BMINDS_INJECTIONS_FOLDER, "BMINDS_INJECTIONS_FOLDER")
    check_input_folders(BMINDS_ATLAS_STPT_FOLDER, "BMINDS_ATLAS_STPT_FOLDER")
    check_input_folders(BMINDS_ZIPPED_DWI_FOLDER, "BMINDS_ZIPPED_DWI_FOLDER")

    # Check the output folders
    check_output_folders(BMINDS_UNZIPPED_DWI_FOLDER, "BMINDS_UNZIPPED_DWI_FOLDER", wipe=False)

    # Unzip all input files
    check_unzipping(BMINDS_ZIPPED_DWI_FOLDER, BMINDS_UNZIPPED_DWI_FOLDER)

    # --------------- Glob the DWI, bval, bvec and tract-tracing data --------------- #
    BMINDS_UNZIPPED_DWI_FILES = glob_files(BMINDS_UNZIPPED_DWI_FOLDER, "nii")
    BMINDS_BVAL_FILES = glob_files(BMINDS_ZIPPED_DWI_FOLDER, "bval")
    BMINDS_BVEC_FILES = glob_files(BMINDS_ZIPPED_DWI_FOLDER, "bvec")
    BMINDS_STREAMLINE_FILES = glob_files(BMINDS_METADATA_FOLDER, "tck")
    BMINDS_INJECTION_FILES = glob_files(BMINDS_INJECTIONS_FOLDER, "nii.gz")
    BMINDS_ATLAS_STPT_FILES = glob_files(BMINDS_ATLAS_STPT_FOLDER, "nii.gz") # Gets both the atlas and stpt files, need to separate
    
    # Get the atlas and stpt files - separate from the mix above
    BMINDS_ATLAS_FILE = [file for file in BMINDS_ATLAS_STPT_FILES if "atlas_segmentation" in file]
    BMINDS_STPT_FILE = [file for file in BMINDS_ATLAS_STPT_FILES if "STPT_template" in file]

    # Check the globbed files
    check_globbed_files(BMINDS_UNZIPPED_DWI_FILES, "BMINDS_UNZIPPED_DWI_FILES")
    check_globbed_files(BMINDS_BVAL_FILES, "BMINDS_BVAL_FILES")
    check_globbed_files(BMINDS_BVEC_FILES, "BMINDS_BVEC_FILES")
    check_globbed_files(BMINDS_STREAMLINE_FILES, "BMINDS_STREAMLINE_FILES")
    check_globbed_files(BMINDS_INJECTION_FILES, "BMINDS_INJECTION_FILES")

    # --------------- Create list of all data for each zone name --------------- #
    ALL_DATA_LIST = create_data_list(BMINDS_UNZIPPED_DWI_FILES, BMINDS_BVAL_FILES, BMINDS_BVEC_FILES, 
                                        BMINDS_STREAMLINE_FILES, BMINDS_INJECTION_FILES)
    

    # --------------- ACTUAL ALGORITHM TIME NOW THAT WE'VE EXTRACTED THE DATA --------------- #





if __name__ == '__main__':
    main()