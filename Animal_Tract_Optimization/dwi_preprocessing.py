# Right now we just want to read the data using nibabel
import os
import sys
import numpy as np
import nibabel as nib
from py_helpers.general_helpers import *
from py_helpers.SC_commands import *
import multiprocessing as mp
import subprocess

def parallel_process(REGION_ID, DWI_FILES, STREAMLINE_FILES, INJECTION_FILES, ATLAS_STPT):

    print("Started parallel process - {}".format(REGION_ID))

    print("DWI FILES: {}".format(DWI_FILES))
    print("STREAMLINE FILES: {}".format(STREAMLINE_FILES))
    print("INJECTION FILES: {}".format(INJECTION_FILES))
    print("ATLAS/STPT FILES: {}".format(ATLAS_STPT))

    # --------------- Creating NiFTi collection commands --------------- #

    # --------------- MRTRIX reconstruction commands --------------- #
    # Define needed arguments array
    ARGS_MRTRIX = [
        REGION_ID,
        DWI_FILES,
        ATLAS_STPT
    ]
    # Get the mrtrix commands array
    MRTRIX_COMMANDS = pre_tractography_commands(ARGS_MRTRIX)

    # --------------- Injection matrix creation commands --------------- #

    # --------------- Calling subprocesses commands --------------- #
   
    # Pre-tractography commands
    for (mrtrix_cmd, cmd_name) in MRTRIX_COMMANDS:
        print("Started {} - {}".format(cmd_name, REGION_ID))
        subprocess.run(mrtrix_cmd, shell=True, check=True)


# Main function
def main():

    # --------------- Get main folder paths, check inputs/outputs, unzip necessary --------------- #
    # Get the folder paths
    hpc = False
    (BMINDS_DATA_FOLDER, BMINDS_OUTPUTS_FOLDER, BMINDS_ATLAS_STPT_FOLDER, BMINDS_CORE_FOLDER, BMINDS_DWI_FOLDER,
        BMINDS_METADATA_FOLDER, BMINDS_INJECTIONS_FOLDER, BMINDS_UNZIPPED_DWI_FOLDER, MAIN_MRTRIX_FOLDER) = get_main_paths(hpc)
    
    # Check the input folders
    check_input_folders(BMINDS_DATA_FOLDER, "BMINDS_DATA_FOLDER")
    check_input_folders(BMINDS_ATLAS_STPT_FOLDER, "BMINDS_ATLAS_STPT_FOLDER")
    check_input_folders(BMINDS_CORE_FOLDER, "BMINDS_CORE_FOLDER")
    check_input_folders(BMINDS_DWI_FOLDER, "BMINDS_DWI_FOLDER")
    check_input_folders(BMINDS_METADATA_FOLDER, "BMINDS_METADATA_FOLDER")
    check_input_folders(BMINDS_INJECTIONS_FOLDER, "BMINDS_INJECTIONS_FOLDER")

    # Check the output folders
    check_output_folders(BMINDS_OUTPUTS_FOLDER, "BMINDS_OUTPUTS_FOLDER", wipe=False)
    check_output_folders(BMINDS_UNZIPPED_DWI_FOLDER, "BMINDS_UNZIPPED_DWI_FOLDER", wipe=False)
    check_output_folders(MAIN_MRTRIX_FOLDER, "MAIN_MRTRIX_FOLDER", wipe=True)

    # Unzip all input files
    check_unzipping(BMINDS_DWI_FOLDER, BMINDS_UNZIPPED_DWI_FOLDER)

    # --------------- Glob the DWI, bval, bvec and tract-tracing data --------------- #
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
         
    # --------------- Preprocessing the data to get the right file formats --------------- #
    if hpc:
        # Get the current region based on the command-line
        region_idx = int(sys.argv[1])
        # Get the data of the indexed region in the list
        (REGION_ID, DWI_FILES, STREAMLINE_FILES, INJECTION_FILES, ATLAS_STPT) = ALL_DATA_LIST[region_idx]
        # Call the parallel process function on this region
        parallel_process(REGION_ID, DWI_FILES, STREAMLINE_FILES, INJECTION_FILES, ATLAS_STPT)
    else:
        # Call the parallel process function on all regions - serially
        for region_idx in range(len(ALL_DATA_LIST)):
            # Get the region data
            (REGION_ID, DWI_FILES, STREAMLINE_FILES, INJECTION_FILES, ATLAS_STPT) = ALL_DATA_LIST[region_idx]
            # Call the parallel process function on this region
            parallel_process(REGION_ID, DWI_FILES, STREAMLINE_FILES, INJECTION_FILES, ATLAS_STPT)


if __name__ == '__main__':
    main()