# Right now we just want to read the data using nibabel
import sys
import subprocess
import argparse

from py_helpers import *
from dwi_helpers import *

def parallel_process(REGION_ID, DWI_FILES, STREAMLINE_FILES, INJECTION_FILES, ATLAS_STPT):

    print("Started parallel process - {}".format(REGION_ID))

    print("DWI FILES: {}".format(DWI_FILES))
    print("STREAMLINE FILES: {}".format(STREAMLINE_FILES))
    print("INJECTION FILES: {}".format(INJECTION_FILES))
    print("ATLAS/STPT FILES: {}".format(ATLAS_STPT))

    # --------------- MRTRIX reconstruction commands --------------- #
    # Define needed arguments array
    ARGS_MRTRIX = [
        REGION_ID,
        DWI_FILES,
        ATLAS_STPT
    ]
    # Get the mrtrix commands array
    MRTRIX_COMMANDS = pre_tractography_commands(ARGS_MRTRIX)

    # --------------- Calling subprocesses commands --------------- #
   
    # Pre-tractography commands
    for (mrtrix_cmd, cmd_name) in MRTRIX_COMMANDS:
        print("Started {} - {}".format(cmd_name, REGION_ID))
        subprocess.run(mrtrix_cmd, shell=True, check=True) 

# Main function
def main():

    # --------------- Get main folder paths, check inputs/outputs, unzip necessary --------------- #
    # Get whether it's HPC or not
    hpc = int(sys.argv[1])
    # Get the folder paths
    (BMINDS_DATA_FOLDER, BMINDS_OUTPUTS_DMRI_FOLDER, BMINDS_OUTPUTS_INJECTIONS_FOLDER, BMINDS_CORE_FOLDER,
    BMINDS_DWI_FOLDER, BMINDS_METADATA_FOLDER, BMINDS_TEMPLATES_FOLDER, BMINDS_ATLAS_FOLDER, BMINDS_STPT_TEMPLATE_FOLDER, 
    BMINDS_TRANSFORMS_FOLDER, BMINDS_INJECTIONS_FOLDER, BMINDS_UNZIPPED_DWI_FOLDER, BMINDS_UNZIPPED_DWI_RESIZED_FOLDER,
    MAIN_MRTRIX_FOLDER_DMRI, MAIN_MRTRIX_FOLDER_INJECTIONS) = get_main_paths(hpc)
    
    # Check the input folders
    check_input_folders(BMINDS_DATA_FOLDER, "BMINDS_DATA_FOLDER")
    check_input_folders(BMINDS_CORE_FOLDER, "BMINDS_CORE_FOLDER")
    check_input_folders(BMINDS_DWI_FOLDER, "BMINDS_DWI_FOLDER")
    check_input_folders(BMINDS_METADATA_FOLDER, "BMINDS_METADATA_FOLDER")
    check_input_folders(BMINDS_TEMPLATES_FOLDER, "BMINDS_TEMPLATES_FOLDER")
    check_input_folders(BMINDS_ATLAS_FOLDER, "BMINDS_ATLAS_FOLDER")
    check_input_folders(BMINDS_STPT_TEMPLATE_FOLDER, "BMINDS_STPT_TEMPLATE_FOLDER")
    check_input_folders(BMINDS_TRANSFORMS_FOLDER, "BMINDS_TRANSFORMS_FOLDER")
    check_input_folders(BMINDS_INJECTIONS_FOLDER, "BMINDS_INJECTIONS_FOLDER")

    # Check the output folders
    check_output_folders(BMINDS_OUTPUTS_DMRI_FOLDER, "BMINDS_OUTPUTS_FOLDER", wipe=False)
    check_output_folders(BMINDS_UNZIPPED_DWI_FOLDER, "BMINDS_UNZIPPED_DWI_FOLDER", wipe=False)
    check_output_folders(BMINDS_UNZIPPED_DWI_RESIZED_FOLDER, "BMINDS_UNZIPPED_DWI_RESIZED_FOLDER", wipe=False)
    check_output_folders(MAIN_MRTRIX_FOLDER_DMRI, "MAIN_MRTRIX_FOLDER", wipe=False)

    # Unzip all input files
    check_unzipping(BMINDS_DWI_FOLDER, BMINDS_UNZIPPED_DWI_FOLDER)

    # --------------- Glob the DWI, bval, bvec and tract-tracing data --------------- #
    BMINDS_UNZIPPED_DWI_FILES = glob_files(BMINDS_UNZIPPED_DWI_FOLDER, "nii")
    BMINDS_BVAL_FILES = glob_files(BMINDS_DWI_FOLDER, "bval")
    BMINDS_BVEC_FILES = glob_files(BMINDS_DWI_FOLDER, "bvec")
    BMINDS_STREAMLINE_FILES = glob_files(BMINDS_METADATA_FOLDER, "tck")
    BMINDS_INJECTION_FILES = glob_files(BMINDS_INJECTIONS_FOLDER, "nii.gz")
    BMINDS_ATLAS_FILES = glob_files(BMINDS_ATLAS_FOLDER, "nii.gz")
    BMINDS_ATLAS_LABEL_FILES = glob_files(BMINDS_ATLAS_FOLDER, "txt")
    BMINDS_STPT_FILES = glob_files(BMINDS_STPT_TEMPLATE_FOLDER, "nii")
    BMINDS_TRANSFORM_FILES = glob_files(BMINDS_TRANSFORMS_FOLDER, "h5")
    
    # Get the atlas and stpt files - separate from the mix above
    BMINDS_ATLAS_FILE = [file for file in BMINDS_ATLAS_FILES if "140_region_atlas_segmentation" in file]
    BMINDS_ATLAS_LABEL_FILE = [file for file in BMINDS_ATLAS_LABEL_FILES if "140_region_atlas_labels" in file]
    BMINDS_STPT_FILE = [file for file in BMINDS_STPT_FILES if "STPT_template_unzipped" in file]
    BMINDS_MBCA_TRANSFORM_FILE = [file for file in BMINDS_TRANSFORM_FILES if "MBCA" in file]

    # Check the globbed files
    check_globbed_files(BMINDS_UNZIPPED_DWI_FILES, "BMINDS_UNZIPPED_DWI_FILES")
    check_globbed_files(BMINDS_BVAL_FILES, "BMINDS_BVAL_FILES")
    check_globbed_files(BMINDS_BVEC_FILES, "BMINDS_BVEC_FILES")
    check_globbed_files(BMINDS_STREAMLINE_FILES, "BMINDS_STREAMLINE_FILES")
    check_globbed_files(BMINDS_INJECTION_FILES, "BMINDS_INJECTION_FILES")
    check_globbed_files(BMINDS_ATLAS_FILE, "BMINDS_ATLAS_FILE")
    check_globbed_files(BMINDS_ATLAS_LABEL_FILE, "BMINDS_ATLAS_LABEL_FILE")
    check_globbed_files(BMINDS_STPT_FILE, "BMINDS_STPT_FILE")
    check_globbed_files(BMINDS_MBCA_TRANSFORM_FILE, "BMINDS_MBCA_TRANSFORM_FILE")

    # --------------- Create list of all data for each zone name --------------- #
    (ALL_DATA_LIST, RESIZED_ALL_DATA_LIST) = create_data_list(BMINDS_UNZIPPED_DWI_FILES, BMINDS_BVAL_FILES, BMINDS_BVEC_FILES, 
                                                                BMINDS_STREAMLINE_FILES, BMINDS_INJECTION_FILES, BMINDS_ATLAS_FILE, 
                                                                BMINDS_ATLAS_LABEL_FILE, BMINDS_STPT_FILE)

    print("Length of All Data List: {}".format(len(ALL_DATA_LIST)))
    print("Length of Resized All Data List: {}".format(len(RESIZED_ALL_DATA_LIST)))
    print("All resized data list: {}".format(RESIZED_ALL_DATA_LIST))

    # --------------- Preprocessing the data to get the right file formats --------------- #
    if hpc:
        # # Get the current region based on the command-line
        # region_idx = int(sys.argv[2])
        # # Get the data of the indexed region in the list
        # (REGION_ID, DWI_FILES, STREAMLINE_FILES, INJECTION_FILES, ATLAS_STPT) = ALL_DATA_LIST[region_idx]
        # # Call the parallel process function on this region
        # parallel_process(REGION_ID, DWI_FILES, STREAMLINE_FILES, INJECTION_FILES, ATLAS_STPT)
        # # Get the data of the indexed region in the list
        # (REGION_ID, DWI_FILES, STREAMLINE_FILES, INJECTION_FILES, ATLAS_STPT) = ALL_DATA_LIST[0]
        # # Call the parallel process function on this region
        # parallel_process(REGION_ID, DWI_FILES, STREAMLINE_FILES, INJECTION_FILES, ATLAS_STPT)
        pass
    else:
        # Call the parallel process function on all regions - serially
        for region_idx in range(len(ALL_DATA_LIST)):
            # Get the region data
            (REGION_ID, DWI_FILES, STREAMLINE_FILES, INJECTION_FILES, ATLAS_STPT) = ALL_DATA_LIST[region_idx]
            # Call the parallel process function on this region
            parallel_process(REGION_ID, DWI_FILES, STREAMLINE_FILES, INJECTION_FILES, ATLAS_STPT)


if __name__ == '__main__':
    main()