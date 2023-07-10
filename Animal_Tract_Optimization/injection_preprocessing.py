# We want to create the injection matrix that was in the paper
# To do this, we have many different tracts, and we want to concatenate them all together
# We want to do this for each injection site
import sys
from py_helpers.general_helpers import *
from injection_helpers.inj_general import *
from injection_helpers.inj_general_commands import *
from injection_helpers.inj_region_commands import *
import argparse

def parallel_process(REGION_ID, DWI_FILES, STREAMLINE_FILES, INJECTION_FILES, ATLAS_STPT, TRANSFORMS):

    print("Started parallel process - {}".format(REGION_ID))

    # --------------- MRTRIX reconstruction commands --------------- #
    ARGS_MRTRIX = [
        REGION_ID,
        DWI_FILES,
        STREAMLINE_FILES,
        INJECTION_FILES,
        ATLAS_STPT
    ]
    # Get the mrtrix commands array
    MRTRIX_COMMANDS = mrtrix_all_region_functions(ARGS_MRTRIX)

    # --------------- Calling subprocesses commands --------------- #
   
    # Injection masks, atlas registration, tract editing commands
    for (mrtrix_cmd, cmd_name) in MRTRIX_COMMANDS:
        print("Started {} - {}".format(cmd_name, REGION_ID))
        subprocess.run(mrtrix_cmd, shell=True, check=True) 

    # Find stats of the streamline files

# Main function
def main():
    # --------------- Get main folder paths, check inputs/outputs, unzip necessary --------------- #
    # Get whether it's HPC or not
    parser = argparse.ArgumentParser()
    parser.add_argument('--hpc', help='Whether or not this is being run on the HPC',
                        action='store_true')
    args = parser.parse_args()
    hpc = args.hpc
    (BMINDS_DATA_FOLDER, BMINDS_OUTPUTS_DMRI_FOLDER, BMINDS_OUTPUTS_INJECTIONS_FOLDER, BMINDS_CORE_FOLDER,
        BMINDS_DWI_FOLDER, BMINDS_METADATA_FOLDER, BMINDS_TEMPLATES_FOLDER, BMINDS_ATLAS_FOLDER, BMINDS_STPT_TEMPLATE_FOLDER, 
        BMINDS_TRANSFORMS_FOLDER, BMINDS_INJECTIONS_FOLDER, BMINDS_UNZIPPED_DWI_FOLDER, MAIN_MRTRIX_FOLDER_DMRI, 
        MAIN_MRTRIX_FOLDER_INJECTIONS) = get_main_paths(hpc)
    
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
    check_output_folders(BMINDS_OUTPUTS_INJECTIONS_FOLDER, "BMINDS_OUTPUTS_FOLDER", wipe=False)
    check_output_folders(BMINDS_UNZIPPED_DWI_FOLDER, "BMINDS_UNZIPPED_DWI_FOLDER", wipe=False)
    check_output_folders(MAIN_MRTRIX_FOLDER_INJECTIONS, "MAIN_MRTRIX_FOLDER_INJECTIONS", wipe=False)

    # --------------- Get the injection files --------------- #
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
    ALL_DATA_LIST = create_data_list(BMINDS_UNZIPPED_DWI_FILES, BMINDS_BVAL_FILES, BMINDS_BVEC_FILES, 
                                     BMINDS_STREAMLINE_FILES, BMINDS_INJECTION_FILES, BMINDS_ATLAS_FILE, 
                                     BMINDS_ATLAS_LABEL_FILE, BMINDS_STPT_FILE, BMINDS_MBCA_TRANSFORM_FILE)
    
    print("Length of all data list: {}".format(len(ALL_DATA_LIST)))

    # --------------- Create the common atlas and combined tracts folders --------------- #
    perform_all_general_mrtrix_functions(ALL_DATA_LIST, BMINDS_MBCA_TRANSFORM_FILE, BMINDS_ATLAS_FILE,
                                            BMINDS_STPT_FILE, BMINDS_ATLAS_LABEL_FILE)

    # --------------- Preprocessing the data to get the right file formats --------------- #
    if hpc:
        # # Get the current region based on the command-line
        # region_idx = int(sys.argv[1])
        # # Get the data of the indexed region in the list
        # (REGION_ID, DWI_FILES, STREAMLINE_FILES, INJECTION_FILES, ATLAS_STPT, TRANSFORMS) = ALL_DATA_LIST[region_idx]
        # # Call the parallel process function on this region
        # parallel_process(REGION_ID, DWI_FILES, STREAMLINE_FILES, INJECTION_FILES, ATLAS_STPT, TRANSFORMS)
        # Get the data of the indexed region in the list
        (REGION_ID, DWI_FILES, STREAMLINE_FILES, INJECTION_FILES, ATLAS_STPT, TRANSFORMS) = ALL_DATA_LIST[0]
        # Call the parallel process function on this region
        parallel_process(REGION_ID, DWI_FILES, STREAMLINE_FILES, INJECTION_FILES, ATLAS_STPT, TRANSFORMS)

    else:
        # Call the parallel process function on all regions - serially
        # for region_idx in range(len(ALL_DATA_LIST)):
        #     # Get the region data
        #     (REGION_ID, DWI_FILES, STREAMLINE_FILES, INJECTION_FILES, ATLAS_STPT, TRANSFORMS) = ALL_DATA_LIST[region_idx]
        #     # Call the parallel process function on this region
        #     parallel_process(REGION_ID, DWI_FILES, STREAMLINE_FILES, INJECTION_FILES, ATLAS_STPT, TRANSFORMS)
        # Get the region data
        (REGION_ID, DWI_FILES, STREAMLINE_FILES, INJECTION_FILES, ATLAS_STPT, TRANSFORMS) = ALL_DATA_LIST[0]
        # Call the parallel process function on this region
        parallel_process(REGION_ID, DWI_FILES, STREAMLINE_FILES, INJECTION_FILES, ATLAS_STPT, TRANSFORMS)
    

if __name__ == "__main__":
    main()
